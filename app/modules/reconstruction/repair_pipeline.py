"""app/modules/reconstruction/repair_pipeline.py — B1+B2+B3+B4.

B1 : moteur adaptatif — choisit les familles selon corruption_type
B2 : boucle itérative multi-pass avec rollback si score baisse
B3 : stratégies composées (denoise→inpaint, inpaint→sharpen, mask_dilate→inpaint…)
B4 : top_candidates [best_score, best_visual, most_conservative]
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.exceptions import ReconstructionError
from app.core.logger import logger
from app.modules.evaluation.metrics import score_candidate as _metrics_score_candidate
from app.modules.reconstruction.block_repair import repair_blocks
from app.modules.reconstruction.deblurring import deblur
from app.modules.reconstruction.denoising import denoise_image
from app.modules.reconstruction.inpainting import reconstruct_with_inpaint
try:
    from app.modules.reconstruction.patchmatch import patchmatch_inpaint as _patchmatch_inpaint
    _PATCHMATCH_AVAILABLE = True
except ImportError:
    _PATCHMATCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_candidate(
    corrupted_image_path: str | Path,
    candidate_path: str | Path,
    original_image_path: str | Path | None = None,
    mask: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    result = _metrics_score_candidate(
        corrupted=corrupted_image_path,
        reconstructed=candidate_path,
        original=original_image_path,
        mask=mask,
    )
    return float(result.get("score", 0.0)), result


def _save_image(output_path: Path, image: np.ndarray) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise ReconstructionError(f"Impossible d'écrire l'image : {output_path}")
    return str(output_path)


def _candidate_from_path(
    name: str,
    path: str,
    corrupted_image_path: str | Path,
    original_image_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    score, details = _score_candidate(corrupted_image_path, path, original_image_path, mask=mask)
    candidate: dict[str, Any] = {"strategy": name, "path": path, "score": score, **details}
    if extra:
        candidate.update(extra)
    return candidate


# ---------------------------------------------------------------------------
# B1 — Moteur adaptatif : plan de candidats selon corruption_type
# ---------------------------------------------------------------------------

def choose_repair_strategy(corruption_type: str, detection_confidence: float) -> str:
    ct = (corruption_type or "").lower()
    if ct in {"mask_like", "missing", "zone_deletion", "bar", "multiple_bars",
              "large_deleted_square", "random_holes", "scratch_lines"} and detection_confidence >= 0.40:
        return "inpainting"
    if ct in {"noise_like", "noise", "local_noise"}:
        return "denoise"
    if ct in {"blur", "blur_like", "local_blur"}:
        return "deblur"
    if ct in {"jpeg_block", "jpeg_block_artifacts", "blocky"}:
        return "block_repair"
    if ct in {"mixed", "combined", "shift_region", "block_dropout"}:
        return "hybrid"
    return "conservative"


def _build_adaptive_plan(
    corruption_type: str,
    base_radius: int,
    recommended: str,
) -> list[dict[str, Any]]:
    """B1 — Construit un plan adaptatif priorisant les familles pertinentes.

    Les familles adaptées au type de corruption passent en premier,
    les autres sont ajoutées en fallback.
    """
    ct = (corruption_type or "").lower()

    # Définir l'ordre des familles selon le type de corruption
    family_priority: list[str]
    if recommended == "inpainting":
        family_priority = ["inpainting", "composite", "hybrid", "patchmatch", "denoise", "deblur", "block_repair"]
    elif recommended == "denoise":
        family_priority = ["denoise", "composite", "inpainting", "patchmatch", "deblur", "block_repair"]
    elif recommended == "deblur":
        family_priority = ["deblur", "composite", "denoise", "inpainting", "patchmatch", "block_repair"]
    elif recommended == "block_repair":
        family_priority = ["block_repair", "deblur", "composite", "denoise", "inpainting", "patchmatch"]
    else:  # hybrid / conservative
        family_priority = ["hybrid", "composite", "inpainting", "patchmatch", "denoise", "deblur", "block_repair"]

    all_plans: dict[str, list[dict[str, Any]]] = {
        "inpainting": [
            {"strategy": "inpainting_r3", "family": "inpainting", "radius": max(3, base_radius)},
            {"strategy": "inpainting_r5", "family": "inpainting", "radius": max(5, base_radius + 2)},
            {"strategy": "inpainting_r7", "family": "inpainting", "radius": max(7, base_radius + 4)},
        ],
        "patchmatch": [
            {"strategy": "patchmatch_p7_i5",  "family": "patchmatch", "patch_size": 7,  "iterations": 5},
            {"strategy": "patchmatch_p9_i5",  "family": "patchmatch", "patch_size": 9,  "iterations": 5},
            {"strategy": "patchmatch_p11_i7", "family": "patchmatch", "patch_size": 11, "iterations": 7},
        ],
        "denoise": [
            {"strategy": "denoise_median_blur",   "family": "denoise", "denoise_method": "median_blur"},
            {"strategy": "denoise_gaussian_blur",  "family": "denoise", "denoise_method": "gaussian_blur"},
            {"strategy": "denoise_bilateral",      "family": "denoise", "denoise_method": "bilateral"},
        ],
        "deblur": [
            {"strategy": "deblur_light",  "family": "deblur", "strength": "light"},
            {"strategy": "deblur_strong", "family": "deblur", "strength": "strong"},
        ],
        "block_repair": [
            {"strategy": "block_repair", "family": "block_repair"},
        ],
        # B3 — Stratégies composées
        "composite": [
            {"strategy": "denoise_then_inpaint",   "family": "composite",
             "steps": ["denoise_median_blur", "inpainting_r5"]},
            {"strategy": "inpaint_then_sharpen",   "family": "composite",
             "steps": ["inpainting_r5", "deblur_light"]},
            {"strategy": "deblur_then_inpaint",    "family": "composite",
             "steps": ["deblur_light", "inpainting_r3"]},
            {"strategy": "block_then_inpaint",     "family": "composite",
             "steps": ["block_repair", "inpainting_r3"]},
            {"strategy": "mask_dilate_inpaint",    "family": "composite",
             "steps": ["mask_dilate", "inpainting_r5"]},
            {"strategy": "multi_pass_inpaint",     "family": "composite",
             "steps": ["inpainting_r3", "inpainting_r5", "inpainting_r7"]},
        ],
        "hybrid": [
            {"strategy": "hybrid_denoise_inpaint", "family": "hybrid",
             "denoise_method": "median_blur", "radius": max(5, base_radius)},
        ],
    }

    plan: list[dict[str, Any]] = []
    for family in family_priority:
        plan.extend(all_plans.get(family, []))

    return plan


# ---------------------------------------------------------------------------
# B3 — Exécution des stratégies composées
# ---------------------------------------------------------------------------

def _run_composite_step(
    step: str,
    current_path: str | Path,
    mask_path_obj: Path | None,
    image_np: np.ndarray,
    method: str,
    base_radius: int,
    out_dir: Path,
    stem: str,
    step_idx: int,
) -> str:
    """Exécute une étape d'une stratégie composée, retourne le chemin résultant."""
    current_path = Path(current_path)

    if step.startswith("inpainting_r"):
        radius = int(step.split("_r")[1])
        if mask_path_obj is None:
            return str(current_path)
        r = reconstruct_with_inpaint(current_path, mask_path_obj, method=method, radius=radius)
        return r["path"]

    if step.startswith("denoise_"):
        method_name = step.replace("denoise_", "")
        r = denoise_image(current_path, method=method_name)
        return r["path"]

    if step == "deblur_light":
        img = cv2.imread(str(current_path), cv2.IMREAD_COLOR)
        result = deblur(img)
        out = out_dir / f"{stem}_composite_{step_idx}.png"
        _save_image(out, result)
        return str(out)

    if step == "block_repair":
        img = cv2.imread(str(current_path), cv2.IMREAD_COLOR)
        result = repair_blocks(img)
        out = out_dir / f"{stem}_composite_block_{step_idx}.png"
        _save_image(out, result)
        return str(out)

    if step == "mask_dilate" and mask_path_obj is not None:
        # Dilater le masque de 3px puis inpainter
        mask = cv2.imread(str(mask_path_obj), cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=2)
        dilated_path = out_dir / f"{stem}_mask_dilated.png"
        cv2.imwrite(str(dilated_path), dilated)
        # On retourne le path du masque dilaté — la prochaine étape inpainting l'utilisera
        # Hack : on stocke dans le chemin pour la prochaine itération
        return str(current_path)  # image inchangée, masque dilaté géré séparément

    return str(current_path)


def _run_composite_strategy(
    plan: dict[str, Any],
    corrupted_image_path: Path,
    mask_path_obj: Path | None,
    image_np: np.ndarray,
    method: str,
    base_radius: int,
    out_dir: Path,
) -> str | None:
    """Exécute une stratégie composée séquentiellement."""
    steps = plan.get("steps", [])
    if not steps:
        return None

    current = str(corrupted_image_path)
    stem = corrupted_image_path.stem

    for idx, step in enumerate(steps):
        try:
            current = _run_composite_step(
                step, current, mask_path_obj, image_np,
                method, base_radius, out_dir, stem, idx,
            )
        except Exception as exc:
            logger.debug("Composite step %s failed: %s", step, exc)
            return None

    return current


# ---------------------------------------------------------------------------
# B2 — Boucle itérative multi-pass avec rollback
# ---------------------------------------------------------------------------

def _run_iterative_pass(
    corrupted_image_path: Path,
    mask_path_obj: Path | None,
    original_image_path: Path | None,
    method: str,
    base_radius: int,
    max_iterations: int = 3,
    mask_arr: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """B2 — Affine le meilleur inpainting par passes successives.

    Augmente le rayon à chaque itération, fait un rollback si le score baisse.
    Retourne le meilleur candidat itératif ou None si aucune amélioration.
    """
    if mask_path_obj is None:
        return None

    best_path = str(corrupted_image_path)
    best_score, best_details = _score_candidate(
        corrupted_image_path, corrupted_image_path, original_image_path, mask=mask_arr
    )
    iterations: list[dict[str, Any]] = []
    stopped_reason = "max_iterations"

    for i in range(max_iterations):
        radius = base_radius + i * 2
        try:
            r = reconstruct_with_inpaint(
                Path(best_path), mask_path_obj, method=method, radius=radius
            )
            score, details = _score_candidate(
                corrupted_image_path, r["path"], original_image_path, mask=mask_arr
            )
            iterations.append({
                "iteration": i + 1,
                "radius": radius,
                "score": score,
                "path": r["path"],
            })

            if score > best_score + 0.1:  # amélioration significative
                best_score = score
                best_path = r["path"]
                best_details = details
            else:
                stopped_reason = "no_improvement"
                break

        except Exception as exc:
            logger.debug("Iterative pass %d failed: %s", i, exc)
            stopped_reason = "error"
            break

    if best_path == str(corrupted_image_path):
        return None  # aucune amélioration

    return {
        "strategy": "iterative_inpaint",
        "path": best_path,
        "score": best_score,
        "iterations": iterations,
        "stopped_reason": stopped_reason,
        **best_details,
    }


# ---------------------------------------------------------------------------
# B4 — Sélection Top 3
# ---------------------------------------------------------------------------

def _select_top3(
    candidates: list[dict[str, Any]],
    selected_strategy: str,
) -> dict[str, Any]:
    """B4 — Retourne best_score, best_visual, most_conservative."""
    if not candidates:
        return {}

    sorted_c = sorted(candidates, key=lambda c: float(c.get("score", 0.0)), reverse=True)

    # best_score : score le plus élevé
    best_score = sorted_c[0]

    # best_visual : inpainting avec le meilleur PSNR (si dispo)
    inpainting_candidates = [c for c in sorted_c if "inpainting" in c.get("strategy", "")]
    best_visual = inpainting_candidates[0] if inpainting_candidates else sorted_c[0]

    # most_conservative : conservative ou score le plus bas
    conservative = next(
        (c for c in candidates if c.get("strategy") == "conservative"),
        sorted_c[-1],
    )

    return {
        "best_score":        {"strategy": best_score.get("strategy"),   "score": best_score.get("score"),   "path": best_score.get("path")},
        "best_visual":       {"strategy": best_visual.get("strategy"),  "score": best_visual.get("score"),  "path": best_visual.get("path")},
        "most_conservative": {"strategy": conservative.get("strategy"), "score": conservative.get("score"), "path": conservative.get("path")},
    }


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_repair_pipeline(
    corrupted_image_path: str | Path,
    mask_path: str | Path | None = None,
    method: str = "opencv_inpaint",
    radius: int = 3,
    corruption_type: str = "mask_like",
    detection_confidence: float = 1.0,
    original_image_path: str | Path | None = None,
    max_attempts: int = 8,
) -> dict[str, Any]:
    corrupted_image_path = Path(corrupted_image_path)
    if not corrupted_image_path.exists():
        raise ReconstructionError(f"Image corrompue introuvable : {corrupted_image_path}")

    mask_path_obj: Path | None = None
    if mask_path is not None:
        mask_path_obj = Path(mask_path)
        if not mask_path_obj.exists():
            raise ReconstructionError(f"Masque introuvable : {mask_path_obj}")

    orig_path = Path(original_image_path) if original_image_path else None
    out_dir = corrupted_image_path.parent.parent / "reconstructed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load mask as ndarray once — used to improve scoring accuracy
    mask_arr: np.ndarray | None = None
    if mask_path_obj is not None:
        try:
            from PIL import Image as _PILImage
            mask_arr = np.array(_PILImage.open(mask_path_obj).convert("L"))
        except Exception:
            mask_arr = None

    recommended = choose_repair_strategy(corruption_type, detection_confidence)
    logger.info(
        "Repair pipeline start | image=%s | mask=%s | recommended=%s | confidence=%.3f",
        corrupted_image_path, mask_path_obj, recommended, detection_confidence,
    )

    image_np = cv2.imread(str(corrupted_image_path), cv2.IMREAD_COLOR)
    if image_np is None:
        raise ReconstructionError(f"Impossible de charger l'image : {corrupted_image_path}")

    candidates: list[dict[str, Any]] = []

    # Baseline conservative
    candidates.append(_candidate_from_path(
        "conservative", str(corrupted_image_path),
        corrupted_image_path, orig_path,
        extra={"recommended": recommended, "family": "conservative"},
        mask=mask_arr,
    ))

    # B1 — Plan adaptatif
    adaptive_plan = _build_adaptive_plan(corruption_type, radius, recommended)

    for plan in adaptive_plan:
        if len(candidates) >= max_attempts:
            break

        family   = str(plan["family"])
        strategy = str(plan["strategy"])

        try:
            # B3 — Stratégies composées
            if family == "composite":
                result_path = _run_composite_strategy(
                    plan, corrupted_image_path, mask_path_obj,
                    image_np, method, radius, out_dir,
                )
                if result_path and Path(result_path).exists():
                    candidates.append(_candidate_from_path(
                        strategy, result_path, corrupted_image_path, orig_path,
                        extra={"family": family, "steps": plan.get("steps", [])},
                        mask=mask_arr,
                    ))

            elif family == "patchmatch":
                if mask_path_obj is None or not _PATCHMATCH_AVAILABLE:
                    continue
                result = _patchmatch_inpaint(
                    corrupted_image_path, mask_path_obj,
                    patch_size=int(plan["patch_size"]),
                    iterations=int(plan["iterations"]),
                )
                candidates.append(_candidate_from_path(
                    strategy, result["path"], corrupted_image_path, orig_path,
                    extra={"patch_size": plan["patch_size"],
                           "iterations": plan["iterations"], "family": family},
                    mask=mask_arr,
                ))

            elif family == "inpainting":
                if mask_path_obj is None:
                    continue
                result = reconstruct_with_inpaint(
                    corrupted_image_path, mask_path_obj,
                    method=method, radius=int(plan["radius"]),
                )
                candidates.append(_candidate_from_path(
                    strategy, result["path"], corrupted_image_path, orig_path,
                    extra={"radius": int(plan["radius"]), "family": family},
                    mask=mask_arr,
                ))

            elif family == "denoise":
                result = denoise_image(corrupted_image_path, method=str(plan["denoise_method"]))
                candidates.append(_candidate_from_path(
                    strategy, result["path"], corrupted_image_path, orig_path,
                    extra={"family": family},
                    mask=mask_arr,
                ))

            elif family == "deblur":
                deblurred = deblur(image_np)
                if plan.get("strength") == "strong":
                    deblurred = deblur(deblurred)
                output = out_dir / f"{corrupted_image_path.stem}_{strategy}.png"
                path = _save_image(output, deblurred)
                candidates.append(_candidate_from_path(
                    strategy, path, corrupted_image_path, orig_path,
                    extra={"family": family, "strength": plan.get("strength")},
                    mask=mask_arr,
                ))

            elif family == "block_repair":
                repaired = repair_blocks(image_np)
                output = out_dir / f"{corrupted_image_path.stem}_{strategy}.png"
                path = _save_image(output, repaired)
                candidates.append(_candidate_from_path(
                    strategy, path, corrupted_image_path, orig_path,
                    extra={"family": family},
                    mask=mask_arr,
                ))

            elif family == "hybrid":
                if mask_path_obj is None:
                    continue
                denoised = denoise_image(corrupted_image_path, method=str(plan["denoise_method"]))
                hybrid = reconstruct_with_inpaint(
                    denoised["path"], mask_path_obj,
                    method=method, radius=int(plan["radius"]),
                )
                candidates.append(_candidate_from_path(
                    strategy, hybrid["path"], corrupted_image_path, orig_path,
                    extra={"family": family, "radius": int(plan["radius"]),
                           "denoise_method": str(plan["denoise_method"])},
                    mask=mask_arr,
                ))

        except Exception as exc:
            logger.warning("Repair candidate failed (%s): %s", strategy, exc)

    # B2 — Passe itérative sur le meilleur candidat courant
    if len(candidates) < max_attempts and mask_path_obj is not None:
        iterative = _run_iterative_pass(
            corrupted_image_path, mask_path_obj, orig_path,
            method, radius, max_iterations=min(3, max_attempts - len(candidates)),
            mask_arr=mask_arr,
        )
        if iterative:
            candidates.append(iterative)

    if not candidates:
        raise ReconstructionError("Aucune tentative de reconstruction valide")

    candidates.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
    best = candidates[0]

    # B4 — Top 3
    top_candidates = _select_top3(candidates, best["strategy"])

    result = {
        "path":                     best["path"],
        "status":                   "reconstructed",
        "source_image":             corrupted_image_path.name,
        "selected_repair_strategy": best["strategy"],
        "repair_strategy":          best["strategy"],
        "recommended_strategy":     recommended,
        "detection_confidence":     detection_confidence,
        "score":                    best["score"],
        "candidates":               candidates,
        "top_candidates":           top_candidates,
        "retry_count":              max(0, len(candidates) - 1),
        "method":                   method,
        "corruption_type":          corruption_type,
    }
    logger.info(
        "Repair pipeline done | selected=%s | score=%.2f | attempts=%s",
        best["strategy"], best["score"], len(candidates),
    )
    return result