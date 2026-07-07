"""app/modules/reconstruction/repair_pipeline.py — B1+B2+B3+B4.

B1 : moteur adaptatif — choisit les familles selon corruption_type
B2 : boucle itérative multi-pass avec rollback si score baisse
B3 : stratégies composées (denoise→inpaint, inpaint→sharpen, mask_dilate→inpaint…)
B4 : top_candidates [best_score, best_visual, most_conservative]
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.core.exceptions import ReconstructionError
from app.core.logger import logger
from app.modules.evaluation.metrics import score_candidate as _metrics_score_candidate
from app.modules.reconstruction.block_repair import repair_blocks
from app.modules.reconstruction.deblurring import deblur
from app.modules.reconstruction.denoising import denoise_image
from app.modules.reconstruction.inpainting import criminisi_inpaint, reconstruct_with_inpaint
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


def _build_forensic_supreme_plan(
    corruption_type: str,
    base_radius: int,
    recommended: str,
) -> list[dict[str, Any]]:
    """Construit le plan exhaustif du mode forensic_supreme."""
    plan = _build_adaptive_plan(corruption_type, base_radius, recommended)
    has_supreme_patchmatch = any(
        p.get("family") == "patchmatch"
        and int(p.get("patch_size", 0)) == 15
        and int(p.get("iterations", 0)) == 20
        for p in plan
    )
    if not has_supreme_patchmatch:
        last_patchmatch_idx = max(
            (idx for idx, p in enumerate(plan) if p.get("family") == "patchmatch"),
            default=-1,
        )
        supreme_patchmatch = {
            "strategy": "patchmatch_p15_i20",
            "family": "patchmatch",
            "patch_size": 15,
            "iterations": 20,
        }
        if last_patchmatch_idx >= 0:
            plan.insert(last_patchmatch_idx + 1, supreme_patchmatch)
        else:
            plan.append(supreme_patchmatch)

    last_inpainting_idx = max(
        (idx for idx, p in enumerate(plan) if p.get("family") == "inpainting"),
        default=-1,
    )
    criminisi_plans = [
        {"strategy": "criminisi_p9", "family": "criminisi", "patch_size": 9},
        {"strategy": "criminisi_p15", "family": "criminisi", "patch_size": 15},
    ]
    existing_criminisi = {str(p.get("strategy")) for p in plan if p.get("family") == "criminisi"}
    criminisi_plans = [p for p in criminisi_plans if p["strategy"] not in existing_criminisi]
    if criminisi_plans:
        insert_at = last_inpainting_idx + 1 if last_inpainting_idx >= 0 else len(plan)
        plan[insert_at:insert_at] = criminisi_plans
    if not any(p.get("strategy") == "meta_regional" for p in plan):
        plan.append({"strategy": "meta_regional", "family": "meta_regional"})
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


def _emit_progress(
    progress_callback: Callable[[str, dict[str, Any]], None] | None,
    phase: str,
    **details: Any,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(phase, details)
    except Exception as exc:
        logger.debug("Progress callback failed (%s): %s", phase, exc)


def _region_type_from_stats(
    variance_value: float,
    gradient_value: float,
    variance_threshold: float,
    gradient_threshold: float,
) -> str:
    if gradient_value >= gradient_threshold:
        return "strong_edges"
    if variance_value >= variance_threshold:
        return "textured"
    return "homogeneous"


def _split_region_mask(region_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(region_mask > 0)
    if len(ys) <= 1:
        return region_mask.copy(), np.zeros_like(region_mask)

    y_span = int(ys.max() - ys.min())
    x_span = int(xs.max() - xs.min())
    first = np.zeros_like(region_mask)
    second = np.zeros_like(region_mask)
    if x_span >= y_span:
        pivot = int(np.median(xs))
        first[(region_mask > 0) & (np.indices(region_mask.shape)[1] <= pivot)] = 255
        second[(region_mask > 0) & (np.indices(region_mask.shape)[1] > pivot)] = 255
    else:
        pivot = int(np.median(ys))
        first[(region_mask > 0) & (np.indices(region_mask.shape)[0] <= pivot)] = 255
        second[(region_mask > 0) & (np.indices(region_mask.shape)[0] > pivot)] = 255
    return first, second


def _segment_mask_regions(
    image_path: str | Path,
    mask_path: str | Path,
    min_regions: int = 2,
    max_regions: int = 6,
) -> list[dict[str, Any]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ReconstructionError(f"Impossible de charger l'image : {image_path}")
    if mask is None:
        raise ReconstructionError(f"Impossible de charger le masque : {mask_path}")

    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    if not (mask > 0).any():
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (9, 9))
    mean_sq = cv2.blur(gray_f * gray_f, (9, 9))
    local_variance = np.maximum(mean_sq - mean * mean, 0.0)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)

    masked = mask > 0
    variance_threshold = float(np.percentile(local_variance[masked], 60))
    gradient_threshold = float(np.percentile(gradient[masked], 70))

    labels = np.zeros(mask.shape, dtype=np.uint8)
    labels[(local_variance >= variance_threshold) & masked] = 2
    labels[(gradient >= gradient_threshold) & masked] = 3

    regions: list[dict[str, Any]] = []
    for label, fallback_type in [(3, "strong_edges"), (2, "textured"), (0, "homogeneous")]:
        class_mask = np.zeros_like(mask)
        if label == 0:
            class_mask[(labels == 0) & masked] = 255
        else:
            class_mask[labels == label] = 255
        if not (class_mask > 0).any():
            continue

        count, component_labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, 8)
        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            region_mask = np.zeros_like(mask)
            region_mask[component_labels == idx] = 255
            pix = region_mask > 0
            region_type = _region_type_from_stats(
                float(local_variance[pix].mean()),
                float(gradient[pix].mean()),
                variance_threshold,
                gradient_threshold,
            )
            if label == 3:
                region_type = "strong_edges"
            elif label == 2 and region_type == "homogeneous":
                region_type = fallback_type
            regions.append({"mask": region_mask, "type": region_type, "area": area})

    if not regions:
        regions.append({"mask": mask.copy(), "type": "homogeneous", "area": int(masked.sum())})

    regions.sort(key=lambda r: int(r["area"]), reverse=True)
    while len(regions) < min_regions and int(regions[0]["area"]) > 1:
        largest = regions.pop(0)
        first, second = _split_region_mask(largest["mask"])
        split_parts = [m for m in (first, second) if (m > 0).any()]
        if len(split_parts) < 2:
            regions.insert(0, largest)
            break
        for part in split_parts:
            pix = part > 0
            regions.append({
                "mask": part,
                "type": _region_type_from_stats(
                    float(local_variance[pix].mean()),
                    float(gradient[pix].mean()),
                    variance_threshold,
                    gradient_threshold,
                ),
                "area": int(pix.sum()),
            })
        regions.sort(key=lambda r: int(r["area"]), reverse=True)

    if len(regions) > max_regions:
        kept = regions[:max_regions - 1]
        merged_mask = np.zeros_like(mask)
        for region in regions[max_regions - 1:]:
            merged_mask[region["mask"] > 0] = 255
        pix = merged_mask > 0
        kept.append({
            "mask": merged_mask,
            "type": _region_type_from_stats(
                float(local_variance[pix].mean()),
                float(gradient[pix].mean()),
                variance_threshold,
                gradient_threshold,
            ),
            "area": int(pix.sum()),
        })
        regions = kept

    return regions[:max_regions]


def _regional_strategy_plan(region_type: str) -> list[dict[str, Any]]:
    if region_type == "strong_edges":
        return [
            {"strategy": "criminisi_p9", "family": "criminisi", "patch_size": 9},
            {"strategy": "criminisi_p15", "family": "criminisi", "patch_size": 15},
            {"strategy": "inpainting_r3", "family": "inpainting", "radius": 3},
        ]
    if region_type == "textured":
        return [
            {"strategy": "patchmatch_p11", "family": "patchmatch", "patch_size": 11, "iterations": 7},
            {"strategy": "criminisi_p15", "family": "criminisi", "patch_size": 15},
            {"strategy": "patchmatch_p9", "family": "patchmatch", "patch_size": 9, "iterations": 5},
        ]
    return [
        {"strategy": "inpainting_r7", "family": "inpainting", "radius": 7},
        {"strategy": "inpainting_r5", "family": "inpainting", "radius": 5},
        {"strategy": "criminisi_p9", "family": "criminisi", "patch_size": 9},
    ]


def _run_regional_strategy(
    plan: dict[str, Any],
    image_path: Path,
    region_mask_path: Path,
    method: str,
) -> dict[str, Any] | None:
    family = str(plan["family"])
    if family == "inpainting":
        return reconstruct_with_inpaint(
            image_path,
            region_mask_path,
            method=method,
            radius=int(plan["radius"]),
        )
    if family == "criminisi":
        return criminisi_inpaint(
            image_path,
            region_mask_path,
            patch_size=int(plan["patch_size"]),
        )
    if family == "patchmatch" and _PATCHMATCH_AVAILABLE:
        return _patchmatch_inpaint(
            image_path,
            region_mask_path,
            patch_size=int(plan["patch_size"]),
            iterations=int(plan["iterations"]),
        )
    return None


def _blend_region(
    base: np.ndarray,
    candidate: np.ndarray,
    region_mask: np.ndarray,
) -> np.ndarray:
    mask = (region_mask > 0).astype(np.float32)
    kernel = max(5, int(round(min(region_mask.shape) * 0.04)) | 1)
    alpha = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    alpha = np.clip(alpha, 0.0, 1.0)[..., np.newaxis]
    blended = base.astype(np.float32) * (1.0 - alpha) + candidate.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _run_meta_regional_strategy(
    input_image_path: Path,
    scoring_image_path: Path,
    mask_path_obj: Path,
    original_image_path: Path | None,
    method: str,
    out_dir: Path,
) -> dict[str, Any] | None:
    regions = _segment_mask_regions(input_image_path, mask_path_obj)
    if len(regions) < 2:
        return None

    fused = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
    if fused is None:
        raise ReconstructionError(f"Impossible de charger l'image : {input_image_path}")

    region_results: list[dict[str, Any]] = []
    total_area = float(sum(int(r["area"]) for r in regions)) or 1.0

    for idx, region in enumerate(regions):
        region_mask = region["mask"]
        region_mask_path = out_dir / f"{input_image_path.stem}_meta_region_{idx}.png"
        cv2.imwrite(str(region_mask_path), region_mask)
        best_candidate: dict[str, Any] | None = None

        for plan in _regional_strategy_plan(str(region["type"])):
            try:
                result = _run_regional_strategy(plan, input_image_path, region_mask_path, method)
                if not result or not Path(result["path"]).exists():
                    continue
                score, details = _score_candidate(
                    scoring_image_path,
                    result["path"],
                    original_image_path,
                    mask=region_mask,
                )
                candidate = {
                    "strategy": str(plan["strategy"]),
                    "path": result["path"],
                    "score": score,
                    "region_type": region["type"],
                    "region_index": idx,
                    "region_area": int(region["area"]),
                    **details,
                }
                if best_candidate is None or score > float(best_candidate.get("score", 0.0)):
                    best_candidate = candidate
            except Exception as exc:
                logger.debug("Meta-regional candidate failed (%s): %s", plan.get("strategy"), exc)

        if best_candidate is None:
            continue

        candidate_img = cv2.imread(str(best_candidate["path"]), cv2.IMREAD_COLOR)
        if candidate_img is None:
            continue
        fused = _blend_region(fused, candidate_img, region_mask)
        region_results.append(best_candidate)

    if not region_results:
        return None

    output_path = out_dir / f"{input_image_path.stem}_meta_regional.png"
    path = _save_image(output_path, fused)
    weighted_score = sum(
        float(r["score"]) * (float(r["region_area"]) / total_area)
        for r in region_results
    )
    _, details = _score_candidate(scoring_image_path, path, original_image_path)
    return {
        "strategy": "meta_regional",
        "path": path,
        "score": float(weighted_score),
        "family": "meta_regional",
        "regions": [
            {
                "index": int(r["region_index"]),
                "type": str(r["region_type"]),
                "area": int(r["region_area"]),
                "selected_strategy": str(r["strategy"]),
                "score": float(r["score"]),
            }
            for r in region_results
        ],
        "region_count": len(region_results),
        "fusion": "gaussian_blending",
        **{k: v for k, v in details.items() if k != "score"},
    }


def _run_repair_plan_candidate(
    plan: dict[str, Any],
    input_image_path: Path,
    scoring_image_path: Path,
    mask_path_obj: Path | None,
    original_image_path: Path | None,
    method: str,
    base_radius: int,
    out_dir: Path,
    mask_arr: np.ndarray | None,
) -> dict[str, Any] | None:
    family = str(plan["family"])
    strategy = str(plan["strategy"])
    image_np = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
    if image_np is None:
        raise ReconstructionError(f"Impossible de charger l'image : {input_image_path}")

    if family == "composite":
        result_path = _run_composite_strategy(
            plan, input_image_path, mask_path_obj,
            image_np, method, base_radius, out_dir,
        )
        if result_path and Path(result_path).exists():
            return _candidate_from_path(
                strategy, result_path, scoring_image_path, original_image_path,
                extra={"family": family, "steps": plan.get("steps", [])},
                mask=mask_arr,
            )
        return None

    if family == "patchmatch":
        if mask_path_obj is None or not _PATCHMATCH_AVAILABLE:
            return None
        result = _patchmatch_inpaint(
            input_image_path, mask_path_obj,
            patch_size=int(plan["patch_size"]),
            iterations=int(plan["iterations"]),
        )
        return _candidate_from_path(
            strategy, result["path"], scoring_image_path, original_image_path,
            extra={"patch_size": plan["patch_size"],
                   "iterations": plan["iterations"], "family": family},
            mask=mask_arr,
        )

    if family == "inpainting":
        if mask_path_obj is None:
            return None
        result = reconstruct_with_inpaint(
            input_image_path, mask_path_obj,
            method=method, radius=int(plan["radius"]),
        )
        return _candidate_from_path(
            strategy, result["path"], scoring_image_path, original_image_path,
            extra={"radius": int(plan["radius"]), "family": family},
            mask=mask_arr,
        )

    if family == "criminisi":
        if mask_path_obj is None:
            return None
        result = criminisi_inpaint(
            input_image_path, mask_path_obj,
            patch_size=int(plan["patch_size"]),
        )
        return _candidate_from_path(
            strategy, result["path"], scoring_image_path, original_image_path,
            extra={"patch_size": int(plan["patch_size"]), "family": family},
            mask=mask_arr,
        )

    if family == "meta_regional":
        if mask_path_obj is None:
            return None
        return _run_meta_regional_strategy(
            input_image_path=input_image_path,
            scoring_image_path=scoring_image_path,
            mask_path_obj=mask_path_obj,
            original_image_path=original_image_path,
            method=method,
            out_dir=out_dir,
        )

    if family == "denoise":
        result = denoise_image(input_image_path, method=str(plan["denoise_method"]))
        return _candidate_from_path(
            strategy, result["path"], scoring_image_path, original_image_path,
            extra={"family": family},
            mask=mask_arr,
        )

    if family == "deblur":
        deblurred = deblur(image_np)
        if plan.get("strength") == "strong":
            deblurred = deblur(deblurred)
        output = out_dir / f"{input_image_path.stem}_{strategy}.png"
        path = _save_image(output, deblurred)
        return _candidate_from_path(
            strategy, path, scoring_image_path, original_image_path,
            extra={"family": family, "strength": plan.get("strength")},
            mask=mask_arr,
        )

    if family == "block_repair":
        repaired = repair_blocks(image_np)
        output = out_dir / f"{input_image_path.stem}_{strategy}.png"
        path = _save_image(output, repaired)
        return _candidate_from_path(
            strategy, path, scoring_image_path, original_image_path,
            extra={"family": family},
            mask=mask_arr,
        )

    if family == "hybrid":
        if mask_path_obj is None:
            return None
        denoised = denoise_image(input_image_path, method=str(plan["denoise_method"]))
        hybrid = reconstruct_with_inpaint(
            denoised["path"], mask_path_obj,
            method=method, radius=int(plan["radius"]),
        )
        return _candidate_from_path(
            strategy, hybrid["path"], scoring_image_path, original_image_path,
            extra={"family": family, "radius": int(plan["radius"]),
                   "denoise_method": str(plan["denoise_method"])},
            mask=mask_arr,
        )

    return None


def _run_forensic_supreme_candidates(
    corrupted_image_path: Path,
    mask_path_obj: Path | None,
    original_image_path: Path | None,
    method: str,
    base_radius: int,
    corruption_type: str,
    recommended: str,
    out_dir: Path,
    mask_arr: np.ndarray | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exécute toutes les familles et réinjecte le meilleur de chaque famille."""
    plan = _build_forensic_supreme_plan(corruption_type, base_radius, recommended)
    families: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in plan:
        family = str(item["family"])
        if family not in grouped:
            grouped[family] = []
            families.append(family)
        grouped[family].append(item)

    current_input = corrupted_image_path
    candidates: list[dict[str, Any]] = []
    family_chain: list[dict[str, Any]] = []

    for family in families:
        family_candidates: list[dict[str, Any]] = []
        family_input = current_input
        _emit_progress(
            progress_callback, "family_started",
            family=family, input_path=str(family_input),
            strategy_count=len(grouped[family]),
        )

        for item in grouped[family]:
            strategy = str(item["strategy"])
            _emit_progress(
                progress_callback, "strategy_started",
                family=family, strategy=strategy, input_path=str(family_input),
            )
            try:
                candidate = _run_repair_plan_candidate(
                    item, family_input, corrupted_image_path, mask_path_obj,
                    original_image_path, method, base_radius, out_dir, mask_arr,
                )
                if candidate:
                    candidate["supreme_input_path"] = str(family_input)
                    candidate["supreme_family"] = family
                    family_candidates.append(candidate)
                    candidates.append(candidate)
                    _emit_progress(
                        progress_callback, "strategy_completed",
                        family=family, strategy=strategy,
                        score=float(candidate.get("score", 0.0)),
                    )
                else:
                    _emit_progress(
                        progress_callback, "strategy_skipped",
                        family=family, strategy=strategy,
                    )
            except Exception as exc:
                logger.warning("Forensic supreme candidate failed (%s): %s", strategy, exc)
                _emit_progress(
                    progress_callback, "strategy_failed",
                    family=family, strategy=strategy, error=str(exc),
                )

        if family_candidates:
            best = max(family_candidates, key=lambda c: float(c.get("score", 0.0)))
            current_input = Path(best["path"])
            family_chain.append({
                "family": family,
                "input_path": str(family_input),
                "selected_strategy": best.get("strategy"),
                "selected_path": best.get("path"),
                "selected_score": best.get("score"),
            })
            _emit_progress(
                progress_callback, "family_completed",
                family=family, selected_strategy=str(best.get("strategy")),
                selected_path=str(best.get("path")),
                selected_score=float(best.get("score", 0.0)),
            )
        else:
            family_chain.append({
                "family": family,
                "input_path": str(family_input),
                "selected_strategy": None,
                "selected_path": None,
                "selected_score": None,
            })
            _emit_progress(
                progress_callback, "family_completed",
                family=family, selected_strategy=None,
            )

    if mask_path_obj is not None:
        iterative = _run_iterative_pass(
            current_input, mask_path_obj, original_image_path,
            method, base_radius, max_iterations=3, mask_arr=mask_arr,
        )
        if iterative:
            iterative["family"] = "iterative"
            iterative["supreme_family"] = "iterative"
            iterative["supreme_input_path"] = str(current_input)
            candidates.append(iterative)
            _emit_progress(
                progress_callback, "strategy_completed",
                family="iterative", strategy="iterative_inpaint",
                score=float(iterative.get("score", 0.0)),
            )

    return candidates, family_chain


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
    forensic_supreme: bool = False,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
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

    family_chain: list[dict[str, Any]] = []

    if forensic_supreme:
        _emit_progress(
            progress_callback, "repair_started",
            mode="forensic_supreme",
            max_attempts=None,
        )
        supreme_candidates, family_chain = _run_forensic_supreme_candidates(
            corrupted_image_path=corrupted_image_path,
            mask_path_obj=mask_path_obj,
            original_image_path=orig_path,
            method=method,
            base_radius=radius,
            corruption_type=corruption_type,
            recommended=recommended,
            out_dir=out_dir,
            mask_arr=mask_arr,
            progress_callback=progress_callback,
        )
        candidates.extend(supreme_candidates)
    else:
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
        "forensic_supreme":         forensic_supreme,
        "family_chain":             family_chain,
        "max_attempts_applied":     None if forensic_supreme else max_attempts,
    }
    logger.info(
        "Repair pipeline done | selected=%s | score=%.2f | attempts=%s",
        best["strategy"], best["score"], len(candidates),
    )
    return result
