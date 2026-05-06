from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.exceptions import ReconstructionError
from app.core.logger import logger
from app.modules.evaluation.comparator import compare_images
from app.modules.reconstruction.block_repair import repair_blocks
from app.modules.reconstruction.deblurring import deblur
from app.modules.reconstruction.denoising import denoise_image
from app.modules.reconstruction.inpainting import reconstruct_with_inpaint


def choose_repair_strategy(corruption_type: str, detection_confidence: float) -> str:
    corruption_type = (corruption_type or "").lower()

    if corruption_type in {"mask_like", "missing", "zone_deletion", "bar"} and detection_confidence >= 0.40:
        return "inpainting"
    if corruption_type in {"noise_like", "noise"}:
        return "denoise"
    if corruption_type in {"blur", "blur_like", "local_blur"}:
        return "deblur"
    if corruption_type in {"jpeg_block", "jpeg_block_artifacts", "blocky"}:
        return "block_repair"
    if corruption_type in {"mixed", "combined", "shift_region", "block_dropout"}:
        return "hybrid"
    return "conservative"


def _blind_score(reference_path: str | Path, candidate_path: str | Path) -> tuple[float, dict[str, float]]:
    ref = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    cand = cv2.imread(str(candidate_path), cv2.IMREAD_COLOR)
    if ref is None or cand is None:
        raise ReconstructionError("Impossible de charger l'image pour scoring blind")
    if ref.shape != cand.shape:
        raise ReconstructionError(
            f"Dimensions incompatibles pour scoring blind : {ref.shape} vs {cand.shape}"
        )

    cand_gray = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    lap_var = float(cv2.Laplacian(cand_gray, cv2.CV_64F).var())
    noise_penalty = float(
        np.mean(
            np.abs(
                cand.astype(np.float32)
                - cv2.GaussianBlur(cand, (3, 3), 0).astype(np.float32)
            )
        )
    )
    dynamic_range = float(cand.max() - cand.min())
    mean_delta = float(
        np.mean(np.abs(cand_gray.astype(np.float32) - ref_gray.astype(np.float32)))
    )

    score = (
        0.45 * min(lap_var / 120.0, 1.0)
        + 0.25 * min(dynamic_range / 255.0, 1.0)
        + 0.30 * max(0.0, 1.0 - min(noise_penalty / 40.0, 1.0))
    )
    score *= max(0.2, 1.0 - min(mean_delta / 80.0, 0.8))
    score_100 = float(max(0.0, min(100.0, score * 100.0)))

    metrics = {
        "sharpness": lap_var,
        "noise_penalty": noise_penalty,
        "dynamic_range": dynamic_range,
        "mean_delta_vs_corrupted": mean_delta,
    }
    return score_100, metrics


def _score_candidate(
    corrupted_image_path: str | Path,
    candidate_path: str | Path,
    original_image_path: str | Path | None = None,
) -> tuple[float, dict[str, Any]]:
    if original_image_path is not None:
        comparison = compare_images(
            original_path=original_image_path,
            corrupted_path=corrupted_image_path,
            reconstructed_path=candidate_path,
        )
        gains = comparison.get("gains", {})
        reconstructed = comparison.get("reconstructed", {})

        psnr = float(reconstructed.get("psnr", 0.0))
        ssim = float(reconstructed.get("ssim", 0.0))
        gain_psnr = float(gains.get("psnr_gain", 0.0))
        gain_ssim = float(gains.get("ssim_gain", 0.0))

        score = (
            (ssim * 60.0)
            + (min(psnr, 100.0) * 0.20)
            + (gain_ssim * 30.0)
            + (gain_psnr * 0.20)
        )
        score = float(max(0.0, min(100.0, score)))

        details = {
            "mode": "supervised",
            "comparison": comparison,
            "psnr": psnr,
            "ssim": ssim,
            "gain_psnr": gain_psnr,
            "gain_ssim": gain_ssim,
        }
        return score, details

    score, metrics = _blind_score(corrupted_image_path, candidate_path)
    return score, {
        "mode": "blind",
        "metrics": metrics,
        "psnr": None,
        "ssim": None,
        "gain_psnr": None,
        "gain_ssim": None,
    }


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
) -> dict[str, Any]:
    score, details = _score_candidate(corrupted_image_path, path, original_image_path)
    candidate: dict[str, Any] = {
        "strategy": name,
        "path": path,
        "score": score,
        **details,
    }
    if extra:
        candidate.update(extra)
    return candidate


def _build_candidate_plan(base_radius: int) -> list[dict[str, Any]]:
    return [
        {"strategy": "inpainting_r3", "family": "inpainting", "radius": max(3, base_radius)},
        {"strategy": "inpainting_r5", "family": "inpainting", "radius": max(5, base_radius + 2)},
        {"strategy": "inpainting_r7", "family": "inpainting", "radius": max(7, base_radius + 4)},
        {"strategy": "denoise_median_blur", "family": "denoise", "denoise_method": "median_blur"},
        {"strategy": "denoise_gaussian_blur", "family": "denoise", "denoise_method": "gaussian_blur"},
        {"strategy": "denoise_bilateral", "family": "denoise", "denoise_method": "bilateral"},
        {"strategy": "deblur_light", "family": "deblur", "strength": "light"},
        {"strategy": "deblur_strong", "family": "deblur", "strength": "strong"},
        {"strategy": "block_repair", "family": "block_repair"},
        {
            "strategy": "hybrid_denoise_inpaint",
            "family": "hybrid",
            "denoise_method": "median_blur",
            "radius": max(5, base_radius),
        },
    ]


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

    recommended = choose_repair_strategy(corruption_type, detection_confidence)
    logger.info(
        "Repair pipeline start | image=%s | mask=%s | recommended=%s | confidence=%.3f",
        corrupted_image_path,
        mask_path_obj,
        recommended,
        detection_confidence,
    )

    image_np = cv2.imread(str(corrupted_image_path), cv2.IMREAD_COLOR)
    if image_np is None:
        raise ReconstructionError(f"Impossible de charger l'image : {corrupted_image_path}")

    candidates: list[dict[str, Any]] = []

    candidates.append(
        _candidate_from_path(
            "conservative",
            str(corrupted_image_path),
            corrupted_image_path,
            original_image_path,
            extra={"recommended": recommended},
        )
    )

    candidate_plan = _build_candidate_plan(radius)

    for plan in candidate_plan:
        if len(candidates) >= max_attempts:
            break

        family = str(plan["family"])
        strategy = str(plan["strategy"])

        try:
            if family == "inpainting":
                if mask_path_obj is None:
                    continue
                result = reconstruct_with_inpaint(
                    corrupted_image_path,
                    mask_path_obj,
                    method=method,
                    radius=int(plan["radius"]),
                )
                candidates.append(
                    _candidate_from_path(
                        strategy,
                        result["path"],
                        corrupted_image_path,
                        original_image_path,
                        extra={"radius": int(plan["radius"]), "family": family},
                    )
                )
                continue

            if family == "denoise":
                result = denoise_image(
                    corrupted_image_path,
                    method=str(plan["denoise_method"]),
                )
                candidates.append(
                    _candidate_from_path(
                        strategy,
                        result["path"],
                        corrupted_image_path,
                        original_image_path,
                        extra={"family": family},
                    )
                )
                continue

            if family == "deblur":
                deblurred = deblur(image_np)
                if plan.get("strength") == "strong":
                    deblurred = deblur(deblurred)
                output = corrupted_image_path.parent.parent / "reconstructed" / (
                    f"{corrupted_image_path.stem}_{strategy}.png"
                )
                path = _save_image(output, deblurred)
                candidates.append(
                    _candidate_from_path(
                        strategy,
                        path,
                        corrupted_image_path,
                        original_image_path,
                        extra={"family": family, "strength": plan.get("strength")},
                    )
                )
                continue

            if family == "block_repair":
                repaired = repair_blocks(image_np)
                output = corrupted_image_path.parent.parent / "reconstructed" / (
                    f"{corrupted_image_path.stem}_{strategy}.png"
                )
                path = _save_image(output, repaired)
                candidates.append(
                    _candidate_from_path(
                        strategy,
                        path,
                        corrupted_image_path,
                        original_image_path,
                        extra={"family": family},
                    )
                )
                continue

            if family == "hybrid":
                if mask_path_obj is None:
                    continue
                denoised = denoise_image(
                    corrupted_image_path,
                    method=str(plan["denoise_method"]),
                )
                hybrid = reconstruct_with_inpaint(
                    denoised["path"],
                    mask_path_obj,
                    method=method,
                    radius=int(plan["radius"]),
                )
                candidates.append(
                    _candidate_from_path(
                        strategy,
                        hybrid["path"],
                        corrupted_image_path,
                        original_image_path,
                        extra={
                            "family": family,
                            "radius": int(plan["radius"]),
                            "denoise_method": str(plan["denoise_method"]),
                        },
                    )
                )
                continue

        except Exception as exc:
            logger.warning("Repair candidate failed (%s): %s", strategy, exc)

    if not candidates:
        raise ReconstructionError("Aucune tentative de reconstruction valide")

    candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    best = candidates[0]

    result = {
        "path": best["path"],
        "status": "reconstructed",
        "source_image": corrupted_image_path.name,
        "selected_repair_strategy": best["strategy"],
        "repair_strategy": best["strategy"],
        "recommended_strategy": recommended,
        "detection_confidence": detection_confidence,
        "score": best["score"],
        "candidates": candidates,
        "retry_count": max(0, len(candidates) - 1),
        "method": method,
        "corruption_type": corruption_type,
    }
    logger.info(
        "Repair pipeline done | selected=%s | score=%.2f | attempts=%s",
        best["strategy"],
        best["score"],
        len(candidates),
    )
    return result
