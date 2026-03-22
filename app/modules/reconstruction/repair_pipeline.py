from pathlib import Path

from app.core.exceptions import ReconstructionError
from app.core.logger import logger
from app.modules.reconstruction.denoising import denoise_image
from app.modules.reconstruction.inpainting import reconstruct_with_inpaint


def choose_repair_strategy(corruption_type: str, detection_confidence: float) -> str:
    if corruption_type == "mask_like" and detection_confidence >= 0.60:
        return "inpainting"

    if corruption_type == "noise_like":
        if detection_confidence < 0.35:
            return "conservative"
        return "denoise"

    if corruption_type == "mixed":
        if detection_confidence >= 0.45:
            return "hybrid"
        return "conservative"

    if detection_confidence >= 0.70:
        return "inpainting"

    return "conservative"


def run_repair_pipeline(
    corrupted_image_path: str | Path,
    mask_path: str | Path | None = None,
    method: str = "opencv_inpaint",
    radius: int = 3,
    corruption_type: str = "unknown",
    detection_confidence: float | None = None,
) -> dict:
    corrupted_image_path = Path(corrupted_image_path)

    if not corrupted_image_path.exists():
        raise ReconstructionError(f"Image corrompue introuvable : {corrupted_image_path}")

    # --- MODE LEGACY : utilisé par les anciens tests ---
    if detection_confidence is None:
        if mask_path is None:
            raise ReconstructionError("mask_path requis pour la reconstruction")

        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise ReconstructionError(f"Masque introuvable : {mask_path}")

        if method != "opencv_inpaint":
            raise ReconstructionError(f"Méthode non supportée : {method}")

        logger.info(
            "Repair pipeline start | image=%s | mask=%s | method=%s",
            corrupted_image_path,
            mask_path,
            method,
        )

        result = reconstruct_with_inpaint(
            image_path=corrupted_image_path,
            mask_path=mask_path,
            method=method,
            radius=radius,
        )

        result["selected_repair_strategy"] = "inpainting"
        result["detection_confidence"] = None

        logger.info("Repair pipeline done")
        return result

    # --- MODE ADAPTATIF AVANCÉ ---
    strategy = choose_repair_strategy(corruption_type, detection_confidence)

    logger.info(
        "Repair pipeline start | image=%s | mask=%s | strategy=%s | confidence=%.3f",
        corrupted_image_path,
        mask_path,
        strategy,
        detection_confidence,
    )

    if strategy == "inpainting":
        if mask_path is None:
            raise ReconstructionError("mask_path requis pour la stratégie inpainting")

        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise ReconstructionError(f"Masque introuvable : {mask_path}")

        result = reconstruct_with_inpaint(
            image_path=corrupted_image_path,
            mask_path=mask_path,
            method=method,
            radius=radius,
        )

    elif strategy == "denoise":
        result = denoise_image(corrupted_image_path, method="median_blur")
        result["repair_strategy"] = "denoise"

    elif strategy == "hybrid":
        if mask_path is None:
            raise ReconstructionError("mask_path requis pour la stratégie hybrid")

        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise ReconstructionError(f"Masque introuvable : {mask_path}")

        denoised = denoise_image(corrupted_image_path, method="median_blur")
        result = reconstruct_with_inpaint(
            image_path=denoised["path"],
            mask_path=mask_path,
            method=method,
            radius=radius,
        )
        result["repair_strategy"] = "hybrid"

    elif strategy == "conservative":
        result = {
            "path": str(corrupted_image_path),
            "status": "reused_corrupted",
            "source_image": corrupted_image_path.name,
            "selected_repair_strategy": "conservative",
            "repair_strategy": "conservative",
            "method": "none",
        }

    else:
        raise ReconstructionError(f"Stratégie non supportée : {strategy}")

    result["selected_repair_strategy"] = strategy
    result["detection_confidence"] = detection_confidence

    logger.info("Repair pipeline done")
    return result