from pathlib import Path
from typing import Any

from app.core.logger import logger
from app.modules.evaluation.metrics import (
    compute_psnr,
    compute_ssim,
    load_image_as_rgb_array,
)


def compare_images(
    original_path: str | Path,
    corrupted_path: str | Path,
    reconstructed_path: str | Path,
) -> dict[str, Any]:
    original = load_image_as_rgb_array(original_path)
    corrupted = load_image_as_rgb_array(corrupted_path)
    reconstructed = load_image_as_rgb_array(reconstructed_path)

    original_vs_corrupted = {
        "psnr": compute_psnr(original, corrupted),
        "ssim": compute_ssim(original, corrupted),
    }

    original_vs_reconstructed = {
        "psnr": compute_psnr(original, reconstructed),
        "ssim": compute_ssim(original, reconstructed),
    }

    improvement = (
        original_vs_reconstructed["psnr"] >= original_vs_corrupted["psnr"]
        and original_vs_reconstructed["ssim"] >= original_vs_corrupted["ssim"]
    )

    result = {
        "original_vs_corrupted": original_vs_corrupted,
        "original_vs_reconstructed": original_vs_reconstructed,
        "improvement": improvement,
    }

    logger.info("Comparaison OK : %s", result)
    return result