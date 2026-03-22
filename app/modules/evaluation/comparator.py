from pathlib import Path

from app.modules.evaluation.metrics import compute_psnr, compute_ssim


def compare_images(
    original_path: str | Path,
    corrupted_path: str | Path,
    reconstructed_path: str | Path,
) -> dict:
    psnr_corrupted = compute_psnr(original_path, corrupted_path)
    ssim_corrupted = compute_ssim(original_path, corrupted_path)

    psnr_reconstructed = compute_psnr(original_path, reconstructed_path)
    ssim_reconstructed = compute_ssim(original_path, reconstructed_path)

    psnr_gain = float(psnr_reconstructed) - float(psnr_corrupted)
    ssim_gain = float(ssim_reconstructed) - float(ssim_corrupted)

    improvement = (
        (psnr_gain > 1.0 and ssim_gain > 0.0)
        or
        (psnr_gain > 0.0 and ssim_gain > 0.01)
    )

    improvement_score = 0.7 * psnr_gain + 30.0 * ssim_gain

    return {
        "original_vs_corrupted": {
            "psnr": psnr_corrupted,
            "ssim": ssim_corrupted,
        },
        "original_vs_reconstructed": {
            "psnr": psnr_reconstructed,
            "ssim": ssim_reconstructed,
        },
        "gains": {
            "psnr_gain": psnr_gain,
            "ssim_gain": ssim_gain,
            "improvement_score": improvement_score,
        },
        "improvement": improvement,
    }