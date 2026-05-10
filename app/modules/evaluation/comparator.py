from pathlib import Path

from app.modules.evaluation.metrics import (
    compute_psnr,
    compute_ssim,
    compute_supervised_score,
    compute_blind_score,
    score_candidate,
)


def compare_images(
    original_path: str | Path,
    corrupted_path: str | Path,
    reconstructed_path: str | Path,
) -> dict:
    """Compare original / corrompu / reconstruit.

    Retour enrichi (Ticket 4) :
        original_vs_corrupted      psnr + ssim
        original_vs_reconstructed  psnr + ssim  (alias : reconstructed)
        gains                      psnr_gain, ssim_gain, improvement_score
        improvement                bool
        supervised_score           dict complet compute_supervised_score
    """
    psnr_corrupted     = compute_psnr(original_path, corrupted_path)
    ssim_corrupted     = compute_ssim(original_path, corrupted_path)
    psnr_reconstructed = compute_psnr(original_path, reconstructed_path)
    ssim_reconstructed = compute_ssim(original_path, reconstructed_path)

    psnr_gain = float(psnr_reconstructed) - float(psnr_corrupted)
    ssim_gain = float(ssim_reconstructed) - float(ssim_corrupted)

    improvement = (
        (psnr_gain > 1.0 and ssim_gain > 0.0)
        or (psnr_gain > 0.0 and ssim_gain > 0.01)
    )
    improvement_score = 0.7 * psnr_gain + 30.0 * ssim_gain

    # Scoring supervisé complet (Ticket 4)
    supervised = compute_supervised_score(
        original=original_path,
        reconstructed=reconstructed_path,
        corrupted=corrupted_path,
    )

    return {
        # Clés historiques (compatibilité avec les tests existants)
        "original_vs_corrupted": {
            "psnr": psnr_corrupted,
            "ssim": ssim_corrupted,
        },
        "original_vs_reconstructed": {
            "psnr": psnr_reconstructed,
            "ssim": ssim_reconstructed,
        },
        # Alias utilisé par repair_pipeline._score_candidate
        "reconstructed": {
            "psnr": psnr_reconstructed,
            "ssim": ssim_reconstructed,
        },
        "corrupted": {
            "psnr": psnr_corrupted,
            "ssim": ssim_corrupted,
        },
        "gains": {
            "psnr_gain":        psnr_gain,
            "ssim_gain":        ssim_gain,
            "improvement_score": improvement_score,
        },
        "improvement": improvement,
        # Nouveau — Ticket 4
        "supervised_score": supervised,
    }