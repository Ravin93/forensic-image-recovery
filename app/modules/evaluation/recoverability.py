from typing import Any


def classify_recoverability(evaluation: dict[str, Any]) -> str:
    """
    Classification simple et stable :
    - recoverable
    - partially_recoverable
    - non_recoverable
    """
    corr = evaluation["original_vs_corrupted"]
    reco = evaluation["original_vs_reconstructed"]

    psnr_corr = float(corr["psnr"])
    ssim_corr = float(corr["ssim"])
    psnr_reco = float(reco["psnr"])
    ssim_reco = float(reco["ssim"])

    psnr_gain = psnr_reco - psnr_corr
    ssim_gain = ssim_reco - ssim_corr

    if psnr_gain >= 5.0 and ssim_reco >= 0.97 and ssim_gain > 0:
        return "recoverable"

    if psnr_gain > 0.5 and ssim_reco >= 0.85 and ssim_gain > 0:
        return "partially_recoverable"

    return "non_recoverable"