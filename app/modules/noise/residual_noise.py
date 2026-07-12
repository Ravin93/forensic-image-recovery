"""app/modules/noise/residual_noise.py + prnu_analyzer.py — K18.

Analyse du bruit residuel et detection d incoherences par PRNU/SPN.
Utile pour detecter les zones collee ou retouchees.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.logger import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_REPORTS_DIR  = _PROJECT_ROOT / "data" / "reports"
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Extraction bruit residuel
# ---------------------------------------------------------------------------

def extract_residual_noise(image: np.ndarray, wavelet_passes: int = 1) -> np.ndarray:
    """Extrait le bruit residuel par soustraction d un filtre lissant.

    Methode : image - denoised (filtre bilateral).
    Approxime le bruit capteur (SPN) sans librairie wavelet externe.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    # Plusieurs passes de debruitage pour approcher le bruit pur
    denoised = gray.copy()
    for _ in range(wavelet_passes):
        denoised = cv2.bilateralFilter(
            denoised.astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75
        ).astype(np.float32)

    noise = gray - denoised
    return noise


def compute_noise_consistency(
    image_path: str | Path,
    block_size: int = 32,
) -> dict[str, Any]:
    """Calcule la coherence du bruit residuel par blocs.

    Un bloc retouche ou colle aura un profil de bruit different
    du reste de l image (variance anormale).

    Returns:
        dict avec consistency, suspicious_regions, heatmap_path
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de charger : {image_path}")

    h, w = img.shape[:2]
    noise = extract_residual_noise(img)

    # Variance du bruit par bloc
    block_variances: list[float] = []
    block_positions: list[tuple[int,int]] = []

    for by in range(0, h - block_size, block_size):
        for bx in range(0, w - block_size, block_size):
            block = noise[by:by+block_size, bx:bx+block_size]
            var = float(np.var(block))
            block_variances.append(var)
            block_positions.append((by, bx))

    if not block_variances:
        return {
            "image_path": str(image_path), "noise_consistency": 1.0,
            "suspicious_regions": [], "heatmap_path": None,
            "mean_variance": 0.0, "std_variance": 0.0,
        }

    arr_var = np.array(block_variances)
    mean_v  = float(np.mean(arr_var))
    std_v   = float(np.std(arr_var))

    # Regions suspectes : variance deviant de > 2 sigma
    suspicious: list[dict[str, Any]] = []
    threshold  = mean_v + 2.0 * std_v

    for var, (by, bx) in zip(block_variances, block_positions):
        if var > threshold:
            suspicious.append({
                "x": bx, "y": by,
                "w": block_size, "h": block_size,
                "variance": round(var, 3),
                "deviation": round((var - mean_v) / max(std_v, 1e-6), 2),
            })

    # Coherence = 1 - (nb suspects / nb total)
    consistency = float(1.0 - len(suspicious) / max(len(block_variances), 1))

    # Heatmap
    heatmap_path = _generate_heatmap(img, noise, block_variances,
                                      block_positions, block_size, image_path.stem)

    logger.info("NoiseConsistency %s | consistency=%.3f suspects=%d",
                image_path.name, consistency, len(suspicious))

    return {
        "image_path":         str(image_path),
        "noise_consistency":  round(consistency, 3),
        "suspicious_regions": suspicious[:20],
        "heatmap_path":       heatmap_path,
        "mean_variance":      round(mean_v, 3),
        "std_variance":       round(std_v, 3),
        "n_blocks_total":     len(block_variances),
        "n_blocks_suspicious": len(suspicious),
    }


def _generate_heatmap(
    original: np.ndarray,
    noise: np.ndarray,
    variances: list[float],
    positions: list[tuple[int,int]],
    block_size: int,
    stem: str,
) -> str | None:
    """Genere une heatmap coloree des variances de bruit."""
    try:
        h, w = original.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        if not variances:
            return None

        arr_var = np.array(variances)
        vmin, vmax = float(arr_var.min()), float(arr_var.max())
        if vmax == vmin:
            return None

        for var, (by, bx) in zip(variances, positions):
            normalized = (var - vmin) / (vmax - vmin)
            heatmap[by:by+block_size, bx:bx+block_size] = normalized

        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend avec l image originale
        overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

        import time
        ts = int(time.time())
        out_path = _REPORTS_DIR / f"heatmap_{stem}_{ts}.png"
        cv2.imwrite(str(out_path), overlay)
        return str(out_path)
    except Exception as exc:
        logger.debug("Heatmap generation echouee : %s", exc)
        return None


# ---------------------------------------------------------------------------
# PRNU — Photo Response Non-Uniformity
# ---------------------------------------------------------------------------

def compute_prnu_signature(image_path: str | Path) -> np.ndarray:
    """Calcule une signature PRNU approximative de l image.

    Methode simplifiee : bruit residuel normalise par la luminance locale.
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de charger : {image_path}")

    noise = extract_residual_noise(img)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
    prnu  = noise / gray
    return prnu


def compare_prnu(
    path_a: str | Path,
    path_b: str | Path,
) -> dict[str, Any]:
    """Compare les signatures PRNU de deux images.

    Utile pour determiner si deux images proviennent du meme appareil.
    """
    pa = compute_prnu_signature(path_a)
    pb = compute_prnu_signature(path_b)

    # Redimensionner si necessaire
    if pa.shape != pb.shape:
        h = min(pa.shape[0], pb.shape[0])
        w = min(pa.shape[1], pb.shape[1])
        pa = pa[:h, :w]
        pb = pb[:h, :w]

    # Correlation de Pearson
    pa_flat = pa.flatten()
    pb_flat = pb.flatten()
    corr    = float(np.corrcoef(pa_flat, pb_flat)[0, 1])
    if np.isnan(corr):
        corr = 0.0

    same_device = corr > 0.5

    return {
        "path_a":       str(path_a),
        "path_b":       str(path_b),
        "correlation":  round(corr, 4),
        "same_device":  same_device,
        "confidence":   round(abs(corr), 3),
    }