"""app/modules/steganalysis/lsb_analyzer.py — K13.

Detection de steganographie LSB (Least Significant Bit).
Analyse la distribution des bits de poids faible pour detecter
des anomalies statistiques indicatives de donnees cachees.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.logger import logger


def analyze_lsb_distribution(image_path: str | Path) -> dict[str, Any]:
    """Analyse la distribution des LSB par canal RGB.

    Une image naturelle a des LSB ~uniformes (entropie proche de 1.0).
    Un LSB modifie montre une distribution trop uniforme ou trop reguliere.

    Returns:
        dict avec entropie, distribution et stats par canal
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de charger : {image_path}")

    channels = cv2.split(img)
    channel_names = ["blue", "green", "red"]
    channel_stats: dict[str, Any] = {}
    entropies: list[float] = []

    for name, ch in zip(channel_names, channels):
        lsb = ch & 1  # bit de poids faible
        n_pixels = lsb.size
        n_ones   = int(lsb.sum())
        n_zeros  = n_pixels - n_ones
        ratio    = n_ones / n_pixels

        # Entropie de Shannon sur les 2 valeurs possibles (0/1)
        p0 = n_zeros / n_pixels
        p1 = n_ones  / n_pixels
        eps = 1e-10
        entropy = float(-(p0 * np.log2(p0 + eps) + p1 * np.log2(p1 + eps)))
        entropies.append(entropy)

        # Distribution sur les 8 valeurs des 3 LSB
        lsb3 = ch & 0b111
        hist, _ = np.histogram(lsb3, bins=8, range=(0, 8))
        hist_norm = (hist / n_pixels).tolist()

        channel_stats[name] = {
            "n_ones":      n_ones,
            "n_zeros":     n_zeros,
            "ratio_ones":  round(ratio, 4),
            "entropy":     round(entropy, 4),
            "lsb3_hist":   [round(v, 4) for v in hist_norm],
        }

    mean_entropy = float(np.mean(entropies))

    return {
        "image_path":    str(image_path),
        "channels":      channel_stats,
        "mean_entropy":  round(mean_entropy, 4),
        "n_pixels":      int(img.shape[0] * img.shape[1]),
    }


def detect_lsb_anomaly(image_path: str | Path) -> dict[str, Any]:
    """Detecte les anomalies LSB indicatives de steganographie.

    Criteres :
    - Entropie LSB trop proche de 1.0 (tous bits aleatoires = suspect)
    - Distribution 0/1 trop equilibree (ratio proche de 0.5)
    - Variance anormalement faible entre canaux

    Retourne un score d anomalie 0.0-1.0 et un flag suspicious.
    """
    dist = analyze_lsb_distribution(image_path)
    channels = dist["channels"]
    mean_ent = dist["mean_entropy"]

    flags: list[str] = []
    sub_scores: list[float] = []

    for name, stats in channels.items():
        entropy = stats["entropy"]
        ratio   = stats["ratio_ones"]

        # Entropie tres proche de 1.0 = LSB trop aleatoires
        entropy_score = float(max(0.0, (entropy - 0.90) / 0.10))
        sub_scores.append(entropy_score)

        # Ratio trop proche de 0.5 = suspect
        balance_score = float(max(0.0, 1.0 - abs(ratio - 0.5) * 20))
        sub_scores.append(balance_score)

        if entropy > 0.97:
            flags.append(f"Canal {name} : entropie LSB tres elevee ({entropy:.3f})")
        if 0.49 <= ratio <= 0.51:
            flags.append(f"Canal {name} : ratio 0/1 trop equilibre ({ratio:.3f})")

    anomaly_score = float(np.mean(sub_scores)) if sub_scores else 0.0
    anomaly_score = round(min(1.0, max(0.0, anomaly_score)), 3)
    suspicious    = anomaly_score > 0.6

    if suspicious:
        flags.append("Score anomalie eleve : steganographie LSB possible")

    level = "forte" if anomaly_score > 0.75 else "moyenne" if anomaly_score > 0.5 else "faible"

    logger.debug("LSB %s | entropy=%.3f score=%.3f suspicious=%s",
                 Path(image_path).name, mean_ent, anomaly_score, suspicious)

    return {
        "image_path":    str(image_path),
        "lsb_entropy":   mean_ent,
        "anomaly_score": anomaly_score,
        "suspicious":    suspicious,
        "suspicion_level": level,
        "flags":         flags,
        "channel_stats": channels,
    }