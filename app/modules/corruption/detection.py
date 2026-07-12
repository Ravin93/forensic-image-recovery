from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.modules.detection.advanced_detector import (
    detect_dark_regions,
    detect_gradient_breaks,
    detect_low_variance_regions,
)
from app.modules.detection.fusion import fuse_detection_masks


# ---------------------------------------------------------------------------
# Détection basique — seuil adaptatif + filtrage morphologique
# ---------------------------------------------------------------------------

def detect_dark_regions_mask(
    image_path: str | Path,
    threshold: int = 22,
) -> Image.Image:
    """Détecte les zones noires/corrompues avec seuil adaptatif.

    Améliorations vs version originale :
    - Seuil adaptatif basé sur le percentile bas de l'image (évite les faux
      positifs sur images naturellement sombres comme couchers de soleil)
    - Filtrage morphologique : supprime les pixels isolés, ne garde que les
      blocs continus (min_area=200px)
    - Détection complémentaire des zones uniformes (fill_mode=black/white/gray)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        # Fallback PIL
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- Seuil adaptatif ---
    # Le seuil fixe de 22 détecte trop de faux positifs sur images sombres.
    # On calcule le 5e percentile de luminosité : si l'image est globalement
    # sombre (coucher de soleil, nuit), on abaisse le seuil pour ne détecter
    # que les zones vraiment anormales (corruption = noir absolu).
    p5 = float(np.percentile(gray, 5))
    p50 = float(np.percentile(gray, 50))

    # Zone noire artificielle = bien en dessous du minimum naturel
    # Si l'image est claire (p50 > 80), seuil = 30
    # Si l'image est sombre (p50 < 40), seuil = max(p5 * 0.6, 8)
    adaptive_threshold = max(8, min(30, p5 * 0.6 if p50 < 60 else threshold))

    # --- Masque zones noires ---
    black_mask = (gray < adaptive_threshold).astype(np.uint8) * 255

    # --- Masque zones blanches (fill_mode=white) ---
    white_mask = (gray > 253).astype(np.uint8) * 255

    # --- Masque zones grises uniformes (fill_mode=gray, valeur ~127) ---
    gray_mask = (
        (gray > 120) & (gray < 135) &
        (cv2.Laplacian(gray, cv2.CV_64F).var() < 5.0)
    ).astype(np.uint8) * 255

    combined = np.maximum(black_mask, np.maximum(white_mask, gray_mask))

    # --- Filtrage morphologique ---
    # Supprime les pixels isolés (bruit) et ne garde que les blocs compacts
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- Filtrage par aire minimale ---
    # Une zone corrompue artificielle est un bloc continu ≥ 200 pixels
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
    filtered = np.zeros_like(combined)
    min_area = max(200, int(h * w * 0.0005))  # 0.05% de l'image minimum
    max_area = int(h * w * 0.60)              # pas plus de 60% de l'image

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtered[labels == i] = 255

    return Image.fromarray(filtered)


# ---------------------------------------------------------------------------
# Détection avancée — fusion multi-critères améliorée
# ---------------------------------------------------------------------------

def detect_advanced_mask(
    image_path: str | Path,
    strategy: str = "weighted_union",
) -> dict[str, Any]:
    """Détection avancée par fusion de plusieurs détecteurs.

    Utilise detect_dark_regions_mask (version améliorée) comme base,
    croisée avec variance et gradient pour réduire les faux positifs.
    """
    # Masque de base amélioré
    base_mask = np.array(detect_dark_regions_mask(image_path))

    # Détecteurs complémentaires
    dark = detect_dark_regions(image_path)
    variance = detect_low_variance_regions(image_path)
    gradient = detect_gradient_breaks(image_path)

    dark_bin = (base_mask > 0)

    # Croisement : ne garder que ce que notre masque amélioré confirme
    variance_mask = ((variance["mask"] > 0) & dark_bin).astype("uint8") * 255
    gradient_mask = ((gradient["mask"] > 0) & dark_bin).astype("uint8") * 255

    fusion = fuse_detection_masks(
        [base_mask, variance_mask, gradient_mask],
        strategy=strategy,
        min_area=120,
        max_area_ratio=0.12,
        dilate_iter=1,
        return_metadata=True,
    )

    return {
        "mask":       Image.fromarray(fusion["mask"]),
        "method":     "advanced_fusion_v2",
        "confidence": fusion["confidence"],
        "area_ratio": fusion["area_ratio"],
        "components": {
            "dark_regions":  dark["score"],
            "low_variance":  variance["score"],
            "gradient_breaks": gradient["score"],
        },
    }