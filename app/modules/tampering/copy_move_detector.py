"""app/modules/tampering/copy_move_detector.py — K17.

Detecte les zones copiees-collees dans une image (copy-move forgery).
Utilise ORB (toujours disponible dans OpenCV) pour le feature matching.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.logger import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MASKS_DIR    = _PROJECT_ROOT / "data" / "masks"
_MASKS_DIR.mkdir(parents=True, exist_ok=True)


def detect_copy_move(
    image_path: str | Path,
    min_matches: int = 10,
    min_distance_px: int = 10,
) -> dict[str, Any]:
    """Detecte les regions copiees-collees par feature matching ORB.

    Algorithme :
    1. Extraire les keypoints ORB
    2. Matcher les descripteurs avec BFMatcher
    3. Filtrer les matches dont les deux points sont
       suffisamment eloignes (copies depuis une autre zone)
    4. Generer un masque des zones suspectes

    Args:
        image_path    : chemin de l image a analyser
        min_matches   : nombre minimum de matches pour declarer suspicion
        min_distance_px : distance minimale en pixels entre les deux zones

    Returns:
        dict avec suspicious, matches, regions, mask_path, confidence
    """
    t0 = time.perf_counter()
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Impossible de charger : {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ORB detector (sans GPU, robuste)
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    result_base: dict[str, Any] = {
        "image_path":   str(image_path),
        "suspicious":   False,
        "matches":      0,
        "strong_matches": 0,
        "regions":      [],
        "mask_path":    None,
        "confidence":   0.0,
        "elapsed_s":    0.0,
        "n_keypoints":  len(keypoints) if keypoints else 0,
    }

    if descriptors is None or len(keypoints) < 2:
        result_base["elapsed_s"] = round(time.perf_counter() - t0, 3)
        return result_base

    # BFMatcher avec distance Hamming (adapte a ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors, descriptors)

    # Filtrer : garder les matches entre keypoints eloignes
    strong: list[dict[str, Any]] = []
    for m in matches:
        if m.queryIdx == m.trainIdx:
            continue  # meme point
        pt1 = keypoints[m.queryIdx].pt
        pt2 = keypoints[m.trainIdx].pt
        dist = float(np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1]))
        if dist >= min_distance_px and m.distance < 50:
            strong.append({
                "pt1": (int(pt1[0]), int(pt1[1])),
                "pt2": (int(pt2[0]), int(pt2[1])),
                "descriptor_dist": int(m.distance),
                "spatial_dist_px": round(dist, 1),
            })

    n_strong = len(strong)
    suspicious = n_strong >= min_matches

    # Masque des zones suspectes
    mask_path_str = None
    if suspicious:
        mask = np.zeros((h, w), dtype=np.uint8)
        radius = max(10, min(w, h) // 40)
        for m in strong[:100]:
            cv2.circle(mask, m["pt1"], radius, 255, -1)
            cv2.circle(mask, m["pt2"], radius, 255, -1)
        ts = int(time.time())
        mask_path = _MASKS_DIR / f"copymove_{image_path.stem}_{ts}.png"
        cv2.imwrite(str(mask_path), mask)
        mask_path_str = str(mask_path)

    # Regions suspectes (clusters grossiers)
    regions = _cluster_regions(strong[:50]) if suspicious else []

    # Score de confiance
    confidence = min(1.0, n_strong / 50.0) if suspicious else 0.0

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(
        "CopyMove | %s | matches=%d strong=%d suspicious=%s | %.3fs",
        image_path.name, len(matches), n_strong, suspicious, elapsed,
    )

    return {
        "image_path":     str(image_path),
        "suspicious":     suspicious,
        "matches":        len(matches),
        "strong_matches": n_strong,
        "regions":        regions,
        "mask_path":      mask_path_str,
        "confidence":     round(confidence, 3),
        "elapsed_s":      elapsed,
        "n_keypoints":    len(keypoints),
    }


def _cluster_regions(matches: list[dict]) -> list[dict[str, Any]]:
    """Groupe les matches proches en regions suspectes."""
    if not matches:
        return []
    regions = []
    used = set()
    for i, m in enumerate(matches):
        if i in used:
            continue
        cluster = [m]
        used.add(i)
        for j, m2 in enumerate(matches):
            if j in used:
                continue
            d = float(np.hypot(
                m["pt1"][0] - m2["pt1"][0],
                m["pt1"][1] - m2["pt1"][1],
            ))
            if d < 40:
                cluster.append(m2)
                used.add(j)
        if len(cluster) >= 3:
            xs = [c["pt1"][0] for c in cluster]
            ys = [c["pt1"][1] for c in cluster]
            regions.append({
                "x": min(xs), "y": min(ys),
                "w": max(xs) - min(xs) + 20,
                "h": max(ys) - min(ys) + 20,
                "n_matches": len(cluster),
            })
    return regions