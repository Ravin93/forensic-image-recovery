"""app/modules/reconstruction/patchmatch.py — K8.

PatchMatch / exemplar-based inpainting.
Algorithme inspire de Barnes et al. 2009.
Fonctionne sans GPU, remplace progressivement la zone masquee
par des patches similaires trouves dans le reste de l image.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import build_reconstructed_image_path, ensure_directories
from app.core.exceptions import ReconstructionError
from app.core.logger import logger


# ---------------------------------------------------------------------------
# Algorithme PatchMatch exemplar-based
# ---------------------------------------------------------------------------

def _find_best_patch(
    source: np.ndarray,
    mask: np.ndarray,
    patch_center: tuple[int, int],
    patch_size: int,
) -> tuple[int, int]:
    """Trouve le patch le plus similaire dans la region non masquee.

    Parcourt les candidats non masques et retourne le centre du meilleur patch
    (celui qui minimise la SSD sur les pixels connus).
    """
    h, w = source.shape[:2]
    half = patch_size // 2
    cy, cx = patch_center

    # Extraire le patch cible (zone connue uniquement)
    y1, y2 = max(0, cy - half), min(h, cy + half + 1)
    x1, x2 = max(0, cx - half), min(w, cx + half + 1)
    target_patch = source[y1:y2, x1:x2].astype(np.float32)
    target_mask  = (mask[y1:y2, x1:x2] == 0)  # True = pixel connu

    if not target_mask.any():
        return cy, cx  # aucun pixel connu → retourner le centre

    best_ssd   = float("inf")
    best_cy_cx = (cy, cx)

    # Echantillonner des candidats aleatoirement (eviter O(n²))
    step = max(1, patch_size // 2)
    for cand_y in range(half, h - half, step):
        for cand_x in range(half, w - half, step):
            # Le patch candidat ne doit pas contenir de pixels masques
            cm = mask[cand_y - half:cand_y + half + 1,
                      cand_x - half:cand_x + half + 1]
            if cm.any():
                continue

            cand_patch = source[cand_y - half:cand_y + half + 1,
                                cand_x - half:cand_x + half + 1].astype(np.float32)

            if cand_patch.shape != target_patch.shape:
                continue

            # SSD sur les pixels connus uniquement
            diff = (cand_patch - target_patch) ** 2
            if target_mask.ndim == 2:
                ssd = float(diff[target_mask].sum())
            else:
                ssd = float(diff[target_mask[..., np.newaxis].repeat(3, axis=2)].sum())

            if ssd < best_ssd:
                best_ssd   = ssd
                best_cy_cx = (cand_y, cand_x)

    return best_cy_cx


def _compute_fill_priority(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Calcule la priorite de remplissage pour chaque pixel masque.

    Pixels sur les contours des zones masquees sont traites en premier
    (strategie onion-peel).
    """
    # Eroder le masque pour trouver la frontiere
    kernel   = np.ones((3, 3), np.uint8)
    eroded   = cv2.erode(mask, kernel, iterations=1)
    frontier = mask - eroded  # pixels en bordure de la zone masquee
    return frontier


def patchmatch_inpaint(
    image_path: str | Path,
    mask_path: str | Path,
    patch_size: int = 7,
    iterations: int = 5,
) -> dict[str, Any]:
    """Reconstruit une image par exemplar-based inpainting (PatchMatch simplifie).

    Args:
        image_path : image corrompue
        mask_path  : masque (blanc = zone a reconstruire)
        patch_size : taille des patches (7, 9 ou 11 recommande)
        iterations : nombre de passes de remplissage

    Returns:
        dict compatible run_repair_pipeline
    """
    ensure_directories()
    image_path = Path(image_path)
    mask_path  = Path(mask_path)

    if not image_path.exists():
        raise ReconstructionError(f"Image introuvable : {image_path}")
    if not mask_path.exists():
        raise ReconstructionError(f"Masque introuvable : {mask_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ReconstructionError(f"Impossible de charger : {image_path}")
    if mask is None:
        raise ReconstructionError(f"Impossible de charger le masque : {mask_path}")

    # Binariser le masque (blanc = zone masquee)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    h, w    = image.shape[:2]
    result  = image.copy()
    half    = patch_size // 2

    t0 = time.perf_counter()

    for iteration in range(iterations):
        frontier = _compute_fill_priority(mask, mask)
        frontier_pixels = list(zip(*np.where(frontier > 0)))

        if not frontier_pixels:
            logger.debug("PatchMatch: frontiere vide a iteration %d", iteration)
            break

        # Melanger pour eviter les artefacts directionnels
        np.random.shuffle(frontier_pixels)

        for (py, px) in frontier_pixels:
            best_cy, best_cx = _find_best_patch(result, mask, (py, px), patch_size)

            # Copier le patch source vers la zone cible
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    ty, tx = py + dy, px + dx
                    sy, sx = best_cy + dy, best_cx + dx
                    if 0 <= ty < h and 0 <= tx < w and 0 <= sy < h and 0 <= sx < w:
                        if mask[ty, tx] > 0:
                            result[ty, tx] = result[sy, sx]
                            mask[ty, tx]   = 0  # marquer comme rempli

        # Arreter si tout est rempli
        if mask.max() == 0:
            logger.debug("PatchMatch: masque vide apres iteration %d", iteration)
            break

    elapsed = time.perf_counter() - t0

    # Sauvegarde
    method_name = f"patchmatch_p{patch_size}_i{iterations}"
    output_path = build_reconstructed_image_path(image_path.name, method_name)
    ok = cv2.imwrite(str(output_path), result)
    if not ok:
        raise ReconstructionError(f"Impossible d'ecrire : {output_path}")

    logger.info("PatchMatch OK | patch=%d iter=%d | %.2fs | %s",
                patch_size, iterations, elapsed, output_path.name)

    return {
        "file":         output_path.name,
        "path":         str(output_path),
        "method":       method_name,
        "status":       "reconstructed",
        "source_image": image_path.name,
        "mask_path":    str(mask_path),
        "patch_size":   patch_size,
        "iterations":   iterations,
        "elapsed_s":    round(elapsed, 3),
    }