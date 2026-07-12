"""app/modules/evaluation/metrics.py — D1+D2+D3+D4.

Scoring supervisé et aveugle enrichis avec score_breakdown détaillé.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from app.core.exceptions import EvaluationError


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_image_as_rgb_array(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    path = Path(image)
    if not path.exists():
        raise EvaluationError(f"Image introuvable : {path}")
    try:
        return np.array(Image.open(path).convert("RGB"))
    except Exception as exc:
        raise EvaluationError(f"Impossible de charger l'image : {path}") from exc


def load_image_as_array(image: str | Path | np.ndarray) -> np.ndarray:
    return load_image_as_rgb_array(image)


def ensure_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise EvaluationError(f"Dimensions incompatibles : {a.shape} vs {b.shape}")


# ---------------------------------------------------------------------------
# Métriques de base
# ---------------------------------------------------------------------------

def compute_psnr(
    image_a: str | Path | np.ndarray,
    image_b: str | Path | np.ndarray,
) -> float:
    arr_a = load_image_as_rgb_array(image_a)
    arr_b = load_image_as_rgb_array(image_b)
    ensure_same_shape(arr_a, arr_b)
    if np.array_equal(arr_a, arr_b):
        return 100.0
    value = peak_signal_noise_ratio(arr_a, arr_b, data_range=255)
    return 100.0 if np.isinf(value) else float(value)


def compute_ssim(
    image_a: str | Path | np.ndarray,
    image_b: str | Path | np.ndarray,
) -> float:
    arr_a = load_image_as_rgb_array(image_a)
    arr_b = load_image_as_rgb_array(image_b)
    ensure_same_shape(arr_a, arr_b)
    return float(structural_similarity(arr_a, arr_b, channel_axis=2, data_range=255))


# ---------------------------------------------------------------------------
# D1 — Score supervisé détaillé (original disponible)
# ---------------------------------------------------------------------------

def compute_supervised_score(
    original: str | Path | np.ndarray,
    reconstructed: str | Path | np.ndarray,
    corrupted: str | Path | np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score 0-100 supervisé avec breakdown détaillé.

    Nouveautés D1 :
    - score_mask_region   : SSIM sur la zone masquée uniquement
    - score_outside       : SSIM hors masque (préservation)
    - outside_preservation: ratio de conservation hors zone
    - score_breakdown     : décomposition de chaque composante
    """
    arr_orig  = load_image_as_rgb_array(original)
    arr_recon = load_image_as_rgb_array(reconstructed)
    arr_corr  = load_image_as_rgb_array(corrupted)
    ensure_same_shape(arr_orig, arr_recon)

    psnr_recon = compute_psnr(arr_orig, arr_recon)
    ssim_recon = compute_ssim(arr_orig, arr_recon)
    psnr_corr  = compute_psnr(arr_orig, arr_corr)
    ssim_corr  = compute_ssim(arr_orig, arr_corr)

    gain_psnr = float(psnr_recon) - float(psnr_corr)
    gain_ssim = float(ssim_recon) - float(ssim_corr)

    # --- Scores zonaux (D1) ---
    score_mask_region   = None
    score_outside       = None
    outside_preservation = None
    zone_ssim           = None
    bin_mask            = None

    if mask is not None and mask.shape[:2] == arr_orig.shape[:2]:
        bin_mask = (mask > 128).astype(bool)
        if bin_mask.any():
            # Zone masquée — RMSE-based (much more sensitive than nMSE)
            orig_m  = arr_orig[bin_mask].reshape(-1, 3)
            recon_m = arr_recon[bin_mask].reshape(-1, 3)
            rmse_m = float(np.sqrt(np.mean((orig_m.astype(np.float32) - recon_m.astype(np.float32)) ** 2)))
            score_mask_region = float(max(0.0, 1.0 - rmse_m / 255.0)) * 100.0

            # Zone SSIM via bounding box (captures structural quality in the damaged area)
            rows_m = np.any(bin_mask, axis=1)
            cols_m = np.any(bin_mask, axis=0)
            r0 = int(np.where(rows_m)[0][0]);  r1 = int(np.where(rows_m)[0][-1]) + 1
            c0 = int(np.where(cols_m)[0][0]);  c1 = int(np.where(cols_m)[0][-1]) + 1
            pad = 8
            r0 = max(0, r0 - pad); r1 = min(arr_orig.shape[0], r1 + pad)
            c0 = max(0, c0 - pad); c1 = min(arr_orig.shape[1], c1 + pad)
            crop_orig  = arr_orig[r0:r1, c0:c1]
            crop_recon = arr_recon[r0:r1, c0:c1]
            if min(crop_orig.shape[:2]) >= 7:
                zone_ssim = float(structural_similarity(crop_orig, crop_recon, channel_axis=2, data_range=255))
            else:
                rmse_z = float(np.sqrt(np.mean((crop_orig.astype(np.float32) - crop_recon.astype(np.float32)) ** 2)))
                zone_ssim = float(max(0.0, 1.0 - rmse_z / 255.0))

        not_mask = ~bin_mask
        if not_mask.any():
            # Hors masque
            orig_o  = arr_orig[not_mask].reshape(-1, 3)
            recon_o = arr_recon[not_mask].reshape(-1, 3)
            corr_o  = arr_corr[not_mask].reshape(-1, 3)
            mse_outside_recon = float(np.mean((orig_o.astype(np.float32) - recon_o.astype(np.float32)) ** 2))
            mse_outside_corr  = float(np.mean((orig_o.astype(np.float32) - corr_o.astype(np.float32)) ** 2))
            score_outside = float(max(0.0, 1.0 - mse_outside_recon / (255.0 ** 2))) * 100.0
            outside_preservation = (
                float(max(0.0, 1.0 - mse_outside_recon / max(mse_outside_corr, 1e-6)))
            )

    # --- Score global (zone-weighted when mask available) ---
    if zone_ssim is not None and bin_mask is not None:
        # The smaller the corrupted zone, the more global SSIM hides its quality.
        # Blend zone SSIM and global SSIM: weight zone more heavily for small masks.
        coverage = float(np.sum(bin_mask)) / float(bin_mask.size)
        # zone_weight: 0.7 for ≤5% masks, down to 0.3 for ≥25% masks
        zone_weight = float(max(0.3, 0.8 - coverage * 2.0))
        effective_ssim = (1.0 - zone_weight) * ssim_recon + zone_weight * zone_ssim
    else:
        effective_ssim = ssim_recon

    score = (
        effective_ssim * 60.0
        + min(float(psnr_recon), 100.0) * 0.20
        + max(0.0, gain_ssim) * 30.0
        + max(0.0, gain_psnr) * 0.20
    )
    score = float(max(0.0, min(100.0, score)))

    # --- D3 : score_breakdown ---
    score_breakdown: dict[str, Any] = {
        "global_score":          score,
        "ssim_component":        round(effective_ssim * 60.0, 3),
        "zone_ssim":             round(zone_ssim, 4) if zone_ssim is not None else None,
        "psnr_component":        round(min(float(psnr_recon), 100.0) * 0.20, 3),
        "gain_ssim_component":   round(max(0.0, gain_ssim) * 30.0, 3),
        "gain_psnr_component":   round(max(0.0, gain_psnr) * 0.20, 3),
        "mask_region_score":     round(score_mask_region, 2) if score_mask_region is not None else None,
        "outside_preservation":  round(outside_preservation * 100.0, 2) if outside_preservation is not None else None,
        "outside_score":         round(score_outside, 2) if score_outside is not None else None,
    }

    return {
        "mode":                  "supervised",
        "score":                 score,
        "psnr":                  float(psnr_recon),
        "ssim":                  float(ssim_recon),
        "zone_ssim":             zone_ssim,
        "gain_psnr":             gain_psnr,
        "gain_ssim":             gain_ssim,
        "psnr_corrupted":        float(psnr_corr),
        "ssim_corrupted":        float(ssim_corr),
        "mask_region_score":     score_mask_region,
        "outside_score":         score_outside,
        "outside_preservation":  outside_preservation,
        "score_breakdown":       score_breakdown,
    }


# ---------------------------------------------------------------------------
# D2 — Score aveugle détaillé (sans original)
# ---------------------------------------------------------------------------

def compute_blind_score(
    corrupted: str | Path | np.ndarray,
    reconstructed: str | Path | np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score aveugle 0-100 enrichi.

    Nouveautés D2 :
    - coherence_color   : cohérence couleur (histogramme)
    - edge_continuity   : continuité des contours (Canny)
    - local_entropy     : entropie locale (complexité texturale)
    - artifact_score    : artefacts de blocs
    - score_breakdown   : décomposition de chaque composante
    """
    arr_c = load_image_as_rgb_array(corrupted)
    arr_r = load_image_as_rgb_array(reconstructed)

    # Chargement BGR pour OpenCV
    if isinstance(corrupted, np.ndarray):
        bgr_c = cv2.cvtColor(arr_c, cv2.COLOR_RGB2BGR)
    else:
        _bgr = cv2.imread(str(corrupted), cv2.IMREAD_COLOR)
        bgr_c = _bgr if _bgr is not None else cv2.cvtColor(arr_c, cv2.COLOR_RGB2BGR)

    bgr_r  = cv2.cvtColor(arr_r, cv2.COLOR_RGB2BGR)
    gray_r = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2GRAY)
    gray_c = cv2.cvtColor(bgr_c, cv2.COLOR_BGR2GRAY)

    # 1. Netteté (variance Laplacien)
    sharpness = float(cv2.Laplacian(gray_r, cv2.CV_64F).var())
    sharpness_score = float(min(sharpness / 120.0, 1.0))

    # 2. Bruit résiduel
    noise_penalty = float(np.mean(np.abs(
        bgr_r.astype(np.float32)
        - cv2.GaussianBlur(bgr_r, (3, 3), 0).astype(np.float32)
    )))
    noise_score = float(max(0.0, 1.0 - min(noise_penalty / 40.0, 1.0)))

    # 3. Continuité des contours (D2 : Canny au lieu de Sobel brut)
    edges_r = cv2.Canny(gray_r, 50, 150)
    edges_c = cv2.Canny(gray_c, 50, 150)
    edge_density_r = float(np.mean(edges_r > 0))
    edge_density_c = float(np.mean(edges_c > 0))
    # On récompense une densité de contours proche de l'image corrompue
    edge_continuity = float(max(0.0, 1.0 - abs(edge_density_r - edge_density_c) / max(edge_density_c, 0.001)))
    edge_continuity = min(edge_continuity, 1.0)

    # 4. Cohérence couleur (D2 : histogramme)
    color_scores = []
    for ch in range(3):
        hist_r = cv2.calcHist([bgr_r], [ch], None, [32], [0, 256]).flatten()
        hist_c = cv2.calcHist([bgr_c], [ch], None, [32], [0, 256]).flatten()
        hist_r /= (hist_r.sum() + 1e-9)
        hist_c /= (hist_c.sum() + 1e-9)
        correlation = float(cv2.compareHist(
            hist_r.astype(np.float32).reshape(-1, 1),
            hist_c.astype(np.float32).reshape(-1, 1),
            cv2.HISTCMP_CORREL,
        ))
        color_scores.append(max(0.0, correlation))
    coherence_color = float(np.mean(color_scores))

    # 5. Entropie locale (D2 : complexité texturale)
    def _local_entropy(gray: np.ndarray, block: int = 8) -> float:
        h, w = gray.shape
        entropies = []
        for by in range(0, h - block, block):
            for bx in range(0, w - block, block):
                patch = gray[by:by+block, bx:bx+block]
                hist, _ = np.histogram(patch, bins=16, range=(0, 256))
                hist = hist / (hist.sum() + 1e-9)
                ent = float(-np.sum(hist * np.log2(hist + 1e-9)))
                entropies.append(ent)
        return float(np.mean(entropies)) if entropies else 0.0

    entropy_r = _local_entropy(gray_r)
    entropy_c = _local_entropy(gray_c)
    # Récompense une entropie proche de l'image corrompue (ni trop lisse ni trop chaotique)
    local_entropy_score = float(max(0.0, 1.0 - abs(entropy_r - entropy_c) / max(entropy_c, 0.1)))

    # 6. Artefacts de blocs (variance locale 8×8)
    h, w = gray_r.shape
    block_vars = []
    for by in range(0, h - 8, 8):
        for bx in range(0, w - 8, 8):
            block_vars.append(float(np.var(gray_r[by:by+8, bx:bx+8].astype(np.float32))))
    mean_block_var = float(np.mean(block_vars)) if block_vars else 0.0
    artifact_score = float(np.clip(mean_block_var / 500.0, 0.0, 1.0))

    # 7. Cohérence locale — split inside/outside mask when mask is available
    mean_delta = float(np.mean(np.abs(
        gray_r.astype(np.float32) - gray_c.astype(np.float32)
    )))
    reconstruction_effort = 1.0  # penalty multiplier for conservative candidates
    mean_delta_inside: float | None = None
    mean_delta_outside: float | None = None

    if mask is not None and mask.shape[:2] == gray_c.shape:
        bin_mask_b = (mask > 128)
        not_mask_b = ~bin_mask_b
        if bin_mask_b.any() and not_mask_b.any():
            mean_delta_inside  = float(np.mean(np.abs(
                gray_r[bin_mask_b].astype(np.float32) - gray_c[bin_mask_b].astype(np.float32)
            )))
            mean_delta_outside = float(np.mean(np.abs(
                gray_r[not_mask_b].astype(np.float32) - gray_c[not_mask_b].astype(np.float32)
            )))
            # Consistency: only penalize changes OUTSIDE the mask (artifacts)
            consistency = float(max(0.2, 1.0 - min(mean_delta_outside / 80.0, 0.8)))
            # Reconstruction effort: reward candidates that actually changed the masked zone
            # Conservative (no change → delta ≈ 0) gets penalized; real inpainting changes pixels
            if mean_delta_inside < 3.0:
                reconstruction_effort = 0.45  # barely touched — very conservative
            elif mean_delta_inside < 15.0:
                reconstruction_effort = 0.60 + mean_delta_inside / 75.0
            else:
                reconstruction_effort = float(min(1.0, 0.80 + mean_delta_inside / 200.0))
        else:
            consistency = float(max(0.2, 1.0 - min(mean_delta / 80.0, 0.8)))
    else:
        consistency = float(max(0.2, 1.0 - min(mean_delta / 80.0, 0.8)))

    # Score composite pondéré (D2)
    raw = (
        0.25 * sharpness_score
        + 0.20 * noise_score
        + 0.20 * edge_continuity
        + 0.15 * coherence_color
        + 0.10 * local_entropy_score
        + 0.10 * artifact_score
    )
    score = float(max(0.0, min(100.0, raw * consistency * reconstruction_effort * 100.0)))

    # D3 : score_breakdown
    score_breakdown: dict[str, Any] = {
        "global_score":              score,
        "sharpness_component":       round(sharpness_score * 0.25 * 100.0, 2),
        "noise_component":           round(noise_score * 0.20 * 100.0, 2),
        "edge_component":            round(edge_continuity * 0.20 * 100.0, 2),
        "color_component":           round(coherence_color * 0.15 * 100.0, 2),
        "entropy_component":         round(local_entropy_score * 0.10 * 100.0, 2),
        "artifact_component":        round(artifact_score * 0.10 * 100.0, 2),
        "consistency_factor":        round(consistency, 3),
        "reconstruction_effort":     round(reconstruction_effort, 3),
        "mean_delta_inside_mask":    round(mean_delta_inside, 2) if mean_delta_inside is not None else None,
        "mean_delta_outside_mask":   round(mean_delta_outside, 2) if mean_delta_outside is not None else None,
        "mask_region_score":         None,
        "outside_preservation":      None,
    }

    return {
        "mode":                       "blind",
        "score":                      score,
        "psnr":                       None,
        "ssim":                       None,
        "gain_psnr":                  None,
        "gain_ssim":                  None,
        "sharpness":                  sharpness,
        "noise_penalty":              noise_penalty,
        "edge_continuity":            edge_continuity,
        "coherence_color":            coherence_color,
        "local_entropy":              entropy_r,
        "mean_block_var":             mean_block_var,
        "mean_delta_vs_corrupted":    mean_delta,
        "mean_delta_inside_mask":     mean_delta_inside,
        "mean_delta_outside_mask":    mean_delta_outside,
        "reconstruction_effort":      reconstruction_effort,
        "sharpness_score":            sharpness_score,
        "noise_score":                noise_score,
        "artifact_score":             artifact_score,
        "consistency":                consistency,
        "score_breakdown":            score_breakdown,
    }


# ---------------------------------------------------------------------------
# Point d'entrée unifié
# ---------------------------------------------------------------------------

def score_candidate(
    corrupted: str | Path | np.ndarray,
    reconstructed: str | Path | np.ndarray,
    original: str | Path | np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Retourne le meilleur scoring disponible avec score_breakdown."""
    if original is not None:
        return compute_supervised_score(original, reconstructed, corrupted, mask=mask)
    return compute_blind_score(corrupted, reconstructed, mask=mask)