"""app/modules/fragments/fragment_clustering.py — K15.

Clustering de fragments visuels par features.
Utilise DBSCAN pour regrouper les fragments appartenant
probablement a la meme image source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.logger import logger


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_fragment_features(image_path: str | Path) -> np.ndarray:
    """Extrait un vecteur de features visuelles pour le clustering.

    Features :
    - Histogramme couleur normalise (3x32 = 96 bins)
    - Entropie locale (1 valeur)
    - Variance de luminance (1 valeur)
    - 3 moments statistiques par canal (9 valeurs)
    Total : 107 dimensions
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de charger : {image_path}")

    features: list[float] = []

    # 1. Histogramme couleur (3 canaux x 32 bins)
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [32], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-9)
        features.extend(hist.tolist())

    # 2. Statistiques par canal (mean, std, skewness)
    for ch in range(3):
        channel = img[:, :, ch].astype(np.float32)
        mean = float(channel.mean())
        std  = float(channel.std())
        skew = float(np.mean(((channel - mean) / (std + 1e-9)) ** 3))
        features.extend([mean / 255.0, std / 255.0, np.clip(skew / 10.0, -1, 1)])

    # 3. Entropie locale (8x8 blocs)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    block_entropies = []
    bsize = max(8, min(h, w) // 8)
    for by in range(0, h - bsize, bsize):
        for bx in range(0, w - bsize, bsize):
            patch = gray[by:by+bsize, bx:bx+bsize]
            hist, _ = np.histogram(patch, bins=16, range=(0,256))
            p = hist / (hist.sum() + 1e-9)
            ent = float(-np.sum(p * np.log2(p + 1e-9)))
            block_entropies.append(ent)
    mean_entropy = float(np.mean(block_entropies)) / 4.0 if block_entropies else 0.0
    features.append(mean_entropy)

    # 4. Variance de luminance
    var_lum = float(gray.astype(np.float32).var()) / (255.0 ** 2)
    features.append(var_lum)

    return np.array(features, dtype=np.float32)


def extract_features_from_paths(
    fragment_paths: list[str | Path],
) -> tuple[np.ndarray, list[int]]:
    """Extrait les features de plusieurs fragments.

    Retourne la matrice de features et les indices valides.
    """
    all_features: list[np.ndarray] = []
    valid_indices: list[int] = []

    for i, p in enumerate(fragment_paths):
        try:
            feat = extract_fragment_features(p)
            all_features.append(feat)
            valid_indices.append(i)
        except Exception as exc:
            logger.warning("Features fragment %d echouees : %s", i, exc)

    if not all_features:
        return np.empty((0, 0)), []

    return np.stack(all_features), valid_indices


# ---------------------------------------------------------------------------
# Clustering DBSCAN
# ---------------------------------------------------------------------------

def cluster_fragments_dbscan(
    fragment_paths: list[str | Path],
    eps: float = 0.3,
    min_samples: int = 2,
) -> dict[str, Any]:
    """Regroupe les fragments par DBSCAN sur leurs features visuelles.

    Args:
        fragment_paths : chemins des images de fragments
        eps            : distance maximale pour etre voisins
        min_samples    : taille minimale d un cluster

    Returns:
        dict avec clusters, noise, n_clusters, coherence_score
    """
    if not fragment_paths:
        return {
            "clusters": [], "noise": [], "n_clusters": 0,
            "n_noise": 0, "coherence_score": 0.0,
        }

    features_matrix, valid_idx = extract_features_from_paths(fragment_paths)

    if len(valid_idx) == 0:
        return {
            "clusters": [], "noise": [], "n_clusters": 0,
            "n_noise": 0, "coherence_score": 0.0,
        }

    # Normalisation L2
    norms = np.linalg.norm(features_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    features_norm = features_matrix / norms

    # DBSCAN via sklearn si dispo, sinon implementation simple
    try:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(features_norm)
    except ImportError:
        labels = _simple_dbscan(features_norm, eps, min_samples)

    # Construire les clusters
    clusters_dict: dict[int, list[int]] = {}
    noise_indices: list[int] = []

    for local_i, label in enumerate(labels):
        global_i = valid_idx[local_i]
        if label == -1:
            noise_indices.append(global_i)
        else:
            clusters_dict.setdefault(int(label), []).append(global_i)

    clusters = [
        {
            "cluster_id": cid,
            "fragments":  [str(fragment_paths[i]) for i in idxs],
            "size":       len(idxs),
        }
        for cid, idxs in sorted(clusters_dict.items())
    ]

    noise = [str(fragment_paths[i]) for i in noise_indices]
    n_clusters = len(clusters)

    # Score de coherence : proportion de fragments clusterises
    n_total = len(valid_idx)
    n_clustered = n_total - len(noise_indices)
    coherence = float(n_clustered / n_total) if n_total > 0 else 0.0

    logger.info("Clustering | fragments=%d clusters=%d noise=%d coherence=%.2f",
                n_total, n_clusters, len(noise_indices), coherence)

    return {
        "clusters":        clusters,
        "noise":           noise,
        "n_clusters":      n_clusters,
        "n_noise":         len(noise_indices),
        "n_fragments":     n_total,
        "coherence_score": round(coherence, 3),
        "eps":             eps,
        "min_samples":     min_samples,
    }


def _simple_dbscan(
    features: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """DBSCAN simplifie sans sklearn (fallback)."""
    n = len(features)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0

    def _neighbors(idx: int) -> list[int]:
        diffs = features - features[idx]
        dists = np.linalg.norm(diffs, axis=1)
        return [i for i, d in enumerate(dists) if d <= eps and i != idx]

    visited = set()
    for i in range(n):
        if i in visited:
            continue
        visited.add(i)
        nbrs = _neighbors(i)
        if len(nbrs) < min_samples - 1:
            continue  # bruit
        labels[i] = cluster_id
        queue = list(nbrs)
        while queue:
            j = queue.pop()
            if j not in visited:
                visited.add(j)
                j_nbrs = _neighbors(j)
                if len(j_nbrs) >= min_samples - 1:
                    queue.extend(j_nbrs)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1

    return labels