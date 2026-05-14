"""app/modules/similarity/perceptual_hash.py — K14.

Hash perceptuel pour comparer des images meme apres
compression, redimensionnement ou legere modification.
Implemente dHash et pHash sans dependance externe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.logger import logger


# ---------------------------------------------------------------------------
# dHash (difference hash) — rapide, robuste aux petites modifications
# ---------------------------------------------------------------------------

def compute_dhash(path: str | Path, hash_size: int = 8) -> str:
    """Calcule le dHash d une image (difference hash).

    Compare les pixels adjacents horizontalement.
    Retourne une chaine hexadecimale de hash_size² / 4 caracteres.
    """
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de charger : {path}")

    # Redimensionner a (hash_size+1) x hash_size
    resized = cv2.resize(img, (hash_size + 1, hash_size),
                         interpolation=cv2.INTER_AREA)

    # Gradient horizontal : pixel gauche > pixel droite ?
    diff = resized[:, :-1] > resized[:, 1:]
    bits = diff.flatten()

    # Convertir en entier puis hex
    hash_int = int("".join("1" if b else "0" for b in bits), 2)
    hex_len  = hash_size * hash_size // 4
    return format(hash_int, "0" + str(hex_len) + "x")


# ---------------------------------------------------------------------------
# pHash (perceptual hash) — robuste aux changements gamma/contraste
# ---------------------------------------------------------------------------

def compute_phash(path: str | Path, hash_size: int = 8) -> str:
    """Calcule le pHash d une image (perceptual hash via DCT).

    Retourne une chaine hexadecimale.
    """
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de charger : {path}")

    # Redimensionner a 32x32
    size = hash_size * 4
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    # DCT
    float_img = resized.astype(np.float32)
    dct = cv2.dct(float_img)

    # Garder le coin top-left (hash_size x hash_size)
    dct_low = dct[:hash_size, :hash_size]

    # Moyenne (sans le coefficient DC)
    mean_val = (dct_low.sum() - dct_low[0, 0]) / (hash_size * hash_size - 1)

    # Binariser
    bits = (dct_low > mean_val).flatten()

    hash_int = int("".join("1" if b else "0" for b in bits), 2)
    hex_len  = hash_size * hash_size // 4
    return format(hash_int, "0" + str(hex_len) + "x")


# ---------------------------------------------------------------------------
# Comparaison
# ---------------------------------------------------------------------------

def _hex_to_bits(hex_str: str) -> list[int]:
    n = int(hex_str, 16)
    bits_len = len(hex_str) * 4
    return [(n >> i) & 1 for i in range(bits_len - 1, -1, -1)]


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Calcule la distance de Hamming entre deux hashes hexadecimaux."""
    if len(hash_a) != len(hash_b):
        raise ValueError(f"Hashes de longueur differente : {len(hash_a)} vs {len(hash_b)}")
    bits_a = _hex_to_bits(hash_a)
    bits_b = _hex_to_bits(hash_b)
    return sum(a != b for a, b in zip(bits_a, bits_b))


def compare_hashes(hash_a: str, hash_b: str) -> dict[str, Any]:
    """Compare deux hashes perceptuels.

    Retourne la distance de Hamming et une appreciation qualitative.
    """
    dist = hamming_distance(hash_a, hash_b)
    total_bits = len(hash_a) * 4
    similarity = 1.0 - dist / total_bits

    if dist == 0:
        label = "identiques"
    elif dist <= 5:
        label = "tres similaires"
    elif dist <= 10:
        label = "similaires"
    elif dist <= 20:
        label = "moderement differentes"
    else:
        label = "differentes"

    return {
        "distance":        dist,
        "max_distance":    total_bits,
        "similarity":      round(similarity, 4),
        "label":           label,
        "same_image":      dist == 0,
        "likely_similar":  dist <= 10,
    }


def compare_images_perceptual(
    path_a: str | Path,
    path_b: str | Path,
    method: str = "dhash",
) -> dict[str, Any]:
    """Compare deux images par hash perceptuel.

    Args:
        path_a, path_b : chemins des images
        method : "dhash" ou "phash"

    Returns:
        dict avec hashes, distance, similarite et label
    """
    fn = compute_dhash if method == "dhash" else compute_phash
    hash_a = fn(path_a)
    hash_b = fn(path_b)
    result = compare_hashes(hash_a, hash_b)
    return {
        "method":   method,
        "hash_a":   hash_a,
        "hash_b":   hash_b,
        "path_a":   str(path_a),
        "path_b":   str(path_b),
        **result,
    }


def compute_all_hashes(path: str | Path) -> dict[str, str]:
    """Calcule dhash et phash pour une image."""
    return {
        "dhash": compute_dhash(path),
        "phash": compute_phash(path),
    }