from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import build_corrupted_image_path, build_mask_path, ensure_directories
from app.core.exceptions import CorruptionError
from app.core.logger import logger
from app.modules.corruption.transforms import (
    apply_bar_corruption,
    apply_block_dropout,
    apply_combined_corruption,
    apply_jpeg_block_artifacts,
    apply_large_deleted_square,
    apply_local_blur,
    apply_local_noise,
    apply_mixed_corruption,
    apply_multiple_bars,
    apply_noise,
    apply_random_holes,
    apply_rectangle_mask,
    apply_scratch_lines,
    apply_shift_region,
    apply_zone_deletion,
    build_imperfect_mask,
    generate_random_rectangle,
)

# ---------------------------------------------------------------------------
# Types supportés
# ---------------------------------------------------------------------------

SUPPORTED_CORRUPTIONS = {
    # Existants
    "rectangle_mask",
    "noise",
    "zone_deletion",
    "combined",
    "bar",
    "local_blur",
    "shift_region",
    "block_dropout",
    "jpeg_block_artifacts",
    "mixed",
    # Nouveaux — Ticket 1
    "scratch_lines",
    "large_deleted_square",
    "multiple_bars",
    "random_holes",
    "local_noise",
}

# Corruptions dont les coords x/y/width/height peuvent être randomisées
_RECT_BASED = {
    "rectangle_mask",
    "noise",
    "zone_deletion",
    "combined",
    "local_blur",
    "shift_region",
    "jpeg_block_artifacts",
    "local_noise",
    # Ces types ont aussi des coordonnées optionnelles
    "bar",
}


def _maybe_randomize_rect(width: int, height: int, kwargs: dict[str, Any]) -> dict[str, Any]:
    # max_size adaptatif : 40% de la plus petite dimension, min 30px
    default_max = max(30, min(width, height) * 4 // 10)
    rect = generate_random_rectangle(
        image_width=width,
        image_height=height,
        min_size=kwargs.pop("min_size", 30),
        max_size=kwargs.pop("max_size", default_max),
    )
    return {**rect, **kwargs}


# ---------------------------------------------------------------------------
# simulate_corruption  (entrée numpy — utilisée par les modules internes)
# ---------------------------------------------------------------------------

def simulate_corruption(
    image: np.ndarray, corruption_type: str, **params: Any
) -> dict[str, Any]:
    """Applique une corruption à un tableau numpy et retourne le résultat.

    Returns:
        dict avec les clés :
            corrupted_image (np.ndarray)
            mask            (np.ndarray)
            corruption_type (str)
            parameters      (dict)
    """
    pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    corrupted, mask, used_params = _dispatch(pil_image, corruption_type, params)
    return {
        "corrupted_image": np.array(corrupted),
        "mask": np.array(mask),
        "corruption_type": corruption_type,
        "parameters": used_params,
    }


# ---------------------------------------------------------------------------
# corrupt_image  (entrée fichier — utilisée par le pipeline et l'API)
# ---------------------------------------------------------------------------

def corrupt_image(
    image_path: str | Path,
    corruption_type: str,
    randomize: bool = False,
    seed: int | None = None,
    imperfect_mask: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Charge une image, applique la corruption, sauvegarde et retourne les chemins.

    Returns:
        dict avec au minimum :
            path / image_path   chemin de l'image corrompue
            mask_path           chemin du masque
            corruption_type     type utilisé
            parameters          params réels appliqués
            status              "corrupted"
    """
    ensure_directories()
    path = Path(image_path)
    if not path.exists():
        raise CorruptionError(f"Image source introuvable : {path}")

    try:
        image = Image.open(path).convert("RGB")
    except OSError as exc:
        raise CorruptionError(f"Impossible d'ouvrir l'image : {path}") from exc

    img_width, img_height = image.size
    corruption_type = corruption_type.strip().lower()
    if corruption_type not in SUPPORTED_CORRUPTIONS:
        raise CorruptionError(
            f"Type de corruption non supporté : '{corruption_type}'. "
            f"Disponibles : {sorted(SUPPORTED_CORRUPTIONS)}"
        )

    # Randomisation des coordonnées rect si demandée
    if randomize and corruption_type in _RECT_BASED:
        kwargs = _maybe_randomize_rect(img_width, img_height, dict(kwargs))

    # Injection du seed si non fourni explicitement
    if seed is not None:
        kwargs.setdefault("seed", seed)

    try:
        corrupted, mask, params = _dispatch(image, corruption_type, kwargs)
    except TypeError as exc:
        raise CorruptionError(
            f"Paramètres invalides pour '{corruption_type}' : {exc}"
        ) from exc
    except ValueError as exc:
        raise CorruptionError(str(exc)) from exc

    if imperfect_mask:
        mask = build_imperfect_mask(mask, seed=seed)

    corrupted_path = build_corrupted_image_path(path.name, corruption_type)
    mask_path = build_mask_path(path.name, corruption_type)
    corrupted.save(corrupted_path)
    mask.save(mask_path)

    result = {
        "file": corrupted_path.name,
        "path": str(corrupted_path),
        "image_path": str(corrupted_path),
        "mask_path": str(mask_path),
        "corruption_type": corruption_type,
        "parameters": params,
        "status": "corrupted",
        "source_image": path.name,
        "randomize": randomize,
        "imperfect_mask": imperfect_mask,
        "supported_types": sorted(SUPPORTED_CORRUPTIONS),
    }
    logger.info("Corruption OK : %s → %s (score masque ≈ %.1f%%)",
                corruption_type, corrupted_path.name,
                100.0 * np.mean(np.array(mask) > 0))
    return result


# ---------------------------------------------------------------------------
# Dispatcher interne (PIL → PIL)
# ---------------------------------------------------------------------------

def _dispatch(
    image: Image.Image, corruption_type: str, kwargs: dict[str, Any]
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Route la PIL Image vers la bonne fonction de transforms.py.

    Le seed est retiré des kwargs si la fonction cible ne le supporte pas,
    évitant les TypeError pour les fonctions déterministes.
    """
    import inspect

    _FN_MAP = {
        "rectangle_mask":      apply_rectangle_mask,
        "noise":               apply_noise,
        "zone_deletion":       apply_zone_deletion,
        "combined":            apply_combined_corruption,
        "bar":                 apply_bar_corruption,
        "local_blur":          apply_local_blur,
        "shift_region":        apply_shift_region,
        "block_dropout":       apply_block_dropout,
        "jpeg_block_artifacts": apply_jpeg_block_artifacts,
        "mixed":               apply_mixed_corruption,
        "scratch_lines":       apply_scratch_lines,
        "large_deleted_square": apply_large_deleted_square,
        "multiple_bars":       apply_multiple_bars,
        "random_holes":        apply_random_holes,
        "local_noise":         apply_local_noise,
    }

    fn = _FN_MAP.get(corruption_type)
    if fn is None:
        raise CorruptionError(f"Type de corruption non géré dans le dispatcher : {corruption_type}")

    # Ne passer que les paramètres que la fonction accepte réellement
    accepted = inspect.signature(fn).parameters
    filtered = {k: v for k, v in kwargs.items() if k in accepted}

    return fn(image, **filtered)