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
    apply_local_blur,
    apply_mixed_corruption,
    apply_noise,
    apply_rectangle_mask,
    apply_shift_region,
    apply_zone_deletion,
    build_imperfect_mask,
    generate_random_rectangle,
)

SUPPORTED_CORRUPTIONS = {
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
}


def _maybe_randomize_rect(width: int, height: int, kwargs: dict[str, Any]) -> dict[str, Any]:
    rect = generate_random_rectangle(
        image_width=width,
        image_height=height,
        min_size=kwargs.pop("min_size", 20),
        max_size=kwargs.pop("max_size", 100),
    )
    return {**rect, **kwargs}


def simulate_corruption(image: np.ndarray, corruption_type: str, **params: Any) -> dict[str, Any]:
    pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")

    if corruption_type == "zone_deletion":
        corrupted, mask, used_params = apply_zone_deletion(pil_image, **params)
    elif corruption_type == "bar":
        corrupted, mask, used_params = apply_bar_corruption(pil_image, **params)
    elif corruption_type == "shift_region":
        corrupted, mask, used_params = apply_shift_region(pil_image, **params)
    elif corruption_type == "rectangle_mask":
        corrupted, mask, used_params = apply_rectangle_mask(pil_image, **params)
    elif corruption_type == "noise":
        corrupted, mask, used_params = apply_noise(pil_image, **params)
    elif corruption_type == "combined":
        corrupted, mask, used_params = apply_combined_corruption(pil_image, **params)
    elif corruption_type == "local_blur":
        corrupted, mask, used_params = apply_local_blur(pil_image, **params)
    elif corruption_type == "block_dropout":
        corrupted, mask, used_params = apply_block_dropout(pil_image, **params)
    elif corruption_type == "jpeg_block_artifacts":
        corrupted, mask, used_params = apply_jpeg_block_artifacts(pil_image, **params)
    elif corruption_type == "mixed":
        corrupted, mask, used_params = apply_mixed_corruption(pil_image, **params)
    else:
        raise ValueError(f"Type de corruption non supporté : {corruption_type}")

    return {
        "corrupted_image": np.array(corrupted),
        "mask": np.array(mask),
        "corruption_type": corruption_type,
        "parameters": used_params,
    }


def corrupt_image(
    image_path: str | Path,
    corruption_type: str,
    randomize: bool = False,
    seed: int | None = None,
    imperfect_mask: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    ensure_directories()
    path = Path(image_path)
    if not path.exists():
        raise CorruptionError(f"Image source introuvable : {path}")

    try:
        image = Image.open(path).convert("RGB")
    except OSError as exc:
        raise CorruptionError(f"Impossible d'ouvrir l'image : {path}") from exc

    width, height = image.size
    corruption_type = corruption_type.strip().lower()
    if corruption_type not in SUPPORTED_CORRUPTIONS:
        raise CorruptionError(f"Type de corruption non supporté : {corruption_type}")

    rect_based = {
        "rectangle_mask",
        "noise",
        "zone_deletion",
        "combined",
        "local_blur",
        "shift_region",
        "jpeg_block_artifacts",
    }
    if randomize and corruption_type in rect_based:
        kwargs = _maybe_randomize_rect(width, height, dict(kwargs))

    try:
        if corruption_type == "rectangle_mask":
            corrupted, mask, params = apply_rectangle_mask(image, **kwargs)
        elif corruption_type == "noise":
            corrupted, mask, params = apply_noise(image, seed=seed, **kwargs)
        elif corruption_type == "zone_deletion":
            corrupted, mask, params = apply_zone_deletion(image, **kwargs)
        elif corruption_type == "combined":
            corrupted, mask, params = apply_combined_corruption(image, seed=seed, **kwargs)
        elif corruption_type == "bar":
            corrupted, mask, params = apply_bar_corruption(image, seed=seed, **kwargs)
        elif corruption_type == "local_blur":
            corrupted, mask, params = apply_local_blur(image, **kwargs)
        elif corruption_type == "shift_region":
            corrupted, mask, params = apply_shift_region(image, **kwargs)
        elif corruption_type == "block_dropout":
            corrupted, mask, params = apply_block_dropout(image, seed=seed, **kwargs)
        elif corruption_type == "jpeg_block_artifacts":
            corrupted, mask, params = apply_jpeg_block_artifacts(image, **kwargs)
        elif corruption_type == "mixed":
            corrupted, mask, params = apply_mixed_corruption(image, seed=seed, **kwargs)
        else:
            raise CorruptionError(f"Type de corruption non supporté : {corruption_type}")
    except TypeError as exc:
        raise CorruptionError(f"Paramètres invalides pour '{corruption_type}' : {exc}") from exc
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
    logger.info("Corruption OK : %s", result)
    return result