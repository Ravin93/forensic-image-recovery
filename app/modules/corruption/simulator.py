from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import build_corrupted_image_path, build_mask_path, ensure_directories
from app.core.exceptions import CorruptionError
from app.core.logger import logger
from app.modules.corruption.transforms import (
    apply_noise,
    apply_rectangle_mask,
    apply_zone_deletion,
)


def corrupt_image(
    image_path: str | Path,
    corruption_type: str,
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

    if corruption_type == "rectangle_mask":
        corrupted, mask, params = apply_rectangle_mask(image, **kwargs)
    elif corruption_type == "noise":
        corrupted, mask, params = apply_noise(image, **kwargs)
    elif corruption_type == "zone_deletion":
        corrupted, mask, params = apply_zone_deletion(image, **kwargs)
    else:
        raise CorruptionError(f"Type de corruption non supporté : {corruption_type}")

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
    }

    logger.info("Corruption OK : %s", result)
    return result