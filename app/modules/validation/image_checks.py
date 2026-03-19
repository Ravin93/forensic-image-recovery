from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

from app.core.exceptions import ValidationError


def check_image_readable(image_path: str | Path) -> None:
    path = Path(image_path)

    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValidationError(f"Image illisible : {path}") from exc


def get_image_info(image_path: str | Path) -> dict[str, Any]:
    path = Path(image_path)

    try:
        with Image.open(path) as img:
            width, height = img.size
            image_format = img.format
            mode = img.mode
    except (UnidentifiedImageError, OSError) as exc:
        raise ValidationError(f"Impossible de lire les métadonnées : {path}") from exc

    return {
        "width": width,
        "height": height,
        "format": image_format,
        "mode": mode,
    }


def check_image_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValidationError("Dimensions invalides")


def check_image_format(
    image_format: str | None,
    allowed_formats: Iterable[str] | None = None,
) -> None:
    if allowed_formats is None:
        allowed_formats = {"JPEG", "JPG"}

    normalized_allowed = {fmt.upper() for fmt in allowed_formats}

    if image_format is None or image_format.upper() not in normalized_allowed:
        raise ValidationError(
            f"Format invalide : {image_format} | formats attendus={sorted(normalized_allowed)}"
        )