from pathlib import Path
from typing import Any, Iterable

from app.core.exceptions import ValidationError
from app.core.logger import logger
from app.modules.validation.image_checks import (
    check_image_dimensions,
    check_image_format,
    check_image_readable,
    get_image_info,
)


def verify_image(
    image_path: str | Path,
    allowed_formats: Iterable[str] | None = None,
) -> dict[str, Any]:
    path = Path(image_path)

    try:
        check_image_readable(path)
        info = get_image_info(path)
        check_image_dimensions(info["width"], info["height"])
        check_image_format(info["format"], allowed_formats=allowed_formats)

        logger.info("Validation OK : %s", path)
        return {
            "valid": True,
            "reason": "Image valide",
            "details": info,
        }

    except ValidationError as exc:
        logger.info("Validation KO : %s | %s", path, exc)
        return {
            "valid": False,
            "reason": str(exc),
            "details": None,
        }