from pathlib import Path
from typing import Any

import cv2

from app.core.config import DEFAULT_INPAINT_RADIUS, build_reconstructed_image_path, ensure_directories
from app.core.exceptions import ReconstructionError
from app.core.logger import logger


def reconstruct_with_inpaint(
    image_path: str | Path,
    mask_path: str | Path,
    method: str = "opencv_inpaint",
    radius: int = DEFAULT_INPAINT_RADIUS,
) -> dict[str, Any]:
    ensure_directories()

    image_path = Path(image_path)
    mask_path = Path(mask_path)

    if not image_path.exists():
        raise ReconstructionError(f"Image corrompue introuvable : {image_path}")
    if not mask_path.exists():
        raise ReconstructionError(f"Masque introuvable : {mask_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ReconstructionError(f"Impossible de charger l'image : {image_path}")
    if mask is None:
        raise ReconstructionError(f"Impossible de charger le masque : {mask_path}")

    if method != "opencv_inpaint":
        raise ReconstructionError(f"Méthode non supportée : {method}")

    reconstructed = cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
    output_path = build_reconstructed_image_path(image_path.name, method)

    ok = cv2.imwrite(str(output_path), reconstructed)
    if not ok:
        raise ReconstructionError(f"Impossible d'écrire l'image reconstruite : {output_path}")

    result = {
        "file": output_path.name,
        "path": str(output_path),
        "method": method,
        "status": "reconstructed",
        "source_image": image_path.name,
        "mask_path": str(mask_path),
    }

    logger.info("Reconstruction OK : %s", result)
    return result