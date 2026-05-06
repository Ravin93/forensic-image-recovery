from pathlib import Path
from typing import Any

import cv2

from app.core.config import RECONSTRUCTED_DIR, ensure_directories
from app.core.exceptions import ReconstructionError


def denoise_image(image_path: str | Path, method: str = "median_blur") -> dict[str, Any]:
    ensure_directories()

    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ReconstructionError(f"Impossible de charger l'image : {path}")

    if method == "median_blur":
        denoised = cv2.medianBlur(image, 5)
    elif method == "gaussian_blur":
        denoised = cv2.GaussianBlur(image, (5, 5), 0)
    elif method == "bilateral":
        denoised = cv2.bilateralFilter(image, 7, 50, 50)
    else:
        raise ReconstructionError(f"Méthode non supportée : {method}")

    output_path = RECONSTRUCTED_DIR / f"{path.stem}_{method}.png"
    ok = cv2.imwrite(str(output_path), denoised)
    if not ok:
        raise ReconstructionError(f"Impossible d'écrire l'image : {output_path}")

    return {
        "method": method,
        "path": str(output_path),
        "status": "denoised",
        "source_image": path.name,
    }

def denoise(image):
    return cv2.medianBlur(image, 5)