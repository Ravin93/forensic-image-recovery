from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.exceptions import CorruptionError


def _load_gray(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise CorruptionError(f"Image introuvable : {path}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise CorruptionError(f"Impossible de charger l'image : {path}")
    return image


def detect_dark_regions(
    image_path: str | Path,
    threshold: int = 22,
) -> dict[str, Any]:
    image = _load_gray(image_path)
    _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)

    return {
        "mask": mask.astype(np.uint8),
        "method": "dark_regions",
        "score": float(np.mean(mask > 0)),
    }


def detect_low_variance_regions(
    image_path: str | Path,
    window_size: int = 11,
    variance_threshold: float = 6.0,
) -> dict[str, Any]:
    image = _load_gray(image_path).astype(np.float32)

    mean = cv2.blur(image, (window_size, window_size))
    mean_sq = cv2.blur(image ** 2, (window_size, window_size))
    variance = np.clip(mean_sq - mean ** 2, 0, None)

    mask = np.where(variance < variance_threshold, 255, 0).astype(np.uint8)

    return {
        "mask": mask,
        "method": "low_variance",
        "score": float(np.mean(mask > 0)),
    }


def detect_gradient_breaks(
    image_path: str | Path,
    edge_threshold: float = 4.0,
) -> dict[str, Any]:
    image = _load_gray(image_path).astype(np.float32)

    sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    mask = np.where(grad_mag < edge_threshold, 255, 0).astype(np.uint8)

    return {
        "mask": mask,
        "method": "gradient_breaks",
        "score": float(np.mean(mask > 0)),
    }