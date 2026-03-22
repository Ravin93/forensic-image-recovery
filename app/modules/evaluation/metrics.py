from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from app.core.exceptions import EvaluationError


def load_image_as_rgb_array(image: str | Path | np.ndarray) -> np.ndarray:
    """
    Charge une image en tableau RGB uint8.
    Compatible avec :
    - str
    - Path
    - np.ndarray
    """
    if isinstance(image, np.ndarray):
        return image

    path = Path(image)
    if not path.exists():
        raise EvaluationError(f"Image introuvable : {path}")

    try:
        return np.array(Image.open(path).convert("RGB"))
    except Exception as exc:
        raise EvaluationError(f"Impossible de charger l'image : {path}") from exc


def load_image_as_array(image: str | Path | np.ndarray) -> np.ndarray:
    """
    Alias de compatibilité interne.
    """
    return load_image_as_rgb_array(image)


def ensure_same_shape(image_a: np.ndarray, image_b: np.ndarray) -> None:
    if image_a.shape != image_b.shape:
        raise EvaluationError(
            f"Dimensions incompatibles : {image_a.shape} vs {image_b.shape}"
        )


def compute_psnr(
    image_a: str | Path | np.ndarray,
    image_b: str | Path | np.ndarray,
) -> float:
    arr_a = load_image_as_rgb_array(image_a)
    arr_b = load_image_as_rgb_array(image_b)

    ensure_same_shape(arr_a, arr_b)

    value = peak_signal_noise_ratio(arr_a, arr_b, data_range=255)

    if np.isinf(value):
        return 100.0

    return float(value)


def compute_ssim(
    image_a: str | Path | np.ndarray,
    image_b: str | Path | np.ndarray,
) -> float:
    arr_a = load_image_as_rgb_array(image_a)
    arr_b = load_image_as_rgb_array(image_b)

    ensure_same_shape(arr_a, arr_b)

    value = structural_similarity(
        arr_a,
        arr_b,
        channel_axis=2,
        data_range=255,
    )
    return float(value)