from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from app.core.exceptions import EvaluationError


def load_image_as_rgb_array(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)

    if not path.exists():
        raise EvaluationError(f"Image introuvable : {path}")

    try:
        image = Image.open(path).convert("RGB")
    except OSError as exc:
        raise EvaluationError(f"Impossible de charger l'image : {path}") from exc

    return np.array(image, dtype=np.uint8)


def ensure_same_shape(image_a: np.ndarray, image_b: np.ndarray) -> None:
    if image_a.shape != image_b.shape:
        raise EvaluationError(
            f"Les images doivent avoir la même taille. "
            f"Reçu: {image_a.shape} vs {image_b.shape}"
        )


def compute_psnr(image_a: np.ndarray, image_b: np.ndarray) -> float:
    ensure_same_shape(image_a, image_b)

    mse = np.mean((image_a.astype(np.float32) - image_b.astype(np.float32)) ** 2)

    if mse == 0:
        return float("inf")

    pixel_max = 255.0
    return float(10.0 * np.log10((pixel_max ** 2) / mse))


def compute_ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    ensure_same_shape(image_a, image_b)

    try:
        score = structural_similarity(
            image_a,
            image_b,
            channel_axis=2,
            data_range=255,
        )
    except Exception as exc:
        raise EvaluationError("Impossible de calculer le SSIM") from exc

    return float(score)