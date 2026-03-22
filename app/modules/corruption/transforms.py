from typing import Any

import numpy as np
from PIL import Image
import cv2

def _to_numpy_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def _empty_mask(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def apply_rectangle_mask(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    fill_value: int = 0,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]

    x2 = min(x + width, w)
    y2 = min(y + height, h)
    x = max(0, x)
    y = max(0, y)

    arr[y:y2, x:x2] = fill_value

    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255

    params = {"x": x, "y": y, "width": x2 - x, "height": y2 - y, "fill_value": fill_value}
    return Image.fromarray(arr), Image.fromarray(mask), params


def apply_noise(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    sigma: float = 25.0,
    seed: int | None = None,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]

    x2 = min(x + width, w)
    y2 = min(y + height, h)
    x = max(0, x)
    y = max(0, y)

    region = arr[y:y2, x:x2].astype(np.float32)
    noise = rng.normal(0, sigma, region.shape)
    region_noisy = np.clip(region + noise, 0, 255).astype(np.uint8)
    arr[y:y2, x:x2] = region_noisy

    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255

    params = {"x": x, "y": y, "width": x2 - x, "height": y2 - y, "sigma": sigma, "seed": seed}
    return Image.fromarray(arr), Image.fromarray(mask), params


def apply_zone_deletion(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]

    x2 = min(x + width, w)
    y2 = min(y + height, h)
    x = max(0, x)
    y = max(0, y)

    arr[y:y2, x:x2] = [255, 255, 255]

    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255

    params = {"x": x, "y": y, "width": x2 - x, "height": y2 - y}
    return Image.fromarray(arr), Image.fromarray(mask), params


def irregular_mask(image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    points = np.array([
        [50, 50],
        [80, 120],
        [140, 100],
        [120, 40]
    ])

    cv2.fillPoly(mask, [points], 255)

    return mask


def localized_noise(image, mask):
    noisy = image.copy()
    noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)

    noisy[mask == 255] = cv2.add(noisy[mask == 255], noise[mask == 255])

    return noisy


def _to_numpy_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def _empty_mask(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def generate_random_rectangle(
    image_width: int,
    image_height: int,
    min_size: int = 20,
    max_size: int = 100,
    rng: np.random.Generator | None = None,
) -> dict[str, int]:
    if rng is None:
        rng = np.random.default_rng()

    width = int(rng.integers(min_size, min(max_size, image_width) + 1))
    height = int(rng.integers(min_size, min(max_size, image_height) + 1))

    max_x = max(1, image_width - width)
    max_y = max(1, image_height - height)

    x = int(rng.integers(0, max_x))
    y = int(rng.integers(0, max_y))

    return {"x": x, "y": y, "width": width, "height": height}


def apply_rectangle_mask(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    fill_value: int = 0,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]

    x2 = min(x + width, w)
    y2 = min(y + height, h)
    x = max(0, x)
    y = max(0, y)

    arr[y:y2, x:x2] = fill_value

    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255

    params = {"x": x, "y": y, "width": x2 - x, "height": y2 - y, "fill_value": fill_value}
    return Image.fromarray(arr), Image.fromarray(mask), params


def apply_noise(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    sigma: float = 25.0,
    seed: int | None = None,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]

    x2 = min(x + width, w)
    y2 = min(y + height, h)
    x = max(0, x)
    y = max(0, y)

    region = arr[y:y2, x:x2].astype(np.float32)
    noise = rng.normal(0, sigma, region.shape)
    region_noisy = np.clip(region + noise, 0, 255).astype(np.uint8)
    arr[y:y2, x:x2] = region_noisy

    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255

    params = {"x": x, "y": y, "width": x2 - x, "height": y2 - y, "sigma": sigma, "seed": seed}
    return Image.fromarray(arr), Image.fromarray(mask), params


def apply_zone_deletion(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]

    x2 = min(x + width, w)
    y2 = min(y + height, h)
    x = max(0, x)
    y = max(0, y)

    arr[y:y2, x:x2] = [255, 255, 255]

    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255

    params = {"x": x, "y": y, "width": x2 - x, "height": y2 - y}
    return Image.fromarray(arr), Image.fromarray(mask), params


def apply_combined_corruption(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    fill_value: int = 0,
    sigma: float = 12.0,
    seed: int | None = None,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """
    Corruption plus réaliste :
    - rectangle masqué
    - léger bruit autour de la zone
    """
    rng = np.random.default_rng(seed)
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]

    x2 = min(x + width, w)
    y2 = min(y + height, h)
    x = max(0, x)
    y = max(0, y)

    # zone centrale fortement corrompue
    arr[y:y2, x:x2] = fill_value

    # halo de bruit autour
    halo = 10
    hx1 = max(0, x - halo)
    hy1 = max(0, y - halo)
    hx2 = min(w, x2 + halo)
    hy2 = min(h, y2 + halo)

    region = arr[hy1:hy2, hx1:hx2].astype(np.float32)
    noise = rng.normal(0, sigma, region.shape)
    region_noisy = np.clip(region + noise, 0, 255).astype(np.uint8)

    # remettre la zone centrale noire après ajout du bruit autour
    arr[hy1:hy2, hx1:hx2] = region_noisy
    arr[y:y2, x:x2] = fill_value

    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255

    params = {
        "x": x,
        "y": y,
        "width": x2 - x,
        "height": y2 - y,
        "fill_value": fill_value,
        "sigma": sigma,
        "seed": seed,
        "mode": "combined",
    }
    return Image.fromarray(arr), Image.fromarray(mask), params


def build_imperfect_mask(
    mask_image: Image.Image,
    dilate_iter: int = 1,
    erode_iter: int = 0,
    add_noise: bool = True,
    noise_ratio: float = 0.01,
    seed: int | None = None,
) -> Image.Image:
    """
    Transforme un masque parfait en masque approximatif :
    - dilatation/érosion
    - bruit aléatoire
    """
    rng = np.random.default_rng(seed)
    mask = np.array(mask_image.convert("L"), dtype=np.uint8)

    kernel = np.ones((5, 5), np.uint8)

    if dilate_iter > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    if erode_iter > 0:
        mask = cv2.erode(mask, kernel, iterations=erode_iter)

    if add_noise and noise_ratio > 0:
        noise = rng.random(mask.shape)
        mask[noise < noise_ratio] = 255
        mask[noise > (1.0 - noise_ratio)] = 0

    return Image.fromarray(mask)