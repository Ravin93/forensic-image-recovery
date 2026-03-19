from typing import Any

import numpy as np
from PIL import Image


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