from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


RGBImageAndMask = tuple[Image.Image, Image.Image, dict[str, Any]]


def _to_numpy_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def _empty_mask(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def _clip_rect(x: int, y: int, width: int, height: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x = max(0, int(x))
    y = max(0, int(y))
    width = max(1, int(width))
    height = max(1, int(height))
    x2 = min(x + width, img_w)
    y2 = min(y + height, img_h)
    return x, y, x2, y2


def generate_random_rectangle(
    image_width: int,
    image_height: int,
    min_size: int = 20,
    max_size: int = 100,
    rng: np.random.Generator | None = None,
) -> dict[str, int]:
    if rng is None:
        rng = np.random.default_rng()

    max_size = max(min_size, min(max_size, image_width, image_height))
    width = int(rng.integers(min_size, max_size + 1))
    height = int(rng.integers(min_size, max_size + 1))

    max_x = max(1, image_width - width + 1)
    max_y = max(1, image_height - height + 1)
    x = int(rng.integers(0, max_x))
    y = int(rng.integers(0, max_y))
    return {"x": x, "y": y, "width": width, "height": height}


def _fill_region(arr: np.ndarray, x: int, y: int, x2: int, y2: int, fill_mode: str, seed: int | None = None) -> None:
    rng = np.random.default_rng(seed)
    region_shape = arr[y:y2, x:x2].shape
    if fill_mode == "black":
        arr[y:y2, x:x2] = 0
    elif fill_mode == "white":
        arr[y:y2, x:x2] = 255
    elif fill_mode == "gray":
        arr[y:y2, x:x2] = 127
    elif fill_mode == "noise":
        arr[y:y2, x:x2] = rng.integers(0, 256, size=region_shape, dtype=np.uint8)
    else:
        raise ValueError(f"fill_mode non supporté : {fill_mode}")


def apply_rectangle_mask(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    fill_value: int = 0,
) -> RGBImageAndMask:
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]
    x, y, x2, y2 = _clip_rect(x, y, width, height, w, h)
    arr[y:y2, x:x2] = np.uint8(fill_value)

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
) -> RGBImageAndMask:
    rng = np.random.default_rng(seed)
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]
    x, y, x2, y2 = _clip_rect(x, y, width, height, w, h)

    region = arr[y:y2, x:x2].astype(np.float32)
    noise = rng.normal(0, sigma, region.shape)
    arr[y:y2, x:x2] = np.clip(region + noise, 0, 255).astype(np.uint8)

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
    fill_mode: str = "white",
    seed: int | None = None,
) -> RGBImageAndMask:
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]
    x, y, x2, y2 = _clip_rect(x, y, width, height, w, h)

    _fill_region(arr, x, y, x2, y2, fill_mode=fill_mode, seed=seed)
    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255
    params = {"x": x, "y": y, "width": x2 - x, "height": y2 - y, "fill_mode": fill_mode, "seed": seed}
    return Image.fromarray(arr), Image.fromarray(mask), params


def apply_bar_corruption(
    image: Image.Image,
    orientation: str = "horizontal",
    thickness: int = 12,
    position: int | None = None,
    count: int = 1,
    gap: int = 20,
    fill_mode: str = "black",
    seed: int | None = None,
) -> RGBImageAndMask:
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]
    mask = _empty_mask(h, w)
    rng = np.random.default_rng(seed)

    thickness = max(1, int(thickness))
    count = max(1, int(count))
    orientation = orientation.lower()
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation doit être 'horizontal' ou 'vertical'")

    if position is None:
        limit = h if orientation == "horizontal" else w
        position = int(rng.integers(0, max(1, limit - thickness + 1)))

    bars = []
    for i in range(count):
        start = position + i * (thickness + gap)
        if orientation == "horizontal":
            if start >= h:
                break
            end = min(h, start + thickness)
            _fill_region(arr, 0, start, w, end, fill_mode=fill_mode, seed=None if seed is None else seed + i)
            mask[start:end, :] = 255
            bars.append({"y": start, "height": end - start})
        else:
            if start >= w:
                break
            end = min(w, start + thickness)
            _fill_region(arr, start, 0, end, h, fill_mode=fill_mode, seed=None if seed is None else seed + i)
            mask[:, start:end] = 255
            bars.append({"x": start, "width": end - start})

    params = {
        "orientation": orientation,
        "thickness": thickness,
        "position": position,
        "count": len(bars),
        "gap": gap,
        "fill_mode": fill_mode,
        "bars": bars,
        "seed": seed,
    }
    return Image.fromarray(arr), Image.fromarray(mask), params


def apply_local_blur(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    kernel_size: int = 9,
) -> RGBImageAndMask:
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]
    x, y, x2, y2 = _clip_rect(x, y, width, height, w, h)
    kernel_size = max(3, int(kernel_size) | 1)

    region = arr[y:y2, x:x2]
    blurred = cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)
    arr[y:y2, x:x2] = blurred

    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255
    params = {"x": x, "y": y, "width": x2 - x, "height": y2 - y, "kernel_size": kernel_size}
    return Image.fromarray(arr), Image.fromarray(mask), params


def apply_shift_region(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    shift_x: int = 10,
    shift_y: int = 0,
    fill_mode: str = "black",
) -> RGBImageAndMask:
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]
    x, y, x2, y2 = _clip_rect(x, y, width, height, w, h)

    region = arr[y:y2, x:x2].copy()
    _fill_region(arr, x, y, x2, y2, fill_mode=fill_mode)

    dest_x1 = np.clip(x + shift_x, 0, w)
    dest_y1 = np.clip(y + shift_y, 0, h)
    dest_x2 = np.clip(dest_x1 + region.shape[1], 0, w)
    dest_y2 = np.clip(dest_y1 + region.shape[0], 0, h)

    paste_w = max(0, dest_x2 - dest_x1)
    paste_h = max(0, dest_y2 - dest_y1)
    if paste_w > 0 and paste_h > 0:
        arr[dest_y1:dest_y2, dest_x1:dest_x2] = region[:paste_h, :paste_w]

    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255
    if paste_w > 0 and paste_h > 0:
        mask[dest_y1:dest_y2, dest_x1:dest_x2] = 255

    params = {
        "x": x,
        "y": y,
        "width": x2 - x,
        "height": y2 - y,
        "shift_x": shift_x,
        "shift_y": shift_y,
        "fill_mode": fill_mode,
    }
    return Image.fromarray(arr), Image.fromarray(mask), params


def apply_block_dropout(
    image: Image.Image,
    block_size: int = 16,
    drop_ratio: float = 0.2,
    fill_mode: str = "black",
    seed: int | None = None,
) -> RGBImageAndMask:
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]
    mask = _empty_mask(h, w)
    rng = np.random.default_rng(seed)
    block_size = max(4, int(block_size))
    drop_ratio = float(np.clip(drop_ratio, 0.0, 1.0))

    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            if rng.random() < drop_ratio:
                y2 = min(h, by + block_size)
                x2 = min(w, bx + block_size)
                _fill_region(arr, bx, by, x2, y2, fill_mode=fill_mode, seed=None if seed is None else seed + bx + by)
                mask[by:y2, bx:x2] = 255

    params = {"block_size": block_size, "drop_ratio": drop_ratio, "fill_mode": fill_mode, "seed": seed}
    return Image.fromarray(arr), Image.fromarray(mask), params


def apply_jpeg_block_artifacts(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    block_size: int = 8,
) -> RGBImageAndMask:
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]
    x, y, x2, y2 = _clip_rect(x, y, width, height, w, h)
    region = arr[y:y2, x:x2]
    block_size = max(4, int(block_size))

    small_w = max(1, region.shape[1] // block_size)
    small_h = max(1, region.shape[0] // block_size)
    down = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    up = cv2.resize(down, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
    arr[y:y2, x:x2] = up

    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255
    params = {"x": x, "y": y, "width": x2 - x, "height": y2 - y, "block_size": block_size}
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
) -> RGBImageAndMask:
    rng = np.random.default_rng(seed)
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]
    x, y, x2, y2 = _clip_rect(x, y, width, height, w, h)

    arr[y:y2, x:x2] = np.uint8(fill_value)
    halo = 10
    hx1 = max(0, x - halo)
    hy1 = max(0, y - halo)
    hx2 = min(w, x2 + halo)
    hy2 = min(h, y2 + halo)
    region = arr[hy1:hy2, hx1:hx2].astype(np.float32)
    noise = rng.normal(0, sigma, region.shape)
    arr[hy1:hy2, hx1:hx2] = np.clip(region + noise, 0, 255).astype(np.uint8)
    arr[y:y2, x:x2] = np.uint8(fill_value)

    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255
    params = {"x": x, "y": y, "width": x2 - x, "height": y2 - y, "fill_value": fill_value, "sigma": sigma, "seed": seed, "mode": "combined"}
    return Image.fromarray(arr), Image.fromarray(mask), params


def apply_mixed_corruption(
    image: Image.Image,
    seed: int | None = None,
    operations: list[dict[str, Any]] | None = None,
) -> RGBImageAndMask:
    rng = np.random.default_rng(seed)
    arr_img = image.copy().convert("RGB")
    total_mask = _empty_mask(arr_img.size[1], arr_img.size[0])
    applied: list[dict[str, Any]] = []

    if operations is None:
        ops = [
            {"type": "bar", "orientation": "horizontal", "thickness": int(rng.integers(8, 20))},
            {"type": "zone_deletion", **generate_random_rectangle(arr_img.size[0], arr_img.size[1], 25, 80, rng), "fill_mode": "noise"},
            {"type": "local_blur", **generate_random_rectangle(arr_img.size[0], arr_img.size[1], 30, 90, rng), "kernel_size": 11},
        ]
    else:
        ops = operations

    for idx, op in enumerate(ops):
        op = dict(op)
        op_type = op.pop("type")
        local_seed = None if seed is None else seed + idx
        if op_type == "bar":
            arr_img, mask_img, params = apply_bar_corruption(arr_img, seed=local_seed, **op)
        elif op_type == "zone_deletion":
            arr_img, mask_img, params = apply_zone_deletion(arr_img, seed=local_seed, **op)
        elif op_type == "noise":
            arr_img, mask_img, params = apply_noise(arr_img, seed=local_seed, **op)
        elif op_type == "local_blur":
            arr_img, mask_img, params = apply_local_blur(arr_img, **op)
        elif op_type == "shift_region":
            arr_img, mask_img, params = apply_shift_region(arr_img, **op)
        elif op_type == "block_dropout":
            arr_img, mask_img, params = apply_block_dropout(arr_img, seed=local_seed, **op)
        elif op_type == "jpeg_block_artifacts":
            arr_img, mask_img, params = apply_jpeg_block_artifacts(arr_img, **op)
        else:
            raise ValueError(f"Opération mixed non supportée : {op_type}")

        total_mask = np.maximum(total_mask, np.array(mask_img.convert("L"), dtype=np.uint8))
        applied.append({"type": op_type, **params})

    return arr_img, Image.fromarray(total_mask), {"operations": applied, "seed": seed}


def build_imperfect_mask(
    mask_image: Image.Image,
    dilate_iter: int = 1,
    erode_iter: int = 0,
    add_noise: bool = True,
    noise_ratio: float = 0.01,
    seed: int | None = None,
) -> Image.Image:
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


# Compatibilité éventuelle avec d'anciens essais

def irregular_mask(image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points = np.array([[50, 50], [80, 120], [140, 100], [120, 40]])
    cv2.fillPoly(mask, [points], 255)
    return mask


def localized_noise(image, mask):
    noisy = image.copy()
    noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
    noisy[mask == 255] = cv2.add(noisy[mask == 255], noise[mask == 255])
    return noisy



def apply_zone_deletion(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    fill_mode: str = "black",
) -> tuple[Image.Image, Image.Image, dict[str, int | str]]:
    image_np = np.array(image.convert("RGB"))
    corrupted = image_np.copy()
    mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

    img_h, img_w = image_np.shape[:2]
    x = max(0, min(int(x), img_w - 1))
    y = max(0, min(int(y), img_h - 1))
    width = max(1, min(int(width), img_w - x))
    height = max(1, min(int(height), img_h - y))

    x2 = x + width
    y2 = y + height

    mask[y:y2, x:x2] = 255

    if fill_mode == "black":
        corrupted[y:y2, x:x2] = 0
    elif fill_mode == "white":
        corrupted[y:y2, x:x2] = 255
    elif fill_mode == "noise":
        corrupted[y:y2, x:x2] = np.random.randint(
            0,
            256,
            (height, width, 3),
            dtype=np.uint8,
        )
    else:
        raise ValueError(f"fill_mode non supporté : {fill_mode}")

    params: dict[str, int | str] = {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "fill_mode": fill_mode,
    }
    return Image.fromarray(corrupted), Image.fromarray(mask), params

def apply_bar_corruption(image: np.ndarray, orientation: str = "horizontal", thickness: int = 12, count: int = 3):
    corrupted = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    h, w = image.shape[:2]

    if orientation == "horizontal":
        positions = np.linspace(0, h - thickness, count, dtype=int)
        for y in positions:
            corrupted[y:y+thickness, :] = 0
            mask[y:y+thickness, :] = 255
    elif orientation == "vertical":
        positions = np.linspace(0, w - thickness, count, dtype=int)
        for x in positions:
            corrupted[:, x:x+thickness] = 0
            mask[:, x:x+thickness] = 255
    else:
        raise ValueError("orientation doit être horizontal ou vertical")

    return corrupted, mask

def apply_shift_region(image: np.ndarray, x: int, y: int, width: int, height: int, dx: int, dy: int):
    corrupted = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    h, w = image.shape[:2]
    x2 = min(x + width, w)
    y2 = min(y + height, h)

    region = image[y:y2, x:x2].copy()

    new_x = max(0, min(w - (x2 - x), x + dx))
    new_y = max(0, min(h - (y2 - y), y + dy))

    corrupted[y:y2, x:x2] = 0
    corrupted[new_y:new_y + (y2 - y), new_x:new_x + (x2 - x)] = region

    mask[y:y2, x:x2] = 255
    mask[new_y:new_y + (y2 - y), new_x:new_x + (x2 - x)] = 255

    return corrupted, mask