from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


RGBImageAndMask = tuple[Image.Image, Image.Image, dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _to_numpy_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def _empty_mask(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def _clip_rect(
    x: int, y: int, width: int, height: int, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    x = max(0, int(x))
    y = max(0, int(y))
    width = max(1, int(width))
    height = max(1, int(height))
    x2 = min(x + width, img_w)
    y2 = min(y + height, img_h)
    return x, y, x2, y2


def _fill_region(
    arr: np.ndarray,
    x: int,
    y: int,
    x2: int,
    y2: int,
    fill_mode: str,
    seed: int | None = None,
) -> None:
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


# ---------------------------------------------------------------------------
# Corruptions existantes (versions PIL, sans doublons)
# ---------------------------------------------------------------------------

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
    fill_mode: str = "black",
    seed: int | None = None,
) -> RGBImageAndMask:
    """Supprime une zone rectangulaire (noir / blanc / gris / bruit)."""
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]
    x, y, x2, y2 = _clip_rect(x, y, width, height, w, h)
    _fill_region(arr, x, y, x2, y2, fill_mode=fill_mode, seed=seed)
    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255
    params = {
        "x": x, "y": y, "width": x2 - x, "height": y2 - y,
        "fill_mode": fill_mode, "seed": seed,
    }
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
    """Une ou plusieurs barres horizontales / verticales."""
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
            _fill_region(arr, 0, start, w, end, fill_mode=fill_mode,
                         seed=None if seed is None else seed + i)
            mask[start:end, :] = 255
            bars.append({"y": start, "height": end - start})
        else:
            if start >= w:
                break
            end = min(w, start + thickness)
            _fill_region(arr, start, 0, end, h, fill_mode=fill_mode,
                         seed=None if seed is None else seed + i)
            mask[:, start:end] = 255
            bars.append({"x": start, "width": end - start})

    params = {
        "orientation": orientation, "thickness": thickness,
        "position": position, "count": len(bars),
        "gap": gap, "fill_mode": fill_mode, "bars": bars, "seed": seed,
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
    arr[y:y2, x:x2] = cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)
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

    dest_x1 = int(np.clip(x + shift_x, 0, w))
    dest_y1 = int(np.clip(y + shift_y, 0, h))
    dest_x2 = int(np.clip(dest_x1 + region.shape[1], 0, w))
    dest_y2 = int(np.clip(dest_y1 + region.shape[0], 0, h))

    paste_w = max(0, dest_x2 - dest_x1)
    paste_h = max(0, dest_y2 - dest_y1)
    if paste_w > 0 and paste_h > 0:
        arr[dest_y1:dest_y2, dest_x1:dest_x2] = region[:paste_h, :paste_w]

    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255
    if paste_w > 0 and paste_h > 0:
        mask[dest_y1:dest_y2, dest_x1:dest_x2] = 255

    params = {
        "x": x, "y": y, "width": x2 - x, "height": y2 - y,
        "shift_x": shift_x, "shift_y": shift_y, "fill_mode": fill_mode,
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
                _fill_region(arr, bx, by, x2, y2, fill_mode=fill_mode,
                             seed=None if seed is None else seed + bx + by)
                mask[by:y2, bx:x2] = 255

    params = {"block_size": block_size, "drop_ratio": drop_ratio,
              "fill_mode": fill_mode, "seed": seed}
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
    hx1 = max(0, x - halo); hy1 = max(0, y - halo)
    hx2 = min(w, x2 + halo); hy2 = min(h, y2 + halo)
    region = arr[hy1:hy2, hx1:hx2].astype(np.float32)
    noise = rng.normal(0, sigma, region.shape)
    arr[hy1:hy2, hx1:hx2] = np.clip(region + noise, 0, 255).astype(np.uint8)
    arr[y:y2, x:x2] = np.uint8(fill_value)
    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255
    params = {
        "x": x, "y": y, "width": x2 - x, "height": y2 - y,
        "fill_value": fill_value, "sigma": sigma, "seed": seed, "mode": "combined",
    }
    return Image.fromarray(arr), Image.fromarray(mask), params


# ---------------------------------------------------------------------------
# Nouvelles corruptions réalistes — Ticket 1
# ---------------------------------------------------------------------------

def apply_scratch_lines(
    image: Image.Image,
    count: int = 5,
    thickness: int = 1,
    angle_range: tuple[float, float] = (-30.0, 30.0),
    color: tuple[int, int, int] | None = None,
    seed: int | None = None,
) -> RGBImageAndMask:
    """Rayures fines ou épaisses simulant des égratignures physiques.

    Args:
        count: nombre de rayures.
        thickness: épaisseur en pixels (1 = fine, 3+ = épaisse).
        angle_range: plage d'angles en degrés (0 = horizontal, 90 = vertical).
        color: couleur RGB de la rayure. None → blanc (255, 255, 255).
        seed: graine aléatoire.
    """
    rng = np.random.default_rng(seed)
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]
    mask = _empty_mask(h, w)

    scratch_color = color if color is not None else (255, 255, 255)
    thickness = max(1, int(thickness))

    for i in range(count):
        angle_deg = float(rng.uniform(angle_range[0], angle_range[1]))
        angle_rad = np.deg2rad(angle_deg)

        # Point de départ aléatoire sur un bord de l'image
        start_x = int(rng.integers(0, w))
        start_y = int(rng.integers(0, h))

        length = int(rng.integers(min(w, h) // 3, max(w, h)))
        end_x = int(start_x + length * np.cos(angle_rad))
        end_y = int(start_y + length * np.sin(angle_rad))

        cv2.line(arr, (start_x, start_y), (end_x, end_y),
                 scratch_color, thickness, lineType=cv2.LINE_AA)
        cv2.line(mask, (start_x, start_y), (end_x, end_y),
                 255, max(thickness, 2), lineType=cv2.LINE_AA)

    params = {
        "count": count, "thickness": thickness,
        "angle_range": list(angle_range), "color": list(scratch_color), "seed": seed,
    }
    return Image.fromarray(arr), Image.fromarray(mask), params


def apply_large_deleted_square(
    image: Image.Image,
    size_ratio: float = 0.4,
    fill_mode: str = "black",
    seed: int | None = None,
) -> RGBImageAndMask:
    """Supprime un grand carré centré ou légèrement décalé.

    Args:
        size_ratio: fraction de la plus petite dimension (0.2–0.7 recommandé).
        fill_mode: 'black' | 'white' | 'gray' | 'noise'.
        seed: graine aléatoire.
    """
    rng = np.random.default_rng(seed)
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]

    side = int(min(w, h) * float(np.clip(size_ratio, 0.1, 0.8)))
    # Centre légèrement aléatoire
    cx = int(rng.integers(side // 2, w - side // 2))
    cy = int(rng.integers(side // 2, h - side // 2))
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    _fill_region(arr, x1, y1, x2, y2, fill_mode=fill_mode, seed=seed)
    mask = _empty_mask(h, w)
    mask[y1:y2, x1:x2] = 255
    params = {
        "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
        "size_ratio": size_ratio, "fill_mode": fill_mode, "seed": seed,
    }
    return Image.fromarray(arr), Image.fromarray(mask), params


def apply_multiple_bars(
    image: Image.Image,
    count: int = 4,
    orientation: str = "horizontal",
    thickness: int = 10,
    fill_mode: str = "black",
    seed: int | None = None,
) -> RGBImageAndMask:
    """Barres multiples réparties aléatoirement (plus dense que apply_bar_corruption).

    Args:
        count: nombre de barres.
        orientation: 'horizontal' ou 'vertical'.
        thickness: épaisseur de chaque barre.
        fill_mode: 'black' | 'white' | 'gray' | 'noise'.
        seed: graine aléatoire.
    """
    rng = np.random.default_rng(seed)
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]
    mask = _empty_mask(h, w)
    orientation = orientation.lower()
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation doit être 'horizontal' ou 'vertical'")

    thickness = max(1, int(thickness))
    bars = []

    limit = h if orientation == "horizontal" else w
    positions = sorted(rng.integers(0, max(1, limit - thickness), size=count).tolist())

    for idx, pos in enumerate(positions):
        local_seed = None if seed is None else seed + idx
        if orientation == "horizontal":
            end = min(h, pos + thickness)
            _fill_region(arr, 0, pos, w, end, fill_mode=fill_mode, seed=local_seed)
            mask[pos:end, :] = 255
            bars.append({"y": pos, "height": end - pos})
        else:
            end = min(w, pos + thickness)
            _fill_region(arr, pos, 0, end, h, fill_mode=fill_mode, seed=local_seed)
            mask[:, pos:end] = 255
            bars.append({"x": pos, "width": end - pos})

    params = {
        "count": count, "orientation": orientation,
        "thickness": thickness, "fill_mode": fill_mode,
        "bars": bars, "seed": seed,
    }
    return Image.fromarray(arr), Image.fromarray(mask), params


def apply_random_holes(
    image: Image.Image,
    count: int = 8,
    min_size: int = 10,
    max_size: int = 40,
    fill_mode: str = "black",
    seed: int | None = None,
) -> RGBImageAndMask:
    """Trous rectangulaires multiples dispersés aléatoirement.

    Args:
        count: nombre de trous.
        min_size / max_size: taille min/max en pixels.
        fill_mode: 'black' | 'white' | 'gray' | 'noise'.
        seed: graine aléatoire.
    """
    rng = np.random.default_rng(seed)
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]
    mask = _empty_mask(h, w)
    holes = []

    for i in range(count):
        hw = int(rng.integers(min_size, max(min_size + 1, min(max_size, w // 2))))
        hh = int(rng.integers(min_size, max(min_size + 1, min(max_size, h // 2))))
        hx = int(rng.integers(0, max(1, w - hw)))
        hy = int(rng.integers(0, max(1, h - hh)))
        local_seed = None if seed is None else seed + i * 7
        _fill_region(arr, hx, hy, hx + hw, hy + hh, fill_mode=fill_mode, seed=local_seed)
        mask[hy:hy + hh, hx:hx + hw] = 255
        holes.append({"x": hx, "y": hy, "width": hw, "height": hh})

    params = {
        "count": count, "min_size": min_size, "max_size": max_size,
        "fill_mode": fill_mode, "holes": holes, "seed": seed,
    }
    return Image.fromarray(arr), Image.fromarray(mask), params


def apply_local_noise(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    sigma: float = 40.0,
    noise_type: str = "gaussian",
    seed: int | None = None,
) -> RGBImageAndMask:
    """Bruit localisé sur une zone précise.

    Différent de apply_noise : supporte gaussian ET salt_pepper.

    Args:
        noise_type: 'gaussian' | 'salt_pepper'.
        sigma: intensité du bruit gaussien (ignoré pour salt_pepper).
        seed: graine aléatoire.
    """
    rng = np.random.default_rng(seed)
    arr = _to_numpy_rgb(image)
    h, w = arr.shape[:2]
    x, y, x2, y2 = _clip_rect(x, y, width, height, w, h)
    region = arr[y:y2, x:x2].astype(np.float32)

    if noise_type == "gaussian":
        noise = rng.normal(0, sigma, region.shape)
        arr[y:y2, x:x2] = np.clip(region + noise, 0, 255).astype(np.uint8)
    elif noise_type == "salt_pepper":
        sp_mask = rng.random(region.shape[:2])
        region_u8 = arr[y:y2, x:x2].copy()
        region_u8[sp_mask < 0.05] = 0    # poivre
        region_u8[sp_mask > 0.95] = 255  # sel
        arr[y:y2, x:x2] = region_u8
    else:
        raise ValueError(f"noise_type non supporté : {noise_type}")

    mask = _empty_mask(h, w)
    mask[y:y2, x:x2] = 255
    params = {
        "x": x, "y": y, "width": x2 - x, "height": y2 - y,
        "sigma": sigma, "noise_type": noise_type, "seed": seed,
    }
    return Image.fromarray(arr), Image.fromarray(mask), params


# ---------------------------------------------------------------------------
# Corruption mixte
# ---------------------------------------------------------------------------

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
            {"type": "bar", "orientation": "horizontal",
             "thickness": int(rng.integers(8, 20))},
            {"type": "zone_deletion",
             **generate_random_rectangle(arr_img.size[0], arr_img.size[1], 25, 80, rng),
             "fill_mode": "noise"},
            {"type": "local_blur",
             **generate_random_rectangle(arr_img.size[0], arr_img.size[1], 30, 90, rng),
             "kernel_size": 11},
        ]
    else:
        ops = operations

    _DISPATCH: dict[str, Any] = {
        "bar": apply_bar_corruption,
        "zone_deletion": apply_zone_deletion,
        "noise": apply_noise,
        "local_blur": apply_local_blur,
        "shift_region": apply_shift_region,
        "block_dropout": apply_block_dropout,
        "jpeg_block_artifacts": apply_jpeg_block_artifacts,
        "scratch_lines": apply_scratch_lines,
        "large_deleted_square": apply_large_deleted_square,
        "multiple_bars": apply_multiple_bars,
        "random_holes": apply_random_holes,
        "local_noise": apply_local_noise,
    }

    for idx, op in enumerate(ops):
        op = dict(op)
        op_type = op.pop("type")
        local_seed = None if seed is None else seed + idx

        fn = _DISPATCH.get(op_type)
        if fn is None:
            raise ValueError(f"Opération mixed non supportée : {op_type}")

        # Injection du seed si la fonction le supporte
        import inspect
        sig = inspect.signature(fn)
        if "seed" in sig.parameters:
            op.setdefault("seed", local_seed)

        arr_img, mask_img, params = fn(arr_img, **op)
        total_mask = np.maximum(total_mask, np.array(mask_img.convert("L"), dtype=np.uint8))
        applied.append({"type": op_type, **params})

    return arr_img, Image.fromarray(total_mask), {"operations": applied, "seed": seed}


# ---------------------------------------------------------------------------
# Masque imparfait (bruit sur le masque exact)
# ---------------------------------------------------------------------------

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