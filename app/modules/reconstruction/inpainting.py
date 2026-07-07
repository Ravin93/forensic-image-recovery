from pathlib import Path
from typing import Any

import cv2
import numpy as np

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


def _patch_bounds(
    y: int,
    x: int,
    half: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, y - half),
        min(height, y + half + 1),
        max(0, x - half),
        min(width, x + half + 1),
    )


def _criminisi_priority(
    gray: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray,
    patch_size: int,
) -> tuple[int, int] | None:
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    frontier = cv2.subtract(mask, eroded)
    points = np.column_stack(np.where(frontier > 0))
    if len(points) == 0:
        return None

    half = patch_size // 2
    height, width = mask.shape
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mask_f = mask.astype(np.float32) / 255.0
    normal_x = cv2.Sobel(mask_f, cv2.CV_32F, 1, 0, ksize=3)
    normal_y = cv2.Sobel(mask_f, cv2.CV_32F, 0, 1, ksize=3)

    best_point: tuple[int, int] | None = None
    best_priority = -1.0
    for y, x in points:
        y1, y2, x1, x2 = _patch_bounds(int(y), int(x), half, height, width)
        known = mask[y1:y2, x1:x2] == 0
        confidence_term = float(confidence[y1:y2, x1:x2][known].mean()) if known.any() else 0.0

        nx = float(normal_x[y, x])
        ny = float(normal_y[y, x])
        norm = (nx * nx + ny * ny) ** 0.5
        if norm > 1e-6:
            nx /= norm
            ny /= norm

        isophote_x = -float(grad_y[y, x])
        isophote_y = float(grad_x[y, x])
        data_term = abs(isophote_x * nx + isophote_y * ny) / 255.0
        priority = confidence_term * (data_term + 0.001)
        if priority > best_priority:
            best_priority = priority
            best_point = (int(y), int(x))

    return best_point


def _find_criminisi_source_patch(
    image: np.ndarray,
    mask: np.ndarray,
    center: tuple[int, int],
    patch_size: int,
) -> tuple[int, int] | None:
    height, width = mask.shape
    half = patch_size // 2
    cy, cx = center
    ty1, ty2, tx1, tx2 = _patch_bounds(cy, cx, half, height, width)
    target = image[ty1:ty2, tx1:tx2].astype(np.float32)
    known = mask[ty1:ty2, tx1:tx2] == 0
    if not known.any():
        return None

    best_center: tuple[int, int] | None = None
    best_ssd = float("inf")
    step = max(1, patch_size // 3)

    for sy in range(half, height - half, step):
        for sx in range(half, width - half, step):
            source_mask = mask[sy - half:sy + half + 1, sx - half:sx + half + 1]
            if source_mask.shape[:2] != target.shape[:2] or source_mask.any():
                continue

            source = image[sy - half:sy + half + 1, sx - half:sx + half + 1].astype(np.float32)
            diff = source - target
            ssd = float((diff[known] ** 2).sum())
            if ssd < best_ssd:
                best_ssd = ssd
                best_center = (sy, sx)

    return best_center


def criminisi_inpaint(
    image_path: str | Path,
    mask_path: str | Path,
    patch_size: int = 9,
) -> dict[str, Any]:
    """Reconstruit une image par inpainting exemplar-based de Criminisi.

    La zone masquée est remplie en priorité sur les contours dont la structure
    locale est la plus forte, puis copiée depuis le patch connu le plus proche
    en SSD sur les pixels déjà observés.
    """
    ensure_directories()
    image_path = Path(image_path)
    mask_path = Path(mask_path)

    if patch_size < 3 or patch_size % 2 == 0:
        raise ReconstructionError("patch_size Criminisi doit être impair et >= 3")
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

    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    result = image.copy()
    confidence = (mask == 0).astype(np.float32)
    half = patch_size // 2
    height, width = mask.shape
    max_steps = height * width

    for _ in range(max_steps):
        if mask.max() == 0:
            break

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        target_center = _criminisi_priority(gray, mask, confidence, patch_size)
        if target_center is None:
            break

        source_center = _find_criminisi_source_patch(result, mask, target_center, patch_size)
        if source_center is None:
            break

        ty, tx = target_center
        sy, sx = source_center
        ty1, ty2, tx1, tx2 = _patch_bounds(ty, tx, half, height, width)
        sy1 = sy - (ty - ty1)
        sx1 = sx - (tx - tx1)
        sy2 = sy1 + (ty2 - ty1)
        sx2 = sx1 + (tx2 - tx1)
        if sy1 < 0 or sx1 < 0 or sy2 > height or sx2 > width:
            break

        fill = mask[ty1:ty2, tx1:tx2] > 0
        if not fill.any():
            mask[ty, tx] = 0
            continue

        source_patch = result[sy1:sy2, sx1:sx2]
        result[ty1:ty2, tx1:tx2][fill] = source_patch[fill]
        confidence_value = float(confidence[ty1:ty2, tx1:tx2][~fill].mean()) if (~fill).any() else 1.0
        confidence[ty1:ty2, tx1:tx2][fill] = confidence_value
        mask[ty1:ty2, tx1:tx2][fill] = 0

    method_name = f"criminisi_p{patch_size}"
    output_path = build_reconstructed_image_path(image_path.name, method_name)
    ok = cv2.imwrite(str(output_path), result)
    if not ok:
        raise ReconstructionError(f"Impossible d'écrire l'image reconstruite : {output_path}")

    result_dict = {
        "file": output_path.name,
        "path": str(output_path),
        "method": method_name,
        "status": "reconstructed",
        "source_image": image_path.name,
        "mask_path": str(mask_path),
        "patch_size": patch_size,
    }
    logger.info("Criminisi reconstruction OK : %s", result_dict)
    return result_dict
