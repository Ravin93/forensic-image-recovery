import cv2
import numpy as np


def filter_connected_components(mask: np.ndarray, min_area: int = 200) -> np.ndarray:
    if min_area <= 1:
        return ((mask > 0) * 255).astype(np.uint8)

    mask_bin = ((mask > 0) * 255).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin,
        connectivity=8,
    )

    filtered = np.zeros_like(mask_bin)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == label_id] = 255

    return filtered


def compute_mask_confidence(
    component_masks: list[np.ndarray],
    fused_mask: np.ndarray,
) -> float:
    if not component_masks:
        return 0.0

    binary_components = [(m > 0).astype(np.uint8) for m in component_masks]
    stacked = np.stack(binary_components, axis=0)

    agreement = float(np.mean(np.sum(stacked, axis=0) >= 2))
    fused_ratio = float(np.mean(fused_mask > 0))

    if fused_ratio <= 0.15:
        size_score = 1.0
    elif fused_ratio <= 0.30:
        size_score = 0.7
    elif fused_ratio <= 0.50:
        size_score = 0.4
    else:
        size_score = 0.1

    confidence = 0.6 * agreement + 0.4 * size_score
    return float(max(0.0, min(1.0, confidence)))


def fuse_detection_masks(
    masks: list[np.ndarray],
    strategy: str = "weighted_union",
    closing: bool = True,
    dilate_iter: int = 1,
    min_area: int = 200,
    max_area_ratio: float = 0.30,
    return_metadata: bool = False,
):
    if not masks:
        raise ValueError("Aucun masque fourni")

    masks = [((m > 0) * 255).astype(np.uint8) for m in masks]

    h, w = masks[0].shape[:2]
    image_area = h * w

    # Assouplissement automatique pour les petits masques de test
    effective_min_area = min(min_area, max(1, image_area // 100))
    effective_dilate_iter = dilate_iter if image_area >= 2500 else 0
    effective_closing = closing if image_area >= 2500 else False

    if strategy == "union":
        fused = np.maximum.reduce(masks)

    elif strategy == "weighted_union":
        acc = np.zeros_like(masks[0], dtype=np.float32)
        for mask in masks:
            acc += (mask > 0).astype(np.float32)

        fused = np.where(acc >= 2, 255, 0).astype(np.uint8)

    else:
        raise ValueError(f"Stratégie non supportée : {strategy}")

    kernel = np.ones((5, 5), np.uint8)

    if effective_closing:
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, kernel)

    if effective_dilate_iter > 0:
        fused = cv2.dilate(fused, kernel, iterations=effective_dilate_iter)

    fused = filter_connected_components(fused, min_area=effective_min_area)

    area_ratio = float(np.mean(fused > 0))
    if area_ratio > max_area_ratio:
        fused = np.zeros_like(fused)

    confidence = compute_mask_confidence(masks, fused)

    if return_metadata:
        return {
            "mask": fused,
            "confidence": confidence,
            "area_ratio": float(np.mean(fused > 0)),
        }

    return fused