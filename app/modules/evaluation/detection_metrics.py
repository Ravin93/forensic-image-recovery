import numpy as np


def _bin(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def compute_iou(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    true = _bin(mask_true)
    pred = _bin(mask_pred)

    inter = np.logical_and(true, pred).sum()
    union = np.logical_or(true, pred).sum()

    if union == 0:
        return 1.0
    return float(inter / union)


def compute_precision_recall(mask_true: np.ndarray, mask_pred: np.ndarray) -> dict[str, float]:
    true = _bin(mask_true)
    pred = _bin(mask_pred)

    tp = np.logical_and(true == 1, pred == 1).sum()
    fp = np.logical_and(true == 0, pred == 1).sum()
    fn = np.logical_and(true == 1, pred == 0).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
    }