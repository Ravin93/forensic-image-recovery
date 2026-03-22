from pathlib import Path
from typing import Any

import cv2
import numpy as np


def classify_corruption_type(
    image_path: str | Path,
    detected_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")

    features: dict[str, float] = {
        "dark_ratio": float(np.mean(image < 25)),
        "bright_ratio": float(np.mean(image > 245)),
        "global_variance": float(np.var(image)),
    }

    lap = cv2.Laplacian(image.astype(np.float32), cv2.CV_32F)
    features["gradient_energy"] = float(np.mean(np.abs(lap)))

    if detected_mask is not None:
        features["mask_ratio"] = float(np.mean(detected_mask > 0))
    else:
        features["mask_ratio"] = 0.0

    if features["dark_ratio"] > 0.003 and features["gradient_energy"] < 20:
        ctype, conf = "mask_like", 0.75
    elif features["global_variance"] > 450 and features["mask_ratio"] > 0.001:
        ctype, conf = "noise_like", 0.65
    elif features["dark_ratio"] > 0.001 and features["global_variance"] > 250:
        ctype, conf = "mixed", 0.60
    else:
        ctype, conf = "unknown", 0.45

    return {
        "corruption_type": ctype,
        "confidence": conf,
        "features": features,
    }