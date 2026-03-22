from pathlib import Path
from typing import Any

from PIL import Image

from app.modules.detection.advanced_detector import (
    detect_dark_regions,
    detect_gradient_breaks,
    detect_low_variance_regions,
)
from app.modules.detection.fusion import fuse_detection_masks


def detect_dark_regions_mask(image_path: str | Path, threshold: int = 22) -> Image.Image:
    result = detect_dark_regions(image_path=image_path, threshold=threshold)
    return Image.fromarray(result["mask"])


def detect_advanced_mask(
    image_path: str | Path,
    strategy: str = "weighted_union",
) -> dict[str, Any]:
    dark = detect_dark_regions(image_path)
    variance = detect_low_variance_regions(image_path)
    gradient = detect_gradient_breaks(image_path)

    dark_bin = (dark["mask"] > 0)
    variance_mask = ((variance["mask"] > 0) & dark_bin).astype("uint8") * 255
    gradient_mask = ((gradient["mask"] > 0) & dark_bin).astype("uint8") * 255

    fusion = fuse_detection_masks(
    [dark["mask"], variance_mask, gradient_mask],
    strategy=strategy,
    min_area=120,
    max_area_ratio=0.12,
    dilate_iter=1,
    return_metadata=True,
)

    return {
    "mask": Image.fromarray(fusion["mask"]),
    "method": "advanced_fusion",
    "confidence": fusion["confidence"],
    "area_ratio": fusion["area_ratio"],
    "components": {
        "dark_regions": dark["score"],
        "low_variance": variance["score"],
        "gradient_breaks": gradient["score"],
    },
}