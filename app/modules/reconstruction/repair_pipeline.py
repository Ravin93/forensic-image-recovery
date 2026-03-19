from pathlib import Path
from typing import Any

from app.core.logger import logger
from app.modules.reconstruction.inpainting import reconstruct_with_inpaint


def run_repair_pipeline(
    corrupted_image_path: str | Path,
    mask_path: str | Path,
    method: str = "opencv_inpaint",
    radius: int = 3,
) -> dict[str, Any]:
    logger.info(
        "Repair pipeline start | image=%s | mask=%s | method=%s",
        corrupted_image_path,
        mask_path,
        method,
    )

    result = reconstruct_with_inpaint(
        image_path=corrupted_image_path,
        mask_path=mask_path,
        method=method,
        radius=radius,
    )

    logger.info("Repair pipeline done")
    return result