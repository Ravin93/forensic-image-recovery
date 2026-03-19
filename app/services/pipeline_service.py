from pathlib import Path
from typing import Any

from app.core.logger import logger
from app.modules.corruption.simulator import corrupt_image
from app.modules.evaluation.comparator import compare_images
from app.modules.reconstruction.repair_pipeline import run_repair_pipeline
from app.modules.reporting.json_report import build_report, save_json_report
from app.modules.validation.verifier import verify_image


def run_demo_pipeline(
    source_image_path: str | Path,
    corruption_type: str = "rectangle_mask",
    corruption_params: dict[str, Any] | None = None,
    inpaint_method: str = "opencv_inpaint",
    radius: int = 3,
) -> dict[str, Any]:
    source_image_path = Path(source_image_path)

    if corruption_params is None:
        corruption_params = {
            "x": 20,
            "y": 20,
            "width": 40,
            "height": 40,
            "fill_value": 0,
        }

    logger.info("Pipeline start | source=%s", source_image_path)

    validation = verify_image(
        source_image_path,
        allowed_formats={"JPEG", "JPG", "PNG"},
    )
    if not validation["valid"]:
        raise ValueError(f"Image source invalide : {validation['reason']}")

    corruption_result = corrupt_image(
        image_path=source_image_path,
        corruption_type=corruption_type,
        **corruption_params,
    )

    reconstruction_result = run_repair_pipeline(
        corrupted_image_path=corruption_result["path"],
        mask_path=corruption_result["mask_path"],
        method=inpaint_method,
        radius=radius,
    )

    comparison_result = compare_images(
        original_path=source_image_path,
        corrupted_path=corruption_result["path"],
        reconstructed_path=reconstruction_result["path"],
    )

    report = build_report(
        source_image=str(source_image_path),
        corruption_result=corruption_result,
        reconstruction_result=reconstruction_result,
        comparison_result=comparison_result,
        status="completed",
    )
    report_path = save_json_report(report)

    result = {
        "source_image": str(source_image_path),
        "validation": validation,
        "corruption": corruption_result,
        "reconstruction": reconstruction_result,
        "evaluation": comparison_result,
        "report_path": str(report_path),
        "status": "completed",
    }

    logger.info("Pipeline done")
    return result