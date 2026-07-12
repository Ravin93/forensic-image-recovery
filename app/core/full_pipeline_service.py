from pathlib import Path
from typing import Any

from app.core.logger import logger
from app.modules.carving.extractor import extract_jpegs_from_dump
from app.modules.validation.verifier import verify_image
from app.services.pipeline_service import run_demo_pipeline


def run_full_pipeline(
    dump_path: str | Path,
    corruption_type: str = "rectangle_mask",
    corruption_params: dict[str, Any] | None = None,
    inpaint_method: str = "opencv_inpaint",
    radius: int = 3,
    execution_mode: str = "assisted",
    detection_mode: str = "basic",
    corruption_level: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    dump_path = Path(dump_path)
    logger.info("FULL PIPELINE START | dump=%s", dump_path)

    extracted_images = extract_jpegs_from_dump(dump_path)
    if not extracted_images:
        raise ValueError("Aucune image extraite du dump")

    valid_images = []
    for img in extracted_images:
        validation = verify_image(img["path"], allowed_formats={"JPEG", "JPG"})
        if validation["valid"]:
            valid_images.append({"extraction": img, "validation": validation})

    if not valid_images:
        raise ValueError("Aucune image valide après validation")

    selected = valid_images[0]

    demo_result = run_demo_pipeline(
        source_image_path=selected["extraction"]["path"],
        corruption_type=corruption_type,
        corruption_params=corruption_params,
        inpaint_method=inpaint_method,
        radius=radius,
        execution_mode=execution_mode,
        detection_mode=detection_mode,
        corruption_level=corruption_level,
        seed=seed,
    )

    return {
        "dump_path": str(dump_path),
        "extracted_count": len(extracted_images),
        "valid_count": len(valid_images),
        "selected_image": selected["extraction"],
        "selected_validation": selected["validation"],
        "pipeline_result": demo_result,
        "status": "completed",
    }