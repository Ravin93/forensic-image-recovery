from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.modules.reconstruction.repair_pipeline import run_repair_pipeline

router = APIRouter()


@router.post("/reconstruction")
def run_reconstruction(data: dict):
    try:
        image_path = data.get("image_path")
        mask_path = data.get("mask_path")

        if not image_path or not mask_path:
            raise HTTPException(status_code=400, detail="Paramètres manquants")

        image_path_obj = Path(image_path)
        mask_path_obj = Path(mask_path)

        if not image_path_obj.exists():
            raise HTTPException(status_code=404, detail="Image introuvable")

        if not mask_path_obj.exists():
            raise HTTPException(status_code=404, detail="Mask introuvable")

        result = run_repair_pipeline(
            corrupted_image_path=image_path_obj,
            mask_path=mask_path_obj,
            method="opencv_inpaint",
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))