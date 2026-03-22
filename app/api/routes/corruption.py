from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.modules.corruption.simulator import corrupt_image

router = APIRouter()


@router.post("/corruption")
def run_corruption_route(data: dict):
    try:
        image_path = data.get("image_path")
        corruption_type = data.get("type")
        params = data.get("params", {})

        if not image_path or not corruption_type:
            raise HTTPException(status_code=400, detail="Paramètres manquants")

        image_path_obj = Path(image_path)

        if not image_path_obj.exists():
            raise HTTPException(status_code=404, detail="Image introuvable")

        result = corrupt_image(
            image_path=image_path_obj,
            corruption_type=corruption_type,
            **params,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))