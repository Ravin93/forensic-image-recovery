from fastapi import APIRouter, HTTPException
from pathlib import Path

from app.modules.validation.verifier import verify_image

router = APIRouter()


@router.post("/validation")
def run_validation(data: dict):
    try:
        image_path = data.get("image_path")

        if not image_path:
            raise HTTPException(status_code=400, detail="image_path requis")

        if not Path(image_path).exists():
            raise HTTPException(status_code=404, detail="Image introuvable")

        result = verify_image(image_path)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))