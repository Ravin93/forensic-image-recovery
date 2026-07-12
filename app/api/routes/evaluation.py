from fastapi import APIRouter, HTTPException
from pathlib import Path

from app.modules.evaluation.comparator import compare_images

router = APIRouter()


@router.post("/evaluation")
def run_evaluation(data: dict):
    try:
        original = data.get("original")
        corrupted = data.get("corrupted")
        reconstructed = data.get("reconstructed")

        if not all([original, corrupted, reconstructed]):
            raise HTTPException(status_code=400, detail="Paramètres manquants")

        for path in [original, corrupted, reconstructed]:
            if not Path(path).exists():
                raise HTTPException(status_code=404, detail=f"{path} introuvable")

        result = compare_images(original, corrupted, reconstructed)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))