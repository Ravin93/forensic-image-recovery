from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.modules.corruption.simulator import SUPPORTED_CORRUPTIONS, corrupt_image

router = APIRouter()


@router.post("/corruption")
def run_corruption_route(data: dict):
    try:
        image_path = data.get("image_path")
        corruption_type = data.get("corruption_type") or data.get("type")
        params = data.get("params", {}) or {}
        randomize = bool(data.get("randomize", False))
        imperfect_mask = bool(data.get("imperfect_mask", False))
        seed = data.get("seed")

        if not image_path or not corruption_type:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Paramètres manquants",
                    "required": ["image_path", "corruption_type"],
                    "supported_types": sorted(SUPPORTED_CORRUPTIONS),
                },
            )

        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise HTTPException(status_code=404, detail="Image introuvable")

        result = corrupt_image(
            image_path=image_path_obj,
            corruption_type=str(corruption_type),
            randomize=randomize,
            imperfect_mask=imperfect_mask,
            seed=seed,
            **params,
        )
        result["supported_types"] = sorted(SUPPORTED_CORRUPTIONS)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
