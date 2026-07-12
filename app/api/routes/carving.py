from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.modules.carving.extractor import extract_jpegs_from_dump
from app.modules.carving.jpeg_scanner import scan_jpeg_offsets

router = APIRouter()


@router.post("/carving")
def run_carving(data: dict):
    try:
        dump_path = data.get("dump_path")

        if not dump_path:
            raise HTTPException(status_code=400, detail="dump_path requis")

        dump_path_obj = Path(dump_path)

        if not dump_path_obj.exists():
            raise HTTPException(status_code=404, detail="Dump introuvable")

        offsets = scan_jpeg_offsets(dump_path_obj)
        extracted = extract_jpegs_from_dump(dump_path_obj)

        return {
            "count": len(extracted),
            "offsets": offsets,
            "extracted": extracted,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))