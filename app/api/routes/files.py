from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter()

BASE_PATHS = {
    "extracted": "data/extracted",
    "corrupted": "data/corrupted",
    "masks": "data/masks",
    "reconstructed": "data/reconstructed",
}


@router.get("/files/{file_type}/{filename}")
def get_file(file_type: str, filename: str):
    if file_type not in BASE_PATHS:
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_path = Path(BASE_PATHS[file_type]) / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)