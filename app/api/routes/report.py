from fastapi import APIRouter, HTTPException
from pathlib import Path
import json

router = APIRouter()


@router.get("/report/{report_id}")
def get_report(report_id: str):
    report_path = Path(f"data/reports/{report_id}.json")

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    with open(report_path, "r") as f:
        return json.load(f)