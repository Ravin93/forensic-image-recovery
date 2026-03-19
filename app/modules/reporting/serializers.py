from pathlib import Path
from typing import Any
from datetime import datetime



def serialize_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(value)


def build_execution_metadata(source_image: str, status: str = "completed") -> dict[str, Any]:
    return {
        "source_image": source_image,
        "status": status,
    }


def build_run_metadata(source_image: str) -> dict:
    return {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now().isoformat(),
        "source_image": source_image,
        "status": "completed",
    }