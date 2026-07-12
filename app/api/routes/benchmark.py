from __future__ import annotations
from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
router = APIRouter(prefix="/benchmark", tags=["benchmark"])
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

@router.post("/run")
def run_benchmark_endpoint(
    image_names: list[str] = Query(default=["demo_real.jpeg"]),
    corruption_types: list[str] = Query(default=[]),
    max_attempts: int = Query(default=4, ge=1, le=8),
    seed: int = Query(default=42),
):
    """Lance le benchmark sur les images de data/input/."""
    from app.modules.benchmark.benchmark_runner import run_benchmark
    image_paths = []
    for name in image_names:
        p = _PROJECT_ROOT / "data" / "input" / name
        if p.exists():
            image_paths.append(p)
    if not image_paths:
        raise HTTPException(status_code=404, detail="Aucune image trouvee dans data/input/")
    try:
        result = run_benchmark(
            image_paths=image_paths,
            corruption_types=corruption_types or None,
            max_attempts=max_attempts,
            seed=seed,
        )
        return {
            "status":   "completed",
            "n_rows":   len(result["rows"]),
            "summary":  result["summary"],
            "paths":    result["paths"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/results")
def list_benchmark_results():
    """Liste les fichiers benchmark JSON disponibles."""
    reports_dir = _PROJECT_ROOT / "data" / "reports"
    files = sorted(reports_dir.glob("benchmark_*.json"), reverse=True)[:10]
    return [{"name": f.name, "path": str(f), "size": f.stat().st_size} for f in files]

@router.get("/download/{filename}")
def download_benchmark_file(filename: str):
    """Telecharge un fichier de benchmark (CSV ou JSON)."""
    if not filename.startswith("benchmark_") or ".." in filename:
        raise HTTPException(status_code=403, detail="Acces refuse")
    path = _PROJECT_ROOT / "data" / "reports" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    media = "text/csv" if filename.endswith(".csv") else "application/json"
    return FileResponse(str(path), media_type=media, filename=filename)