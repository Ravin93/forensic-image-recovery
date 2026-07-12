"""app/api/routes/analysis.py — I1+I2+I3+I4.

Endpoints :
    POST /analysis/start              Lance une analyse complète (async via BackgroundTasks)
    GET  /analysis/{id}/status        Statut courant
    GET  /analysis/{id}/result        Résultat complet (si completed)
    GET  /analysis/{id}/download      Télécharge le rapport PDF
    GET  /analysis/                   Liste des analyses récentes
    DELETE /analysis/{id}             Supprime une analyse
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.modules.analysis.analysis_store import (
    create_analysis,
    delete_analysis,
    get_result,
    get_status,
    list_analyses,
    new_analysis_id,
    save_analysis_file,
    save_result,
    update_status,
)

router = APIRouter(prefix="/analysis", tags=["analysis"])

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_INPUT_DIR    = _PROJECT_ROOT / "data" / "input"
_INPUT_DIR.mkdir(parents=True, exist_ok=True)

_SUPPORTED_CORRUPTIONS = {
    "rectangle_mask", "noise", "zone_deletion", "combined", "bar",
    "local_blur", "shift_region", "block_dropout", "jpeg_block_artifacts",
    "mixed", "scratch_lines", "large_deleted_square", "multiple_bars",
    "random_holes", "local_noise",
}


# ---------------------------------------------------------------------------
# I4 — Tâche de fond (BackgroundTasks FastAPI)
# ---------------------------------------------------------------------------

def _run_analysis_task(
    analysis_id: str,
    source_path: Path,
    corruption_type: str,
    severity: str,
    execution_mode: str,
    max_attempts: int,
    seed: int | None,
) -> None:
    """Tâche exécutée en arrière-plan par FastAPI BackgroundTasks."""
    try:
        update_status(analysis_id, "running")

        from app.services.pipeline_service import run_demo_pipeline
        result = run_demo_pipeline(
            source_image_path=source_path,
            corruption_type=corruption_type,
            corruption_params={},
            execution_mode=execution_mode,
            seed=seed,
            max_attempts=max_attempts,
            randomize=True,
        )

        # Sauvegarder les fichiers dans data/analyses/{id}/
        for key, path_key in [
            ("original",      "original_image"),
            ("corrupted",     "corrupted_image"),
            ("reconstructed", "reconstructed_image"),
            ("mask",          "mask_path"),
        ]:
            p = result.get(path_key)
            if p and Path(p).exists():
                save_analysis_file(analysis_id, key, p)

        # Rapport JSON
        report_path = result.get("report_path")
        if report_path and Path(report_path).exists():
            save_analysis_file(analysis_id, "report_json", report_path)

        # Générer le PDF
        try:
            import json as _json
            from app.modules.reporting.pdf_report import generate_pdf_report
            from app.modules.analysis.analysis_store import analysis_dir
            with open(report_path, encoding="utf-8") as f:
                report_data = _json.load(f)
            pdf_path = analysis_dir(analysis_id) / "report.pdf"
            generate_pdf_report(report_data, output_path=pdf_path)
            save_analysis_file(analysis_id, "report_pdf", pdf_path, "report.pdf")
        except Exception as pdf_err:
            pass  # PDF optionnel

        # Résultat complet
        save_result(analysis_id, result)
        update_status(analysis_id, "completed", extra={
            "score":                    result.get("repair_score"),
            "selected_repair_strategy": result.get("selected_repair_strategy"),
            "corruption_type":          corruption_type,
        })

    except Exception as exc:
        update_status(analysis_id, "failed", extra={"error": str(exc)})


# ---------------------------------------------------------------------------
# I1 — POST /analysis/start
# ---------------------------------------------------------------------------

@router.post("/start")
async def start_analysis(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    corruption_type: str = Form("scratch_lines"),
    severity: str = Form("medium"),
    execution_mode: str = Form("assisted"),
    max_attempts: int = Form(8, ge=1, le=11),
    seed: int | None = Form(None),
) -> dict[str, Any]:
    """Lance une analyse complète en arrière-plan.

    Retourne immédiatement un `analysis_id` pour suivre la progression
    via GET /analysis/{id}/status.
    """
    # Validation
    ct = corruption_type.strip().lower()
    if ct not in _SUPPORTED_CORRUPTIONS:
        raise HTTPException(status_code=422, detail=f"Type non supporté : '{ct}'")
    if severity not in ("light", "medium", "heavy"):
        raise HTTPException(status_code=422, detail=f"Sévérité invalide : '{severity}'")

    suffix = Path(image.filename or "img.png").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=422, detail=f"Format non supporté : '{suffix}'")

    # Sauvegarde de l'upload
    ts = int(time.time())
    source_path = _INPUT_DIR / f"analysis_{ts}{suffix}"
    try:
        content = await image.read()
        source_path.write_bytes(content)
    finally:
        await image.close()

    # Création de l'analyse
    analysis_id = new_analysis_id()
    create_analysis(analysis_id, request_info={
        "corruption_type": ct,
        "severity":        severity,
        "execution_mode":  execution_mode,
        "max_attempts":    max_attempts,
        "seed":            seed,
        "filename":        image.filename,
    })

    # Lancement en arrière-plan (I4)
    background_tasks.add_task(
        _run_analysis_task,
        analysis_id=analysis_id,
        source_path=source_path,
        corruption_type=ct,
        severity=severity,
        execution_mode=execution_mode,
        max_attempts=max_attempts,
        seed=seed,
    )

    return {
        "analysis_id": analysis_id,
        "status":      "pending",
        "message":     "Analyse lancée. Suivez la progression via GET /analysis/{id}/status",
        "status_url":  f"/analysis/{analysis_id}/status",
        "result_url":  f"/analysis/{analysis_id}/result",
        "download_url": f"/analysis/{analysis_id}/download",
    }


# ---------------------------------------------------------------------------
# I3 — GET /analysis/{id}/status
# ---------------------------------------------------------------------------

@router.get("/{analysis_id}/status")
def get_analysis_status(analysis_id: str) -> dict[str, Any]:
    """Retourne le statut courant d'une analyse."""
    try:
        return get_status(analysis_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Analyse introuvable : {analysis_id}")


# ---------------------------------------------------------------------------
# I3 — GET /analysis/{id}/result
# ---------------------------------------------------------------------------

@router.get("/{analysis_id}/result")
def get_analysis_result(analysis_id: str) -> dict[str, Any]:
    """Retourne le résultat complet d'une analyse terminée."""
    try:
        status = get_status(analysis_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Analyse introuvable : {analysis_id}")

    if status["status"] == "pending":
        raise HTTPException(status_code=202, detail="Analyse en attente de démarrage.")
    if status["status"] == "running":
        raise HTTPException(status_code=202, detail="Analyse en cours, réessayez dans quelques secondes.")
    if status["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Analyse échouée : {status.get('error')}")

    try:
        return get_result(analysis_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Résultat non disponible.")


# ---------------------------------------------------------------------------
# I3 — GET /analysis/{id}/download
# ---------------------------------------------------------------------------

@router.get("/{analysis_id}/download")
def download_analysis_pdf(analysis_id: str):
    """Télécharge le rapport PDF de l'analyse."""
    try:
        status = get_status(analysis_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Analyse introuvable : {analysis_id}")

    if status["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Analyse non terminée (statut : {status['status']})"
        )

    pdf_path = Path(status.get("files", {}).get("report_pdf", ""))
    if not pdf_path.exists():
        # Tentative de génération à la volée
        try:
            from app.modules.analysis.analysis_store import get_result, analysis_dir
            import json as _json
            from app.modules.reporting.pdf_report import generate_pdf_report
            result = get_result(analysis_id)
            pdf_path = analysis_dir(analysis_id) / "report.pdf"
            generate_pdf_report(result, output_path=pdf_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Impossible de générer le PDF : {exc}")

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"forensic_report_{analysis_id}.pdf",
    )


# ---------------------------------------------------------------------------
# Liste et suppression
# ---------------------------------------------------------------------------

@router.get("/")
def list_recent_analyses(limit: int = 20) -> list[dict[str, Any]]:
    """Retourne les analyses récentes (max 50)."""
    return list_analyses(limit=min(limit, 50))


@router.delete("/{analysis_id}")
def delete_analysis_endpoint(analysis_id: str) -> dict[str, str]:
    """Supprime une analyse et tous ses fichiers."""
    if not delete_analysis(analysis_id):
        raise HTTPException(status_code=404, detail=f"Analyse introuvable : {analysis_id}")
    return {"status": "deleted", "analysis_id": analysis_id}