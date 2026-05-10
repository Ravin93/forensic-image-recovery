# Ajouter dans app/api/routes/report.py
# (ou remplacer si le fichier existe déjà)

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.core.config import REPORTS_DIR

router = APIRouter(tags=["reports"])


@router.get("/reports/json/{report_id}")
def get_json_report(report_id: str):
    """Retourne le rapport JSON par son ID."""
    path = REPORTS_DIR / f"report_{report_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Rapport introuvable : {report_id}")
    return FileResponse(str(path), media_type="application/json")


@router.get("/reports/pdf/{report_id}")
def download_pdf_report(report_id: str):
    """Télécharge le rapport PDF par son ID (Ticket E2).

    Si le PDF n'existe pas encore mais que le JSON existe,
    génère le PDF à la volée.
    """
    pdf_path = REPORTS_DIR / f"report_{report_id}.pdf"

    # PDF déjà généré
    if pdf_path.exists():
        return FileResponse(
            str(pdf_path),
            media_type="application/pdf",
            filename=f"forensic_report_{report_id}.pdf",
        )

    # JSON disponible → génère le PDF à la volée
    json_path = REPORTS_DIR / f"report_{report_id}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Rapport introuvable : {report_id}")

    try:
        import json
        from app.modules.reporting.pdf_report import generate_pdf_report
        with open(json_path, encoding="utf-8") as f:
            report = json.load(f)
        pdf_path = generate_pdf_report(report, output_path=pdf_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur génération PDF : {exc}")

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"forensic_report_{report_id}.pdf",
    )