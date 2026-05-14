from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.core.config import REPORTS_DIR

router = APIRouter(tags=["reports"])


@router.get("/reports/json/{report_id}")
def get_json_report(report_id: str):
    path = REPORTS_DIR / f"report_{report_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Rapport introuvable : {report_id}")
    return FileResponse(str(path), media_type="application/json")


@router.get("/reports/pdf/{report_id}")
def download_pdf_report(report_id: str):
    """Telecharge le rapport PDF (force le download)."""
    pdf_path = REPORTS_DIR / f"report_{report_id}.pdf"
    if pdf_path.exists():
        return FileResponse(str(pdf_path), media_type="application/pdf",
                           filename=f"forensic_report_{report_id}.pdf")
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
        raise HTTPException(status_code=500, detail=f"Erreur generation PDF : {exc}")
    return FileResponse(str(pdf_path), media_type="application/pdf",
                       filename=f"forensic_report_{report_id}.pdf")


@router.get("/reports/html/{report_id}")
def view_html_report(report_id: str):
    """Affiche le rapport HTML directement dans le navigateur (sans telechargement)."""
    html_path = REPORTS_DIR / f"report_{report_id}.html"
    if html_path.exists():
        # Pas de filename= : le navigateur affiche inline dans un nouvel onglet
        return FileResponse(str(html_path), media_type="text/html")
    json_path = REPORTS_DIR / f"report_{report_id}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Rapport introuvable : {report_id}")
    try:
        import json
        from app.modules.reporting.html_report import generate_html_report
        with open(json_path, encoding="utf-8") as f:
            report_data = json.load(f)
        html_path = generate_html_report(report_data, output_path=html_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur generation HTML : {exc}")
    return FileResponse(str(html_path), media_type="text/html")