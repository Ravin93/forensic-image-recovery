"""Tests Ticket E1+E2 — rapport PDF."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from app.modules.reporting.pdf_report import generate_pdf_report


@pytest.fixture
def sample_report():
    return {
        "run_id": "testpdf001",
        "timestamp": "2026-05-08T12:00:00",
        "status": "completed",
        "input": {
            "source_image": "data/input/demo_real.jpeg",
            "corruption_type": "scratch_lines",
            "execution_mode": "assisted",
        },
        "corruption": {
            "type": "scratch_lines",
            "parameters": {"count": 5, "thickness": 2},
            "image_path": None,
            "mask_path": None,
            "randomize": False,
            "imperfect_mask": False,
        },
        "reconstruction": {
            "best_candidate": {
                "strategy": "inpainting_r5",
                "path": None,
                "score": 74.3,
                "psnr": 28.1,
                "ssim": 0.87,
                "gain_psnr": 4.2,
                "gain_ssim": 0.12,
                "mode": "supervised",
            },
            "all_candidates": [
                {"strategy": "inpainting_r5", "score": 74.3, "psnr": 28.1, "ssim": 0.87, "gain_psnr": 4.2, "gain_ssim": 0.12, "mode": "supervised"},
                {"strategy": "conservative",  "score": 40.0, "psnr": 23.9, "ssim": 0.75, "gain_psnr": 0.0, "gain_ssim": 0.0,  "mode": "supervised"},
            ],
            "selected_strategy": "inpainting_r5",
            "score": 74.3,
            "retry_count": 5,
        },
        "metrics": {
            "original_vs_corrupted":    {"psnr": 23.9, "ssim": 0.75},
            "original_vs_reconstructed": {"psnr": 28.1, "ssim": 0.87},
            "gains": {"psnr_gain": 4.2, "ssim_gain": 0.12, "improvement_score": 6.5},
            "detection_metrics": {"iou": 0.82, "precision": 0.91, "recall": 0.88},
        },
        "analysis": {
            "detection_quality": "good",
            "repair_effectiveness": "improved",
            "score_level": "good",
            "notes": ["Gain PSNR significatif : +4.2 dB.", "Score de reconstruction bon."],
        },
    }


def test_pdf_generated(sample_report, tmp_path):
    out = tmp_path / "test_report.pdf"
    path = generate_pdf_report(sample_report, output_path=out)
    assert path.exists()
    assert path.suffix == ".pdf"
    assert path.stat().st_size > 1000  # PDF non vide


def test_pdf_default_path(sample_report, monkeypatch, tmp_path):
    import app.modules.reporting.pdf_report as pr
    monkeypatch.setattr(pr, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(pr, "ensure_directories", lambda: None)
    path = generate_pdf_report(sample_report)
    assert path.exists()
    assert "testpdf001" in path.name


def test_pdf_with_missing_images(sample_report, tmp_path):
    """Le PDF doit se générer même si les images sont absentes."""
    sample_report["corruption"]["image_path"] = "/nonexistent/image.png"
    out = tmp_path / "test_no_img.pdf"
    path = generate_pdf_report(sample_report, output_path=out)
    assert path.exists()


def test_api_pdf_endpoint_not_found():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    resp = client.get("/reports/pdf/nonexistent000")
    assert resp.status_code == 404


def test_api_pdf_endpoint_generates_from_json(tmp_path, monkeypatch):
    """Si le JSON existe, l'endpoint génère le PDF à la volée."""
    import json as _json
    from fastapi.testclient import TestClient
    from app.main import app
    import app.api.routes.report as report_route

    monkeypatch.setattr(report_route, "REPORTS_DIR", tmp_path)

    # Créer un JSON de test
    report_data = {
        "run_id": "apipdf001",
        "timestamp": "2026-05-08T12:00:00",
        "status": "completed",
        "input": {}, "corruption": {}, "reconstruction": {"best_candidate": {}, "all_candidates": []},
        "metrics": {}, "analysis": {"detection_quality": "good", "repair_effectiveness": "improved",
                                     "score_level": "good", "notes": []},
    }
    json_path = tmp_path / "report_apipdf001.json"
    json_path.write_text(_json.dumps(report_data), encoding="utf-8")

    client = TestClient(app)
    resp = client.get("/reports/pdf/apipdf001")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"