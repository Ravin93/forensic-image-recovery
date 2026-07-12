"""Tests E4 — rapport HTML."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_report():
    return {
        "run_id": "htmltest001", "timestamp": "2026-05-11T12:00:00",
        "status": "completed",
        "input": {"source_image": None, "corruption_type": "scratch_lines",
                  "execution_mode": "assisted"},
        "corruption": {"type": "scratch_lines",
                       "parameters": {"count": 4, "thickness": 2},
                       "image_path": None, "mask_path": None},
        "reconstruction": {
            "best_candidate": {"strategy": "inpainting_r5", "path": None,
                               "score": 74.3, "psnr": 28.1, "ssim": 0.87,
                               "gain_psnr": 4.2, "gain_ssim": 0.12, "mode": "supervised"},
            "all_candidates": [
                {"strategy": "inpainting_r5", "score": 74.3, "psnr": 28.1,
                 "ssim": 0.87, "mode": "supervised"},
                {"strategy": "conservative",  "score": 40.0, "psnr": 23.9,
                 "ssim": 0.75, "mode": "supervised"},
            ],
            "selected_strategy": "inpainting_r5",
            "score": 74.3, "retry_count": 5,
        },
        "metrics": {
            "original_vs_corrupted":    {"psnr": 23.9, "ssim": 0.75},
            "original_vs_reconstructed": {"psnr": 28.1, "ssim": 0.87},
            "gains": {"psnr_gain": 4.2, "ssim_gain": 0.12},
            "detection_metrics": {"iou": 0.82, "precision": 0.91},
        },
        "analysis": {
            "detection_quality": "good", "repair_effectiveness": "improved",
            "score_level": "good",
            "notes": ["Gain PSNR significatif.", "Pipeline correct."],
        },
    }


class TestHtmlReport:
    def test_generates_html_file(self, sample_report, tmp_path):
        from app.modules.reporting.html_report import generate_html_report
        out = tmp_path / "test.html"
        path = generate_html_report(sample_report, output_path=out)
        assert path.exists()
        assert path.suffix == ".html"

    def test_html_is_valid_structure(self, sample_report, tmp_path):
        from app.modules.reporting.html_report import generate_html_report
        out = tmp_path / "test.html"
        path = generate_html_report(sample_report, output_path=out)
        content = path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "htmltest001" in content
        assert "inpainting_r5" in content
        assert "74.1" in content or "74.3" in content

    def test_html_contains_score(self, sample_report, tmp_path):
        from app.modules.reporting.html_report import generate_html_report
        path = generate_html_report(sample_report, output_path=tmp_path/"t.html")
        content = path.read_text()
        assert "FORENSIC IMAGE RECOVERY" in content
        assert "scratch_lines" in content

    def test_html_contains_candidates_table(self, sample_report, tmp_path):
        from app.modules.reporting.html_report import generate_html_report
        path = generate_html_report(sample_report, output_path=tmp_path/"t.html")
        content = path.read_text()
        assert "conservative" in content
        assert "Stratégie" in content or "strat" in content.lower()

    def test_html_contains_analysis(self, sample_report, tmp_path):
        from app.modules.reporting.html_report import generate_html_report
        path = generate_html_report(sample_report, output_path=tmp_path/"t.html")
        content = path.read_text()
        assert "GOOD" in content or "good" in content.lower()
        assert "Gain PSNR" in content or "gain" in content.lower()

    def test_html_self_contained_no_external_js(self, sample_report, tmp_path):
        """Le HTML ne doit pas dependre de scripts externes (self-contained)."""
        from app.modules.reporting.html_report import generate_html_report
        path = generate_html_report(sample_report, output_path=tmp_path/"t.html")
        content = path.read_text()
        # Pas de CDN JS (les fonts Google sont ok en mode display)
        assert "cdn.jsdelivr.net" not in content
        assert "unpkg.com" not in content

    def test_default_output_path(self, sample_report, tmp_path, monkeypatch):
        import app.modules.reporting.html_report as hr
        monkeypatch.setattr(hr, "REPORTS_DIR", tmp_path)
        monkeypatch.setattr(hr, "ensure_directories", lambda: None)
        path = hr.generate_html_report(sample_report)
        assert "htmltest001" in path.name
        assert path.suffix == ".html"

    def test_api_endpoint_generates_html(self, tmp_path, monkeypatch):
        """L'endpoint /reports/html/{id} genere le HTML depuis le JSON."""
        import json as _json
        from fastapi.testclient import TestClient
        from app.main import app
        import app.api.routes.report as rr
        monkeypatch.setattr(rr, "REPORTS_DIR", tmp_path)
        report_data = {"run_id": "apie4test", "timestamp": "2026-05-11",
                       "status": "completed", "input": {}, "corruption": {},
                       "reconstruction": {"best_candidate": {}, "all_candidates": []},
                       "metrics": {}, "analysis": {"detection_quality": "good",
                       "repair_effectiveness": "improved", "score_level": "good", "notes": []}}
        json_path = tmp_path / "report_apie4test.json"
        json_path.write_text(_json.dumps(report_data), encoding="utf-8")
        client = TestClient(app)
        resp = client.get("/reports/html/apie4test")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "<!DOCTYPE html>" in resp.text

    def test_api_404_for_unknown(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.get("/reports/html/nonexistent999")
        assert resp.status_code == 404