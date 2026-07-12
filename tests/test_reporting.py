"""Tests Ticket 8 — rapport JSON enrichi."""
from __future__ import annotations
from pathlib import Path
import pytest
from app.modules.reporting.json_report import build_report, save_json_report, build_analysis_block


@pytest.fixture
def mock_results():
    corruption = {
        "corruption_type": "scratch_lines", "parameters": {"count": 3},
        "image_path": "/data/corrupted/img.png", "mask_path": "/data/masks/img_mask.png",
        "randomize": False, "imperfect_mask": False,
    }
    reconstruction = {
        "path": "/data/reconstructed/img_best.png",
        "selected_repair_strategy": "inpainting_r5",
        "recommended_strategy": "inpainting",
        "score": 74.3, "retry_count": 5,
        "detection_confidence": 1.0, "method": "opencv_inpaint",
        "corruption_type": "scratch_lines",
        "candidates": [
            {"strategy": "inpainting_r5", "path": "/data/reconstructed/img_r5.png",
             "score": 74.3, "psnr": 28.1, "ssim": 0.87, "gain_psnr": 4.2, "gain_ssim": 0.12, "mode": "supervised"},
            {"strategy": "conservative",  "path": "/data/corrupted/img.png",
             "score": 40.0, "psnr": 23.9, "ssim": 0.75, "gain_psnr": 0.0, "gain_ssim": 0.0,  "mode": "supervised"},
        ],
    }
    comparison = {
        "original_vs_corrupted":    {"psnr": 23.9, "ssim": 0.75},
        "original_vs_reconstructed": {"psnr": 28.1, "ssim": 0.87},
        "reconstructed": {"psnr": 28.1, "ssim": 0.87},
        "corrupted":     {"psnr": 23.9, "ssim": 0.75},
        "gains": {"psnr_gain": 4.2, "ssim_gain": 0.12, "improvement_score": 6.54},
        "improvement": True,
        "supervised_score": {"mode": "supervised", "score": 74.3, "psnr": 28.1, "ssim": 0.87,
                             "gain_psnr": 4.2, "gain_ssim": 0.12},
    }
    extra = {
        "execution_mode": "assisted", "detection_mode": "basic",
        "detected_mask_path": "/data/masks/img_mask.png",
        "detection_metrics": {"iou": 0.82, "precision": 0.91, "recall": 0.88},
        "recoverability_status": "good",
        "detection_confidence": 1.0,
    }
    return "source.png", corruption, reconstruction, comparison, extra


class TestBuildReport:
    def test_top_level_sections(self, mock_results):
        src, corr, recon, comp, extra = mock_results
        report = build_report(src, corr, recon, comp, extra=extra)
        for section in ("run_id", "timestamp", "status", "input",
                        "corruption", "reconstruction", "metrics", "analysis"):
            assert section in report, f"Section manquante : {section}"

    def test_input_section(self, mock_results):
        src, corr, recon, comp, extra = mock_results
        report = build_report(src, corr, recon, comp, extra=extra)
        inp = report["input"]
        assert inp["source_image"] == "source.png"
        assert inp["corruption_type"] == "scratch_lines"
        assert inp["execution_mode"] == "assisted"

    def test_corruption_section(self, mock_results):
        src, corr, recon, comp, extra = mock_results
        report = build_report(src, corr, recon, comp, extra=extra)
        c = report["corruption"]
        assert c["type"] == "scratch_lines"
        assert "parameters" in c
        assert "image_path" in c
        assert "mask_path" in c

    def test_reconstruction_section_best_candidate(self, mock_results):
        src, corr, recon, comp, extra = mock_results
        report = build_report(src, corr, recon, comp, extra=extra)
        r = report["reconstruction"]
        assert "best_candidate" in r
        assert "all_candidates" in r
        assert r["best_candidate"]["strategy"] == "inpainting_r5"
        assert r["best_candidate"]["score"] == 74.3
        assert len(r["all_candidates"]) == 2

    def test_reconstruction_all_candidates_structure(self, mock_results):
        src, corr, recon, comp, extra = mock_results
        report = build_report(src, corr, recon, comp, extra=extra)
        for c in report["reconstruction"]["all_candidates"]:
            for key in ("strategy", "path", "score", "psnr", "ssim"):
                assert key in c, f"Clé manquante dans candidat : {key}"

    def test_metrics_section(self, mock_results):
        src, corr, recon, comp, extra = mock_results
        report = build_report(src, corr, recon, comp, extra=extra)
        m = report["metrics"]
        assert "original_vs_corrupted" in m
        assert "original_vs_reconstructed" in m
        assert "gains" in m
        assert "detection_metrics" in m
        assert "supervised_score" in m

    def test_analysis_section(self, mock_results):
        src, corr, recon, comp, extra = mock_results
        report = build_report(src, corr, recon, comp, extra=extra)
        a = report["analysis"]
        for key in ("detection_quality", "repair_effectiveness", "score_level", "notes"):
            assert key in a, f"Clé manquante dans analysis : {key}"

    def test_analysis_good_iou(self, mock_results):
        src, corr, recon, comp, extra = mock_results
        report = build_report(src, corr, recon, comp, extra=extra)
        assert report["analysis"]["detection_quality"] == "good"

    def test_analysis_improved(self, mock_results):
        src, corr, recon, comp, extra = mock_results
        report = build_report(src, corr, recon, comp, extra=extra)
        assert report["analysis"]["repair_effectiveness"] == "improved"

    def test_analysis_score_level(self, mock_results):
        src, corr, recon, comp, extra = mock_results
        report = build_report(src, corr, recon, comp, extra=extra)
        assert report["analysis"]["score_level"] in ("excellent", "good", "medium", "poor")

    def test_notes_not_empty(self, mock_results):
        src, corr, recon, comp, extra = mock_results
        report = build_report(src, corr, recon, comp, extra=extra)
        assert len(report["analysis"]["notes"]) >= 1

    def test_json_serializable(self, mock_results):
        """Le rapport doit être sérialisable sans erreur."""
        import json
        src, corr, recon, comp, extra = mock_results
        report = build_report(src, corr, recon, comp, extra=extra)
        dumped = json.dumps(report)
        assert len(dumped) > 100

    def test_status_field(self, mock_results):
        src, corr, recon, comp, extra = mock_results
        report = build_report(src, corr, recon, comp, status="completed", extra=extra)
        assert report["status"] == "completed"


class TestSaveJsonReport:
    def test_file_created(self, mock_results, tmp_path, monkeypatch):
        import app.modules.reporting.json_report as jr
        monkeypatch.setattr(jr, "REPORTS_DIR", tmp_path)
        monkeypatch.setattr(jr, "ensure_directories", lambda: None)
        src, corr, recon, comp, extra = mock_results
        report = build_report(src, corr, recon, comp, extra=extra)
        path = jr.save_json_report(report)
        assert Path(path).exists()

    def test_file_content_valid_json(self, mock_results, tmp_path, monkeypatch):
        import json, app.modules.reporting.json_report as jr
        monkeypatch.setattr(jr, "REPORTS_DIR", tmp_path)
        monkeypatch.setattr(jr, "ensure_directories", lambda: None)
        src, corr, recon, comp, extra = mock_results
        report = build_report(src, corr, recon, comp, extra=extra)
        path = jr.save_json_report(report)
        with open(path) as f:
            data = json.load(f)
        assert data["status"] == "completed"
        assert "reconstruction" in data