"""Tests K19 — Rapport forensique legal."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_report(tmp_path):
    arr = np.zeros((32,32,3),dtype=np.uint8)+100
    p = tmp_path/"orig.png"; Image.fromarray(arr).save(p)
    return {
        "run_id": "legaltest01",
        "timestamp": "2026-05-12T10:00:00",
        "status": "completed",
        "input": {"source_image": str(p), "execution_mode": "assisted"},
        "corruption": {"type": "scratch_lines", "image_path": str(p),
                       "mask_path": None, "parameters": {"count": 4}},
        "reconstruction": {
            "best_candidate": {"strategy": "inpainting_r3", "path": str(p),
                               "score": 74.3, "psnr": 28.1, "ssim": 0.87,
                               "mode": "supervised"},
            "all_candidates": [
                {"strategy": "inpainting_r3", "score": 74.3, "psnr": 28.1,
                 "ssim": 0.87, "mode": "supervised"},
            ],
            "selected_strategy": "inpainting_r3",
        },
        "metrics": {
            "original_vs_corrupted": {"psnr": 23.9, "ssim": 0.75},
            "original_vs_reconstructed": {"psnr": 28.1, "ssim": 0.87},
            "gains": {"psnr_gain": 4.2, "ssim_gain": 0.12},
        },
        "analysis": {"detection_quality": "good",
                     "repair_effectiveness": "improved",
                     "score_level": "good", "notes": []},
    }


class TestLegalReport:
    def test_generates_json(self, sample_report, tmp_path):
        from app.modules.reporting.legal_report import generate_legal_report
        out = tmp_path/"legal.json"
        path = generate_legal_report(sample_report, output_path=out)
        assert path.exists()

    def test_has_disclaimer(self, sample_report, tmp_path):
        from app.modules.reporting.legal_report import generate_legal_report
        path = generate_legal_report(sample_report, output_path=tmp_path/"l.json")
        data = json.loads(path.read_text())
        assert "disclaimer" in data
        assert len(data["disclaimer"]) > 50

    def test_has_chain_of_custody(self, sample_report, tmp_path):
        from app.modules.reporting.legal_report import generate_legal_report
        path = generate_legal_report(sample_report, output_path=tmp_path/"l.json")
        data = json.loads(path.read_text())
        assert "chain_of_custody" in data
        assert isinstance(data["chain_of_custody"], list)
        assert len(data["chain_of_custody"]) >= 1

    def test_has_sha256(self, sample_report, tmp_path):
        from app.modules.reporting.legal_report import generate_legal_report
        path = generate_legal_report(sample_report, output_path=tmp_path/"l.json")
        data = json.loads(path.read_text())
        assert "sha256_original" in data
        assert len(data["sha256_original"]) == 64  # SHA-256 valide

    def test_has_table_of_methods(self, sample_report, tmp_path):
        from app.modules.reporting.legal_report import generate_legal_report
        path = generate_legal_report(sample_report, output_path=tmp_path/"l.json")
        data = json.loads(path.read_text())
        assert "table_of_methods" in data
        assert len(data["table_of_methods"]) >= 1

    def test_has_legal_note(self, sample_report, tmp_path):
        from app.modules.reporting.legal_report import generate_legal_report
        path = generate_legal_report(sample_report, output_path=tmp_path/"l.json")
        data = json.loads(path.read_text())
        assert "legal_note" in data
        assert "hypothese" in data["legal_note"].lower() or "probant" in data["legal_note"].lower()

    def test_forensic_mode_safe(self, sample_report, tmp_path):
        from app.modules.reporting.legal_report import generate_legal_report
        path = generate_legal_report(sample_report, output_path=tmp_path/"l.json")
        data = json.loads(path.read_text())
        assert data["forensic_mode"] == "forensic_safe"

    def test_chain_of_custody_has_sha256(self, sample_report, tmp_path):
        from app.modules.reporting.legal_report import generate_legal_report
        path = generate_legal_report(sample_report, output_path=tmp_path/"l.json")
        data = json.loads(path.read_text())
        for step in data["chain_of_custody"]:
            assert "sha256" in step

    def test_certifications_present(self, sample_report, tmp_path):
        from app.modules.reporting.legal_report import generate_legal_report
        path = generate_legal_report(sample_report, output_path=tmp_path/"l.json")
        data = json.loads(path.read_text())
        assert "certifications" in data
        assert data["certifications"]["probatoire"] == False

    def test_html_section_generated(self, sample_report, tmp_path):
        from app.modules.reporting.legal_report import (
            build_chain_of_custody, get_legal_html_section
        )
        inp  = sample_report["input"]
        corr = sample_report["corruption"]
        best = sample_report["reconstruction"]["best_candidate"]
        chain = build_chain_of_custody(
            inp.get("source_image"), corr.get("image_path"),
            best.get("path"), "test001",
        )
        html = get_legal_html_section(chain, "forensic_safe", "abc123")
        assert "FORENSIC-SAFE" in html or "hypothese" in html.lower()
        assert "SHA-256" in html or "sha" in html.lower()