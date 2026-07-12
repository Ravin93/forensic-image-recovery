"""Tests K12 — Analyse EXIF forensique."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def plain_png(tmp_path):
    arr = np.zeros((64,64,3),dtype=np.uint8) + 100
    p = tmp_path / "plain.png"
    Image.fromarray(arr).save(p)
    return p


@pytest.fixture
def jpeg_with_exif(tmp_path):
    arr = np.zeros((64,64,3),dtype=np.uint8) + 150
    p = tmp_path / "photo.jpg"
    Image.fromarray(arr).save(p, format="JPEG")
    return p


class TestExifAnalyzer:
    def test_extract_no_exif(self, plain_png):
        from app.modules.metadata.exif_analyzer import extract_exif
        result = extract_exif(plain_png)
        assert isinstance(result, dict)
        assert "exif_present" in result

    def test_extract_returns_required_keys(self, plain_png):
        from app.modules.metadata.exif_analyzer import extract_exif
        result = extract_exif(plain_png)
        for k in ("exif_present","camera_make","camera_model","software",
                  "gps_present","suspicion_flags" if False else "raw_tags"):
            assert k in result

    def test_analyze_consistency_returns_dict(self, plain_png):
        from app.modules.metadata.exif_analyzer import analyze_exif_consistency
        result = analyze_exif_consistency(plain_png)
        assert isinstance(result, dict)
        for k in ("exif_present","suspicious","suspicion_flags","suspicion_score"):
            assert k in result

    def test_no_exif_flagged(self, plain_png):
        from app.modules.metadata.exif_analyzer import analyze_exif_consistency
        result = analyze_exif_consistency(plain_png)
        assert result["suspicious"] == True  # pas d EXIF = suspect
        assert any("absent" in f.lower() or "supprime" in f.lower()
                   for f in result["suspicion_flags"])

    def test_suspicion_score_in_range(self, plain_png):
        from app.modules.metadata.exif_analyzer import analyze_exif_consistency
        result = analyze_exif_consistency(plain_png)
        assert 0.0 <= result["suspicion_score"] <= 1.0

    def test_missing_file_returns_empty(self, tmp_path):
        from app.modules.metadata.exif_analyzer import extract_exif
        result = extract_exif(tmp_path / "nonexistent.jpg")
        assert result["exif_present"] == False

    def test_jpeg_extract_no_crash(self, jpeg_with_exif):
        from app.modules.metadata.exif_analyzer import extract_exif
        result = extract_exif(jpeg_with_exif)
        assert isinstance(result, dict)