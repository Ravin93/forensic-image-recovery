"""Tests K11 — OCR sur zones reconstruites."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont


@pytest.fixture
def blank_image(tmp_path):
    arr = np.zeros((64,64,3),dtype=np.uint8)+200
    p = tmp_path/"blank.png"; Image.fromarray(arr).save(p); return p


@pytest.fixture
def text_image(tmp_path):
    img = Image.new("RGB",(200,60),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((10,10),"TEST 123",fill=(0,0,0))
    p = tmp_path/"text.png"; img.save(p); return p


class TestOCR:
    def test_run_ocr_returns_dict(self, blank_image):
        from app.modules.ocr.text_detector import run_ocr
        r = run_ocr(blank_image)
        assert isinstance(r, dict)

    def test_required_keys(self, blank_image):
        from app.modules.ocr.text_detector import run_ocr
        r = run_ocr(blank_image)
        for k in ("text","confidence","backend","available","char_count","word_count"):
            assert k in r, f"Cle manquante : {k}"

    def test_no_crash_on_blank_image(self, blank_image):
        from app.modules.ocr.text_detector import run_ocr
        r = run_ocr(blank_image)
        assert r["char_count"] >= 0

    def test_missing_file_no_crash(self, tmp_path):
        from app.modules.ocr.text_detector import run_ocr
        r = run_ocr(tmp_path/"nonexistent.png")
        assert r["text"] == ""
        assert "error" in r

    def test_compare_before_after_returns_dict(self, blank_image):
        from app.modules.ocr.text_detector import compare_ocr_before_after
        r = compare_ocr_before_after(blank_image, blank_image)
        assert isinstance(r, dict)

    def test_compare_required_keys(self, blank_image):
        from app.modules.ocr.text_detector import compare_ocr_before_after
        r = compare_ocr_before_after(blank_image, blank_image)
        for k in ("corrupted_text","reconstructed_text","text_gain",
                  "confidence","ocr_available","text_recovered"):
            assert k in r, f"Cle manquante : {k}"

    def test_text_gain_in_range(self, blank_image):
        from app.modules.ocr.text_detector import compare_ocr_before_after
        r = compare_ocr_before_after(blank_image, blank_image)
        assert 0.0 <= r["text_gain"] <= 1.0

    def test_same_image_zero_gain(self, blank_image):
        from app.modules.ocr.text_detector import compare_ocr_before_after
        r = compare_ocr_before_after(blank_image, blank_image)
        assert r["text_gain"] == 0.0