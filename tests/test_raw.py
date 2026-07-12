"""Tests K10 — Chargement RAW."""
from __future__ import annotations
from pathlib import Path
import pytest


class TestRawLoader:
    def test_is_raw_file_true(self, tmp_path):
        from app.modules.raw.raw_loader import is_raw_file
        assert is_raw_file(tmp_path/"photo.dng") == True
        assert is_raw_file(tmp_path/"photo.CR2") == True
        assert is_raw_file(tmp_path/"photo.nef") == True

    def test_is_raw_file_false(self, tmp_path):
        from app.modules.raw.raw_loader import is_raw_file
        assert is_raw_file(tmp_path/"photo.jpg") == False
        assert is_raw_file(tmp_path/"photo.png") == False

    def test_rawpy_availability_check(self):
        from app.modules.raw.raw_loader import is_rawpy_available
        result = is_rawpy_available()
        assert isinstance(result, bool)

    def test_missing_file_error(self, tmp_path):
        from app.modules.raw.raw_loader import load_raw_as_rgb
        r = load_raw_as_rgb(tmp_path/"nonexistent.dng")
        assert "error" in r
        assert r["image"] is None

    def test_non_raw_extension_error(self, tmp_path):
        from app.modules.raw.raw_loader import load_raw_as_rgb
        p = tmp_path/"img.jpg"; p.write_bytes(b"fake")
        r = load_raw_as_rgb(p)
        assert "error" in r

    def test_metadata_missing_file(self, tmp_path):
        from app.modules.raw.raw_loader import extract_raw_metadata
        r = extract_raw_metadata(tmp_path/"nonexistent.dng")
        assert "error" in r
        assert r["available"] == False

    def test_load_returns_dict(self, tmp_path):
        from app.modules.raw.raw_loader import load_raw_as_rgb
        p = tmp_path/"test.dng"; p.write_bytes(b"fake raw data")
        r = load_raw_as_rgb(p)
        assert isinstance(r, dict)
        assert "available" in r