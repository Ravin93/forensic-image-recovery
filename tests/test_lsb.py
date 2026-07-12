"""Tests K13 — Detection steganographie LSB."""
from __future__ import annotations
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def normal_image(tmp_path):
    arr = np.zeros((64,64,3),dtype=np.uint8)
    for i in range(64):
        arr[i,:,0]=i*4; arr[:,i,1]=i*4; arr[i,:,2]=200-i*3
    p = tmp_path/"normal.png"; Image.fromarray(arr).save(p); return p


@pytest.fixture
def lsb_modified_image(tmp_path):
    """Image avec LSB artificiellement modifies (steganographie simulee)."""
    arr = np.zeros((64,64,3),dtype=np.uint8)
    for i in range(64):
        arr[i,:,0]=i*4; arr[:,i,1]=i*4
    rng = np.random.default_rng(42)
    # Remplacer tous les LSB par des bits aleatoires
    arr[:,:,0] = (arr[:,:,0] & 0xFE) | rng.integers(0,2,(64,64),dtype=np.uint8)
    arr[:,:,1] = (arr[:,:,1] & 0xFE) | rng.integers(0,2,(64,64),dtype=np.uint8)
    arr[:,:,2] = (arr[:,:,2] & 0xFE) | rng.integers(0,2,(64,64),dtype=np.uint8)
    p = tmp_path/"lsb_mod.png"; Image.fromarray(arr).save(p); return p


class TestLSBAnalyzer:
    def test_analyze_returns_dict(self, normal_image):
        from app.modules.steganalysis.lsb_analyzer import analyze_lsb_distribution
        r = analyze_lsb_distribution(normal_image)
        assert isinstance(r, dict)

    def test_required_keys_distribution(self, normal_image):
        from app.modules.steganalysis.lsb_analyzer import analyze_lsb_distribution
        r = analyze_lsb_distribution(normal_image)
        for k in ("channels","mean_entropy","n_pixels"):
            assert k in r, f"Cle manquante : {k}"

    def test_required_keys_anomaly(self, normal_image):
        from app.modules.steganalysis.lsb_analyzer import detect_lsb_anomaly
        r = detect_lsb_anomaly(normal_image)
        for k in ("lsb_entropy","anomaly_score","suspicious","suspicion_level","flags"):
            assert k in r, f"Cle manquante : {k}"

    def test_anomaly_score_in_range(self, normal_image):
        from app.modules.steganalysis.lsb_analyzer import detect_lsb_anomaly
        r = detect_lsb_anomaly(normal_image)
        assert 0.0 <= r["anomaly_score"] <= 1.0

    def test_modified_higher_score(self, normal_image, lsb_modified_image):
        from app.modules.steganalysis.lsb_analyzer import detect_lsb_anomaly
        r_normal = detect_lsb_anomaly(normal_image)
        r_mod    = detect_lsb_anomaly(lsb_modified_image)
        # Image modifiee doit avoir score >= image normale
        assert r_mod["anomaly_score"] >= r_normal["anomaly_score"] - 0.1

    def test_suspicion_level_valid(self, normal_image):
        from app.modules.steganalysis.lsb_analyzer import detect_lsb_anomaly
        r = detect_lsb_anomaly(normal_image)
        assert r["suspicion_level"] in ("faible","moyenne","forte")

    def test_channel_stats_present(self, normal_image):
        from app.modules.steganalysis.lsb_analyzer import analyze_lsb_distribution
        r = analyze_lsb_distribution(normal_image)
        for ch in ("blue","green","red"):
            assert ch in r["channels"]
            assert "entropy" in r["channels"][ch]

    def test_missing_file_raises(self, tmp_path):
        from app.modules.steganalysis.lsb_analyzer import detect_lsb_anomaly
        with pytest.raises(FileNotFoundError):
            detect_lsb_anomaly(tmp_path/"nonexistent.png")