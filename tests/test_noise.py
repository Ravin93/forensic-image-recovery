"""Tests K18 — Bruit residuel et PRNU."""
from __future__ import annotations
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def test_image(tmp_path):
    arr = np.zeros((64,64,3),dtype=np.uint8)
    for i in range(64):
        arr[i,:,0]=i*4; arr[:,i,1]=i*4; arr[i,:,2]=200-i*3
    p = tmp_path/"img.png"; Image.fromarray(arr).save(p); return p


class TestResidualNoise:
    def test_compute_consistency_returns_dict(self, test_image):
        from app.modules.noise.residual_noise import compute_noise_consistency
        r = compute_noise_consistency(test_image)
        assert isinstance(r, dict)

    def test_required_keys(self, test_image):
        from app.modules.noise.residual_noise import compute_noise_consistency
        r = compute_noise_consistency(test_image)
        for k in ("noise_consistency","suspicious_regions","mean_variance","std_variance"):
            assert k in r, f"Cle manquante : {k}"

    def test_consistency_in_range(self, test_image):
        from app.modules.noise.residual_noise import compute_noise_consistency
        r = compute_noise_consistency(test_image)
        assert 0.0 <= r["noise_consistency"] <= 1.0

    def test_heatmap_created(self, test_image):
        from app.modules.noise.residual_noise import compute_noise_consistency
        r = compute_noise_consistency(test_image, block_size=16)
        # heatmap peut etre None si image trop petite ou uniforme
        if r["heatmap_path"] is not None:
            from pathlib import Path
            assert Path(r["heatmap_path"]).exists()

    def test_extract_noise_shape(self, test_image):
        import cv2
        from app.modules.noise.residual_noise import extract_residual_noise
        img = cv2.imread(str(test_image))
        noise = extract_residual_noise(img)
        assert noise.shape == img.shape[:2]

    def test_prnu_compare_returns_dict(self, test_image):
        from app.modules.noise.residual_noise import compare_prnu
        r = compare_prnu(test_image, test_image)
        assert isinstance(r, dict)
        for k in ("correlation","same_device","confidence"):
            assert k in r

    def test_prnu_same_image_high_correlation(self, test_image):
        from app.modules.noise.residual_noise import compare_prnu
        r = compare_prnu(test_image, test_image)
        # Meme image = correlation elevee
        assert r["correlation"] > 0.5

    def test_missing_file_raises(self, tmp_path):
        from app.modules.noise.residual_noise import compute_noise_consistency
        with pytest.raises(FileNotFoundError):
            compute_noise_consistency(tmp_path/"nonexistent.png")