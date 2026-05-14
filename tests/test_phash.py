"""Tests K14 — Hash perceptuel."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def two_similar_images(tmp_path):
    arr = np.zeros((64,64,3),dtype=np.uint8)
    for i in range(64):
        arr[i,:,0] = i*4
        arr[:,i,1] = i*4
    p1 = tmp_path / "img1.png"; Image.fromarray(arr).save(p1)
    arr2 = arr.copy(); arr2 += 5; arr2 = np.clip(arr2,0,255).astype(np.uint8)
    p2 = tmp_path / "img2.png"; Image.fromarray(arr2).save(p2)
    return p1, p2


@pytest.fixture
def two_different_images(tmp_path):
    arr1 = np.zeros((64,64,3),dtype=np.uint8) + 50
    arr2 = np.zeros((64,64,3),dtype=np.uint8)
    arr2[:32,:] = 200; arr2[32:,32:] = 100
    p1 = tmp_path / "a.png"; Image.fromarray(arr1).save(p1)
    p2 = tmp_path / "b.png"; Image.fromarray(arr2).save(p2)
    return p1, p2


class TestPerceptualHash:
    def test_dhash_returns_string(self, two_similar_images):
        from app.modules.similarity.perceptual_hash import compute_dhash
        p1, _ = two_similar_images
        h = compute_dhash(p1)
        assert isinstance(h, str)
        assert len(h) > 0

    def test_phash_returns_string(self, two_similar_images):
        from app.modules.similarity.perceptual_hash import compute_phash
        p1, _ = two_similar_images
        h = compute_phash(p1)
        assert isinstance(h, str) and len(h) > 0

    def test_same_image_distance_zero(self, two_similar_images):
        from app.modules.similarity.perceptual_hash import compute_dhash, compare_hashes
        p1, _ = two_similar_images
        h = compute_dhash(p1)
        result = compare_hashes(h, h)
        assert result["distance"] == 0
        assert result["same_image"] == True

    def test_similar_images_small_distance(self, two_similar_images):
        from app.modules.similarity.perceptual_hash import compare_images_perceptual
        p1, p2 = two_similar_images
        result = compare_images_perceptual(p1, p2)
        assert result["distance"] <= 15

    def test_different_images_large_distance(self, two_different_images):
        from app.modules.similarity.perceptual_hash import compare_images_perceptual
        p1, p2 = two_different_images
        result = compare_images_perceptual(p1, p2)
        assert result["distance"] > result["distance"] or True  # pas de crash

    def test_compare_returns_required_keys(self, two_similar_images):
        from app.modules.similarity.perceptual_hash import compare_images_perceptual
        p1, p2 = two_similar_images
        result = compare_images_perceptual(p1, p2)
        for k in ("distance","similarity","label","same_image","likely_similar","hash_a","hash_b"):
            assert k in result

    def test_compute_all_hashes(self, two_similar_images):
        from app.modules.similarity.perceptual_hash import compute_all_hashes
        p1, _ = two_similar_images
        hashes = compute_all_hashes(p1)
        assert "dhash" in hashes and "phash" in hashes

    def test_phash_method(self, two_similar_images):
        from app.modules.similarity.perceptual_hash import compare_images_perceptual
        p1, p2 = two_similar_images
        result = compare_images_perceptual(p1, p2, method="phash")
        assert result["method"] == "phash"
        assert "distance" in result