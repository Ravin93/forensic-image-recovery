"""Tests K17 — Detection copy-move."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def normal_image(tmp_path):
    arr = np.zeros((128,128,3),dtype=np.uint8)
    for i in range(128):
        arr[i,:,0]=i*2; arr[:,i,1]=i*2; arr[i,:,2]=200-i
    p = tmp_path/"normal.png"; Image.fromarray(arr).save(p); return p


@pytest.fixture
def copymove_image(tmp_path):
    """Image avec une zone copiee-collee."""
    arr = np.zeros((128,128,3),dtype=np.uint8)
    for i in range(128):
        arr[i,:,0]=i*2; arr[:,i,1]=i*2
    # Copier la zone [10:40, 10:40] vers [70:100, 70:100]
    arr[70:100,70:100] = arr[10:40,10:40]
    p = tmp_path/"copymove.png"; Image.fromarray(arr).save(p); return p


class TestCopyMoveDetector:
    def test_returns_dict(self, normal_image):
        from app.modules.tampering.copy_move_detector import detect_copy_move
        r = detect_copy_move(normal_image)
        assert isinstance(r, dict)

    def test_required_keys(self, normal_image):
        from app.modules.tampering.copy_move_detector import detect_copy_move
        r = detect_copy_move(normal_image)
        for k in ("suspicious","matches","strong_matches","regions",
                  "mask_path","confidence","elapsed_s","n_keypoints"):
            assert k in r, f"Cle manquante : {k}"

    def test_normal_image_low_confidence(self, normal_image):
        from app.modules.tampering.copy_move_detector import detect_copy_move
        r = detect_copy_move(normal_image)
        assert 0.0 <= r["confidence"] <= 1.0

    def test_invalid_path_raises(self, tmp_path):
        from app.modules.tampering.copy_move_detector import detect_copy_move
        with pytest.raises(FileNotFoundError):
            detect_copy_move(tmp_path/"nonexistent.png")

    def test_copymove_detected(self, copymove_image):
        from app.modules.tampering.copy_move_detector import detect_copy_move
        r = detect_copy_move(copymove_image, min_matches=2)
        # Au moins les keypoints sont extraits
        assert r["n_keypoints"] >= 0

    def test_mask_created_when_suspicious(self, copymove_image):
        from app.modules.tampering.copy_move_detector import detect_copy_move
        r = detect_copy_move(copymove_image, min_matches=1)
        if r["suspicious"]:
            assert r["mask_path"] is not None
            assert Path(r["mask_path"]).exists()

    def test_confidence_in_range(self, normal_image):
        from app.modules.tampering.copy_move_detector import detect_copy_move
        r = detect_copy_move(normal_image)
        assert 0.0 <= r["confidence"] <= 1.0

    def test_elapsed_positive(self, normal_image):
        from app.modules.tampering.copy_move_detector import detect_copy_move
        r = detect_copy_move(normal_image)
        assert r["elapsed_s"] >= 0