"""Tests C2+C3 — endpoint /reconstruction/repair-with-mask."""
from __future__ import annotations
import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def _gradient_png() -> bytes:
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(128):
        arr[i, :, 0] = i * 2
        arr[:, i, 1] = i * 2
        arr[i, :, 2] = 255 - i * 2
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()

def _mask_png(white_zone: bool = True) -> bytes:
    arr = np.zeros((128, 128), dtype=np.uint8)
    if white_zone:
        arr[30:70, 30:70] = 255  # zone blanche centrale
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()

def _post(image_bytes=None, mask_bytes=None, img_name="img.png", mask_name="mask.png", max_attempts=2):
    return client.post(
        "/reconstruction/repair-with-mask",
        files={
            "image": (img_name, io.BytesIO(image_bytes or _gradient_png()), "image/png"),
            "mask":  (mask_name, io.BytesIO(mask_bytes  or _mask_png()),   "image/png"),
        },
        data={"max_attempts": str(max_attempts)},
    )

def test_repair_with_mask_200():
    resp = _post()
    assert resp.status_code == 200, resp.text

def test_repair_with_mask_response_structure():
    resp = _post()
    data = resp.json()
    for key in ("reconstructed_image","mask_path","score","selected_repair_strategy","candidates","status"):
        assert key in data, f"Clé manquante : {key}"

def test_repair_score_in_range():
    resp = _post()
    assert 0.0 <= float(resp.json()["score"]) <= 100.0

def test_repair_candidates_not_empty():
    resp = _post()
    assert len(resp.json()["candidates"]) >= 1

def test_repair_empty_mask_returns_422():
    resp = _post(mask_bytes=_mask_png(white_zone=False))
    assert resp.status_code == 422
    assert "vide" in resp.json()["detail"].lower() or "noir" in resp.json()["detail"].lower()

def test_repair_invalid_image_format():
    resp = _post(image_bytes=b"not an image", img_name="file.txt")
    assert resp.status_code == 422

def test_repair_non_png_mask():
    resp = client.post(
        "/reconstruction/repair-with-mask",
        files={
            "image": ("img.png", io.BytesIO(_gradient_png()), "image/png"),
            "mask":  ("mask.jpg", io.BytesIO(_mask_png()),   "image/jpeg"),
        },
        data={"max_attempts": "2"},
    )
    assert resp.status_code == 422

def test_repair_jpeg_image():
    arr = np.zeros((128, 128, 3), dtype=np.uint8) + 100
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    resp = _post(image_bytes=buf.getvalue(), img_name="img.jpg", max_attempts=2)
    assert resp.status_code == 200

def test_repair_execution_mode_is_user_mask():
    resp = _post()
    assert resp.json().get("execution_mode") == "user_mask"