"""Tests C4 — comparaison masque auto vs masque utilisateur."""
from __future__ import annotations
import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def _gradient_png(size=128) -> bytes:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        arr[i, :, 0] = i * 2
        arr[:, i, 1] = i * 2
        arr[i, :, 2] = 255 - i * 2
    # Ajouter une zone noire (simule une corruption detectable)
    arr[40:80, 40:80] = 0
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _mask_png(size=128, zone=(40, 80, 40, 80)) -> bytes:
    """Masque PNG avec une zone blanche."""
    arr = np.zeros((size, size), dtype=np.uint8)
    y1, y2, x1, x2 = zone
    arr[y1:y2, x1:x2] = 255
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _post_compare(client, image_bytes=None, mask_bytes=None,
                  detection_mode="basic", max_attempts=2):
    return client.post(
        "/reconstruction/compare-masks",
        files={
            "image":     ("img.png", io.BytesIO(image_bytes or _gradient_png()), "image/png"),
            "user_mask": ("mask.png", io.BytesIO(mask_bytes or _mask_png()),    "image/png"),
        },
        data={"detection_mode": detection_mode, "max_attempts": str(max_attempts)},
    )


class TestCompareMasks:
    def test_returns_200(self, client):
        resp = _post_compare(client)
        assert resp.status_code == 200, resp.text

    def test_response_structure(self, client):
        resp = _post_compare(client)
        data = resp.json()
        for key in ("user_mask", "auto_mask", "winner", "comparison_note",
                    "score", "reconstructed_image", "status"):
            assert key in data, f"Cle manquante : {key}"

    def test_winner_is_valid(self, client):
        resp = _post_compare(client)
        assert resp.json()["winner"] in ("user_mask", "auto_mask")

    def test_user_mask_section_structure(self, client):
        resp = _post_compare(client)
        um = resp.json()["user_mask"]
        for key in ("mask_path", "score", "status", "selected_repair_strategy"):
            assert key in um, f"Cle manquante dans user_mask : {key}"

    def test_auto_mask_section_structure(self, client):
        resp = _post_compare(client)
        am = resp.json()["auto_mask"]
        for key in ("mask_path", "score", "status", "detection_confidence"):
            assert key in am, f"Cle manquante dans auto_mask : {key}"

    def test_score_is_best_of_two(self, client):
        resp = _post_compare(client)
        data = resp.json()
        score_user = data["user_mask"]["score"] or 0
        score_auto = data["auto_mask"]["score"] or 0
        assert float(data["score"]) >= min(score_user, score_auto) - 0.01

    def test_comparison_note_present(self, client):
        resp = _post_compare(client)
        note = resp.json()["comparison_note"]
        assert isinstance(note, str) and len(note) > 5

    def test_advanced_detection_mode(self, client):
        resp = _post_compare(client, detection_mode="advanced", max_attempts=2)
        assert resp.status_code == 200
        assert resp.json()["auto_mask"]["detection_mode"] == "advanced"

    def test_empty_user_mask_rejected(self, client):
        empty_mask = np.zeros((128, 128), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(empty_mask).save(buf, format="PNG")
        resp = _post_compare(client, mask_bytes=buf.getvalue())
        assert resp.status_code == 422

    def test_invalid_mask_format_rejected(self, client):
        resp = client.post(
            "/reconstruction/compare-masks",
            files={
                "image":     ("img.png", io.BytesIO(_gradient_png()), "image/png"),
                "user_mask": ("mask.jpg", io.BytesIO(b"fake"), "image/jpeg"),
            },
            data={"detection_mode": "basic", "max_attempts": "2"},
        )
        assert resp.status_code == 422

    def test_both_reconstructions_attempted(self, client):
        """Les deux masques doivent avoir tenté une reconstruction."""
        resp = _post_compare(client, max_attempts=2)
        data = resp.json()
        assert data["user_mask"]["status"] in ("completed", "failed")
        assert data["auto_mask"]["status"] in ("completed", "failed")

    def test_best_result_matches_winner(self, client):
        """reconstructed_image doit correspondre au winner."""
        resp = _post_compare(client, max_attempts=2)
        data = resp.json()
        winner = data["winner"]
        best_recon = data["reconstructed_image"]
        winner_recon = data[winner]["reconstructed_image"]
        assert best_recon == winner_recon