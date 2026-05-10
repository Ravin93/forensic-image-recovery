"""Tests Ticket 5 + Ticket 7 — API POST /pipeline/corrupt-and-repair.

Couvre :
  - upload image valide → réponse complète
  - corruption sélectionnée → champ corruption_type dans la réponse
  - reconstruction générée → reconstructed_image existe
  - score retourné → float dans [0, 100]
  - erreur propre si fichier invalide (non-image)
  - erreur propre si type de corruption non supporté
  - erreur propre si sévérité invalide
  - tous les types de corruption acceptés par l'endpoint
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gradient_png_bytes() -> bytes:
    """Image 128×128 gradient RGB encodée en PNG (bytes)."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(128):
        arr[i, :, 0] = i * 2
        arr[:, i, 1] = i * 2
        arr[i, :, 2] = 255 - i * 2
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _post_corrupt_and_repair(
    image_bytes: bytes = None,
    filename: str = "test.png",
    corruption_type: str = "scratch_lines",
    severity: str = "medium",
    max_attempts: int = 3,
    execution_mode: str = "assisted",
    seed: int = 42,
):
    if image_bytes is None:
        image_bytes = _gradient_png_bytes()
    return client.post(
        "/pipeline/corrupt-and-repair",
        files={"image": (filename, io.BytesIO(image_bytes), "image/png")},
        data={
            "corruption_type": corruption_type,
            "severity": severity,
            "max_attempts": str(max_attempts),
            "execution_mode": execution_mode,
            "seed": str(seed),
        },
    )


# ---------------------------------------------------------------------------
# Tests nominaux
# ---------------------------------------------------------------------------

def test_corrupt_and_repair_returns_200():
    resp = _post_corrupt_and_repair()
    assert resp.status_code == 200, resp.text


def test_corrupt_and_repair_response_structure():
    resp = _post_corrupt_and_repair()
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "original_image", "corrupted_image", "reconstructed_image",
        "mask_path", "score", "selected_repair_strategy",
        "retry_count", "candidates", "corruption_type",
        "execution_mode", "report_path", "status",
    ):
        assert key in data, f"Clé manquante dans la réponse : {key}"


def test_corrupt_and_repair_corruption_type_in_response():
    resp = _post_corrupt_and_repair(corruption_type="random_holes")
    assert resp.status_code == 200
    assert resp.json()["corruption_type"] == "random_holes"


def test_corrupt_and_repair_score_in_range():
    resp = _post_corrupt_and_repair()
    assert resp.status_code == 200
    score = resp.json()["score"]
    assert 0.0 <= float(score) <= 100.0, f"Score hors plage : {score}"


def test_corrupt_and_repair_candidates_not_empty():
    resp = _post_corrupt_and_repair()
    assert resp.status_code == 200
    candidates = resp.json()["candidates"]
    assert len(candidates) >= 1


def test_corrupt_and_repair_candidate_structure():
    resp = _post_corrupt_and_repair()
    assert resp.status_code == 200
    for c in resp.json()["candidates"]:
        assert "strategy" in c
        assert "path" in c
        assert "score" in c


def test_corrupt_and_repair_jpeg_input():
    """L'endpoint accepte aussi les JPEG."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr + 100).save(buf, format="JPEG")
    resp = _post_corrupt_and_repair(
        image_bytes=buf.getvalue(),
        filename="test.jpg",
        corruption_type="multiple_bars",
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests d'erreur (Ticket 7)
# ---------------------------------------------------------------------------

def test_corrupt_and_repair_invalid_file_format():
    """Un fichier texte doit retourner 422."""
    resp = client.post(
        "/pipeline/corrupt-and-repair",
        files={"image": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        data={"corruption_type": "scratch_lines", "severity": "medium", "max_attempts": "2"},
    )
    assert resp.status_code == 422


def test_corrupt_and_repair_unsupported_corruption_type():
    """Un type de corruption inexistant doit retourner 422."""
    resp = _post_corrupt_and_repair(corruption_type="alien_corruption")
    assert resp.status_code == 422
    assert "alien_corruption" in resp.json()["detail"]


def test_corrupt_and_repair_invalid_severity():
    """Une sévérité invalide doit retourner 422."""
    resp = _post_corrupt_and_repair(severity="extreme")
    assert resp.status_code == 422


def test_corrupt_and_repair_missing_image():
    """Pas de fichier uploadé → 422."""
    resp = client.post(
        "/pipeline/corrupt-and-repair",
        data={"corruption_type": "scratch_lines"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Smoke test tous les types de corruption via l'API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("corruption_type", [
    "scratch_lines",
    "large_deleted_square",
    "multiple_bars",
    "random_holes",
    "local_noise",
    "zone_deletion",
    "bar",
    "block_dropout",
    "mixed",
])
def test_all_corruption_types_via_api(corruption_type: str):
    resp = _post_corrupt_and_repair(
        corruption_type=corruption_type,
        max_attempts=2,
        seed=0,
    )
    assert resp.status_code == 200, (
        f"{corruption_type}: status {resp.status_code} — {resp.text[:200]}"
    )
    assert resp.json()["status"] == "completed"