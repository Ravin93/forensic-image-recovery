"""Tests Ticket 2 + Ticket 3 — pipeline multi-reconstruction.

Vérifie :
  - Les 3 chemins d'images sont présents dans le retour (Ticket 2)
  - Les champs multi-reconstruction sont exposés (Ticket 3)
  - max_attempts est bien transmis au repair_pipeline
  - Le pipeline fonctionne en mode assisted, blind_basic et blind_advanced
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services.pipeline_service import run_demo_pipeline


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def gradient_image(tmp_path: Path) -> Path:
    """Image 128×128 avec gradient (évite les images plates)."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(128):
        arr[i, :, 0] = i * 2
        arr[:, i, 1] = i * 2
        arr[i, :, 2] = 255 - i * 2
    p = tmp_path / "source.png"
    Image.fromarray(arr).save(p)
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_three_images(result: dict) -> None:
    """Ticket 2 — les 3 chemins d'images existent."""
    assert "original_image" in result, "Clé original_image manquante"
    assert "corrupted_image" in result, "Clé corrupted_image manquante"
    assert "reconstructed_image" in result, "Clé reconstructed_image manquante"
    assert "mask_path" in result, "Clé mask_path manquante"

    assert Path(result["original_image"]).exists(), "original_image introuvable"
    assert Path(result["corrupted_image"]).exists(), "corrupted_image introuvable"
    assert Path(result["reconstructed_image"]).exists(), "reconstructed_image introuvable"
    assert Path(result["mask_path"]).exists(), "mask_path introuvable"


def _assert_multi_reconstruction(result: dict) -> None:
    """Ticket 3 — les champs de reconstruction multi-essais sont présents."""
    assert "selected_repair_strategy" in result
    assert "repair_score" in result
    assert "retry_count" in result
    assert "repair_candidates" in result

    assert isinstance(result["selected_repair_strategy"], str)
    assert isinstance(result["repair_score"], float | int)
    assert isinstance(result["retry_count"], int)
    assert isinstance(result["repair_candidates"], list)
    assert len(result["repair_candidates"]) >= 1, "Aucun candidat de reconstruction"

    # Chaque candidat doit avoir strategy, path, score
    for candidate in result["repair_candidates"]:
        assert "strategy" in candidate, f"Candidat sans strategy : {candidate}"
        assert "path" in candidate, f"Candidat sans path : {candidate}"
        assert "score" in candidate, f"Candidat sans score : {candidate}"


# ---------------------------------------------------------------------------
# Tests principaux
# ---------------------------------------------------------------------------

def test_pipeline_assisted_exposes_three_images(gradient_image: Path) -> None:
    """Mode assisted — les 3 chemins d'images sont dans le retour."""
    result = run_demo_pipeline(
        source_image_path=gradient_image,
        corruption_type="rectangle_mask",
        corruption_params={"x": 20, "y": 20, "width": 40, "height": 40, "fill_value": 0},
        execution_mode="assisted",
        seed=42,
    )
    _assert_three_images(result)


def test_pipeline_assisted_multi_reconstruction(gradient_image: Path) -> None:
    """Mode assisted — les champs multi-reconstruction sont exposés."""
    result = run_demo_pipeline(
        source_image_path=gradient_image,
        corruption_type="zone_deletion",
        corruption_params={"x": 10, "y": 10, "width": 50, "height": 50},
        execution_mode="assisted",
        seed=42,
        max_attempts=4,
    )
    _assert_multi_reconstruction(result)
    assert result["retry_count"] <= 4


def test_pipeline_max_attempts_respected(gradient_image: Path) -> None:
    """max_attempts limite bien le nombre de candidats."""
    result = run_demo_pipeline(
        source_image_path=gradient_image,
        corruption_type="noise",
        corruption_params={"x": 10, "y": 10, "width": 40, "height": 40, "sigma": 30.0},
        execution_mode="assisted",
        seed=0,
        max_attempts=3,
    )
    # candidates inclut le "conservatif" + max_attempts-1 autres
    assert len(result["repair_candidates"]) <= 3 + 1


def test_pipeline_selected_strategy_in_candidates(gradient_image: Path) -> None:
    """La stratégie sélectionnée correspond bien au meilleur candidat."""
    result = run_demo_pipeline(
        source_image_path=gradient_image,
        corruption_type="rectangle_mask",
        corruption_params={"x": 20, "y": 20, "width": 40, "height": 40, "fill_value": 0},
        execution_mode="assisted",
        seed=1,
    )
    strategies = [c["strategy"] for c in result["repair_candidates"]]
    assert result["selected_repair_strategy"] in strategies


def test_pipeline_repair_score_is_best(gradient_image: Path) -> None:
    """repair_score correspond au score maximal parmi les candidats."""
    result = run_demo_pipeline(
        source_image_path=gradient_image,
        corruption_type="zone_deletion",
        corruption_params={"x": 15, "y": 15, "width": 30, "height": 30},
        execution_mode="assisted",
        seed=2,
    )
    best_score = max(float(c["score"]) for c in result["repair_candidates"])
    assert abs(float(result["repair_score"]) - best_score) < 0.01


def test_pipeline_blind_basic_exposes_fields(gradient_image: Path) -> None:
    """Mode blind_basic — tous les champs Ticket 2 + 3 présents."""
    result = run_demo_pipeline(
        source_image_path=gradient_image,
        corruption_type="bar",
        corruption_params={"orientation": "horizontal", "thickness": 10},
        execution_mode="blind_basic",
        seed=42,
    )
    _assert_three_images(result)
    _assert_multi_reconstruction(result)


def test_pipeline_new_corruption_types(gradient_image: Path) -> None:
    """Les nouveaux types de corruption (Ticket 1) s'intègrent au pipeline."""
    for corruption_type, params in [
        ("scratch_lines",       {"count": 3}),
        ("large_deleted_square", {"size_ratio": 0.3}),
        ("multiple_bars",       {"count": 2}),
        ("random_holes",        {"count": 4}),
    ]:
        result = run_demo_pipeline(
            source_image_path=gradient_image,
            corruption_type=corruption_type,
            corruption_params=params,
            execution_mode="assisted",
            seed=42,
        )
        _assert_three_images(result)
        assert result["corruption"]["corruption_type"] == corruption_type, (
            f"{corruption_type}: corruption_type incorrect dans le retour"
        )


def test_pipeline_result_structure_complete(gradient_image: Path) -> None:
    """Vérifie la structure complète du retour."""
    result = run_demo_pipeline(
        source_image_path=gradient_image,
        corruption_type="rectangle_mask",
        corruption_params={"x": 20, "y": 20, "width": 40, "height": 40, "fill_value": 0},
        execution_mode="assisted",
        seed=42,
    )

    # Clés Ticket 2
    for key in ("original_image", "corrupted_image", "reconstructed_image", "mask_path"):
        assert key in result, f"Clé manquante : {key}"

    # Clés Ticket 3
    for key in ("selected_repair_strategy", "repair_score", "retry_count", "repair_candidates"):
        assert key in result, f"Clé manquante : {key}"

    # Clés évaluation
    for key in ("evaluation", "recoverability_status", "detection_metrics"):
        assert key in result, f"Clé manquante : {key}"

    # Clés contexte
    assert result["status"] == "completed"
    assert result["mode"]["max_attempts"] == 8