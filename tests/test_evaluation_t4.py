"""Tests Ticket 4 — scoring supervisé et blind enrichis.

Vérifie :
  - compute_supervised_score retourne les bons champs avec des valeurs cohérentes
  - compute_blind_score retourne les bons champs
  - score_candidate délègue correctement selon présence de l'original
  - compare_images expose supervised_score et les alias reconstructed/corrupted
  - Les scores sont dans [0, 100]
  - Le score supervisé est meilleur quand la reconstruction est proche de l'original
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.modules.evaluation.metrics import (
    compute_blind_score,
    compute_supervised_score,
    score_candidate,
)
from app.modules.evaluation.comparator import compare_images


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_images(tmp_path: Path):
    """Retourne (original, corrupted, reconstructed) comme chemins PNG."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        arr[i, :, 0] = i * 4
        arr[:, i, 1] = i * 4
        arr[i, :, 2] = 255 - i * 4

    orig = tmp_path / "original.png"
    Image.fromarray(arr).save(orig)

    # Corrompu : zone noire
    arr_c = arr.copy()
    arr_c[10:30, 10:30] = 0
    corr = tmp_path / "corrupted.png"
    Image.fromarray(arr_c).save(corr)

    # Reconstruit : proche de l'original (légère différence)
    arr_r = arr.copy()
    arr_r[10:30, 10:30] = arr[10:30, 10:30] + 5
    arr_r = np.clip(arr_r, 0, 255).astype(np.uint8)
    recon = tmp_path / "reconstructed.png"
    Image.fromarray(arr_r).save(recon)

    return orig, corr, recon


@pytest.fixture
def identical_images(tmp_path: Path):
    """Original et reconstruit identiques (score max attendu)."""
    arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    orig = tmp_path / "orig.png"
    recon = tmp_path / "recon.png"
    corr = tmp_path / "corr.png"
    Image.fromarray(arr).save(orig)
    Image.fromarray(arr).save(recon)
    arr_c = np.zeros_like(arr)
    Image.fromarray(arr_c).save(corr)
    return orig, corr, recon


# ---------------------------------------------------------------------------
# Tests compute_supervised_score
# ---------------------------------------------------------------------------

class TestSupervisedScore:
    def test_returns_required_keys(self, tmp_images):
        orig, corr, recon = tmp_images
        result = compute_supervised_score(orig, recon, corr)
        for key in ("mode", "score", "psnr", "ssim", "gain_psnr", "gain_ssim",
                    "psnr_corrupted", "ssim_corrupted"):
            assert key in result, f"Clé manquante : {key}"

    def test_mode_is_supervised(self, tmp_images):
        orig, corr, recon = tmp_images
        result = compute_supervised_score(orig, recon, corr)
        assert result["mode"] == "supervised"

    def test_score_in_range(self, tmp_images):
        orig, corr, recon = tmp_images
        result = compute_supervised_score(orig, recon, corr)
        assert 0.0 <= result["score"] <= 100.0

    def test_perfect_reconstruction_high_score(self, identical_images):
        orig, corr, recon = identical_images
        result = compute_supervised_score(orig, recon, corr)
        assert result["score"] >= 50.0, f"Score trop bas pour reconstruction parfaite : {result['score']}"

    def test_gains_are_floats(self, tmp_images):
        orig, corr, recon = tmp_images
        result = compute_supervised_score(orig, recon, corr)
        assert isinstance(result["gain_psnr"], float)
        assert isinstance(result["gain_ssim"], float)

    def test_good_recon_better_than_corrupted(self, tmp_images):
        """La reconstruction proche de l'original > image corrompue."""
        orig, corr, recon = tmp_images
        result = compute_supervised_score(orig, recon, corr)
        assert result["psnr"] > result["psnr_corrupted"], (
            "PSNR reconstruit devrait dépasser PSNR corrompu"
        )


# ---------------------------------------------------------------------------
# Tests compute_blind_score
# ---------------------------------------------------------------------------

class TestBlindScore:
    def test_returns_required_keys(self, tmp_images):
        _, corr, recon = tmp_images
        result = compute_blind_score(corr, recon)
        for key in ("mode", "score", "psnr", "ssim", "sharpness",
                    "noise_penalty", "edge_continuity", "consistency"):
            assert key in result, f"Clé manquante : {key}"

    def test_mode_is_blind(self, tmp_images):
        _, corr, recon = tmp_images
        result = compute_blind_score(corr, recon)
        assert result["mode"] == "blind"

    def test_score_in_range(self, tmp_images):
        _, corr, recon = tmp_images
        result = compute_blind_score(corr, recon)
        assert 0.0 <= result["score"] <= 100.0

    def test_psnr_ssim_are_none(self, tmp_images):
        """En mode blind, PSNR et SSIM sont None (pas d'original)."""
        _, corr, recon = tmp_images
        result = compute_blind_score(corr, recon)
        assert result["psnr"] is None
        assert result["ssim"] is None

    def test_sharpness_is_positive(self, tmp_images):
        _, corr, recon = tmp_images
        result = compute_blind_score(corr, recon)
        assert result["sharpness"] >= 0.0

    def test_accepts_numpy_arrays(self):
        """compute_blind_score accepte des tableaux numpy directement."""
        arr = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        result = compute_blind_score(arr, arr)
        assert 0.0 <= result["score"] <= 100.0


# ---------------------------------------------------------------------------
# Tests score_candidate (point d'entrée unifié)
# ---------------------------------------------------------------------------

class TestScoreCandidate:
    def test_with_original_returns_supervised(self, tmp_images):
        orig, corr, recon = tmp_images
        result = score_candidate(corr, recon, original=orig)
        assert result["mode"] == "supervised"

    def test_without_original_returns_blind(self, tmp_images):
        _, corr, recon = tmp_images
        result = score_candidate(corr, recon, original=None)
        assert result["mode"] == "blind"

    def test_score_always_in_range(self, tmp_images):
        orig, corr, recon = tmp_images
        for original in (orig, None):
            result = score_candidate(corr, recon, original=original)
            assert 0.0 <= result["score"] <= 100.0, (
                f"Score hors plage en mode {'supervised' if original else 'blind'} : {result['score']}"
            )


# ---------------------------------------------------------------------------
# Tests compare_images enrichi
# ---------------------------------------------------------------------------

class TestCompareImages:
    def test_returns_supervised_score_key(self, tmp_images):
        orig, corr, recon = tmp_images
        result = compare_images(orig, corr, recon)
        assert "supervised_score" in result, "Clé supervised_score manquante"

    def test_supervised_score_structure(self, tmp_images):
        orig, corr, recon = tmp_images
        result = compare_images(orig, corr, recon)
        sup = result["supervised_score"]
        assert sup["mode"] == "supervised"
        assert 0.0 <= sup["score"] <= 100.0

    def test_alias_reconstructed_present(self, tmp_images):
        """Alias 'reconstructed' requis par repair_pipeline._score_candidate."""
        orig, corr, recon = tmp_images
        result = compare_images(orig, corr, recon)
        assert "reconstructed" in result
        assert "psnr" in result["reconstructed"]
        assert "ssim" in result["reconstructed"]

    def test_alias_corrupted_present(self, tmp_images):
        orig, corr, recon = tmp_images
        result = compare_images(orig, corr, recon)
        assert "corrupted" in result

    def test_backward_compat_original_vs_corrupted(self, tmp_images):
        """Les clés historiques restent présentes."""
        orig, corr, recon = tmp_images
        result = compare_images(orig, corr, recon)
        assert "original_vs_corrupted" in result
        assert "original_vs_reconstructed" in result
        assert "gains" in result
        assert "improvement" in result

    def test_improvement_true_when_better(self, tmp_images):
        """La reconstruction proche de l'original doit être marquée improvement=True."""
        orig, corr, recon = tmp_images
        result = compare_images(orig, corr, recon)
        # Notre fixture recon est très proche de orig, corr est très dégradé
        assert result["improvement"] is True