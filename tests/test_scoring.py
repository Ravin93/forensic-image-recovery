"""Tests D1+D2+D3+D4 — scoring avancé détaillé."""
from __future__ import annotations
import io
import numpy as np
import pytest
from PIL import Image
from pathlib import Path


@pytest.fixture
def trio(tmp_path):
    """Retourne (original, corrupted, reconstructed) comme chemins PNG."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        arr[i, :, 0] = i * 4
        arr[:, i, 1] = i * 4
        arr[i, :, 2] = 255 - i * 4
    orig = tmp_path / "orig.png"; Image.fromarray(arr).save(orig)
    arr_c = arr.copy(); arr_c[10:30, 10:30] = 0
    corr = tmp_path / "corr.png"; Image.fromarray(arr_c).save(corr)
    arr_r = arr.copy(); arr_r[10:30, 10:30] = arr[10:30, 10:30] + 3
    arr_r = np.clip(arr_r, 0, 255).astype(np.uint8)
    recon = tmp_path / "recon.png"; Image.fromarray(arr_r).save(recon)
    mask = np.zeros((64, 64), dtype=np.uint8); mask[10:30, 10:30] = 255
    return orig, corr, recon, mask


class TestSupervisedDetailed:
    """D1 — Score supervisé avec zones masque/hors-masque."""

    def test_required_keys(self, trio):
        from app.modules.evaluation.metrics import compute_supervised_score
        orig, corr, recon, mask = trio
        r = compute_supervised_score(orig, recon, corr, mask=mask)
        for k in ("mode","score","psnr","ssim","gain_psnr","gain_ssim",
                  "mask_region_score","outside_score","outside_preservation","score_breakdown"):
            assert k in r, f"Clé manquante D1 : {k}"

    def test_mask_region_score_not_none_when_mask_provided(self, trio):
        from app.modules.evaluation.metrics import compute_supervised_score
        orig, corr, recon, mask = trio
        r = compute_supervised_score(orig, recon, corr, mask=mask)
        assert r["mask_region_score"] is not None
        assert 0.0 <= r["mask_region_score"] <= 100.0

    def test_outside_preservation_not_none(self, trio):
        from app.modules.evaluation.metrics import compute_supervised_score
        orig, corr, recon, mask = trio
        r = compute_supervised_score(orig, recon, corr, mask=mask)
        assert r["outside_preservation"] is not None

    def test_no_mask_zones_are_none(self, trio):
        from app.modules.evaluation.metrics import compute_supervised_score
        orig, corr, recon, _ = trio
        r = compute_supervised_score(orig, recon, corr, mask=None)
        assert r["mask_region_score"] is None
        assert r["outside_score"] is None

    def test_score_in_range(self, trio):
        from app.modules.evaluation.metrics import compute_supervised_score
        orig, corr, recon, mask = trio
        r = compute_supervised_score(orig, recon, corr, mask=mask)
        assert 0.0 <= r["score"] <= 100.0


class TestBlindDetailed:
    """D2 — Score aveugle enrichi (cohérence couleur, entropie, contours)."""

    def test_required_keys(self, trio):
        from app.modules.evaluation.metrics import compute_blind_score
        _, corr, recon, _ = trio
        r = compute_blind_score(corr, recon)
        for k in ("mode","score","sharpness","noise_penalty","edge_continuity",
                  "coherence_color","local_entropy","score_breakdown"):
            assert k in r, f"Clé manquante D2 : {k}"

    def test_new_d2_metrics_present(self, trio):
        from app.modules.evaluation.metrics import compute_blind_score
        _, corr, recon, _ = trio
        r = compute_blind_score(corr, recon)
        assert "edge_continuity"  in r
        assert "coherence_color"  in r
        assert "local_entropy"    in r

    def test_score_in_range(self, trio):
        from app.modules.evaluation.metrics import compute_blind_score
        _, corr, recon, _ = trio
        r = compute_blind_score(corr, recon)
        assert 0.0 <= r["score"] <= 100.0

    def test_accepts_numpy(self):
        from app.modules.evaluation.metrics import compute_blind_score
        arr = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        r = compute_blind_score(arr, arr)
        assert 0.0 <= r["score"] <= 100.0


class TestScoreBreakdown:
    """D3 — score_breakdown exposé dans les deux modes."""

    def test_supervised_breakdown_keys(self, trio):
        from app.modules.evaluation.metrics import compute_supervised_score
        orig, corr, recon, mask = trio
        bd = compute_supervised_score(orig, recon, corr, mask=mask)["score_breakdown"]
        for k in ("global_score","ssim_component","psnr_component",
                  "gain_ssim_component","gain_psnr_component","mask_region_score"):
            assert k in bd, f"Clé breakdown manquante : {k}"

    def test_blind_breakdown_keys(self, trio):
        from app.modules.evaluation.metrics import compute_blind_score
        _, corr, recon, _ = trio
        bd = compute_blind_score(corr, recon)["score_breakdown"]
        for k in ("global_score","sharpness_component","noise_component",
                  "edge_component","color_component","entropy_component","consistency_factor"):
            assert k in bd, f"Clé breakdown blind manquante : {k}"

    def test_breakdown_sum_consistent(self, trio):
        """La somme des composantes doit être proche du score global."""
        from app.modules.evaluation.metrics import compute_supervised_score
        orig, corr, recon, _ = trio
        r = compute_supervised_score(orig, recon, corr)
        bd = r["score_breakdown"]
        total = (bd["ssim_component"] + bd["psnr_component"]
                 + bd["gain_ssim_component"] + bd["gain_psnr_component"])
        assert abs(total - bd["global_score"]) < 0.5, (
            f"Somme composantes ({total:.2f}) ≠ global ({bd['global_score']:.2f})"
        )


class TestScoreCandidateAPI:
    """D3 — score_candidate expose score_breakdown."""

    def test_with_original_has_breakdown(self, trio):
        from app.modules.evaluation.metrics import score_candidate
        orig, corr, recon, _ = trio
        r = score_candidate(corr, recon, original=orig)
        assert "score_breakdown" in r

    def test_blind_has_breakdown(self, trio):
        from app.modules.evaluation.metrics import score_candidate
        _, corr, recon, _ = trio
        r = score_candidate(corr, recon, original=None)
        assert "score_breakdown" in r

    def test_with_mask_exposes_zone_scores(self, trio):
        from app.modules.evaluation.metrics import score_candidate
        orig, corr, recon, mask = trio
        r = score_candidate(corr, recon, original=orig, mask=mask)
        assert r["mask_region_score"] is not None
        assert r["outside_preservation"] is not None


class TestAPIBreakdown:
    """D4 — score_breakdown remonte dans l'API /pipeline/corrupt-and-repair."""

    def test_candidates_have_score_breakdown(self):
        import io as _io
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)

        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            arr[i, :, 0] = i * 4
            arr[:, i, 1] = i * 4
        buf = _io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")

        resp = client.post(
            "/pipeline/corrupt-and-repair",
            files={"image": ("t.png", _io.BytesIO(buf.getvalue()), "image/png")},
            data={"corruption_type": "scratch_lines", "severity": "light",
                  "max_attempts": "2", "seed": "42"},
        )
        assert resp.status_code == 200
        data = resp.json()
        candidates = data.get("candidates", [])
        assert len(candidates) >= 1
        # Au moins le meilleur candidat doit avoir score_breakdown
        best_strategy = data.get("selected_repair_strategy")
        best = next((c for c in candidates if c.get("strategy") == best_strategy), candidates[0])
        assert "score_breakdown" in best, "score_breakdown absent du candidat"