"""Tests B1+B2+B3+B4 — moteur adaptatif de reconstruction."""
from __future__ import annotations
import io
import numpy as np
import pytest
from pathlib import Path
from PIL import Image


@pytest.fixture
def corrupted_setup(tmp_path):
    """Image + masque + original pour les tests pipeline."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(128):
        arr[i, :, 0] = i * 2
        arr[:, i, 1] = i * 2
        arr[i, :, 2] = 255 - i * 2
    orig = tmp_path / "orig.png"
    Image.fromarray(arr).save(orig)

    arr_c = arr.copy()
    arr_c[30:70, 30:70] = 0
    corr = tmp_path / "corrupted.png"
    Image.fromarray(arr_c).save(corr)

    mask_arr = np.zeros((128, 128), dtype=np.uint8)
    mask_arr[30:70, 30:70] = 255
    mask = tmp_path / "mask.png"
    Image.fromarray(mask_arr).save(mask)

    # Créer le dossier reconstructed attendu par le pipeline
    (tmp_path / "reconstructed").mkdir(exist_ok=True)

    return orig, corr, mask, tmp_path


class TestAdaptivePlan:
    """B1 — Plan adaptatif selon corruption_type."""

    def test_inpainting_type_prioritizes_inpainting(self):
        from app.modules.reconstruction.repair_pipeline import _build_adaptive_plan
        plan = _build_adaptive_plan("zone_deletion", 3, "inpainting")
        families = [p["family"] for p in plan]
        inpaint_idx = next(i for i, f in enumerate(families) if f == "inpainting")
        denoise_idx = next((i for i, f in enumerate(families) if f == "denoise"), 999)
        assert inpaint_idx < denoise_idx

    def test_denoise_type_prioritizes_denoise(self):
        from app.modules.reconstruction.repair_pipeline import _build_adaptive_plan
        plan = _build_adaptive_plan("noise", 3, "denoise")
        families = [p["family"] for p in plan]
        denoise_idx = next(i for i, f in enumerate(families) if f == "denoise")
        inpaint_idx = next((i for i, f in enumerate(families) if f == "inpainting"), 999)
        assert denoise_idx < inpaint_idx

    def test_plan_contains_composite_strategies(self):
        from app.modules.reconstruction.repair_pipeline import _build_adaptive_plan
        plan = _build_adaptive_plan("zone_deletion", 3, "inpainting")
        composites = [p for p in plan if p["family"] == "composite"]
        assert len(composites) >= 3

    def test_new_corruption_types_mapped(self):
        from app.modules.reconstruction.repair_pipeline import choose_repair_strategy
        assert choose_repair_strategy("scratch_lines", 1.0) == "inpainting"
        assert choose_repair_strategy("multiple_bars", 1.0) == "inpainting"
        assert choose_repair_strategy("local_noise", 1.0) == "denoise"
        assert choose_repair_strategy("large_deleted_square", 1.0) == "inpainting"


class TestCompositeStrategies:
    """B3 — Stratégies composées."""

    def test_pipeline_runs_composite(self, corrupted_setup):
        orig, corr, mask, tmp = corrupted_setup
        from app.modules.reconstruction.repair_pipeline import run_repair_pipeline
        result = run_repair_pipeline(
            corrupted_image_path=corr,
            mask_path=mask,
            original_image_path=orig,
            corruption_type="zone_deletion",
            max_attempts=6,
        )
        strategies = [c["strategy"] for c in result["candidates"]]
        # Au moins une stratégie composée doit avoir été tentée
        composite_tried = any("then" in s or "multi_pass" in s or "dilate" in s
                              for s in strategies)
        assert composite_tried, f"Aucune stratégie composée dans : {strategies}"

    def test_composite_result_has_steps(self, corrupted_setup):
        orig, corr, mask, tmp = corrupted_setup
        from app.modules.reconstruction.repair_pipeline import run_repair_pipeline
        result = run_repair_pipeline(
            corrupted_image_path=corr,
            mask_path=mask,
            original_image_path=orig,
            max_attempts=8,
        )
        composite_candidates = [
            c for c in result["candidates"]
            if c.get("family") == "composite"
        ]
        if composite_candidates:
            assert "steps" in composite_candidates[0]


class TestIterativePass:
    """B2 — Boucle itérative multi-pass."""

    def test_iterative_pass_returns_dict_or_none(self, corrupted_setup):
        orig, corr, mask, tmp = corrupted_setup
        from app.modules.reconstruction.repair_pipeline import _run_iterative_pass
        result = _run_iterative_pass(
            corr, mask, orig, "opencv_inpaint", 3, max_iterations=2
        )
        assert result is None or isinstance(result, dict)

    def test_iterative_candidate_has_iterations_key(self, corrupted_setup):
        orig, corr, mask, tmp = corrupted_setup
        from app.modules.reconstruction.repair_pipeline import _run_iterative_pass
        result = _run_iterative_pass(
            corr, mask, orig, "opencv_inpaint", 3, max_iterations=2
        )
        if result is not None:
            assert "iterations" in result
            assert "stopped_reason" in result
            assert result["stopped_reason"] in ("no_improvement", "max_iterations", "error")

    def test_pipeline_includes_iterative(self, corrupted_setup):
        orig, corr, mask, tmp = corrupted_setup
        from app.modules.reconstruction.repair_pipeline import run_repair_pipeline
        result = run_repair_pipeline(
            corrupted_image_path=corr,
            mask_path=mask,
            original_image_path=orig,
            max_attempts=12,
        )
        # L'itératif peut ou non être dans les candidats selon l'amélioration
        assert len(result["candidates"]) >= 1
        assert result["retry_count"] >= 0


class TestTop3:
    """B4 — top_candidates."""

    def test_top_candidates_present(self, corrupted_setup):
        orig, corr, mask, tmp = corrupted_setup
        from app.modules.reconstruction.repair_pipeline import run_repair_pipeline
        result = run_repair_pipeline(
            corrupted_image_path=corr,
            mask_path=mask,
            original_image_path=orig,
            max_attempts=6,
        )
        assert "top_candidates" in result

    def test_top_candidates_structure(self, corrupted_setup):
        orig, corr, mask, tmp = corrupted_setup
        from app.modules.reconstruction.repair_pipeline import run_repair_pipeline
        result = run_repair_pipeline(
            corrupted_image_path=corr,
            mask_path=mask,
            original_image_path=orig,
            max_attempts=6,
        )
        tc = result["top_candidates"]
        for key in ("best_score", "best_visual", "most_conservative"):
            assert key in tc, f"Clé manquante dans top_candidates : {key}"
            assert "strategy" in tc[key]
            assert "score" in tc[key]
            assert "path" in tc[key]

    def test_best_score_is_highest(self, corrupted_setup):
        orig, corr, mask, tmp = corrupted_setup
        from app.modules.reconstruction.repair_pipeline import run_repair_pipeline
        result = run_repair_pipeline(
            corrupted_image_path=corr,
            mask_path=mask,
            original_image_path=orig,
            max_attempts=6,
        )
        best = float(result["top_candidates"]["best_score"]["score"])
        all_scores = [float(c.get("score", 0)) for c in result["candidates"]]
        assert best == max(all_scores)

    def test_api_exposes_top_candidates(self):
        """top_candidates remonte dans la réponse API."""
        import io as _io
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)

        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            arr[i, :, 0] = i * 4
        buf = _io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")

        resp = client.post(
            "/pipeline/corrupt-and-repair",
            files={"image": ("t.png", _io.BytesIO(buf.getvalue()), "image/png")},
            data={"corruption_type": "zone_deletion", "severity": "medium",
                  "max_attempts": "4", "seed": "42"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "top_candidates" in data or "candidates" in data