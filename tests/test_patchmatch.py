"""Tests K8 — PatchMatch inpainting."""
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def image_and_mask(tmp_path):
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(128):
        arr[i, :, 0] = i * 2
        arr[:, i, 1] = i * 2
        arr[i, :, 2] = 200 - i
    img_path = tmp_path / "test.png"
    Image.fromarray(arr).save(img_path)

    mask_arr = np.zeros((128, 128), dtype=np.uint8)
    mask_arr[40:70, 40:70] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask_arr).save(mask_path)

    (tmp_path / "reconstructed").mkdir(exist_ok=True)
    return img_path, mask_path


class TestPatchMatch:
    def test_returns_dict(self, image_and_mask):
        from app.modules.reconstruction.patchmatch import patchmatch_inpaint
        img, mask = image_and_mask
        result = patchmatch_inpaint(img, mask, patch_size=7, iterations=2)
        assert isinstance(result, dict)

    def test_result_file_exists(self, image_and_mask):
        from app.modules.reconstruction.patchmatch import patchmatch_inpaint
        img, mask = image_and_mask
        result = patchmatch_inpaint(img, mask, patch_size=7, iterations=2)
        assert Path(result["path"]).exists()

    def test_dimensions_preserved(self, image_and_mask):
        from app.modules.reconstruction.patchmatch import patchmatch_inpaint
        img, mask = image_and_mask
        result = patchmatch_inpaint(img, mask, patch_size=7, iterations=2)
        import cv2
        out = cv2.imread(result["path"])
        src = cv2.imread(str(img))
        assert out.shape == src.shape

    def test_result_has_required_keys(self, image_and_mask):
        from app.modules.reconstruction.patchmatch import patchmatch_inpaint
        img, mask = image_and_mask
        result = patchmatch_inpaint(img, mask, patch_size=7, iterations=2)
        for k in ("file","path","method","status","source_image","mask_path",
                  "patch_size","iterations","elapsed_s"):
            assert k in result, f"Cle manquante : {k}"

    def test_method_name_includes_params(self, image_and_mask):
        from app.modules.reconstruction.patchmatch import patchmatch_inpaint
        img, mask = image_and_mask
        result = patchmatch_inpaint(img, mask, patch_size=9, iterations=3)
        assert "p9" in result["method"]
        assert "i3" in result["method"]

    def test_scoreable_by_metrics(self, image_and_mask):
        from app.modules.reconstruction.patchmatch import patchmatch_inpaint
        from app.modules.evaluation.metrics import compute_psnr, compute_ssim
        img, mask = image_and_mask
        result = patchmatch_inpaint(img, mask, patch_size=7, iterations=2)
        psnr = compute_psnr(img, result["path"])
        ssim = compute_ssim(img, result["path"])
        assert psnr >= 0
        assert 0.0 <= ssim <= 1.0

    def test_in_repair_pipeline(self, image_and_mask):
        """Verifie que patchmatch est dans le plan adaptatif et scoreable via le pipeline."""
        from app.modules.reconstruction.repair_pipeline import (
            _build_adaptive_plan, _candidate_from_path
        )
        img, mask = image_and_mask
        # Verifier que patchmatch est dans le plan (peu importe max_attempts)
        plan = _build_adaptive_plan("zone_deletion", 3, "inpainting")
        pm_plans = [p for p in plan if p["family"] == "patchmatch"]
        assert len(pm_plans) >= 1, f"Famille patchmatch absente du plan : {[p['family'] for p in plan]}"
        # Verifier que les strategies ont les bons parametres
        strategies = [p["strategy"] for p in pm_plans]
        assert any("p7" in s for s in strategies), f"patchmatch_p7 absent : {strategies}"

    def test_invalid_image_raises(self, tmp_path):
        from app.modules.reconstruction.patchmatch import patchmatch_inpaint
        from app.core.exceptions import ReconstructionError
        mask = tmp_path / "mask.png"
        Image.fromarray(np.zeros((32,32),dtype=np.uint8)).save(mask)
        with pytest.raises(ReconstructionError):
            patchmatch_inpaint(tmp_path / "nonexistent.png", mask)