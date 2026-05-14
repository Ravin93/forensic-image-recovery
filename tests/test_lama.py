"""Tests K9 — LaMa adapter."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def test_image(tmp_path):
    arr = np.zeros((64,64,3),dtype=np.uint8)+100
    p = tmp_path/"img.png"; Image.fromarray(arr).save(p); return p

@pytest.fixture
def test_mask(tmp_path):
    arr = np.zeros((64,64),dtype=np.uint8)
    arr[20:40,20:40]=255
    p = tmp_path/"mask.png"; Image.fromarray(arr).save(p); return p


class TestLamaAdapter:
    def test_lama_status_returns_dict(self):
        from app.modules.ai.lama_adapter import get_lama_status
        s = get_lama_status()
        for k in ("enabled","has_torch","has_model","available","reason"):
            assert k in s

    def test_disabled_by_default(self):
        from app.modules.ai.lama_adapter import is_lama_enabled
        import os
        os.environ.pop("LAMA_ENABLED", None)
        assert is_lama_enabled() == False

    def test_run_lama_returns_dict(self, test_image, test_mask):
        from app.modules.ai.lama_adapter import run_lama_inpainting
        r = run_lama_inpainting(test_image, test_mask)
        assert isinstance(r, dict)

    def test_required_keys(self, test_image, test_mask):
        from app.modules.ai.lama_adapter import run_lama_inpainting
        r = run_lama_inpainting(test_image, test_mask)
        for k in ("strategy","forensic_mode","warning","available","status"):
            assert k in r

    def test_strategy_name(self, test_image, test_mask):
        from app.modules.ai.lama_adapter import run_lama_inpainting
        r = run_lama_inpainting(test_image, test_mask)
        assert r["strategy"] == "lama_inpainting"

    def test_forensic_mode_generative(self, test_image, test_mask):
        from app.modules.ai.lama_adapter import run_lama_inpainting
        r = run_lama_inpainting(test_image, test_mask)
        assert r["forensic_mode"] == "generative"

    def test_disabled_skip(self, test_image, test_mask):
        import os
        os.environ["LAMA_ENABLED"] = "false"
        from app.modules.ai.lama_adapter import run_lama_inpainting
        r = run_lama_inpainting(test_image, test_mask)
        assert r["status"] == "skipped"
        os.environ.pop("LAMA_ENABLED", None)

    def test_warning_not_probant(self, test_image, test_mask):
        from app.modules.ai.lama_adapter import run_lama_inpainting
        r = run_lama_inpainting(test_image, test_mask)
        warning = r.get("warning","").lower()
        assert "probant" in warning or "generatif" in warning or "generative" in warning