"""Tests du module corruption — Ticket 1.

Structure :
  - Tests existants (rectangle_mask, noise, zone_deletion, erreurs)
  - Tests nouveaux types (scratch_lines, large_deleted_square, multiple_bars,
                          random_holes, local_noise)
  - Tests de validation : image réellement modifiée, masque non vide
  - Test apply_* direct (indépendant du filesystem)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.exceptions import CorruptionError
from app.modules.corruption.simulator import corrupt_image
from app.modules.corruption.transforms import (
    apply_bar_corruption,
    apply_block_dropout,
    apply_large_deleted_square,
    apply_local_blur,
    apply_local_noise,
    apply_multiple_bars,
    apply_noise,
    apply_random_holes,
    apply_rectangle_mask,
    apply_scratch_lines,
    apply_zone_deletion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_gradient_image() -> Image.Image:
    """Image 128×128 avec gradient RGB.

    Une image uniforme trompe local_blur et jpeg_block_artifacts :
    le flou d'une zone plate = identique, downscale/upscale aussi.
    Le gradient garantit qu'une modification dans n'importe quelle zone
    sera toujours détectable par np.array_equal.
    """
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(128):
        arr[i, :, 0] = i * 2        # rouge : gradient vertical
        arr[:, i, 1] = i * 2        # vert  : gradient horizontal
        arr[i, :, 2] = 255 - i * 2  # bleu  : gradient inverse
    return Image.fromarray(arr)


@pytest.fixture
def blue_image(tmp_path: Path) -> Path:
    """Image 128×128 avec gradient RGB."""
    p = tmp_path / "source.png"
    _make_gradient_image().save(p)
    return p


@pytest.fixture
def blue_pil() -> Image.Image:
    """PIL Image 128×128 avec gradient RGB (pas de filesystem)."""
    return _make_gradient_image()


def _arr(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


# ---------------------------------------------------------------------------
# Tests existants (conservés tels quels)
# ---------------------------------------------------------------------------

def test_corrupt_image_rectangle_mask(blue_image: Path) -> None:
    result = corrupt_image(
        image_path=blue_image,
        corruption_type="rectangle_mask",
        x=20, y=20, width=40, height=40, fill_value=0,
    )
    assert Path(result["path"]).exists()
    assert Path(result["mask_path"]).exists()
    assert result["status"] == "corrupted"
    assert result["corruption_type"] == "rectangle_mask"


def test_corrupt_image_noise(blue_image: Path) -> None:
    result = corrupt_image(
        image_path=blue_image,
        corruption_type="noise",
        x=10, y=10, width=30, height=30, sigma=20.0, seed=42,
    )
    assert Path(result["path"]).exists()
    assert Path(result["mask_path"]).exists()
    assert result["corruption_type"] == "noise"


def test_corrupt_image_zone_deletion(blue_image: Path) -> None:
    result = corrupt_image(
        image_path=blue_image,
        corruption_type="zone_deletion",
        x=15, y=15, width=20, height=20,
    )
    assert Path(result["path"]).exists()
    assert Path(result["mask_path"]).exists()
    assert result["corruption_type"] == "zone_deletion"


def test_corrupt_missing_source() -> None:
    with pytest.raises(CorruptionError):
        corrupt_image(
            image_path="missing.png",
            corruption_type="rectangle_mask",
            x=10, y=10, width=20, height=20, fill_value=0,
        )


def test_corrupt_unsupported_type(blue_image: Path) -> None:
    with pytest.raises(CorruptionError):
        corrupt_image(image_path=blue_image, corruption_type="unsupported_mode")


# ---------------------------------------------------------------------------
# Helpers de validation réutilisables
# ---------------------------------------------------------------------------

def _assert_valid_output(
    corrupted: Image.Image,
    mask: Image.Image,
    original: Image.Image,
    params: dict,
) -> None:
    """Vérifie les invariants communs à toute corruption."""
    arr_orig = _arr(original)
    arr_corr = _arr(corrupted)
    arr_mask = np.array(mask.convert("L"))

    # 1. Dimensions préservées
    assert arr_corr.shape == arr_orig.shape, "La corruption change les dimensions"

    # 2. Masque non vide
    assert arr_mask.max() == 255, "Le masque est complètement vide"

    # 3. Image réellement modifiée (au moins 1 pixel différent)
    assert not np.array_equal(arr_corr, arr_orig), "L'image n'a pas été modifiée"

    # 4. Les pixels hors masque restent inchangés (invariant fort)
    outside = arr_mask == 0
    assert np.array_equal(
        arr_corr[outside], arr_orig[outside]
    ), "Des pixels hors masque ont été modifiés"

    # 5. params est un dict non vide
    assert isinstance(params, dict) and len(params) > 0


# ---------------------------------------------------------------------------
# Tests — Nouveaux types via corrupt_image (filesystem)
# ---------------------------------------------------------------------------

class TestScratchLines:
    def test_filesystem(self, blue_image: Path) -> None:
        result = corrupt_image(
            image_path=blue_image,
            corruption_type="scratch_lines",
            count=5, thickness=2, seed=0,
        )
        assert Path(result["path"]).exists()
        assert Path(result["mask_path"]).exists()
        assert result["corruption_type"] == "scratch_lines"

    def test_direct_transform(self, blue_pil: Image.Image) -> None:
        corrupted, mask, params = apply_scratch_lines(blue_pil, count=3, seed=1)
        _assert_valid_output(corrupted, mask, blue_pil, params)
        assert params["count"] == 3

    def test_thick_scratches(self, blue_pil: Image.Image) -> None:
        corrupted, mask, params = apply_scratch_lines(
            blue_pil, count=2, thickness=5, seed=2
        )
        _assert_valid_output(corrupted, mask, blue_pil, params)


class TestLargeDeletedSquare:
    def test_filesystem(self, blue_image: Path) -> None:
        result = corrupt_image(
            image_path=blue_image,
            corruption_type="large_deleted_square",
            size_ratio=0.4, seed=0,
        )
        assert Path(result["path"]).exists()
        assert result["corruption_type"] == "large_deleted_square"

    def test_direct_transform(self, blue_pil: Image.Image) -> None:
        corrupted, mask, params = apply_large_deleted_square(
            blue_pil, size_ratio=0.4, seed=3
        )
        _assert_valid_output(corrupted, mask, blue_pil, params)

    def test_large_zone_deleted(self, blue_pil: Image.Image) -> None:
        """Le carré supprimé doit couvrir une fraction significative de l'image."""
        corrupted, mask, params = apply_large_deleted_square(
            blue_pil, size_ratio=0.5, fill_mode="black", seed=4
        )
        arr_mask = np.array(mask.convert("L"))
        ratio = np.mean(arr_mask > 0)
        assert ratio > 0.05, f"Moins de 5% de l'image supprimée ({ratio:.2%})"

    def test_fill_modes(self, blue_pil: Image.Image) -> None:
        for mode in ("black", "white", "gray", "noise"):
            corrupted, mask, params = apply_large_deleted_square(
                blue_pil, size_ratio=0.3, fill_mode=mode, seed=5
            )
            _assert_valid_output(corrupted, mask, blue_pil, params)


class TestMultipleBars:
    def test_filesystem(self, blue_image: Path) -> None:
        result = corrupt_image(
            image_path=blue_image,
            corruption_type="multiple_bars",
            count=4, seed=0,
        )
        assert Path(result["path"]).exists()
        assert result["corruption_type"] == "multiple_bars"

    def test_direct_horizontal(self, blue_pil: Image.Image) -> None:
        corrupted, mask, params = apply_multiple_bars(
            blue_pil, count=3, orientation="horizontal", seed=6
        )
        _assert_valid_output(corrupted, mask, blue_pil, params)
        assert params["count"] == 3

    def test_direct_vertical(self, blue_pil: Image.Image) -> None:
        corrupted, mask, params = apply_multiple_bars(
            blue_pil, count=2, orientation="vertical", seed=7
        )
        _assert_valid_output(corrupted, mask, blue_pil, params)

    def test_invalid_orientation(self, blue_pil: Image.Image) -> None:
        with pytest.raises(ValueError):
            apply_multiple_bars(blue_pil, orientation="diagonal")


class TestRandomHoles:
    def test_filesystem(self, blue_image: Path) -> None:
        result = corrupt_image(
            image_path=blue_image,
            corruption_type="random_holes",
            count=6, seed=0,
        )
        assert Path(result["path"]).exists()
        assert result["corruption_type"] == "random_holes"

    def test_direct_transform(self, blue_pil: Image.Image) -> None:
        corrupted, mask, params = apply_random_holes(blue_pil, count=5, seed=8)
        _assert_valid_output(corrupted, mask, blue_pil, params)
        assert len(params["holes"]) == 5

    def test_hole_coordinates_valid(self, blue_pil: Image.Image) -> None:
        w, h = blue_pil.size
        _, _, params = apply_random_holes(blue_pil, count=10, seed=9)
        for hole in params["holes"]:
            assert 0 <= hole["x"] < w
            assert 0 <= hole["y"] < h
            assert hole["width"] > 0
            assert hole["height"] > 0


class TestLocalNoise:
    def test_filesystem(self, blue_image: Path) -> None:
        result = corrupt_image(
            image_path=blue_image,
            corruption_type="local_noise",
            x=10, y=10, width=40, height=40, seed=0,
        )
        assert Path(result["path"]).exists()
        assert result["corruption_type"] == "local_noise"

    def test_gaussian(self, blue_pil: Image.Image) -> None:
        corrupted, mask, params = apply_local_noise(
            blue_pil, x=10, y=10, width=40, height=40,
            sigma=30.0, noise_type="gaussian", seed=10,
        )
        _assert_valid_output(corrupted, mask, blue_pil, params)

    def test_salt_pepper(self, blue_pil: Image.Image) -> None:
        corrupted, mask, params = apply_local_noise(
            blue_pil, x=10, y=10, width=40, height=40,
            noise_type="salt_pepper", seed=11,
        )
        _assert_valid_output(corrupted, mask, blue_pil, params)

    def test_invalid_noise_type(self, blue_pil: Image.Image) -> None:
        with pytest.raises(ValueError):
            apply_local_noise(
                blue_pil, x=0, y=0, width=30, height=30,
                noise_type="unknown",
            )


# ---------------------------------------------------------------------------
# Tests — tous les types via corrupt_image (smoke test complet)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("corruption_type,extra_kwargs", [
    ("rectangle_mask",     {"x": 10, "y": 10, "width": 30, "height": 30, "fill_value": 0}),
    ("noise",              {"x": 10, "y": 10, "width": 30, "height": 30, "sigma": 20.0}),
    ("zone_deletion",      {"x": 10, "y": 10, "width": 30, "height": 30}),
    ("combined",           {"x": 10, "y": 10, "width": 30, "height": 30}),
    ("bar",                {"orientation": "horizontal", "thickness": 8}),
    ("local_blur",         {"x": 10, "y": 10, "width": 30, "height": 30, "kernel_size": 9}),
    ("shift_region",       {"x": 10, "y": 10, "width": 30, "height": 30, "shift_x": 5}),
    ("block_dropout",      {"block_size": 16, "drop_ratio": 0.3}),
    ("jpeg_block_artifacts", {"x": 10, "y": 10, "width": 30, "height": 30}),
    ("mixed",              {}),
    ("scratch_lines",      {"count": 3}),
    ("large_deleted_square", {"size_ratio": 0.3}),
    ("multiple_bars",      {"count": 2}),
    ("random_holes",       {"count": 4}),
    ("local_noise",        {"x": 10, "y": 10, "width": 30, "height": 30}),
])
def test_all_corruption_types_smoke(
    blue_image: Path,
    corruption_type: str,
    extra_kwargs: dict,
) -> None:
    result = corrupt_image(
        image_path=blue_image,
        corruption_type=corruption_type,
        seed=42,
        **extra_kwargs,
    )
    assert Path(result["path"]).exists(), f"{corruption_type}: fichier corrompu absent"
    assert Path(result["mask_path"]).exists(), f"{corruption_type}: masque absent"
    assert result["status"] == "corrupted"
    assert result["corruption_type"] == corruption_type

    # Vérifie que l'image est réellement modifiée
    orig = np.array(Image.open(blue_image).convert("RGB"))
    corr = np.array(Image.open(result["path"]).convert("RGB"))
    assert not np.array_equal(orig, corr), f"{corruption_type}: image non modifiée"

    # Masque non vide
    m = np.array(Image.open(result["mask_path"]).convert("L"))
    assert m.max() == 255, f"{corruption_type}: masque vide"