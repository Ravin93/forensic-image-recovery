from pathlib import Path

import pytest
from PIL import Image

from app.core.exceptions import EvaluationError
from app.modules.corruption.simulator import corrupt_image
from app.modules.evaluation.comparator import compare_images
from app.modules.evaluation.metrics import (
    compute_psnr,
    compute_ssim,
    ensure_same_shape,
    load_image_as_rgb_array,
)


def test_load_image_as_rgb_array(tmp_path: Path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (64, 64), color="red").save(image_path)

    arr = load_image_as_rgb_array(image_path)
    assert arr.shape == (64, 64, 3)


def test_load_image_as_rgb_array_missing():
    with pytest.raises(EvaluationError):
        load_image_as_rgb_array("missing.png")


def test_ensure_same_shape_ok(tmp_path: Path):
    image1 = tmp_path / "a.png"
    image2 = tmp_path / "b.png"
    Image.new("RGB", (32, 32), color="red").save(image1)
    Image.new("RGB", (32, 32), color="blue").save(image2)

    arr1 = load_image_as_rgb_array(image1)
    arr2 = load_image_as_rgb_array(image2)
    ensure_same_shape(arr1, arr2)


def test_ensure_same_shape_invalid(tmp_path: Path):
    image1 = tmp_path / "a.png"
    image2 = tmp_path / "b.png"
    Image.new("RGB", (32, 32), color="red").save(image1)
    Image.new("RGB", (64, 64), color="blue").save(image2)

    arr1 = load_image_as_rgb_array(image1)
    arr2 = load_image_as_rgb_array(image2)

    with pytest.raises(EvaluationError):
        ensure_same_shape(arr1, arr2)


def test_compute_psnr_and_ssim(tmp_path: Path):
    image1 = tmp_path / "a.png"
    image2 = tmp_path / "b.png"
    Image.new("RGB", (64, 64), color="red").save(image1)
    Image.new("RGB", (64, 64), color="red").save(image2)

    arr1 = load_image_as_rgb_array(image1)
    arr2 = load_image_as_rgb_array(image2)

    psnr = compute_psnr(arr1, arr2)
    ssim = compute_ssim(arr1, arr2)

    assert isinstance(psnr, float)
    assert isinstance(ssim, float)
    assert ssim == pytest.approx(1.0, rel=1e-6)


def test_compare_images(tmp_path: Path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (128, 128), color="blue").save(image_path)

    corrupted = corrupt_image(
        image_path=image_path,
        corruption_type="rectangle_mask",
        x=20,
        y=20,
        width=40,
        height=40,
        fill_value=0,
    )

    result = compare_images(
        original_path=image_path,
        corrupted_path=corrupted["path"],
        reconstructed_path=corrupted["path"],
    )

    assert "original_vs_corrupted" in result
    assert "original_vs_reconstructed" in result
    assert "psnr" in result["original_vs_corrupted"]
    assert "ssim" in result["original_vs_corrupted"]


def test_compare_images_different_sizes(tmp_path: Path):
    original = tmp_path / "original.png"
    corrupted = tmp_path / "corrupted.png"
    reconstructed = tmp_path / "reconstructed.png"

    Image.new("RGB", (128, 128), color="red").save(original)
    Image.new("RGB", (64, 64), color="red").save(corrupted)
    Image.new("RGB", (128, 128), color="red").save(reconstructed)

    with pytest.raises(EvaluationError):
        compare_images(original, corrupted, reconstructed)