from pathlib import Path

import pytest
from PIL import Image

from app.core.exceptions import CorruptionError
from app.modules.corruption.simulator import corrupt_image


def test_corrupt_image_rectangle_mask(tmp_path: Path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (128, 128), color="blue").save(image_path)

    result = corrupt_image(
        image_path=image_path,
        corruption_type="rectangle_mask",
        x=20,
        y=20,
        width=40,
        height=40,
        fill_value=0,
    )

    assert Path(result["path"]).exists()
    assert Path(result["mask_path"]).exists()
    assert result["status"] == "corrupted"
    assert result["corruption_type"] == "rectangle_mask"


def test_corrupt_image_noise(tmp_path: Path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (128, 128), color="blue").save(image_path)

    result = corrupt_image(
        image_path=image_path,
        corruption_type="noise",
        x=10,
        y=10,
        width=30,
        height=30,
        sigma=20.0,
        seed=42,
    )

    assert Path(result["path"]).exists()
    assert Path(result["mask_path"]).exists()
    assert result["corruption_type"] == "noise"


def test_corrupt_image_zone_deletion(tmp_path: Path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (128, 128), color="blue").save(image_path)

    result = corrupt_image(
        image_path=image_path,
        corruption_type="zone_deletion",
        x=15,
        y=15,
        width=20,
        height=20,
    )

    assert Path(result["path"]).exists()
    assert Path(result["mask_path"]).exists()
    assert result["corruption_type"] == "zone_deletion"


def test_corrupt_missing_source():
    with pytest.raises(CorruptionError):
        corrupt_image(
            image_path="missing.png",
            corruption_type="rectangle_mask",
            x=10,
            y=10,
            width=20,
            height=20,
            fill_value=0,
        )


def test_corrupt_unsupported_type(tmp_path: Path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (128, 128), color="blue").save(image_path)

    with pytest.raises(CorruptionError):
        corrupt_image(
            image_path=image_path,
            corruption_type="unsupported_mode",
        )