from pathlib import Path

import pytest
from PIL import Image

from app.core.exceptions import ReconstructionError
from app.modules.corruption.simulator import corrupt_image
from app.modules.reconstruction.repair_pipeline import run_repair_pipeline


def test_reconstruction_pipeline(tmp_path: Path):
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

    result = run_repair_pipeline(
        corrupted_image_path=corrupted["path"],
        mask_path=corrupted["mask_path"],
    )

    assert Path(result["path"]).exists()
    assert result["status"] == "reconstructed"


def test_reconstruction_missing_corrupted_image(tmp_path: Path):
    mask_path = tmp_path / "mask.png"
    Image.new("L", (64, 64), color=0).save(mask_path)

    with pytest.raises(ReconstructionError):
        run_repair_pipeline(
            corrupted_image_path="missing.png",
            mask_path=mask_path,
        )


def test_reconstruction_missing_mask(tmp_path: Path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (64, 64), color="blue").save(image_path)

    with pytest.raises(ReconstructionError):
        run_repair_pipeline(
            corrupted_image_path=image_path,
            mask_path="missing_mask.png",
        )


def test_reconstruction_unsupported_method(tmp_path: Path):
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

    with pytest.raises(ReconstructionError):
        run_repair_pipeline(
            corrupted_image_path=corrupted["path"],
            mask_path=corrupted["mask_path"],
            method="unknown_method",
        )