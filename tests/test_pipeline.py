from pathlib import Path

import pytest
from PIL import Image

from app.services.pipeline_service import run_demo_pipeline


def test_run_demo_pipeline_png(tmp_path: Path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (128, 128), color="green").save(image_path)

    result = run_demo_pipeline(
        source_image_path=image_path,
        corruption_type="rectangle_mask",
        corruption_params={
            "x": 10,
            "y": 10,
            "width": 30,
            "height": 30,
            "fill_value": 0,
        },
        inpaint_method="opencv_inpaint",
        radius=3,
    )

    assert result["status"] == "completed"
    assert Path(result["corruption"]["path"]).exists()
    assert Path(result["corruption"]["mask_path"]).exists()
    assert Path(result["reconstruction"]["path"]).exists()
    assert Path(result["report_path"]).exists()


def test_run_demo_pipeline_jpeg(tmp_path: Path):
    image_path = tmp_path / "source.jpg"
    Image.new("RGB", (128, 128), color="yellow").save(image_path, format="JPEG")

    result = run_demo_pipeline(
        source_image_path=image_path,
        corruption_type="noise",
        corruption_params={
            "x": 15,
            "y": 15,
            "width": 25,
            "height": 25,
            "sigma": 15.0,
            "seed": 42,
        },
        inpaint_method="opencv_inpaint",
        radius=3,
    )

    assert result["status"] == "completed"
    assert "evaluation" in result


def test_run_demo_pipeline_invalid_source(tmp_path: Path):
    bad_path = tmp_path / "fake.jpg"
    bad_path.write_bytes(b"not an image")

    with pytest.raises(ValueError):
        run_demo_pipeline(
            source_image_path=bad_path,
            corruption_type="rectangle_mask",
            corruption_params={
                "x": 10,
                "y": 10,
                "width": 30,
                "height": 30,
                "fill_value": 0,
            },
        )


def test_run_demo_pipeline_blind_advanced(tmp_path: Path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (128, 128), color="blue").save(image_path)

    result = run_demo_pipeline(
        source_image_path=image_path,
        corruption_level=60,
        execution_mode="blind_advanced",
        detection_mode="advanced",
        seed=42,
    )

    assert result["status"] == "completed"
    assert "detection_metrics" in result
    assert "recoverability_status" in result