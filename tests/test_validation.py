from pathlib import Path

import pytest
from PIL import Image

from app.core.exceptions import ValidationError
from app.modules.validation.image_checks import (
    check_image_dimensions,
    check_image_format,
    check_image_readable,
    get_image_info,
)
from app.modules.validation.verifier import verify_image


def test_check_image_readable_valid(tmp_path: Path):
    image_path = tmp_path / "valid.jpg"
    Image.new("RGB", (32, 32), color="red").save(image_path, format="JPEG")

    check_image_readable(image_path)


def test_check_image_readable_invalid(tmp_path: Path):
    bad_path = tmp_path / "fake.jpg"
    bad_path.write_bytes(b"not a real image")

    with pytest.raises(ValidationError):
        check_image_readable(bad_path)


def test_get_image_info_valid(tmp_path: Path):
    image_path = tmp_path / "valid.jpg"
    Image.new("RGB", (40, 20), color="blue").save(image_path, format="JPEG")

    info = get_image_info(image_path)
    assert info["width"] == 40
    assert info["height"] == 20
    assert info["format"] == "JPEG"


def test_check_image_dimensions_ok():
    check_image_dimensions(10, 20)


def test_check_image_dimensions_invalid():
    with pytest.raises(ValidationError):
        check_image_dimensions(0, 20)

    with pytest.raises(ValidationError):
        check_image_dimensions(10, 0)


def test_check_image_format_jpeg_ok():
    check_image_format("JPEG", allowed_formats={"JPEG", "JPG"})


def test_check_image_format_png_rejected_for_forensic():
    with pytest.raises(ValidationError):
        check_image_format("PNG", allowed_formats={"JPEG", "JPG"})


def test_check_image_format_png_allowed_for_generic():
    check_image_format("PNG", allowed_formats={"JPEG", "JPG", "PNG"})


def test_verify_valid_jpeg(tmp_path: Path):
    image_path = tmp_path / "valid.jpg"
    Image.new("RGB", (32, 32), color="red").save(image_path, format="JPEG")

    result = verify_image(image_path, allowed_formats={"JPEG", "JPG"})
    assert result["valid"] is True
    assert result["details"]["width"] == 32
    assert result["details"]["height"] == 32


def test_verify_invalid_file(tmp_path: Path):
    bad_path = tmp_path / "fake.jpg"
    bad_path.write_bytes(b"not a real image")

    result = verify_image(bad_path, allowed_formats={"JPEG", "JPG"})
    assert result["valid"] is False
    assert result["details"] is None


def test_verify_png_rejected_in_forensic_mode(tmp_path: Path):
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (32, 32), color="green").save(image_path, format="PNG")

    result = verify_image(image_path, allowed_formats={"JPEG", "JPG"})
    assert result["valid"] is False


def test_verify_png_allowed_in_generic_mode(tmp_path: Path):
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (32, 32), color="green").save(image_path, format="PNG")

    result = verify_image(image_path, allowed_formats={"JPEG", "JPG", "PNG"})
    assert result["valid"] is True