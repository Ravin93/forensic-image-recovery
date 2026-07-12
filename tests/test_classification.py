from pathlib import Path

from PIL import Image

from app.modules.classification.corruption_classifier import classify_corruption_type


def test_classify_corruption_type(tmp_path: Path):
    img_path = tmp_path / "test.png"
    Image.new("RGB", (64, 64), color="black").save(img_path)

    result = classify_corruption_type(img_path)
    assert "corruption_type" in result
    assert "confidence" in result