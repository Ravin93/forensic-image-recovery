from pathlib import Path

import pytest
from PIL import Image

from app.core.exceptions import CarvingError
from app.modules.carving.extractor import extract_jpegs_from_dump
from app.modules.carving.jpeg_scanner import (
    scan_jpeg_offsets,
    scan_jpeg_offsets_from_bytes,
)
from app.modules.carving.signature import (
    find_all_jpeg_starts,
    find_next_jpeg_end_offset_exclusive,
    is_jpeg_end,
    is_jpeg_start,
)


def test_is_jpeg_start():
    data = b"\x00\x11\xFF\xD8\x22"
    assert is_jpeg_start(data, 2) is True
    assert is_jpeg_start(data, 1) is False
    assert is_jpeg_start(data, 99) is False


def test_is_jpeg_end():
    data = b"\x00\xFF\xD9\x11"
    assert is_jpeg_end(data, 1) is True
    assert is_jpeg_end(data, 0) is False
    assert is_jpeg_end(data, 99) is False


def test_find_all_jpeg_starts():
    data = b"\xFF\xD8\x00\xFF\xD8\x11"
    assert find_all_jpeg_starts(data) == [0, 3]


def test_find_next_jpeg_end_offset_exclusive():
    data = b"\xAA\xFF\xD8\x01\x02\xFF\xD9\xBB"
    assert find_next_jpeg_end_offset_exclusive(data, 2) == 7


def test_find_next_jpeg_end_offset_exclusive_none():
    data = b"\xAA\xFF\xD8\x01\x02\x03\x04"
    assert find_next_jpeg_end_offset_exclusive(data, 2) is None


def test_scan_jpeg_offsets_from_bytes():
    content = b"\x00\xFF\xD8\x00\x11\xFF\xD8\xFF\xD9"
    offsets = scan_jpeg_offsets_from_bytes(content)
    assert offsets == [1, 5]


def test_scan_jpeg_offsets_file(tmp_path: Path):
    dump_path = tmp_path / "dump.bin"
    dump_path.write_bytes(b"\x00\xFF\xD8\x00\x11\xFF\xD8\xFF\xD9")

    offsets = scan_jpeg_offsets(dump_path)
    assert offsets == [1, 5]


def test_scan_jpeg_offsets_missing_file():
    with pytest.raises(CarvingError):
        scan_jpeg_offsets("missing_dump.bin")


def test_extract_simple_jpeg_from_dump(tmp_path: Path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (64, 64), color="red").save(image_path, format="JPEG")
    jpeg_bytes = image_path.read_bytes()

    dump_path = tmp_path / "dump.bin"
    dump_path.write_bytes(b"\x00\x01\x02" + jpeg_bytes + b"\x99\x88")

    extracted = extract_jpegs_from_dump(dump_path)

    assert len(extracted) >= 1
    assert extracted[0]["status"] == "extracted"
    assert extracted[0]["source_dump"] == "dump.bin"
    assert extracted[0]["start_offset"] >= 0
    assert extracted[0]["end_offset_exclusive"] > extracted[0]["start_offset"]
    assert Path(extracted[0]["path"]).exists()


def test_extract_dump_without_jpeg(tmp_path: Path):
    dump_path = tmp_path / "dump.bin"
    dump_path.write_bytes(b"\x00\x11\x22\x33\x44\x55")

    extracted = extract_jpegs_from_dump(dump_path)
    assert extracted == []


def test_extract_dump_with_soi_but_no_eoi(tmp_path: Path):
    dump_path = tmp_path / "dump.bin"
    dump_path.write_bytes(b"\x00\xFF\xD8\x11\x22\x33\x44")

    extracted = extract_jpegs_from_dump(dump_path)
    assert extracted == []


def test_extract_missing_dump():
    with pytest.raises(CarvingError):
        extract_jpegs_from_dump("missing_dump.bin")