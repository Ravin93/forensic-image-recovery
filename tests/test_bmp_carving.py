import os
from pathlib import Path

import pytest
from PIL import Image

from app.core.config import BMP_SIGNATURE
from app.modules.carving.extractor import extract_bmps_from_dump
from app.modules.carving.jpeg_scanner import (
    scan_bmp_offsets,
    scan_bmp_offsets_from_bytes,
)
from app.modules.carving.signature import (
    find_all_bmp_headers,
    is_bmp_start,
    parse_bmp_header,
)
from app.core.exceptions import CarvingError


def _make_bmp_bytes(color=(120, 200, 90), size=(10, 10)) -> bytes:
    """Génère un vrai BMP via Pillow et retourne ses octets."""
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="BMP")
    return buffer.getvalue()


def test_is_bmp_start():
    data = b"\x00\x11" + BMP_SIGNATURE + b"\x22"
    assert is_bmp_start(data, 2) is True
    assert is_bmp_start(data, 1) is False
    assert is_bmp_start(data, 99) is False
    assert is_bmp_start(data, -1) is False


def test_parse_bmp_header_valid():
    bmp = _make_bmp_bytes()
    header = parse_bmp_header(bmp, 0)
    assert header is not None
    assert header["start_offset"] == 0
    assert header["file_size"] == len(bmp)
    assert header["end_offset_exclusive"] == len(bmp)
    assert header["dib_size"] in {12, 40, 52, 56, 64, 108, 124}
    assert header["bpp"] in {1, 4, 8, 16, 24, 32}


def test_parse_bmp_header_rejects_false_positive():
    # "BM" suivi d'octets aléatoires : header incohérent -> None
    lure = BMP_SIGNATURE + os.urandom(40)
    assert parse_bmp_header(lure, 0) is None


def test_parse_bmp_header_rejects_truncated():
    # Trop court pour lire tout le header
    assert parse_bmp_header(BMP_SIGNATURE + b"\x00" * 5, 0) is None


def test_find_all_bmp_headers_skips_content_and_lures():
    bmp = _make_bmp_bytes()
    lure = BMP_SIGNATURE + os.urandom(40)
    dump = os.urandom(64) + lure + os.urandom(64) + bmp + os.urandom(32)

    headers = find_all_bmp_headers(dump)
    # Le leurre est rejeté, seul le vrai BMP est trouvé
    assert len(headers) == 1
    assert headers[0]["file_size"] == len(bmp)


def test_scan_bmp_offsets_from_bytes_rejects_lure():
    bmp = _make_bmp_bytes()
    lure = BMP_SIGNATURE + os.urandom(40)
    prefix = os.urandom(50)
    dump = prefix + lure + os.urandom(50) + bmp + os.urandom(20)

    offsets = scan_bmp_offsets_from_bytes(dump)
    assert offsets == [len(prefix) + len(lure) + 50]


def test_scan_bmp_offsets_file(tmp_path: Path):
    bmp = _make_bmp_bytes()
    dump_path = tmp_path / "dump.bin"
    dump_path.write_bytes(b"\x00\x01\x02" + bmp + b"\x99\x88")

    offsets = scan_bmp_offsets(dump_path)
    assert offsets == [3]


def test_scan_bmp_offsets_missing_file():
    with pytest.raises(CarvingError):
        scan_bmp_offsets("missing_dump.bin")


def test_extract_simple_bmp_from_dump(tmp_path: Path):
    bmp = _make_bmp_bytes(color=(120, 200, 90), size=(10, 10))
    lure = BMP_SIGNATURE + os.urandom(40)

    dump_path = tmp_path / "dump.bin"
    dump_path.write_bytes(os.urandom(30) + lure + os.urandom(30) + bmp + os.urandom(15))

    extracted = extract_bmps_from_dump(dump_path)

    # Un seul BMP valide : le leurre est ignoré
    assert len(extracted) == 1
    item = extracted[0]
    assert item["status"] == "extracted"
    assert item["source_dump"] == "dump.bin"
    assert item["format"] == "bmp"
    assert item["start_offset"] >= 0
    assert item["end_offset_exclusive"] > item["start_offset"]

    carved_path = Path(item["path"])
    assert carved_path.exists()

    # Le BMP carvé se rouvre avec Pillow et est identique à l'original
    with Image.open(carved_path) as carved:
        assert carved.format == "BMP"
        assert carved.size == (10, 10)
        assert carved.convert("RGB").getpixel((5, 5)) == (120, 200, 90)
    assert carved_path.read_bytes() == bmp


def test_extract_dump_without_bmp(tmp_path: Path):
    dump_path = tmp_path / "dump.bin"
    dump_path.write_bytes(b"\x00\x11\x22\x33\x44\x55")

    assert extract_bmps_from_dump(dump_path) == []


def test_extract_dump_with_bmp_signature_but_invalid_header(tmp_path: Path):
    dump_path = tmp_path / "dump.bin"
    dump_path.write_bytes(BMP_SIGNATURE + os.urandom(40))

    assert extract_bmps_from_dump(dump_path) == []


def test_extract_missing_bmp_dump():
    with pytest.raises(CarvingError):
        extract_bmps_from_dump("missing_dump.bin")
