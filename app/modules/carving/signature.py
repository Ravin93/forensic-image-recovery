import struct

from app.core.config import (
    BMP_SIGNATURE,
    JPEG_EOI,
    JPEG_SOI,
    VALID_BMP_BPP,
    VALID_DIB_SIZES,
)


def is_jpeg_start(buffer: bytes, offset: int) -> bool:
    if offset < 0 or offset + len(JPEG_SOI) > len(buffer):
        return False
    return buffer[offset:offset + len(JPEG_SOI)] == JPEG_SOI


def is_jpeg_end(buffer: bytes, offset: int) -> bool:
    if offset < 0 or offset + len(JPEG_EOI) > len(buffer):
        return False
    return buffer[offset:offset + len(JPEG_EOI)] == JPEG_EOI


def find_all_jpeg_starts(buffer: bytes) -> list[int]:
    results: list[int] = []
    for i in range(len(buffer) - 1):
        if is_jpeg_start(buffer, i):
            results.append(i)
    return results


def find_next_jpeg_end_offset_exclusive(buffer: bytes, start_offset: int) -> int | None:
    """
    Retourne l'offset de fin EXCLUSIF du prochain marqueur EOI.
    Si EOI commence à i, on retourne i + 2.
    """
    for i in range(start_offset, len(buffer) - 1):
        if is_jpeg_end(buffer, i):
            return i + 2
    return None


# ---------------------------------------------------------------------------
# BMP : pas de footer — la taille est lue dans le header et validée.
# ---------------------------------------------------------------------------

def is_bmp_start(buffer: bytes, offset: int) -> bool:
    if offset < 0 or offset + 2 > len(buffer):
        return False
    return buffer[offset:offset + 2] == BMP_SIGNATURE


def parse_bmp_header(buffer: bytes, offset: int) -> dict | None:
    """Lit + valide un header BMP. Retourne None si incohérent (faux positif)."""
    if offset + 30 > len(buffer):
        return None
    if buffer[offset:offset + 2] != BMP_SIGNATURE:
        return None
    file_size = struct.unpack_from("<I", buffer, offset + 2)[0]
    reserved1 = struct.unpack_from("<H", buffer, offset + 6)[0]
    reserved2 = struct.unpack_from("<H", buffer, offset + 8)[0]
    pixel_off = struct.unpack_from("<I", buffer, offset + 10)[0]
    dib_size  = struct.unpack_from("<I", buffer, offset + 14)[0]
    planes    = struct.unpack_from("<H", buffer, offset + 26)[0]
    bpp       = struct.unpack_from("<H", buffer, offset + 28)[0]
    # Filtres anti-faux-positifs : champs réservés nuls, tailles connues.
    if reserved1 != 0 or reserved2 != 0:
        return None
    if dib_size not in VALID_DIB_SIZES:
        return None
    if planes != 1:
        return None
    if bpp not in VALID_BMP_BPP:
        return None
    if file_size < 14 + dib_size or offset + file_size > len(buffer):
        return None
    if pixel_off < 14 or pixel_off >= file_size:
        return None
    return {
        "start_offset": offset,
        "file_size": file_size,
        "end_offset_exclusive": offset + file_size,
        "dib_size": dib_size,
        "bpp": bpp,
    }


def find_all_bmp_headers(buffer: bytes) -> list[dict]:
    """Trouve tous les BMP valides ; saute le contenu d'un BMP identifié."""
    results: list[dict] = []
    i = 0
    n = len(buffer)
    while i < n - 1:
        if buffer[i:i + 2] == BMP_SIGNATURE:
            header = parse_bmp_header(buffer, i)
            if header is not None:
                results.append(header)
                i = header["end_offset_exclusive"]
                continue
        i += 1
    return results