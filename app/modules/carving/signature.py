from app.core.config import JPEG_EOI, JPEG_SOI


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