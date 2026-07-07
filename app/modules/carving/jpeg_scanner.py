from pathlib import Path

from app.core.exceptions import CarvingError
from app.core.logger import logger
from app.modules.carving.signature import find_all_jpeg_starts

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def scan_jpeg_offsets_from_bytes(content: bytes) -> list[int]:
    offsets = find_all_jpeg_starts(content)
    logger.info("Scanner buffer : %s offsets JPEG trouvés", len(offsets))
    return offsets


def scan_png_offsets_from_bytes(content: bytes) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        offset = content.find(PNG_SIGNATURE, start)
        if offset < 0:
            break
        offsets.append(offset)
        start = offset + len(PNG_SIGNATURE)
    logger.info("Scanner buffer : %s offsets PNG trouvés", len(offsets))
    return offsets


def scan_jpeg_offsets(file_path: str | Path) -> list[int]:
    path = Path(file_path)

    if not path.exists():
        raise CarvingError(f"Fichier introuvable : {path}")

    try:
        with path.open("rb") as f:
            content = f.read()
    except OSError as exc:
        raise CarvingError(f"Impossible de lire le dump : {path}") from exc

    offsets = scan_jpeg_offsets_from_bytes(content)
    logger.info("Scanner fichier : %s offsets trouvés dans %s", len(offsets), path)
    return offsets


def scan_png_offsets(file_path: str | Path) -> list[int]:
    path = Path(file_path)

    if not path.exists():
        raise CarvingError(f"Fichier introuvable : {path}")

    try:
        with path.open("rb") as f:
            content = f.read()
    except OSError as exc:
        raise CarvingError(f"Impossible de lire le dump : {path}") from exc

    offsets = scan_png_offsets_from_bytes(content)
    logger.info("Scanner fichier : %s offsets PNG trouvés dans %s", len(offsets), path)
    return offsets
