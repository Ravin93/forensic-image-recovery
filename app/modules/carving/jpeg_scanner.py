from pathlib import Path

from app.core.exceptions import CarvingError
from app.core.logger import logger
from app.modules.carving.signature import find_all_jpeg_starts


def scan_jpeg_offsets_from_bytes(content: bytes) -> list[int]:
    offsets = find_all_jpeg_starts(content)
    logger.info("Scanner buffer : %s offsets JPEG trouvés", len(offsets))
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