from pathlib import Path
from typing import Any

from app.core.config import (
    DEFAULT_MIN_JPEG_SIZE,
    EXTRACTED_DIR,
    build_extracted_image_path,
    build_timestamp,
    ensure_directories,
)
from app.core.exceptions import CarvingError
from app.core.logger import logger
from app.modules.carving.jpeg_scanner import scan_jpeg_offsets_from_bytes, scan_png_offsets_from_bytes
from app.modules.carving.signature import find_all_bmp_headers, find_next_jpeg_end_offset_exclusive

PNG_END_SIGNATURE = b"IEND\xaeB`\x82"


def extract_jpegs_from_dump(file_path: str | Path) -> list[dict[str, Any]]:
    ensure_directories()
    path = Path(file_path)

    if not path.exists():
        raise CarvingError(f"Dump introuvable : {path}")

    try:
        with path.open("rb") as f:
            content = f.read()
    except OSError as exc:
        raise CarvingError(f"Impossible de lire le dump : {path}") from exc

    starts = scan_jpeg_offsets_from_bytes(content)
    extracted: list[dict[str, Any]] = []

    for idx, start_offset in enumerate(starts, start=1):
        end_offset_exclusive = find_next_jpeg_end_offset_exclusive(content, start_offset + 2)

        if end_offset_exclusive is None:
            logger.info("Aucune fin JPEG trouvée pour start_offset=%s", start_offset)
            continue

        jpeg_bytes = content[start_offset:end_offset_exclusive]
        size = end_offset_exclusive - start_offset

        if size < DEFAULT_MIN_JPEG_SIZE:
            logger.info(
                "Extraction ignorée : taille trop petite (%s octets) à offset %s",
                size,
                start_offset,
            )
            continue

        output_path = build_extracted_image_path(path.name, idx)

        try:
            with output_path.open("wb") as out:
                out.write(jpeg_bytes)
        except OSError as exc:
            raise CarvingError(f"Impossible d'écrire le fichier extrait : {output_path}") from exc

        item = {
            "index": idx,
            "source_dump": path.name,
            "file": output_path.name,
            "path": str(output_path),
            "start_offset": start_offset,
            "end_offset_exclusive": end_offset_exclusive,
            "size": size,
            "status": "extracted",
        }
        extracted.append(item)

        logger.info(
            "Image extraite : %s | start_offset=%s | end_offset_exclusive=%s | size=%s",
            output_path.name,
            start_offset,
            end_offset_exclusive,
            size,
        )

    return extracted


def extract_pngs_from_dump(file_path: str | Path) -> list[dict[str, Any]]:
    ensure_directories()
    path = Path(file_path)

    if not path.exists():
        raise CarvingError(f"Dump introuvable : {path}")

    try:
        with path.open("rb") as f:
            content = f.read()
    except OSError as exc:
        raise CarvingError(f"Impossible de lire le dump : {path}") from exc

    starts = scan_png_offsets_from_bytes(content)
    extracted: list[dict[str, Any]] = []

    for idx, start_offset in enumerate(starts, start=1):
        end_signature_offset = content.find(PNG_END_SIGNATURE, start_offset)
        if end_signature_offset < 0:
            logger.info("Aucune fin PNG trouvée pour start_offset=%s", start_offset)
            continue

        end_offset_exclusive = end_signature_offset + len(PNG_END_SIGNATURE)
        png_bytes = content[start_offset:end_offset_exclusive]
        size = end_offset_exclusive - start_offset

        if size < len(PNG_END_SIGNATURE):
            logger.info(
                "Extraction PNG ignorée : taille trop petite (%s octets) à offset %s",
                size,
                start_offset,
            )
            continue

        output_path = EXTRACTED_DIR / f"{path.stem}_{build_timestamp()}_img_{idx:03d}.png"

        try:
            with output_path.open("wb") as out:
                out.write(png_bytes)
        except OSError as exc:
            raise CarvingError(f"Impossible d'écrire le fichier extrait : {output_path}") from exc

        item = {
            "index": idx,
            "source_dump": path.name,
            "file": output_path.name,
            "path": str(output_path),
            "start_offset": start_offset,
            "end_offset_exclusive": end_offset_exclusive,
            "size": size,
            "status": "extracted",
            "format": "png",
        }
        extracted.append(item)

        logger.info(
            "PNG extrait : %s | start_offset=%s | end_offset_exclusive=%s | size=%s",
            output_path.name,
            start_offset,
            end_offset_exclusive,
            size,
        )

    return extracted


def extract_bmps_from_dump(file_path: str | Path) -> list[dict[str, Any]]:
    ensure_directories()
    path = Path(file_path)

    if not path.exists():
        raise CarvingError(f"Dump introuvable : {path}")

    try:
        with path.open("rb") as f:
            content = f.read()
    except OSError as exc:
        raise CarvingError(f"Impossible de lire le dump : {path}") from exc

    # BMP n'a pas de footer : la taille est lue et validée dans le header.
    headers = find_all_bmp_headers(content)
    extracted: list[dict[str, Any]] = []

    for idx, header in enumerate(headers, start=1):
        start_offset = header["start_offset"]
        end_offset_exclusive = header["end_offset_exclusive"]
        bmp_bytes = content[start_offset:end_offset_exclusive]
        size = end_offset_exclusive - start_offset

        output_path = EXTRACTED_DIR / f"{path.stem}_{build_timestamp()}_img_{idx:03d}.bmp"

        try:
            with output_path.open("wb") as out:
                out.write(bmp_bytes)
        except OSError as exc:
            raise CarvingError(f"Impossible d'écrire le fichier extrait : {output_path}") from exc

        item = {
            "index": idx,
            "source_dump": path.name,
            "file": output_path.name,
            "path": str(output_path),
            "start_offset": start_offset,
            "end_offset_exclusive": end_offset_exclusive,
            "size": size,
            "status": "extracted",
            "format": "bmp",
        }
        extracted.append(item)

        logger.info(
            "BMP extrait : %s | start_offset=%s | end_offset_exclusive=%s | size=%s",
            output_path.name,
            start_offset,
            end_offset_exclusive,
            size,
        )

    return extracted
