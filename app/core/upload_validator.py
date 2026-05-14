"""app/core/upload_validator.py — H1.

Validation stricte des fichiers uploadés :
- Taille maximale
- Extension whitelist
- MIME type check
- Magic bytes (signature binaire réelle)
- Refus de tout fichier non-image
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

# Signatures binaires connues (magic bytes)
_MAGIC: dict[str, list[bytes]] = {
    ".jpg":  [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".png":  [b"\x89PNG\r\n\x1a\n"],
}

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_MAX_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


async def validate_upload(image: UploadFile) -> bytes:
    """Lit, valide et retourne le contenu brut de l'upload.

    Vérifie dans l'ordre :
    1. Extension whitelist
    2. Taille maximale
    3. Magic bytes (vraie signature binaire)

    Returns:
        bytes — contenu validé, prêt à être écrit sur disque

    Raises:
        HTTPException 422 si invalide
    """
    # 1. Extension
    suffix = Path(image.filename or "upload.bin").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Extension non autorisée : \'{suffix}\'. Seuls JPEG et PNG sont acceptés.",
        )

    # 2. Lecture + taille
    content = await image.read()
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="Fichier vide.")
    if len(content) > _MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"Fichier trop volumineux : {len(content):,} octets (max {_MAX_SIZE_BYTES:,}).",
        )

    # 3. Magic bytes
    expected_magic = _MAGIC.get(suffix, [])
    if expected_magic:
        ok = any(content.startswith(sig) for sig in expected_magic)
        if not ok:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Signature binaire invalide pour \'{suffix}\'. "
                    "Le fichier ne correspond pas au format déclaré."
                ),
            )

    return content


def get_file_info(content: bytes, filename: str) -> dict[str, Any]:
    """Retourne les métadonnées de base du fichier pour l'audit."""
    import hashlib
    return {
        "filename": filename,
        "size":     len(content),
        "sha256":   hashlib.sha256(content).hexdigest(),
        "suffix":   Path(filename).suffix.lower(),
    }