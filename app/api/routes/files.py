from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

router = APIRouter(tags=["files"])

# ---------------------------------------------------------------------------
# Répertoires autorisés — SEULS ces dossiers sont accessibles via /files/serve
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ALLOWED_DIRS = {
    (_PROJECT_ROOT / "data" / "corrupted").resolve(),
    (_PROJECT_ROOT / "data" / "reconstructed").resolve(),
    (_PROJECT_ROOT / "data" / "masks").resolve(),
    (_PROJECT_ROOT / "data" / "input").resolve(),
    (_PROJECT_ROOT / "data" / "extracted").resolve(),
    (_PROJECT_ROOT / "data" / "reports").resolve(),
}

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _is_safe_path(requested: Path) -> bool:
    """Vérifie que le chemin résolu est bien dans un répertoire autorisé."""
    try:
        resolved = requested.resolve()
    except Exception:
        return False
    for allowed in _ALLOWED_DIRS:
        try:
            resolved.relative_to(allowed)
            return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# Endpoint existant — inchangé
# ---------------------------------------------------------------------------

BASE_PATHS = {
    "extracted":     "data/extracted",
    "corrupted":     "data/corrupted",
    "masks":         "data/masks",
    "reconstructed": "data/reconstructed",
}


@router.get("/files/{file_type}/{filename}")
def get_file(file_type: str, filename: str):
    if file_type not in BASE_PATHS:
        raise HTTPException(status_code=400, detail="Invalid file type")
    file_path = Path(BASE_PATHS[file_type]) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


# ---------------------------------------------------------------------------
# Ticket H2 — /files/serve sécurisé
# ---------------------------------------------------------------------------

@router.get("/files/serve")
def serve_file(
    path: str = Query(..., description="Chemin absolu vers le fichier image"),
):
    """Sert un fichier image par chemin absolu.

    Sécurité :
    - Seuls data/corrupted, data/reconstructed, data/masks, data/input,
      data/extracted, data/reports sont accessibles.
    - Path traversal (../../etc/passwd) → 403.
    - Extensions non-image → 403.
    - Liens symboliques pointant hors de data/ → 403.
    """
    requested = Path(path)

    # 1. Extension whitelist
    suffix = requested.suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=403,
            detail=f"Extension non autorisée : '{suffix}'.",
        )

    # 2. Path traversal / dossier autorisé
    if not _is_safe_path(requested):
        raise HTTPException(
            status_code=403,
            detail="Accès refusé : chemin hors des répertoires autorisés.",
        )

    # 3. Existence et fichier régulier
    resolved = requested.resolve()
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Fichier introuvable.")
    if not resolved.is_file():
        raise HTTPException(status_code=403, detail="Le chemin ne pointe pas vers un fichier.")

    # 4. Symlink sortant de data/
    if requested.is_symlink() and not _is_safe_path(resolved):
        raise HTTPException(status_code=403, detail="Lien symbolique non autorisé.")

    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return FileResponse(str(resolved), media_type=media_types.get(suffix, "application/octet-stream"))