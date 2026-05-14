"""app/core/file_cleanup.py — H3.

Nettoyage automatique des fichiers temporaires anciens :
- data/input/upload_*.{jpg,jpeg,png}   > 24h
- data/corrupted/*                     > 48h
- data/reconstructed/*                 > 48h
- data/masks/*                         > 48h
- data/analyses/*/                     > 7 jours (dossiers complets)

Usage manuel :
    from app.core.file_cleanup import run_cleanup
    run_cleanup()

Intégré dans FastAPI via lifespan ou appel périodique.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from app.core.logger import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# (répertoire, pattern, âge max en secondes)
_CLEANUP_RULES: list[tuple[Path, str, int]] = [
    (_PROJECT_ROOT / "data" / "input",         "upload_*",  24 * 3600),
    (_PROJECT_ROOT / "data" / "corrupted",     "*",         48 * 3600),
    (_PROJECT_ROOT / "data" / "reconstructed", "*",         48 * 3600),
    (_PROJECT_ROOT / "data" / "masks",         "*",         48 * 3600),
]
_ANALYSES_DIR = _PROJECT_ROOT / "data" / "analyses"
_ANALYSES_MAX_AGE = 7 * 24 * 3600   # 7 jours


def run_cleanup(dry_run: bool = False) -> dict[str, int]:
    """Supprime les fichiers anciens selon les règles définies.

    Args:
        dry_run: si True, liste sans supprimer (pour audit/test)

    Returns:
        dict avec le nombre de fichiers/dossiers supprimés par catégorie
    """
    now = time.time()
    stats: dict[str, int] = {}

    # Fichiers individuels
    for directory, pattern, max_age in _CLEANUP_RULES:
        if not directory.exists():
            continue
        key = directory.name
        count = 0
        for p in directory.glob(pattern):
            if not p.is_file():
                continue
            age = now - p.stat().st_mtime
            if age > max_age:
                if not dry_run:
                    try:
                        p.unlink()
                        logger.debug("Cleanup supprimé : %s (age=%.0fs)", p.name, age)
                    except OSError as exc:
                        logger.warning("Cleanup erreur suppression %s : %s", p, exc)
                count += 1
        if count:
            logger.info("Cleanup %s : %d fichier(s) %s", key, count, "(dry_run)" if dry_run else "supprimé(s)")
        stats[key] = count

    # Dossiers d'analyses
    count_analyses = 0
    if _ANALYSES_DIR.exists():
        for d in _ANALYSES_DIR.iterdir():
            if not d.is_dir():
                continue
            age = now - d.stat().st_mtime
            if age > _ANALYSES_MAX_AGE:
                if not dry_run:
                    try:
                        shutil.rmtree(str(d))
                        logger.debug("Cleanup analyse supprimée : %s (age=%.0fs)", d.name, age)
                    except OSError as exc:
                        logger.warning("Cleanup erreur suppression analyse %s : %s", d, exc)
                count_analyses += 1
    stats["analyses"] = count_analyses

    total = sum(stats.values())
    logger.info("Cleanup terminé : %d élément(s) %s", total, "(dry_run)" if dry_run else "supprimé(s)")
    return stats


def schedule_cleanup_on_startup(app) -> None:
    """Branche le nettoyage automatique sur le démarrage FastAPI via lifespan."""
    from contextlib import asynccontextmanager

    original_lifespan = getattr(app, "router", app).lifespan_context

    @asynccontextmanager
    async def _lifespan(app):
        # Cleanup au démarrage
        try:
            stats = run_cleanup()
            logger.info("Cleanup démarrage : %s", stats)
        except Exception as exc:
            logger.warning("Cleanup démarrage échoué : %s", exc)
        # Déléguer au lifespan original si présent
        if original_lifespan is not None:
            async with original_lifespan(app):
                yield
        else:
            yield

    app.router.lifespan_context = _lifespan