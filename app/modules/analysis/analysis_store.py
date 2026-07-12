"""app/modules/analysis/analysis_store.py — I1+I2.

Stockage persistant des analyses dans data/analyses/{analysis_id}/
avec gestion des statuts pending/running/completed/failed.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from app.core.config import ensure_directories
from app.core.logger import logger

_ANALYSES_DIR = Path(__file__).resolve().parents[3] / "data" / "analyses"


def _analyses_dir() -> Path:
    _ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    return _ANALYSES_DIR


def new_analysis_id() -> str:
    """Génère un ID unique pour une analyse."""
    return uuid.uuid4().hex[:12]


def analysis_dir(analysis_id: str) -> Path:
    return _analyses_dir() / analysis_id


def create_analysis(
    analysis_id: str,
    request_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Crée le répertoire et le fichier status.json initial (status=pending)."""
    adir = analysis_dir(analysis_id)
    adir.mkdir(parents=True, exist_ok=True)

    status = {
        "analysis_id":  analysis_id,
        "status":        "pending",
        "created_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "request":       request_info or {},
        "files":         {},
        "error":         None,
    }
    _write_status(analysis_id, status)
    logger.info("Analysis créée : %s", analysis_id)
    return status


def update_status(
    analysis_id: str,
    status: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Met à jour le statut d'une analyse (running/completed/failed)."""
    current = get_status(analysis_id)
    current["status"]     = status
    current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if extra:
        current.update(extra)
    _write_status(analysis_id, current)
    return current


def update_progress(
    analysis_id: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Met à jour les champs de progression sans changer le statut courant."""
    current = get_status(analysis_id)
    current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    current.update(extra)
    _write_status(analysis_id, current)
    return current


def record_strategy_completed(
    analysis_id: str,
    strategy: str,
    score: float | None,
    elapsed_s: float,
) -> dict[str, Any]:
    """Ajoute une stratégie complétée au status.json d'une analyse."""
    current = get_status(analysis_id)
    history = list(current.get("strategy_completed_events") or [])
    event = {
        "phase": "strategy_completed",
        "strategy": strategy,
        "score": score,
        "elapsed_s": elapsed_s,
    }
    history.append(event)
    current["strategy_completed_events"] = history
    current["strategies_completed"] = len(history)
    current["last_strategy"] = strategy
    current["last_score"] = score
    current["elapsed_s"] = elapsed_s
    current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_status(analysis_id, current)
    return current


def get_status(analysis_id: str) -> dict[str, Any]:
    """Retourne le statut courant d'une analyse."""
    path = analysis_dir(analysis_id) / "status.json"
    if not path.exists():
        raise FileNotFoundError(f"Analyse introuvable : {analysis_id}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_analysis_file(
    analysis_id: str,
    key: str,
    src_path: str | Path,
    filename: str | None = None,
) -> str:
    """Copie un fichier dans data/analyses/{id}/ et enregistre son chemin."""
    src = Path(src_path)
    if not src.exists():
        return ""
    dest_name = filename or src.name
    dest = analysis_dir(analysis_id) / dest_name
    shutil.copy2(str(src), str(dest))

    current = get_status(analysis_id)
    current["files"][key] = str(dest)
    current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_status(analysis_id, current)
    return str(dest)


def save_result(analysis_id: str, result: dict[str, Any]) -> None:
    """Sauvegarde le résultat complet en JSON."""
    out = analysis_dir(analysis_id) / "result.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    current = get_status(analysis_id)
    current["files"]["result"] = str(out)
    current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_status(analysis_id, current)


def get_result(analysis_id: str) -> dict[str, Any]:
    """Retourne le résultat complet d'une analyse terminée."""
    path = analysis_dir(analysis_id) / "result.json"
    if not path.exists():
        raise FileNotFoundError(f"Résultat non disponible pour : {analysis_id}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def list_analyses(limit: int = 50) -> list[dict[str, Any]]:
    """Retourne les dernières analyses triées par date décroissante."""
    base = _analyses_dir()
    statuses = []
    for d in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "status.json").exists():
            try:
                statuses.append(get_status(d.name))
            except Exception:
                pass
        if len(statuses) >= limit:
            break
    return statuses


def delete_analysis(analysis_id: str) -> bool:
    """Supprime une analyse et tous ses fichiers."""
    adir = analysis_dir(analysis_id)
    if not adir.exists():
        return False
    shutil.rmtree(str(adir))
    logger.info("Analysis supprimée : %s", analysis_id)
    return True


def _write_status(analysis_id: str, status: dict[str, Any]) -> None:
    path = analysis_dir(analysis_id) / "status.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False, default=str)
