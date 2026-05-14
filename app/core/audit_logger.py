"""app/core/audit_logger.py — H4.

Logging d'audit complet pour chaque requête API :
- request_id  (UUID court)
- IP client
- filename + SHA-256
- corruption_type
- processing_time
- status (success/error)
- HTTP status code

Les entrées sont écrites dans data/logs/audit.jsonl (une ligne JSON par requête).
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from app.core.logger import logger

_LOGS_DIR = Path(__file__).resolve().parents[2] / "data" / "logs"
_AUDIT_FILE = _LOGS_DIR / "audit.jsonl"


def _ensure_log_dir() -> None:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)


def new_request_id() -> str:
    return uuid.uuid4().hex[:8]


def log_audit_entry(
    request_id: str,
    ip: str,
    endpoint: str,
    filename: str | None,
    sha256: str | None,
    corruption_type: str | None,
    processing_time_s: float | None,
    status: str,
    http_status: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Écrit une entrée d'audit dans data/logs/audit.jsonl.

    Chaque entrée est une ligne JSON indépendante (format JSONL).
    """
    _ensure_log_dir()
    entry = {
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "request_id":      request_id,
        "ip":              ip,
        "endpoint":        endpoint,
        "filename":        filename,
        "sha256":          sha256,
        "corruption_type": corruption_type,
        "processing_time_s": round(processing_time_s, 3) if processing_time_s is not None else None,
        "status":          status,
        "http_status":     http_status,
    }
    if extra:
        entry.update(extra)

    try:
        with _AUDIT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as exc:
        logger.warning("Audit log erreur écriture : %s", exc)

    logger.info(
        "AUDIT | req=%s ip=%s ep=%s file=%s status=%s time=%.3fs",
        request_id, ip, endpoint, filename or "—",
        status, processing_time_s or 0,
    )


def read_audit_log(limit: int = 100) -> list[dict[str, Any]]:
    """Retourne les dernières entrées du log d'audit."""
    if not _AUDIT_FILE.exists():
        return []
    lines = _AUDIT_FILE.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in reversed(lines[-limit:]):
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return entries