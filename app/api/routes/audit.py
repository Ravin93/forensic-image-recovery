"""app/api/routes/audit.py — H4.
Endpoint pour consulter les logs d'audit.
"""
from fastapi import APIRouter, Query
from app.core.audit_logger import read_audit_log

router = APIRouter(prefix="/audit", tags=["audit"])

@router.get("/logs")
def get_audit_logs(limit: int = Query(default=50, ge=1, le=500)):
    """Retourne les dernières entrées du log d'audit."""
    return read_audit_log(limit=limit)