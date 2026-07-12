"""Tests Ticket H2 — sécurité /files/serve."""
from __future__ import annotations
import io
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_path_traversal_etc_passwd():
    resp = client.get("/files/serve", params={"path": "/etc/passwd"})
    assert resp.status_code == 403, f"Path traversal non bloqué : {resp.status_code}"


def test_path_traversal_relative():
    resp = client.get("/files/serve", params={"path": "../../../../etc/passwd"})
    assert resp.status_code == 403


def test_path_traversal_encoded():
    resp = client.get("/files/serve?path=%2Fetc%2Fpasswd")
    assert resp.status_code == 403


def test_forbidden_extension_py():
    resp = client.get("/files/serve", params={"path": "/data/corrupted/script.py"})
    assert resp.status_code == 403


def test_forbidden_extension_sh():
    resp = client.get("/files/serve", params={"path": "/data/corrupted/run.sh"})
    assert resp.status_code == 403


def test_forbidden_extension_json():
    resp = client.get("/files/serve", params={"path": "/data/reports/report.json"})
    assert resp.status_code == 403


def test_system_file_blocked():
    resp = client.get("/files/serve", params={"path": "/etc/hosts"})
    assert resp.status_code == 403


def test_home_directory_blocked():
    resp = client.get("/files/serve", params={"path": "/Users/ravin/.ssh/id_rsa"})
    assert resp.status_code == 403


def test_valid_png_in_allowed_dir(tmp_path):
    """Un PNG dans data/corrupted doit être servi (404 si absent, pas 403)."""
    from pathlib import Path
    # On teste juste que le chemin passe la validation de sécurité
    # (le fichier n'existe pas → 404, pas 403)
    import os
    project_root = Path(__file__).resolve().parents[1]
    fake_path = project_root / "data" / "corrupted" / "nonexistent_test.png"
    resp = client.get("/files/serve", params={"path": str(fake_path)})
    # 404 = chemin valide mais fichier absent → sécurité OK
    # 403 = chemin bloqué → erreur
    assert resp.status_code == 404, f"Chemin valide bloqué à tort : {resp.status_code}"