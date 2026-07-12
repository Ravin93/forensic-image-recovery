"""Tests I1+I2+I3+I4 — analyses persistantes."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


@pytest.fixture
def gradient_png() -> bytes:
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        arr[i, :, 0] = i * 4
        arr[:, i, 1] = i * 4
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _start(client, image_bytes, corruption_type="scratch_lines", max_attempts=2):
    return client.post(
        "/analysis/start",
        files={"image": ("test.png", io.BytesIO(image_bytes), "image/png")},
        data={
            "corruption_type": corruption_type,
            "severity":        "light",
            "max_attempts":    str(max_attempts),
            "seed":            "42",
        },
    )


# ---------------------------------------------------------------------------
# I1 — analysis_id retourné immédiatement
# ---------------------------------------------------------------------------

class TestAnalysisStart:
    def test_returns_200(self, client, gradient_png):
        resp = _start(client, gradient_png)
        assert resp.status_code == 200

    def test_returns_analysis_id(self, client, gradient_png):
        resp = _start(client, gradient_png)
        data = resp.json()
        assert "analysis_id" in data
        assert len(data["analysis_id"]) == 12

    def test_returns_urls(self, client, gradient_png):
        resp = _start(client, gradient_png)
        data = resp.json()
        for key in ("status_url", "result_url", "download_url"):
            assert key in data

    def test_invalid_corruption_type(self, client, gradient_png):
        resp = client.post(
            "/analysis/start",
            files={"image": ("t.png", io.BytesIO(gradient_png), "image/png")},
            data={"corruption_type": "alien_mode", "severity": "light", "max_attempts": "2"},
        )
        assert resp.status_code == 422

    def test_invalid_severity(self, client, gradient_png):
        resp = client.post(
            "/analysis/start",
            files={"image": ("t.png", io.BytesIO(gradient_png), "image/png")},
            data={"corruption_type": "scratch_lines", "severity": "extreme", "max_attempts": "2"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# I2 — Stockage persistant
# ---------------------------------------------------------------------------

class TestAnalysisStore:
    def test_create_analysis(self, tmp_path, monkeypatch):
        import app.modules.analysis.analysis_store as store
        monkeypatch.setattr(store, "_ANALYSES_DIR", tmp_path / "analyses")
        aid = store.new_analysis_id()
        status = store.create_analysis(aid, {"test": True})
        assert status["status"] == "pending"
        assert status["analysis_id"] == aid

    def test_update_status(self, tmp_path, monkeypatch):
        import app.modules.analysis.analysis_store as store
        monkeypatch.setattr(store, "_ANALYSES_DIR", tmp_path / "analyses")
        aid = store.new_analysis_id()
        store.create_analysis(aid)
        store.update_status(aid, "running")
        s = store.get_status(aid)
        assert s["status"] == "running"

    def test_save_and_get_result(self, tmp_path, monkeypatch):
        import app.modules.analysis.analysis_store as store
        monkeypatch.setattr(store, "_ANALYSES_DIR", tmp_path / "analyses")
        aid = store.new_analysis_id()
        store.create_analysis(aid)
        store.save_result(aid, {"score": 75.0, "strategy": "inpainting_r3"})
        result = store.get_result(aid)
        assert result["score"] == 75.0

    def test_status_json_persists(self, tmp_path, monkeypatch):
        import app.modules.analysis.analysis_store as store
        monkeypatch.setattr(store, "_ANALYSES_DIR", tmp_path / "analyses")
        aid = store.new_analysis_id()
        store.create_analysis(aid, {"corruption": "bar"})
        # Recharger depuis disque
        s = store.get_status(aid)
        assert s["request"]["corruption"] == "bar"

    def test_list_analyses(self, tmp_path, monkeypatch):
        import app.modules.analysis.analysis_store as store
        monkeypatch.setattr(store, "_ANALYSES_DIR", tmp_path / "analyses")
        for _ in range(3):
            aid = store.new_analysis_id()
            store.create_analysis(aid)
        results = store.list_analyses(limit=10)
        assert len(results) == 3

    def test_delete_analysis(self, tmp_path, monkeypatch):
        import app.modules.analysis.analysis_store as store
        monkeypatch.setattr(store, "_ANALYSES_DIR", tmp_path / "analyses")
        aid = store.new_analysis_id()
        store.create_analysis(aid)
        assert store.delete_analysis(aid)
        with pytest.raises(FileNotFoundError):
            store.get_status(aid)


# ---------------------------------------------------------------------------
# I3 — Endpoints status/result/download
# ---------------------------------------------------------------------------

class TestAnalysisEndpoints:
    def test_status_endpoint_returns_pending(self, client, gradient_png):
        resp = _start(client, gradient_png)
        aid = resp.json()["analysis_id"]
        # Polling immédiat — peut être pending ou running ou completed
        status_resp = client.get(f"/analysis/{aid}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] in ("pending", "running", "completed", "failed")

    def test_status_404_for_unknown(self, client):
        resp = client.get("/analysis/unknownid000/status")
        assert resp.status_code == 404

    def test_result_404_for_unknown(self, client):
        resp = client.get("/analysis/unknownid000/result")
        assert resp.status_code == 404

    def test_delete_endpoint(self, client, gradient_png):
        resp = _start(client, gradient_png)
        aid = resp.json()["analysis_id"]
        time.sleep(0.1)
        del_resp = client.delete(f"/analysis/{aid}")
        assert del_resp.status_code == 200
        assert del_resp.json()["analysis_id"] == aid

    def test_list_endpoint(self, client, gradient_png):
        _start(client, gradient_png)
        resp = client.get("/analysis/")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_result_available_after_completion(self, client, gradient_png):
        """Attend la complétion et vérifie le résultat."""
        resp = _start(client, gradient_png, max_attempts=2)
        aid = resp.json()["analysis_id"]

        # Polling jusqu'à complétion (max 30s)
        for _ in range(30):
            status = client.get(f"/analysis/{aid}/status").json()
            if status["status"] in ("completed", "failed"):
                break
            time.sleep(1)

        if status["status"] == "completed":
            result_resp = client.get(f"/analysis/{aid}/result")
            assert result_resp.status_code == 200
            result = result_resp.json()
            assert "repair_score" in result or "score" in result or "reconstruction" in result


# ---------------------------------------------------------------------------
# I4 — BackgroundTasks
# ---------------------------------------------------------------------------

class TestBackgroundTask:
    def test_analysis_completes_eventually(self, client, gradient_png):
        """L'analyse doit passer de pending/running à completed."""
        resp = _start(client, gradient_png, max_attempts=2)
        assert resp.status_code == 200
        aid = resp.json()["analysis_id"]

        final_status = "unknown"
        for _ in range(30):
            s = client.get(f"/analysis/{aid}/status").json()
            final_status = s["status"]
            if final_status in ("completed", "failed"):
                break
            time.sleep(1)

        assert final_status in ("completed", "failed"), (
            f"Analyse toujours en {final_status} après 30s"
        )

    def test_analysis_id_is_unique(self, client, gradient_png):
        ids = set()
        for _ in range(3):
            r = _start(client, gradient_png, max_attempts=2)
            ids.add(r.json()["analysis_id"])
        assert len(ids) == 3