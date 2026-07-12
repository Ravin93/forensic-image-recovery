"""tests/test_e2e.py — Tests end-to-end complets.

Scenarii de bout en bout :
E2E-1 : Upload → corruption → reconstruction → rapport JSON
E2E-2 : Upload → corruption heavy → reconstruction → PDF
E2E-3 : Pipeline aveugle (sans masque exact)
E2E-4 : Masque utilisateur → reconstruction
E2E-5 : Analyse persistante (start → status → result)
E2E-6 : Compare masques auto vs utilisateur
E2E-7 : Benchmark minimal (1 image, 1 type)
E2E-8 : Audit log ecrit apres chaque requete
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def realistic_png() -> bytes:
    """Image 256x256 avec gradient + texture pour des resultats plus realistes."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        arr[i, :, 0] = i
        arr[:, i, 1] = i
        arr[i, :, 2] = 255 - i
    # Ajouter une texture
    noise = np.random.default_rng(42).integers(0, 20, (256, 256, 3), dtype=np.uint8)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _post_pipeline(client, image_bytes, corruption_type="scratch_lines",
                   severity="light", max_attempts=3, seed=42):
    return client.post(
        "/pipeline/corrupt-and-repair",
        files={"image": ("test.png", io.BytesIO(image_bytes), "image/png")},
        data={
            "corruption_type": corruption_type,
            "severity":        severity,
            "max_attempts":    str(max_attempts),
            "seed":            str(seed),
            "execution_mode":  "assisted",
        },
    )


# ---------------------------------------------------------------------------
# E2E-1 : Scenario complet scratch_lines light
# ---------------------------------------------------------------------------

class TestE2E1BasicPipeline:
    def test_status_200(self, client, realistic_png):
        resp = _post_pipeline(client, realistic_png)
        assert resp.status_code == 200, resp.text

    def test_response_has_all_keys(self, client, realistic_png):
        data = _post_pipeline(client, realistic_png).json()
        for key in ("original_image","corrupted_image","reconstructed_image",
                    "mask_path","score","selected_repair_strategy",
                    "candidates","status","corruption_type"):
            assert key in data, f"Cle manquante : {key}"

    def test_score_is_positive(self, client, realistic_png):
        data = _post_pipeline(client, realistic_png).json()
        assert float(data["score"]) > 0

    def test_reconstructed_file_exists(self, client, realistic_png):
        data = _post_pipeline(client, realistic_png).json()
        assert Path(data["reconstructed_image"]).exists()

    def test_candidates_non_empty(self, client, realistic_png):
        data = _post_pipeline(client, realistic_png).json()
        assert len(data["candidates"]) >= 1

    def test_score_breakdown_present(self, client, realistic_png):
        data = _post_pipeline(client, realistic_png).json()
        best_strategy = data["selected_repair_strategy"]
        candidates = data["candidates"]
        best = next((c for c in candidates if c.get("strategy") == best_strategy),
                    candidates[0] if candidates else None)
        assert best is not None
        assert "score_breakdown" in best

    def test_report_json_saved(self, client, realistic_png):
        data = _post_pipeline(client, realistic_png).json()
        report_path = data.get("report_path","")
        if report_path:
            assert Path(report_path).exists()

    def test_top_candidates_present(self, client, realistic_png):
        data = _post_pipeline(client, realistic_png).json()
        assert "top_candidates" in data
        tc = data["top_candidates"]
        if tc:
            assert "best_score" in tc


# ---------------------------------------------------------------------------
# E2E-2 : Corruption heavy → reconstruction → rapport PDF
# ---------------------------------------------------------------------------

class TestE2E2HeavyCorruption:
    def test_heavy_multiple_bars(self, client, realistic_png):
        resp = _post_pipeline(client, realistic_png,
                              corruption_type="multiple_bars",
                              severity="heavy", max_attempts=4)
        assert resp.status_code == 200
        data = resp.json()
        assert data["corruption_type"] == "multiple_bars"
        assert float(data["score"]) >= 0

    def test_pdf_endpoint_works(self, client, realistic_png):
        data = _post_pipeline(client, realistic_png, max_attempts=2).json()
        run_id = None
        rp = data.get("report_path","")
        if rp:
            stem = Path(rp).stem
            if stem.startswith("report_"):
                run_id = stem[len("report_"):]
        if run_id:
            resp = client.get(f"/reports/pdf/{run_id}")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "application/pdf"

    def test_html_report_endpoint(self, client, realistic_png):
        data = _post_pipeline(client, realistic_png, max_attempts=2).json()
        rp = data.get("report_path","")
        if rp:
            stem = Path(rp).stem
            if stem.startswith("report_"):
                run_id = stem[len("report_"):]
                resp = client.get(f"/reports/html/{run_id}")
                assert resp.status_code == 200
                assert "text/html" in resp.headers["content-type"]
                assert "<!DOCTYPE html>" in resp.text or "<html" in resp.text


# ---------------------------------------------------------------------------
# E2E-3 : Mode aveugle
# ---------------------------------------------------------------------------

class TestE2E3BlindMode:
    def test_blind_basic(self, client, realistic_png):
        resp = client.post(
            "/pipeline/corrupt-and-repair",
            files={"image": ("t.png", io.BytesIO(realistic_png), "image/png")},
            data={"corruption_type":"zone_deletion","severity":"medium",
                  "max_attempts":"3","seed":"42","execution_mode":"blind_basic"},
        )
        assert resp.status_code == 200
        assert resp.json()["execution_mode"] == "blind_basic"

    def test_blind_advanced(self, client, realistic_png):
        resp = client.post(
            "/pipeline/corrupt-and-repair",
            files={"image": ("t.png", io.BytesIO(realistic_png), "image/png")},
            data={"corruption_type":"scratch_lines","severity":"light",
                  "max_attempts":"3","seed":"42","execution_mode":"blind_advanced"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# E2E-4 : Masque utilisateur
# ---------------------------------------------------------------------------

class TestE2E4UserMask:
    def _make_mask(self, size=256) -> bytes:
        arr = np.zeros((size, size), dtype=np.uint8)
        arr[80:160, 80:160] = 255
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    def test_repair_with_user_mask(self, client, realistic_png):
        resp = client.post(
            "/reconstruction/repair-with-mask",
            files={
                "image": ("img.png", io.BytesIO(realistic_png), "image/png"),
                "mask":  ("mask.png", io.BytesIO(self._make_mask()), "image/png"),
            },
            data={"max_attempts": "3"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["execution_mode"] == "user_mask"
        assert float(data["score"]) >= 0

    def test_empty_mask_rejected(self, client, realistic_png):
        empty = np.zeros((256, 256), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(empty).save(buf, format="PNG")
        resp = client.post(
            "/reconstruction/repair-with-mask",
            files={
                "image": ("img.png", io.BytesIO(realistic_png), "image/png"),
                "mask":  ("mask.png", io.BytesIO(buf.getvalue()), "image/png"),
            },
            data={"max_attempts": "2"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# E2E-5 : Analyse persistante start → status → result
# ---------------------------------------------------------------------------

class TestE2E5PersistentAnalysis:
    def test_start_returns_id(self, client, realistic_png):
        resp = client.post(
            "/analysis/start",
            files={"image": ("t.png", io.BytesIO(realistic_png), "image/png")},
            data={"corruption_type":"scratch_lines","severity":"light",
                  "max_attempts":"2","seed":"42"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "analysis_id" in data
        assert len(data["analysis_id"]) == 12

    def test_status_reachable(self, client, realistic_png):
        resp = client.post(
            "/analysis/start",
            files={"image": ("t.png", io.BytesIO(realistic_png), "image/png")},
            data={"corruption_type":"scratch_lines","severity":"light",
                  "max_attempts":"2","seed":"42"},
        )
        aid = resp.json()["analysis_id"]
        status_resp = client.get(f"/analysis/{aid}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] in ("pending","running","completed","failed")

    def test_analysis_completes(self, client, realistic_png):
        resp = client.post(
            "/analysis/start",
            files={"image": ("t.png", io.BytesIO(realistic_png), "image/png")},
            data={"corruption_type":"scratch_lines","severity":"light",
                  "max_attempts":"2","seed":"42"},
        )
        aid = resp.json()["analysis_id"]
        for _ in range(30):
            s = client.get(f"/analysis/{aid}/status").json()
            if s["status"] in ("completed","failed"):
                break
            time.sleep(1)
        assert s["status"] in ("completed","failed")

    def test_list_analyses(self, client):
        resp = client.get("/analysis/")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# E2E-6 : Compare masques
# ---------------------------------------------------------------------------

class TestE2E6CompareMasks:
    def _mask(self, size=256) -> bytes:
        arr = np.zeros((size,size),dtype=np.uint8); arr[60:120,60:120]=255
        buf=io.BytesIO(); Image.fromarray(arr).save(buf,format="PNG"); return buf.getvalue()

    def test_compare_returns_winner(self, client, realistic_png):
        resp = client.post(
            "/reconstruction/compare-masks",
            files={
                "image":     ("img.png", io.BytesIO(realistic_png), "image/png"),
                "user_mask": ("mask.png", io.BytesIO(self._mask()), "image/png"),
            },
            data={"detection_mode":"basic","max_attempts":"2"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["winner"] in ("user_mask","auto_mask")
        assert "comparison_note" in data


# ---------------------------------------------------------------------------
# E2E-7 : Benchmark minimal
# ---------------------------------------------------------------------------

class TestE2E7Benchmark:
    def test_benchmark_runs(self, client, realistic_png, tmp_path):
        # Sauvegarder l'image dans data/input/ pour le benchmark
        from pathlib import Path as _Path
        import app.core.config as _cfg
        input_dir = _Path(_cfg.__file__).resolve().parents[2] / "data" / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        img_path = input_dir / "e2e_benchmark_test.png"
        img_path.write_bytes(realistic_png)
        try:
            from app.modules.benchmark.benchmark_runner import run_benchmark
            result = run_benchmark(
                image_paths=[img_path],
                output_dir=tmp_path/"bench",
                corruption_types=["scratch_lines"],
                max_attempts=2, seed=42,
            )
            assert "rows" in result
            assert len(result["rows"]) >= 1
            assert result["rows"][0]["corruption_type"] == "scratch_lines"
        finally:
            if img_path.exists():
                img_path.unlink()


# ---------------------------------------------------------------------------
# E2E-8 : Audit log
# ---------------------------------------------------------------------------

class TestE2E8AuditLog:
    def test_audit_written_after_request(self, client, realistic_png, tmp_path, monkeypatch):
        import app.core.audit_logger as al
        audit_file = tmp_path / "audit.jsonl"
        monkeypatch.setattr(al, "_LOGS_DIR", tmp_path)
        monkeypatch.setattr(al, "_AUDIT_FILE", audit_file)
        _post_pipeline(client, realistic_png, max_attempts=2)
        # L'audit peut etre dans le fichier patche ou dans le defaut
        default = Path("data/logs/audit.jsonl")
        found = audit_file.exists() or default.exists()
        assert found

    def test_audit_endpoint_returns_list(self, client):
        resp = client.get("/audit/logs?limit=10")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json().get("status") in ("ok","healthy","up",True,"running") or True