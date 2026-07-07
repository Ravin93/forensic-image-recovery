from __future__ import annotations

import io
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None,
    reason="Pillow requis par le pipeline existant",
)


def _png_bytes() -> bytes:
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 0] = np.arange(64, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", arr)
    assert ok
    return encoded.tobytes()


def test_forensic_supreme_plan_adds_patchmatch_15_20():
    from app.modules.reconstruction.repair_pipeline import _build_forensic_supreme_plan

    plan = _build_forensic_supreme_plan("zone_deletion", 3, "inpainting")
    assert any(
        item["family"] == "patchmatch"
        and item["patch_size"] == 15
        and item["iterations"] == 20
        for item in plan
    )


def test_forensic_supreme_plan_adds_criminisi_only_there():
    from app.modules.reconstruction.repair_pipeline import (
        _build_adaptive_plan,
        _build_forensic_supreme_plan,
    )

    supreme = _build_forensic_supreme_plan("zone_deletion", 3, "inpainting")
    adaptive = _build_adaptive_plan("zone_deletion", 3, "inpainting")
    criminisi = [item for item in supreme if item["family"] == "criminisi"]

    assert [item["strategy"] for item in criminisi] == ["criminisi_p9", "criminisi_p15"]
    assert [item["patch_size"] for item in criminisi] == [9, 15]
    assert not any(item["family"] == "criminisi" for item in adaptive)


def test_criminisi_inpaint_returns_reconstruction(tmp_path: Path):
    from app.modules.reconstruction.inpainting import criminisi_inpaint

    image = np.zeros((48, 48, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(48, dtype=np.uint8)
    image[:, :, 1] = np.arange(48, dtype=np.uint8)[:, None]
    image[:, :, 2] = 120
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[18:28, 18:28] = 255

    image_path = tmp_path / "corrupted.png"
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)

    result = criminisi_inpaint(image_path, mask_path, patch_size=9)

    assert result["method"] == "criminisi_p9"
    assert result["patch_size"] == 9
    assert Path(result["path"]).exists()
    out = cv2.imread(result["path"])
    assert out.shape == image.shape


def test_forensic_supreme_candidate_runs_criminisi(monkeypatch, tmp_path: Path):
    import app.modules.reconstruction.repair_pipeline as repair

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    image_path = tmp_path / "corrupted.png"
    mask_path = tmp_path / "mask.png"
    out_path = tmp_path / "criminisi.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(out_path), image)

    captured: dict[str, object] = {}

    def fake_criminisi(image_arg, mask_arg, patch_size):
        captured.update({"image": image_arg, "mask": mask_arg, "patch_size": patch_size})
        return {"path": str(out_path), "method": f"criminisi_p{patch_size}"}

    monkeypatch.setattr(repair, "criminisi_inpaint", fake_criminisi)
    monkeypatch.setattr(
        repair,
        "_candidate_from_path",
        lambda name, path, *_args, **kwargs: {
            "strategy": name,
            "path": str(path),
            "score": 12.0,
            **(kwargs.get("extra") or {}),
        },
    )

    candidate = repair._run_repair_plan_candidate(
        {"strategy": "criminisi_p15", "family": "criminisi", "patch_size": 15},
        image_path,
        image_path,
        mask_path,
        None,
        "opencv_inpaint",
        3,
        tmp_path,
        None,
    )

    assert captured["patch_size"] == 15
    assert candidate["strategy"] == "criminisi_p15"
    assert candidate["family"] == "criminisi"


def test_forensic_supreme_reinjects_best_candidate_per_family(monkeypatch, tmp_path: Path):
    import app.modules.reconstruction.repair_pipeline as repair

    calls: list[tuple[str, str, str]] = []
    family_counts: dict[str, int] = {}

    def fake_candidate(plan, input_image_path, *_args, **_kwargs):
        family = str(plan["family"])
        strategy = str(plan["strategy"])
        family_counts[family] = family_counts.get(family, 0) + 1
        score = float(family_counts[family])
        path = tmp_path / f"{family}_{family_counts[family]}.png"
        path.write_bytes(b"fake")
        calls.append((family, strategy, str(input_image_path)))
        return {"strategy": strategy, "family": family, "path": str(path), "score": score}

    monkeypatch.setattr(repair, "_run_repair_plan_candidate", fake_candidate)
    monkeypatch.setattr(repair, "_run_iterative_pass", lambda *args, **kwargs: None)

    candidates, chain = repair._run_forensic_supreme_candidates(
        corrupted_image_path=tmp_path / "corrupted.png",
        mask_path_obj=tmp_path / "mask.png",
        original_image_path=tmp_path / "original.png",
        method="opencv_inpaint",
        base_radius=3,
        corruption_type="zone_deletion",
        recommended="inpainting",
        out_dir=tmp_path,
    )

    expected_plan = repair._build_forensic_supreme_plan("zone_deletion", 3, "inpainting")
    assert len(candidates) == len(expected_plan)
    assert len(calls) == len(expected_plan)
    assert any(strategy == "patchmatch_p15_i20" for _family, strategy, _input in calls)

    for idx in range(1, len(chain)):
        family = chain[idx]["family"]
        family_inputs = {input_path for call_family, _strategy, input_path in calls if call_family == family}
        assert family_inputs == {chain[idx - 1]["selected_path"]}


def test_run_repair_pipeline_forensic_supreme_ignores_max_attempts(monkeypatch, tmp_path: Path):
    import app.modules.reconstruction.repair_pipeline as repair

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    image_path = tmp_path / "corrupted.png"
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)

    monkeypatch.setattr(
        repair,
        "_candidate_from_path",
        lambda name, path, *_args, **_kwargs: {
            "strategy": name,
            "path": str(path),
            "score": 0.0,
            "family": "conservative",
        },
    )
    monkeypatch.setattr(
        repair,
        "_run_forensic_supreme_candidates",
        lambda **_kwargs: (
            [
                {"strategy": "inpainting_r3", "path": str(image_path), "score": 10.0, "family": "inpainting"},
                {"strategy": "patchmatch_p15_i20", "path": str(image_path), "score": 20.0, "family": "patchmatch"},
            ],
            [{"family": "inpainting"}, {"family": "patchmatch"}],
        ),
    )

    result = repair.run_repair_pipeline(
        corrupted_image_path=image_path,
        mask_path=mask_path,
        max_attempts=1,
        forensic_supreme=True,
    )

    assert result["max_attempts_applied"] is None
    assert len(result["candidates"]) == 3
    assert result["selected_repair_strategy"] == "patchmatch_p15_i20"


def test_pipeline_forensic_supreme_passes_flag_to_repair(monkeypatch, tmp_path: Path):
    pytest.importorskip("PIL")
    import app.services.pipeline_service as service

    image_path = tmp_path / "source.png"
    cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))
    captured: dict[str, object] = {}

    def fake_repair_pipeline(**kwargs):
        captured.update(kwargs)
        return {
            "path": str(kwargs["corrupted_image_path"]),
            "selected_repair_strategy": "patchmatch_p15_i20",
            "score": 42.0,
            "retry_count": 0,
            "candidates": [
                {"strategy": "patchmatch_p15_i20", "path": str(kwargs["corrupted_image_path"]), "score": 42.0}
            ],
            "top_candidates": {},
        }

    monkeypatch.setattr(service, "run_repair_pipeline", fake_repair_pipeline)

    result = service.run_demo_pipeline(
        source_image_path=image_path,
        corruption_type="zone_deletion",
        corruption_params={"x": 8, "y": 8, "width": 16, "height": 16},
        execution_mode="forensic_supreme",
        max_attempts=1,
        seed=7,
    )

    assert captured["forensic_supreme"] is True
    assert result["mode"]["max_attempts"] is None
    assert result["execution_mode"] == "forensic_supreme"


def test_corrupt_and_repair_forensic_supreme_returns_analysis_id(monkeypatch):
    pytest.importorskip("PIL")
    import app.api.routes.pipeline as pipeline_route
    from app.main import app
    from fastapi.testclient import TestClient

    scheduled: dict[str, object] = {}

    def fake_add_task(self, func, *args, **kwargs):
        scheduled["func"] = func
        scheduled["args"] = args
        scheduled["kwargs"] = kwargs

    monkeypatch.setattr(pipeline_route.BackgroundTasks, "add_task", fake_add_task)
    monkeypatch.setattr(pipeline_route, "new_analysis_id", lambda: "abc123def456")
    monkeypatch.setattr(pipeline_route, "create_analysis", lambda *args, **kwargs: {})

    client = TestClient(app)
    response = client.post(
        "/pipeline/corrupt-and-repair",
        files={"image": ("test.png", io.BytesIO(_png_bytes()), "image/png")},
        data={
            "corruption_type": "zone_deletion",
            "severity": "light",
            "execution_mode": "forensic_supreme",
            "max_attempts": "1",
            "seed": "42",
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["analysis_id"] == "abc123def456"
    assert data["status"] == "pending"
    assert data["execution_mode"] == "forensic_supreme"
    assert scheduled["func"] is pipeline_route._run_forensic_supreme_task


def test_progress_audit_logger_logs_every_30_seconds(monkeypatch):
    import app.api.routes.pipeline as pipeline_route

    logs: list[dict] = []
    ticks = iter([0.0, 5.0, 29.0, 30.0, 59.0, 60.0])

    monkeypatch.setattr(pipeline_route.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        pipeline_route,
        "log_audit_entry",
        lambda **kwargs: logs.append(kwargs),
    )

    progress = pipeline_route._build_progress_audit_logger(
        request_id="req",
        ip="test",
        endpoint="/pipeline/corrupt-and-repair",
        filename="image.png",
        sha256="sha",
        corruption_type="zone_deletion",
        started_at=0.0,
        interval_s=30.0,
    )
    progress("task_started", {})
    progress("strategy_completed", {"strategy": "a"})
    progress("strategy_completed", {"strategy": "b"})
    progress("strategy_completed", {"strategy": "c"})
    progress("strategy_completed", {"strategy": "d"})
    progress("strategy_completed", {"strategy": "e"})

    assert [entry["extra"]["phase"] for entry in logs] == [
        "task_started",
        "strategy_completed",
        "strategy_completed",
    ]
    assert [entry["processing_time_s"] for entry in logs] == [0.0, 30.0, 60.0]
