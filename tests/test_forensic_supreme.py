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


def test_forensic_supreme_plan_adds_meta_regional_only_there():
    from app.modules.reconstruction.repair_pipeline import (
        _build_adaptive_plan,
        _build_forensic_supreme_plan,
    )

    supreme = _build_forensic_supreme_plan("zone_deletion", 3, "inpainting")
    adaptive = _build_adaptive_plan("zone_deletion", 3, "inpainting")

    assert any(item["strategy"] == "meta_regional" for item in supreme)
    assert not any(item.get("strategy") == "meta_regional" for item in adaptive)


def test_meta_regional_segmentation_bounds_regions(tmp_path: Path):
    from app.modules.reconstruction.repair_pipeline import _segment_mask_regions

    image = np.zeros((72, 72, 3), dtype=np.uint8)
    image[:, :36] = 40
    rng = np.random.default_rng(7)
    image[:, 36:] = rng.integers(0, 255, size=(72, 36, 3), dtype=np.uint8)
    cv2.line(image, (8, 8), (64, 64), (255, 255, 255), 2)

    mask = np.zeros((72, 72), dtype=np.uint8)
    mask[12:60, 10:62] = 255
    image_path = tmp_path / "masked.png"
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)

    regions = _segment_mask_regions(image_path, mask_path)

    assert 2 <= len(regions) <= 6
    assert sum(int(region["area"]) for region in regions) == int((mask > 0).sum())
    assert {region["type"] for region in regions} <= {"homogeneous", "textured", "strong_edges"}


def test_meta_regional_strategy_fuses_regions(monkeypatch, tmp_path: Path):
    import app.modules.reconstruction.repair_pipeline as repair

    image = np.zeros((40, 40, 3), dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[:, :20] = 255
    mask[:, 20:] = 255
    image_path = tmp_path / "corrupted.png"
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)

    region_a = np.zeros((40, 40), dtype=np.uint8)
    region_b = np.zeros((40, 40), dtype=np.uint8)
    region_a[:, :20] = 255
    region_b[:, 20:] = 255
    monkeypatch.setattr(
        repair,
        "_segment_mask_regions",
        lambda *_args, **_kwargs: [
            {"mask": region_a, "type": "homogeneous", "area": int((region_a > 0).sum())},
            {"mask": region_b, "type": "textured", "area": int((region_b > 0).sum())},
        ],
    )

    def fake_run(plan, _image_path, _region_mask_path, _method):
        value = 80 if plan["strategy"] in {"inpainting_r7", "patchmatch_p11"} else 20
        out = np.zeros((40, 40, 3), dtype=np.uint8) + value
        path = tmp_path / f"{plan['strategy']}.png"
        cv2.imwrite(str(path), out)
        return {"path": str(path)}

    def fake_score(_corrupted, candidate_path, _original=None, mask=None):
        score = 90.0 if "inpainting_r7" in str(candidate_path) or "patchmatch_p11" in str(candidate_path) else 10.0
        return score, {
            "mode": "supervised",
            "score": score,
            "psnr": 31.0,
            "ssim": 0.91,
            "gain_psnr": 2.5,
            "gain_ssim": 0.08,
            "score_breakdown": {"global_score": score},
            "region_pixels": int((mask > 0).sum()) if mask is not None else 0,
        }

    monkeypatch.setattr(repair, "_run_regional_strategy", fake_run)
    monkeypatch.setattr(repair, "_score_candidate", fake_score)

    candidate = repair._run_meta_regional_strategy(
        input_image_path=image_path,
        scoring_image_path=image_path,
        mask_path_obj=mask_path,
        original_image_path=image_path,
        method="opencv_inpaint",
        out_dir=tmp_path,
    )

    assert candidate["strategy"] == "meta_regional"
    assert candidate["selected_strategy"] == "meta_regional"
    assert candidate["selected_score"] == 90.0
    for key in ("psnr", "ssim", "mode", "gain_psnr", "gain_ssim", "score_breakdown"):
        assert key in candidate
    assert candidate["region_count"] == 2
    assert candidate["fusion"] == "gaussian_blending"
    assert {r["selected_strategy"] for r in candidate["regions"]} == {"inpainting_r7", "patchmatch_p11"}
    assert Path(candidate["path"]).exists()


def test_run_repair_pipeline_includes_meta_regional_candidate(monkeypatch, tmp_path: Path):
    import app.modules.reconstruction.repair_pipeline as repair

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
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
            "mode": "supervised",
            "psnr": 0.0,
            "ssim": 0.0,
            "gain_psnr": 0.0,
            "gain_ssim": 0.0,
            "score_breakdown": {},
        },
    )
    monkeypatch.setattr(
        repair,
        "_run_forensic_supreme_candidates",
        lambda **_kwargs: (
            [
                {
                    "strategy": "meta_regional",
                    "path": str(image_path),
                    "score": 33.0,
                    "family": "meta_regional",
                    "mode": "supervised",
                    "psnr": 30.0,
                    "ssim": 0.9,
                    "gain_psnr": 1.0,
                    "gain_ssim": 0.05,
                    "score_breakdown": {"global_score": 33.0},
                },
            ],
            [{"family": "meta_regional", "selected_strategy": "meta_regional", "selected_score": 33.0}],
        ),
    )

    result = repair.run_repair_pipeline(
        corrupted_image_path=image_path,
        mask_path=mask_path,
        forensic_supreme=True,
    )

    meta = [c for c in result["candidates"] if c["strategy"] == "meta_regional"]
    assert len(meta) == 1
    for key in ("psnr", "ssim", "mode", "gain_psnr", "gain_ssim", "score_breakdown"):
        assert key in meta[0]


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

    (tmp_path / "corrupted.png").write_bytes(b"source")
    calls: list[tuple[str, str, str]] = []
    copies: list[tuple[str, str]] = []
    family_counts: dict[str, int] = {}
    real_copy = repair.shutil.copy

    def tracked_copy(src, dst, *args, **kwargs):
        copies.append((str(src), str(dst)))
        return real_copy(src, dst, *args, **kwargs)

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
    monkeypatch.setattr(repair.shutil, "copy", tracked_copy)

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

    for _family, _strategy, input_path in calls:
        path = Path(input_path)
        assert path.name.startswith("sup_")
        assert path.exists()

    for (source_path, _short_path), (family, _strategy, _input_path) in zip(copies, calls):
        if family == chain[0]["family"]:
            assert source_path == str(tmp_path / "corrupted.png")
        else:
            previous = chain[[entry["family"] for entry in chain].index(family) - 1]
            assert source_path == previous["selected_path"]


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
        "strategy_completed",
        "strategy_completed",
        "strategy_completed",
    ]
    assert [entry["extra"].get("strategy") for entry in logs[1:]] == ["a", "b", "c", "d", "e"]
    assert [entry["processing_time_s"] for entry in logs] == [0.0, 5.0, 29.0, 30.0, 59.0, 60.0]


def test_progress_audit_logger_updates_analysis_status(monkeypatch, tmp_path: Path):
    import app.api.routes.pipeline as pipeline_route
    import app.modules.analysis.analysis_store as store

    monkeypatch.setattr(store, "_ANALYSES_DIR", tmp_path / "analyses")
    monkeypatch.setattr(pipeline_route, "log_audit_entry", lambda **_kwargs: None)
    ticks = iter([0.0, 1.7, 2.4, 3.1])
    monkeypatch.setattr(pipeline_route.time, "perf_counter", lambda: next(ticks))

    analysis_id = store.new_analysis_id()
    store.create_analysis(analysis_id)
    store.update_status(analysis_id, "running")
    progress = pipeline_route._build_progress_audit_logger(
        request_id="req",
        ip="test",
        endpoint="/pipeline/corrupt-and-repair",
        filename="image.png",
        sha256="sha",
        corruption_type="zone_deletion",
        started_at=0.0,
        analysis_id=analysis_id,
    )

    progress("supreme_plan", {"total_strategies": 23})
    progress("strategy_completed", {"strategy": "inpainting_r3", "score": 12.5})
    progress("strategy_completed", {"strategy": "inpainting_r5", "score": 24.0})
    progress("strategy_completed", {"strategy": "inpainting_r7", "score": 37.6})

    status = store.get_status(analysis_id)
    assert status["status"] == "running"
    assert status["strategies_completed"] == 3
    assert status["total_strategies"] == 23
    assert status["last_strategy"] == "inpainting_r7"
    assert status["last_score"] == 37.6
    assert status["elapsed_s"] == 3.1
    assert [event["strategy"] for event in status["strategy_completed_events"]] == [
        "inpainting_r3",
        "inpainting_r5",
        "inpainting_r7",
    ]


def test_progress_audit_logger_updates_running_strategy_status(monkeypatch, tmp_path: Path):
    import app.api.routes.pipeline as pipeline_route
    import app.modules.analysis.analysis_store as store

    monkeypatch.setattr(store, "_ANALYSES_DIR", tmp_path / "analyses")
    monkeypatch.setattr(pipeline_route, "log_audit_entry", lambda **_kwargs: None)
    ticks = iter([2.0])
    monkeypatch.setattr(pipeline_route.time, "perf_counter", lambda: next(ticks))

    analysis_id = store.new_analysis_id()
    store.create_analysis(analysis_id)
    store.update_status(analysis_id, "running", extra={"strategies_completed": 2})
    progress = pipeline_route._build_progress_audit_logger(
        request_id="req",
        ip="test",
        endpoint="/pipeline/corrupt-and-repair",
        filename="image.png",
        sha256="sha",
        corruption_type="zone_deletion",
        started_at=0.0,
        analysis_id=analysis_id,
    )

    progress("strategy_running", {"strategy": "patchmatch_p15_i20"})
    status = store.get_status(analysis_id)
    assert status["phase"] == "strategy_running"
    assert status["strategies_completed"] == 2
    assert status["last_strategy"] == "patchmatch_p15_i20"
    assert status["elapsed_s"] == 2.0


def test_supreme_strategy_heartbeat_emits_running_progress(monkeypatch, tmp_path: Path):
    import time as _time
    import app.modules.reconstruction.repair_pipeline as repair

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(repair, "_SUPREME_RUNNING_UPDATE_INTERVAL_S", 0.01)
    stop_event, thread = repair._start_supreme_strategy_heartbeat(
        lambda phase, details: events.append((phase, details)),
        "patchmatch",
        "patchmatch_p15_i20",
        tmp_path / "sup_input.png",
    )

    assert stop_event is not None
    assert thread is not None
    _time.sleep(0.04)
    stop_event.set()
    thread.join(timeout=0.2)

    assert any(phase == "strategy_running" for phase, _details in events)
    assert all(details["strategy"] == "patchmatch_p15_i20" for _phase, details in events)
