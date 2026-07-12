from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile

from app.api.schemas.requests import RunPipelineRequest
from app.api.schemas.responses import CandidateResponse, CorruptAndRepairResponse
from app.core.audit_logger import log_audit_entry, new_request_id
from app.core.full_pipeline_service import run_full_pipeline
from app.core.upload_validator import get_file_info, validate_upload
from app.modules.analysis.analysis_store import (
    create_analysis,
    new_analysis_id,
    record_strategy_completed,
    save_analysis_file,
    save_result,
    update_status,
    update_progress,
)
from app.services.pipeline_service import run_demo_pipeline

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_INPUT_DIR    = _PROJECT_ROOT / "data" / "input"
_INPUT_DIR.mkdir(parents=True, exist_ok=True)

_SUPPORTED_CORRUPTIONS = {
    "rectangle_mask", "noise", "zone_deletion", "combined", "bar",
    "local_blur", "shift_region", "block_dropout", "jpeg_block_artifacts",
    "mixed", "scratch_lines", "large_deleted_square", "multiple_bars",
    "random_holes", "local_noise",
}

_SEVERITY_DEFAULTS: dict[str, dict[str, Any]] = {
    "light":  {"count": 2, "size_ratio": 0.2, "sigma": 15.0, "drop_ratio": 0.1,
               "min_size": 20, "max_size": 80},
    "medium": {"count": 4, "size_ratio": 0.35, "sigma": 30.0, "drop_ratio": 0.25,
               "min_size": 30, "max_size": 120},
    "heavy":  {"count": 8, "size_ratio": 0.55, "sigma": 50.0, "drop_ratio": 0.45,
               "min_size": 50, "max_size": 200},
}

_RECT_BASED = {
    "rectangle_mask", "noise", "zone_deletion", "combined",
    "local_blur", "shift_region", "jpeg_block_artifacts", "local_noise",
}


def _build_progress_audit_logger(
    request_id: str,
    ip: str,
    endpoint: str,
    filename: str | None,
    sha256: str | None,
    corruption_type: str | None,
    started_at: float,
    interval_s: float = 30.0,
    analysis_id: str | None = None,
):
    last_logged = {"t": started_at}

    def _log(phase: str, details: dict[str, Any]) -> None:
        now = time.perf_counter()
        elapsed_s = now - started_at
        if analysis_id and phase == "supreme_plan":
            update_progress(analysis_id, {
                "total_strategies": int(details.get("total_strategies", 0) or 0),
            })
        elif analysis_id and phase == "strategy_completed":
            record_strategy_completed(
                analysis_id,
                str(details.get("strategy") or ""),
                details.get("score"),
                elapsed_s,
            )
        elif analysis_id and phase == "strategy_running":
            update_progress(analysis_id, {
                "phase": "strategy_running",
                "last_strategy": str(details.get("strategy") or ""),
                "elapsed_s": elapsed_s,
            })
        force = phase in {"task_started", "task_completed", "task_failed", "strategy_completed"}
        if not force and now - last_logged["t"] < interval_s:
            return
        last_logged["t"] = now
        log_audit_entry(
            request_id=request_id,
            ip=ip,
            endpoint=endpoint,
            filename=filename,
            sha256=sha256,
            corruption_type=corruption_type,
            processing_time_s=elapsed_s,
            status="progress",
            http_status=202,
            extra={"phase": phase, **details},
        )

    return _log


def _run_forensic_supreme_task(
    analysis_id: str,
    source_path: Path,
    corruption_type: str,
    corruption_params: dict[str, Any],
    randomize: bool,
    seed: int | None,
    request_id: str,
    client_ip: str,
    filename: str,
    sha256: str | None,
    started_at: float,
) -> None:
    progress = _build_progress_audit_logger(
        request_id=request_id,
        ip=client_ip,
        endpoint="/pipeline/corrupt-and-repair",
        filename=filename,
        sha256=sha256,
        corruption_type=corruption_type,
        started_at=started_at,
        analysis_id=analysis_id,
    )

    try:
        update_status(analysis_id, "running")
        progress("task_started", {"analysis_id": analysis_id, "execution_mode": "forensic_supreme"})

        result = run_demo_pipeline(
            source_image_path=source_path,
            corruption_type=corruption_type,
            corruption_params=corruption_params,
            randomize=randomize,
            execution_mode="forensic_supreme",
            seed=seed,
            progress_callback=progress,
        )

        for key, path_key in [
            ("original", "original_image"),
            ("corrupted", "corrupted_image"),
            ("reconstructed", "reconstructed_image"),
            ("mask", "mask_path"),
            ("report_json", "report_path"),
        ]:
            p = result.get(path_key)
            if p and Path(p).exists():
                save_analysis_file(analysis_id, key, p)

        save_result(analysis_id, result)
        update_status(analysis_id, "completed", extra={
            "score": result.get("repair_score"),
            "selected_repair_strategy": result.get("selected_repair_strategy"),
            "corruption_type": corruption_type,
            "execution_mode": "forensic_supreme",
        })
        progress("task_completed", {
            "analysis_id": analysis_id,
            "score": result.get("repair_score"),
            "selected_repair_strategy": result.get("selected_repair_strategy"),
        })

        log_audit_entry(
            request_id=request_id, ip=client_ip,
            endpoint="/pipeline/corrupt-and-repair",
            filename=filename, sha256=sha256,
            corruption_type=corruption_type,
            processing_time_s=time.perf_counter() - started_at,
            status="success", http_status=200,
            extra={"analysis_id": analysis_id, "execution_mode": "forensic_supreme"},
        )

    except Exception as exc:
        update_status(analysis_id, "failed", extra={"error": str(exc)})
        progress("task_failed", {"analysis_id": analysis_id, "error": str(exc)})
        log_audit_entry(
            request_id=request_id, ip=client_ip,
            endpoint="/pipeline/corrupt-and-repair",
            filename=filename, sha256=sha256,
            corruption_type=corruption_type,
            processing_time_s=time.perf_counter() - started_at,
            status="error", http_status=500,
            extra={"analysis_id": analysis_id, "detail": str(exc)},
        )


@router.post("/run")
def run_pipeline(request: RunPipelineRequest):
    try:
        result = run_full_pipeline(
            dump_path=request.dump_path,
            execution_mode=request.execution_mode,
            detection_mode=request.detection_mode,
            corruption_level=request.corruption_level,
            seed=request.seed,
        )
        pipeline_result = result["pipeline_result"]
        return {
            "status": result["status"],
            "report_path": pipeline_result["report_path"],
            "summary": {
                "extracted": result["extracted_count"],
                "valid": result["valid_count"],
                "execution_mode": pipeline_result["execution_mode"],
                "detection_mode": pipeline_result["detection_mode"],
                "recoverability_status": pipeline_result["recoverability_status"],
                "selected_repair_strategy": pipeline_result["reconstruction"].get(
                    "selected_repair_strategy"
                ),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/corrupt-and-repair",
    response_model=CorruptAndRepairResponse | dict[str, Any],
    summary="Upload → corruption → reconstruction automatique",
)
async def corrupt_and_repair(
    background_tasks: BackgroundTasks,
    request: Request,
    image: UploadFile = File(...),
    corruption_type: str = Form("scratch_lines"),
    severity: str = Form("medium"),
    max_attempts: int = Form(8, ge=1, le=20),
    execution_mode: str = Form("assisted"),
    seed: int | None = Form(None),
) -> CorruptAndRepairResponse:

    req_id     = new_request_id()
    client_ip  = request.client.host if request.client else "unknown"
    t_start    = time.perf_counter()
    filename   = image.filename or "upload"
    sha256     = None

    try:
        ct  = corruption_type.strip().lower()
        sev = severity.strip().lower()
        exec_mode = execution_mode.strip().lower()

        if ct not in _SUPPORTED_CORRUPTIONS:
            raise HTTPException(status_code=422,
                detail=f"Type non supporte : '{ct}'.")
        if sev not in _SEVERITY_DEFAULTS:
            raise HTTPException(status_code=422,
                detail=f"Severite invalide : '{sev}'. Valeurs : light | medium | heavy")

        # H1 — magic bytes + taille
        content  = await validate_upload(image)
        file_info = get_file_info(content, filename)
        sha256   = file_info["sha256"]

        ts = int(time.time())
        source_path = _INPUT_DIR / f"upload_{ts}{file_info['suffix']}"
        source_path.write_bytes(content)

        corruption_params = dict(_SEVERITY_DEFAULTS[sev])
        needs_randomize = ct in _RECT_BASED and not all(
            k in corruption_params for k in ("x", "y", "width", "height")
        )

        if exec_mode == "forensic_supreme":
            analysis_id = new_analysis_id()
            create_analysis(analysis_id, request_info={
                "corruption_type": ct,
                "severity": sev,
                "execution_mode": exec_mode,
                "max_attempts": None,
                "seed": seed,
                "filename": filename,
                "sha256": sha256,
            })
            background_tasks.add_task(
                _run_forensic_supreme_task,
                analysis_id=analysis_id,
                source_path=source_path,
                corruption_type=ct,
                corruption_params={} if needs_randomize else corruption_params,
                randomize=needs_randomize,
                seed=seed,
                request_id=req_id,
                client_ip=client_ip,
                filename=filename,
                sha256=sha256,
                started_at=t_start,
            )
            log_audit_entry(
                request_id=req_id, ip=client_ip,
                endpoint="/pipeline/corrupt-and-repair",
                filename=filename, sha256=sha256,
                corruption_type=ct,
                processing_time_s=time.perf_counter() - t_start,
                status="accepted", http_status=202,
                extra={"analysis_id": analysis_id, "execution_mode": exec_mode},
            )
            return {
                "analysis_id": analysis_id,
                "status": "pending",
                "execution_mode": exec_mode,
                "message": "Analyse forensic_supreme lancée en arrière-plan.",
                "status_url": f"/analysis/{analysis_id}/status",
                "result_url": f"/analysis/{analysis_id}/result",
                "download_url": f"/analysis/{analysis_id}/download",
            }

        result = run_demo_pipeline(
            source_image_path=source_path,
            corruption_type=ct,
            corruption_params={} if needs_randomize else corruption_params,
            randomize=needs_randomize,
            execution_mode=exec_mode,
            seed=seed,
            max_attempts=max_attempts,
        )

        elapsed = time.perf_counter() - t_start

        # H4 — audit success
        import app.core.audit_logger as _al
        _al.log_audit_entry(
            request_id=req_id, ip=client_ip,
            endpoint="/pipeline/corrupt-and-repair",
            filename=filename, sha256=sha256,
            corruption_type=ct,
            processing_time_s=elapsed,
            status="success", http_status=200,
            extra={"score": result.get("repair_score"), "severity": sev},
        )

        candidates_out = [
            CandidateResponse(
                strategy=str(c.get("strategy", "")),
                path=str(c.get("path", "")),
                score=float(c.get("score", 0.0)),
                mode=c.get("mode"),
                psnr=c.get("psnr"),
                ssim=c.get("ssim"),
                gain_psnr=c.get("gain_psnr"),
                gain_ssim=c.get("gain_ssim"),
                score_breakdown=c.get("score_breakdown"),
            )
            for c in result.get("repair_candidates", [])
        ]

        return CorruptAndRepairResponse(
            original_image=result["original_image"],
            corrupted_image=result["corrupted_image"],
            reconstructed_image=result["reconstructed_image"],
            mask_path=result["mask_path"],
            score=float(result.get("repair_score") or 0.0),
            selected_repair_strategy=str(result.get("selected_repair_strategy") or ""),
            retry_count=int(result.get("retry_count") or 0),
            candidates=candidates_out,
            corruption_type=ct,
            execution_mode=exec_mode,
            report_path=str(result.get("report_path") or ""),
            status=result.get("status", "completed"),
            top_candidates=result.get("top_candidates"),
        )

    except HTTPException as exc:
        elapsed = time.perf_counter() - t_start
        import app.core.audit_logger as _al
        _al.log_audit_entry(
            request_id=req_id, ip=client_ip,
            endpoint="/pipeline/corrupt-and-repair",
            filename=filename, sha256=sha256,
            corruption_type=corruption_type,
            processing_time_s=elapsed,
            status="error", http_status=exc.status_code,
            extra={"detail": str(exc.detail)},
        )
        raise

    except Exception as exc:
        elapsed = time.perf_counter() - t_start
        import app.core.audit_logger as _al
        _al.log_audit_entry(
            request_id=req_id, ip=client_ip,
            endpoint="/pipeline/corrupt-and-repair",
            filename=filename, sha256=sha256,
            corruption_type=corruption_type,
            processing_time_s=elapsed,
            status="error", http_status=500,
            extra={"detail": str(exc)},
        )
        raise HTTPException(status_code=500, detail=str(exc))
