from fastapi import APIRouter, HTTPException

from app.api.schemas.requests import RunPipelineRequest
from app.services.full_pipeline_service import run_full_pipeline

router = APIRouter()


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
                "selected_repair_strategy": pipeline_result["reconstruction"].get("selected_repair_strategy"),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))