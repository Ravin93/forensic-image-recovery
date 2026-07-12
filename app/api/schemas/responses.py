from typing import Any, Optional
from pydantic import BaseModel


# --- Existant ---

class RunPipelineResponse(BaseModel):
    status: str
    report_path: str
    summary: Any


class HealthResponse(BaseModel):
    status: str


# --- Ticket 5 : corrupt-and-repair ---

class CandidateResponse(BaseModel):
    strategy: str
    path: str
    score: float
    mode: Optional[str] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    gain_psnr: Optional[float] = None
    gain_ssim: Optional[float] = None
    mask_region_score: Optional[float] = None
    outside_score: Optional[float] = None
    outside_preservation: Optional[float] = None
    score_breakdown: Optional[dict] = None

    model_config = {"extra": "ignore"}


class CorruptAndRepairResponse(BaseModel):
    original_image: str
    corrupted_image: str
    reconstructed_image: str
    mask_path: str
    score: float
    selected_repair_strategy: str
    retry_count: int
    candidates: list[CandidateResponse]
    top_candidates: Optional[dict] = None
    corruption_type: str
    execution_mode: str
    report_path: str
    status: str