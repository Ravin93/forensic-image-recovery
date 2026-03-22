from pydantic import BaseModel
from typing import Any


class RunPipelineResponse(BaseModel):
    status: str
    report_path: str
    summary: Any


class HealthResponse(BaseModel):
    status: str