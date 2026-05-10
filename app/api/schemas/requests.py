from typing import Any, Optional
from pydantic import BaseModel, Field


# --- Existant ---

class RunPipelineRequest(BaseModel):
    dump_path: str
    execution_mode: str = "assisted"
    detection_mode: str = "basic"
    corruption_level: Optional[int] = None
    seed: Optional[int] = None


# --- Ticket 5 : corrupt-and-repair ---

class CorruptAndRepairParams(BaseModel):
    """Paramètres optionnels de corruption (coordonnées, sévérité…)."""
    severity: Optional[str] = Field(None, description="'light' | 'medium' | 'heavy'")
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fill_mode: Optional[str] = None
    count: Optional[int] = None
    size_ratio: Optional[float] = None
    orientation: Optional[str] = None
    thickness: Optional[int] = None
    sigma: Optional[float] = None
    kernel_size: Optional[int] = None
    block_size: Optional[int] = None
    drop_ratio: Optional[float] = None

    model_config = {"extra": "allow"}   # passe le reste tel quel à corrupt_image