from typing import Optional

from pydantic import BaseModel


class RunPipelineRequest(BaseModel):
    dump_path: str
    execution_mode: str = "assisted"
    detection_mode: str = "basic"
    corruption_level: Optional[int] = None
    seed: Optional[int] = None