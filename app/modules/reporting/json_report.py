import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.core.config import REPORTS_DIR, ensure_directories
from app.core.logger import logger


def _to_jsonable(value: Any) -> Any:
    """
    Convertit récursivement les types non sérialisables JSON :
    - numpy scalars -> python scalars
    - numpy arrays -> listes
    - Path -> str
    - inf / nan -> str contrôlée
    """
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return value

    return value


def build_report(
    source_image: str,
    corruption_result: dict[str, Any],
    reconstruction_result: dict[str, Any],
    comparison_result: dict[str, Any],
    status: str = "completed",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "run_id": uuid4().hex[:12],
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "source_image": source_image,
        "corruption": corruption_result,
        "reconstruction": reconstruction_result,
        "evaluation": comparison_result,
    }
    if extra:
        report.update(extra)
    return _to_jsonable(report)


def save_json_report(report: dict[str, Any], report_name: str | None = None) -> Path:
    ensure_directories()

    report = _to_jsonable(report)

    if report_name is None:
        report_name = f"report_{report['run_id']}.json"

    output_path = REPORTS_DIR / report_name

    if "analysis" not in report:
        report["analysis"] = build_analysis_block(report)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Rapport JSON sauvegardé : %s", output_path)
    return output_path

def build_analysis_block(report: dict[str, Any]) -> dict[str, Any]:
    detection_metrics = report.get("detection_metrics") or {}
    evaluation = report.get("evaluation") or {}
    gains = evaluation.get("gains") or {}

    iou = float(detection_metrics.get("iou", 0.0))
    improvement = bool(evaluation.get("improvement", False))
    improvement_score = float(gains.get("improvement_score", 0.0))

    if iou >= 0.6:
        detection_quality = "good"
    elif iou >= 0.3:
        detection_quality = "medium"
    else:
        detection_quality = "poor"

    if improvement and improvement_score > 0:
        repair_effectiveness = "improved"
    elif abs(improvement_score) < 0.5:
        repair_effectiveness = "neutral"
    else:
        repair_effectiveness = "degraded"

    notes = []
    if detection_quality == "poor":
        notes.append("La détection automatique est peu précise sur ce cas.")
    if repair_effectiveness == "degraded":
        notes.append("La reconstruction a dégradé l'image ou n'a pas apporté de gain mesurable.")
    if detection_quality == "good" and repair_effectiveness == "improved":
        notes.append("Le pipeline a correctement localisé et amélioré la zone corrompue.")

    return {
        "detection_quality": detection_quality,
        "repair_effectiveness": repair_effectiveness,
        "notes": notes,
    }