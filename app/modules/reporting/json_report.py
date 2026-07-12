import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from app.core.config import REPORTS_DIR, ensure_directories
from app.core.logger import logger


# ---------------------------------------------------------------------------
# Sérialisation JSON robuste
# ---------------------------------------------------------------------------

def _to_jsonable(value: Any) -> Any:
    """Convertit récursivement les types non sérialisables JSON."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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


# ---------------------------------------------------------------------------
# Construction du rapport — Ticket 8
# ---------------------------------------------------------------------------

def build_report(
    source_image: str,
    corruption_result: dict[str, Any],
    reconstruction_result: dict[str, Any],
    comparison_result: dict[str, Any],
    status: str = "completed",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construit le rapport JSON structuré.

    Structure de sortie (Ticket 8) :
        run_id, timestamp, status
        input       → source, format, validation
        corruption  → type, params, paths
        reconstruction → best_candidate, all_candidates, strategy, score, retry_count
        metrics     → psnr, ssim, gains, supervised_score, detection_metrics
        analysis    → detection_quality, repair_effectiveness, notes
    """
    extra = extra or {}

    # --- Section input ---
    input_section = {
        "source_image":   source_image,
        "corruption_type": corruption_result.get("corruption_type"),
        "execution_mode": extra.get("execution_mode"),
        "detection_mode": extra.get("detection_mode"),
        "seed":           extra.get("seed"),
    }

    # --- Section corruption ---
    corruption_section = {
        "type":          corruption_result.get("corruption_type"),
        "parameters":    corruption_result.get("parameters", {}),
        "image_path":    corruption_result.get("image_path") or corruption_result.get("path"),
        "mask_path":     corruption_result.get("mask_path"),
        "detected_mask": extra.get("detected_mask_path"),
        "randomize":     corruption_result.get("randomize", False),
        "imperfect_mask": corruption_result.get("imperfect_mask", False),
    }

    # --- Section reconstruction ---
    candidates = reconstruction_result.get("candidates", [])
    best = candidates[0] if candidates else {}

    best_candidate = {
        "strategy": reconstruction_result.get("selected_repair_strategy"),
        "path":     reconstruction_result.get("path"),
        "score":    reconstruction_result.get("score"),
        "psnr":     best.get("psnr"),
        "ssim":     best.get("ssim"),
        "gain_psnr": best.get("gain_psnr"),
        "gain_ssim": best.get("gain_ssim"),
        "mode":     best.get("mode"),
    }

    # Candidats allégés (sans la clé "comparison" volumineuse)
    all_candidates = [
        {
            "strategy":  c.get("strategy"),
            "path":      c.get("path"),
            "score":     c.get("score"),
            "psnr":      c.get("psnr"),
            "ssim":      c.get("ssim"),
            "gain_psnr": c.get("gain_psnr"),
            "gain_ssim": c.get("gain_ssim"),
            "mode":      c.get("mode"),
        }
        for c in candidates
    ]

    reconstruction_section = {
        "best_candidate":         best_candidate,
        "all_candidates":         all_candidates,
        "selected_strategy":      reconstruction_result.get("selected_repair_strategy"),
        "recommended_strategy":   reconstruction_result.get("recommended_strategy"),
        "score":                  reconstruction_result.get("score"),
        "retry_count":            reconstruction_result.get("retry_count", 0),
        "detection_confidence":   reconstruction_result.get("detection_confidence"),
        "method":                 reconstruction_result.get("method"),
        "corruption_type_used":   reconstruction_result.get("corruption_type"),
    }

    # --- Section metrics ---
    evaluation = comparison_result or {}
    gains = evaluation.get("gains", {})
    orig_vs_corr = evaluation.get("original_vs_corrupted", {})
    orig_vs_recon = evaluation.get("original_vs_reconstructed", {})
    supervised = evaluation.get("supervised_score", {})

    metrics_section = {
        "original_vs_corrupted": {
            "psnr": orig_vs_corr.get("psnr"),
            "ssim": orig_vs_corr.get("ssim"),
        },
        "original_vs_reconstructed": {
            "psnr": orig_vs_recon.get("psnr"),
            "ssim": orig_vs_recon.get("ssim"),
        },
        "gains": {
            "psnr_gain":        gains.get("psnr_gain"),
            "ssim_gain":        gains.get("ssim_gain"),
            "improvement_score": gains.get("improvement_score"),
        },
        "improvement": evaluation.get("improvement", False),
        "supervised_score": supervised,
        "detection_metrics": extra.get("detection_metrics", {}),
        "recoverability_status": extra.get("recoverability_status"),
    }

    # --- Rapport final ---
    report: dict[str, Any] = {
        "run_id":    uuid4().hex[:12],
        "timestamp": datetime.now().isoformat(),
        "status":    status,
        "input":         input_section,
        "corruption":    corruption_section,
        "reconstruction": reconstruction_section,
        "metrics":       metrics_section,
        # Champs complémentaires conservés pour compatibilité
        "corruption_classification": extra.get("corruption_classification"),
        "detection_confidence":      extra.get("detection_confidence"),
    }

    # Bloc d'analyse (calculé en dernier, utilise le rapport complet)
    report["analysis"] = build_analysis_block_v2(report)

    return _to_jsonable(report)


# ---------------------------------------------------------------------------
# Analyse automatique enrichie
# ---------------------------------------------------------------------------

def build_analysis_block(report: dict[str, Any]) -> dict[str, Any]:
    """Compatibilité ascendante — délègue à build_analysis_block_v2."""
    return build_analysis_block_v2(report)


def build_analysis_block_v2(report: dict[str, Any]) -> dict[str, Any]:
    """Analyse qualitative enrichie du résultat pipeline.

    Critères :
        detection_quality    good / medium / poor  (IoU)
        repair_effectiveness improved / neutral / degraded
        score_level          excellent / good / medium / poor
        notes                liste de messages lisibles
    """
    # Métriques de détection
    metrics = report.get("metrics") or report.get("detection_metrics") or {}
    detection_metrics = metrics.get("detection_metrics") or metrics
    iou = float(detection_metrics.get("iou", 0.0))

    # Gains
    gains_src = (
        (report.get("metrics") or {}).get("gains")
        or (report.get("evaluation") or {}).get("gains")
        or {}
    )
    improvement = bool(
        (report.get("metrics") or report.get("evaluation") or {}).get("improvement", False)
    )
    improvement_score = float(gains_src.get("improvement_score", 0.0))
    psnr_gain = float(gains_src.get("psnr_gain") or 0.0)
    ssim_gain = float(gains_src.get("ssim_gain") or 0.0)

    # Score global
    recon = report.get("reconstruction") or {}
    score = float(recon.get("score") or 0.0)

    # --- Qualité détection ---
    if iou >= 0.6:
        detection_quality = "good"
    elif iou >= 0.3:
        detection_quality = "medium"
    else:
        detection_quality = "poor"

    # --- Efficacité réparation ---
    if improvement and improvement_score > 0:
        repair_effectiveness = "improved"
    elif abs(improvement_score) < 0.5:
        repair_effectiveness = "neutral"
    else:
        repair_effectiveness = "degraded"

    # --- Niveau de score global ---
    if score >= 80:
        score_level = "excellent"
    elif score >= 60:
        score_level = "good"
    elif score >= 40:
        score_level = "medium"
    else:
        score_level = "poor"

    # --- Notes automatiques ---
    notes = []
    if detection_quality == "poor":
        notes.append("La détection automatique est peu précise sur ce cas.")
    if repair_effectiveness == "degraded":
        notes.append("La reconstruction a dégradé l'image ou n'a pas apporté de gain mesurable.")
    if detection_quality == "good" and repair_effectiveness == "improved":
        notes.append("Le pipeline a correctement localisé et amélioré la zone corrompue.")
    if psnr_gain > 3.0:
        notes.append(f"Gain PSNR significatif : +{psnr_gain:.1f} dB.")
    if ssim_gain > 0.05:
        notes.append(f"Gain SSIM notable : +{ssim_gain:.3f}.")
    if score_level == "excellent":
        notes.append("Score de reconstruction excellent (≥ 80/100).")
    if not notes:
        notes.append("Résultat dans la norme — aucune anomalie détectée.")

    return {
        "detection_quality":   detection_quality,
        "repair_effectiveness": repair_effectiveness,
        "score_level":         score_level,
        "score":               score,
        "iou":                 iou,
        "psnr_gain":           psnr_gain,
        "ssim_gain":           ssim_gain,
        "notes":               notes,
    }


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_json_report(report: dict[str, Any], report_name: str | None = None) -> Path:
    ensure_directories()
    report = _to_jsonable(report)

    if report_name is None:
        report_name = f"report_{report['run_id']}.json"

    output_path = REPORTS_DIR / report_name

    # Assure la présence du bloc analysis (si build_report n'a pas été appelé)
    if "analysis" not in report:
        report["analysis"] = build_analysis_block_v2(report)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Rapport JSON sauvegardé : %s", output_path)
    return output_path