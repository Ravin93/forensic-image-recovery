from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas.requests import CorruptAndRepairParams
from app.api.schemas.responses import CorruptAndRepairResponse
from app.core.full_pipeline_service import run_full_pipeline
from app.api.schemas.requests import RunPipelineRequest
from app.services.pipeline_service import run_demo_pipeline

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# ---------------------------------------------------------------------------
# Endpoint existant — conservé tel quel
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Ticket 5 — POST /pipeline/corrupt-and-repair
# ---------------------------------------------------------------------------

_SUPPORTED_CORRUPTIONS = {
    "rectangle_mask", "noise", "zone_deletion", "combined", "bar",
    "local_blur", "shift_region", "block_dropout", "jpeg_block_artifacts",
    "mixed", "scratch_lines", "large_deleted_square", "multiple_bars",
    "random_holes", "local_noise",
}

# Sévérité → paramètres par défaut pour les types sans coordonnées
_SEVERITY_DEFAULTS: dict[str, dict[str, Any]] = {
    # Paramètres génériques par sévérité.
    # Pour les types rect-based (zone_deletion, local_noise, etc.),
    # min_size/max_size sont utilisés par _maybe_randomize_rect dans simulator.py.
    "light":  {"count": 2, "size_ratio": 0.2, "sigma": 15.0, "drop_ratio": 0.1,
               "min_size": 20, "max_size": 80},
    "medium": {"count": 4, "size_ratio": 0.35, "sigma": 30.0, "drop_ratio": 0.25,
               "min_size": 30, "max_size": 120},
    "heavy":  {"count": 8, "size_ratio": 0.55, "sigma": 50.0, "drop_ratio": 0.45,
               "min_size": 50, "max_size": 200},
}


def _build_corruption_params(
    corruption_type: str,
    params: CorruptAndRepairParams,
    img_width: int,
    img_height: int,
) -> dict[str, Any]:
    """Construit les kwargs de corruption à partir du formulaire.

    Si aucune coordonnée x/y/width/height n'est fournie et que le type le
    nécessite, on randomise via le flag randomize=True de corrupt_image.
    """
    severity = (params.severity or "medium").lower()
    defaults = _SEVERITY_DEFAULTS.get(severity, _SEVERITY_DEFAULTS["medium"])

    # Tous les champs non-None du formulaire (sauf severity)
    extra = {
        k: v for k, v in params.model_dump(exclude={"severity"}).items()
        if v is not None
    }
    # Compléter avec les defaults de sévérité pour les clés absentes
    for k, v in defaults.items():
        extra.setdefault(k, v)

    return extra


@router.post(
    "/corrupt-and-repair",
    response_model=CorruptAndRepairResponse,
    summary="Upload → corruption → reconstruction automatique",
    description=(
        "Dépose une image, applique une dégradation réaliste, "
        "lance la reconstruction multi-essais et retourne les 3 images + score."
    ),
)
async def corrupt_and_repair(
    image: UploadFile = File(..., description="Image source (JPEG ou PNG)"),
    corruption_type: str = Form(
        "scratch_lines",
        description=(
            "Type de dégradation : rectangle_mask | noise | zone_deletion | "
            "combined | bar | local_blur | shift_region | block_dropout | "
            "jpeg_block_artifacts | mixed | scratch_lines | large_deleted_square | "
            "multiple_bars | random_holes | local_noise"
        ),
    ),
    severity: str = Form("medium", description="'light' | 'medium' | 'heavy'"),
    max_attempts: int = Form(8, ge=1, le=20, description="Nombre max de stratégies testées"),
    execution_mode: str = Form("assisted", description="'assisted' | 'blind_basic' | 'blind_advanced'"),
    seed: int | None = Form(None, description="Graine aléatoire (reproductibilité)"),
) -> CorruptAndRepairResponse:

    # --- Validation du type de corruption ---
    corruption_type = corruption_type.strip().lower()
    if corruption_type not in _SUPPORTED_CORRUPTIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Type de corruption non supporté : '{corruption_type}'. "
                f"Valeurs acceptées : {sorted(_SUPPORTED_CORRUPTIONS)}"
            ),
        )

    # --- Validation du fichier ---
    allowed_exts = {".jpg", ".jpeg", ".png"}
    suffix = Path(image.filename or "image.png").suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(
            status_code=422,
            detail=f"Format non supporté : '{suffix}'. Utilisez JPEG ou PNG.",
        )

    # --- Sauvegarde dans data/input/ (accessible par /files/serve) ---
    from app.core.config import ensure_directories
    ensure_directories()
    import time
    ts = int(time.time())
    input_dir = Path(__file__).resolve().parents[3] / "data" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_path = input_dir / f"upload_{ts}{suffix}"
    try:
        with source_path.open("wb") as f:
            shutil.copyfileobj(image.file, f)
    finally:
        image.file.close()

    # --- Paramètres de corruption ---
    severity_val = severity.strip().lower()
    if severity_val not in _SEVERITY_DEFAULTS:
        raise HTTPException(
            status_code=422,
            detail=f"Sévérité non supportée : '{severity_val}'. Valeurs : light | medium | heavy",
        )
    corruption_params = dict(_SEVERITY_DEFAULTS[severity_val])

    # Types qui nécessitent x/y/width/height → on active randomize=True
    # pour que corrupt_image génère les coordonnées automatiquement.
    _RECT_BASED = {
        "rectangle_mask", "noise", "zone_deletion", "combined",
        "local_blur", "shift_region", "jpeg_block_artifacts", "local_noise",
    }
    needs_randomize = corruption_type in _RECT_BASED and not all(
        k in corruption_params for k in ("x", "y", "width", "height")
    )

    # --- Appel pipeline ---
    try:
        result = run_demo_pipeline(
            source_image_path=source_path,
            corruption_type=corruption_type,
            corruption_params={} if needs_randomize else corruption_params,
            randomize=needs_randomize,
            execution_mode=execution_mode,
            seed=seed,
            max_attempts=max_attempts,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # --- Sérialisation des candidats ---
    candidates_out = []
    for c in result.get("repair_candidates", []):
        candidates_out.append(
            CandidateResponse(
                strategy=str(c.get("strategy", "")),
                path=str(c.get("path", "")),
                score=float(c.get("score", 0.0)),
                mode=c.get("mode"),
                psnr=c.get("psnr"),
                ssim=c.get("ssim"),
                gain_psnr=c.get("gain_psnr"),
                gain_ssim=c.get("gain_ssim"),
            )
        )

    return CorruptAndRepairResponse(
        original_image=result["original_image"],
        corrupted_image=result["corrupted_image"],
        reconstructed_image=result["reconstructed_image"],
        mask_path=result["mask_path"],
        score=float(result.get("repair_score") or 0.0),
        selected_repair_strategy=str(result.get("selected_repair_strategy") or ""),
        retry_count=int(result.get("retry_count") or 0),
        candidates=candidates_out,
        corruption_type=corruption_type,
        execution_mode=execution_mode,
        report_path=str(result.get("report_path") or ""),
        status=result.get("status", "completed"),
        top_candidates=result.get("top_candidates"),
    )


# Import local pour éviter circular import dans la réponse
from app.api.schemas.responses import CandidateResponse  # noqa: E402