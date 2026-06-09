from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from app.modules.corruption.detection import detect_advanced_mask, detect_dark_regions_mask
from app.modules.reconstruction.repair_pipeline import run_repair_pipeline

router = APIRouter(tags=["reconstruction"])

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MASKS_DIR    = _PROJECT_ROOT / "data" / "masks"
_INPUT_DIR    = _PROJECT_ROOT / "data" / "input"
_MASKS_DIR.mkdir(parents=True, exist_ok=True)
_INPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers partagés
# ---------------------------------------------------------------------------

def _save_image(content: bytes, suffix: str, prefix: str = "upload") -> Path:
    ts = int(time.time())
    p = _INPUT_DIR / f"{prefix}_{ts}{suffix}"
    p.write_bytes(content)
    return p


def _save_mask(mask_arr: np.ndarray, prefix: str = "mask") -> Path:
    ts = int(time.time())
    p = _MASKS_DIR / f"{prefix}_{ts}.png"
    Image.fromarray(mask_arr).save(str(p))
    return p


def _binarize_mask(content: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(content)).convert("L")
    arr = np.array(pil)
    return ((arr > 128).astype(np.uint8)) * 255


# ---------------------------------------------------------------------------
# Endpoint existant
# ---------------------------------------------------------------------------

@router.post("/reconstruction")
def run_reconstruction(data: dict):
    try:
        image_path = data.get("image_path")
        mask_path  = data.get("mask_path")
        if not image_path or not mask_path:
            raise HTTPException(status_code=400, detail="Parametres manquants")
        ip = Path(image_path)
        mp = Path(mask_path)
        if not ip.exists():
            raise HTTPException(status_code=404, detail="Image introuvable")
        if not mp.exists():
            raise HTTPException(status_code=404, detail="Masque introuvable")
        return run_repair_pipeline(corrupted_image_path=ip, mask_path=mp,
                                   method="opencv_inpaint")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# C2+C3 — POST /reconstruction/repair-with-mask
# ---------------------------------------------------------------------------

@router.post("/reconstruction/repair-with-mask")
async def repair_with_user_mask(
    image: UploadFile = File(...),
    mask:  UploadFile = File(...),
    max_attempts: int = Form(8, ge=1, le=11),
    original_image_path: str | None = Form(None),
):
    img_suffix = Path(image.filename or "img.png").suffix.lower()
    if img_suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=422, detail=f"Format image non supporte : {img_suffix}")
    if Path(mask.filename or "mask.png").suffix.lower() != ".png":
        raise HTTPException(status_code=422, detail="Le masque doit etre un PNG")

    img_content  = await image.read(); await image.close()
    mask_content = await mask.read();  await mask.close()

    image_path = _save_image(img_content, img_suffix, "usermask_upload")

    try:
        binary_mask = _binarize_mask(mask_content)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Masque invalide : {exc}")

    if binary_mask.max() == 0:
        raise HTTPException(status_code=422,
            detail="Le masque est vide — dessinez en blanc la zone a reparer.")

    mask_path = _save_mask(binary_mask, "usermask")

    orig_path = Path(original_image_path) if original_image_path else None
    if orig_path and not orig_path.exists():
        orig_path = None

    try:
        result = run_repair_pipeline(
            corrupted_image_path=image_path, mask_path=mask_path,
            method="opencv_inpaint", corruption_type="mask_like",
            detection_confidence=1.0, original_image_path=orig_path,
            max_attempts=max_attempts,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "original_image":           str(image_path),
        "corrupted_image":          str(image_path),
        "reconstructed_image":      result["path"],
        "mask_path":                str(mask_path),
        "score":                    result.get("score", 0.0),
        "selected_repair_strategy": result.get("selected_repair_strategy", ""),
        "retry_count":              result.get("retry_count", 0),
        "candidates":               result.get("candidates", []),
        "status":                   "completed",
        "report_path":              "",
        "corruption_type":          "user_mask",
        "execution_mode":           "user_mask",
    }


# ---------------------------------------------------------------------------
# C4 — POST /reconstruction/compare-masks
# ---------------------------------------------------------------------------

@router.post(
    "/reconstruction/compare-masks",
    summary="Compare masque automatique vs masque utilisateur",
    description=(
        "Reconstruit la meme image deux fois : "
        "une fois avec le masque dessin\u00e9 manuellement, "
        "une fois avec un masque detect\u00e9 automatiquement. "
        "Retourne les deux resultats + le meilleur."
    ),
)
async def compare_masks(
    image: UploadFile = File(..., description="Image corrompue (JPEG ou PNG)"),
    user_mask: UploadFile = File(..., description="Masque utilisateur (PNG blanc = zone a reconstruire)"),
    detection_mode: str = Form("basic", description="'basic' ou 'advanced' pour la detection auto"),
    max_attempts: int = Form(6, ge=1, le=11),
    original_image_path: str | None = Form(None),
):
    """C4 — Reconstruit avec masque auto ET masque utilisateur, compare les scores."""
    # Validation
    img_suffix = Path(image.filename or "img.png").suffix.lower()
    if img_suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=422, detail=f"Format non supporte : {img_suffix}")
    if Path(user_mask.filename or "mask.png").suffix.lower() != ".png":
        raise HTTPException(status_code=422, detail="Le masque doit etre un PNG")

    img_content  = await image.read();     await image.close()
    mask_content = await user_mask.read(); await user_mask.close()

    # Sauvegarde image
    image_path = _save_image(img_content, img_suffix, "compare_upload")

    # Masque utilisateur
    try:
        user_mask_arr = _binarize_mask(mask_content)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Masque invalide : {exc}")
    if user_mask_arr.max() == 0:
        raise HTTPException(status_code=422,
            detail="Le masque utilisateur est vide.")
    user_mask_path = _save_mask(user_mask_arr, "compare_user_mask")

    # Masque automatique
    try:
        if detection_mode == "advanced":
            det = detect_advanced_mask(str(image_path))
            auto_mask_img = det["mask"]
            auto_confidence = float(det.get("confidence", 0.35))
        else:
            auto_mask_img = detect_dark_regions_mask(str(image_path))
            auto_confidence = 0.35

        auto_mask_arr = np.array(auto_mask_img.convert("L"), dtype=np.uint8)
        auto_mask_path = _save_mask(auto_mask_arr, "compare_auto_mask")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection auto echouee : {exc}")

    orig_path = Path(original_image_path) if original_image_path else None
    if orig_path and not orig_path.exists():
        orig_path = None

    errors = []

    # Reconstruction avec masque utilisateur
    try:
        result_user = run_repair_pipeline(
            corrupted_image_path=image_path,
            mask_path=user_mask_path,
            method="opencv_inpaint",
            corruption_type="mask_like",
            detection_confidence=1.0,
            original_image_path=orig_path,
            max_attempts=max_attempts,
        )
    except Exception as exc:
        result_user = None
        errors.append(f"user_mask: {exc}")

    # Reconstruction avec masque automatique
    try:
        result_auto = run_repair_pipeline(
            corrupted_image_path=image_path,
            mask_path=auto_mask_path,
            method="opencv_inpaint",
            corruption_type="mask_like",
            detection_confidence=auto_confidence,
            original_image_path=orig_path,
            max_attempts=max_attempts,
        )
    except Exception as exc:
        result_auto = None
        errors.append(f"auto_mask: {exc}")

    if result_user is None and result_auto is None:
        raise HTTPException(status_code=500,
            detail=f"Les deux reconstructions ont echoue : {errors}")

    # Comparaison des scores
    score_user = float(result_user.get("score", 0.0)) if result_user else -1.0
    score_auto = float(result_auto.get("score", 0.0)) if result_auto else -1.0

    if score_user >= score_auto:
        winner = "user_mask"
        best_result = result_user
        best_mask   = str(user_mask_path)
    else:
        winner = "auto_mask"
        best_result = result_auto
        best_mask   = str(auto_mask_path)

    # Taux de couverture : quelle proportion de la zone auto l'utilisateur a couverte
    bin_user = (user_mask_arr > 128)
    bin_auto = (auto_mask_arr > 128)
    if bin_auto.any():
        coverage_ratio = float(np.sum(bin_user & bin_auto)) / float(np.sum(bin_auto))
    else:
        coverage_ratio = 1.0

    # Analyse qualitative
    diff = abs(score_user - score_auto)
    cov_pct = round(coverage_ratio * 100)
    if diff < 1.0:
        comparison_note = f"Les deux masques donnent des resultats equivalents. (couverture : {cov_pct}%)"
    elif winner == "user_mask":
        comparison_note = f"Le masque utilisateur est meilleur de {diff:.1f} points. (couverture : {cov_pct}%)"
    else:
        comparison_note = f"La detection automatique est meilleure de {diff:.1f} points. (couverture : {cov_pct}%)"

    return {
        # Meilleur resultat (compatible avec le front)
        "original_image":           str(image_path),
        "corrupted_image":          str(image_path),
        "reconstructed_image":      best_result["path"] if best_result else "",
        "mask_path":                best_mask,
        "score":                    max(score_user, score_auto),
        "selected_repair_strategy": (best_result or {}).get("selected_repair_strategy", ""),
        "winner":                   winner,
        "comparison_note":          comparison_note,

        # Resultat masque utilisateur
        "user_mask": {
            "mask_path":                str(user_mask_path),
            "reconstructed_image":      result_user["path"] if result_user else None,
            "score":                    score_user if score_user >= 0 else None,
            "selected_repair_strategy": (result_user or {}).get("selected_repair_strategy"),
            "retry_count":              (result_user or {}).get("retry_count", 0),
            "status":                   "completed" if result_user else "failed",
        },

        # Resultat masque automatique
        "auto_mask": {
            "mask_path":                str(auto_mask_path),
            "detection_mode":           detection_mode,
            "detection_confidence":     auto_confidence,
            "reconstructed_image":      result_auto["path"] if result_auto else None,
            "score":                    score_auto if score_auto >= 0 else None,
            "selected_repair_strategy": (result_auto or {}).get("selected_repair_strategy"),
            "retry_count":              (result_auto or {}).get("retry_count", 0),
            "status":                   "completed" if result_auto else "failed",
        },

        "coverage_ratio":            round(coverage_ratio, 4),
        "status": "completed",
        "errors": errors if errors else None,
    }