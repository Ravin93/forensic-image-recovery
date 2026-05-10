from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.modules.reconstruction.repair_pipeline import run_repair_pipeline

router = APIRouter(tags=["reconstruction"])

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MASKS_DIR = _PROJECT_ROOT / "data" / "masks"
_MASKS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Endpoint existant — conservé
# ---------------------------------------------------------------------------

@router.post("/reconstruction")
def run_reconstruction(data: dict):
    try:
        image_path = data.get("image_path")
        mask_path  = data.get("mask_path")
        if not image_path or not mask_path:
            raise HTTPException(status_code=400, detail="Paramètres manquants")
        image_path_obj = Path(image_path)
        mask_path_obj  = Path(mask_path)
        if not image_path_obj.exists():
            raise HTTPException(status_code=404, detail="Image introuvable")
        if not mask_path_obj.exists():
            raise HTTPException(status_code=404, detail="Mask introuvable")
        result = run_repair_pipeline(
            corrupted_image_path=image_path_obj,
            mask_path=mask_path_obj,
            method="opencv_inpaint",
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Ticket C2+C3 — POST /reconstruction/repair-with-mask
# ---------------------------------------------------------------------------

@router.post("/reconstruction/repair-with-mask")
async def repair_with_user_mask(
    image: UploadFile = File(..., description="Image corrompue (JPEG ou PNG)"),
    mask:  UploadFile = File(..., description="Masque PNG dessiné par l'utilisateur (blanc = zone à reconstruire)"),
    max_attempts: int = Form(8, ge=1, le=11),
    original_image_path: str | None = Form(None, description="Chemin optionnel vers l'original pour scoring supervisé"),
):
    """Reconstruit une image en utilisant un masque dessiné manuellement.

    Le masque doit être un PNG en niveaux de gris ou RGBA où les zones
    blanches (>128) indiquent les zones à reconstruire.

    Retourne le même format que /pipeline/corrupt-and-repair pour
    compatibilité avec l'affichage front.
    """
    # --- Validation image ---
    img_suffix = Path(image.filename or "img.png").suffix.lower()
    if img_suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=422, detail=f"Format image non supporté : {img_suffix}")

    # --- Validation masque ---
    mask_suffix = Path(mask.filename or "mask.png").suffix.lower()
    if mask_suffix != ".png":
        raise HTTPException(status_code=422, detail="Le masque doit être un fichier PNG")

    # --- Sauvegarde image dans data/input/ ---
    input_dir = _PROJECT_ROOT / "data" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    image_path = input_dir / f"usermask_upload_{ts}{img_suffix}"
    try:
        content = await image.read()
        image_path.write_bytes(content)
    finally:
        await image.close()

    # --- Lecture et normalisation du masque ---
    try:
        mask_content = await mask.read()
        pil_mask = Image.open(io.BytesIO(mask_content)).convert("L")
        mask_arr = np.array(pil_mask)
        # Binarisation : blanc (>128) = zone à reconstruire
        binary_mask = ((mask_arr > 128).astype(np.uint8)) * 255
        pil_binary = Image.fromarray(binary_mask)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Masque invalide : {exc}")
    finally:
        await mask.close()

    if binary_mask.max() == 0:
        raise HTTPException(
            status_code=422,
            detail="Le masque est entièrement noir — aucune zone à reconstruire. "
                   "Dessinez en blanc les zones à réparer.",
        )

    # --- Sauvegarde masque dans data/masks/ ---
    mask_path = _MASKS_DIR / f"usermask_{ts}.png"
    pil_binary.save(str(mask_path))

    # --- Reconstruction ---
    orig_path = Path(original_image_path) if original_image_path else None
    if orig_path and not orig_path.exists():
        orig_path = None  # ignore si introuvable

    try:
        result = run_repair_pipeline(
            corrupted_image_path=image_path,
            mask_path=mask_path,
            method="opencv_inpaint",
            corruption_type="mask_like",
            detection_confidence=1.0,
            original_image_path=orig_path,
            max_attempts=max_attempts,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # --- Réponse compatible front ---
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