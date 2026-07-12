"""app/modules/ai/lama_adapter.py — K9.

Adaptateur LaMa (Large Mask inpainting) via simple-lama-inpainting.
Active si LAMA_ENABLED=true + simple_lama_inpainting installé.
Utilise MPS sur Mac M1 automatiquement.

Ne crash jamais si les dépendances sont absentes.
Résultat marqué GÉNÉRATIF / NON PROBANT forensic.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from app.core.logger import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ── Disponibilité ─────────────────────────────────────────────────────────────

def is_lama_enabled() -> bool:
    return os.getenv("LAMA_ENABLED", "false").lower() in ("true", "1", "yes")


def _detect_device() -> str:
    """Détecte le meilleur device disponible : MPS (M1) > CUDA > CPU."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"


def is_lama_available() -> bool:
    if not is_lama_enabled():
        return False
    try:
        import simple_lama_inpainting  # noqa: F401
        import torch                   # noqa: F401
        return True
    except ImportError:
        return False


def get_lama_status() -> dict[str, Any]:
    enabled       = is_lama_enabled()
    has_torch     = False
    has_simple    = False
    device        = "none"

    if enabled:
        try:
            import torch  # noqa: F401
            has_torch = True
            device = _detect_device()
        except ImportError:
            pass
        try:
            import simple_lama_inpainting  # noqa: F401
            has_simple = True
        except ImportError:
            pass

    available = enabled and has_torch and has_simple

    if not enabled:
        reason = "désactivé — LAMA_ENABLED=false"
    elif not has_torch:
        reason = "torch absent (pip install torch)"
    elif not has_simple:
        reason = "simple-lama-inpainting absent (pip install simple-lama-inpainting)"
    else:
        reason = f"disponible — device={device}"

    return {
        "enabled":       enabled,
        "has_torch":     has_torch,
        "has_simple_lama": has_simple,
        "available":     available,
        "device":        device,
        "reason":        reason,
    }


# ── Inférence ─────────────────────────────────────────────────────────────────

# Singleton pour éviter de recharger le modèle à chaque appel
_lama_model = None

def _get_model():
    """Charge le modèle LaMa une seule fois (singleton)."""
    global _lama_model
    if _lama_model is None:
        from simple_lama_inpainting import SimpleLama
        device = _detect_device()
        logger.info("Chargement modèle LaMa sur device=%s", device)
        _lama_model = SimpleLama(device=device)
        logger.info("Modèle LaMa chargé")
    return _lama_model


def run_lama_inpainting(
    image_path: str | Path,
    mask_path:  str | Path,
) -> dict[str, Any]:
    """Lance LaMa inpainting si disponible.

    Returns toujours un dict structuré, même si LaMa est absent.
    Le résultat est marqué GÉNÉRATIF / NON PROBANT forensic.
    """
    image_path = Path(image_path)
    mask_path  = Path(mask_path)

    base: dict[str, Any] = {
        "strategy":      "lama_inpainting",
        "forensic_mode": "generative",
        "warning":       (
            "Résultat génératif par deep learning (LaMa). "
            "Non probant en contexte forensique. "
            "À utiliser uniquement à des fins de visualisation."
        ),
        "available": False,
        "path":      None,
        "status":    "unavailable",
    }

    status = get_lama_status()
    base["lama_status"] = status

    if not status["available"]:
        base["status"]      = "skipped"
        base["skip_reason"] = status["reason"]
        logger.info("LaMa skip : %s", status["reason"])
        return base

    # ── Inférence réelle ──────────────────────────────────────────────────────
    try:
        from PIL import Image as PILImage
        import cv2
        import numpy as np

        t0 = time.perf_counter()

        # Charger image + masque
        img_pil  = PILImage.open(str(image_path)).convert("RGB")
        mask_pil = PILImage.open(str(mask_path)).convert("L")

        # Redimensionner le masque si nécessaire
        if img_pil.size != mask_pil.size:
            mask_pil = mask_pil.resize(img_pil.size, PILImage.NEAREST)

        # Binariser le masque (LaMa attend 0/255)
        import numpy as np
        mask_arr = np.array(mask_pil)
        mask_arr = (mask_arr > 127).astype(np.uint8) * 255
        mask_pil = PILImage.fromarray(mask_arr)

        # Inférence LaMa
        model  = _get_model()
        result = model(img_pil, mask_pil)

        # Sauvegarder
        from app.core.config import build_reconstructed_image_path
        output_path = build_reconstructed_image_path(
            image_path.name, "lama_inpainting"
        )
        result.save(str(output_path))

        elapsed = round(time.perf_counter() - t0, 3)
        logger.info(
            "LaMa inpainting OK | device=%s | %.3fs | %s",
            status["device"], elapsed, output_path.name,
        )

        base.update({
            "available":    True,
            "path":         str(output_path),
            "file":         output_path.name,
            "status":       "completed",
            "source_image": image_path.name,
            "mask_path":    str(mask_path),
            "device":       status["device"],
            "elapsed_s":    elapsed,
        })

    except Exception as exc:
        logger.warning("LaMa inpainting échoué : %s", exc)
        base["status"] = "error"
        base["error"]  = str(exc)

    return base