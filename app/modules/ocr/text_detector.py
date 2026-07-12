"""app/modules/ocr/text_detector.py + ocr_comparator.py — K11.

Analyse OCR sur zones reconstruites. Utilise Tesseract si disponible,
sinon retourne un resultat vide sans crash.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.logger import logger


# ---------------------------------------------------------------------------
# Detection disponibilite OCR
# ---------------------------------------------------------------------------

def _is_tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _is_easyocr_available() -> bool:
    try:
        import importlib
        importlib.import_module("easyocr")
        return True
    except ImportError:
        return False


def _ocr_backend() -> str:
    if _is_tesseract_available():
        return "tesseract"
    if _is_easyocr_available():
        return "easyocr"
    return "none"


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def run_ocr(image_path: str | Path) -> dict[str, Any]:
    """Extrait le texte d une image.

    Utilise Tesseract si disponible, sinon EasyOCR, sinon retourne vide.
    Ne crash jamais.
    """
    image_path = Path(image_path)
    base: dict[str, Any] = {
        "text":       "",
        "confidence": 0.0,
        "backend":    "none",
        "available":  False,
        "words":      [],
        "char_count": 0,
        "word_count": 0,
    }

    if not image_path.exists():
        base["error"] = f"Image introuvable : {image_path}"
        return base

    backend = _ocr_backend()
    base["backend"] = backend

    if backend == "none":
        base["error"] = "Aucun backend OCR disponible (pip install pytesseract)"
        return base

    base["available"] = True

    try:
        if backend == "tesseract":
            import pytesseract
            from PIL import Image as PILImage
            img = PILImage.open(str(image_path))
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT,
                                             lang="fra+eng")
            words = []
            confidences = []
            for i, word in enumerate(data["text"]):
                word = word.strip()
                if not word:
                    continue
                conf = float(data["conf"][i])
                if conf > 0:
                    words.append({"word": word, "confidence": conf / 100.0})
                    confidences.append(conf)
            text = " ".join(w["word"] for w in words)
            avg_conf = float(sum(confidences) / len(confidences)) / 100.0 if confidences else 0.0

        elif backend == "easyocr":
            import easyocr
            import numpy as np
            import cv2
            reader = easyocr.Reader(["fr","en"], gpu=False, verbose=False)
            results = reader.readtext(str(image_path))
            words = []
            texts = []
            confidences = []
            for (bbox, text_val, conf) in results:
                words.append({"word": text_val, "confidence": float(conf)})
                texts.append(text_val)
                confidences.append(float(conf))
            text = " ".join(texts)
            avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0

        base.update({
            "text":       text.strip(),
            "confidence": round(avg_conf, 3),
            "words":      words[:50],  # limiter la taille
            "char_count": len(text.strip()),
            "word_count": len(text.strip().split()) if text.strip() else 0,
        })

    except Exception as exc:
        logger.warning("OCR echec sur %s : %s", image_path.name, exc)
        base["error"] = str(exc)

    return base


# ---------------------------------------------------------------------------
# Comparaison OCR avant/apres
# ---------------------------------------------------------------------------

def compare_ocr_before_after(
    corrupted_path: str | Path,
    reconstructed_path: str | Path,
) -> dict[str, Any]:
    """Compare le texte detecte avant et apres reconstruction.

    Calcule un gain OCR base sur le nombre de caracteres recuperes.
    """
    ocr_before = run_ocr(corrupted_path)
    ocr_after  = run_ocr(reconstructed_path)

    text_before = ocr_before.get("text", "")
    text_after  = ocr_after.get("text", "")

    chars_before = ocr_before.get("char_count", 0)
    chars_after  = ocr_after.get("char_count",  0)

    # Gain : proportion de caracteres recuperes
    if chars_before == 0 and chars_after == 0:
        text_gain = 0.0
    elif chars_before == 0:
        text_gain = 1.0  # texte recupere depuis zero
    else:
        text_gain = float(max(0.0, chars_after - chars_before)) / max(chars_before, 1)
        text_gain = min(text_gain, 1.0)

    avg_conf = (
        (ocr_before.get("confidence", 0.0) + ocr_after.get("confidence", 0.0)) / 2.0
    )

    return {
        "corrupted_text":    text_before,
        "reconstructed_text": text_after,
        "chars_before":      chars_before,
        "chars_after":       chars_after,
        "words_before":      ocr_before.get("word_count", 0),
        "words_after":       ocr_after.get("word_count",  0),
        "text_gain":         round(text_gain, 3),
        "confidence":        round(avg_conf, 3),
        "backend":           ocr_before.get("backend", "none"),
        "ocr_available":     ocr_before.get("available", False),
        "text_recovered":    chars_after > chars_before,
    }