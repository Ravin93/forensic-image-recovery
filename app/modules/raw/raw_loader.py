"""app/modules/raw/raw_loader.py — K10.

Chargement de fichiers RAW photo (DNG, CR2, NEF, ARW).
Utilise rawpy si disponible, sinon retourne une erreur propre.
Module optionnel : pas de crash si rawpy absent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.core.logger import logger

_RAW_EXTENSIONS = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf",
                   ".rw2", ".pef", ".srw", ".raf", ".x3f"}


def is_rawpy_available() -> bool:
    try:
        import rawpy  # noqa: F401
        return True
    except ImportError:
        return False


def is_raw_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _RAW_EXTENSIONS


def load_raw_as_rgb(path: str | Path) -> dict[str, Any]:
    """Charge un fichier RAW et le convertit en tableau RGB uint8.

    Returns:
        dict avec image (np.ndarray), width, height, shape, path
        ou error si rawpy absent ou format non supporte
    """
    path = Path(path)
    base: dict[str, Any] = {
        "path":      str(path),
        "available": False,
        "image":     None,
        "width":     None,
        "height":    None,
        "shape":     None,
    }

    if not path.exists():
        base["error"] = f"Fichier introuvable : {path}"
        return base

    if not is_raw_file(path):
        base["error"] = f"Extension non RAW : {path.suffix}"
        return base

    if not is_rawpy_available():
        base["error"] = "rawpy non installe (pip install rawpy)"
        return base

    try:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=False,
                output_bps=8,
            )
        base.update({
            "available": True,
            "image":     rgb,
            "width":     rgb.shape[1],
            "height":    rgb.shape[0],
            "shape":     rgb.shape,
        })
        logger.info("RAW charge : %s | %dx%d", path.name, rgb.shape[1], rgb.shape[0])
    except Exception as exc:
        base["error"] = f"Erreur lecture RAW : {exc}"
        logger.warning("RAW load echec %s : %s", path.name, exc)

    return base


def save_raw_as_png(
    path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Charge un RAW et le sauvegarde en PNG pour le pipeline.

    Returns:
        dict avec output_path, ou error
    """
    result = load_raw_as_rgb(path)
    if not result.get("available") or result.get("image") is None:
        return result

    import cv2
    from app.core.config import ensure_directories
    ensure_directories()

    path = Path(path)
    if output_path is None:
        from app.core.config import INPUT_DIR
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = INPUT_DIR / (path.stem + "_from_raw.png")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = result["image"]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok  = cv2.imwrite(str(output_path), bgr)

    if not ok:
        result["error"] = f"Impossible d ecrire : {output_path}"
    else:
        result["output_path"] = str(output_path)
        logger.info("RAW converti : %s → %s", path.name, output_path.name)

    return result


def extract_raw_metadata(path: str | Path) -> dict[str, Any]:
    """Extrait les metadonnees d un fichier RAW.

    Returns:
        dict avec camera, iso, exposition, balance_blancs, dimensions, bits
    """
    path = Path(path)
    base: dict[str, Any] = {
        "path":          str(path),
        "available":     False,
        "camera_make":   None,
        "camera_model":  None,
        "iso":           None,
        "exposure":      None,
        "white_balance": None,
        "width":         None,
        "height":        None,
        "bits_per_pixel": None,
        "color_desc":    None,
    }

    if not path.exists():
        base["error"] = f"Fichier introuvable : {path}"
        return base

    if not is_rawpy_available():
        base["error"] = "rawpy non installe"
        return base

    try:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            base.update({
                "available":     True,
                "width":         raw.sizes.width,
                "height":        raw.sizes.height,
                "bits_per_pixel": raw.num_colors,
                "color_desc":    raw.color_desc.decode("ascii", errors="replace"),
                "white_balance": list(raw.camera_whitebalance),
            })
            # Tenter d extraire via EXIF embarque
            try:
                from app.modules.metadata.exif_analyzer import extract_exif
                exif = extract_exif(path)
                base["camera_make"]  = exif.get("camera_make")
                base["camera_model"] = exif.get("camera_model")
                base["iso"]          = exif.get("iso")
                base["exposure"]     = exif.get("exposure")
            except Exception:
                pass
    except Exception as exc:
        base["error"] = str(exc)

    return base