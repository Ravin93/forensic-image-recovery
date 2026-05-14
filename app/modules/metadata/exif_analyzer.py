"""app/modules/metadata/exif_analyzer.py — K12.

Extraction et analyse forensique des metadonnees EXIF.
Detecte les incoherences, logiciels de retouche, GPS, metadata stripping.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.logger import logger

# Logiciels suspects (retouche, deepfake, generation)
_SUSPICIOUS_SOFTWARE = {
    "photoshop", "gimp", "lightroom", "affinity", "pixelmator",
    "capture one", "darktable", "rawtherapee", "luminar",
    "dall-e", "stable diffusion", "midjourney", "firefly",
    "topaz", "imagemagick", "paint.net",
}


def extract_exif(path: str | Path) -> dict[str, Any]:
    """Extrait les metadonnees EXIF d une image.

    Utilise Pillow (toujours disponible). piexif si installe.
    Retourne un dict vide sans crash si pas d EXIF.
    """
    path = Path(path)
    result: dict[str, Any] = {
        "exif_present": False,
        "camera_make":  None,
        "camera_model": None,
        "software":     None,
        "datetime_original": None,
        "datetime_modified": None,
        "gps_present":  False,
        "gps_lat":      None,
        "gps_lon":      None,
        "orientation":  None,
        "resolution_x": None,
        "resolution_y": None,
        "iso":          None,
        "exposure":     None,
        "focal_length": None,
        "flash":        None,
        "color_space":  None,
        "compression":  None,
        "bits_per_sample": None,
        "raw_tags":     {},
    }

    if not path.exists():
        return result

    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS

        with Image.open(str(path)) as img:
            exif_data = img._getexif()  # type: ignore[attr-defined]

        if not exif_data:
            return result

        result["exif_present"] = True
        raw: dict[str, Any] = {}

        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, str(tag_id))
            raw[tag] = value

            if tag == "Make":
                result["camera_make"] = str(value)
            elif tag == "Model":
                result["camera_model"] = str(value)
            elif tag == "Software":
                result["software"] = str(value)
            elif tag == "DateTimeOriginal":
                result["datetime_original"] = str(value)
            elif tag == "DateTime":
                result["datetime_modified"] = str(value)
            elif tag == "Orientation":
                result["orientation"] = int(value)
            elif tag == "XResolution":
                try: result["resolution_x"] = float(value)
                except: pass
            elif tag == "YResolution":
                try: result["resolution_y"] = float(value)
                except: pass
            elif tag == "ISOSpeedRatings":
                result["iso"] = value
            elif tag == "ExposureTime":
                try: result["exposure"] = float(value)
                except: pass
            elif tag == "FocalLength":
                try: result["focal_length"] = float(value)
                except: pass
            elif tag == "Flash":
                result["flash"] = value
            elif tag == "ColorSpace":
                result["color_space"] = value
            elif tag == "Compression":
                result["compression"] = value
            elif tag == "BitsPerSample":
                result["bits_per_sample"] = value
            elif tag == "GPSInfo":
                result["gps_present"] = True
                gps_data: dict[str, Any] = {}
                if isinstance(value, dict):
                    for gps_tag_id, gps_val in value.items():
                        gps_tag = GPSTAGS.get(gps_tag_id, str(gps_tag_id))
                        gps_data[gps_tag] = gps_val
                    # Convertir coordonnees
                    if "GPSLatitude" in gps_data and "GPSLatitudeRef" in gps_data:
                        try:
                            lat = _dms_to_decimal(gps_data["GPSLatitude"],
                                                  gps_data["GPSLatitudeRef"])
                            result["gps_lat"] = round(lat, 6)
                        except Exception:
                            pass
                    if "GPSLongitude" in gps_data and "GPSLongitudeRef" in gps_data:
                        try:
                            lon = _dms_to_decimal(gps_data["GPSLongitude"],
                                                  gps_data["GPSLongitudeRef"])
                            result["gps_lon"] = round(lon, 6)
                        except Exception:
                            pass

        result["raw_tags"] = {k: str(v) for k, v in raw.items()
                              if not isinstance(v, bytes)}

    except Exception as exc:
        logger.debug("EXIF extraction echouee : %s", exc)

    return result


def _dms_to_decimal(dms: tuple, ref: str) -> float:
    """Convertit degres/minutes/secondes en decimal."""
    d, m, s = dms
    decimal = float(d) + float(m) / 60 + float(s) / 3600
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def analyze_exif_consistency(path: str | Path) -> dict[str, Any]:
    """Analyse forensique des metadonnees EXIF.

    Retourne les flags de suspicion et un score de confiance.
    """
    exif = extract_exif(path)
    flags: list[str] = []

    # EXIF absent
    if not exif["exif_present"]:
        flags.append("EXIF absent ou supprime")

    # Logiciel de retouche
    sw = (exif.get("software") or "").lower()
    if sw:
        for suspect in _SUSPICIOUS_SOFTWARE:
            if suspect in sw:
                flags.append(f"Logiciel de retouche detecte : {exif['software']}")
                break

    # Incoherence de dates
    dt_orig = exif.get("datetime_original")
    dt_mod  = exif.get("datetime_modified")
    if dt_orig and dt_mod and dt_orig != dt_mod:
        flags.append(f"Date originale ({dt_orig}) != date modification ({dt_mod})")

    # GPS present
    if exif.get("gps_present"):
        coord_str = ""
        if exif.get("gps_lat") is not None:
            coord_str = f" ({exif['gps_lat']}, {exif['gps_lon']})"
        flags.append("Coordonnees GPS presentes" + coord_str)

    # Orientation suspecte
    orient = exif.get("orientation")
    if orient and orient not in (1, 2, 3, 4, 5, 6, 7, 8):
        flags.append(f"Orientation EXIF suspecte : {orient}")

    # Camera absente mais EXIF present
    if exif["exif_present"] and not exif.get("camera_model"):
        flags.append("EXIF present mais modele appareil absent")

    # Score suspicion (0.0 = clean, 1.0 = tres suspect)
    suspicion_score = min(1.0, len(flags) * 0.2)

    return {
        "path":            str(path),
        "exif_present":    exif["exif_present"],
        "camera":          (exif.get("camera_make") or "") + " " + (exif.get("camera_model") or ""),
        "software":        exif.get("software"),
        "datetime_original": exif.get("datetime_original"),
        "gps_present":     exif.get("gps_present"),
        "gps_lat":         exif.get("gps_lat"),
        "gps_lon":         exif.get("gps_lon"),
        "iso":             exif.get("iso"),
        "exposure":        exif.get("exposure"),
        "suspicion_flags": flags,
        "suspicion_score": round(suspicion_score, 2),
        "suspicious":      len(flags) > 0,
        "raw_tags":        exif.get("raw_tags", {}),
    }