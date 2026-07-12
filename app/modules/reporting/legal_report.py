"""app/modules/reporting/legal_report.py — K19.

Rapport forensique legal : PDF + HTML enrichis avec
- Hash SHA-256 original
- Chaine de conservation (chain of custody)
- Disclaimer juridique
- Formulations prudentes (hypotheses, non-probant)
- Table des methodes utilisees
- Mode forensic-safe / generatif
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.logger import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_REPORTS_DIR  = _PROJECT_ROOT / "data" / "reports"

_DISCLAIMER_FR = """AVERTISSEMENT JURIDIQUE

Ce rapport a ete produit par un systeme automatise d analyse d images
forensiques. Les resultats presentes constituent des hypotheses visuelles
generees par algorithmes et ne sauraient en aucun cas constituer
une preuve judiciaire.

Toute reference a un contenu original est formule comme hypothese
compatible avec le contexte observe, sans certitude probatoire.

Ce document ne remplace pas l expertise d un specialiste forensique
habilite et ne peut etre utilise comme seul element de preuve dans
une procedure judiciaire.

Systeme : Forensic Image Recovery v1.0
Methodes : OpenCV Inpainting / PatchMatch / Reconstruction adaptative
"""

_LEGAL_NOTE = (
    "La reconstruction suggere une hypothese visuelle compatible "
    "avec le contexte, sans certitude probatoire."
)


# ---------------------------------------------------------------------------
# Chain of custody
# ---------------------------------------------------------------------------

def compute_sha256(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_chain_of_custody(
    original_path: str | Path | None,
    corrupted_path: str | Path | None,
    reconstructed_path: str | Path | None,
    report_id: str,
) -> list[dict[str, Any]]:
    """Construit la chaine de conservation (chain of custody)."""
    chain = []
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    if original_path:
        chain.append({
            "step":        "1_original",
            "label":       "Image source recue",
            "path":        str(original_path),
            "sha256":      compute_sha256(original_path),
            "timestamp":   ts,
            "action":      "reception",
        })

    if corrupted_path:
        chain.append({
            "step":        "2_corrupted",
            "label":       "Image apres degradation simulee",
            "path":        str(corrupted_path),
            "sha256":      compute_sha256(corrupted_path),
            "timestamp":   ts,
            "action":      "corruption_appliquee",
        })

    if reconstructed_path:
        chain.append({
            "step":        "3_reconstructed",
            "label":       "Image reconstruite par algorithme",
            "path":        str(reconstructed_path),
            "sha256":      compute_sha256(reconstructed_path),
            "timestamp":   ts,
            "action":      "reconstruction_algorithmique",
            "note":        _LEGAL_NOTE,
        })

    return chain


# ---------------------------------------------------------------------------
# Rapport legal JSON
# ---------------------------------------------------------------------------

def generate_legal_report(
    report: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """Genere un rapport forensique legal en JSON enrichi.

    Ajoute par rapport au rapport standard :
    - chain_of_custody
    - sha256 de chaque fichier
    - disclaimer
    - table_of_methods
    - forensic_mode
    - legal_note
    """
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = report.get("run_id", "unknown")
    if output_path is None:
        output_path = _REPORTS_DIR / f"legal_report_{run_id}.json"
    output_path = Path(output_path)

    inp   = report.get("input", {})
    corr  = report.get("corruption", {})
    recon = report.get("reconstruction", {})
    best  = recon.get("best_candidate", {})

    # Chain of custody
    chain = build_chain_of_custody(
        original_path=inp.get("source_image"),
        corrupted_path=corr.get("image_path"),
        reconstructed_path=best.get("path"),
        report_id=run_id,
    )

    # Table des methodes
    candidates = recon.get("all_candidates", [])
    table_of_methods = [
        {
            "strategy":   c.get("strategy", "—"),
            "score":      c.get("score"),
            "psnr":       c.get("psnr"),
            "ssim":       c.get("ssim"),
            "mode":       c.get("mode", "blind"),
            "forensic_safe": "patchmatch" not in str(c.get("strategy","")).lower()
                             and "lama" not in str(c.get("strategy","")).lower(),
        }
        for c in candidates
    ]

    # Mode forensic
    strategy = recon.get("selected_strategy") or best.get("strategy") or ""
    is_generative = "lama" in strategy.lower()
    forensic_mode = "generative" if is_generative else "forensic_safe"

    legal_data = {
        **report,
        "legal_report":       True,
        "generated_at":       datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "report_version":     "1.0-legal",
        "analysis_id":        run_id,
        "chain_of_custody":   chain,
        "table_of_methods":   table_of_methods,
        "forensic_mode":      forensic_mode,
        "sha256_original":    compute_sha256(inp.get("source_image", "")),
        "sha256_reconstructed": compute_sha256(best.get("path", "")),
        "disclaimer":         _DISCLAIMER_FR,
        "legal_note":         _LEGAL_NOTE,
        "warning":            (
            "Ce resultat est GENERATIF et non probant forensic."
            if is_generative else
            "Reconstruction algorithmique — hypothese visuelle uniquement."
        ),
        "certifications": {
            "probatoire":    False,
            "expertise_humaine_requise": True,
            "methode_reproductible": True,
            "open_source": True,
        },
    }

    output_path.write_text(
        json.dumps(legal_data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Rapport legal genere : %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Section legale pour PDF/HTML
# ---------------------------------------------------------------------------

def get_legal_html_section(
    chain: list[dict[str, Any]],
    forensic_mode: str = "forensic_safe",
    sha256_original: str = "",
) -> str:
    """Genere la section HTML legale pour le rapport HTML."""
    mode_color  = "#ff3d71" if forensic_mode == "generative" else "#ffaa00"
    mode_label  = "GENERATIF — NON PROBANT" if forensic_mode == "generative" else "FORENSIC-SAFE — HYPOTHESE VISUELLE"

    chain_rows = ""
    for step in chain:
        sha = step.get("sha256","")
        sha_display = sha[:16] + "..." if len(sha) > 16 else sha
        chain_rows += (
            "<tr>"
            + "<td>" + str(step.get("label","")) + "</td>"
            + "<td style='font-family:monospace;font-size:.65rem'>" + sha_display + "</td>"
            + "<td>" + str(step.get("action","")) + "</td>"
            + "<td>" + str(step.get("timestamp","")) + "</td>"
            + "</tr>"
        )

    return (
        "<div style='border:1px solid #ffaa00;padding:1rem;margin:1rem 0;background:rgba(255,170,0,.05)'>"
        + "<div style='font-family:monospace;font-size:.7rem;color:" + mode_color + ";margin-bottom:.5rem'>"
        + "⚖ MODE : " + mode_label + "</div>"
        + "<p style='font-size:.8rem;color:#e8e8f0;margin-bottom:.8rem'>" + _LEGAL_NOTE + "</p>"
        + "<table style='width:100%;border-collapse:collapse;font-family:monospace;font-size:.65rem'>"
        + "<thead><tr>"
        + "<th style='text-align:left;padding:.3rem;border-bottom:1px solid #2a2a3a;color:#6b6b80'>ETAPE</th>"
        + "<th style='text-align:left;padding:.3rem;border-bottom:1px solid #2a2a3a;color:#6b6b80'>SHA-256</th>"
        + "<th style='text-align:left;padding:.3rem;border-bottom:1px solid #2a2a3a;color:#6b6b80'>ACTION</th>"
        + "<th style='text-align:left;padding:.3rem;border-bottom:1px solid #2a2a3a;color:#6b6b80'>TIMESTAMP</th>"
        + "</tr></thead><tbody>" + chain_rows + "</tbody></table>"
        + "<pre style='margin-top:1rem;font-size:.6rem;color:#6b6b80;white-space:pre-wrap'>"
        + _DISCLAIMER_FR + "</pre>"
        + "</div>"
    )