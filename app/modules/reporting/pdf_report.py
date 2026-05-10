"""app/modules/reporting/pdf_report.py — Ticket E1.

Génère un rapport PDF forensique complet :
  - Image originale / corrompue / masque / reconstruite
  - Type de dégradation + paramètres
  - Algorithmes testés avec scores
  - Meilleure stratégie sélectionnée
  - Métriques PSNR / SSIM / gains
  - Limites et notes d'analyse
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.core.config import REPORTS_DIR, ensure_directories
from app.core.logger import logger

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
_BG       = colors.HexColor("#0a0a0f")
_SURFACE  = colors.HexColor("#111118")
_ACCENT   = colors.HexColor("#00e5ff")
_ACCENT2  = colors.HexColor("#ff3d71")
_ACCENT3  = colors.HexColor("#a259ff")
_SUCCESS  = colors.HexColor("#00e096")
_TEXT     = colors.HexColor("#e8e8f0")
_MUTED    = colors.HexColor("#6b6b80")
_BORDER   = colors.HexColor("#2a2a3a")
_WHITE    = colors.white
_BLACK    = colors.black

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=_ACCENT,
            spaceAfter=4,
            leading=26,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=9,
            textColor=_MUTED,
            spaceAfter=12,
            letterSpacing=2,
        ),
        "section": ParagraphStyle(
            "section",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=_ACCENT,
            spaceBefore=14,
            spaceAfter=6,
            letterSpacing=1,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=8.5,
            textColor=_TEXT,
            spaceAfter=4,
            leading=13,
        ),
        "mono": ParagraphStyle(
            "mono",
            fontName="Courier",
            fontSize=7.5,
            textColor=_ACCENT,
            spaceAfter=2,
            leading=11,
        ),
        "label": ParagraphStyle(
            "label",
            fontName="Helvetica-Bold",
            fontSize=7,
            textColor=_MUTED,
            spaceAfter=1,
            letterSpacing=1,
        ),
        "note": ParagraphStyle(
            "note",
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=_MUTED,
            spaceAfter=3,
            leading=11,
        ),
        "score_big": ParagraphStyle(
            "score_big",
            fontName="Helvetica-Bold",
            fontSize=28,
            textColor=_SUCCESS,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "centered": ParagraphStyle(
            "centered",
            fontName="Helvetica",
            fontSize=8,
            textColor=_MUTED,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hr(color=_BORDER, thickness=0.5) -> HRFlowable:
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=2)


def _load_image_rl(path: str | Path | None, max_w: float, max_h: float) -> RLImage | None:
    """Charge une image pour ReportLab, redimensionnée proportionnellement."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        from PIL import Image as PILImage
        with PILImage.open(p) as img:
            w, h = img.size
        ratio = min(max_w / w, max_h / h)
        return RLImage(str(p), width=w * ratio, height=h * ratio)
    except Exception:
        return None


def _fmt(value: Any, decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _score_color(score: float) -> colors.Color:
    if score >= 80:
        return _SUCCESS
    if score >= 60:
        return _ACCENT
    if score >= 40:
        return colors.HexColor("#ffaa00")
    return _ACCENT2


# ---------------------------------------------------------------------------
# Sections du rapport
# ---------------------------------------------------------------------------

def _section_header(story: list, title: str, styles: dict) -> None:
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(f"// {title.upper()}", styles["section"]))
    story.append(_hr(_ACCENT, thickness=0.8))


def _build_cover(story: list, report: dict, styles: dict) -> None:
    run_id    = report.get("run_id", "—")
    timestamp = report.get("timestamp", "—")
    status    = report.get("status", "—").upper()

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("FORENSIC IMAGE RECOVERY", styles["title"]))
    story.append(Paragraph("RECONSTRUCTION ANALYSIS REPORT", styles["subtitle"]))
    story.append(_hr(_ACCENT, thickness=1.5))
    story.append(Spacer(1, 4 * mm))

    meta = [
        ["RUN ID",    run_id],
        ["TIMESTAMP", timestamp],
        ["STATUS",    status],
    ]
    t = Table(meta, colWidths=[4 * cm, 13 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Courier"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",  (0, 0), (0, -1), _MUTED),
        ("TEXTCOLOR",  (1, 0), (1, -1), _ACCENT),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 6 * mm))


def _build_images_section(story: list, report: dict, styles: dict, page_w: float) -> None:
    _section_header(story, "Analyse visuelle", styles)

    inp   = report.get("input", {})
    corr  = report.get("corruption", {})
    recon = report.get("reconstruction", {})

    orig_path  = inp.get("source_image")
    corr_path  = corr.get("image_path")
    mask_path  = corr.get("mask_path") or corr.get("detected_mask")
    recon_path = recon.get("best_candidate", {}).get("path") or report.get("reconstructed_image")

    img_w = (page_w - 4 * cm) / 4 - 3 * mm
    img_h = img_w * 0.75

    cells = []
    labels = []
    for label, path in [
        ("Originale", orig_path),
        ("Corrompue", corr_path),
        ("Masque", mask_path),
        ("Reconstruite", recon_path),
    ]:
        img = _load_image_rl(path, img_w, img_h)
        cells.append(img if img else Paragraph("Non disponible", styles["note"]))
        labels.append(Paragraph(label, styles["centered"]))

    img_table = Table([cells, labels], colWidths=[img_w + 2 * mm] * 4)
    img_table.setStyle(TableStyle([
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOX",         (0, 0), (0, 0), 0.5, _ACCENT),
        ("BOX",         (1, 0), (1, 0), 0.5, _ACCENT2),
        ("BOX",         (2, 0), (2, 0), 0.5, _MUTED),
        ("BOX",         (3, 0), (3, 0), 0.5, _SUCCESS),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 4 * mm))


def _build_corruption_section(story: list, report: dict, styles: dict) -> None:
    _section_header(story, "Dégradation appliquée", styles)
    corr = report.get("corruption", {})
    inp  = report.get("input", {})

    rows = [
        ["Type",           corr.get("type") or inp.get("corruption_type") or "—"],
        ["Mode exécution", inp.get("execution_mode") or "—"],
        ["Randomisé",      str(corr.get("randomize", False))],
        ["Masque imparfait", str(corr.get("imperfect_mask", False))],
    ]
    params = corr.get("parameters", {})
    for k, v in params.items():
        if not isinstance(v, (dict, list)):
            rows.append([f"  {k}", str(v)])

    t = Table(rows, colWidths=[5 * cm, 12 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",  (0, 0), (0, -1), _MUTED),
        ("TEXTCOLOR",  (1, 0), (1, -1), _TEXT),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_SURFACE, _BG]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(t)


def _build_reconstruction_section(story: list, report: dict, styles: dict) -> None:
    _section_header(story, "Reconstruction multi-stratégies", styles)
    recon = report.get("reconstruction", {})
    best  = recon.get("best_candidate", {})

    score = float(best.get("score") or recon.get("score") or 0)
    sc    = _score_color(score)

    # Score + stratégie dans un tableau pour éviter le chevauchement
    sc_table = Table(
        [[
            Paragraph(f"{score:.1f}<font size='14' color='#6b6b80'>/100</font>",
                      ParagraphStyle("scn", fontName="Helvetica-Bold", fontSize=28,
                                     textColor=sc, alignment=TA_RIGHT, spaceAfter=0)),
            Paragraph(
                f"<font color='#6b6b80' size='7'>STRATÉGIE SÉLECTIONNÉE</font><br/>"
                f"{best.get('strategy') or recon.get('selected_strategy') or '—'}",
                ParagraphStyle("sc2", fontName="Courier", fontSize=9,
                               textColor=_ACCENT3, alignment=TA_LEFT, spaceAfter=0, leading=14)),
        ]],
        colWidths=[8*cm, 9*cm],
    )
    sc_table.setStyle(TableStyle([
        ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BACKGROUND",    (0, 0), (-1, -1), _SURFACE),
        ("LINEAFTER",     (0, 0), (0, -1), 0.5, _BORDER),
    ]))
    story.append(sc_table)
    story.append(Spacer(1, 4*mm))

    # Métriques best candidate
    metrics_rows = [
        ["PSNR",       _fmt(best.get("psnr"), 1, " dB")],
        ["SSIM",       _fmt(best.get("ssim"), 4)],
        ["Gain PSNR",  _fmt(best.get("gain_psnr"), 2, " dB")],
        ["Gain SSIM",  _fmt(best.get("gain_ssim"), 4)],
        ["Mode scoring", str(best.get("mode") or "—")],
        ["Tentatives", str(recon.get("retry_count", "—"))],
    ]
    t = Table(metrics_rows, colWidths=[5 * cm, 12 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",  (0, 0), (0, -1), _MUTED),
        ("TEXTCOLOR",  (1, 0), (1, -1), _TEXT),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_SURFACE, _BG]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    # Tableau tous les candidats
    candidates = recon.get("all_candidates", [])
    if candidates:
        story.append(Paragraph("Tous les candidats testés :", styles["label"]))
        story.append(Spacer(1, 2 * mm))

        header = ["Stratégie", "Score", "PSNR", "SSIM", "Mode"]
        rows = [header]
        for c in candidates:
            s = float(c.get("score") or 0)
            rows.append([
                str(c.get("strategy") or "—"),
                f"{s:.1f}",
                _fmt(c.get("psnr"), 1, " dB"),
                _fmt(c.get("ssim"), 3),
                str(c.get("mode") or "—"),
            ])

        col_w = [6*cm, 2.5*cm, 2.8*cm, 2.5*cm, 3.2*cm]
        t2 = Table(rows, colWidths=col_w, repeatRows=1)

        style = [
            ("FONTNAME",   (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTNAME",   (0, 1), (-1, -1), "Courier"),
            ("FONTSIZE",   (0, 0), (-1, -1), 7.5),
            ("TEXTCOLOR",  (0, 0), (-1, 0),  _MUTED),
            ("TEXTCOLOR",  (0, 1), (-1, -1), _TEXT),
            ("BACKGROUND", (0, 0), (-1, 0),  _SURFACE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_BG, _SURFACE]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("GRID",          (0, 0), (-1, -1), 0.3, _BORDER),
        ]
        # Surligner la meilleure ligne
        best_strategy = best.get("strategy") or recon.get("selected_strategy")
        for i, c in enumerate(candidates, start=1):
            if c.get("strategy") == best_strategy:
                style.append(("TEXTCOLOR", (0, i), (-1, i), _ACCENT))
                style.append(("FONTNAME",  (0, i), (-1, i), "Courier-Bold"))

        t2.setStyle(TableStyle(style))
        story.append(t2)


def _build_metrics_section(story: list, report: dict, styles: dict) -> None:
    _section_header(story, "Métriques d'évaluation", styles)
    metrics = report.get("metrics", {})

    ovc  = metrics.get("original_vs_corrupted", {})
    ovr  = metrics.get("original_vs_reconstructed", {})
    gains = metrics.get("gains", {})
    det  = metrics.get("detection_metrics", {})

    rows = [
        ["", "PSNR", "SSIM"],
        ["Original vs Corrompu",     _fmt(ovc.get("psnr"), 1, " dB"),  _fmt(ovc.get("ssim"), 4)],
        ["Original vs Reconstruit",  _fmt(ovr.get("psnr"), 1, " dB"),  _fmt(ovr.get("ssim"), 4)],
        ["Gain",
         _fmt(gains.get("psnr_gain"), 2, " dB"),
         _fmt(gains.get("ssim_gain"), 4)],
    ]
    t = Table(rows, colWidths=[7*cm, 4*cm, 4*cm])
    t.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",   (0, 1), (-1, -1), "Courier"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",  (0, 0), (-1, 0),  _MUTED),
        ("TEXTCOLOR",  (0, 1), (-1, -1), _TEXT),
        ("TEXTCOLOR",  (1, 3), (2, 3),   _SUCCESS),
        ("BACKGROUND", (0, 0), (-1, 0),  _SURFACE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_BG, _SURFACE]),
        ("GRID",       (0, 0), (-1, -1), 0.3, _BORDER),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(t)

    if det:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph("Métriques de détection du masque :", styles["label"]))
        det_rows = [[k.upper(), _fmt(v, 3)] for k, v in det.items() if not isinstance(v, dict)]
        if det_rows:
            td = Table(det_rows, colWidths=[5*cm, 12*cm])
            td.setStyle(TableStyle([
                ("FONTNAME",  (0, 0), (-1, -1), "Courier"),
                ("FONTSIZE",  (0, 0), (-1, -1), 7.5),
                ("TEXTCOLOR", (0, 0), (0, -1),  _MUTED),
                ("TEXTCOLOR", (1, 0), (1, -1),  _ACCENT),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING",    (0, 0), (-1, -1), 2),
                ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ]))
            story.append(td)


def _build_analysis_section(story: list, report: dict, styles: dict) -> None:
    _section_header(story, "Analyse et conclusions", styles)
    analysis = report.get("analysis", {})

    dq  = analysis.get("detection_quality", "—")
    re_ = analysis.get("repair_effectiveness", "—")
    sl  = analysis.get("score_level", "—")
    notes = analysis.get("notes", [])

    color_map = {
        "good": _SUCCESS, "medium": colors.HexColor("#ffaa00"),
        "poor": _ACCENT2, "improved": _SUCCESS, "neutral": _MUTED,
        "degraded": _ACCENT2, "excellent": _SUCCESS,
    }

    rows = [
        ["Qualité détection",    dq.upper()],
        ["Efficacité réparation", re_.upper()],
        ["Niveau de score",      sl.upper()],
    ]
    t = Table(rows, colWidths=[6*cm, 11*cm])
    row_styles = [
        ("FONTNAME",  (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), _MUTED),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_SURFACE, _BG]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]
    for i, (_, val) in enumerate(rows):
        c = color_map.get(val.lower(), _TEXT)
        row_styles.append(("TEXTCOLOR", (1, i), (1, i), c))
        row_styles.append(("FONTNAME",  (1, i), (1, i), "Helvetica-Bold"))
    t.setStyle(TableStyle(row_styles))
    story.append(t)

    if notes:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph("Notes automatiques :", styles["label"]))
        for note in notes:
            story.append(Paragraph(f"• {note}", styles["note"]))


def _build_limits_section(story: list, styles: dict) -> None:
    _section_header(story, "Limites du système", styles)
    limits = [
        "En mode aveugle, la détection automatique peut confondre des zones naturellement "
        "sombres (canal, ombres) avec des zones corrompues, générant des faux positifs.",
        "L'inpainting OpenCV (Navier-Stokes / Telea) est efficace sur les petites zones "
        "mais peut produire des artefacts sur les grandes surfaces (>15% de l'image).",
        "Le scoring supervisé (PSNR/SSIM) nécessite l'image originale. Sans elle, "
        "le scoring aveugle est une estimation heuristique.",
        "Les corruptions de type 'shift_region' ou 'block_dropout' sur des images "
        "texturées complexes peuvent donner des reconstructions sous-optimales.",
        "Performance : le pipeline complet (11 stratégies) prend ~2-5s selon la "
        "résolution de l'image et le hardware disponible.",
    ]
    for limit in limits:
        story.append(Paragraph(f"— {limit}", styles["note"]))
        story.append(Spacer(1, 1 * mm))


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def generate_pdf_report(
    report: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """Génère le rapport PDF et retourne son chemin.

    Args:
        report:      Dict retourné par build_report() de json_report.py
        output_path: Chemin de sortie. Si None, sauvegardé dans data/reports/

    Returns:
        Path vers le fichier PDF généré.
    """
    ensure_directories()

    if output_path is None:
        run_id = report.get("run_id", "unknown")
        output_path = REPORTS_DIR / f"report_{run_id}.pdf"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()
    page_w, page_h = A4
    margin = 2 * cm

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title=f"Forensic Report {report.get('run_id', '')}",
        author="Forensic Image Recovery System",
    )

    story: list = []

    # Page de couverture
    _build_cover(story, report, styles)

    # Images côte à côte
    _build_images_section(story, report, styles, page_w - 2 * margin)

    # Dégradation
    _build_corruption_section(story, report, styles)

    # Reconstruction
    _build_reconstruction_section(story, report, styles)

    # Métriques
    _build_metrics_section(story, report, styles)

    # Analyse
    _build_analysis_section(story, report, styles)

    # Limites
    _build_limits_section(story, styles)

    # Pied de page
    story.append(Spacer(1, 6 * mm))
    story.append(_hr(_BORDER))
    story.append(Paragraph(
        f"Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — "
        f"Forensic Image Recovery System v1.0",
        ParagraphStyle("footer", fontName="Helvetica", fontSize=7,
                       textColor=_MUTED, alignment=TA_CENTER),
    ))

    doc.build(story)
    logger.info("Rapport PDF généré : %s", output_path)
    return output_path