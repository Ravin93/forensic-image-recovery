"""app/modules/reporting/html_report.py — E4.

Genere un rapport HTML autonome (self-contained).
"""
from __future__ import annotations
import base64
from datetime import datetime
from pathlib import Path
from typing import Any
from app.core.config import REPORTS_DIR, ensure_directories
from app.core.logger import logger


def _img_to_b64(path) -> str | None:
    if path is None: return None
    p = Path(path)
    if not p.exists(): return None
    try:
        suffix = p.suffix.lower().lstrip(".")
        mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png"}.get(suffix,"image/png")
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        return "data:" + mime + ";base64," + data
    except Exception: return None


def _fmt(value, decimals: int = 2, suffix: str = "") -> str:
    if value is None: return "—"
    try: return f"{float(value):.{decimals}f}{suffix}"
    except: return str(value)


def _score_color(score: float) -> str:
    if score >= 80: return "#4d7c5f"
    if score >= 60: return "#0d9488"
    if score >= 40: return "#7d6b2f"
    return "#9b4444"


def _qual_class(val: str) -> str:
    v = (val or "").lower()
    if v in ("good","improved","excellent"): return "qual-good"
    if v in ("medium","neutral"): return "qual-medium"
    return "qual-poor"


def _pill_class(score: float) -> str:
    if score >= 70: return "pill pill-h"
    if score >= 40: return "pill pill-m"
    return "pill pill-l"


def _img_tag(b64: str | None, alt: str = "") -> str:
    if b64:
        return '<img src="' + b64 + '" alt="' + alt + '" loading="lazy">'
    return '<div class="img-placeholder">Non disponible</div>'

_CSS = """
:root{
  --bg:#f5f0e8;--surface:#edeae3;--surface2:#e4e0d8;--border:#cfc9be;
  --text:#1e1c18;--muted:#6b6358;
  --accent:#0d9488;--accent-bg:#e6f4f3;--accent-dk:#0f766e;
  --success:#4d7c5f;--error:#9b4444;--warn:#7d6b2f;
  --sans:system-ui,-apple-system,'Segoe UI',Arial,sans-serif;
  --mono:'Courier New',Courier,monospace
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--sans);line-height:1.6;padding:2rem;max-width:1100px;margin:0 auto}
h1{font-size:1.6rem;font-weight:800;color:var(--text);margin-bottom:.25rem;letter-spacing:-.01em}
h2{font-family:var(--sans);font-size:.62rem;font-weight:700;color:var(--accent);letter-spacing:.16em;text-transform:uppercase;margin:2rem 0 .75rem;padding-bottom:.35rem;border-bottom:1px solid var(--border)}
h3{font-size:.7rem;color:var(--muted);font-family:var(--mono);letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem}
.meta{font-family:var(--mono);font-size:.72rem;color:var(--muted);margin-bottom:1.5rem;display:flex;flex-wrap:wrap;gap:.3rem 1.5rem}
.meta span{color:var(--accent)}
.brand{font-size:.62rem;color:var(--muted);margin-bottom:1.5rem;display:flex;align-items:center;gap:.5rem}
.brand-dot{width:8px;height:8px;background:var(--accent);border-radius:3px;display:inline-block}
.images{display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;margin-bottom:1rem}
.img-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.img-label{font-family:var(--mono);font-size:.58rem;color:var(--muted);padding:.35rem .65rem;border-bottom:1px solid var(--border);letter-spacing:.08em;text-transform:uppercase;display:flex;gap:.35rem;align-items:center;background:var(--surface2)}
.dot{width:6px;height:6px;border-radius:50%;display:inline-block;flex-shrink:0}
.img-card img{width:100%;display:block;max-height:180px;object-fit:cover;background:var(--surface2)}
.img-placeholder{width:100%;height:120px;background:var(--surface2);display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:.62rem;color:var(--muted)}
.score-box{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.25rem 1.75rem;display:grid;grid-template-columns:auto 1fr auto;gap:1.5rem;align-items:center;margin-bottom:1rem}
.score-num{font-family:var(--mono);font-size:2.4rem;font-weight:700}
.score-label{font-family:var(--mono);font-size:.58rem;color:var(--muted);letter-spacing:.12em;margin-bottom:.25rem;text-transform:uppercase}
.bar-track{height:4px;background:var(--border);border-radius:2px;overflow:hidden}
.bar-fill{height:100%;background:var(--accent);border-radius:2px}
.strategy{font-family:var(--mono);font-size:.68rem;color:var(--accent-dk);border:1px solid var(--accent);border-radius:999px;padding:.25rem .75rem;background:var(--accent-bg)}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.7rem;margin-bottom:1rem}
th{text-align:left;padding:.45rem .75rem;color:var(--muted);font-size:.6rem;letter-spacing:.1em;border-bottom:1px solid var(--border);text-transform:uppercase;background:var(--surface2)}
td{padding:.4rem .75rem;border-bottom:1px solid var(--border);color:var(--text)}
tr.best td{color:var(--accent-dk);font-weight:600}
tr:last-child td{border-bottom:none}
.pill{display:inline-block;padding:.1rem .5rem;border-radius:999px;font-size:.62rem;font-weight:600}
.pill-h{background:rgba(77,124,95,.15);color:#4d7c5f}
.pill-m{background:rgba(125,107,47,.15);color:#7d6b2f}
.pill-l{background:rgba(155,68,68,.12);color:#9b4444}
.metrics-grid{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-bottom:1rem}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.9rem 1.1rem}
.metric-row{display:flex;justify-content:space-between;align-items:center;padding:.3rem 0;border-bottom:1px solid var(--border);font-size:.78rem}
.metric-row:last-child{border-bottom:none}
.metric-key{color:var(--muted);font-family:var(--mono);font-size:.67rem}
.metric-val{color:var(--text);font-family:var(--mono);font-size:.75rem}
.metric-gain{color:var(--success);font-family:var(--mono);font-size:.67rem;font-weight:600}
.analysis-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem 1.4rem;margin-bottom:1rem}
.qual-badge{display:inline-block;padding:.2rem .65rem;border-radius:999px;font-family:var(--mono);font-size:.62rem;letter-spacing:.08em;text-transform:uppercase;margin-right:.4rem;font-weight:600}
.qual-good{background:rgba(77,124,95,.12);color:#4d7c5f;border:1px solid #4d7c5f}
.qual-medium{background:rgba(125,107,47,.12);color:#7d6b2f;border:1px solid #7d6b2f}
.qual-poor{background:rgba(155,68,68,.10);color:#9b4444;border:1px solid #9b4444}
.notes{margin-top:.75rem;list-style:none}
.notes li{font-size:.8rem;color:var(--text);margin-bottom:.3rem;padding-left:1rem;line-height:1.5}
.notes li::before{content:"▸ ";color:var(--accent)}
.limits{list-style:none;margin-top:.5rem}
.limits li{font-size:.78rem;color:var(--muted);padding:.45rem .8rem;border-left:2px solid var(--warn);margin-bottom:.4rem;background:var(--surface);border-radius:0 6px 6px 0;line-height:1.55}
.limits li::before{content:"⚠ ";color:var(--warn)}
footer{margin-top:2.5rem;padding-top:.9rem;border-top:1px solid var(--border);font-family:var(--mono);font-size:.6rem;color:var(--muted);text-align:center}
@media(max-width:800px){.images{grid-template-columns:1fr 1fr}.metrics-grid{grid-template-columns:1fr}.score-box{grid-template-columns:1fr}}
"""


def generate_html_report(report: dict, output_path=None) -> Path:
    """Genere le rapport HTML self-contained et retourne son chemin."""
    ensure_directories()
    if output_path is None:
        run_id = report.get("run_id","unknown")
        output_path = REPORTS_DIR / ("report_" + run_id + ".html")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_id    = report.get("run_id","—")
    timestamp = report.get("timestamp","—")
    status    = report.get("status","—").upper()
    inp       = report.get("input",{})
    corr      = report.get("corruption",{})
    recon     = report.get("reconstruction",{})
    met       = report.get("metrics",{})
    ana       = report.get("analysis",{})

    orig_b64  = _img_to_b64(inp.get("source_image"))
    corr_b64  = _img_to_b64(corr.get("image_path"))
    mask_b64  = _img_to_b64(corr.get("mask_path") or corr.get("detected_mask"))
    best      = recon.get("best_candidate",{})
    recon_b64 = _img_to_b64(best.get("path"))

    ct_rows = ""
    ct_type   = corr.get("type") or inp.get("corruption_type") or "—"
    exec_mode = inp.get("execution_mode") or "—"
    ct_rows += '<div class="metric-row"><span class="metric-key">Type</span><span class="metric-val">' + ct_type + "</span></div>"
    ct_rows += '<div class="metric-row"><span class="metric-key">Mode</span><span class="metric-val">' + exec_mode + "</span></div>"
    for k, v in (corr.get("parameters") or {}).items():
        if not isinstance(v,(dict,list)):
            ct_rows += '<div class="metric-row"><span class="metric-key">' + str(k) + '</span><span class="metric-val">' + str(v) + "</span></div>"

    score       = float(best.get("score") or recon.get("score") or 0)
    strategy    = recon.get("selected_strategy") or best.get("strategy") or "—"
    retry_count = recon.get("retry_count", len(recon.get("all_candidates",[])))

    candidates = recon.get("all_candidates",[])
    cand_rows = ""
    for c in candidates:
        s      = float(c.get("score") or 0)
        is_best = c.get("strategy") == strategy
        cls    = 'class="best"' if is_best else ""
        pill   = _pill_class(s)
        prefix = "▶ " if is_best else ""
        cand_rows += ("<tr " + cls + ">"
            + "<td>" + prefix + str(c.get("strategy") or "—") + "</td>"
            + '<td><span class="' + pill + '">' + f"{s:.1f}" + "</span></td>"
            + "<td>" + _fmt(c.get("psnr"),1," dB") + "</td>"
            + "<td>" + _fmt(c.get("ssim"),4) + "</td>"
            + '<td style="color:var(--muted)">' + str(c.get("mode") or "—") + "</td>"
            + "</tr>")

    ovc   = met.get("original_vs_corrupted",{})
    ovr   = met.get("original_vs_reconstructed",{})
    gains = met.get("gains",{})
    met_rows = (
        '<div class="metric-row"><span class="metric-key">PSNR corrompu</span><span class="metric-val">' + _fmt(ovc.get("psnr"),1," dB") + "</span></div>"
        + '<div class="metric-row"><span class="metric-key">SSIM corrompu</span><span class="metric-val">' + _fmt(ovc.get("ssim"),4) + "</span></div>"
        + '<div class="metric-row"><span class="metric-key">PSNR reconstruit</span><span class="metric-val">' + _fmt(ovr.get("psnr"),1," dB") + "</span></div>"
        + '<div class="metric-row"><span class="metric-key">SSIM reconstruit</span><span class="metric-val">' + _fmt(ovr.get("ssim"),4) + "</span></div>"
        + '<div class="metric-row"><span class="metric-key">Gain PSNR</span><span class="metric-gain">+' + _fmt(gains.get("psnr_gain"),2," dB") + "</span></div>"
        + '<div class="metric-row"><span class="metric-key">Gain SSIM</span><span class="metric-gain">+' + _fmt(gains.get("ssim_gain"),4) + "</span></div>"
    )

    det = met.get("detection_metrics",{})
    det_rows = ""
    for k, v in det.items():
        if not isinstance(v,dict):
            det_rows += '<div class="metric-row"><span class="metric-key">' + k.upper() + '</span><span class="metric-val">' + _fmt(v,3) + "</span></div>"

    dq  = ana.get("detection_quality","—")
    re_ = ana.get("repair_effectiveness","—")
    sl  = ana.get("score_level","—")
    notes = ana.get("notes",[])
    notes_html = "".join("<li>" + n + "</li>" for n in notes) or "<li>Aucune note.</li>"

    score_str = f"{score:.1f}"
    score_col = _score_color(score)

    parts = [
        "<!DOCTYPE html>\n<html lang='fr'>\n<head>\n",
        "<meta charset='UTF-8'>\n",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>\n",
        "<title>Forensic Report " + run_id + "</title>\n",
        "<style>" + _CSS + "</style>\n",
        "</head>\n<body>\n",
        "<h1>FORENSIC IMAGE RECOVERY</h1>\n",
        '<p class="meta"><span>RUN ID : ' + run_id + "</span>",
        "<span>DATE : " + timestamp + "</span>",
        "<span>STATUT : " + status + "</span></p>\n",
        "<h2>// Analyse visuelle</h2>\n",
        '<div class="images">',
        '<div class="img-card"><div class="img-label"><span class="dot" style="background:#00e5ff"></span>Originale</div>' + _img_tag(orig_b64,"Originale") + "</div>",
        '<div class="img-card"><div class="img-label"><span class="dot" style="background:#ff3d71"></span>Corrompue</div>' + _img_tag(corr_b64,"Corrompue") + "</div>",
        '<div class="img-card"><div class="img-label"><span class="dot" style="background:#6b6b80"></span>Masque</div>' + _img_tag(mask_b64,"Masque") + "</div>",
        '<div class="img-card"><div class="img-label"><span class="dot" style="background:#00e096"></span>Reconstruite</div>' + _img_tag(recon_b64,"Reconstruite") + "</div>",
        "</div>\n",
        "<h2>// Dégradation appliquée</h2>\n",
        '<div class="metric-card" style="margin-bottom:1rem">' + ct_rows + "</div>\n",
        "<h2>// Reconstruction — Score global</h2>\n",
        '<div class="score-box">',
        "<div><div class='score-label'>SCORE GLOBAL</div>",
        "<div class='score-num' style='color:" + score_col + "'>" + score_str + "<span style='font-size:1.2rem;opacity:.4'>/100</span></div></div>",
        "<div><div class='score-label'>Qualité de reconstruction</div>",
        "<div class='bar-track'><div class='bar-fill' style='width:" + score_str + "%'></div></div>",
        "<div style='font-family:monospace;font-size:.65rem;color:var(--muted);margin-top:.5rem'>" + str(retry_count) + " stratégie(s) testée(s)</div></div>",
        "<div style='text-align:right'><div class='score-label'>Stratégie sélectionnée</div>",
        "<div class='strategy'>" + str(strategy) + "</div></div></div>\n",
        "<h2>// Candidats testés</h2>\n",
        "<table><thead><tr><th>Stratégie</th><th>Score</th><th>PSNR</th><th>SSIM</th><th>Mode</th></tr></thead><tbody>",
        cand_rows + "</tbody></table>\n",
        "<h2>// Métriques d'évaluation</h2>\n",
        '<div class="metrics-grid"><div class="metric-card"><h3>Comparaison globale</h3>' + met_rows + "</div>",
        '<div class="metric-card"><h3>Détection du masque</h3>' + det_rows + "</div></div>\n",
        "<h2>// Analyse et conclusions</h2>\n",
        '<div class="analysis-card"><div style="margin-bottom:.8rem">',
        '<span class="qual-badge ' + _qual_class(dq)  + '">Détection : ' + dq.upper()  + "</span>",
        '<span class="qual-badge ' + _qual_class(re_) + '">Réparation : ' + re_.upper() + "</span>",
        '<span class="qual-badge ' + _qual_class(sl)  + '">Score : ' + sl.upper()  + "</span></div>",
        '<ul class="notes">' + notes_html + "</ul></div>\n",
        "<h2>// Limites du système</h2>\n",
        '<ul class="limits">',
        "<li>En mode aveugle, la détection peut confondre zones sombres naturelles et zones corrompues.</li>",
        "<li>L'inpainting OpenCV est efficace sur les petites zones, artefacts possibles au-delà de 15%.</li>",
        "<li>Le scoring supervisé nécessite l'image originale. Sans elle, score heuristique.</li>",
        "<li>Corruptions complexes (shift_region, block_dropout) peuvent donner des résultats sous-optimaux.</li>",
        "<li>Pipeline complet : 2–5s selon résolution et hardware.</li>",
        "</ul>\n",
        "<footer>Généré le " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " — Forensic Image Recovery System v1.0</footer>\n",
        "</body>\n</html>",
    ]

    html = "".join(parts)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Rapport HTML genere : %s", output_path)
    return output_path