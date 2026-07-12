"""app/modules/timeline/timeline_builder.py — K16.

Reconstruction de la timeline forensique d une analyse.
Enregistre chaque evenement avec timestamp, duree et details.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.logger import logger

_EVENTS = [
    "upload", "validation", "corruption", "detection",
    "reconstruction", "scoring", "report_generation",
    "download",
]


class ForensicTimeline:
    """Collecteur d evenements forensiques avec timestamps."""

    def __init__(self, analysis_id: str = ""):
        self.analysis_id = analysis_id
        self._events: list[dict[str, Any]] = []
        self._t_start = time.perf_counter()

    def record(
        self,
        event: str,
        details: dict[str, Any] | None = None,
        duration_s: float | None = None,
    ) -> None:
        """Enregistre un evenement dans la timeline."""
        entry: dict[str, Any] = {
            "event":      event,
            "time":       datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "elapsed_s":  round(time.perf_counter() - self._t_start, 3),
            "details":    details or {},
        }
        if duration_s is not None:
            entry["duration_s"] = round(duration_s, 3)
        self._events.append(entry)
        logger.debug("Timeline [%s] %s", self.analysis_id, event)

    def to_dict(self) -> dict[str, Any]:
        """Retourne la timeline complete sous forme de dict."""
        total = time.perf_counter() - self._t_start
        return {
            "analysis_id":  self.analysis_id,
            "started_at":   self._events[0]["time"] if self._events else None,
            "total_s":      round(total, 3),
            "n_events":     len(self._events),
            "timeline":     self._events,
        }

    def get_event(self, event_name: str) -> dict[str, Any] | None:
        """Retourne le dernier evenement avec ce nom."""
        for e in reversed(self._events):
            if e["event"] == event_name:
                return e
        return None

    def has_event(self, event_name: str) -> bool:
        return any(e["event"] == event_name for e in self._events)


def build_timeline_from_report(report: dict[str, Any]) -> dict[str, Any]:
    """Reconstruit une timeline a partir d un rapport JSON existant.

    Utile quand on n a pas capture la timeline en temps reel.
    """
    tl = ForensicTimeline(analysis_id=report.get("run_id", ""))
    ts = report.get("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))

    inp   = report.get("input", {})
    corr  = report.get("corruption", {})
    recon = report.get("reconstruction", {})
    met   = report.get("metrics", {})

    tl.record("upload", {
        "filename": inp.get("source_image", ""),
        "mode":     inp.get("execution_mode", ""),
    })

    tl.record("validation", {
        "format": "JPEG/PNG",
        "status": "ok",
    })

    ct = corr.get("type", "")
    if ct:
        tl.record("corruption", {
            "type":       ct,
            "parameters": corr.get("parameters", {}),
        })

    tl.record("detection", {
        "mask_path": corr.get("mask_path", ""),
        "iou":       met.get("detection_metrics", {}).get("iou"),
    })

    best = recon.get("best_candidate", {})
    n_cand = len(recon.get("all_candidates", []))
    tl.record("reconstruction", {
        "strategy":    best.get("strategy", ""),
        "n_candidates": n_cand,
        "score":       best.get("score"),
    })

    tl.record("scoring", {
        "psnr":      best.get("psnr"),
        "ssim":      best.get("ssim"),
        "gain_psnr": met.get("gains", {}).get("psnr_gain"),
    })

    if report.get("run_id"):
        tl.record("report_generation", {
            "run_id":  report["run_id"],
            "formats": ["json", "pdf", "html"],
        })

    return tl.to_dict()


def get_timeline_html_section(timeline: dict[str, Any]) -> str:
    """Genere la section HTML de la timeline pour le rapport."""
    events = timeline.get("timeline", [])
    if not events:
        return ""

    rows = ""
    for ev in events:
        rows += (
            "<tr>"
            + "<td style='color:#00e5ff;font-family:monospace;font-size:.65rem'>"
            + str(ev.get("time","")) + "</td>"
            + "<td style='font-family:monospace;font-size:.7rem'>"
            + str(ev.get("event","")).upper() + "</td>"
            + "<td style='font-family:monospace;font-size:.65rem;color:#6b6b80'>"
            + str(round(ev.get("elapsed_s",0),3)) + "s</td>"
            + "<td style='font-size:.72rem'>"
            + ", ".join(f"{k}={v}" for k,v in ev.get("details",{}).items()
                        if v is not None)[:60] + "</td>"
            + "</tr>"
        )

    return (
        "<div style='background:#111118;border:1px solid #2a2a3a;padding:1rem;margin:1rem 0'>"
        + "<div style='font-family:monospace;font-size:.7rem;color:#00e5ff;margin-bottom:.5rem'>"
        + "// TIMELINE FORENSIQUE (" + str(len(events)) + " evenements)"
        + " — duree totale : " + str(timeline.get("total_s",0)) + "s</div>"
        + "<table style='width:100%;border-collapse:collapse;font-size:.72rem'>"
        + "<thead><tr>"
        + "<th style='text-align:left;color:#6b6b80;padding:.3rem;border-bottom:1px solid #2a2a3a'>TIMESTAMP</th>"
        + "<th style='text-align:left;color:#6b6b80;padding:.3rem;border-bottom:1px solid #2a2a3a'>EVENEMENT</th>"
        + "<th style='text-align:left;color:#6b6b80;padding:.3rem;border-bottom:1px solid #2a2a3a'>T+</th>"
        + "<th style='text-align:left;color:#6b6b80;padding:.3rem;border-bottom:1px solid #2a2a3a'>DETAILS</th>"
        + "</tr></thead><tbody>" + rows + "</tbody></table></div>"
    )