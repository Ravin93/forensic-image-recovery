"""Tests K16 — Timeline forensique."""
from __future__ import annotations
import pytest


class TestTimeline:
    def test_record_event(self):
        from app.modules.timeline.timeline_builder import ForensicTimeline
        tl = ForensicTimeline("test001")
        tl.record("upload", {"filename": "test.png"})
        d = tl.to_dict()
        assert d["n_events"] == 1
        assert d["timeline"][0]["event"] == "upload"

    def test_timestamps_present(self):
        from app.modules.timeline.timeline_builder import ForensicTimeline
        tl = ForensicTimeline()
        tl.record("upload"); tl.record("validation"); tl.record("reconstruction")
        d = tl.to_dict()
        for ev in d["timeline"]:
            assert "time" in ev
            assert "elapsed_s" in ev

    def test_chronological_order(self):
        from app.modules.timeline.timeline_builder import ForensicTimeline
        tl = ForensicTimeline()
        for evt in ["upload","validation","corruption","detection","reconstruction"]:
            tl.record(evt)
        d = tl.to_dict()
        elapsed = [e["elapsed_s"] for e in d["timeline"]]
        assert elapsed == sorted(elapsed)

    def test_has_event(self):
        from app.modules.timeline.timeline_builder import ForensicTimeline
        tl = ForensicTimeline()
        tl.record("upload")
        assert tl.has_event("upload") == True
        assert tl.has_event("download") == False

    def test_get_event(self):
        from app.modules.timeline.timeline_builder import ForensicTimeline
        tl = ForensicTimeline()
        tl.record("upload", {"filename": "img.png"})
        ev = tl.get_event("upload")
        assert ev is not None
        assert ev["details"]["filename"] == "img.png"

    def test_build_from_report(self):
        from app.modules.timeline.timeline_builder import build_timeline_from_report
        report = {
            "run_id": "tl001", "timestamp": "2026-05-12T10:00:00",
            "input": {"source_image": "img.png", "execution_mode": "assisted"},
            "corruption": {"type": "scratch_lines", "parameters": {}, "mask_path": "mask.png"},
            "reconstruction": {
                "best_candidate": {"strategy": "inpainting_r3", "score": 74.0,
                                   "psnr": 28.0, "ssim": 0.87},
                "all_candidates": [{}]
            },
            "metrics": {"detection_metrics": {"iou": 0.9}, "gains": {"psnr_gain": 4.0}},
        }
        d = build_timeline_from_report(report)
        assert d["n_events"] >= 4
        events = [e["event"] for e in d["timeline"]]
        for expected in ("upload","reconstruction","scoring"):
            assert expected in events

    def test_html_section_generated(self):
        from app.modules.timeline.timeline_builder import ForensicTimeline, get_timeline_html_section
        tl = ForensicTimeline("html_test")
        tl.record("upload"); tl.record("reconstruction", {"strategy": "inpainting_r3"})
        html = get_timeline_html_section(tl.to_dict())
        assert "TIMELINE" in html or "timeline" in html.lower()
        assert "upload" in html.lower() or "UPLOAD" in html

    def test_total_time_positive(self):
        from app.modules.timeline.timeline_builder import ForensicTimeline
        import time
        tl = ForensicTimeline()
        tl.record("start")
        time.sleep(0.01)
        tl.record("end")
        assert tl.to_dict()["total_s"] > 0