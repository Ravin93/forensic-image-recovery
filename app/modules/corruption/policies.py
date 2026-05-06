from __future__ import annotations

from typing import Any


def clamp_corruption_level(level: int) -> int:
    return max(0, min(100, int(level)))


def build_corruption_profile(level: int) -> dict[str, Any]:
    """Traduit un niveau 0..100 en scénario de corruption plus réaliste."""
    level = clamp_corruption_level(level)

    if level <= 10:
        return {
            "label": "very_low",
            "corruption_type": "rectangle_mask",
            "corruption_params": {"fill_value": 0, "min_size": 20, "max_size": 40},
            "randomize": True,
            "imperfect_mask": False,
            "use_mask": True,
        }
    if level <= 20:
        return {
            "label": "low",
            "corruption_type": "bar",
            "corruption_params": {"orientation": "horizontal", "thickness": 8, "count": 1, "fill_mode": "black"},
            "randomize": False,
            "imperfect_mask": False,
            "use_mask": True,
        }
    if level <= 35:
        return {
            "label": "moderate",
            "corruption_type": "zone_deletion",
            "corruption_params": {"fill_mode": "white", "min_size": 30, "max_size": 80},
            "randomize": True,
            "imperfect_mask": False,
            "use_mask": True,
        }
    if level <= 50:
        return {
            "label": "moderate_plus",
            "corruption_type": "local_blur",
            "corruption_params": {"kernel_size": 11, "min_size": 50, "max_size": 120},
            "randomize": True,
            "imperfect_mask": False,
            "use_mask": True,
        }
    if level <= 65:
        return {
            "label": "high",
            "corruption_type": "jpeg_block_artifacts",
            "corruption_params": {"block_size": 8, "min_size": 70, "max_size": 140},
            "randomize": True,
            "imperfect_mask": True,
            "use_mask": True,
        }
    if level <= 80:
        return {
            "label": "very_high",
            "corruption_type": "block_dropout",
            "corruption_params": {"block_size": 16, "drop_ratio": 0.25, "fill_mode": "black"},
            "randomize": False,
            "imperfect_mask": True,
            "use_mask": False,
        }
    return {
        "label": "extreme",
        "corruption_type": "mixed",
        "corruption_params": {},
        "randomize": False,
        "imperfect_mask": True,
        "use_mask": False,
    }
