from typing import Any


def clamp_corruption_level(level: int) -> int:
    return max(0, min(100, int(level)))


def build_corruption_profile(level: int) -> dict[str, Any]:
    """
    Traduit un niveau de corruption (0..100) en scénario stable.
    """
    level = clamp_corruption_level(level)

    if level <= 10:
        return {
            "label": "low",
            "corruption_type": "rectangle_mask",
            "corruption_params": {
                "fill_value": 0,
                "min_size": 20,
                "max_size": 40,
            },
            "randomize": True,
            "imperfect_mask": False,
            "use_mask": True,
        }

    if level <= 20:
        return {
            "label": "light",
            "corruption_type": "rectangle_mask",
            "corruption_params": {
                "fill_value": 0,
                "min_size": 40,
                "max_size": 70,
            },
            "randomize": True,
            "imperfect_mask": False,
            "use_mask": True,
        }

    if level <= 40:
        return {
            "label": "moderate",
            "corruption_type": "noise",
            "corruption_params": {
                "sigma": 18.0,
                "min_size": 50,
                "max_size": 90,
            },
            "randomize": True,
            "imperfect_mask": False,
            "use_mask": True,
        }

    if level <= 60:
        return {
            "label": "medium_high",
            "corruption_type": "combined",
            "corruption_params": {
                "fill_value": 0,
                "sigma": 10.0,
                "min_size": 60,
                "max_size": 110,
            },
            "randomize": True,
            "imperfect_mask": False,
            "use_mask": True,
        }

    if level <= 80:
        return {
            "label": "high",
            "corruption_type": "combined",
            "corruption_params": {
                "fill_value": 0,
                "sigma": 16.0,
                "min_size": 80,
                "max_size": 140,
            },
            "randomize": True,
            "imperfect_mask": True,
            "use_mask": False,
        }

    return {
        "label": "extreme",
        "corruption_type": "combined",
        "corruption_params": {
            "fill_value": 0,
            "sigma": 24.0,
            "min_size": 110,
            "max_size": 180,
        },
        "randomize": True,
        "imperfect_mask": True,
        "use_mask": False,
    }