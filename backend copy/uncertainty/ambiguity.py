"""
backend/uncertainty/ambiguity.py

Summaries of stable vs threshold-sensitive (ambiguous) change, and a check
of the chosen threshold against the estimated sensor/compression noise.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from backend import config


def ambiguity_summary(stable: np.ndarray, ambiguous: np.ndarray, base_mask: np.ndarray) -> Dict[str, Any]:
    n = stable.size
    s, a, b = int(stable.sum()), int(ambiguous.sum()), int(base_mask.sum())
    retained = int((base_mask & stable).sum())
    return {
        "stable_pixels": s,
        "stable_pct": round(s / n * 100.0, 4),
        "ambiguous_pixels": a,
        "ambiguous_pct": round(a / n * 100.0, 4),
        "base_changed_pixels": b,
        "base_retained_at_strictest": retained,
        "retention_of_base": None if b == 0 else round(retained / b, 4),
        "ambiguous_share_of_all_changed": None if (s + a) == 0 else round(a / (s + a), 4),
        "note": "Threshold stability is robustness to the threshold choice, not proof of real-world change.",
    }


def noise_floor_check(
    sigma_a: float,
    sigma_b: float,
    base_threshold: int,
    cfg: config.ChangeParams = config.CHANGE,
) -> Dict[str, Any]:
    """Is the base threshold comfortably above the expected noise of |A-B|?"""
    sigma_diff = math.hypot(sigma_a, sigma_b)
    multiple = None if sigma_diff < 1e-9 else base_threshold / sigma_diff
    below = multiple is not None and multiple < cfg.noise_floor_multiplier
    return {
        "sigma_a": round(sigma_a, 3),
        "sigma_b": round(sigma_b, 3),
        "sigma_difference": round(sigma_diff, 3),
        "base_threshold": base_threshold,
        "threshold_over_sigma": None if multiple is None else round(multiple, 3),
        "required_multiple": cfg.noise_floor_multiplier,
        "threshold_within_noise": bool(below),
        "note": (
            "Noise is an estimate (Immerkaer) and is biased upward on strongly textured scenes."
        ),
    }
