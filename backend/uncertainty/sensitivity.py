"""
backend/uncertainty/sensitivity.py

Threshold sensitivity analysis for pixel-difference change detection.

A single threshold gives a single, falsely definitive number. This module
re-runs the same comparison across several thresholds and reports how much
the answer moves.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from backend import config


class ThresholdError(ValueError):
    pass


def normalise_thresholds(
    base: int,
    thresholds: Optional[Iterable[int]] = None,
    limits: config.Limits = config.LIMITS,
) -> List[int]:
    """Validate and return the sorted, de-duplicated threshold list, always including `base`."""
    if not (1 <= int(base) <= 254):
        raise ThresholdError("base_threshold must be between 1 and 254 gray levels")
    values = list(config.CHANGE.sensitivity_thresholds if thresholds is None else thresholds)
    for t in values:
        if isinstance(t, bool) or not isinstance(t, (int, np.integer)):
            raise ThresholdError("thresholds must be integers")
        if not (1 <= int(t) <= 254):
            raise ThresholdError("each threshold must be between 1 and 254 gray levels")
    values = sorted({int(t) for t in values} | {int(base)})
    if len(values) < limits.min_thresholds:
        raise ThresholdError(f"at least {limits.min_thresholds} distinct thresholds are required")
    if len(values) > limits.max_thresholds:
        raise ThresholdError(f"at most {limits.max_thresholds} thresholds are allowed")
    return values


def run_sensitivity(
    diff: np.ndarray,
    thresholds: List[int],
    base: int,
    cfg: config.ChangeParams = config.CHANGE,
) -> Dict[str, Any]:
    n = diff.size
    base_mask = diff > base
    base_count = int(base_mask.sum())
    rows: List[Dict[str, Any]] = []
    for t in thresholds:
        m = diff > t
        count = int(m.sum())
        union = int((m | base_mask).sum())
        inter = int((m & base_mask).sum())
        rows.append({
            "threshold": int(t),
            "is_base": int(t) == int(base),
            "changed_pixels": count,
            "changed_pct": round(count / n * 100.0, 4),
            "delta_vs_base_pp": round((count - base_count) / n * 100.0, 4),
            "relative_to_base": None if base_count == 0 else round(count / base_count, 4),
            "iou_with_base": 1.0 if union == 0 else round(inter / union, 4),
        })
    pcts = [r["changed_pct"] for r in rows]
    lo, hi, mean = min(pcts), max(pcts), float(np.mean(pcts))
    if hi == 0:
        index, level = 0.0, "NO_CHANGE"
    else:
        index = (hi - lo) / mean
        level = "LOW" if index < cfg.sensitivity_low else "HIGH" if index >= cfg.sensitivity_high else "MODERATE"
    return {
        "thresholds": rows,
        "base_threshold": int(base),
        "changed_pct_min": round(lo, 4),
        "changed_pct_max": round(hi, 4),
        "changed_pct_range_pp": round(hi - lo, 4),
        "sensitivity_index": round(float(index), 4),
        "sensitivity_level": level,
        "definition": "sensitivity_index = (max - min) / mean of changed-area percentage across the tested thresholds",
        "level_rule": f"LOW < {cfg.sensitivity_low} <= MODERATE < {cfg.sensitivity_high} <= HIGH",
    }
