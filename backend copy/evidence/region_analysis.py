"""
backend/evidence/region_analysis.py

How is the change distributed? Concentrated in one place, clustered,
spread evenly, or scattered noise?
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from backend import config
from backend.imgops import cohesion


def location_phrase(cx: float, cy: float) -> str:
    """Normalised centroid (0..1, origin top-left) -> words about the image frame."""
    v = "upper" if cy < 1 / 3 else "middle" if cy < 2 / 3 else "lower"
    h = "left" if cx < 1 / 3 else "center" if cx < 2 / 3 else "right"
    if v == "middle" and h == "center":
        return "the center of the frame"
    if v == "middle":
        return f"the {h} side of the frame"
    if h == "center":
        return f"the {v} part of the frame"
    return f"the {v}-{h} of the frame"


def concentration(
    cells: List[Dict[str, Any]],
    base_mask: np.ndarray,
    cfg: config.ChangeParams = config.CHANGE,
) -> Dict[str, Any]:
    total = int(base_mask.sum())
    n_cells = len(cells)
    if total == 0:
        return {
            "pattern": "NONE", "total_changed_pixels": 0, "dominant_cell": None, "top_cells": [],
            "top_cell_share": 0.0, "top3_share": 0.0, "normalised_entropy": None, "gini": None,
            "spatial_cohesion": None, "scattered": False, "centroid_norm": None, "centroid_phrase": None,
        }
    counts = np.array([c["changed_pixel_count"] for c in cells], dtype=np.float64)
    shares = counts / counts.sum()
    order = np.argsort(-shares)
    top1 = float(shares[order[0]])
    top3 = float(shares[order[:3]].sum())
    nz = shares[shares > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    h_norm = entropy / math.log(n_cells) if n_cells > 1 else 0.0
    srt = np.sort(shares)
    idx = np.arange(1, n_cells + 1)
    gini = float((2 * (idx * srt).sum() / (n_cells * srt.sum())) - (n_cells + 1) / n_cells)

    coh = cohesion(base_mask)
    h, w = base_mask.shape
    ys, xs = np.nonzero(base_mask)
    cx, cy = float(xs.mean() / w), float(ys.mean() / h)

    pattern = (
        "CONCENTRATED" if top1 >= cfg.concentrated_top_cell_share
        else "DISTRIBUTED" if h_norm >= cfg.distributed_entropy
        else "CLUSTERED"
    )
    dominant = cells[int(order[0])]
    return {
        "pattern": pattern,
        "total_changed_pixels": total,
        "dominant_cell": dominant["region_id"],
        "dominant_cell_phrase": location_phrase(
            (dominant["bbox_norm"][0] + dominant["bbox_norm"][2]) / 2,
            (dominant["bbox_norm"][1] + dominant["bbox_norm"][3]) / 2,
        ),
        "top_cells": [cells[int(i)]["region_id"] for i in order[:3] if shares[int(i)] > 0],
        "top_cell_share": round(top1, 4),
        "top3_share": round(top3, 4),
        "normalised_entropy": round(h_norm, 4),
        "gini": round(gini, 4),
        "spatial_cohesion": round(coh, 4),
        "scattered": bool(coh < cfg.cohesion_scattered),
        "centroid_norm": [round(cx, 4), round(cy, 4)],
        "centroid_phrase": location_phrase(cx, cy),
        "rules": (
            f"CONCENTRATED if the top cell holds >= {cfg.concentrated_top_cell_share:.0%} of changed pixels; "
            f"DISTRIBUTED if normalised entropy >= {cfg.distributed_entropy}; otherwise CLUSTERED. "
            f"'Scattered' if cohesion < {cfg.cohesion_scattered} (share of changed pixels with >= 3 changed neighbours)."
        ),
    }
