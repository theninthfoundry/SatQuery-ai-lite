"""
backend/evidence/spatial_grid.py

Per-cell spatial evidence for a change analysis. Every number in a cell
record is measured from the actual difference array; nothing is templated
or hard-coded.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from backend import config
from backend.imgops import laplacian, to_gray

EVIDENCE_STRENGTH_RULE = (
    "NONE if change% < {min:.0f}. Otherwise STRONG if change% >= {strong:.0f} and retention at the strictest "
    "threshold >= {sret:.2f}; MODERATE if retention >= {mret:.2f}; else WEAK. "
    "A local quality warning demotes the level by one step."
)


def strength_rule_text(cfg: config.ChangeParams = config.CHANGE) -> str:
    return EVIDENCE_STRENGTH_RULE.format(
        min=cfg.min_cell_change_pct, strong=cfg.strong_cell_change_pct,
        sret=cfg.strong_retention, mret=cfg.moderate_retention,
    )


def grid_edges(size: int, n: int) -> np.ndarray:
    return np.linspace(0, size, n + 1).round().astype(int)


def _grade(change_pct: float, retention, quality_warning: bool, cfg: config.ChangeParams) -> str:
    if change_pct < cfg.min_cell_change_pct:
        return "NONE"
    ret = 0.0 if retention is None else retention
    if change_pct >= cfg.strong_cell_change_pct and ret >= cfg.strong_retention:
        level = "STRONG"
    elif ret >= cfg.moderate_retention:
        level = "MODERATE"
    else:
        level = "WEAK"
    if quality_warning:
        level = {"STRONG": "MODERATE", "MODERATE": "WEAK", "WEAK": "WEAK"}[level]
    return level


def build_spatial_grid(
    pixels_a: np.ndarray,
    pixels_b: np.ndarray,
    diff: np.ndarray,
    base_mask: np.ndarray,
    stable: np.ndarray,
    ambiguous: np.ndarray,
    grid_size: int,
    base_threshold: int,
    strict_threshold: int,
    cfg: config.ChangeParams = config.CHANGE,
    qt: config.QualityThresholds = config.QUALITY,
) -> Dict[str, Any]:
    h, w = diff.shape
    gray_a, gray_b = to_gray(pixels_a), to_gray(pixels_b)
    lap_a, lap_b = laplacian(gray_a), laplacian(gray_b)
    ys, xs = grid_edges(h, grid_size), grid_edges(w, grid_size)
    total_changed = int(base_mask.sum())

    cells: List[Dict[str, Any]] = []
    for r in range(grid_size):
        for c in range(grid_size):
            y0, y1, x0, x1 = int(ys[r]), int(ys[r + 1]), int(xs[c]), int(xs[c + 1])
            area = (y1 - y0) * (x1 - x0)
            d = diff[y0:y1, x0:x1]
            changed = int(base_mask[y0:y1, x0:x1].sum())
            stable_n = int(stable[y0:y1, x0:x1].sum())
            amb_n = int(ambiguous[y0:y1, x0:x1].sum())
            pct = changed / area * 100.0 if area else 0.0
            retention = None if changed == 0 else stable_n / changed

            ga, gb = gray_a[y0:y1, x0:x1], gray_b[y0:y1, x0:x1]
            clip_a = float(((ga <= qt.clip_low_level) | (ga >= qt.clip_high_level)).mean() * 100.0)
            clip_b = float(((gb <= qt.clip_low_level) | (gb >= qt.clip_high_level)).mean() * 100.0)
            std_a, std_b = float(ga.std()), float(gb.std())
            la = lap_a[max(0, y0 - 1):max(0, y1 - 1), max(0, x0 - 1):max(0, x1 - 1)]
            lb = lap_b[max(0, y0 - 1):max(0, y1 - 1), max(0, x0 - 1):max(0, x1 - 1)]
            sharp_a = float(la.var()) if la.size else 0.0
            sharp_b = float(lb.var()) if lb.size else 0.0

            reasons: List[str] = []
            if max(clip_a, clip_b) >= qt.clip_high_pct:
                reasons.append(f"saturation: {max(clip_a, clip_b):.0f}% of pixels clipped in at least one image")
            if max(std_a, std_b) < config.ALIGNMENT.min_texture_std and pct > 0:
                reasons.append("featureless in both images: the difference is a brightness shift of a uniform area and may reflect exposure rather than surface")
            warning = bool(reasons)
            strength = _grade(pct, retention, warning, cfg)

            if strength == "NONE":
                interp = (
                    f"No pixel-difference evidence in this region ({pct:.1f}% of pixels exceed threshold {base_threshold})."
                )
            else:
                interp = (
                    f"{'Elevated' if strength in ('STRONG', 'MODERATE') else 'Weak'} pixel difference in this region: "
                    f"{pct:.1f}% of pixels exceed threshold {base_threshold} (mean difference {float(d.mean()):.1f} gray levels); "
                    + (
                        f"{retention * 100:.0f}% of the changed pixels persist at threshold {strict_threshold}."
                        if retention is not None else "no changed pixels persist at the strictest threshold."
                    )
                )
            if warning:
                interp += " Quality caution: " + "; ".join(reasons) + "."

            cells.append({
                "region_id": f"grid_{r * grid_size + c}",
                "row": r,
                "col": c,
                "bbox_px": [x0, y0, x1, y1],
                "bbox_norm": [round(x0 / w, 5), round(y0 / h, 5), round(x1 / w, 5), round(y1 / h, 5)],
                "analysed_pixel_count": int(area),
                "changed_pixel_count": changed,
                "change_percentage": round(pct, 3),
                "share_of_total_change_pct": round(changed / total_changed * 100.0, 3) if total_changed else 0.0,
                "mean_abs_difference": round(float(d.mean()), 3),
                "p95_abs_difference": round(float(np.percentile(d, 95)), 3) if area else 0.0,
                "stable_changed_pixels": stable_n,
                "ambiguous_pixels": amb_n,
                "retention_at_strictest": None if retention is None else round(retention, 4),
                "local_quality": {
                    "clipped_pct_a": round(clip_a, 2), "clipped_pct_b": round(clip_b, 2),
                    "contrast_std_a": round(std_a, 2), "contrast_std_b": round(std_b, 2),
                    "sharpness_a": round(sharp_a, 2), "sharpness_b": round(sharp_b, 2),
                },
                "quality_warning": warning,
                "quality_warning_reasons": reasons,
                "evidence_type": "pixel_intensity_difference",
                "evidence_strength": strength,
                "interpretation": interp,
            })

    return {
        "grid_size": grid_size,
        "cell_count": len(cells),
        "base_threshold": base_threshold,
        "strictest_threshold": strict_threshold,
        "evidence_strength_rule": strength_rule_text(cfg),
        "cells": cells,
    }
