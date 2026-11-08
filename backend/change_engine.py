"""
backend/change_engine.py

Change Intelligence Engine: pixel-difference change detection wrapped in
the diagnostics, spatial evidence and uncertainty analysis that make its
output interpretable.

Input contract: two (H, W, 3) uint8 arrays ALREADY on the same working
grid. Resampling decisions (and their provenance) belong to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from backend import config
from backend.diagnostics import _overlap_crops
from backend.evidence import evidence_summary
from backend.evidence.region_analysis import concentration
from backend.evidence.spatial_grid import build_spatial_grid
from backend.imgops import noise_sigma, to_gray
from backend.render import build_change_layers
from backend.uncertainty.ambiguity import ambiguity_summary, noise_floor_check
from backend.uncertainty.sensitivity import run_sensitivity
from backend.uncertainty.stability import stability_masks

METHOD_NAME = "pixel_intensity_difference"


@dataclass
class ChangeAnalysis:
    evidence: Dict[str, Any]
    layers: List[Dict[str, Any]]
    base_mask: np.ndarray


def analyze_change(
    pixels_a: np.ndarray,
    pixels_b: np.ndarray,
    alignment: Dict[str, Any],
    thresholds: List[int],
    base_threshold: int,
    grid_size: int,
    cfg: config.ChangeParams = config.CHANGE,
    render_layers: bool = True,
) -> ChangeAnalysis:
    if pixels_a.shape != pixels_b.shape:
        raise ValueError("analyze_change requires both images on the same working grid")
    h, w = pixels_a.shape[:2]
    n = h * w

    gray_a, gray_b = to_gray(pixels_a), to_gray(pixels_b)
    signed = gray_b - gray_a
    diff = np.abs(signed)

    base_mask = diff > base_threshold
    t_min, t_max = min(thresholds), max(thresholds)
    stable, ambiguous = stability_masks(diff, t_min, t_max)

    changed = int(base_mask.sum())
    measurements = {
        "method": METHOD_NAME,
        "base_threshold": int(base_threshold),
        "analysed_pixels": int(n),
        "changed_pixels": changed,
        "changed_pct": round(changed / n * 100.0, 4),
        "mean_abs_difference": round(float(diff.mean()), 3),
        "p95_abs_difference": round(float(np.percentile(diff, 95)), 3),
        "max_abs_difference": round(float(diff.max()), 3),
        "gray_definition": config.PROCESSING.gray_definition,
    }

    sensitivity = run_sensitivity(diff, thresholds, base_threshold, cfg)
    ambiguity = ambiguity_summary(stable, ambiguous, base_mask)
    grid = build_spatial_grid(
        pixels_a, pixels_b, diff, base_mask, stable, ambiguous,
        grid_size, base_threshold, t_max, cfg,
    )
    conc = concentration(grid["cells"], base_mask, cfg)

    sigma_a, sigma_b = noise_sigma(gray_a), noise_sigma(gray_b)
    noise_floor = noise_floor_check(sigma_a, sigma_b, base_threshold, cfg)

    median_signed = float(np.median(signed))
    corrected = np.abs(signed - median_signed)
    radiometric = {
        "median_signed_difference": round(median_signed, 3),
        "global_offset_flag": bool(abs(median_signed) >= cfg.global_offset_warn),
        "flag_threshold": cfg.global_offset_warn,
        "offset_corrected_changed_pct": round(float((corrected > base_threshold).mean() * 100.0), 4),
        "note": "Diagnostic only: the reported measurement is NOT offset-corrected.",
    }

    alignment_impact: Dict[str, Any] = {"shift_compensated_changed_pct": None}
    if alignment.get("dx_px") is not None:
        idy, idx = int(round(alignment["dy_px"])), int(round(alignment["dx_px"]))
        if (idy, idx) != (0, 0):
            ca, cb = _overlap_crops(gray_a, gray_b, idy, idx)
            if ca.size:
                alignment_impact = {
                    "shift_compensated_changed_pct": round(float((np.abs(cb - ca) > base_threshold).mean() * 100.0), 4),
                    "applied_shift_px": [idx, idy],
                    "note": "What-if diagnostic over the overlapping region; the reported measurement is not shift-corrected.",
                }

    evidence: Dict[str, Any] = {
        "kind": "change_analysis",
        "measurements": measurements,
        "spatial_evidence": grid,
        "concentration": conc,
        "sensitivity": sensitivity,
        "stability": ambiguity,
        "noise_floor": noise_floor,
        "radiometric": radiometric,
        "alignment_impact": alignment_impact,
    }
    evidence["statements"] = evidence_summary.change_statements(
        measurements, conc, sensitivity, ambiguity, alignment, noise_floor, radiometric, alignment_impact,
    )

    layers = (
        build_change_layers(pixels_a, pixels_b, diff, base_mask, stable, ambiguous, grid, base_threshold, t_max)
        if render_layers else []
    )
    return ChangeAnalysis(evidence=evidence, layers=layers, base_mask=base_mask)
