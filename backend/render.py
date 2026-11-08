"""
backend/render.py

Visual evidence layers for change analysis. Each layer is rendered from the
same working grid so the interface can switch between them (and map grid
cells onto any of them) without re-running the analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from backend.imgops import blend, colormap_difference, desaturate, encode_png_b64

CHANGE_COLOR = (226, 87, 76)
AMBIGUOUS_COLOR = (232, 163, 61)
STABLE_COLOR = (226, 87, 76)
STRENGTH_TINT = {"STRONG": ((226, 87, 76), 0.45), "MODERATE": ((232, 163, 61), 0.32), "WEAK": ((94, 200, 216), 0.22)}


def draw_grid_lines(img: np.ndarray, grid: Dict[str, Any], alpha: float = 0.55) -> np.ndarray:
    out = img.astype(np.float32).copy()
    h, w = img.shape[:2]
    xs, ys = set(), set()
    for c in grid["cells"]:
        x0, y0, x1, y1 = c["bbox_px"]
        xs.update((x0, x1))
        ys.update((y0, y1))
    for x in xs:
        if 0 < x < w:
            out[:, x] = out[:, x] * (1 - alpha) + 255 * alpha
    for y in ys:
        if 0 < y < h:
            out[y, :] = out[y, :] * (1 - alpha) + 255 * alpha
    return out.astype(np.uint8)


def _layer(layer_id: str, label: str, description: str, arr: np.ndarray) -> Dict[str, Any]:
    return {
        "id": layer_id,
        "label": label,
        "description": description,
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "data": encode_png_b64(arr),
    }


def build_change_layers(
    pixels_a: np.ndarray,
    pixels_b: np.ndarray,
    diff: np.ndarray,
    base_mask: np.ndarray,
    stable: np.ndarray,
    ambiguous: np.ndarray,
    grid: Dict[str, Any],
    base_threshold: int,
    strictest_threshold: int,
) -> List[Dict[str, Any]]:
    layers: List[Dict[str, Any]] = []
    layers.append(_layer("before", "Time A", "Earlier image on the shared working grid.", pixels_a))
    layers.append(_layer("after", "Time B", "Later image on the shared working grid (resampled onto A's grid if sizes differ).", pixels_b))
    layers.append(_layer(
        "difference", "Raw difference",
        "Absolute gray-level difference |A-B|, fixed colour scale 0 (dark) to 128+ (bright). No threshold applied.",
        colormap_difference(diff),
    ))
    layers.append(_layer(
        "change_mask", "Change mask",
        f"Pixels with |A-B| > {base_threshold} gray levels, drawn over image B.",
        blend(pixels_b, base_mask, CHANGE_COLOR, 0.65),
    ))

    tint = desaturate(pixels_b, 0.5)
    for c in grid["cells"]:
        spec = STRENGTH_TINT.get(c["evidence_strength"])
        if spec is None:
            continue
        x0, y0, x1, y1 = c["bbox_px"]
        region = np.zeros(base_mask.shape, dtype=bool)
        region[y0:y1, x0:x1] = True
        tint = blend(tint, region, spec[0], spec[1])
    layers.append(_layer(
        "evidence_grid", "Spatial evidence grid",
        "Grid cells tinted by evidence strength: red STRONG, amber MODERATE, cyan WEAK, untinted NONE.",
        draw_grid_lines(tint, grid),
    ))

    unc = desaturate(pixels_b, 0.65)
    unc = blend(unc, ambiguous, AMBIGUOUS_COLOR, 0.7)
    unc = blend(unc, stable, STABLE_COLOR, 0.75)
    layers.append(_layer(
        "uncertain", "Stable vs uncertain",
        f"Red: changed at every tested threshold (|A-B| > {strictest_threshold}). Amber: changed only at some thresholds (threshold-sensitive).",
        unc,
    ))

    comb = blend(pixels_b, ambiguous, AMBIGUOUS_COLOR, 0.45)
    comb = blend(comb, stable, STABLE_COLOR, 0.6)
    layers.append(_layer(
        "combined", "Combined evidence",
        "Image B with stable change (red), threshold-sensitive change (amber) and the evidence grid.",
        draw_grid_lines(comb, grid, 0.4),
    ))
    return layers
