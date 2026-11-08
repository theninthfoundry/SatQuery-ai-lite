"""
backend/tools/change_tool.py

Change detection and comparative change questions for bi-temporal pairs.
Both now run on the Change Intelligence Engine (backend/change_engine.py),
so every answer comes with alignment-aware, spatially resolved,
threshold-sensitivity-tested evidence.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from backend import config
from backend.change_engine import analyze_change
from backend.controller import ToolResult, detect_keyword
from backend.diagnostics import estimate_alignment
from backend.evidence.evidence_summary import change_headline
from backend.image_utils import CLASS_LABELS, compute_landcover_stats
from backend.imgops import to_gray
from backend.uncertainty.sensitivity import normalise_thresholds

TOOL_NAME_DESCRIBE = "RS-ChangeIntel v1 (pixel differencing + spatial evidence + threshold sensitivity)"
TOOL_NAME_VQA = "RS-ChangeVQA v1 (RGB class-share delta + change evidence)"


def _engine(pixels_a, pixels_b, context: Optional[Dict[str, Any]]):
    ctx = context or {}
    base = int(ctx.get("base_threshold", config.CHANGE.base_threshold))
    thresholds = normalise_thresholds(base, ctx.get("thresholds"))
    grid_size = int(ctx.get("grid_size", config.CHANGE.grid_size))
    alignment = ctx.get("alignment") or estimate_alignment(to_gray(pixels_a), to_gray(pixels_b))
    result = analyze_change(pixels_a, pixels_b, alignment, thresholds, base, grid_size)
    return result, alignment


def run_describe(
    pixels_a: np.ndarray, pixels_b: np.ndarray, query: str, context: Optional[Dict[str, Any]] = None
) -> ToolResult:
    """Detect and describe changes between two co-gridded temporal images."""
    result, alignment = _engine(pixels_a, pixels_b, context)
    ev = result.evidence
    answer = change_headline(ev["measurements"], ev["concentration"])
    ev["alignment"] = alignment
    combined = next(layer for layer in result.layers if layer["id"] == "combined")
    return ToolResult(answer=answer, overlay_base64=combined["data"], evidence=ev, layers=result.layers)


def run_vqa(
    pixels_a: np.ndarray, pixels_b: np.ndarray, query: str, context: Optional[Dict[str, Any]] = None
) -> ToolResult:
    """Answer a comparative question (increased / decreased / unchanged)."""
    result, alignment = _engine(pixels_a, pixels_b, context)
    stats_a = compute_landcover_stats(pixels_a)
    stats_b = compute_landcover_stats(pixels_b)
    named = detect_keyword(query.lower())
    keyword = named or "built-up"
    margin = config.CHANGE.trend_margin_pp
    pa, pb = stats_a.percentages[keyword], stats_b.percentages[keyword]
    delta = pb - pa
    trend = "remained within the decision margin" if abs(delta) < margin else ("increased" if delta > 0 else "decreased")

    assumed = "" if named else "No target class was named, so built-up surfaces are assumed. "
    answer = (
        f"{assumed}By colour-threshold classification, {CLASS_LABELS[keyword]} coverage {trend} between the two dates "
        f"(from {pa:.1f}% to {pb:.1f}% of pixels, {'+' if delta >= 0 else ''}{delta:.1f} percentage points; "
        f"changes smaller than {margin:.0f} points are treated as unchanged)."
    )
    ev = result.evidence
    ev["alignment"] = alignment
    ev["class_delta"] = {
        "target_class": keyword, "target_named_in_query": bool(named),
        "share_a_pct": round(pa, 3), "share_b_pct": round(pb, 3), "delta_pp": round(delta, 3),
        "decision_margin_pp": margin, "decision": "UNCHANGED" if abs(delta) < margin else ("INCREASED" if delta > 0 else "DECREASED"),
        "method": "RGB colour-threshold classes on each image independently; not a trained classifier",
    }
    combined = next(layer for layer in result.layers if layer["id"] == "combined")
    return ToolResult(answer=answer, overlay_base64=combined["data"], evidence=ev, layers=result.layers)
