"""
backend/tools/change_tool.py

Change detection (pixel-differencing) and change-VQA (class-delta comparison)
for bi-temporal image pairs.
"""

from __future__ import annotations

import numpy as np

from backend.controller import ToolResult, detect_keyword
from backend.image_utils import (
    CLASS_LABELS,
    compute_change_mask,
    compute_landcover_stats,
    describe_location,
    numpy_to_base64,
    render_change_overlay,
)


TOOL_NAME_DESCRIBE = "RS-ChangeNet-Heuristic v0.1 (pixel-differencing)"
TOOL_NAME_VQA = "RS-ChangeVQA-Heuristic v0.1 (class-delta comparison)"


def run_describe(pixels_a: np.ndarray, pixels_b: np.ndarray, query: str) -> ToolResult:
    """Detect and describe changes between two temporal images."""
    diff = compute_change_mask(pixels_a, pixels_b)
    location = describe_location(diff)

    if diff.changed_pct > 1:
        answer = (
            f"Change detected across approximately {diff.changed_pct:.1f}% of the "
            f"co-registered area, concentrated in the {location}."
        )
    else:
        answer = (
            f"No significant change was detected between the two acquisitions "
            f"({diff.changed_pct:.1f}% of pixels differ, within the no-change threshold)."
        )

    confidence = min(0.92, 0.5 + diff.changed_pct / 70)
    overlay = render_change_overlay(pixels_b, diff.mask)
    overlay_b64 = numpy_to_base64(overlay)

    return ToolResult(
        answer=answer,
        confidence=confidence,
        overlay_base64=overlay_b64,
        evidence={"changedPct": diff.changed_pct, "location": location},
    )


def run_vqa(pixels_a: np.ndarray, pixels_b: np.ndarray, query: str) -> ToolResult:
    """Answer a comparative question about change between two dates."""
    stats_a = compute_landcover_stats(pixels_a)
    stats_b = compute_landcover_stats(pixels_b)
    keyword = detect_keyword(query.lower()) or "built-up"

    pa = stats_a.percentages[keyword]
    pb = stats_b.percentages[keyword]
    delta = pb - pa

    if abs(delta) < 2:
        trend = "remained largely unchanged"
    elif delta > 0:
        trend = "increased"
    else:
        trend = "decreased"

    answer = (
        f"{CLASS_LABELS[keyword]} coverage has {trend} between the two dates "
        f"(from {pa:.1f}% to {pb:.1f}%, a "
        f"{'+' if delta >= 0 else ''}{delta:.1f} percentage-point shift)."
    )

    diff = compute_change_mask(pixels_a, pixels_b)
    confidence = min(0.9, 0.55 + abs(delta) / 25)
    overlay = render_change_overlay(pixels_b, diff.mask)
    overlay_b64 = numpy_to_base64(overlay)

    return ToolResult(
        answer=answer,
        confidence=confidence,
        overlay_base64=overlay_b64,
        evidence={"pa": pa, "pb": pb, "delta": delta},
    )
