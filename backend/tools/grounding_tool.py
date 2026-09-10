"""
backend/tools/grounding_tool.py

Text-guided region grounding via threshold segmentation + bounding box.
"""

from __future__ import annotations

import numpy as np

from backend.controller import ToolResult, detect_keyword
from backend.image_utils import (
    CLASS_LABELS,
    build_highlight_overlay,
    compute_landcover_stats,
    numpy_to_base64,
    top_class,
)


TOOL_NAME = "RS-Ground-Heuristic v0.1 (threshold segmentation)"


def run(pixels: np.ndarray, query: str) -> ToolResult:
    """Locate and highlight a queried landcover region."""
    stats = compute_landcover_stats(pixels)
    keyword = detect_keyword(query.lower()) or top_class(stats)
    pct = stats.percentages[keyword]

    overlay = build_highlight_overlay(pixels, stats.class_map, keyword, draw_bbox=True)
    overlay_b64 = numpy_to_base64(overlay)

    if pct > 1:
        answer = (
            f'Located and outlined the region(s) matching "{CLASS_LABELS[keyword]}" '
            f"— approximately {pct:.1f}% of the frame, shown in the overlay."
        )
        confidence = min(0.9, 0.55 + pct / 120)
    else:
        answer = (
            f'No region matching "{CLASS_LABELS[keyword]}" could be '
            f"localized with sufficient confidence."
        )
        confidence = 0.4

    return ToolResult(
        answer=answer,
        confidence=confidence,
        overlay_base64=overlay_b64,
        evidence=stats.percentages,
    )
