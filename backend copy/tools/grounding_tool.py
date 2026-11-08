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
    named = detect_keyword(query.lower())
    keyword = named or top_class(stats)
    pct = stats.percentages[keyword]

    overlay = build_highlight_overlay(pixels, stats.class_map, keyword, draw_bbox=True)
    overlay_b64 = numpy_to_base64(overlay)

    assumed = "" if named else (
        f"No target class was named in the query, so the dominant class ({CLASS_LABELS[keyword]}) is shown instead. "
    )
    if pct > 1:
        answer = (
            f'{assumed}Outlined the pixels matching "{CLASS_LABELS[keyword]}" by colour threshold '
            f"- {pct:.1f}% of the frame (bounding box spans all matching pixels, including isolated ones)."
        )
    else:
        answer = (
            f'{assumed}Only {pct:.1f}% of pixels match "{CLASS_LABELS[keyword]}" by colour threshold; '
            "no region is outlined."
        )

    return ToolResult(
        answer=answer,
        overlay_base64=overlay_b64,
        evidence={"class_percentages": stats.percentages, "target_class": keyword, "target_named_in_query": bool(named)},
    )
