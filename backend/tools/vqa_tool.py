"""
backend/tools/vqa_tool.py

Visual Question Answering via keyword-grounded stats lookup.
"""

from __future__ import annotations

import numpy as np

from backend.controller import ToolResult, detect_keyword
from backend.image_utils import (
    CLASS_LABELS,
    build_highlight_overlay,
    compute_landcover_stats,
    numpy_to_base64,
)


TOOL_NAME = "RS-VQA-Heuristic v0.1 (keyword-grounded stats lookup)"


def run(pixels: np.ndarray, query: str) -> ToolResult:
    """Answer a visual question about a single image."""
    stats = compute_landcover_stats(pixels)
    q = query.lower()
    keyword = detect_keyword(q)
    overlay_b64 = None

    if keyword and any(p in q for p in ("how much", "what percentage", "what proportion", "what fraction")):
        answer = (
            f"Approximately {stats.percentages[keyword]:.1f}% of the image "
            f"is classified as {CLASS_LABELS[keyword]}."
        )
        confidence = min(0.9, 0.6 + stats.percentages[keyword] / 150)
        overlay = build_highlight_overlay(pixels, stats.class_map, keyword, False)
        overlay_b64 = numpy_to_base64(overlay)

    elif keyword and any(p in q for p in ("is there", "are there", "does", "any ")):
        present = stats.percentages[keyword] > 2
        if present:
            answer = (
                f"Yes — {CLASS_LABELS[keyword]} is present, "
                f"covering about {stats.percentages[keyword]:.1f}% of the scene."
            )
            confidence = min(0.9, 0.6 + stats.percentages[keyword] / 150)
            overlay = build_highlight_overlay(pixels, stats.class_map, keyword, False)
            overlay_b64 = numpy_to_base64(overlay)
        else:
            answer = (
                f"No significant {CLASS_LABELS[keyword]} was detected "
                f"in this scene ({stats.percentages[keyword]:.1f}%)."
            )
            confidence = 0.58

    elif keyword:
        answer = (
            f"{CLASS_LABELS[keyword]} covers about "
            f"{stats.percentages[keyword]:.1f}% of the scene."
        )
        confidence = 0.65
        overlay = build_highlight_overlay(pixels, stats.class_map, keyword, False)
        overlay_b64 = numpy_to_base64(overlay)

    else:
        order = sorted(
            ["vegetation", "water", "built-up", "bare soil"],
            key=lambda c: stats.percentages[c],
            reverse=True,
        )
        answer = (
            f"Based on spectral composition, the scene is most consistent with "
            f"{CLASS_LABELS[order[0]]} ({stats.percentages[order[0]]:.1f}%), "
            f"followed by {CLASS_LABELS[order[1]]} ({stats.percentages[order[1]]:.1f}%)."
        )
        confidence = min(0.85, 0.55 + stats.percentages[order[0]] / 200)

    return ToolResult(
        answer=answer,
        confidence=confidence,
        overlay_base64=overlay_b64,
        evidence=stats.percentages,
    )
