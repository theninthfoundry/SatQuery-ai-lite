"""
backend/tools/caption_tool.py

Scene captioning via spectral composition analysis.
"""

from __future__ import annotations

import numpy as np

from backend.controller import ToolResult
from backend.image_utils import (
    CLASS_LABELS,
    compute_landcover_stats,
)


TOOL_NAME = "RS-Caption-Heuristic v0.1 (spectral composition classifier)"


def run(pixels: np.ndarray, query: str) -> ToolResult:
    """Generate a scene caption based on landcover composition."""
    stats = compute_landcover_stats(pixels)

    order = sorted(
        ["vegetation", "water", "built-up", "bare soil"],
        key=lambda c: stats.percentages[c],
        reverse=True,
    )
    dom = order[0]

    sentences = [
        f"The image is dominated by {CLASS_LABELS[dom]} "
        f"({stats.percentages[dom]:.1f}% of the analysed area)."
    ]

    rest = [c for c in order[1:] if stats.percentages[c] > 3]
    if rest:
        parts = [f"{CLASS_LABELS[c]} ({stats.percentages[c]:.1f}%)" for c in rest]
        sentences.append(f"Secondary cover includes {', '.join(parts)}.")

    confidence = min(0.93, 0.55 + stats.percentages[dom] / 180)

    return ToolResult(
        answer=" ".join(sentences),
        confidence=confidence,
        overlay_base64=None,
        evidence=stats.percentages,
    )
