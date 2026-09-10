"""
backend/tools/fusion_tool.py

Optical–SAR fusion analysis: combines optical spectral signatures with SAR
structural texture to identify water and built-up regions.
"""

from __future__ import annotations

import numpy as np

from backend.controller import ToolResult
from backend.image_utils import (
    compute_landcover_stats,
    compute_sar_structure,
    fuse_classes,
    numpy_to_base64,
    render_fusion_overlay,
)


TOOL_NAME = "RS-Fusion-Heuristic v0.1 (Sobel structure + spectral fusion)"


def run(pixels_optical: np.ndarray, pixels_sar: np.ndarray, query: str) -> ToolResult:
    """Fuse optical and SAR images for cross-modal classification."""
    optical_stats = compute_landcover_stats(pixels_optical)
    sar_struct = compute_sar_structure(pixels_sar)
    fused = fuse_classes(optical_stats, sar_struct)

    answer = (
        f"Combining optical spectral signatures with SAR structural texture: "
        f"water-covered area is estimated at {fused.water_pct:.1f}% and "
        f"built-up area at {fused.built_pct:.1f}% "
        f"(cross-modal agreement {fused.agreement_pct:.0f}%)."
    )

    confidence = min(0.92, 0.55 + fused.agreement_pct / 180)
    overlay = render_fusion_overlay(pixels_optical, fused.water_mask, fused.built_mask)
    overlay_b64 = numpy_to_base64(overlay)

    return ToolResult(
        answer=answer,
        confidence=confidence,
        overlay_base64=overlay_b64,
        evidence={
            "waterPct": fused.water_pct,
            "builtPct": fused.built_pct,
            "agreementPct": fused.agreement_pct,
        },
    )
