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
    overlay = render_fusion_overlay(pixels_optical, fused.water_mask, fused.built_mask)
    overlay_b64 = numpy_to_base64(overlay)
    layers = [
        {"id": "optical", "label": "Optical", "description": "Optical input on the working grid.", "width": int(pixels_optical.shape[1]), "height": int(pixels_optical.shape[0]), "data": numpy_to_base64(pixels_optical)},
        {"id": "sar", "label": "SAR", "description": "SAR input on the optical working grid.", "width": int(pixels_sar.shape[1]), "height": int(pixels_sar.shape[0]), "data": numpy_to_base64(pixels_sar)},
        {"id": "fused", "label": "Fused water / built-up", "description": "Cyan: optical water AND SAR dark region. Amber: optical built-up/bare AND SAR edge.", "width": int(overlay.shape[1]), "height": int(overlay.shape[0]), "data": overlay_b64},
    ]

    return ToolResult(
        answer=answer,
        overlay_base64=overlay_b64,
        layers=layers,
        evidence={
            "waterPct": fused.water_pct,
            "builtPct": fused.built_pct,
            "agreementPct": fused.agreement_pct,
            "sar_edge_pct": sar_struct.edge_pct,
            "sar_dark_pct": sar_struct.dark_pct,
            "agreement_definition": "mean of (optical&SAR / optical|SAR) over the water and built-up proxies",
        },
    )
