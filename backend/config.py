"""
backend/config.py

Every number that decides a diagnostic outcome lives here, in one place, so
that thresholds are explicit, documented and testable rather than buried in
the code that applies them.

None of these are universal truths about satellite imagery. They are
operating points chosen for the 8-bit PNG/JPEG/TIFF imagery the Lite
deployment accepts, and every one of them is echoed back to the caller in
the response's `provenance.parameters` / `methodology` blocks.

Deployment limits can be overridden with environment variables
(SATQUERY_MAX_IMAGE_BYTES, ...). Diagnostic thresholds are code-level
constants on purpose: changing one changes scientific behaviour and should
go through review and a version bump.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

ENGINE_VERSION = "1.0.0"
SCHEMA_VERSION = "2.0"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class Limits:
    """Input/transport limits. Defaults are derived from Vercel's 4.5 MB
    serverless request-body limit: two base64-encoded images (x1.37) must fit."""

    max_image_bytes: int = _env_int("SATQUERY_MAX_IMAGE_BYTES", 1_600_000)
    max_total_image_bytes: int = _env_int("SATQUERY_MAX_TOTAL_IMAGE_BYTES", 3_200_000)
    max_request_bytes: int = _env_int("SATQUERY_MAX_REQUEST_BYTES", 4_400_000)
    max_pixels: int = _env_int("SATQUERY_MAX_PIXELS", 25_000_000)
    min_dimension: int = 32
    max_query_chars: int = 500
    max_images: int = 2
    min_grid_size: int = 2
    max_grid_size: int = 12
    min_thresholds: int = 2
    max_thresholds: int = 7
    max_stac_bytes: int = 1_000_000
    max_stac_assets: int = 500


@dataclass(frozen=True)
class Processing:
    """Working-resolution policy."""

    single_max_side: int = 380      # single-image tasks (matches v0 behaviour)
    pair_max_side: int = 384        # pair tasks: common grid derived from image A
    gray_definition: str = "mean of R,G,B"


@dataclass(frozen=True)
class QualityThresholds:
    dark_mean: float = 40.0
    bright_mean: float = 215.0
    low_contrast_std: float = 25.0
    moderate_contrast_std: float = 50.0
    clip_low_level: int = 2           # gray <= this counts as clipped-dark
    clip_high_level: int = 253        # gray >= this counts as clipped-bright
    clip_elevated_pct: float = 5.0
    clip_high_pct: float = 20.0
    clip_severe_pct: float = 40.0
    sharp_low_lapvar: float = 30.0
    sharp_very_low_lapvar: float = 8.0
    noise_warn_sigma: float = 6.0
    noise_high_sigma: float = 12.0
    low_res_min_dim: int = 256
    very_low_res_min_dim: int = 128
    unusable_std: float = 1.5         # essentially a flat image
    edge_magnitude: float = 40.0      # Sobel magnitude counted as an edge


@dataclass(frozen=True)
class AlignmentThresholds:
    warn_shift_px: float = 0.75
    unreliable_shift_px: float = 4.0
    peak_warn: float = 0.10
    peak_unreliable: float = 0.03
    overlap_warn: float = 0.97
    overlap_unreliable: float = 0.85
    aspect_warn: float = 0.02
    aspect_unreliable: float = 0.10
    ncc_warn: float = 0.50
    min_texture_std: float = 3.0


@dataclass(frozen=True)
class ChangeParams:
    base_threshold: int = 28
    sensitivity_thresholds: Tuple[int, ...] = (20, 28, 40)
    grid_size: int = 4
    min_cell_change_pct: float = 2.0       # below this a cell shows "no change evidence"
    strong_cell_change_pct: float = 10.0
    strong_retention: float = 0.70
    moderate_retention: float = 0.40
    sensitivity_low: float = 0.35          # (max-min)/mean of changed-area across thresholds
    sensitivity_high: float = 0.80
    global_offset_warn: float = 8.0        # |median(B-A)| in gray levels
    noise_floor_multiplier: float = 3.0    # threshold should exceed k * sigma_diff
    cohesion_scattered: float = 0.35       # share of changed px with >=3 changed neighbours
    concentrated_top_cell_share: float = 0.40
    distributed_entropy: float = 0.90
    trend_margin_pp: float = 2.0           # change-VQA "unchanged" margin (percentage points)


LIMITS = Limits()
PROCESSING = Processing()
QUALITY = QualityThresholds()
ALIGNMENT = AlignmentThresholds()
CHANGE = ChangeParams()


def as_dict() -> Dict[str, Any]:
    """Everything above as plain data, for /api/capabilities and provenance."""
    return {
        "engine_version": ENGINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "limits": asdict(LIMITS),
        "processing": asdict(PROCESSING),
        "quality_thresholds": asdict(QUALITY),
        "alignment_thresholds": asdict(ALIGNMENT),
        "change_parameters": {
            **asdict(CHANGE),
            "sensitivity_thresholds": list(CHANGE.sensitivity_thresholds),
        },
    }
