"""
backend/diagnostics.py

Independent diagnostics that run BEFORE any analysis tool and decide
whether the imagery can support the question being asked.

  1. assess_image_quality  - brightness, contrast, clipping, sharpness, noise,
                             edge density, resolution -> flags + implications
  2. estimate_alignment    - translation-only phase correlation between two
                             images -> GOOD / WARNING / UNRELIABLE
  3. assess_suitability    - PROCEED / PROCEED_WITH_WARNINGS /
                             REQUEST_BETTER_IMAGERY / REJECT

Honesty rules baked into this module
------------------------------------
* Quality metrics are *diagnostic indicators*, not a universal satellite
  image quality score. They are computed on the working-resolution image
  and depend on scene content (a calm water scene legitimately has low
  Laplacian variance).
* The alignment estimate is a visual TRANSLATION estimate. It does not
  perform orthorectification, terrain correction, rotation/scale estimation
  or geospatial registration, and says so in every result.
* Every threshold comes from backend.config and is echoed in the response.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend import config
from backend.imgops import (
    laplacian_variance,
    noise_sigma,
    sobel_magnitude,
    to_gray,
)

SEV_INFO = "INFO"
SEV_WARNING = "WARNING"
SEV_SEVERE = "SEVERE"

ALIGNMENT_LIMITATION = (
    "This diagnostic estimates visual translation only. It does not perform "
    "geospatial orthorectification, terrain correction, rotation/scale "
    "estimation or full image registration."
)


def _flag(code: str, severity: str, message: str, implication: Optional[str] = None) -> Dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "implication": implication}


# ── 1. Image quality ─────────────────────────────────────────────────────

def assess_image_quality(
    pixels: np.ndarray,
    original_size: Tuple[int, int],
    qt: config.QualityThresholds = config.QUALITY,
) -> Dict[str, Any]:
    """Quality indicators for one (H, W, 3) uint8 working-resolution image.

    `original_size` is (width, height) of the uploaded file, used for the
    resolution check; every other metric is measured on `pixels`."""
    gray = to_gray(pixels)
    h, w = gray.shape
    mean = float(gray.mean())
    std = float(gray.std())
    p2, p98 = (float(v) for v in np.percentile(gray, [2, 98]))
    clip_low = float((gray <= qt.clip_low_level).mean() * 100.0)
    clip_high = float((gray >= qt.clip_high_level).mean() * 100.0)
    clip_total = clip_low + clip_high
    lapvar = laplacian_variance(gray)
    edge_density = float((sobel_magnitude(gray) > qt.edge_magnitude).mean() * 100.0)
    sigma = noise_sigma(gray)

    px = pixels.astype(np.int16)
    channel_spread = px.max(axis=2) - px.min(axis=2)
    is_grayscale = bool((channel_spread <= 2).mean() >= 0.99)
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(px.max(axis=2) > 0, channel_spread / np.maximum(px.max(axis=2), 1), 0.0)
    mean_saturation = float(sat.mean())

    orig_w, orig_h = original_size
    min_dim = int(min(orig_w, orig_h))

    # ── levels ──
    brightness_level = "DARK" if mean < qt.dark_mean else "BRIGHT" if mean > qt.bright_mean else "NORMAL"
    contrast_level = (
        "LOW" if std < qt.low_contrast_std else "MODERATE" if std < qt.moderate_contrast_std else "GOOD"
    )
    clipping_level = (
        "SEVERE" if clip_total >= qt.clip_severe_pct
        else "HIGH" if clip_total >= qt.clip_high_pct
        else "ELEVATED" if clip_total >= qt.clip_elevated_pct
        else "LOW"
    )
    sharpness_level = (
        "VERY_LOW" if lapvar < qt.sharp_very_low_lapvar
        else "LOW" if lapvar < qt.sharp_low_lapvar
        else "NORMAL"
    )
    noise_level = (
        "HIGH" if sigma >= qt.noise_high_sigma else "ELEVATED" if sigma >= qt.noise_warn_sigma else "LOW"
    )
    resolution_level = (
        "VERY_LOW" if min_dim < qt.very_low_res_min_dim
        else "LOW" if min_dim < qt.low_res_min_dim
        else "ADEQUATE"
    )
    unusable = std < qt.unusable_std

    # ── flags ──
    flags: List[Dict[str, Any]] = []
    if unusable:
        flags.append(_flag(
            "UNUSABLE_FLAT_IMAGE", SEV_SEVERE,
            f"Image is essentially uniform (std {std:.2f} gray levels).",
            "There is no surface structure to analyse.",
        ))
    if brightness_level != "NORMAL":
        flags.append(_flag(
            f"BRIGHTNESS_{brightness_level}", SEV_WARNING,
            f"Image is unusually {brightness_level.lower()} (mean gray {mean:.0f}/255).",
            "Colour-threshold classes and intensity differences are less reliable at this exposure.",
        ))
    if contrast_level == "LOW" and not unusable:
        flags.append(_flag(
            "CONTRAST_LOW", SEV_WARNING,
            f"Low contrast (std {std:.1f} gray levels).",
            "Surface features may be hard to distinguish from one another.",
        ))
    if clipping_level in ("ELEVATED",):
        flags.append(_flag(
            "CLIPPING_ELEVATED", SEV_INFO,
            f"{clip_total:.1f}% of pixels are at the dark/bright extremes "
            f"({clip_low:.1f}% dark, {clip_high:.1f}% bright).",
            "Detail in saturated areas cannot be assessed. Dark values may include no-data fill.",
        ))
    elif clipping_level in ("HIGH", "SEVERE"):
        flags.append(_flag(
            f"CLIPPING_{clipping_level}",
            SEV_SEVERE if clipping_level == "SEVERE" else SEV_WARNING,
            f"{clip_total:.1f}% of pixels are at the dark/bright extremes "
            f"({clip_low:.1f}% dark, {clip_high:.1f}% bright).",
            "Detail in saturated areas is lost; changes there cannot be assessed. "
            "Dark values may include no-data fill.",
        ))
    if sharpness_level != "NORMAL" and not unusable:
        flags.append(_flag(
            f"SHARPNESS_{sharpness_level}", SEV_WARNING,
            f"Low sharpness (Laplacian variance {lapvar:.1f}).",
            "Small surface changes may be difficult to distinguish. "
            "Smooth scenes such as calm water also read as low-sharpness.",
        ))
    if noise_level != "LOW":
        flags.append(_flag(
            f"NOISE_{noise_level}", SEV_WARNING,
            f"Estimated noise sigma {sigma:.1f} gray levels.",
            "Pixel differences smaller than a few sigma may be noise rather than surface change.",
        ))
    if resolution_level == "VERY_LOW":
        flags.append(_flag(
            "RESOLUTION_VERY_LOW", SEV_SEVERE,
            f"Very low resolution (shorter side {min_dim}px).",
            "Too few pixels to resolve surface features.",
        ))
    elif resolution_level == "LOW":
        flags.append(_flag(
            "RESOLUTION_LOW", SEV_WARNING,
            f"Low resolution (shorter side {min_dim}px).",
            "Small features will not be resolved.",
        ))

    severities = {f["severity"] for f in flags}
    overall = (
        "UNUSABLE" if unusable
        else "POOR" if SEV_SEVERE in severities
        else "ACCEPTABLE" if SEV_WARNING in severities
        else "GOOD"
    )

    return {
        "overall": overall,
        "metrics": {
            "brightness_mean": round(mean, 2),
            "contrast_std": round(std, 2),
            "dynamic_range_p2_p98": round(p98 - p2, 2),
            "clipped_dark_pct": round(clip_low, 3),
            "clipped_bright_pct": round(clip_high, 3),
            "clipped_total_pct": round(clip_total, 3),
            "sharpness_laplacian_variance": round(lapvar, 2),
            "edge_density_pct": round(edge_density, 3),
            "noise_sigma_estimate": round(sigma, 2),
            "original_width": int(orig_w),
            "original_height": int(orig_h),
            "original_megapixels": round(orig_w * orig_h / 1e6, 3),
            "working_width": int(w),
            "working_height": int(h),
            "is_grayscale": is_grayscale,
            "mean_saturation": round(mean_saturation, 4),
        },
        "levels": {
            "brightness": brightness_level,
            "contrast": contrast_level,
            "clipping": clipping_level,
            "sharpness": sharpness_level,
            "noise": noise_level,
            "resolution": resolution_level,
        },
        "flags": flags,
        "analysis_implications": [f["implication"] for f in flags if f["implication"]],
        "interpretation_note": (
            "Diagnostic indicators measured on the working-resolution image. "
            "They are not a universal satellite-image quality score and depend on scene content."
        ),
    }


# ── 2. Temporal alignment ────────────────────────────────────────────────

def aspect_check(
    size_a: Tuple[int, int],
    size_b: Tuple[int, int],
    at: config.AlignmentThresholds = config.ALIGNMENT,
) -> Dict[str, Any]:
    """Compare the ORIGINAL (width, height) of two uploads."""
    ar_a = size_a[0] / size_a[1]
    ar_b = size_b[0] / size_b[1]
    mismatch = abs(ar_a / ar_b - 1.0)
    state = "UNRELIABLE" if mismatch >= at.aspect_unreliable else "WARNING" if mismatch >= at.aspect_warn else "GOOD"
    return {
        "size_a": list(size_a),
        "size_b": list(size_b),
        "same_dimensions": tuple(size_a) == tuple(size_b),
        "aspect_ratio_a": round(ar_a, 4),
        "aspect_ratio_b": round(ar_b, 4),
        "aspect_mismatch": round(mismatch, 4),
        "state": state,
    }


def _overlap_crops(a: np.ndarray, b: np.ndarray, dy: int, dx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Crops of a and b that depict the same content when b[y+dy, x+dx] == a[y, x]."""
    h, w = a.shape
    ay0, ay1 = max(0, -dy), min(h, h - dy)
    ax0, ax1 = max(0, -dx), min(w, w - dx)
    by0, by1 = ay0 + dy, ay1 + dy
    bx0, bx1 = ax0 + dx, ax1 + dx
    if ay1 <= ay0 or ax1 <= ax0:
        return a[:0, :0], b[:0, :0]
    return a[ay0:ay1, ax0:ax1], b[by0:by1, bx0:bx1]


def _ncc(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 256:
        return None
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = math.sqrt(float((a * a).sum() * (b * b).sum()))
    if denom < 1e-9:
        return None
    return float((a * b).sum() / denom)


def _subpixel(c_minus: float, c0: float, c_plus: float) -> float:
    denom = c_minus - 2.0 * c0 + c_plus
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (c_minus - c_plus) / denom, -0.5, 0.5))


def estimate_alignment(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    at: config.AlignmentThresholds = config.ALIGNMENT,
    aspect: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Translation-only alignment diagnostic between two same-sized gray images.

    Convention: (dx, dy) is how far image CONTENT moved from A to B, so
    ``b[y + dy, x + dx] ~= a[y, x]``; +dx is right, +dy is down.
    """
    if gray_a.shape != gray_b.shape:
        raise ValueError("estimate_alignment needs images on the same working grid")
    h, w = gray_a.shape
    reasons: List[str] = []
    base: Dict[str, Any] = {
        "method": "phase_correlation_translation",
        "working_grid": [w, h],
        "limitation": ALIGNMENT_LIMITATION,
    }

    std_a, std_b = float(gray_a.std()), float(gray_b.std())
    if min(std_a, std_b) < at.min_texture_std:
        reasons.append(
            "Insufficient texture in at least one image to estimate alignment "
            f"(std {min(std_a, std_b):.1f} < {at.min_texture_std})."
        )
        return {
            **base, "state": "UNRELIABLE", "dx_px": None, "dy_px": None, "magnitude_px": None,
            "peak_correlation": None, "ncc_unshifted": None, "ncc_aligned": None,
            "valid_overlap_fraction": None, "reasons": reasons, "aspect": aspect,
        }

    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float64)
    a = (gray_a.astype(np.float64) - gray_a.mean()) * win
    b = (gray_b.astype(np.float64) - gray_b.mean()) * win
    cross = np.fft.rfft2(b) * np.conj(np.fft.rfft2(a))
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.irfft2(cross, s=(h, w))

    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    peak = float(corr[iy, ix])
    sub_y = _subpixel(corr[(iy - 1) % h, ix], corr[iy, ix], corr[(iy + 1) % h, ix])
    sub_x = _subpixel(corr[iy, (ix - 1) % w], corr[iy, ix], corr[iy, (ix + 1) % w])
    dy = (iy if iy <= h // 2 else iy - h) + sub_y
    dx = (ix if ix <= w // 2 else ix - w) + sub_x
    magnitude = float(math.hypot(dx, dy))

    idy, idx = int(round(dy)), int(round(dx))
    overlap = max(0, h - abs(idy)) * max(0, w - abs(idx)) / float(h * w)
    ca, cb = _overlap_crops(gray_a, gray_b, idy, idx)
    ncc_aligned = _ncc(ca, cb)
    ncc_unshifted = _ncc(gray_a, gray_b)

    state = "GOOD"

    def worsen(new_state: str) -> None:
        nonlocal state
        order = {"GOOD": 0, "WARNING": 1, "UNRELIABLE": 2}
        if order[new_state] > order[state]:
            state = new_state

    if magnitude >= at.unreliable_shift_px:
        worsen("UNRELIABLE")
        reasons.append(
            f"Estimated displacement {magnitude:.1f}px (dx {dx:+.1f}, dy {dy:+.1f}) is large; "
            "pixel-wise differences would mostly reflect the shift."
        )
    elif magnitude >= at.warn_shift_px:
        worsen("WARNING")
        reasons.append(f"Estimated displacement {magnitude:.1f}px (dx {dx:+.1f}, dy {dy:+.1f}); interpret changes near edges with caution.")
    if overlap < at.overlap_unreliable:
        worsen("UNRELIABLE")
        reasons.append(f"Valid overlap after shift is only {overlap * 100:.0f}% of the frame.")
    elif overlap < at.overlap_warn:
        worsen("WARNING")
        reasons.append(f"Valid overlap after shift is {overlap * 100:.0f}% of the frame.")
    if peak < at.peak_unreliable:
        worsen("UNRELIABLE")
        reasons.append(
            f"No coherent correlation peak ({peak:.3f}); alignment cannot be confirmed. "
            "The images may show different areas or differ too strongly."
        )
    elif peak < at.peak_warn:
        worsen("WARNING")
        reasons.append(
            f"Weak correlation peak ({peak:.3f}); alignment cannot be confirmed. "
            "This can also result from extensive genuine change, rotation or scale differences."
        )
    if ncc_aligned is not None and ncc_aligned < at.ncc_warn:
        worsen("WARNING")
        reasons.append(f"Low similarity after alignment (NCC {ncc_aligned:.2f}).")
    if aspect is not None and aspect.get("state") in ("WARNING", "UNRELIABLE"):
        worsen(aspect["state"])
        reasons.append(
            f"Original aspect ratios differ by {aspect['aspect_mismatch'] * 100:.1f}%; "
            "the second image was resampled onto the first image's grid, so spatial correspondence is not guaranteed."
        )
    if state == "GOOD":
        reasons.append("Estimated translation is below the warning threshold and correlation is coherent.")

    return {
        **base,
        "state": state,
        "dx_px": round(float(dx), 3) + 0.0,
        "dy_px": round(float(dy), 3) + 0.0,
        "magnitude_px": round(magnitude, 3),
        "dx_fraction_of_width": round(float(dx) / w, 5),
        "dy_fraction_of_height": round(float(dy) / h, 5),
        "peak_correlation": round(peak, 4),
        "ncc_unshifted": None if ncc_unshifted is None else round(ncc_unshifted, 4),
        "ncc_aligned": None if ncc_aligned is None else round(ncc_aligned, 4),
        "valid_overlap_fraction": round(overlap, 4),
        "reasons": reasons,
        "aspect": aspect,
    }


# ── 3. Suitability ───────────────────────────────────────────────────────

_DECISION_ORDER = ["PROCEED", "PROCEED_WITH_WARNINGS", "REQUEST_BETTER_IMAGERY", "REJECT"]


def _worse(a: str, b: str) -> str:
    return a if _DECISION_ORDER.index(a) >= _DECISION_ORDER.index(b) else b


def assess_suitability(
    qualities: List[Dict[str, Any]],
    alignment: Optional[Dict[str, Any]] = None,
    requires_alignment: bool = False,
) -> Dict[str, Any]:
    """Combine per-image quality (and, for temporal comparison, alignment)
    into one explicit decision with the reasons that produced it."""
    decision = "PROCEED"
    reasons: List[str] = []
    for i, q in enumerate(qualities):
        name = f"image {i + 1}" if len(qualities) > 1 else "image"
        if q["overall"] == "UNUSABLE":
            decision = _worse(decision, "REJECT")
            reasons.append(f"{name}: unusable ({q['flags'][0]['message']})")
            continue
        sev = [f for f in q["flags"] if f["severity"] == SEV_SEVERE]
        warn = [f for f in q["flags"] if f["severity"] == SEV_WARNING]
        if sev:
            decision = _worse(decision, "REQUEST_BETTER_IMAGERY")
            reasons.extend(f"{name}: {f['message']}" for f in sev)
        if warn:
            decision = _worse(decision, "PROCEED_WITH_WARNINGS")
            reasons.extend(f"{name}: {f['message']}" for f in warn)
    if alignment is not None and requires_alignment:
        if alignment["state"] == "UNRELIABLE":
            decision = _worse(decision, "REQUEST_BETTER_IMAGERY")
            reasons.append("alignment: unreliable - " + "; ".join(alignment["reasons"][:2]))
        elif alignment["state"] == "WARNING":
            decision = _worse(decision, "PROCEED_WITH_WARNINGS")
            reasons.append("alignment: warning - " + "; ".join(alignment["reasons"][:2]))
    return {
        "decision": decision,
        "reasons": reasons,
        "rule": (
            "REJECT if an input is unusable; REQUEST_BETTER_IMAGERY if any SEVERE flag or UNRELIABLE alignment "
            "(for temporal comparison); PROCEED_WITH_WARNINGS if any WARNING flag or WARNING alignment; else PROCEED."
        ),
    }
