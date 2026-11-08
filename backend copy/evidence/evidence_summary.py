"""
backend/evidence/evidence_summary.py

Natural-language statements generated FROM the measurements. Every
statement carries a `basis` object naming the numbers it was built from, so
the interface (and a reviewer) can trace wording back to data. Nothing here
is a canned finding: with different measurements the sentences, and the
decision of which sentences to emit, change.
"""

from __future__ import annotations

from typing import Any, Dict, List

NO_CHANGE_PCT = 1.0  # below this share of pixels, headline reports no substantial pixel change


def _pct(x: float) -> str:
    return f"{x:.1f}%"


def change_statements(
    measurements: Dict[str, Any],
    concentration: Dict[str, Any],
    sensitivity: Dict[str, Any],
    ambiguity: Dict[str, Any],
    alignment: Dict[str, Any],
    noise_floor: Dict[str, Any],
    radiometric: Dict[str, Any],
    alignment_impact: Dict[str, Any],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    t = sensitivity["base_threshold"]

    out.append({
        "id": "magnitude",
        "text": (
            f"{_pct(measurements['changed_pct'])} of pixels ({measurements['changed_pixels']:,} of "
            f"{measurements['analysed_pixels']:,}) differ by more than {t} gray levels "
            f"(mean absolute difference over the frame: {measurements['mean_abs_difference']:.1f})."
        ),
        "basis": {
            "changed_pct": measurements["changed_pct"], "changed_pixels": measurements["changed_pixels"],
            "analysed_pixels": measurements["analysed_pixels"], "base_threshold": t,
            "mean_abs_difference": measurements["mean_abs_difference"],
        },
    })

    pattern = concentration["pattern"]
    if pattern == "CONCENTRATED":
        out.append({
            "id": "distribution",
            "text": (
                f"The change is concentrated in {concentration['dominant_cell_phrase']}: grid cell "
                f"{concentration['dominant_cell']} holds {_pct(concentration['top_cell_share'] * 100)} of all changed pixels."
            ),
            "basis": {k: concentration[k] for k in ("pattern", "dominant_cell", "top_cell_share")},
        })
    elif pattern == "CLUSTERED":
        cells = ", ".join(concentration["top_cells"])
        out.append({
            "id": "distribution",
            "text": (
                f"The change forms several clusters, led by {cells} "
                f"({_pct(concentration['top3_share'] * 100)} of changed pixels combined); no single region dominates."
            ),
            "basis": {k: concentration[k] for k in ("pattern", "top_cells", "top3_share")},
        })
    elif pattern == "DISTRIBUTED":
        out.append({
            "id": "distribution",
            "text": (
                "The change is spread across the frame with no dominant region "
                f"(the largest cell holds {_pct(concentration['top_cell_share'] * 100)} of changed pixels)."
            ),
            "basis": {k: concentration[k] for k in ("pattern", "top_cell_share", "normalised_entropy")},
        })
    if pattern != "NONE" and concentration.get("scattered"):
        out.append({
            "id": "scatter",
            "text": (
                f"Changed pixels are scattered rather than forming compact regions (cohesion {concentration['spatial_cohesion']:.2f}), "
                "which is typical of noise or fine texture differences rather than discrete surface changes."
            ),
            "basis": {"spatial_cohesion": concentration["spatial_cohesion"]},
        })

    if measurements["changed_pixels"] > 0 or sensitivity["sensitivity_level"] != "NO_CHANGE":
        ths = ", ".join(str(r["threshold"]) for r in sensitivity["thresholds"])
        ret = ambiguity["retention_of_base"]
        out.append({
            "id": "sensitivity",
            "text": (
                (
                    f"Across thresholds {ths}, the changed area ranges from {_pct(sensitivity['changed_pct_min'])} to "
                    f"{_pct(sensitivity['changed_pct_max'])}"
                    if sensitivity["changed_pct_range_pp"] >= 0.05
                    else f"Across thresholds {ths}, the changed area stays at about {_pct(sensitivity['changed_pct_max'])}"
                )
                + f" ({sensitivity['sensitivity_level'].lower().replace('_', ' ')} threshold sensitivity)."
                + (
                    f" {ret * 100:.0f}% of the pixels changed at threshold {t} remain changed at the strictest threshold; "
                    f"{_pct(ambiguity['ambiguous_pct'])} of the frame is threshold-sensitive and should be interpreted cautiously."
                    if ret is not None else ""
                )
            ),
            "basis": {
                "changed_pct_min": sensitivity["changed_pct_min"], "changed_pct_max": sensitivity["changed_pct_max"],
                "sensitivity_level": sensitivity["sensitivity_level"], "retention_of_base": ret,
                "ambiguous_pct": ambiguity["ambiguous_pct"],
            },
        })

    st = alignment["state"]
    if alignment.get("dx_px") is None:
        out.append({"id": "alignment", "text": "Alignment could not be estimated: " + " ".join(alignment["reasons"]),
                    "basis": {"state": st}})
    else:
        out.append({
            "id": "alignment",
            "text": (
                f"Alignment diagnostic: {st} (estimated translation dx {alignment['dx_px']:+.1f}px, dy {alignment['dy_px']:+.1f}px; "
                f"correlation peak {alignment['peak_correlation']:.2f}; valid overlap {alignment['valid_overlap_fraction'] * 100:.0f}%). "
                "This is a visual translation estimate, not geospatial registration."
            ),
            "basis": {k: alignment[k] for k in ("state", "dx_px", "dy_px", "peak_correlation", "valid_overlap_fraction")},
        })
    if alignment_impact.get("shift_compensated_changed_pct") is not None:
        out.append({
            "id": "alignment_impact",
            "text": (
                f"If the estimated shift were compensated, the changed area would be "
                f"{_pct(alignment_impact['shift_compensated_changed_pct'])} instead of {_pct(measurements['changed_pct'])} "
                "(what-if diagnostic; the reported measurement is not shift-corrected)."
            ),
            "basis": alignment_impact,
        })

    if radiometric["global_offset_flag"]:
        out.append({
            "id": "radiometric_offset",
            "text": (
                f"Image B is on average {abs(radiometric['median_signed_difference']):.1f} gray levels "
                f"{'brighter' if radiometric['median_signed_difference'] > 0 else 'darker'} than image A across the frame. "
                "A global brightness offset can come from exposure, season or atmosphere rather than surface change; "
                f"after removing it, the changed area would be {_pct(radiometric['offset_corrected_changed_pct'])}."
            ),
            "basis": radiometric,
        })
    if noise_floor["threshold_within_noise"]:
        out.append({
            "id": "noise_floor",
            "text": (
                f"The base threshold ({t}) is only {noise_floor['threshold_over_sigma']:.1f}x the estimated noise of the "
                f"difference image (sigma {noise_floor['sigma_difference']:.1f}); at least "
                f"{noise_floor['required_multiple']:.0f}x is recommended, so some detected change may be noise."
            ),
            "basis": noise_floor,
        })
    return out


def change_headline(measurements: Dict[str, Any], concentration: Dict[str, Any]) -> str:
    """The measured headline only. The pipeline appends the reliability caveat."""
    pct = measurements["changed_pct"]
    t = measurements["base_threshold"]
    if pct < NO_CHANGE_PCT:
        return f"No substantial pixel-level change was detected: {_pct(pct)} of pixels differ by more than {t} gray levels."
    where = (
        concentration["dominant_cell_phrase"] if concentration["pattern"] == "CONCENTRATED" else concentration["centroid_phrase"]
    )
    return (
        f"Pixel-level change was detected across {_pct(pct)} of the frame, "
        f"{'concentrated in' if concentration['pattern'] == 'CONCENTRATED' else 'centred on'} {where}."
    )


def reliability_suffix(level: str, concerns: List[str]) -> str:
    if level == "UNSUPPORTED":
        return " This result should NOT be relied on: " + (concerns[0].rstrip(".") if concerns else "diagnostics failed") + "."
    if level == "SUPPORTED":
        return ""
    return f" Evidence level {level}: " + "; ".join(c.rstrip(".") for c in concerns[:2]) + "."
