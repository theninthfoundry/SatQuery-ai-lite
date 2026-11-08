"""
backend/reliability.py

Replaces the single "confidence" number of v0.

A heuristic image-processing pipeline has no calibrated probability of being
right, so none is reported. Instead the response lists the concrete factors
that bear on trust, and a coarse evidence level derived by a stated rule:

    SUPPORTED   no diagnostic raised a concern
    CAUTION     exactly one concern
    WEAK        two or more concerns
    UNSUPPORTED at least one severe concern (the measurement should not be relied on)

SUPPORTED does NOT mean "correct". It means none of the implemented
diagnostics found a reason for doubt.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

RULE = (
    "SUPPORTED: no concerns; CAUTION: 1 concern; WEAK: 2 or more concerns; "
    "UNSUPPORTED: any severe concern. SUPPORTED means no implemented diagnostic found a reason for doubt - not that the result is correct."
)


def _factor(name: str, status: str, detail: str, value: Any = None) -> Dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, "value": value}


def assess_reliability(
    qualities: List[Dict[str, Any]],
    suitability: Dict[str, Any],
    alignment: Optional[Dict[str, Any]] = None,
    change: Optional[Dict[str, Any]] = None,
    method_concern: Optional[str] = None,
    extra_concerns: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    factors: List[Dict[str, Any]] = []

    for i, q in enumerate(qualities):
        who = f"image {i + 1}" if len(qualities) > 1 else "image"
        sev = [f for f in q["flags"] if f["severity"] == "SEVERE"]
        warn = [f for f in q["flags"] if f["severity"] == "WARNING"]
        if sev:
            factors.append(_factor(f"quality:{who}", "SEVERE", "; ".join(f["message"] for f in sev), q["overall"]))
        elif warn:
            factors.append(_factor(f"quality:{who}", "CONCERN", "; ".join(f["message"] for f in warn), q["overall"]))
        else:
            factors.append(_factor(f"quality:{who}", "OK", "No quality concerns raised.", q["overall"]))

    if alignment is not None:
        st = alignment["state"]
        factors.append(_factor(
            "alignment",
            {"GOOD": "OK", "WARNING": "CONCERN", "UNRELIABLE": "SEVERE"}[st],
            " ".join(alignment["reasons"][:2]),
            st,
        ))

    if change is not None:
        sens = change["sensitivity"]["sensitivity_level"]
        factors.append(_factor(
            "threshold_sensitivity",
            "CONCERN" if sens == "HIGH" else "OK",
            f"Changed area varies {change['sensitivity']['changed_pct_min']:.1f}%-{change['sensitivity']['changed_pct_max']:.1f}% "
            f"across thresholds ({sens.lower().replace('_', ' ')} sensitivity).",
            sens,
        ))
        nf = change["noise_floor"]
        factors.append(_factor(
            "noise_floor",
            "CONCERN" if nf["threshold_within_noise"] else "OK",
            "Base threshold is within the estimated noise of the difference image."
            if nf["threshold_within_noise"] else "Base threshold is above the estimated noise floor.",
            nf["threshold_over_sigma"],
        ))
        rad = change["radiometric"]
        factors.append(_factor(
            "radiometric_offset",
            "CONCERN" if rad["global_offset_flag"] else "OK",
            f"Median signed difference {rad['median_signed_difference']:+.1f} gray levels.",
            rad["median_signed_difference"],
        ))
        conc = change["concentration"]
        if conc["pattern"] != "NONE":
            factors.append(_factor(
                "spatial_coherence",
                "CONCERN" if conc["scattered"] else "OK",
                f"Spatial cohesion of changed pixels {conc['spatial_cohesion']:.2f}.",
                conc["spatial_cohesion"],
            ))

    if method_concern:
        factors.append(_factor("method", "CONCERN", method_concern))
    for extra in extra_concerns or []:
        factors.append(extra)

    severe = [f for f in factors if f["status"] == "SEVERE"]
    concerns = [f for f in factors if f["status"] == "CONCERN"]
    if suitability["decision"] == "REQUEST_BETTER_IMAGERY" and not severe:
        severe = [_factor("suitability", "SEVERE", "; ".join(suitability["reasons"][:2]))]
        factors.append(severe[0])
    level = "UNSUPPORTED" if severe else "WEAK" if len(concerns) >= 2 else "CAUTION" if len(concerns) == 1 else "SUPPORTED"

    summary = [f["detail"] for f in severe + concerns]
    return {
        "level": level,
        "factors": factors,
        "concerns": summary,
        "rule": RULE,
        "confidence": None,
        "confidence_note": (
            "No calibrated probability is reported: the analysis is heuristic image processing, "
            "not a trained model with measured accuracy."
        ),
    }
