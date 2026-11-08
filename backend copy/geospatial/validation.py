"""
backend/geospatial/validation.py

Compatibility checks between two inspected rasters. These are metadata
checks only; they say nothing about whether the pixels actually agree.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_ORDER = {"OK": 0, "INFO": 0, "WARNING": 1, "SEVERE": 2}


def _f(code: str, severity: str, message: str) -> Dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def _iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    w = max(0.0, min(a["east"], b["east"]) - max(a["west"], b["west"]))
    h = max(0.0, min(a["north"], b["north"]) - max(a["south"], b["south"]))
    inter = w * h
    area = lambda r: max(0.0, r["east"] - r["west"]) * max(0.0, r["north"] - r["south"])
    union = area(a) + area(b) - inter
    return 0.0 if union <= 0 else inter / union


def validate_geospatial_pair(meta_a: Optional[Dict[str, Any]], meta_b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    findings: List[Dict[str, str]] = []
    ga = bool(meta_a and meta_a.get("georeferenced"))
    gb = bool(meta_b and meta_b.get("georeferenced"))

    if not ga and not gb:
        findings.append(_f(
            "GEO_NO_GEOREFERENCING", "INFO",
            "Neither input carries georeferencing metadata. Spatial correspondence between the images is assumed, not verified, "
            "and no ground measurements (area, distance, coordinates) are possible.",
        ))
    elif ga != gb:
        findings.append(_f(
            "GEO_ASYMMETRIC", "WARNING",
            "Only one input is georeferenced, so the two images cannot be checked for spatial correspondence.",
        ))
    else:
        ca, cb = meta_a["crs"], meta_b["crs"]
        if ca["user_defined"] or cb["user_defined"] or ca["epsg"] is None or cb["epsg"] is None:
            findings.append(_f("GEO_CRS_UNKNOWN", "WARNING", "At least one CRS is user-defined or has no EPSG code; CRS equality cannot be confirmed."))
        elif ca["epsg"] != cb["epsg"]:
            findings.append(_f("GEO_CRS_MISMATCH", "SEVERE",
                               f"CRS differs (EPSG:{ca['epsg']} vs EPSG:{cb['epsg']}). Lite does not reproject, so pixel-wise comparison is invalid."))
        else:
            pa, pb = meta_a["pixel_size"], meta_b["pixel_size"]
            if pa and pb:
                rel = max(abs(pa["x"] - pb["x"]) / max(pa["x"], 1e-12), abs(pa["y"] - pb["y"]) / max(pa["y"], 1e-12))
                if rel > 0.01:
                    findings.append(_f("GEO_RESOLUTION_MISMATCH", "WARNING",
                                       f"Pixel sizes differ by {rel * 100:.1f}%; one image must be resampled onto the other's grid."))
            ba, bb = meta_a["bounds_native"], meta_b["bounds_native"]
            if ba and bb:
                iou = _iou(ba, bb)
                if iou < 0.5:
                    findings.append(_f("GEO_EXTENT_MISMATCH", "SEVERE", f"Footprints overlap poorly (IoU {iou:.2f})."))
                elif iou < 0.9:
                    findings.append(_f("GEO_EXTENT_PARTIAL", "WARNING", f"Footprints only partly overlap (IoU {iou:.2f})."))
        if meta_a.get("rotated_transform") or meta_b.get("rotated_transform"):
            findings.append(_f("GEO_ROTATED", "WARNING", "A rotated/sheared transform is present; north-up grid assumptions do not hold."))
        if (meta_a["width"], meta_a["height"]) != (meta_b["width"], meta_b["height"]):
            findings.append(_f("GEO_DIMENSION_MISMATCH", "WARNING", "Raster dimensions differ."))

    worst = max((_ORDER[f["severity"]] for f in findings), default=0)
    state = {0: "OK", 1: "WARNING", 2: "SEVERE"}[worst]
    return {"state": state, "findings": findings, "scope": "metadata checks only; pixel agreement is not evaluated"}
