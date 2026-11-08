"""
backend/capabilities.py

The agent's knowledge of what it can and cannot do.

Before any tool runs, the query is checked against the capabilities the
deployment actually has. A request that depends on information the input
does not carry (calibrated SAR, spectral bands, georeferencing, acquisition
metadata) is answered with an explicit explanation, never with a
plausible-looking guess.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from backend import config

# ── Registry (served at /api/capabilities) ───────────────────────────────

TASKS = [
    {"id": "caption", "label": "Scene captioning", "modes": ["single"], "method": "RGB colour-threshold land-cover composition", "status": "HEURISTIC"},
    {"id": "vqa", "label": "Visual question answering", "modes": ["single"], "method": "keyword-grounded lookup over RGB colour-threshold statistics", "status": "HEURISTIC"},
    {"id": "grounding", "label": "Region grounding", "modes": ["single"], "method": "colour-threshold segmentation + bounding box", "status": "HEURISTIC"},
    {"id": "change-describe", "label": "Change detection", "modes": ["bitemporal"], "method": "pixel intensity differencing with alignment, spatial-evidence and threshold-sensitivity diagnostics", "status": "DIAGNOSTIC"},
    {"id": "change-vqa", "label": "Comparative change question", "modes": ["bitemporal"], "method": "RGB class-share delta + the change analysis above", "status": "HEURISTIC"},
    {"id": "fusion", "label": "Optical-SAR joint analysis", "modes": ["cross"], "method": "colour classes + Sobel/dark-region SAR proxy", "status": "HEURISTIC"},
]

NOT_SUPPORTED = [
    "Spectral indices (NDVI, NDWI, MNDWI, NDBI): need NIR/SWIR bands, not available in 8-bit RGB/grayscale input.",
    "Calibrated SAR analysis (sigma0, backscatter in dB, polarimetry, speckle filtering, coherence).",
    "Physical measurements (hectares, square kilometres, distances) and geographic coordinates of detected features.",
    "Geospatial registration, orthorectification, reprojection and terrain correction.",
    "Object detection/counting (buildings, vehicles, ships, trees).",
    "Acquisition date / sensor identification from PNG/JPEG input (no metadata to read).",
    "Trained vision-language models: there is none in this deployment; every tool is classical image processing.",
]

GEOSPATIAL_PHASES = [
    {"phase": "A", "name": "GeoTIFF metadata inspection", "status": "READY", "detail": "dimensions, bands, bit depth, CRS (EPSG), transform, pixel size, bounds, NoData, COG-like layout"},
    {"phase": "B", "name": "STAC item metadata inspection", "status": "READY", "detail": "item JSON supplied by the caller; nothing is fetched"},
    {"phase": "C", "name": "Native raster processing", "status": "NOT_IMPLEMENTED", "detail": "requires Rasterio/GDAL; runtime and size constraints on Vercel must be evaluated first"},
]


def registry() -> Dict[str, Any]:
    return {
        "engine_version": config.ENGINE_VERSION,
        "schema_version": config.SCHEMA_VERSION,
        "tasks": TASKS,
        "input_modes": {
            "single": {"images": 1}, "cross": {"images": 2, "roles": ["optical", "sar"]}, "bitemporal": {"images": 2, "roles": ["earlier", "later"]},
        },
        "supported_formats": ["PNG", "JPEG", "TIFF/GeoTIFF (8-bit grayscale or RGB)"],
        "diagnostics": [
            "image_quality", "temporal_alignment_translation", "image_suitability", "spatial_evidence_grid",
            "threshold_sensitivity", "stable_vs_ambiguous_regions", "noise_floor_check", "radiometric_offset_check",
        ],
        "geospatial": GEOSPATIAL_PHASES,
        "not_supported": NOT_SUPPORTED,
        "limits": config.as_dict()["limits"],
        "parameters": config.as_dict(),
        "confidence_policy": "No calibrated confidence is reported. See `reliability` in analysis responses.",
    }


# ── Query gate ───────────────────────────────────────────────────────────

_SPECTRAL = re.compile(r"\b(ndvi|ndwi|mndwi|ndbi|nbr|evi|savi|nir|near[- ]infrared|swir|red[- ]edge|spectral (index|indices|bands?)|multispectral bands?)\b", re.I)
_SAR_CAL = re.compile(r"\b(sigma[- ]?0|sigma[- ]nought|backscatter\w*|calibrat\w*|decibels?|db|polari[sz]\w*|vv|vh|hh|hv|speckle|coherence|interferom\w*)\b", re.I)
_AREA = re.compile(r"(\bhectares?\b|\bacres?\b|\bsq\.?\s?(km|m|mi|ft)\b|\bsquare\s+(kilo)?met(er|re)s?\b|\bkm2\b|km²|\bm2\b|m²)", re.I)
_COORDS = re.compile(r"\b(latitude|longitude|lat/?lon|coordinates?|geo-?coordinates?|gps|where on earth|which (city|country|region))\b", re.I)
_ACQ = re.compile(r"\b(acquisition (date|time)|capture date|which satellite|what satellite|which sensor|what sensor|what (year|date) (was|is))\b", re.I)
_COUNT = re.compile(r"\b(how many|count|number of)\b.{0,30}\b(buildings?|houses?|vehicles?|cars?|trucks?|ships?|boats?|aircraft|planes?|trees?|roads?|people)\b", re.I)
_CHANGE_INTENT = re.compile(r"\b(what changed|chang(e|ed|es|ing)|differ\w*|compar\w*|before and after|increase[d]?|decrease[d]?|grown|grew|shrunk|expan\w+)\b", re.I)
_FUSION_INTENT = re.compile(r"\b(optical and sar|sar and optical|optical\W+sar|both sensors|fuse|fusion|together with (the )?sar)\b", re.I)


def _unavail(code: str, requested: str, requires: str, available: str, alternatives: List[str]) -> Dict[str, Any]:
    return {
        "kind": "UNAVAILABLE_CAPABILITY",
        "code": code,
        "requested": requested,
        "requires": requires,
        "available": available,
        "explanation": f"{requested} requires {requires}. This input provides: {available}.",
        "alternatives": alternatives,
    }


def check_query(query: str, mode: str, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the list of unmet requirements for this query on these inputs
    (empty list means the query may proceed)."""
    unmet: List[Dict[str, Any]] = []
    avail = ", ".join(sorted({f"{i['format']} {i['band_description']}" for i in inputs})) or "8-bit imagery"
    georef = any(i.get("georeferenced") for i in inputs)

    if mode == "single" and _CHANGE_INTENT.search(query):
        unmet.append({
            "kind": "INSUFFICIENT_INPUT", "code": "CHANGE_NEEDS_TWO_OBSERVATIONS",
            "requested": "change analysis", "requires": "two observations of the same area at different times",
            "available": "one image",
            "explanation": "Change analysis requires two observations of the same geographic region at different times; only one image was provided.",
            "alternatives": ["Switch to Bi-temporal mode and upload an earlier and a later image."],
        })
    if mode == "single" and _FUSION_INTENT.search(query):
        unmet.append({
            "kind": "INSUFFICIENT_INPUT", "code": "FUSION_NEEDS_OPTICAL_AND_SAR",
            "requested": "optical-SAR joint analysis", "requires": "a co-registered optical image and a SAR image",
            "available": "one image",
            "explanation": "Joint optical-SAR analysis requires both an optical and a SAR image; only one image was provided.",
            "alternatives": ["Switch to Optical + SAR mode and upload both images."],
        })
    if _SPECTRAL.search(query):
        unmet.append(_unavail(
            "SPECTRAL_BANDS_UNAVAILABLE", "Spectral-index analysis", "multispectral bands (e.g. Red + NIR, Green + SWIR)", avail,
            ["Ask for a colour-based vegetation / water / built-up estimate from RGB (clearly labelled as a heuristic)."],
        ))
    if _SAR_CAL.search(query):
        unmet.append(_unavail(
            "CALIBRATED_SAR_UNAVAILABLE", "Calibrated SAR analysis", "calibrated SAR data (sigma0, polarisation, orbit/processing metadata)", avail,
            ["Use Optical + SAR mode for the structural proxy (edges / dark regions), which makes no radiometric claims."],
        ))
    if _AREA.search(query):
        unmet.append(_unavail(
            "PHYSICAL_AREA_UNAVAILABLE", "Physical area measurement", "valid georeferencing (CRS + transform) and a geospatial processing pipeline",
            "georeferencing metadata is " + ("present but Lite v1 does not compute ground measurements" if georef else "absent"),
            ["Ask for the percentage of the image covered (pixel share), which needs no georeferencing."],
        ))
    if _COORDS.search(query):
        unmet.append(_unavail(
            "GEOCOORDINATES_UNAVAILABLE", "Geographic coordinates of features", "georeferencing and a geospatial processing pipeline",
            "georeferencing metadata is " + ("present (see the geospatial block of this response) but features are not localised in geographic space" if georef else "absent"),
            ["Use /api/diagnostics on a GeoTIFF to inspect CRS and bounds."],
        ))
    if _ACQ.search(query):
        unmet.append(_unavail(
            "ACQUISITION_METADATA_UNAVAILABLE", "Acquisition date / sensor identification", "image metadata or a STAC item",
            avail, ["Submit the STAC item to /api/stac/inspect to read acquisition date and sensor."],
        ))
    if _COUNT.search(query):
        unmet.append(_unavail(
            "OBJECT_COUNTING_UNAVAILABLE", "Object counting", "a trained object-detection model", avail,
            ["Ask for the share of the image classified as built-up (a pixel-colour heuristic, not a count)."],
        ))
    return unmet
