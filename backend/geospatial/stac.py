"""
backend/geospatial/stac.py

STAC Item METADATA inspection (Geospatial Lite Bridge, Phase B).

Input is a STAC Item JSON object supplied by the caller. This module never
fetches anything: no asset hrefs are opened and no catalog is queried, which
removes the SSRF and data-egress surface from the serverless deployment.
Inspecting metadata is not image analysis, and the output says so: assets
are classified by what Lite can currently do with them (nothing, until
Phase C raster processing exists, except rendered PNG/JPEG previews).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend import config

_ORDER = {"INFO": 0, "WARNING": 1, "SEVERE": 2}


class StacError(ValueError):
    pass


def _f(code: str, severity: str, message: str) -> Dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def _parse_dt(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _bbox2d(bbox: Any) -> Optional[List[float]]:
    if not isinstance(bbox, list) or len(bbox) not in (4, 6):
        return None
    try:
        v = [float(x) for x in bbox]
    except (TypeError, ValueError):
        return None
    return v if len(v) == 4 else [v[0], v[1], v[3], v[4]]


def _bbox_iou(a: List[float], b: List[float]) -> Optional[float]:
    if a[0] > a[2] or b[0] > b[2]:
        return None  # antimeridian-crossing boxes are not handled
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if union <= 0 else inter / union


def _modality(props: Dict[str, Any], collection: Optional[str], assets: Dict[str, Any]) -> Dict[str, Any]:
    text = " ".join(str(x).lower() for x in (collection, props.get("platform"), props.get("constellation")) if x)
    if any(k in props for k in ("sar:polarizations", "sar:instrument_mode", "sar:frequency_band")) or "sentinel-1" in text or "sentinel1" in text:
        return {"value": "sar", "basis": "SAR extension fields or Sentinel-1 identifiers"}
    has_eo = "eo:bands" in props or any("eo:bands" in a for a in assets.values() if isinstance(a, dict))
    if has_eo or "sentinel-2" in text or "sentinel2" in text or "landsat" in text:
        return {"value": "optical", "basis": "EO extension bands or optical platform identifiers"}
    return {"value": "unknown", "basis": "no SAR/EO extension fields or recognised platform"}


def inspect_stac_item(item: Any, limits: config.Limits = config.LIMITS) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise StacError("STAC item must be a JSON object")
    if item.get("type") != "Feature":
        raise StacError("not a STAC Item: 'type' must be 'Feature'")
    if not isinstance(item.get("id"), str) or not item["id"]:
        raise StacError("STAC item is missing a string 'id'")
    props = item.get("properties")
    if not isinstance(props, dict):
        raise StacError("STAC item is missing 'properties'")
    assets = item.get("assets", {})
    if assets is None:
        assets = {}
    if not isinstance(assets, dict):
        raise StacError("'assets' must be an object")
    if len(assets) > limits.max_stac_assets:
        raise StacError(f"too many assets (max {limits.max_stac_assets})")

    findings: List[Dict[str, str]] = []
    dt = _parse_dt(props.get("datetime"))
    start, end = _parse_dt(props.get("start_datetime")), _parse_dt(props.get("end_datetime"))
    if dt is None and start is None:
        findings.append(_f("STAC_NO_DATETIME", "WARNING", "No parseable acquisition datetime (datetime / start_datetime)."))
    bbox = _bbox2d(item.get("bbox"))
    if bbox is None:
        findings.append(_f("STAC_NO_BBOX", "WARNING", "Missing or malformed bbox; spatial extent is unknown."))

    collection = item.get("collection")
    platform = props.get("platform") or props.get("constellation")
    eo_bands = props.get("eo:bands")
    bands: List[Dict[str, Any]] = []
    if isinstance(eo_bands, list):
        for b in eo_bands[:64]:
            if isinstance(b, dict):
                bands.append({k: b.get(k) for k in ("name", "common_name", "center_wavelength", "full_width_half_max") if k in b})

    asset_rows: List[Dict[str, Any]] = []
    for key, a in assets.items():
        if not isinstance(a, dict):
            continue
        mtype = str(a.get("type") or "").lower()
        roles = a.get("roles") if isinstance(a.get("roles"), list) else []
        is_tiff = "tiff" in mtype or "geotiff" in mtype
        is_preview = mtype in ("image/png", "image/jpeg", "image/jpg")
        row = {
            "key": key, "type": a.get("type"), "roles": roles, "title": a.get("title"),
            "href": a.get("href") if isinstance(a.get("href"), str) else None,
            "is_cog_or_geotiff": is_tiff,
            "is_rendered_preview": is_preview,
            "lite_v1_can_analyse": is_preview,
            "lite_v1_note": (
                "Rendered preview: can be downloaded by the user and uploaded as PNG/JPEG (no georeferencing)."
                if is_preview else
                "Raster data asset: requires Phase C raster processing (not implemented in Lite v1)." if is_tiff else
                "Not an image asset."
            ),
        }
        if isinstance(a.get("eo:bands"), list):
            row["band_names"] = [b.get("name") for b in a["eo:bands"] if isinstance(b, dict)][:16]
        asset_rows.append(row)

    cloud = props.get("eo:cloud_cover")
    cloud = float(cloud) if isinstance(cloud, (int, float)) and not isinstance(cloud, bool) else None
    epsg = props.get("proj:epsg")
    if epsg is None and isinstance(props.get("proj:code"), str) and props["proj:code"].upper().startswith("EPSG:"):
        try:
            epsg = int(props["proj:code"].split(":", 1)[1])
        except ValueError:
            epsg = None

    return {
        "id": item["id"],
        "stac_version": item.get("stac_version"),
        "collection": collection,
        "platform": platform,
        "instruments": props.get("instruments"),
        "datetime": props.get("datetime"),
        "start_datetime": props.get("start_datetime"),
        "end_datetime": props.get("end_datetime"),
        "gsd_m": props.get("gsd"),
        "bbox": bbox,
        "cloud_cover_pct": cloud,
        "epsg": epsg,
        "sar": {k: props.get(k) for k in ("sar:polarizations", "sar:instrument_mode", "sar:frequency_band", "sat:orbit_state") if k in props},
        "bands": bands,
        "modality": _modality(props, collection, assets),
        "assets": asset_rows,
        "analysable_asset_count": sum(1 for a in asset_rows if a["lite_v1_can_analyse"]),
        "findings": findings,
        "scope_note": "Metadata inspection only. Nothing was fetched; no asset was opened or analysed.",
        "_dt": dt or start,
    }


def compare_stac_items(a: Dict[str, Any], b: Dict[str, Any], purpose: str) -> Dict[str, Any]:
    """Check two inspected items for suitability. purpose: 'change' or 'fusion'."""
    if purpose not in ("change", "fusion"):
        raise StacError("purpose must be 'change' or 'fusion'")
    findings: List[Dict[str, str]] = []
    ma, mb = a["modality"]["value"], b["modality"]["value"]
    if purpose == "change":
        if "unknown" in (ma, mb):
            findings.append(_f("STAC_MODALITY_UNKNOWN", "WARNING", "Modality could not be inferred for at least one item."))
        elif ma != mb:
            findings.append(_f("STAC_MODALITY_MISMATCH", "SEVERE", f"Change analysis needs the same modality; got {ma} and {mb}."))
        if a.get("platform") and b.get("platform") and a["platform"] != b["platform"]:
            findings.append(_f("STAC_PLATFORM_DIFFERS", "INFO", f"Different platforms ({a['platform']} vs {b['platform']}); radiometry may differ."))
    else:
        if {ma, mb} != {"optical", "sar"}:
            findings.append(_f("STAC_FUSION_NEEDS_OPTICAL_AND_SAR", "SEVERE", f"Fusion needs one optical and one SAR item; got {ma} and {mb}."))

    da, db = a.get("_dt"), b.get("_dt")
    gap_days = None
    if da and db:
        gap_days = abs((db - da).total_seconds()) / 86400.0
        if purpose == "change":
            if gap_days == 0:
                findings.append(_f("STAC_SAME_TIME", "WARNING", "Both items share the same acquisition time; there is no temporal baseline."))
            if db < da:
                findings.append(_f("STAC_ORDER", "INFO", "Item B is earlier than item A; treat B as the 'before' image."))
        elif gap_days > 6:
            findings.append(_f("STAC_FUSION_TIME_GAP", "WARNING", f"{gap_days:.1f} days between acquisitions; surface conditions may differ between sensors."))
    else:
        findings.append(_f("STAC_TIME_UNKNOWN", "WARNING", "Acquisition time unknown for at least one item."))

    for label, it in (("A", a), ("B", b)):
        cc = it.get("cloud_cover_pct")
        if cc is not None and cc > 20:
            findings.append(_f("STAC_CLOUDY", "WARNING", f"Item {label} reports {cc:.0f}% cloud cover."))

    iou = None
    if a.get("bbox") and b.get("bbox"):
        iou = _bbox_iou(a["bbox"], b["bbox"])
        if iou is None:
            findings.append(_f("STAC_BBOX_UNSUPPORTED", "WARNING", "Bounding boxes crossing the antimeridian are not compared."))
        elif iou < 0.5:
            findings.append(_f("STAC_FOOTPRINT_MISMATCH", "SEVERE", f"Footprints overlap poorly (bbox IoU {iou:.2f})."))
        elif iou < 0.9:
            findings.append(_f("STAC_FOOTPRINT_PARTIAL", "WARNING", f"Footprints only partly overlap (bbox IoU {iou:.2f})."))
    if a.get("epsg") and b.get("epsg") and a["epsg"] != b["epsg"]:
        findings.append(_f("STAC_CRS_DIFFERS", "WARNING", f"Different CRS (EPSG:{a['epsg']} vs EPSG:{b['epsg']}); reprojection would be required."))

    worst = max((_ORDER[f["severity"]] for f in findings), default=0)
    return {
        "purpose": purpose,
        "state": {0: "COMPATIBLE", 1: "COMPATIBLE_WITH_WARNINGS", 2: "INCOMPATIBLE"}[worst],
        "gap_days": None if gap_days is None else round(gap_days, 3),
        "bbox_iou": None if iou is None else round(iou, 4),
        "findings": findings,
    }


def strip_private(inspected: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in inspected.items() if not k.startswith("_")}
