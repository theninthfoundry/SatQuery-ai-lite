"""
backend/report.py

Structured report derived ONLY from an analysis result. The interface
renders / downloads what is built here; it writes no findings of its own,
so the on-screen result, the JSON, the CSV and the printable report cannot
disagree.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List


def _csv_safe(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _to_csv(header: List[str], rows: List[List[Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow([_csv_safe(v) for v in r])
    return buf.getvalue()


def regions_csv(evidence: Dict[str, Any]) -> str:
    grid = evidence.get("spatial_evidence")
    if not grid:
        return ""
    header = [
        "region_id", "row", "col", "x0", "y0", "x1", "y1", "analysed_pixels", "changed_pixels", "change_pct",
        "share_of_total_change_pct", "mean_abs_difference", "p95_abs_difference", "stable_changed_pixels",
        "ambiguous_pixels", "retention_at_strictest", "evidence_strength", "quality_warning", "interpretation",
    ]
    rows = []
    for c in grid["cells"]:
        x0, y0, x1, y1 = c["bbox_px"]
        rows.append([
            c["region_id"], c["row"], c["col"], x0, y0, x1, y1, c["analysed_pixel_count"], c["changed_pixel_count"],
            c["change_percentage"], c["share_of_total_change_pct"], c["mean_abs_difference"], c["p95_abs_difference"],
            c["stable_changed_pixels"], c["ambiguous_pixels"], c["retention_at_strictest"], c["evidence_strength"],
            c["quality_warning"], c["interpretation"],
        ])
    return _to_csv(header, rows)


def sensitivity_csv(evidence: Dict[str, Any]) -> str:
    sens = evidence.get("sensitivity")
    if not sens:
        return ""
    header = ["threshold", "is_base", "changed_pixels", "changed_pct", "delta_vs_base_pp", "relative_to_base", "iou_with_base"]
    rows = [[r["threshold"], r["is_base"], r["changed_pixels"], r["changed_pct"], r["delta_vs_base_pp"], r["relative_to_base"], r["iou_with_base"]] for r in sens["thresholds"]]
    return _to_csv(header, rows)


def build_report(result: Dict[str, Any]) -> Dict[str, Any]:
    ev = result.get("evidence") or {}
    diag = result.get("diagnostics") or {}
    sections: List[Dict[str, Any]] = []

    sections.append({"title": "Analysis summary", "paragraphs": [result["answer"]], "bullets": [
        f"Task: {result['task_label']} ({result['tool']})",
        f"Status: {result['status']}",
        f"Evidence level: {result['reliability']['level']}",
        f"Analysis ID: {result['analysis_id']}",
    ]})

    sections.append({"title": "Inputs", "paragraphs": [], "bullets": [
        f"{i['role']}: {i['format']} {i['original_size'][0]}x{i['original_size'][1]}px, {i['bytes']:,} bytes, sha256 {i['sha256'][:16]}..., "
        f"georeferenced: {'yes' if i['georeferenced'] else 'no'}"
        for i in result["provenance"]["inputs"]
    ]})

    if diag.get("quality"):
        bullets = []
        for i, q in enumerate(diag["quality"]):
            bullets.append(f"Image {i + 1}: {q['overall']} (sharpness {q['levels']['sharpness']}, contrast {q['levels']['contrast']}, "
                           f"clipping {q['levels']['clipping']}, noise {q['levels']['noise']}, resolution {q['levels']['resolution']})")
            bullets.extend(f"  - {f['message']}" for f in q["flags"])
        sections.append({"title": "Image quality", "paragraphs": [diag["suitability"]["decision"].replace("_", " ").title()], "bullets": bullets})

    al = diag.get("alignment")
    if al and al.get("state") not in (None, "NOT_ASSESSED"):
        sections.append({"title": "Temporal alignment", "paragraphs": [al["limitation"]], "bullets": [f"State: {al['state']}"] + list(al["reasons"])})

    if ev.get("statements"):
        sections.append({"title": "Findings and evidence", "paragraphs": [], "bullets": [s["text"] for s in ev["statements"]]})

    if ev.get("spatial_evidence"):
        cells = [c for c in ev["spatial_evidence"]["cells"] if c["evidence_strength"] != "NONE"]
        sections.append({"title": "Spatial evidence (cells with evidence)", "paragraphs": [ev["spatial_evidence"]["evidence_strength_rule"]],
                         "bullets": [f"{c['region_id']} [{c['evidence_strength']}]: {c['interpretation']}" for c in cells] or ["No cell shows pixel-difference evidence."]})

    sections.append({"title": "Reliability factors", "paragraphs": [result["reliability"]["rule"]],
                     "bullets": [f"{f['name']} [{f['status']}]: {f['detail']}" for f in result["reliability"]["factors"]]})
    sections.append({"title": "Methodology", "paragraphs": [result["methodology"]["name"]], "bullets": list(result["methodology"]["steps"])})
    sections.append({"title": "Limitations", "paragraphs": [], "bullets": list(result["limitations"])})
    sections.append({"title": "Provenance", "paragraphs": [], "bullets": [
        f"Engine {result['engine_version']}, schema {result['schema_version']}",
        f"Created {result['created_at']}",
        f"Parameters: {result['provenance']['parameters']}",
    ] + list(result["provenance"]["processing_steps"])})

    md: List[str] = [f"# SatQuery AI Lite analysis report", "", f"**Query:** {result['query']}", ""]
    for s in sections:
        md.append(f"## {s['title']}")
        md.extend(s["paragraphs"])
        md.extend(f"- {b}" for b in s["bullets"])
        md.append("")
    return {
        "sections": sections,
        "markdown": "\n".join(md),
        "regions_csv": regions_csv(ev),
        "sensitivity_csv": sensitivity_csv(ev),
    }
