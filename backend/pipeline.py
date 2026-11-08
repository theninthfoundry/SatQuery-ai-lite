"""
backend/pipeline.py

The analysis pipeline. One entry point, `run_analysis`, takes an already
schema-validated request and returns `(http_status, response_dict)`.

Stage order (each stage is a trace entry):
  1. request / input-count check
  2. task classification
  3. hardened decode of every image (limits, formats, bit depth, hashing)
  4. capability gate (can this input answer this question at all?)
  5. working-grid preparation
  6. diagnostics: quality, alignment, suitability, geospatial pair checks
  7. tool / engine execution
  8. reliability assessment (replaces v0's invented confidence)
  9. provenance, methodology, limitations, report

The interface renders this response and invents nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend import capabilities, config, diagnostics
from backend.controller import (
    REQUIRED_IMAGE_COUNT,
    TASK_LABELS,
    InputMode,
    TaskType,
    classify_task,
)
from backend.evidence.evidence_summary import reliability_suffix
from backend.geospatial.validation import validate_geospatial_pair
from backend.imgops import encode_png_b64, to_gray
from backend.ingest import DecodedImage, InputError, decode_image, prepare_pair, prepare_single
from backend.methodology import limitations_for, methodology_for
from backend.reliability import assess_reliability
from backend.report import build_report
from backend.tools import caption_tool, change_tool, fusion_tool, grounding_tool, vqa_tool
from backend.uncertainty.sensitivity import ThresholdError, normalise_thresholds

TOOL_NAMES = {
    TaskType.CAPTION: caption_tool.TOOL_NAME,
    TaskType.VQA: vqa_tool.TOOL_NAME,
    TaskType.GROUNDING: grounding_tool.TOOL_NAME,
    TaskType.CHANGE_DESCRIBE: change_tool.TOOL_NAME_DESCRIBE,
    TaskType.CHANGE_VQA: change_tool.TOOL_NAME_VQA,
    TaskType.FUSION: fusion_tool.TOOL_NAME,
}

ROLES = {
    InputMode.SINGLE: ["image"],
    InputMode.CROSS: ["optical", "sar"],
    InputMode.BITEMPORAL: ["time_a", "time_b"],
}

COLOUR_TASKS = {TaskType.CAPTION, TaskType.VQA, TaskType.GROUNDING, TaskType.CHANGE_VQA, TaskType.FUSION}


# ── helpers ──────────────────────────────────────────────────────────────

def json_safe(obj: Any) -> Any:
    """Recursively convert numpy types and non-finite floats to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


class _Trace:
    def __init__(self) -> None:
        self.entries: List[Dict[str, str]] = []
        self.timings: Dict[str, float] = {}
        self._t0 = time.perf_counter()
        self._last = self._t0

    def log(self, label: str, detail: str) -> None:
        now = time.perf_counter()
        self.timings[label] = round((now - self._last) * 1000.0, 1)
        self._last = now
        self.entries.append({"time": time.strftime("%H:%M:%S", time.gmtime()), "label": label, "detail": detail})


def _base(request_id: str, req: Dict[str, Any], trace: _Trace) -> Dict[str, Any]:
    return {
        "schema_version": config.SCHEMA_VERSION,
        "engine_version": config.ENGINE_VERSION,
        "request_id": request_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": req.get("query"),
        "mode": req.get("mode"),
        "trace": trace.entries,
        "timings_ms": trace.timings,
        "confidence": None,
    }


def error_response(
    request_id: str, req: Dict[str, Any], trace: _Trace, status: str, code: str, message: str,
    http: int, extra: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[str, Any]]:
    body = _base(request_id, req, trace)
    body.update({
        "status": status, "ok": False,
        "error": {"code": code, "message": message},
        "answer": "", "task": None, "task_label": None, "tool": None,
    })
    if extra:
        body.update(extra)
    return http, json_safe(body)


def _band_description(img: DecodedImage) -> str:
    return "8-bit grayscale" if img.source_mode in ("L", "LA", "1") else "8-bit RGB"


def _canonical_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ── main entry ───────────────────────────────────────────────────────────

def run_analysis(req: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    request_id = str(uuid.uuid4())
    trace = _Trace()
    mode = InputMode(req["mode"])
    query: str = req["query"]
    options = req.get("options") or {}
    trace.log("QUERY", query)

    # 1 ─ input count
    required = REQUIRED_IMAGE_COUNT[mode]
    images_b64: List[str] = req["images"]
    task = classify_task(query, mode)
    task_label = TASK_LABELS[task]
    trace.log("TASK CLASSIFIED", f"{task.value} - {task_label}")
    if len(images_b64) != required:
        msg = f"{mode.value} mode requires exactly {required} image(s); {len(images_b64)} provided."
        trace.log("INPUT CHECK", msg)
        trace.log("HALT", "Wrong number of images for the selected mode.")
        return error_response(request_id, req, trace, "INSUFFICIENT_INPUT", "WRONG_IMAGE_COUNT", msg, 422,
                              {"task": task.value, "task_label": task_label})

    # 2 ─ change-parameter validation
    try:
        base_t = int(options.get("base_threshold", config.CHANGE.base_threshold))
        thresholds = normalise_thresholds(base_t, options.get("thresholds"))
        grid_size = int(options.get("grid_size", config.CHANGE.grid_size))
        if not (config.LIMITS.min_grid_size <= grid_size <= config.LIMITS.max_grid_size):
            raise ThresholdError(f"grid_size must be between {config.LIMITS.min_grid_size} and {config.LIMITS.max_grid_size}")
    except (ThresholdError, ValueError, TypeError) as e:
        trace.log("HALT", str(e))
        return error_response(request_id, req, trace, "REJECTED_INPUT", "INVALID_OPTIONS", str(e), 422,
                              {"task": task.value, "task_label": task_label})

    # 3 ─ decode
    decoded: List[DecodedImage] = []
    try:
        total = 0
        for i, b64 in enumerate(images_b64):
            d = decode_image(b64)
            total += d.nbytes
            decoded.append(d)
        if total > config.LIMITS.max_total_image_bytes:
            raise InputError("IMAGES_TOO_LARGE", f"Combined image size {total / 1e6:.2f} MB exceeds the "
                             f"{config.LIMITS.max_total_image_bytes / 1e6:.1f} MB limit.", 413)
    except InputError as e:
        trace.log("INPUT CHECK", f"{e.code}: {e.message}")
        trace.log("HALT", "Input rejected before processing.")
        extra = {"task": task.value, "task_label": task_label}
        if e.details.get("geospatial") is not None:
            extra["geospatial"] = e.details["geospatial"]
        return error_response(request_id, req, trace, "REJECTED_INPUT", e.code, e.message, e.status, extra)
    trace.log("INPUT CHECK", f"{len(decoded)}/{required} image(s) decoded and validated: "
              + ", ".join(f"{d.fmt} {d.original_size[0]}x{d.original_size[1]}" for d in decoded))

    roles = ROLES[mode]
    client_notes = {n["index"]: n for n in (req.get("client_notes") or [])}
    inputs_meta: List[Dict[str, Any]] = []
    for i, d in enumerate(decoded):
        inputs_meta.append({
            "index": i, "role": roles[i], "sha256": d.sha256, "bytes": d.nbytes, "format": d.fmt,
            "source_mode": d.source_mode, "original_size": list(d.original_size),
            "band_description": _band_description(d),
            "georeferenced": bool(d.geospatial and d.geospatial["georeferenced"]),
            "geospatial": d.geospatial, "geospatial_error": d.geospatial_error,
            "warnings": d.warnings,
            "client_preprocessing": client_notes.get(i),
        })

    # 4 ─ capability gate
    unmet = capabilities.check_query(query, mode.value, inputs_meta)
    if unmet:
        kind = unmet[0]["kind"]
        status = "INSUFFICIENT_INPUT" if kind == "INSUFFICIENT_INPUT" else "UNAVAILABLE_CAPABILITY"
        trace.log("CAPABILITY CHECK", "; ".join(u["code"] for u in unmet))
        trace.log("HALT", "The request depends on information or capabilities this deployment/input does not have.")
        body = _base(request_id, req, trace)
        body.update({
            "status": status, "ok": False, "task": task.value, "task_label": task_label, "tool": None,
            "answer": " ".join(u["explanation"] for u in unmet),
            "error": {"code": unmet[0]["code"], "message": unmet[0]["explanation"]},
            "unmet_requirements": unmet,
            "supported_alternatives": sorted({a for u in unmet for a in u["alternatives"]}),
            "inputs": inputs_meta,
        })
        return (422 if status == "INSUFFICIENT_INPUT" else 200), json_safe(body)
    trace.log("CAPABILITY CHECK", "query is answerable with the provided inputs")

    # 5 ─ working grid
    if mode == InputMode.SINGLE:
        arr, grid_info = prepare_single(decoded[0])
        arrays = [arr]
    else:
        a, b, grid_info = prepare_pair(decoded[0], decoded[1])
        arrays = [a, b]
    for i, arr in enumerate(arrays):
        inputs_meta[i]["working_size"] = [int(arr.shape[1]), int(arr.shape[0])]
    trace.log("WORKING GRID", f"{arrays[0].shape[1]}x{arrays[0].shape[0]}"
              + (" (resampled)" if grid_info["resampled"] not in (False, [False, False]) else " (native)"))

    # 6 ─ diagnostics
    qualities = [diagnostics.assess_image_quality(arr, decoded[i].original_size) for i, arr in enumerate(arrays)]
    alignment: Optional[Dict[str, Any]] = None
    aspect: Optional[Dict[str, Any]] = None
    geo_pair: Optional[Dict[str, Any]] = None
    if mode != InputMode.SINGLE:
        aspect = diagnostics.aspect_check(decoded[0].original_size, decoded[1].original_size)
        geo_pair = validate_geospatial_pair(decoded[0].geospatial, decoded[1].geospatial)
    if mode == InputMode.BITEMPORAL:
        alignment = diagnostics.estimate_alignment(to_gray(arrays[0]), to_gray(arrays[1]), aspect=aspect)
    elif mode == InputMode.CROSS:
        alignment = {
            "method": "not_assessed", "state": "NOT_ASSESSED", "limitation": diagnostics.ALIGNMENT_LIMITATION,
            "reasons": ["Cross-modal (optical vs SAR) alignment cannot be estimated by intensity correlation; co-registration is assumed."],
            "aspect": aspect,
        }
    suitability = diagnostics.assess_suitability(
        qualities, alignment if mode == InputMode.BITEMPORAL else None, requires_alignment=mode == InputMode.BITEMPORAL
    )
    trace.log("DIAGNOSTICS", f"quality {', '.join(q['overall'] for q in qualities)}; "
              + (f"alignment {alignment['state']}; " if alignment else "") + f"suitability {suitability['decision']}")

    diagnostics_block: Dict[str, Any] = {
        "quality": qualities, "alignment": alignment, "suitability": suitability,
        "geospatial_pair": geo_pair,
    }
    if suitability["decision"] == "REJECT":
        trace.log("HALT", "An input is unusable for analysis.")
        body = _base(request_id, req, trace)
        body.update({
            "status": "REJECTED_INPUT", "ok": False, "task": task.value, "task_label": task_label, "tool": None,
            "answer": "The input cannot be analysed: " + "; ".join(suitability["reasons"]),
            "error": {"code": "UNUSABLE_IMAGE", "message": "; ".join(suitability["reasons"])},
            "diagnostics": diagnostics_block, "inputs": inputs_meta,
        })
        return 422, json_safe(body)

    # 7 ─ tool execution
    tool_name = TOOL_NAMES[task]
    trace.log("TOOL SELECTED", tool_name)
    context = {
        "base_threshold": base_t, "thresholds": thresholds, "grid_size": grid_size, "alignment": alignment,
    }
    try:
        if task == TaskType.CAPTION:
            result = caption_tool.run(arrays[0], query)
        elif task == TaskType.VQA:
            result = vqa_tool.run(arrays[0], query)
        elif task == TaskType.GROUNDING:
            result = grounding_tool.run(arrays[0], query)
        elif task == TaskType.CHANGE_DESCRIBE:
            result = change_tool.run_describe(arrays[0], arrays[1], query, context)
        elif task == TaskType.CHANGE_VQA:
            result = change_tool.run_vqa(arrays[0], arrays[1], query, context)
        else:
            result = fusion_tool.run(arrays[0], arrays[1], query)
    except Exception as e:  # noqa: BLE001 - surfaced in the trace, never as a stack trace
        trace.log("ERROR", f"{type(e).__name__} during tool execution")
        return error_response(request_id, req, trace, "ERROR", "TOOL_FAILURE",
                              "The analysis tool failed unexpectedly on this input.", 500,
                              {"task": task.value, "task_label": task_label, "tool": tool_name,
                               "diagnostics": json_safe(diagnostics_block)})
    trace.log("EXECUTION COMPLETE", "tool finished; evidence generated")

    # 8 ─ reliability
    method_concern = None
    if task in (TaskType.CAPTION, TaskType.VQA, TaskType.GROUNDING, TaskType.CHANGE_VQA, TaskType.FUSION):
        method_concern = "Result comes from fixed RGB colour rules / proxies, not a trained model."
    extra: List[Dict[str, Any]] = []
    if task in (TaskType.CAPTION, TaskType.VQA, TaskType.GROUNDING) and qualities[0]["metrics"]["is_grayscale"]:
        extra.append({"name": "colour_input", "status": "CONCERN", "value": None,
                      "detail": "Input is grayscale; colour-threshold land-cover classes are not meaningful."})
    if task == TaskType.CHANGE_VQA and any(q["metrics"]["is_grayscale"] for q in qualities):
        extra.append({"name": "colour_input", "status": "CONCERN", "value": None,
                      "detail": "At least one input is grayscale; colour-threshold class shares are not meaningful."})
    if task == TaskType.FUSION:
        if qualities[0]["metrics"]["is_grayscale"]:
            extra.append({"name": "optical_modality", "status": "CONCERN", "value": None,
                          "detail": "The optical image is single-channel; colour-threshold classes are not meaningful."})
        if not qualities[1]["metrics"]["is_grayscale"] and qualities[1]["metrics"]["mean_saturation"] > 0.15:
            extra.append({"name": "sar_modality", "status": "CONCERN", "value": qualities[1]["metrics"]["mean_saturation"],
                          "detail": "The SAR image is a colour image (mean saturation "
                                    f"{qualities[1]['metrics']['mean_saturation']:.2f}); it may be an optical image or a colourised product."})
    if aspect and mode == InputMode.CROSS and aspect["state"] != "GOOD":
        extra.append({"name": "aspect_ratio", "status": "SEVERE" if aspect["state"] == "UNRELIABLE" else "CONCERN",
                      "value": aspect["aspect_mismatch"],
                      "detail": f"Original aspect ratios differ by {aspect['aspect_mismatch'] * 100:.1f}%; the second image was resampled onto the first image's grid."})
    if geo_pair:
        for f in geo_pair["findings"]:
            if f["severity"] in ("WARNING", "SEVERE"):
                extra.append({"name": f"geospatial:{f['code']}", "status": "SEVERE" if f["severity"] == "SEVERE" else "CONCERN",
                              "value": None, "detail": f["message"]})

    change_ev = result.evidence if task in (TaskType.CHANGE_DESCRIBE, TaskType.CHANGE_VQA) else None
    reliability = assess_reliability(
        qualities, suitability, alignment if mode == InputMode.BITEMPORAL else None, change_ev,
        method_concern=method_concern, extra_concerns=extra,
    )
    answer = result.answer + reliability_suffix(reliability["level"], reliability["concerns"])
    trace.log("RELIABILITY", f"evidence level {reliability['level']}")

    # 9 ─ layers, provenance, report
    layers = result.layers
    if not layers:
        layers = [{"id": "original", "label": "Input", "description": "Input on the working grid.",
                   "width": int(arrays[0].shape[1]), "height": int(arrays[0].shape[0]), "data": encode_png_b64(arrays[0])}]
        if result.overlay_base64:
            layers.append({"id": "overlay", "label": "Visual evidence", "description": "Pixels matching the queried class.",
                           "width": int(arrays[0].shape[1]), "height": int(arrays[0].shape[0]), "data": result.overlay_base64})

    parameters = {
        "mode": mode.value, "task": task.value,
        "base_threshold": base_t, "thresholds": thresholds, "grid_size": grid_size,
        "single_max_side": config.PROCESSING.single_max_side, "pair_max_side": config.PROCESSING.pair_max_side,
    }
    processing_steps = [
        "Base64 decoded strictly; SHA-256 computed over the raw bytes",
        "Decoded with Pillow; first frame only; converted to 8-bit RGB",
        (f"Resized to {grid_info['working_size'][0]}x{grid_info['working_size'][1]} (LANCZOS)" if grid_info["resampled"] not in (False, [False, False])
         else f"Used at native size {grid_info['working_size'][0]}x{grid_info['working_size'][1]}"),
        f"gray = {config.PROCESSING.gray_definition}",
    ]
    for im in inputs_meta:
        if im["georeferenced"]:
            processing_steps.append(
                f"input {im['index'] + 1} is georeferenced; pixels were analysed in image space and no georeferenced measurement was produced")
    analysis_id = _canonical_hash({
        "engine": config.ENGINE_VERSION, "schema": config.SCHEMA_VERSION, "mode": mode.value, "task": task.value,
        "query": " ".join(query.lower().split()), "inputs": [d.sha256 for d in decoded], "parameters": parameters,
    })

    body = _base(request_id, req, trace)
    warn_status = (
        reliability["level"] != "SUPPORTED" or suitability["decision"] != "PROCEED"
        or any(im["warnings"] for im in inputs_meta)
    )
    body.update({
        "status": "COMPLETED_WITH_WARNINGS" if warn_status else "COMPLETED",
        "ok": True,
        "error": None,
        "analysis_id": analysis_id,
        "task": task.value,
        "task_label": task_label,
        "tool": tool_name,
        "answer": answer,
        "reliability": reliability,
        "diagnostics": diagnostics_block,
        "evidence": result.evidence,
        "layers": layers,
        "overlay_base64": result.overlay_base64,
        "inputs": inputs_meta,
        "methodology": methodology_for(task.value, parameters),
        "limitations": limitations_for(task.value),
        "provenance": {
            "analysis_id": analysis_id,
            "inputs": [{k: im[k] for k in ("index", "role", "sha256", "bytes", "format", "original_size", "working_size",
                                          "georeferenced", "client_preprocessing")} for im in inputs_meta],
            "working_grid": grid_info,
            "parameters": parameters,
            "processing_steps": processing_steps,
            "replay_note": "The same inputs, query and parameters produce the same analysis_id and the same measurements.",
        },
    })
    trace.log("REPORT READY", "evidence, methodology, limitations and report generated")
    body["trace"] = trace.entries
    body["timings_ms"] = trace.timings
    safe = json_safe(body)
    safe["report"] = build_report(safe)
    return 200, safe


# ── /api/diagnostics ─────────────────────────────────────────────────────

def run_diagnostics(req: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """Quality, alignment, suitability and geospatial metadata for 1-2 images,
    without running any analysis tool."""
    request_id = str(uuid.uuid4())
    trace = _Trace()
    images_b64: List[str] = req["images"]
    mode = "single" if len(images_b64) == 1 else "pair"
    trace.log("DIAGNOSTICS REQUEST", f"{len(images_b64)} image(s)")
    decoded: List[DecodedImage] = []
    try:
        total = 0
        for b64 in images_b64:
            d = decode_image(b64)
            total += d.nbytes
            decoded.append(d)
        if total > config.LIMITS.max_total_image_bytes:
            raise InputError("IMAGES_TOO_LARGE", f"Combined image size {total / 1e6:.2f} MB exceeds the "
                             f"{config.LIMITS.max_total_image_bytes / 1e6:.1f} MB limit.", 413)
    except InputError as e:
        trace.log("HALT", f"{e.code}: {e.message}")
        extra = {"geospatial": e.details["geospatial"]} if e.details.get("geospatial") is not None else None
        return error_response(request_id, {"query": None, "mode": mode}, trace, "REJECTED_INPUT", e.code, e.message, e.status, extra)

    if len(decoded) == 1:
        arr, grid_info = prepare_single(decoded[0])
        arrays = [arr]
    else:
        a, b, grid_info = prepare_pair(decoded[0], decoded[1])
        arrays = [a, b]
    qualities = [diagnostics.assess_image_quality(arr, decoded[i].original_size) for i, arr in enumerate(arrays)]
    aspect = alignment = geo_pair = None
    if len(decoded) == 2:
        aspect = diagnostics.aspect_check(decoded[0].original_size, decoded[1].original_size)
        alignment = diagnostics.estimate_alignment(to_gray(arrays[0]), to_gray(arrays[1]), aspect=aspect)
        geo_pair = validate_geospatial_pair(decoded[0].geospatial, decoded[1].geospatial)
    suitability = diagnostics.assess_suitability(qualities, alignment, requires_alignment=alignment is not None)
    trace.log("DIAGNOSTICS", f"suitability {suitability['decision']}")
    body = _base(request_id, {"query": None, "mode": mode}, trace)
    body.update({
        "status": "COMPLETED", "ok": True, "error": None,
        "diagnostics": {"quality": qualities, "alignment": alignment, "suitability": suitability, "geospatial_pair": geo_pair},
        "inputs": [{
            "index": i, "sha256": d.sha256, "bytes": d.nbytes, "format": d.fmt, "source_mode": d.source_mode,
            "original_size": list(d.original_size), "working_size": [int(arrays[i].shape[1]), int(arrays[i].shape[0])],
            "georeferenced": bool(d.geospatial and d.geospatial["georeferenced"]),
            "geospatial": d.geospatial, "geospatial_error": d.geospatial_error, "warnings": d.warnings,
        } for i, d in enumerate(decoded)],
        "working_grid": grid_info,
    })
    return 200, json_safe(body)
