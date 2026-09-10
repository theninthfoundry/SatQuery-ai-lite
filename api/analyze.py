"""
api/analyze.py

Vercel Serverless Function — single POST endpoint for the SatQuery AI
agentic pipeline. Receives base64 images + query, runs the full
controller logic, returns JSON with answer, confidence, overlay, and trace.
"""

from __future__ import annotations

import sys
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure the project root is on the Python path so `backend.*` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.controller import (
    InputMode,
    TaskType,
    TraceStep,
    classify_task,
    check_compatibility,
    TASK_LABELS,
)
from backend.image_utils import (
    decode_base64_image,
    image_to_numpy,
    resize_for_processing,
)
from backend.tools import caption_tool, vqa_tool, grounding_tool, change_tool, fusion_tool


# ── FastAPI app ───────────────────────────────────────────────────────────

app = FastAPI(title="SatQuery AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    mode: str              # "single" | "cross" | "bitemporal"
    query: str
    images: List[str]      # base64 data-URLs


class TraceEntry(BaseModel):
    time: str
    label: str
    detail: str


class AnalyzeResponse(BaseModel):
    request_id: str
    ok: bool
    query: str
    mode: str
    task: str
    task_label: str
    tool: str
    answer: str
    confidence: float
    evidence: Dict[str, Any]
    overlay_base64: Optional[str]
    trace: List[TraceEntry]
    error: Optional[str] = None


# ── Tool registry ────────────────────────────────────────────────────────

TOOL_NAMES = {
    TaskType.CAPTION: caption_tool.TOOL_NAME,
    TaskType.VQA: vqa_tool.TOOL_NAME,
    TaskType.GROUNDING: grounding_tool.TOOL_NAME,
    TaskType.CHANGE_DESCRIBE: change_tool.TOOL_NAME_DESCRIBE,
    TaskType.CHANGE_VQA: change_tool.TOOL_NAME_VQA,
    TaskType.FUSION: fusion_tool.TOOL_NAME,
}


# ── Health check ──────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "satquery-ai"}


# ── Main endpoint ─────────────────────────────────────────────────────────

@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    request_id = str(uuid.uuid4())
    trace: List[TraceEntry] = []

    def log(label: str, detail: str):
        trace.append(TraceEntry(
            time=time.strftime("%H:%M:%S", time.gmtime()),
            label=label,
            detail=detail,
        ))

    # 1. Parse mode
    try:
        mode = InputMode(req.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {req.mode}")

    log("QUERY", req.query)

    # 2. Classify task
    task = classify_task(req.query, mode)
    task_label = TASK_LABELS[task]
    log("TASK CLASSIFIED", f"{task.value} — {task_label}")

    # 3. Check compatibility
    ok, msg = check_compatibility(mode, len(req.images))
    log("INPUT CHECK", msg)
    if not ok:
        log("HALT", "Insufficient inputs for the selected task.")
        return AnalyzeResponse(
            request_id=request_id, ok=False, query=req.query, mode=req.mode,
            task=task.value, task_label=task_label, tool="",
            answer="", confidence=0.0, evidence={},
            overlay_base64=None, trace=trace, error=msg,
        )

    # 4. Select tool
    tool_name = TOOL_NAMES[task]
    log("TOOL SELECTED", tool_name)

    # 5. Decode and resize images
    try:
        pil_images = [decode_base64_image(img) for img in req.images]
        np_images = [image_to_numpy(resize_for_processing(img, req.mode)) for img in pil_images]
    except Exception as e:
        log("ERROR", f"Image decode failed: {e}")
        return AnalyzeResponse(
            request_id=request_id, ok=False, query=req.query, mode=req.mode,
            task=task.value, task_label=task_label, tool=tool_name,
            answer="", confidence=0.0, evidence={},
            overlay_base64=None, trace=trace, error=str(e),
        )

    # 6. Execute tool
    try:
        if task == TaskType.CAPTION:
            result = caption_tool.run(np_images[0], req.query)
        elif task == TaskType.VQA:
            result = vqa_tool.run(np_images[0], req.query)
        elif task == TaskType.GROUNDING:
            result = grounding_tool.run(np_images[0], req.query)
        elif task == TaskType.CHANGE_DESCRIBE:
            result = change_tool.run_describe(np_images[0], np_images[1], req.query)
        elif task == TaskType.CHANGE_VQA:
            result = change_tool.run_vqa(np_images[0], np_images[1], req.query)
        elif task == TaskType.FUSION:
            result = fusion_tool.run(np_images[0], np_images[1], req.query)
        else:
            raise ValueError(f"Unknown task type: {task}")
    except Exception as e:
        log("ERROR", str(e))
        return AnalyzeResponse(
            request_id=request_id, ok=False, query=req.query, mode=req.mode,
            task=task.value, task_label=task_label, tool=tool_name,
            answer="", confidence=0.0, evidence={},
            overlay_base64=None, trace=trace, error=str(e),
        )

    log("EXECUTION COMPLETE", f"confidence {round(result.confidence * 100)}%")
    log("REPORT READY", "Evidence overlay and execution trace available for download.")

    return AnalyzeResponse(
        request_id=request_id,
        ok=True,
        query=req.query,
        mode=req.mode,
        task=task.value,
        task_label=task_label,
        tool=tool_name,
        answer=result.answer,
        confidence=result.confidence,
        evidence=result.evidence,
        overlay_base64=result.overlay_base64,
        trace=trace,
    )
