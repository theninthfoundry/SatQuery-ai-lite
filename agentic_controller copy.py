"""
agentic_controller.py

The orchestration core of SatQuery AI. This mirrors the architecture used in the
browser prototype (index.html) but is structured for real model backends instead
of the classical-CV heuristics used there.

Responsibilities (per the problem statement's "Agentic Model and Tool
Orchestration" section):
  1. Interpret the query and classify the requested task.
  2. Check the number, modality, format, metadata, and compatibility of inputs.
  3. Select one or more tools from a predefined registry.
  4. Configure only permitted task parameters and execute the selected workflow.
  5. Combine textual and spatial outputs, estimate confidence, and return
     visual evidence.
  6. Produce an auditable execution summary (task, tools, parameters, outputs).

This file intentionally contains NO deep-learning code — it is transport/
routing logic only. Real specialist models are plugged in via the ToolRegistry
(see tools/) as thin wrappers implementing the `SpecialistTool` interface.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class Modality(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"


class InputMode(str, Enum):
    SINGLE = "single"
    CROSS_MODAL = "cross_modal"       # co-registered optical + SAR
    BI_TEMPORAL = "bi_temporal"       # two dates, same modality


class TaskType(str, Enum):
    CAPTION = "caption"
    VQA = "vqa"
    GROUNDING = "grounding"
    CHANGE_DESCRIBE = "change_describe"
    CHANGE_VQA = "change_vqa"
    FUSION = "fusion"


@dataclass
class ImageInput:
    """One uploaded/registered image plus the metadata the controller checks."""
    path: str
    modality: Modality
    format: str                 # "GeoTIFF" | "TIFF" | "PNG" | "JPEG"
    acquisition_time: Optional[str] = None   # ISO timestamp, required for bi-temporal
    crs: Optional[str] = None                # coordinate reference system
    bounds: Optional[tuple] = None           # georeferenced extent, for co-registration checks
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class QueryRequest:
    query: str
    images: List[ImageInput]
    mode: InputMode


@dataclass
class ExecutionStep:
    name: str
    detail: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolResult:
    answer: str
    confidence: float
    visual_evidence_paths: List[str] = field(default_factory=list)
    structured_output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionReport:
    request_id: str
    query: str
    mode: InputMode
    task: TaskType
    tools_used: List[str]
    parameters: Dict[str, Any]
    steps: List[ExecutionStep]
    result: Optional[ToolResult]
    ok: bool
    error: Optional[str] = None


class SpecialistTool:
    """Interface every real tool wrapper (tools/*.py) must implement."""

    name: str = "unnamed-tool"
    required_mode: InputMode = InputMode.SINGLE

    def run(self, images: List[ImageInput], query: str, params: Dict[str, Any]) -> ToolResult:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Task classification — replace/extend with an actual intent classifier
# (e.g. a small fine-tuned text model or an LLM function-calling call) once
# available; the regex rules below are a transparent, auditable baseline.
# --------------------------------------------------------------------------

_CHANGE_COMPARATIVE = re.compile(r"increase|decrease|remain|grown|shrunk|more or less", re.I)
_GROUNDING_CUES = re.compile(r"highlight|locate|show me|point out|where is", re.I)
_QUESTION_CUES = re.compile(r"^(what|how|is there|are there|does|has|how much|how many)", re.I)


def classify_task(query: str, mode: InputMode) -> TaskType:
    q = query.strip()
    if mode == InputMode.BI_TEMPORAL:
        return TaskType.CHANGE_VQA if _CHANGE_COMPARATIVE.search(q) else TaskType.CHANGE_DESCRIBE
    if mode == InputMode.CROSS_MODAL:
        return TaskType.FUSION
    if _GROUNDING_CUES.search(q):
        return TaskType.GROUNDING
    if _QUESTION_CUES.search(q):
        return TaskType.VQA
    return TaskType.CAPTION


# --------------------------------------------------------------------------
# Input compatibility checking
# --------------------------------------------------------------------------

REQUIRED_IMAGE_COUNT = {
    InputMode.SINGLE: 1,
    InputMode.CROSS_MODAL: 2,
    InputMode.BI_TEMPORAL: 2,
}

SUPPORTED_FORMATS = {"GeoTIFF", "TIFF", "PNG", "JPEG"}


def check_compatibility(request: QueryRequest) -> ExecutionStep:
    required = REQUIRED_IMAGE_COUNT[request.mode]
    have = len(request.images)
    if have < required:
        raise ValueError(f"{request.mode.value} mode requires {required} image(s); {have} provided.")

    for img in request.images:
        if img.format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{img.format}' for {img.path}.")

    if request.mode == InputMode.CROSS_MODAL:
        modalities = {img.modality for img in request.images}
        if modalities != {Modality.OPTICAL, Modality.SAR}:
            raise ValueError("Cross-modal mode requires exactly one optical and one SAR image.")
        # In production: also verify request.images share CRS + overlapping bounds
        # (co-registration check) before proceeding.

    if request.mode == InputMode.BI_TEMPORAL:
        if any(img.acquisition_time is None for img in request.images):
            raise ValueError("Bi-temporal mode requires acquisition_time on both images.")
        times = sorted(request.images, key=lambda i: i.acquisition_time)
        if times[0].acquisition_time == times[1].acquisition_time:
            raise ValueError("Bi-temporal images must have distinct acquisition times.")

    return ExecutionStep("INPUT CHECK", f"{have}/{required} image(s) present; format & modality validated.")


# --------------------------------------------------------------------------
# Tool registry — real deployments register model-backed SpecialistTool
# instances here (see tools/). The names below are illustrative.
# --------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[TaskType, SpecialistTool] = {}

    def register(self, task: TaskType, tool: SpecialistTool) -> None:
        self._tools[task] = tool

    def get(self, task: TaskType) -> SpecialistTool:
        if task not in self._tools:
            raise KeyError(f"No tool registered for task '{task.value}'.")
        return self._tools[task]


class AgenticController:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def handle(self, request: QueryRequest) -> ExecutionReport:
        request_id = str(uuid.uuid4())
        steps: List[ExecutionStep] = []
        steps.append(ExecutionStep("QUERY RECEIVED", request.query))

        task = classify_task(request.query, request.mode)
        steps.append(ExecutionStep("TASK CLASSIFIED", task.value))

        try:
            steps.append(check_compatibility(request))
        except ValueError as e:
            steps.append(ExecutionStep("HALT", str(e)))
            return ExecutionReport(
                request_id=request_id, query=request.query, mode=request.mode,
                task=task, tools_used=[], parameters={}, steps=steps,
                result=None, ok=False, error=str(e),
            )

        tool = self.registry.get(task)
        steps.append(ExecutionStep("TOOL SELECTED", tool.name))

        params = self._permitted_params(task, request)
        steps.append(ExecutionStep("PARAMETERS", str(params)))

        try:
            result = tool.run(request.images, request.query, params)
        except Exception as e:  # noqa: BLE001 — surfaced in the audit trail on purpose
            steps.append(ExecutionStep("ERROR", str(e)))
            return ExecutionReport(
                request_id=request_id, query=request.query, mode=request.mode,
                task=task, tools_used=[tool.name], parameters=params, steps=steps,
                result=None, ok=False, error=str(e),
            )

        steps.append(ExecutionStep("EXECUTION COMPLETE", f"confidence={result.confidence:.2f}"))

        return ExecutionReport(
            request_id=request_id, query=request.query, mode=request.mode,
            task=task, tools_used=[tool.name], parameters=params, steps=steps,
            result=result, ok=True,
        )

    @staticmethod
    def _permitted_params(task: TaskType, request: QueryRequest) -> Dict[str, Any]:
        """
        Only pass through parameters the task is allowed to use — this is what
        the brief calls "configure only permitted task parameters". Extend per
        tool as real models expose real options (e.g. change threshold, grounding
        top-k boxes, confidence calibration temperature).
        """
        base = {"mode": request.mode.value}
        if task in (TaskType.CHANGE_DESCRIBE, TaskType.CHANGE_VQA):
            base["change_threshold"] = 0.3
        if task == TaskType.GROUNDING:
            base["max_regions"] = 3
        return base
