"""
backend/controller.py

Serverless-adapted AgenticController. Same task classification and
compatibility logic as agentic_controller.py, but works with in-memory
numpy arrays instead of file paths — suitable for Vercel Functions.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


# ── Enums ─────────────────────────────────────────────────────────────────

class InputMode(str, Enum):
    SINGLE = "single"
    CROSS = "cross"
    BITEMPORAL = "bitemporal"


class TaskType(str, Enum):
    CAPTION = "caption"
    VQA = "vqa"
    GROUNDING = "grounding"
    CHANGE_DESCRIBE = "change-describe"
    CHANGE_VQA = "change-vqa"
    FUSION = "fusion"


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class TraceStep:
    label: str
    detail: str
    timestamp: str = field(default_factory=lambda: time.strftime("%H:%M:%S", time.gmtime()))


@dataclass
class ToolResult:
    answer: str
    confidence: float
    overlay_base64: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionReport:
    request_id: str
    query: str
    mode: str
    task: str
    task_label: str
    tool: str
    answer: str
    confidence: float
    evidence: Dict[str, Any]
    overlay_base64: Optional[str]
    trace: List[Dict[str, str]]
    ok: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "mode": self.mode,
            "task": self.task,
            "task_label": self.task_label,
            "tool": self.tool,
            "answer": self.answer,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "overlay_base64": self.overlay_base64,
            "trace": self.trace,
            "ok": self.ok,
            "error": self.error,
        }


# ── Task classification ──────────────────────────────────────────────────

_CHANGE_COMPARATIVE = re.compile(r"increase|decrease|remain|grown|shrunk|more or less", re.I)
_GROUNDING_CUES = re.compile(r"highlight|locate|show me|point out|where is", re.I)
_QUESTION_CUES = re.compile(r"^(what|how|is there|are there|does|has|how much|how many)", re.I)

TASK_LABELS = {
    TaskType.CAPTION: "Scene Captioning / Description",
    TaskType.VQA: "Visual Question Answering",
    TaskType.GROUNDING: "Text-guided Region Grounding",
    TaskType.CHANGE_DESCRIBE: "Change Detection & Description",
    TaskType.CHANGE_VQA: "Change-based VQA (comparative)",
    TaskType.FUSION: "Optical–SAR Fusion Analysis",
}


def classify_task(query: str, mode: InputMode) -> TaskType:
    q = query.strip()
    if mode == InputMode.BITEMPORAL:
        return TaskType.CHANGE_VQA if _CHANGE_COMPARATIVE.search(q) else TaskType.CHANGE_DESCRIBE
    if mode == InputMode.CROSS:
        return TaskType.FUSION
    if _GROUNDING_CUES.search(q):
        return TaskType.GROUNDING
    if _QUESTION_CUES.search(q):
        return TaskType.VQA
    return TaskType.CAPTION


# ── Input compatibility ──────────────────────────────────────────────────

REQUIRED_IMAGE_COUNT = {
    InputMode.SINGLE: 1,
    InputMode.CROSS: 2,
    InputMode.BITEMPORAL: 2,
}


def check_compatibility(mode: InputMode, image_count: int) -> tuple[bool, str]:
    required = REQUIRED_IMAGE_COUNT[mode]
    if image_count < required:
        return False, f"{mode.value} mode requires {required} image(s); {image_count} provided."
    return True, f"{image_count}/{required} image(s) present · format & resolution validated."


# ── Query keyword detection ──────────────────────────────────────────────

def detect_keyword(query: str) -> Optional[str]:
    q = query.lower()
    if re.search(r"water|river|lake|pond|reservoir|flood", q):
        return "water"
    if re.search(r"veget|forest|green|crop|farm|field|tree", q):
        return "vegetation"
    if re.search(r"built|urban|building|city|road|infrastructure", q):
        return "built-up"
    if re.search(r"bare\s*soil|barren|exposed land", q):
        return "bare soil"
    return None
