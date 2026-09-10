# SatQuery AI — implementation notes

This folder plus `satquery-ai-prototype.html` together are one build, split
across what's actually runnable in a chat sandbox versus what needs a real
GPU/data environment. Read this before presenting either piece as "done."

## What's real and working right now

**`satquery-ai-prototype.html`** is a complete, self-contained, runnable web
app (open it in any browser — no server needed). It implements the full
*agentic pipeline* end to end:

- Mode-aware input intake (single image / co-registered optical+SAR pair /
  bi-temporal pair), with client-side format checking.
- A query classifier that routes natural-language questions to the correct
  task (VQA, captioning, grounding, change detection/VQA, optical-SAR
  fusion) — the same rules as `agentic_controller.classify_task`.
- Input-compatibility checking before execution.
- A tool registry that executes the selected specialist function.
- All five mandatory functional-scope items produce a real answer plus a
  rendered visual-evidence overlay: land-cover captioning, VQA, text-guided
  region grounding, bi-temporal change detection/change-VQA, and optical-SAR
  fused classification.
- A confidence score, a live execution trace (the auditable summary the
  brief asks for), and a downloadable JSON report.

**The "specialist models" in that demo are classical computer-vision
heuristics** — RGB thresholding for land-cover classes, pixel-differencing
for change, and a Sobel edge/intensity proxy for SAR structure — not
fine-tuned vision-language models. They run instantly, in-browser, on
PNG/JPEG only.

## What's a scaffold, not a finished system

The problem statement's mandatory requirement — *"At least one visual or
vision-language component must be fine-tuned or otherwise adapted using
BigEarthNet"* — needs GPU training on real satellite imagery. That can't
happen inside a chat sandbox (no GPU, no access to dataset/model hosts).
So instead of faking it, this folder gives you the real integration
surface:

- `agentic_controller.py` — the orchestration/routing logic, framework-
  agnostic, with the exact same task-classification and compatibility-
  checking rules as the browser demo, but built around a `SpecialistTool`
  interface real models implement.
- `tools/vqa_tool.py` — a worked example of that interface: swap
  `RemoteSensingVQAModel.predict` for a real forward pass through a
  BigEarthNet/RSVQA-adapted checkpoint. Copy this pattern for the other four
  tools (`caption_ground_tool.py`, `change_tool.py`, `fusion_tool.py` —
  stubs, not yet written).
- `training/finetune_bigearthnet.py` — a CLIP-style contrastive-adaptation
  recipe skeleton (BigEarthNet Sentinel-1/2 patches + text -> RS-adapted
  encoder). Every function raises `NotImplementedError` with a comment on
  what real code goes there; run it on your own GPU machine, not here.

## Suggested path to full compliance

1. Get BigEarthNet locally (dataset link in the original problem statement)
   and a GPU box (local, Colab Pro, or a cloud instance).
2. Fill in `training/finetune_bigearthnet.py`, fine-tune an OpenCLIP (or
   similar) backbone on BigEarthNet image-text pairs.
3. Build task heads on top of that encoder:
   - VQA/captioning: a lightweight decoder (e.g. BLIP-style) fine-tuned on
     VRSBench/RSVQA.
   - Grounding: add a detection/segmentation head, fine-tuned on VRSBench's
     grounding split.
   - Change: a Siamese variant of the encoder + a change-VQA head, fine-tuned
     on CDVQA.
   - Fusion: a second encoder branch for SAR, fused with the optical branch
     (e.g. cross-attention) before the task heads.
4. Wrap each fine-tuned model in a `SpecialistTool` per `tools/vqa_tool.py`'s
   pattern, register it in `agentic_controller.ToolRegistry`.
5. Put a FastAPI layer in front of `AgenticController.handle(...)` and point
   a GUI at it — you can reuse `satquery-ai-prototype.html`'s frontend
   layout/JS structure and swap its client-side heuristics for API calls.
6. Evaluate on the prescribed VRSBench / RSVQA / CDVQA test splits, then on
   the ISRO/SAC Cartosat-2S + RISAT evaluation set once released.

## File layout

```
satquery-ai-prototype.html     <- runnable demo, open directly in a browser
backend/
  agentic_controller.py        <- orchestration core (real, framework-agnostic)
  tools/
    __init__.py
    vqa_tool.py                 <- worked SpecialistTool example
  training/
    finetune_bigearthnet.py    <- fine-tuning recipe skeleton
  requirements.txt
  README.md                    <- this file
```
