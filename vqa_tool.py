"""
tools/vqa_tool.py

Reference implementation of a SpecialistTool wrapper for single-image
remote-sensing VQA. This is the pattern to copy for the other four
mandatory tools (captioning/grounding, change-VQA, optical-SAR fusion).

Swap `RemoteSensingVQAModel` for a real fine-tuned checkpoint — e.g. a
CLIP/BLIP-style vision-language model adapted on BigEarthNet image-text
pairs, or an open remote-sensing VLM such as GeoChat/RSGPT-style
architectures, further fine-tuned for VQA on RSVQA. None of that
fine-tuning can run inside a sandboxed chat environment (needs GPU
compute + dataset access) — this file just fixes the *contract* the
controller expects, so training and integration are decoupled.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agentic_controller import ImageInput, SpecialistTool, ToolResult, InputMode


class RemoteSensingVQAModel:
    """
    Thin wrapper around a real model checkpoint. Replace the body of
    `predict` with an actual forward pass, e.g.:

        from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
        processor = AutoProcessor.from_pretrained("<your-bigearthnet-adapted-checkpoint>")
        model = AutoModelForVisualQuestionAnswering.from_pretrained("<checkpoint>")

        def predict(self, image_path, question):
            image = load_raster(image_path)  # GDAL/rasterio -> PIL/np array
            inputs = processor(images=image, text=question, return_tensors="pt")
            out = model.generate(**inputs)
            answer = processor.decode(out[0], skip_special_tokens=True)
            confidence = softmax_confidence(out)  # e.g. mean token logprob -> [0,1]
            return answer, confidence
    """

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        # self.model = load_model(checkpoint_path)  # TODO: real load

    def predict(self, image_path: str, question: str) -> tuple[str, float]:
        raise NotImplementedError(
            "Plug in a real BigEarthNet/RSVQA-adapted VQA model here. "
            "See training/finetune_bigearthnet.py for the adaptation recipe."
        )


class VQATool(SpecialistTool):
    name = "RS-VQA-v1"
    required_mode = InputMode.SINGLE

    def __init__(self, model: RemoteSensingVQAModel):
        self.model = model

    def run(self, images: List[ImageInput], query: str, params: Dict[str, Any]) -> ToolResult:
        image = images[0]
        answer, confidence = self.model.predict(image.path, query)
        return ToolResult(
            answer=answer,
            confidence=confidence,
            visual_evidence_paths=[],       # attach attention/heatmap overlay if the model exposes one
            structured_output={"question": query, "image": image.path},
        )
