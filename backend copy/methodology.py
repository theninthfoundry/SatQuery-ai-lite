"""
backend/methodology.py

Plain-language description of HOW each result is produced and WHAT could
make it wrong. Returned with every analysis so the interface never has to
write (or invent) methodology text of its own.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend import config

GLOBAL_LIMITATIONS = [
    "All analysis is classical image processing on 8-bit renderings. No trained remote-sensing model is used and none is claimed.",
    "Results are heuristic estimates measured in image space. Without georeferencing, nothing here is a physical (ground) measurement.",
    "Radiometric calibration of the inputs is unknown; brightness differences may reflect exposure, atmosphere or season rather than the surface.",
]

_COLOUR_HEURISTIC = [
    "Land-cover classes come from fixed RGB colour rules (water: blue-dominant and not bright; vegetation: green-dominant; built-up: low saturation and bright; bare soil: red-dominant). They confuse shadows with water, grey bare ground with built-up, and are not valid for grayscale/SAR input.",
    "Percentages are shares of pixels on the working-resolution image, not of ground area.",
]

_TASKS: Dict[str, Dict[str, Any]] = {
    "caption": {
        "name": "Scene composition summary",
        "steps": ["Decode and resize to the working resolution", "Classify every pixel with the RGB colour rules", "Rank classes by pixel share and describe the dominant and secondary classes"],
        "limitations": _COLOUR_HEURISTIC + ["The description names only the four colour classes; it does not identify objects."],
    },
    "vqa": {
        "name": "Keyword-grounded visual question answering",
        "steps": ["Detect the target class named in the question by keyword", "Classify pixels with the RGB colour rules", "Answer from the measured pixel shares (percentage / presence / composition)"],
        "limitations": _COLOUR_HEURISTIC + ["The question is understood by keyword matching, not language understanding; unnamed targets fall back to the composition summary."],
    },
    "grounding": {
        "name": "Colour-threshold region grounding",
        "steps": ["Detect the target class named in the query (or fall back to the dominant class, and say so)", "Classify pixels with the RGB colour rules", "Highlight matching pixels and draw the box spanning all of them"],
        "limitations": _COLOUR_HEURISTIC + ["The bounding box spans every matching pixel, including isolated ones, so it can be much larger than the main region."],
    },
    "change-describe": {
        "name": "Change Intelligence Engine: pixel differencing with evidence and sensitivity analysis",
        "steps": [
            "Decode both images and place them on a common grid derived from image A",
            "Run image-quality diagnostics on each image",
            "Estimate translation between the images by phase correlation (diagnostic only)",
            "Compute |A-B| on gray = mean(R,G,B) and mark pixels above the base threshold",
            "Divide the frame into a grid and measure changed pixels, intensity, local quality and evidence strength per cell",
            "Repeat the comparison at several thresholds; separate stable change from threshold-sensitive (ambiguous) change",
            "Check the threshold against the estimated noise floor and check for a global brightness offset",
            "Generate statements from the measurements and derive the evidence level from the diagnostics",
        ],
        "limitations": [
            "Pixel differencing reports ANY intensity difference. It cannot distinguish real surface change from illumination, season, shadows, clouds, haze or sensor differences.",
            "The alignment diagnostic estimates visual translation only. It is not geospatial registration, and rotation, scale or parallax differences are not estimated.",
            "The reported change area is not shift-corrected or offset-corrected; the what-if values are diagnostics.",
            "Grid statistics depend on the grid size; per-cell percentages are relative to the cell, not the frame.",
            "Threshold stability means robustness to the threshold choice. It is not evidence that a change is real.",
        ],
    },
    "change-vqa": {
        "name": "Comparative change question over RGB class shares",
        "steps": [
            "Run the full change analysis (see change-describe)",
            "Classify each image independently with the RGB colour rules",
            "Compare the target class's pixel share between the dates against a fixed decision margin",
        ],
        "limitations": _COLOUR_HEURISTIC + [
            "Each date is classified independently, so illumination or seasonal colour shifts alter the class shares without any surface change.",
            "The increased/decreased/unchanged decision uses a fixed margin and is not statistically tested.",
        ],
    },
    "fusion": {
        "name": "Optical-SAR proxy fusion",
        "steps": [
            "Classify the optical image with the RGB colour rules",
            "Derive SAR-like structure from the second image: Sobel edges (structure) and dark smooth areas (possible water)",
            "Mark water where optical water and a dark smooth area agree; mark built-up where optical built-up/bare and an edge agree",
            "Report cross-modal agreement as the overlap ratio of the two proxies",
        ],
        "limitations": _COLOUR_HEURISTIC + [
            "The SAR branch is a visual proxy on an 8-bit image. It is not calibrated SAR processing: no sigma0, no speckle filtering, no polarisation, no geometry.",
            "Co-registration of the two images is assumed, not verified; cross-modal alignment cannot be assessed by correlation.",
            "Agreement is measured between two heuristics and is not a measure of correctness.",
        ],
    },
}


def methodology_for(task: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    t = _TASKS[task]
    return {
        "name": t["name"],
        "steps": t["steps"],
        "parameters": parameters,
        "gray_definition": config.PROCESSING.gray_definition,
        "confidence_policy": "No calibrated confidence is reported; see `reliability`.",
    }


def limitations_for(task: str) -> List[str]:
    return list(_TASKS[task]["limitations"]) + GLOBAL_LIMITATIONS
