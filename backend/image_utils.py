"""
backend/image_utils.py

Shared image processing utilities for the SatQuery AI serverless backend.
Ports the client-side heuristic CV logic (pixel classification, landcover
stats, overlay rendering, change detection, SAR structure analysis) from
JavaScript to Python using numpy + Pillow.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ── Colour constants (matching the frontend overlay colours) ──────────────

CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "water": (94, 200, 216),
    "vegetation": (79, 209, 165),
    "built-up": (232, 163, 61),
    "bare soil": (192, 138, 90),
}

CHANGE_COLOR: Tuple[int, int, int] = (226, 87, 76)

CLASS_LABELS: Dict[str, str] = {
    "water": "water bodies",
    "vegetation": "vegetation",
    "built-up": "built-up / impervious surfaces",
    "bare soil": "bare soil",
    "other": "unclassified surface",
}


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class LandcoverStats:
    counts: Dict[str, int]
    total: int
    percentages: Dict[str, float]
    class_map: np.ndarray          # 1-D array of class indices per pixel
    width: int
    height: int

    # Class index → label mapping used by class_map
    INDEX_TO_LABEL: List[str] = field(
        default_factory=lambda: ["water", "vegetation", "built-up", "bare soil", "other"]
    )


@dataclass
class ChangeMaskResult:
    mask: np.ndarray               # uint8, 1-D
    changed_pct: float
    grid: List[int]                # 9-cell grid counts
    width: int
    height: int


@dataclass
class SARStructure:
    edge: np.ndarray               # uint8, 1-D
    dark: np.ndarray               # uint8, 1-D
    edge_pct: float
    dark_pct: float
    width: int
    height: int


@dataclass
class FusionResult:
    water_mask: np.ndarray
    built_mask: np.ndarray
    water_pct: float
    built_pct: float
    agreement_pct: float


# ── Image I/O ─────────────────────────────────────────────────────────────

WORK_SIZE = 320
SINGLE_MAX = 380


def decode_base64_image(data_url: str) -> Image.Image:
    """Decode a data-URL (data:image/png;base64,...) or raw base64 string."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def resize_for_processing(img: Image.Image, mode: str) -> Image.Image:
    """Resize to working resolution, matching the frontend logic."""
    if mode == "single":
        scale = min(1.0, SINGLE_MAX / max(img.width, img.height))
        w = max(1, round(img.width * scale))
        h = max(1, round(img.height * scale))
    else:
        w, h = WORK_SIZE, WORK_SIZE
    return img.resize((w, h), Image.LANCZOS)


def image_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a (H, W, 3) uint8 numpy array."""
    return np.asarray(img, dtype=np.uint8)


def numpy_to_base64(arr: np.ndarray) -> str:
    """Convert a (H, W, 3) uint8 array to a data-URL PNG string."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ── Pixel-level landcover classification ──────────────────────────────────

def classify_pixel(r: int, g: int, b: int) -> int:
    """
    Classify a single pixel into one of 5 landcover classes.
    Returns an integer index: 0=water, 1=vegetation, 2=built-up, 3=bare soil, 4=other.
    Same heuristic thresholds as the JS frontend.
    """
    brightness = (int(r) + int(g) + int(b)) / 3.0
    if b > r + 12 and b >= g - 5 and brightness < 195:
        return 0  # water
    if g > r + 12 and g > b + 8:
        return 1  # vegetation
    mx = max(r, g, b)
    mn = min(r, g, b)
    sat = 0.0 if mx == 0 else (mx - mn) / mx
    if sat < 0.15 and brightness > 90:
        return 2  # built-up
    if r > b + 15 and r >= g - 5:
        return 3  # bare soil
    return 4  # other


# Vectorised version for performance on full images
def classify_image(pixels: np.ndarray) -> np.ndarray:
    """
    Classify all pixels in a (H, W, 3) uint8 array.
    Returns a (H*W,) int array of class indices.
    """
    flat = pixels.reshape(-1, 3).astype(np.int16)
    r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]
    brightness = (r + g + b) / 3.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.where(mx == 0, 0.0, (mx - mn) / mx)

    result = np.full(len(flat), 4, dtype=np.int32)  # default: other

    # Order matters — later rules override earlier ones (same as JS)
    bare = (r > b + 15) & (r >= g - 5)
    result[bare] = 3

    built = (sat < 0.15) & (brightness > 90)
    result[built] = 2

    veg = (g > r + 12) & (g > b + 8)
    result[veg] = 1

    water = (b > r + 12) & (b >= g - 5) & (brightness < 195)
    result[water] = 0

    return result


def compute_landcover_stats(pixels: np.ndarray) -> LandcoverStats:
    """Compute landcover statistics for a (H, W, 3) image."""
    h, w = pixels.shape[:2]
    n = h * w
    class_map = classify_image(pixels)
    labels = ["water", "vegetation", "built-up", "bare soil", "other"]
    counts = {labels[i]: int(np.sum(class_map == i)) for i in range(5)}
    percentages = {k: (v / n) * 100 for k, v in counts.items()}
    return LandcoverStats(
        counts=counts, total=n, percentages=percentages,
        class_map=class_map, width=w, height=h,
    )


def top_class(stats: LandcoverStats) -> str:
    """Return the dominant landcover class (excluding 'other')."""
    candidates = ["water", "vegetation", "built-up", "bare soil"]
    return max(candidates, key=lambda c: stats.percentages[c])


# ── Overlay rendering ─────────────────────────────────────────────────────

def build_highlight_overlay(
    pixels: np.ndarray,
    class_map: np.ndarray,
    target_class: str,
    draw_bbox: bool = False,
) -> np.ndarray:
    """
    Blend a colour overlay onto pixels where class_map matches target_class.
    Returns a (H, W, 3) uint8 array.
    """
    labels = ["water", "vegetation", "built-up", "bare soil", "other"]
    target_idx = labels.index(target_class) if target_class in labels else 4
    h, w = pixels.shape[:2]
    out = pixels.astype(np.float32).copy()
    flat_out = out.reshape(-1, 3)

    color = np.array(CLASS_COLORS.get(target_class, (255, 255, 255)), dtype=np.float32)
    mask = class_map == target_idx
    flat_out[mask] = flat_out[mask] * 0.4 + color * 0.6

    out = flat_out.reshape(h, w, 3).astype(np.uint8)

    if draw_bbox and np.sum(mask) > 25:
        ys, xs = np.where(mask.reshape(h, w))
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        c = tuple(int(v) for v in color)
        # Draw rectangle
        out[min_y, min_x:max_x+1] = c
        out[max_y, min_x:max_x+1] = c
        out[min_y:max_y+1, min_x] = c
        out[min_y:max_y+1, max_x] = c

    return out


def render_change_overlay(pixels_b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend change-colour onto Time-B image where mask is 1."""
    h, w = pixels_b.shape[:2]
    out = pixels_b.astype(np.float32).copy()
    flat_out = out.reshape(-1, 3)
    color = np.array(CHANGE_COLOR, dtype=np.float32)
    changed = mask.astype(bool)
    flat_out[changed] = flat_out[changed] * 0.35 + color * 0.65
    return flat_out.reshape(h, w, 3).astype(np.uint8)


def render_fusion_overlay(
    pixels_optical: np.ndarray,
    water_mask: np.ndarray,
    built_mask: np.ndarray,
) -> np.ndarray:
    """Blend water + built-up colours onto optical image based on fused masks."""
    h, w = pixels_optical.shape[:2]
    out = pixels_optical.astype(np.float32).copy()
    flat_out = out.reshape(-1, 3)
    water_c = np.array(CLASS_COLORS["water"], dtype=np.float32)
    built_c = np.array(CLASS_COLORS["built-up"], dtype=np.float32)
    wm = water_mask.astype(bool)
    bm = built_mask.astype(bool) & ~wm  # built-up only where not water
    flat_out[wm] = flat_out[wm] * 0.35 + water_c * 0.65
    flat_out[bm] = flat_out[bm] * 0.35 + built_c * 0.65
    return flat_out.reshape(h, w, 3).astype(np.uint8)


# ── Change detection ──────────────────────────────────────────────────────

def compute_change_mask(pixels_a: np.ndarray, pixels_b: np.ndarray) -> ChangeMaskResult:
    """Compute per-pixel change mask between two co-registered images."""
    h, w = pixels_a.shape[:2]
    n = h * w
    threshold = 28

    gray_a = pixels_a.astype(np.float32).reshape(-1, 3).mean(axis=1)
    gray_b = pixels_b.astype(np.float32).reshape(-1, 3).mean(axis=1)
    diff = np.abs(gray_a - gray_b)
    mask = (diff > threshold).astype(np.uint8)
    changed = int(mask.sum())

    # 3x3 spatial grid counts
    grid = [0] * 9
    if changed > 0:
        coords = np.argwhere(mask.reshape(h, w))
        for y, x in coords:
            gx = min(2, int(x / (w / 3)))
            gy = min(2, int(y / (h / 3)))
            grid[gy * 3 + gx] += 1

    return ChangeMaskResult(
        mask=mask, changed_pct=(changed / n) * 100,
        grid=grid, width=w, height=h,
    )


def describe_location(diff: ChangeMaskResult) -> str:
    """Describe where the change is concentrated."""
    if diff.changed_pct < 1:
        return "no single concentrated region"
    labels = [
        "top-left", "top-center", "top-right",
        "middle-left", "center", "middle-right",
        "bottom-left", "bottom-center", "bottom-right",
    ]
    max_idx = max(range(9), key=lambda i: diff.grid[i])
    return labels[max_idx] + " of the frame"


# ── SAR structure analysis (Sobel) ────────────────────────────────────────

def compute_sar_structure(pixels: np.ndarray) -> SARStructure:
    """Compute edge map and dark regions via Sobel filter on grayscale SAR image."""
    h, w = pixels.shape[:2]
    n = h * w
    gray = pixels.astype(np.float32).mean(axis=2)

    # Sobel kernels
    edge = np.zeros((h, w), dtype=np.uint8)
    dark = np.zeros((h, w), dtype=np.uint8)
    edge_count = 0
    dark_count = 0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            region = gray[y-1:y+2, x-1:x+2]
            gx = (
                -region[0, 0] + region[0, 2]
                - 2*region[1, 0] + 2*region[1, 2]
                - region[2, 0] + region[2, 2]
            )
            gy = (
                -region[0, 0] - 2*region[0, 1] - region[0, 2]
                + region[2, 0] + 2*region[2, 1] + region[2, 2]
            )
            mag = math.sqrt(gx * gx + gy * gy)
            if mag > 60:
                edge[y, x] = 1
                edge_count += 1
            if gray[y, x] < 70 and mag < 25:
                dark[y, x] = 1
                dark_count += 1

    return SARStructure(
        edge=edge.ravel(), dark=dark.ravel(),
        edge_pct=(edge_count / n) * 100,
        dark_pct=(dark_count / n) * 100,
        width=w, height=h,
    )


def fuse_classes(
    optical_stats: LandcoverStats,
    sar_struct: SARStructure,
) -> FusionResult:
    """Cross-modal fusion of optical landcover classes with SAR structure."""
    n = len(optical_stats.class_map)
    labels = optical_stats.INDEX_TO_LABEL

    optical_water = optical_stats.class_map == labels.index("water")
    optical_built = (optical_stats.class_map == labels.index("built-up")) | \
                    (optical_stats.class_map == labels.index("bare soil"))
    sar_dark = sar_struct.dark.astype(bool)
    sar_edge = sar_struct.edge.astype(bool)

    water_mask = (optical_water & sar_dark).astype(np.uint8)
    built_mask = (optical_built & sar_edge).astype(np.uint8)

    water_count = int(water_mask.sum())
    built_count = int(built_mask.sum())

    # Agreement calculation
    water_relevant = int((optical_water | sar_dark).sum())
    water_agree = int((optical_water & sar_dark).sum())
    built_relevant = int((optical_built | sar_edge).sum())
    built_agree = int((optical_built & sar_edge).sum())

    w_ratio = water_agree / water_relevant if water_relevant else 0
    b_ratio = built_agree / built_relevant if built_relevant else 0
    agreement_pct = (w_ratio + b_ratio) / 2 * 100

    return FusionResult(
        water_mask=water_mask, built_mask=built_mask,
        water_pct=(water_count / n) * 100,
        built_pct=(built_count / n) * 100,
        agreement_pct=agreement_pct,
    )
