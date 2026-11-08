"""
backend/imgops.py

Small, dependency-free (numpy + Pillow) image operations shared by the
diagnostics, evidence and uncertainty engines. Everything here is
vectorised and deterministic.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Tuple

import numpy as np
from PIL import Image


def to_gray(pixels: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (H, W) float32 gray = mean of R, G, B.

    The mean (not a perceptual luma) is used deliberately: it is what the
    v0 change-detection threshold was defined against, and it keeps the
    documented thresholds comparable between releases."""
    if pixels.ndim == 2:
        return pixels.astype(np.float32)
    return pixels.astype(np.float32).mean(axis=2)


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """3x3 Sobel gradient magnitude. Border pixels are 0 (same convention as
    the original per-pixel loop this replaces)."""
    h, w = gray.shape
    mag = np.zeros((h, w), dtype=np.float32)
    if h < 3 or w < 3:
        return mag
    g = gray.astype(np.float32)
    gx = (
        -g[:-2, :-2] + g[:-2, 2:]
        - 2 * g[1:-1, :-2] + 2 * g[1:-1, 2:]
        - g[2:, :-2] + g[2:, 2:]
    )
    gy = (
        -g[:-2, :-2] - 2 * g[:-2, 1:-1] - g[:-2, 2:]
        + g[2:, :-2] + 2 * g[2:, 1:-1] + g[2:, 2:]
    )
    mag[1:-1, 1:-1] = np.hypot(gx, gy)
    return mag


def laplacian(gray: np.ndarray) -> np.ndarray:
    """4-neighbour Laplacian over the interior, shape (H-2, W-2)."""
    g = gray.astype(np.float32)
    return (
        g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    )


def laplacian_variance(gray: np.ndarray) -> float:
    if min(gray.shape) < 3:
        return 0.0
    return float(laplacian(gray).var())


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer (1996) fast noise-standard-deviation estimate.

    It assumes additive white noise and is biased upwards on strongly
    textured scenes, so it is reported as an *estimate*."""
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    g = gray.astype(np.float64)
    n = (
        g[:-2, :-2] - 2 * g[:-2, 1:-1] + g[:-2, 2:]
        - 2 * g[1:-1, :-2] + 4 * g[1:-1, 1:-1] - 2 * g[1:-1, 2:]
        + g[2:, :-2] - 2 * g[2:, 1:-1] + g[2:, 2:]
    )
    return float(math.sqrt(math.pi / 2.0) / (6.0 * (w - 2) * (h - 2)) * np.abs(n).sum())


def neighbour_count(mask: np.ndarray) -> np.ndarray:
    """Number of True 8-neighbours for every pixel of a boolean mask."""
    m = mask.astype(np.uint8)
    p = np.pad(m, 1)
    h, w = m.shape
    total = np.zeros((h, w), dtype=np.uint8)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            total += p[dy:dy + h, dx:dx + w]
    return total


def cohesion(mask: np.ndarray) -> float:
    """Fraction of True pixels that have at least 3 True 8-neighbours.

    Compact blobs score near 1; salt-and-pepper noise scores near 0."""
    n = int(mask.sum())
    if n == 0:
        return 0.0
    counts = neighbour_count(mask)
    return float(((counts >= 3) & mask).sum() / n)


# ── Colour mapping & encoding ────────────────────────────────────────────

_DIFF_STOPS = np.array(
    [
        [0.00, 10, 10, 28],
        [0.25, 74, 22, 112],
        [0.50, 182, 54, 92],
        [0.75, 246, 142, 42],
        [1.00, 252, 240, 164],
    ],
    dtype=np.float32,
)


def colormap_difference(diff: np.ndarray, full_scale: float = 128.0) -> np.ndarray:
    """Map an absolute-difference array (gray levels) to an RGB heat map.

    A *fixed* full scale is used (not per-image normalisation) so two
    analyses can be compared visually."""
    t = np.clip(diff / full_scale, 0.0, 1.0)
    rgb = np.stack(
        [np.interp(t, _DIFF_STOPS[:, 0], _DIFF_STOPS[:, c]) for c in (1, 2, 3)], axis=-1
    )
    return rgb.astype(np.uint8)


def blend(base: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    """Alpha-blend `color` over `base` where `mask` is True. Returns a copy."""
    out = base.astype(np.float32).copy()
    c = np.array(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + c * alpha
    return out.astype(np.uint8)


def desaturate(pixels: np.ndarray, amount: float = 0.6) -> np.ndarray:
    gray = to_gray(pixels)[..., None]
    out = pixels.astype(np.float32) * (1.0 - amount) + gray * amount
    return out.astype(np.uint8)


def encode_png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=6)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
