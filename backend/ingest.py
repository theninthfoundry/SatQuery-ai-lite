"""
backend/ingest.py

Hardened image intake. Everything an upload can do to a serverless function
is bounded here BEFORE expensive processing: encoded size, decoded pixel
count, format allow-list, bit depth, frames, and malformed base64.

Raw bytes are hashed (SHA-256) so every analysis can be tied to the exact
inputs it consumed.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from backend import config
from backend.geospatial.metadata import TiffParseError, inspect_tiff, is_tiff
from backend.imgops import to_gray

Image.MAX_IMAGE_PIXELS = config.LIMITS.max_pixels * 2  # Pillow errors at 2x; we check explicitly first

ALLOWED_FORMATS = {"PNG", "JPEG", "TIFF"}
_DATA_URL = re.compile(r"^data:([A-Za-z0-9.+/-]+);base64,", re.I)


class InputError(Exception):
    """A rejected input. `code` is stable and machine-readable."""

    def __init__(self, code: str, message: str, status: int = 400, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details or {}


@dataclass
class DecodedImage:
    pixels: Image.Image                 # RGB at original resolution
    sha256: str
    nbytes: int
    fmt: str
    original_size: Tuple[int, int]      # (width, height)
    source_mode: str
    warnings: List[str] = field(default_factory=list)
    geospatial: Optional[Dict[str, Any]] = None
    geospatial_error: Optional[str] = None


def decode_data_url(data: str, limits: config.Limits = config.LIMITS) -> bytes:
    """Strictly decode a base64 payload (optionally a data URL) to bytes."""
    if not isinstance(data, str) or not data.strip():
        raise InputError("EMPTY_IMAGE", "An image payload is empty.")
    if len(data) > int(limits.max_image_bytes * 4 / 3) + 128:
        raise InputError(
            "IMAGE_TOO_LARGE",
            f"Image exceeds the {limits.max_image_bytes / 1e6:.1f} MB per-image limit.",
            status=413,
            details={"limit_bytes": limits.max_image_bytes},
        )
    m = _DATA_URL.match(data)
    body = data[m.end():] if m else data
    body = re.sub(r"\s+", "", body)
    try:
        raw = base64.b64decode(body, validate=True)
    except (binascii.Error, ValueError):
        raise InputError("INVALID_BASE64", "Image payload is not valid base64.")
    if not raw:
        raise InputError("EMPTY_IMAGE", "Decoded image has zero bytes.")
    if len(raw) > limits.max_image_bytes:
        raise InputError(
            "IMAGE_TOO_LARGE",
            f"Image is {len(raw) / 1e6:.2f} MB; the limit is {limits.max_image_bytes / 1e6:.1f} MB.",
            status=413,
            details={"limit_bytes": limits.max_image_bytes, "size_bytes": len(raw)},
        )
    return raw


def decode_image(data: str, limits: config.Limits = config.LIMITS) -> DecodedImage:
    raw = decode_data_url(data, limits)
    sha = hashlib.sha256(raw).hexdigest()

    geo: Optional[Dict[str, Any]] = None
    geo_err: Optional[str] = None
    if is_tiff(raw):
        try:
            geo = inspect_tiff(raw)
        except (TiffParseError, ValueError, OverflowError) as e:
            geo_err = f"TIFF metadata could not be read: {e}"

    if geo is not None and not geo["pixel_analysis_supported"]:
        # Pillow often cannot open multi-band / 16-bit rasters at all; the metadata is still valuable.
        if geo["width"] * geo["height"] > limits.max_pixels:
            raise InputError("IMAGE_TOO_MANY_PIXELS", "Raster exceeds the pixel limit.", status=413)
        raise InputError("UNSUPPORTED_RASTER", geo["pixel_analysis_unsupported_reason"], status=415, details={"geospatial": geo})

    try:
        img = Image.open(io.BytesIO(raw))
        fmt = (img.format or "").upper()
        if fmt not in ALLOWED_FORMATS:
            raise InputError(
                "UNSUPPORTED_FORMAT",
                f"Unsupported image format '{fmt or 'unknown'}'. Supported: PNG, JPEG, TIFF/GeoTIFF (8-bit).",
                status=415,
            )
        w, h = img.size
        if w * h > limits.max_pixels:
            raise InputError(
                "IMAGE_TOO_MANY_PIXELS",
                f"Image has {w * h / 1e6:.1f} megapixels; the limit is {limits.max_pixels / 1e6:.0f} MP.",
                status=413,
                details={"limit_pixels": limits.max_pixels},
            )
        if min(w, h) < limits.min_dimension:
            raise InputError(
                "IMAGE_TOO_SMALL",
                f"Image is {w}x{h}px; the shorter side must be at least {limits.min_dimension}px.",
            )
        mode = img.mode
        if mode in ("I", "F") or mode.startswith("I;"):
            raise InputError(
                "UNSUPPORTED_BIT_DEPTH",
                "16/32-bit rasters are not supported in Lite v1 (silently rescaling would alter radiometry).",
                status=415,
                details={"geospatial": geo},
            )
        if fmt == "TIFF" and geo is not None and not geo["pixel_analysis_supported"]:
            raise InputError(
                "UNSUPPORTED_RASTER", geo["pixel_analysis_unsupported_reason"], status=415, details={"geospatial": geo}
            )
        warnings: List[str] = []
        frames = getattr(img, "n_frames", 1)
        if frames and frames > 1:
            warnings.append(f"File has {frames} frames/pages; only the first is analysed.")
        img.load()
    except InputError:
        raise
    except Image.DecompressionBombError:
        raise InputError("IMAGE_TOO_MANY_PIXELS", "Image exceeds the decompression safety limit.", status=413)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError):
        raise InputError("CORRUPT_IMAGE", "The file could not be decoded as an image (corrupt or unsupported).")

    try:
        exif_orient = img.getexif().get(0x0112)
    except Exception:
        exif_orient = None
    if exif_orient and exif_orient != 1:
        img = ImageOps.exif_transpose(img)
        warnings.append(f"EXIF orientation {exif_orient} was applied before analysis.")
    if mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in img.info):
        warnings.append("Alpha/transparency channel discarded.")
    if mode == "CMYK":
        warnings.append("CMYK converted to RGB.")
    rgb = img.convert("RGB")

    return DecodedImage(
        pixels=rgb, sha256=sha, nbytes=len(raw), fmt=fmt, original_size=(w, h), source_mode=mode,
        warnings=warnings, geospatial=geo, geospatial_error=geo_err,
    )


def _fit(size: Tuple[int, int], max_side: int) -> Tuple[int, int]:
    w, h = size
    scale = min(1.0, max_side / max(w, h))
    return max(1, round(w * scale)), max(1, round(h * scale))


def prepare_single(img: DecodedImage, proc: config.Processing = config.PROCESSING) -> Tuple[np.ndarray, Dict[str, Any]]:
    size = _fit(img.original_size, proc.single_max_side)
    resized = size != img.original_size
    arr = np.asarray(img.pixels.resize(size, Image.LANCZOS) if resized else img.pixels, dtype=np.uint8)
    return arr, {"working_size": list(size), "resampled": resized, "resampling": "LANCZOS" if resized else None}


def prepare_pair(
    a: DecodedImage, b: DecodedImage, proc: config.Processing = config.PROCESSING
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Put both images on a common grid derived from image A.

    B is resampled onto A's working grid only if its size differs; if the
    original aspect ratios differ the resampling distorts B, which the
    alignment diagnostic reports."""
    size = _fit(a.original_size, proc.pair_max_side)
    a_re = size != a.original_size
    b_re = size != b.original_size
    arr_a = np.asarray(a.pixels.resize(size, Image.LANCZOS) if a_re else a.pixels, dtype=np.uint8)
    arr_b = np.asarray(b.pixels.resize(size, Image.LANCZOS) if b_re else b.pixels, dtype=np.uint8)
    return arr_a, arr_b, {
        "working_size": list(size),
        "resampled": [a_re, b_re],
        "resampling": "LANCZOS" if (a_re or b_re) else None,
        "grid_source": "image A",
    }
