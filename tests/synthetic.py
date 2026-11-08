"""Deterministic synthetic imagery for tests. No fixtures, no network, no files."""
from __future__ import annotations

import base64
import io
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


def scene(size: Tuple[int, int] = (256, 256), seed: int = 7) -> np.ndarray:
    """A textured, field-like RGB scene (uint8, HxWx3) with real edges."""
    w, h = size
    rng = np.random.default_rng(seed)
    base = rng.random((h // 8 + 2, w // 8 + 2, 3)).astype(np.float32)
    img = Image.fromarray((base * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) * 0.9 + 12
    # hard-edged parcels so there is structure to register against
    for _ in range(14):
        x0, y0 = int(rng.integers(0, w - 24)), int(rng.integers(0, h - 24))
        x1, y1 = x0 + int(rng.integers(16, w // 3)), y0 + int(rng.integers(16, h // 3))
        arr[y0:y1, x0:x1] = rng.integers(40, 220, 3)
    arr += rng.normal(0, 1.5, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def add_patch(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int, color=(235, 235, 235)) -> np.ndarray:
    out = arr.copy()
    out[y0:y1, x0:x1] = color
    return out


def shift(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Content moves by (dx, dy): out[y+dy, x+dx] == arr[y, x]."""
    return np.roll(arr, shift=(dy, dx), axis=(0, 1))


def blur(arr: np.ndarray, radius: float = 4.0) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius)))


def add_noise(arr: np.ndarray, sigma: float, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(arr.astype(np.float32) + rng.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)


def brighten(arr: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(arr.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def jpeg_bytes(arr: np.ndarray, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def data_url(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def png_data_url(arr: np.ndarray) -> str:
    return data_url(png_bytes(arr))


# ── Minimal GeoTIFF writer (uncompressed, single strip) ──────────────────
import struct as _struct


def make_geotiff(
    arr: np.ndarray,
    epsg: int = 32644,
    origin=(500000.0, 2000000.0),
    pixel=(10.0, 10.0),
    geographic: bool = False,
    nodata=None,
    georeference: bool = True,
    bits: int = 8,
) -> bytes:
    """Build a classic little-endian GeoTIFF from an (H, W) or (H, W, B) array.
    8-bit arrays decode in Pillow; 16-bit/multiband files exercise the
    metadata-only path."""
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w, bands = arr.shape
    dtype = np.uint8 if bits == 8 else np.dtype("<u2")
    pixel_bytes = arr.astype(dtype).tobytes()

    entries = []  # (tag, type, count, payload bytes)

    def short(tag, vals):
        entries.append((tag, 3, len(vals), _struct.pack("<%dH" % len(vals), *vals)))

    def long(tag, vals):
        entries.append((tag, 4, len(vals), _struct.pack("<%dI" % len(vals), *vals)))

    def dbl(tag, vals):
        entries.append((tag, 12, len(vals), _struct.pack("<%dd" % len(vals), *vals)))

    long(256, [w])
    long(257, [h])
    short(258, [bits] * bands)
    short(259, [1])
    short(262, [2 if bands >= 3 else 1])
    entries.append((273, 4, 1, b"PLACEHOLDER"))  # patched below
    short(277, [bands])
    long(278, [h])
    long(279, [len(pixel_bytes)])
    short(284, [1])
    short(339, [1] * bands)
    if georeference:
        dbl(33550, [pixel[0], pixel[1], 0.0])
        dbl(33922, [0.0, 0.0, 0.0, origin[0], origin[1], 0.0])
        keys = [(1024, 2 if geographic else 1), (1025, 1), ((2048 if geographic else 3072), epsg)]
        directory = [1, 1, 0, len(keys)]
        for k, v in keys:
            directory += [k, 0, 1, v]
        short(34735, directory)
    if nodata is not None:
        txt = (str(nodata) + "\x00").encode()
        entries.append((42113, 2, len(txt), txt))
    entries.sort(key=lambda e: e[0])

    pixel_off = 8
    extra_off = pixel_off + len(pixel_bytes)
    extra_off += extra_off % 2
    extra = b""
    rows = []
    for tag, typ, count, payload in entries:
        if tag == 273:
            value = _struct.pack("<I", pixel_off)
        elif len(payload) <= 4:
            value = payload.ljust(4, b"\x00")
        else:
            value = _struct.pack("<I", extra_off + len(extra))
            extra += payload + (b"\x00" if len(payload) % 2 else b"")
        rows.append(_struct.pack("<HHI", tag, typ, count) + value)
    ifd_off = extra_off + len(extra)
    ifd = _struct.pack("<H", len(rows)) + b"".join(rows) + _struct.pack("<I", 0)
    out = b"II*\x00" + _struct.pack("<I", ifd_off)
    out += pixel_bytes + (b"\x00" * (extra_off - pixel_off - len(pixel_bytes))) + extra + ifd
    return out
