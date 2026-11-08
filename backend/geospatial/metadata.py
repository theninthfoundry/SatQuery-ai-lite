"""
backend/geospatial/metadata.py

GeoTIFF metadata inspection with NO raster dependencies.

This is a small, bounds-checked reader for the TIFF/BigTIFF tag directory
and the GeoTIFF GeoKeys. It reads metadata only; it never decodes pixels and
it never executes anything found in the file.

Scope (Geospatial Lite Bridge, Phase A): dimensions, bands, bit depth,
compression, tiling/overview layout, NoData, CRS identification, the
georeferencing transform, pixel size and geographic/projected bounds.

Deliberately NOT done here: reprojection (no PROJ in Lite), orthorectification,
resampling, or any physical measurement. Bounds are therefore reported in the
file's own CRS; WGS84 bounds are provided only for EPSG:4326 and EPSG:3857.
"""

from __future__ import annotations

import math
import re
import struct
from typing import Any, Dict, List, Optional, Tuple

MAGICS = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")

_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8}
_TYPE_CODE = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 11: "f", 12: "d", 16: "Q", 17: "q", 18: "Q"}

_WANTED = {
    254, 256, 257, 258, 259, 262, 277, 284, 322, 323, 338, 339,
    33550, 33922, 34264, 34735, 34736, 34737, 42112, 42113,
}
_MAX_VALUES = 4096
_MAX_IFDS = 64


class TiffParseError(ValueError):
    pass


def is_tiff(data: bytes) -> bool:
    return data[:4] in MAGICS


class _Reader:
    def __init__(self, data: bytes):
        if not is_tiff(data):
            raise TiffParseError("not a TIFF file")
        self.d = data
        self.e = "<" if data[:2] == b"II" else ">"
        self.big = data[2:4] in (b"+\x00", b"\x00+")
        if self.big:
            if len(data) < 16:
                raise TiffParseError("truncated BigTIFF header")
            bs, _zero = struct.unpack(self.e + "HH", data[4:8])
            if bs != 8:
                raise TiffParseError("unsupported BigTIFF offset size")
            self.first = struct.unpack(self.e + "Q", data[8:16])[0]
        else:
            if len(data) < 8:
                raise TiffParseError("truncated TIFF header")
            self.first = struct.unpack(self.e + "I", data[4:8])[0]

    def _need(self, off: int, n: int) -> None:
        if off < 0 or n < 0 or off + n > len(self.d):
            raise TiffParseError("offset outside file (truncated or corrupt)")

    def read_ifd(self, off: int) -> Tuple[Dict[int, Tuple[int, int, bytes]], int]:
        e = self.e
        if self.big:
            self._need(off, 8)
            n = struct.unpack(e + "Q", self.d[off:off + 8])[0]
            entry, base, vsz = 20, off + 8, 8
        else:
            self._need(off, 2)
            n = struct.unpack(e + "H", self.d[off:off + 2])[0]
            entry, base, vsz = 12, off + 2, 4
        if n > 4096:
            raise TiffParseError("implausible IFD entry count")
        self._need(base, n * entry + vsz)
        tags: Dict[int, Tuple[int, int, bytes]] = {}
        for i in range(n):
            p = base + i * entry
            if self.big:
                tag, typ, count = struct.unpack(e + "HHQ", self.d[p:p + 12])
                raw = self.d[p + 12:p + 20]
            else:
                tag, typ, count = struct.unpack(e + "HHI", self.d[p:p + 8])
                raw = self.d[p + 8:p + 12]
            tags[tag] = (typ, count, raw)
        nxt_p = base + n * entry
        nxt = struct.unpack(e + ("Q" if self.big else "I"), self.d[nxt_p:nxt_p + vsz])[0]
        return tags, nxt

    def values(self, entry: Tuple[int, int, bytes]) -> List[Any]:
        typ, count, raw = entry
        size = _TYPE_SIZE.get(typ)
        if size is None:
            return []
        if count > _MAX_VALUES:
            raise TiffParseError("tag value count exceeds safety limit")
        total = size * count
        inline = 8 if self.big else 4
        if total <= inline:
            buf = raw[:total]
        else:
            off = struct.unpack(self.e + ("Q" if self.big else "I"), raw[:inline])[0]
            self._need(off, total)
            buf = self.d[off:off + total]
        if typ == 2:
            return [buf.split(b"\x00", 1)[0].decode("latin-1", "replace")]
        if typ in (5, 10):
            code = "I" if typ == 5 else "i"
            nums = struct.unpack(self.e + code * (2 * count), buf)
            return [nums[i] / nums[i + 1] if nums[i + 1] else None for i in range(0, len(nums), 2)]
        if typ == 7:
            return [bytes(buf)]
        return list(struct.unpack(self.e + _TYPE_CODE[typ] * count, buf))


# ── GeoKeys ──────────────────────────────────────────────────────────────

_USER_DEFINED = 32767
_GEOKEY_NAMES = {
    1024: "GTModelType", 1025: "GTRasterType", 1026: "GTCitation",
    2048: "GeographicType", 2049: "GeogCitation", 2050: "GeogGeodeticDatum",
    3072: "ProjectedCSType", 3073: "PCSCitation", 3076: "ProjLinearUnits",
}
_LINEAR_UNITS = {9001: ("metre", 1.0), 9002: ("foot", 0.3048), 9003: ("US survey foot", 1200 / 3937)}


def _parse_geokeys(directory: List[int], doubles: List[float], ascii_blob: str) -> Dict[str, Any]:
    keys: Dict[str, Any] = {}
    if len(directory) < 4:
        return keys
    n = directory[3]
    for i in range(n):
        base = 4 + i * 4
        if base + 4 > len(directory):
            break
        key_id, loc, count, val = directory[base:base + 4]
        name = _GEOKEY_NAMES.get(key_id)
        if name is None:
            continue
        if loc == 0:
            keys[name] = val
        elif loc == 34736 and val + count <= len(doubles):
            keys[name] = doubles[val:val + count][0] if count == 1 else doubles[val:val + count]
        elif loc == 34737:
            keys[name] = ascii_blob[val:val + count].rstrip("|").strip()
    return keys


def _metres_epsg(epsg: Optional[int]) -> bool:
    if epsg is None:
        return False
    return 32601 <= epsg <= 32660 or 32701 <= epsg <= 32760 or epsg in (3857, 3395, 3035, 3857)


def _mercator_to_lonlat(x: float, y: float) -> Tuple[float, float]:
    r = 6378137.0
    lon = math.degrees(x / r)
    lat = math.degrees(2 * math.atan(math.exp(y / r)) - math.pi / 2)
    return lon, lat


def _band_descriptions(xml: Optional[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not xml:
        return out
    for m in re.finditer(r'<Item\s+[^>]*name="DESCRIPTION"[^>]*sample="(\d+)"[^>]*>([^<]*)</Item>', xml):
        out[int(m.group(1))] = m.group(2).strip()
    for m in re.finditer(r'<Item\s+[^>]*sample="(\d+)"[^>]*name="DESCRIPTION"[^>]*>([^<]*)</Item>', xml):
        out[int(m.group(1))] = m.group(2).strip()
    return out


def inspect_tiff(data: bytes) -> Dict[str, Any]:
    """Return a metadata dictionary for a TIFF / GeoTIFF byte string.

    Raises TiffParseError if the file is not a parseable TIFF."""
    r = _Reader(data)
    ifds: List[Dict[int, Tuple[int, int, bytes]]] = []
    seen = set()
    off = r.first
    while off and off not in seen and len(ifds) < _MAX_IFDS:
        seen.add(off)
        tags, off = r.read_ifd(off)
        ifds.append(tags)
    if not ifds:
        raise TiffParseError("no image directory found")

    def val(tags, tag, default=None):
        if tag not in tags:
            return default
        try:
            v = r.values(tags[tag])
        except (TiffParseError, struct.error):
            return default
        return v

    t0 = ifds[0]
    width = (val(t0, 256) or [None])[0]
    height = (val(t0, 257) or [None])[0]
    if not width or not height:
        raise TiffParseError("missing image dimensions")
    spp = (val(t0, 277) or [1])[0]
    bits = val(t0, 258) or [1]
    sample_format = (val(t0, 339) or [1])[0]
    compression = (val(t0, 259) or [1])[0]
    photometric = (val(t0, 262) or [None])[0]
    planar = (val(t0, 284) or [1])[0]
    tiled = 322 in t0 and 323 in t0
    tile = [(val(t0, 322) or [None])[0], (val(t0, 323) or [None])[0]] if tiled else None
    overviews = sum(1 for t in ifds[1:] if ((val(t, 254) or [0])[0] & 1))

    nodata_raw = val(t0, 42113)
    nodata: Optional[float] = None
    if nodata_raw:
        try:
            nodata = float(nodata_raw[0].strip())
        except ValueError:
            nodata = None

    pixel_scale = val(t0, 33550)
    tiepoints = val(t0, 33922) or []
    transform = val(t0, 34264)
    geokeys = _parse_geokeys(
        val(t0, 34735) or [], val(t0, 34736) or [], (val(t0, 34737) or [""])[0]
    )
    band_desc = _band_descriptions((val(t0, 42112) or [None])[0])

    # ── CRS ──
    model_type = geokeys.get("GTModelType")
    raster_type = geokeys.get("GTRasterType")
    proj_code = geokeys.get("ProjectedCSType")
    geog_code = geokeys.get("GeographicType")
    crs_kind = {1: "projected", 2: "geographic", 3: "geocentric"}.get(model_type, "unknown")
    epsg: Optional[int] = None
    user_defined = False
    if crs_kind == "projected":
        if proj_code in (None, 0):
            pass
        elif proj_code == _USER_DEFINED:
            user_defined = True
        else:
            epsg = int(proj_code)
    elif crs_kind == "geographic":
        if geog_code in (None, 0):
            pass
        elif geog_code == _USER_DEFINED:
            user_defined = True
        else:
            epsg = int(geog_code)
    linear = _LINEAR_UNITS.get(geokeys.get("ProjLinearUnits"))
    if crs_kind == "geographic":
        units = "degree"
    elif linear:
        units = linear[0]
    elif _metres_epsg(epsg):
        units = "metre"
    else:
        units = "unknown"

    # ── transform ──
    georef = "none"
    origin = None
    size = None
    corners: Optional[List[Tuple[float, float]]] = None
    rotated = False
    notes: List[str] = []
    if transform and len(transform) >= 16:
        georef = "affine"
        m = transform
        px = lambda c, rr: (m[0] * c + m[1] * rr + m[3], m[4] * c + m[5] * rr + m[7])
        corners = [px(0, 0), px(width, 0), px(0, height), px(width, height)]
        size = (math.hypot(m[0], m[4]), math.hypot(m[1], m[5]))
        rotated = bool(abs(m[1]) > 1e-12 or abs(m[4]) > 1e-12)
        origin = corners[0]
    elif pixel_scale and len(tiepoints) >= 6 and len(tiepoints) == 6:
        georef = "tiepoint_scale"
        i, j, _k, x, y, _z = tiepoints[:6]
        sx, sy = pixel_scale[0], pixel_scale[1]
        x0, y0 = x - i * sx, y + j * sy
        if raster_type == 2:  # PixelIsPoint: tiepoint is the pixel centre
            x0 -= sx / 2.0
            y0 += sy / 2.0
        corners = [(x0, y0), (x0 + width * sx, y0), (x0, y0 - height * sy), (x0 + width * sx, y0 - height * sy)]
        size = (sx, sy)
        origin = (x0, y0)
    elif len(tiepoints) > 6:
        georef = "gcps"
        notes.append("File uses multiple ground control points; a single affine transform is not defined and bounds are not computed.")

    bounds_native = None
    if corners:
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        bounds_native = {"west": min(xs), "south": min(ys), "east": max(xs), "north": max(ys)}

    bounds_wgs84 = None
    if bounds_native:
        if epsg == 4326:
            bounds_wgs84 = dict(bounds_native)
        elif epsg == 3857:
            w, s = _mercator_to_lonlat(bounds_native["west"], bounds_native["south"])
            e, n = _mercator_to_lonlat(bounds_native["east"], bounds_native["north"])
            bounds_wgs84 = {"west": w, "south": s, "east": e, "north": n}
        else:
            notes.append("WGS84 bounds are only derived for EPSG:4326 and EPSG:3857 (Lite has no reprojection library).")

    gsd = None
    if size:
        if units == "metre":
            gsd = {"x_m": size[0], "y_m": size[1], "approximate": False}
        elif units in ("foot", "US survey foot") and linear:
            gsd = {"x_m": size[0] * linear[1], "y_m": size[1] * linear[1], "approximate": False}
        elif units == "degree" and bounds_native:
            lat = (bounds_native["north"] + bounds_native["south"]) / 2.0
            gsd = {
                "x_m": size[0] * 111_320.0 * math.cos(math.radians(lat)),
                "y_m": size[1] * 111_320.0,
                "approximate": True,
            }

    pixel_ok = (
        spp in (1, 3, 4) and all(b == 8 for b in bits) and sample_format == 1 and planar == 1
    )
    reason = None
    if not pixel_ok:
        reason = (
            f"Lite v1 analyses 8-bit unsigned grayscale/RGB rasters only; this file has {spp} band(s), "
            f"bits per sample {bits}, sample format {sample_format}. Metadata is shown, pixel analysis is not available."
        )

    bands = []
    for b in range(spp):
        bands.append({"index": b + 1, "description": band_desc.get(b), "bits": bits[b] if b < len(bits) else bits[-1]})

    def rnd(v):
        return None if v is None else round(float(v), 8)

    return {
        "format": "BigTIFF" if r.big else "TIFF",
        "width": int(width),
        "height": int(height),
        "band_count": int(spp),
        "bands": bands,
        "bits_per_sample": list(bits),
        "sample_format": int(sample_format),
        "photometric": photometric,
        "compression": int(compression),
        "planar_configuration": int(planar),
        "tiled": bool(tiled),
        "tile_size": tile,
        "ifd_count": len(ifds),
        "overview_count": overviews,
        "cog_like_layout": bool(tiled and overviews > 0),
        "cog_note": "Heuristic: tiled with overviews. COG compliance (IFD/data ordering) is not validated.",
        "nodata": nodata,
        "georeferenced": georef in ("affine", "tiepoint_scale"),
        "georeferencing": georef,
        "rotated_transform": rotated,
        "crs": {
            "kind": crs_kind,
            "epsg": epsg,
            "user_defined": user_defined,
            "units": units,
            "citation": geokeys.get("PCSCitation") or geokeys.get("GeogCitation") or geokeys.get("GTCitation"),
            "raster_type": {1: "PixelIsArea", 2: "PixelIsPoint"}.get(raster_type),
        },
        "origin": None if origin is None else [rnd(origin[0]), rnd(origin[1])],
        "pixel_size": None if size is None else {"x": rnd(size[0]), "y": rnd(size[1]), "units": units},
        "ground_sample_distance": None if gsd is None else {k: (rnd(v) if k != "approximate" else v) for k, v in gsd.items()},
        "bounds_native": None if bounds_native is None else {k: rnd(v) for k, v in bounds_native.items()},
        "bounds_wgs84": None if bounds_wgs84 is None else {k: rnd(v) for k, v in bounds_wgs84.items()},
        "pixel_analysis_supported": bool(pixel_ok),
        "pixel_analysis_unsupported_reason": reason,
        "notes": notes,
        "scope_note": (
            "Metadata inspection only. No raster processing, reprojection or physical measurement is performed in Lite v1."
        ),
    }
