"""Geospatial Lite Bridge: GeoTIFF metadata, pair validation, STAC item inspection."""
import io
import struct

import numpy as np
import pytest
from PIL import Image

from backend.geospatial.metadata import TiffParseError, inspect_tiff, is_tiff
from backend.geospatial.stac import StacError, compare_stac_items, inspect_stac_item, strip_private
from backend.geospatial.validation import validate_geospatial_pair
from tests import synthetic as S


def gt(arr, **kw):
    return S.make_geotiff(arr, **kw)


# ── GeoTIFF metadata ─────────────────────────────────────────────────────

def test_projected_geotiff_metadata(scene):
    m = inspect_tiff(gt(scene[:100, :120], epsg=32644, origin=(500000.0, 2000000.0), pixel=(10.0, 10.0), nodata=0))
    assert (m["width"], m["height"], m["band_count"]) == (120, 100, 3)
    assert m["georeferenced"] is True and m["georeferencing"] == "tiepoint_scale"
    assert m["crs"] == {"kind": "projected", "epsg": 32644, "user_defined": False, "units": "metre",
                        "citation": None, "raster_type": "PixelIsArea"}
    assert m["pixel_size"] == {"x": 10.0, "y": 10.0, "units": "metre"}
    assert m["bounds_native"] == {"west": 500000.0, "south": 1999000.0, "east": 501200.0, "north": 2000000.0}
    assert m["bounds_wgs84"] is None and any("EPSG:4326" in n for n in m["notes"])
    assert m["ground_sample_distance"] == {"x_m": 10.0, "y_m": 10.0, "approximate": False}
    assert m["nodata"] == 0.0 and m["bits_per_sample"] == [8, 8, 8]
    assert m["pixel_analysis_supported"] is True
    assert m["tiled"] is False and m["cog_like_layout"] is False
    assert "Metadata inspection only" in m["scope_note"]


def test_geographic_geotiff_has_wgs84_bounds_and_approximate_gsd(scene):
    m = inspect_tiff(gt(scene[..., 0], epsg=4326, geographic=True, origin=(78.3, 17.5), pixel=(0.0001, 0.0001)))
    assert m["crs"]["kind"] == "geographic" and m["crs"]["epsg"] == 4326 and m["crs"]["units"] == "degree"
    assert m["band_count"] == 1
    b = m["bounds_wgs84"]
    assert b["west"] == pytest.approx(78.3) and b["north"] == pytest.approx(17.5)
    assert b["east"] == pytest.approx(78.3 + 256 * 0.0001) and b["south"] == pytest.approx(17.5 - 256 * 0.0001)
    assert m["ground_sample_distance"]["approximate"] is True
    assert m["ground_sample_distance"]["y_m"] == pytest.approx(11.132, abs=0.01)


def test_web_mercator_bounds_are_converted(scene):
    m = inspect_tiff(gt(scene[:64, :64], epsg=3857, origin=(0.0, 0.0), pixel=(1000.0, 1000.0)))
    assert m["bounds_wgs84"]["west"] == pytest.approx(0.0)
    assert m["bounds_wgs84"]["east"] == pytest.approx(0.5749, abs=1e-3)


def test_geotiff_without_georeferencing_is_reported_as_such(scene):
    m = inspect_tiff(gt(scene, georeference=False))
    assert m["georeferenced"] is False and m["georeferencing"] == "none"
    assert m["crs"]["epsg"] is None and m["bounds_native"] is None and m["pixel_size"] is None


def test_16bit_multiband_is_metadata_only(scene):
    m = inspect_tiff(gt(np.zeros((40, 50, 4), np.uint16), bits=16, nodata=-9999))
    assert m["band_count"] == 4 and m["bits_per_sample"] == [16] * 4
    assert m["pixel_analysis_supported"] is False and "8-bit" in m["pixel_analysis_unsupported_reason"]
    assert m["nodata"] == -9999.0


@pytest.mark.parametrize("cut", [0, 3, 7, 12, 100, 5000, 40000])
def test_truncated_tiff_never_crashes(scene, cut):
    raw = gt(scene)
    try:
        inspect_tiff(raw[:cut])
    except TiffParseError:
        pass


def test_corrupt_tiff_structures_are_contained(scene):
    raw = bytearray(gt(scene))
    for pos in range(8, len(raw), 997):
        raw[pos] ^= 0xFF
    try:
        inspect_tiff(bytes(raw))
    except (TiffParseError, struct.error):
        pass
    with pytest.raises(TiffParseError):
        inspect_tiff(b"II*\x00" + struct.pack("<I", 10**9))      # IFD offset outside the file
    with pytest.raises(TiffParseError):
        inspect_tiff(b"not a tiff")
    loop = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", 0) + struct.pack("<I", 8)  # IFD that points to itself
    with pytest.raises(TiffParseError):
        inspect_tiff(loop)


def test_is_tiff_magic():
    assert is_tiff(b"II*\x00....") and is_tiff(b"MM\x00*....") and is_tiff(b"II+\x00....")
    assert not is_tiff(b"\x89PNG\r\n\x1a\n")


# ── end to end through the API ───────────────────────────────────────────

def b64(raw):
    return S.data_url(raw, "image/tiff")


def post(client, mode, query, imgs):
    return client.post("/api/analyze", json={"mode": mode, "query": query, "images": imgs})


def test_geotiff_input_is_analysed_and_metadata_is_preserved(client, scene):
    a, c = scene[:200, :200], S.add_patch(scene[:200, :200], 30, 30, 120, 120)
    j = post(client, "bitemporal", "What changed?", [b64(gt(a)), b64(gt(c))]).json()
    assert j["ok"] is True
    assert all(i["georeferenced"] for i in j["inputs"])
    assert j["inputs"][0]["geospatial"]["crs"]["epsg"] == 32644
    steps = " ".join(j["provenance"]["processing_steps"])
    assert "analysed in image space" in steps and "no georeferenced measurement" in steps
    assert j["diagnostics"]["geospatial_pair"]["state"] == "OK"
    assert not any("hectare" in x.lower() for x in [j["answer"]])


def test_geotiff_pair_with_different_crs_is_flagged_severe(client, scene):
    a = scene[:200, :200]
    j = post(client, "bitemporal", "What changed?", [b64(gt(a, epsg=32644)), b64(gt(a, epsg=32643))]).json()
    codes = [f["code"] for f in j["diagnostics"]["geospatial_pair"]["findings"]]
    assert "GEO_CRS_MISMATCH" in codes
    assert j["reliability"]["level"] == "UNSUPPORTED"


def test_geotiff_pair_with_disjoint_footprints_is_flagged(client, scene):
    a = scene[:200, :200]
    j = post(client, "bitemporal", "What changed?", [b64(gt(a, origin=(500000.0, 2000000.0))), b64(gt(a, origin=(600000.0, 2100000.0)))]).json()
    assert "GEO_EXTENT_MISMATCH" in [f["code"] for f in j["diagnostics"]["geospatial_pair"]["findings"]]


def test_one_georeferenced_one_not(client, scene):
    a = scene[:200, :200]
    j = post(client, "bitemporal", "What changed?", [b64(gt(a)), S.png_data_url(a)]).json()
    assert "GEO_ASYMMETRIC" in [f["code"] for f in j["diagnostics"]["geospatial_pair"]["findings"]]


def test_unsupported_raster_returns_metadata_not_a_crash(client):
    raw = gt(np.zeros((60, 70, 4), np.uint16), bits=16, epsg=32644)
    r = post(client, "single", "Describe the scene", [b64(raw)])
    j = r.json()
    assert r.status_code == 415 and j["error"]["code"] == "UNSUPPORTED_RASTER"
    assert j["geospatial"]["crs"]["epsg"] == 32644 and j["geospatial"]["band_count"] == 4


def test_area_questions_stay_unavailable_even_for_a_georeferenced_input(client, scene):
    j = post(client, "single", "How many hectares of vegetation are there?", [b64(gt(scene))]).json()
    assert j["status"] == "UNAVAILABLE_CAPABILITY" and j["error"]["code"] == "PHYSICAL_AREA_UNAVAILABLE"
    assert "present but Lite v1 does not compute ground measurements" in j["answer"]


def test_diagnostics_reports_geospatial_block(client, scene):
    r = client.post("/api/diagnostics", json={"images": [b64(gt(scene))]})
    assert r.json()["inputs"][0]["geospatial"]["pixel_size"]["x"] == 10.0


# ── pair validation unit tests ───────────────────────────────────────────

def meta(**kw):
    return inspect_tiff(gt(np.zeros((50, 50), np.uint8), **kw))


def test_pair_validation_rules():
    ok = validate_geospatial_pair(meta(), meta())
    assert ok["state"] == "OK" and ok["findings"] == []
    res = validate_geospatial_pair(meta(pixel=(10.0, 10.0)), meta(pixel=(20.0, 20.0)))
    assert "GEO_RESOLUTION_MISMATCH" in [f["code"] for f in res["findings"]]
    assert validate_geospatial_pair(None, None)["findings"][0]["code"] == "GEO_NO_GEOREFERENCING"
    part = validate_geospatial_pair(meta(origin=(500000.0, 2000000.0)), meta(origin=(500150.0, 2000000.0)))
    assert "GEO_EXTENT_PARTIAL" in [f["code"] for f in part["findings"]]


# ── STAC ─────────────────────────────────────────────────────────────────

def s2(id="S2A_43QHU_20240115", dt="2024-01-15T05:30:00Z", cloud=5.0, bbox=(78.0, 17.0, 79.0, 18.0), **extra):
    item = {
        "type": "Feature", "stac_version": "1.0.0", "id": id, "collection": "sentinel-2-l2a",
        "bbox": list(bbox), "geometry": {"type": "Polygon", "coordinates": []},
        "properties": {"datetime": dt, "platform": "sentinel-2a", "gsd": 10, "eo:cloud_cover": cloud, "proj:epsg": 32644,
                       "eo:bands": [{"name": "B04", "common_name": "red"}, {"name": "B08", "common_name": "nir"}], **extra},
        "assets": {
            "visual": {"href": "https://example.invalid/visual.tif", "type": "image/tiff; application=geotiff; profile=cloud-optimized", "roles": ["visual"]},
            "B08": {"href": "https://example.invalid/B08.tif", "type": "image/tiff; application=geotiff", "roles": ["data"], "eo:bands": [{"name": "B08"}]},
            "thumbnail": {"href": "https://example.invalid/t.jpg", "type": "image/jpeg", "roles": ["thumbnail"]},
            "metadata": {"href": "https://example.invalid/m.xml", "type": "application/xml"},
        },
    }
    return item


def s1(id="S1A_IW_20240117", dt="2024-01-17T00:10:00Z", bbox=(78.0, 17.0, 79.0, 18.0)):
    return {
        "type": "Feature", "stac_version": "1.0.0", "id": id, "collection": "sentinel-1-grd", "bbox": list(bbox),
        "geometry": None,
        "properties": {"datetime": dt, "platform": "sentinel-1a", "sar:polarizations": ["VV", "VH"], "sar:instrument_mode": "IW", "sat:orbit_state": "ascending"},
        "assets": {"vv": {"href": "https://example.invalid/vv.tif", "type": "image/tiff; application=geotiff", "roles": ["data"]}},
    }


def test_stac_item_inspection():
    r = inspect_stac_item(s2())
    assert r["collection"] == "sentinel-2-l2a" and r["platform"] == "sentinel-2a" and r["epsg"] == 32644
    assert r["modality"]["value"] == "optical" and r["cloud_cover_pct"] == 5.0
    assert [b["name"] for b in r["bands"]] == ["B04", "B08"]
    by = {a["key"]: a for a in r["assets"]}
    assert by["visual"]["is_cog_or_geotiff"] and not by["visual"]["lite_v1_can_analyse"]
    assert by["thumbnail"]["lite_v1_can_analyse"] and by["thumbnail"]["is_rendered_preview"]
    assert not by["metadata"]["is_cog_or_geotiff"]
    assert r["analysable_asset_count"] == 1
    assert "Nothing was fetched" in r["scope_note"]
    assert "_dt" not in strip_private(r)


def test_stac_sar_modality_inference():
    r = inspect_stac_item(s1())
    assert r["modality"]["value"] == "sar" and r["sar"]["sar:polarizations"] == ["VV", "VH"]
    assert r["sar"]["sat:orbit_state"] == "ascending"


@pytest.mark.parametrize("bad", [None, [], "x", {}, {"type": "Feature"}, {"type": "Feature", "id": "a"},
                                  {"type": "Feature", "id": "a", "properties": {}, "assets": []}])
def test_invalid_stac_items_are_rejected(bad):
    with pytest.raises(StacError):
        inspect_stac_item(bad)


def test_stac_missing_fields_are_findings_not_inventions():
    item = {"type": "Feature", "id": "x", "properties": {}, "assets": {}}
    r = inspect_stac_item(item)
    assert {f["code"] for f in r["findings"]} == {"STAC_NO_DATETIME", "STAC_NO_BBOX"}
    assert r["datetime"] is None and r["bbox"] is None and r["modality"]["value"] == "unknown"


def test_stac_asset_limit():
    item = s2()
    item["assets"] = {f"a{i}": {"type": "image/tiff"} for i in range(501)}
    with pytest.raises(StacError):
        inspect_stac_item(item)


def codes(cmp):
    return {f["code"] for f in cmp["findings"]}


def test_stac_change_pair_compatibility():
    a, b = inspect_stac_item(s2()), inspect_stac_item(s2(id="later", dt="2024-03-15T05:30:00Z"))
    c = compare_stac_items(a, b, "change")
    assert c["state"] == "COMPATIBLE" and c["gap_days"] == pytest.approx(60, abs=0.1) and c["bbox_iou"] == 1.0
    assert compare_stac_items(a, inspect_stac_item(s1()), "change")["state"] == "INCOMPATIBLE"
    assert "STAC_SAME_TIME" in codes(compare_stac_items(a, inspect_stac_item(s2(id="dup")), "change"))
    cloudy = compare_stac_items(a, inspect_stac_item(s2(id="c", dt="2024-02-01T00:00:00Z", cloud=60)), "change")
    assert "STAC_CLOUDY" in codes(cloudy) and cloudy["state"] == "COMPATIBLE_WITH_WARNINGS"
    far = compare_stac_items(a, inspect_stac_item(s2(id="f", dt="2024-02-01T00:00:00Z", bbox=(10, 10, 11, 11))), "change")
    assert "STAC_FOOTPRINT_MISMATCH" in codes(far) and far["state"] == "INCOMPATIBLE"
    rev = compare_stac_items(b, a, "change")
    assert "STAC_ORDER" in codes(rev)


def test_stac_fusion_pair_compatibility():
    o, s = inspect_stac_item(s2()), inspect_stac_item(s1())
    ok = compare_stac_items(o, s, "fusion")
    assert ok["state"] == "COMPATIBLE" and ok["gap_days"] == pytest.approx(1.77, abs=0.05)
    assert compare_stac_items(o, inspect_stac_item(s2(id="b")), "fusion")["state"] == "INCOMPATIBLE"
    late = compare_stac_items(o, inspect_stac_item(s1(dt="2024-02-20T00:00:00Z")), "fusion")
    assert "STAC_FUSION_TIME_GAP" in codes(late)
    with pytest.raises(StacError):
        compare_stac_items(o, s, "bogus")


def test_stac_antimeridian_bbox_is_not_misreported():
    wrap = inspect_stac_item(s2(bbox=(179.0, 10.0, -179.0, 11.0)))
    c = compare_stac_items(wrap, inspect_stac_item(s2(id="b", dt="2024-02-01T00:00:00Z")), "change")
    assert "STAC_BBOX_UNSUPPORTED" in codes(c)


def test_stac_endpoint(client):
    r = client.post("/api/stac/inspect", json={"items": [s2()]})
    j = r.json()
    assert r.status_code == 200 and j["ok"] and j["items"][0]["id"] == "S2A_43QHU_20240115" and j["comparison"] is None
    r = client.post("/api/stac/inspect", json={"items": [s2(), s1()], "purpose": "fusion"})
    assert r.json()["comparison"]["state"] == "COMPATIBLE"
    assert client.post("/api/stac/inspect", json={"items": [s2(), s1()]}).json()["error"]["code"] == "PURPOSE_REQUIRED"
    assert client.post("/api/stac/inspect", json={"items": [{"type": "x"}]}).json()["error"]["code"] == "INVALID_STAC_ITEM"
    assert client.post("/api/stac/inspect", json={"items": []}).status_code == 422
    assert client.post("/api/stac/inspect", json={"items": [s2()] * 3}).status_code == 422


def test_stac_never_touches_the_network(monkeypatch):
    import socket

    def no_net(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", no_net)
    monkeypatch.setattr(socket, "create_connection", no_net)
    inspect_stac_item(s2())
