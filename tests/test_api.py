"""API contract: validation, hardening, structured errors, capability gating, provenance."""
import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

from backend import config
from tests import synthetic as S


def post(client, mode, query, imgs, **kw):
    body = {"mode": mode, "query": query, "images": [i if isinstance(i, str) else S.png_data_url(i) for i in imgs]}
    body.update(kw)
    return client.post("/api/analyze", json=body)


def err(r):
    return r.json()["error"]["code"]


# ── health / capabilities ────────────────────────────────────────────────

def test_health_and_capabilities(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["engine_version"] == config.ENGINE_VERSION
    c = client.get("/api/capabilities").json()
    assert {t["id"] for t in c["tasks"]} == {"caption", "vqa", "grounding", "change-describe", "change-vqa", "fusion"}
    assert c["limits"]["max_image_bytes"] == config.LIMITS.max_image_bytes
    assert any("Trained vision-language models" in x for x in c["not_supported"])
    phases = {p["phase"]: p["status"] for p in c["geospatial"]}
    assert phases == {"A": "READY", "B": "READY", "C": "NOT_IMPLEMENTED"}
    assert "No calibrated confidence" in c["confidence_policy"]


# ── request validation ───────────────────────────────────────────────────

@pytest.mark.parametrize("query", ["", "   ", "\n\t "])
def test_empty_query_rejected(client, scene, query):
    r = post(client, "single", query, [scene])
    assert r.status_code == 422 and err(r) == "INVALID_REQUEST"
    assert "Value error" not in r.json()["error"]["message"]


def test_overlong_query_rejected(client, scene):
    assert post(client, "single", "x" * 501, [scene]).status_code == 422
    assert post(client, "single", "x" * 500, [scene]).status_code == 200


def test_control_characters_are_normalised(client, scene):
    r = post(client, "single", "Describe\x00 the\x07 scene", [scene])
    assert r.status_code == 200 and r.json()["query"] == "Describe the scene"


@pytest.mark.parametrize("body", [
    {"mode": "bogus", "query": "q", "images": ["x"]},
    {"mode": "single", "query": "q"},
    {"mode": "single", "query": "q", "images": []},
    {"mode": "single", "query": "q", "images": ["a", "b", "c"]},
    {"mode": "single", "query": "q", "images": ["a"], "unexpected": 1},
    {"mode": "single", "query": "q", "images": ["a"], "options": {"grid_size": 99}},
    {"mode": "single", "query": "q", "images": ["a"], "options": {"base_threshold": 0}},
    {"mode": "single", "query": 5, "images": ["a"]},
])
def test_malformed_requests_get_structured_422(client, body):
    r = client.post("/api/analyze", json=body)
    assert r.status_code == 422
    j = r.json()
    assert j["ok"] is False and j["status"] == "REJECTED_INPUT" and j["error"]["code"] == "INVALID_REQUEST"


def test_non_json_body_is_structured(client):
    r = client.post("/api/analyze", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["ok"] is False


def test_wrong_image_count_for_mode(client, scene):
    r = post(client, "bitemporal", "what changed", [scene])
    assert r.status_code == 422 and r.json()["status"] == "INSUFFICIENT_INPUT" and err(r) == "WRONG_IMAGE_COUNT"
    r = post(client, "single", "describe", [scene, scene])
    assert err(r) == "WRONG_IMAGE_COUNT"


def test_invalid_thresholds_rejected(client, scene):
    r = post(client, "bitemporal", "what changed", [scene, scene], options={"thresholds": [0, 28]})
    assert r.status_code == 422 and err(r) == "INVALID_OPTIONS"
    r = post(client, "bitemporal", "what changed", [scene, scene], options={"thresholds": list(range(1, 30))})
    assert r.status_code == 422


# ── image hardening ──────────────────────────────────────────────────────

def test_malformed_base64(client):
    for payload in ["!!!not base64!!!", "data:image/png;base64,@@@@", "AAAA="]:
        r = post(client, "single", "describe", [payload])
        assert r.status_code == 400 and r.json()["ok"] is False
        assert err(r) in ("INVALID_BASE64", "CORRUPT_IMAGE")


def test_empty_image_payload(client):
    assert err(post(client, "single", "describe", ["data:image/png;base64,"])) == "EMPTY_IMAGE"
    assert err(post(client, "single", "describe", ["   "])) == "EMPTY_IMAGE"


def test_unsupported_format_gif(client, scene):
    buf = io.BytesIO(); Image.fromarray(scene).save(buf, format="GIF")
    r = post(client, "single", "describe", [S.data_url(buf.getvalue(), "image/gif")])
    assert r.status_code == 415 and err(r) == "UNSUPPORTED_FORMAT"


def test_mime_prefix_is_not_trusted(client, scene):
    r = post(client, "single", "describe", [S.data_url(S.png_bytes(scene), "image/jpeg")])
    assert r.status_code == 200


def test_corrupt_and_truncated_images(client, scene):
    raw = S.png_bytes(scene)
    assert err(post(client, "single", "describe", [S.data_url(raw[: len(raw) // 2])])) == "CORRUPT_IMAGE"
    assert err(post(client, "single", "describe", [S.data_url(b"this is not an image at all")])) == "CORRUPT_IMAGE"
    assert err(post(client, "single", "describe", [S.data_url(raw[:8] + b"\x00" * 50)])) == "CORRUPT_IMAGE"


def test_oversized_file_rejected_before_decoding(client):
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (900, 900, 3), dtype=np.uint8)   # incompressible -> > 1.6 MB PNG
    raw = S.png_bytes(noise)
    assert len(raw) > config.LIMITS.max_image_bytes
    r = post(client, "single", "describe", [S.data_url(raw)])
    assert r.status_code == 413 and err(r) == "IMAGE_TOO_LARGE"


def test_pixel_count_limit_uses_header_not_decode(client):
    huge = np.zeros((5200, 5200, 3), np.uint8)         # 27 MP, compresses to a few KB
    raw = S.png_bytes(huge)
    assert len(raw) < config.LIMITS.max_image_bytes
    r = post(client, "single", "describe", [S.data_url(raw)])
    assert r.status_code == 413 and err(r) == "IMAGE_TOO_MANY_PIXELS"


def test_too_small_image(client):
    r = post(client, "single", "describe", [np.zeros((20, 20, 3), np.uint8)])
    assert r.status_code == 400 and err(r) == "IMAGE_TOO_SMALL"


def test_16bit_png_is_rejected_not_silently_rescaled(client):
    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(1).random((64, 64)) * 65535).astype(np.uint16)).save(buf, format="PNG")
    r = post(client, "single", "describe", [S.data_url(buf.getvalue())])
    assert r.status_code == 415 and err(r) == "UNSUPPORTED_BIT_DEPTH"


def test_rgba_and_palette_inputs_are_handled(client, scene):
    rgba = np.dstack([scene, np.full(scene.shape[:2], 128, np.uint8)])
    r = post(client, "single", "describe the scene", [rgba])
    assert r.status_code == 200
    assert any("Alpha" in w for w in r.json()["inputs"][0]["warnings"])
    assert r.json()["status"] == "COMPLETED_WITH_WARNINGS"


def test_total_request_size_limit(client):
    body = '{"mode":"single","query":"q","images":["' + "A" * 5_000_000 + '"]}'
    r = client.post("/api/analyze", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_combined_image_size_limit(client, scene, monkeypatch):
    import dataclasses

    raw = S.png_bytes(scene)
    monkeypatch.setattr(config, "LIMITS", dataclasses.replace(config.LIMITS, max_total_image_bytes=len(raw) + 10))
    r = post(client, "bitemporal", "what changed", [scene, scene])
    assert r.status_code == 413 and err(r) == "IMAGES_TOO_LARGE"


# ── structured failure, never a stack trace ──────────────────────────────

def test_unexpected_tool_error_is_contained(client, scene, monkeypatch):
    from backend.tools import vqa_tool

    def boom(*a, **k):
        raise RuntimeError("secret internal detail /srv/app/path")

    monkeypatch.setattr(vqa_tool, "run", boom)
    r = post(client, "single", "How much vegetation cover is present?", [scene])
    assert r.status_code == 500
    j = r.json()
    assert j["status"] == "ERROR" and j["error"]["code"] == "TOOL_FAILURE"
    assert "secret" not in json.dumps(j) and "Traceback" not in json.dumps(j)
    assert j["trace"][-1]["label"] == "ERROR"


def test_unhandled_exception_returns_structured_500(scene, monkeypatch):
    from fastapi.testclient import TestClient
    from api.analyze import app
    import backend.app as appmod

    def boom(_):
        raise ValueError("kaboom /etc/passwd")

    monkeypatch.setattr(appmod, "run_analysis", boom)
    c = TestClient(app, raise_server_exceptions=False)
    r = post(c, "single", "describe", [scene])
    assert r.status_code == 500 and r.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "kaboom" not in r.text


def test_unknown_route_and_method(client):
    assert client.get("/api/nope").status_code == 404
    assert client.get("/api/analyze").status_code == 405
    assert client.get("/api/nope").json()["ok"] is False


# ── capability-aware agent ───────────────────────────────────────────────

@pytest.mark.parametrize("query,code", [
    ("Calculate NDVI for this image", "SPECTRAL_BANDS_UNAVAILABLE"),
    ("What is the NIR reflectance here?", "SPECTRAL_BANDS_UNAVAILABLE"),
    ("Give me the calibrated sigma0 backscatter in dB", "CALIBRATED_SAR_UNAVAILABLE"),
    ("How many hectares of water are there?", "PHYSICAL_AREA_UNAVAILABLE"),
    ("What is the area in square kilometres?", "PHYSICAL_AREA_UNAVAILABLE"),
    ("What are the latitude and longitude of the lake?", "GEOCOORDINATES_UNAVAILABLE"),
    ("Which satellite acquired this and what sensor?", "ACQUISITION_METADATA_UNAVAILABLE"),
    ("How many buildings are in this image?", "OBJECT_COUNTING_UNAVAILABLE"),
])
def test_unavailable_capabilities_are_explained_not_guessed(client, scene, query, code):
    r = post(client, "single", query, [scene])
    j = r.json()
    assert r.status_code == 200 and j["status"] == "UNAVAILABLE_CAPABILITY" and j["ok"] is False
    assert j["error"]["code"] == code
    assert j["unmet_requirements"][0]["requires"] and j["supported_alternatives"]
    assert "answer" in j and "requires" in j["answer"]
    assert "evidence" not in j and j["tool"] is None          # no tool ran, nothing was invented


def test_rgb_image_asked_for_calibrated_sar_names_what_is_missing(client, scene):
    j = post(client, "single", "Run calibrated SAR analysis on this", [scene]).json()
    assert "calibrated SAR data" in j["answer"] and "8-bit RGB" in j["answer"]


def test_change_question_with_one_image_is_insufficient_input(client, scene):
    r = post(client, "single", "What changed here?", [scene])
    assert r.status_code == 422 and r.json()["status"] == "INSUFFICIENT_INPUT"
    assert "two observations" in r.json()["answer"]


def test_fusion_question_with_one_image_is_insufficient_input(client, scene):
    r = post(client, "single", "Use optical and SAR together to find water", [scene])
    assert r.json()["status"] == "INSUFFICIENT_INPUT"


def test_ordinary_queries_are_not_blocked(client, scene):
    for q in ["Describe the land-cover and major objects visible in this image.", "How much vegetation cover is present?",
              "Highlight the water body referred to in the query.", "Highlight the built-up area in this image."]:
        assert post(client, "single", q, [scene]).json()["ok"] is True, q
    for q in ["What changed between these two dates, and where did the change occur?",
              "Has the built-up area increased, decreased, or remained unchanged?"]:
        assert post(client, "bitemporal", q, [scene, S.add_patch(scene, 10, 10, 90, 90)]).json()["ok"] is True, q


# ── honesty invariants ───────────────────────────────────────────────────

def _numeric_confidence(obj, path=""):
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "confidence" in k.lower() and isinstance(v, (int, float)) and not isinstance(v, bool):
                hits.append(path + "/" + k)
            hits += _numeric_confidence(v, path + "/" + k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += _numeric_confidence(v, f"{path}[{i}]")
    return hits


@pytest.mark.parametrize("mode,query,n", [
    ("single", "Describe the scene", 1), ("single", "How much water is there?", 1), ("single", "Highlight the vegetation", 1),
    ("bitemporal", "What changed?", 2), ("bitemporal", "Has built-up increased?", 2),
    ("cross", "Use optical and SAR together to find water", 2),
])
def test_no_numeric_confidence_anywhere(client, scene, mode, query, n):
    imgs = [scene, S.add_patch(scene, 20, 20, 120, 120)][:n]
    j = post(client, mode, query, imgs).json()
    assert j["ok"] is True
    assert j["confidence"] is None
    assert _numeric_confidence(j) == []
    assert j["reliability"]["level"] in ("SUPPORTED", "CAUTION", "WEAK", "UNSUPPORTED")
    assert j["reliability"]["confidence_note"]


def test_single_image_heuristics_never_claim_supported(client, scene):
    for q in ["Describe the scene", "How much water is there?", "Highlight the vegetation"]:
        j = post(client, "single", q, [scene]).json()
        assert j["reliability"]["level"] != "SUPPORTED"
        assert any(f["name"] == "method" for f in j["reliability"]["factors"])


def test_response_contains_the_full_evidence_contract(client, scene):
    j = post(client, "bitemporal", "What changed?", [scene, S.add_patch(scene, 20, 20, 120, 120)]).json()
    for key in ("schema_version", "engine_version", "analysis_id", "status", "answer", "reliability", "diagnostics",
                "evidence", "layers", "methodology", "limitations", "provenance", "trace", "report", "inputs"):
        assert key in j, key
    assert j["methodology"]["steps"] and j["limitations"]
    assert j["diagnostics"]["quality"] and j["diagnostics"]["alignment"] and j["diagnostics"]["suitability"]
    assert "does not perform geospatial" in j["diagnostics"]["alignment"]["limitation"] or "translation only" in j["diagnostics"]["alignment"]["limitation"]


def test_missing_geospatial_metadata_is_explicit(client, scene):
    j = post(client, "bitemporal", "What changed?", [scene, S.add_patch(scene, 20, 20, 120, 120)]).json()
    assert all(i["georeferenced"] is False for i in j["inputs"])
    findings = j["diagnostics"]["geospatial_pair"]["findings"]
    assert findings[0]["code"] == "GEO_NO_GEOREFERENCING" and "no ground measurements" in findings[0]["message"]


def test_provenance_hashes_inputs_and_is_replayable(client, scene):
    b = S.add_patch(scene, 20, 20, 120, 120)
    j1 = post(client, "bitemporal", "What changed?", [scene, b]).json()
    j2 = post(client, "bitemporal", "What changed?", [scene, b]).json()
    import hashlib
    assert j1["provenance"]["inputs"][0]["sha256"] == hashlib.sha256(S.png_bytes(scene)).hexdigest()
    assert j1["analysis_id"] == j2["analysis_id"] and j1["request_id"] != j2["request_id"]
    assert j1["evidence"] == j2["evidence"] and j1["answer"] == j2["answer"]
    j3 = post(client, "bitemporal", "What changed?", [scene, b], options={"base_threshold": 40}).json()
    assert j3["analysis_id"] != j1["analysis_id"]


def test_misaligned_pair_is_not_reported_as_reliable(client, scene):
    j = post(client, "bitemporal", "What changed?", [scene, S.shift(scene, 12, 9)]).json()
    assert j["ok"] is True and j["status"] == "COMPLETED_WITH_WARNINGS"
    assert j["diagnostics"]["alignment"]["state"] == "UNRELIABLE"
    assert j["reliability"]["level"] == "UNSUPPORTED"
    assert "should NOT be relied on" in j["answer"]
    assert j["diagnostics"]["suitability"]["decision"] == "REQUEST_BETTER_IMAGERY"


def test_pair_with_different_dimensions_is_handled_and_disclosed(client, scene):
    small = np.asarray(Image.fromarray(scene).resize((200, 200)))
    j = post(client, "bitemporal", "What changed?", [scene, small]).json()
    assert j["ok"] is True
    assert j["provenance"]["working_grid"]["resampled"] == [False, True]
    assert j["diagnostics"]["alignment"]["aspect"]["same_dimensions"] is False


def test_distorted_aspect_ratio_is_flagged_unreliable(client, scene):
    wide = np.asarray(Image.fromarray(scene).resize((400, 200)))
    j = post(client, "bitemporal", "What changed?", [scene, wide]).json()
    assert j["diagnostics"]["alignment"]["state"] == "UNRELIABLE"
    assert j["reliability"]["level"] == "UNSUPPORTED"


def test_unusable_image_is_rejected(client, scene):
    flat = np.full((128, 128, 3), 90, np.uint8)
    r = post(client, "single", "Describe the scene", [flat])
    assert r.status_code == 422 and err(r) == "UNUSABLE_IMAGE"
    assert r.json()["diagnostics"]["suitability"]["decision"] == "REJECT"


def test_cross_mode_modality_plausibility(client, scene):
    colour_as_sar = S.scene((256, 256), seed=3)
    j = post(client, "cross", "Use optical and SAR together to identify built-up and water-covered regions.", [scene, colour_as_sar]).json()
    assert any(f["name"] == "sar_modality" for f in j["reliability"]["factors"])
    assert j["diagnostics"]["alignment"]["state"] == "NOT_ASSESSED"
    assert {l["id"] for l in j["layers"]} == {"optical", "sar", "fused"}


def test_options_are_honoured(client, scene):
    b = S.add_patch(scene, 20, 20, 120, 120)
    j = post(client, "bitemporal", "What changed?", [scene, b], options={"grid_size": 6, "thresholds": [10, 28, 60], "base_threshold": 28}).json()
    assert j["evidence"]["spatial_evidence"]["grid_size"] == 6 and len(j["evidence"]["spatial_evidence"]["cells"]) == 36
    assert [r["threshold"] for r in j["evidence"]["sensitivity"]["thresholds"]] == [10, 28, 60]


def test_diagnostics_endpoint(client, scene):
    r = client.post("/api/diagnostics", json={"images": [S.png_data_url(scene), S.png_data_url(S.shift(scene, 5, 3))]})
    j = r.json()
    assert r.status_code == 200 and j["diagnostics"]["alignment"]["dx_px"] == pytest.approx(5, abs=0.3)
    assert len(j["inputs"]) == 2 and "layers" not in j
    r = client.post("/api/diagnostics", json={"images": [S.png_data_url(S.blur(scene, 5))]})
    assert r.json()["diagnostics"]["quality"][0]["levels"]["sharpness"] in ("LOW", "VERY_LOW")
    assert client.post("/api/diagnostics", json={"images": ["bad"]}).status_code == 400
