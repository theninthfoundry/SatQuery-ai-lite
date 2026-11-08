"""Report parity, CSV safety, and equivalence of the vectorised legacy tools."""
import csv
import io
import math

import numpy as np

from backend import image_utils as U
from backend.report import _csv_safe, build_report
from tests import synthetic as S


def analyse(client, scene, **opts):
    b = S.add_patch(scene, 60, 40, 150, 130)
    r = client.post("/api/analyze", json={"mode": "bitemporal", "query": "What changed?", "options": opts,
                                         "images": [S.png_data_url(scene), S.png_data_url(b)]})
    return r.json()


def test_report_is_derived_from_the_result(client, scene):
    j = analyse(client, scene)
    rep = j["report"]
    assert j["answer"] in rep["markdown"]
    for s in j["evidence"]["statements"]:
        assert s["text"] in rep["markdown"]
    titles = [s["title"] for s in rep["sections"]]
    for needed in ("Analysis summary", "Inputs", "Image quality", "Temporal alignment", "Findings and evidence",
                   "Reliability factors", "Methodology", "Limitations", "Provenance"):
        assert needed in titles
    assert j["analysis_id"] in rep["markdown"]


def test_regions_csv_matches_the_evidence_exactly(client, scene):
    j = analyse(client, scene, grid_size=5)
    rows = list(csv.DictReader(io.StringIO(j["report"]["regions_csv"])))
    cells = j["evidence"]["spatial_evidence"]["cells"]
    assert len(rows) == len(cells) == 25
    for row, c in zip(rows, cells):
        assert row["region_id"] == c["region_id"]
        assert int(row["changed_pixels"]) == c["changed_pixel_count"]
        assert float(row["change_pct"]) == c["change_percentage"]
        assert row["evidence_strength"] == c["evidence_strength"]
    assert sum(int(r["changed_pixels"]) for r in rows) == j["evidence"]["measurements"]["changed_pixels"]
    assert sum(int(r["analysed_pixels"]) for r in rows) == j["evidence"]["measurements"]["analysed_pixels"]


def test_sensitivity_csv_matches_the_evidence(client, scene):
    j = analyse(client, scene, thresholds=[10, 28, 70])
    rows = list(csv.DictReader(io.StringIO(j["report"]["sensitivity_csv"])))
    assert [int(r["threshold"]) for r in rows] == [10, 28, 70]
    for row, t in zip(rows, j["evidence"]["sensitivity"]["thresholds"]):
        assert int(row["changed_pixels"]) == t["changed_pixels"]


def test_displayed_area_equals_report_area(client, scene):
    j = analyse(client, scene)
    pct = j["evidence"]["measurements"]["changed_pct"]
    assert f"{pct:.1f}%" in j["answer"]
    assert f"{pct:.1f}%" in j["report"]["markdown"]
    layer_ids = {l["id"] for l in j["layers"]}
    assert "change_mask" in layer_ids


def test_csv_formula_injection_guard():
    for bad in ("=1+1", "+cmd", "-2", "@SUM(A1)"):
        assert _csv_safe(bad).startswith("'")
    assert _csv_safe("normal") == "normal" and _csv_safe(5) == 5


def test_single_image_report_has_no_change_sections(client, scene):
    j = client.post("/api/analyze", json={"mode": "single", "query": "Describe the scene", "images": [S.png_data_url(scene)]}).json()
    assert j["report"]["regions_csv"] == "" and j["report"]["sensitivity_csv"] == ""
    assert "Temporal alignment" not in [s["title"] for s in j["report"]["sections"]]


# ── legacy tools: vectorised == original per-pixel loops ─────────────────

def test_vectorised_sar_structure_matches_reference_loop():
    pixels = S.scene((96, 80), 3)
    h, w = pixels.shape[:2]
    gray = pixels.astype(np.float32).mean(axis=2)
    edge = np.zeros((h, w), np.uint8)
    dark = np.zeros((h, w), np.uint8)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            r = gray[y - 1:y + 2, x - 1:x + 2]
            gx = -r[0, 0] + r[0, 2] - 2 * r[1, 0] + 2 * r[1, 2] - r[2, 0] + r[2, 2]
            gy = -r[0, 0] - 2 * r[0, 1] - r[0, 2] + r[2, 0] + 2 * r[2, 1] + r[2, 2]
            m = math.sqrt(gx * gx + gy * gy)
            edge[y, x] = m > 60
            dark[y, x] = gray[y, x] < 70 and m < 25
    s = U.compute_sar_structure(pixels)
    assert (s.edge == edge.ravel()).all() and (s.dark == dark.ravel()).all()


def test_vectorised_grid_counts_are_consistent():
    a = S.scene((90, 60), 1)
    b = S.add_patch(a, 10, 5, 40, 30)
    res = U.compute_change_mask(a, b)
    assert sum(res.grid) == int(res.mask.sum())
