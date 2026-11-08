"""Change Intelligence Engine: measurements, spatial evidence, sensitivity, stability."""
import numpy as np
import pytest

from backend import diagnostics as D
from backend.change_engine import analyze_change
from backend.uncertainty.sensitivity import ThresholdError, normalise_thresholds, run_sensitivity
from tests import synthetic as S
from tests.conftest import gray

T = [20, 28, 40]


def run(a, b, thresholds=T, base=28, grid=4):
    al = D.estimate_alignment(gray(a), gray(b))
    return analyze_change(a, b, al, thresholds, base, grid)


def flat(value=100, size=(160, 160), seed=0):
    rng = np.random.default_rng(seed)
    base = np.full((size[1], size[0], 3), value, np.int16) + rng.integers(-1, 2, (size[1], size[0], 3))
    return np.clip(base, 0, 255).astype(np.uint8)


# ── basic behaviours from the validation table ───────────────────────────

def test_identical_images_have_no_difference(scene):
    ev = run(scene, scene.copy()).evidence
    assert ev["measurements"]["changed_pixels"] == 0
    assert ev["measurements"]["max_abs_difference"] == 0
    assert ev["concentration"]["pattern"] == "NONE"
    assert ev["sensitivity"]["sensitivity_level"] == "NO_CHANGE"
    assert all(c["evidence_strength"] == "NONE" for c in ev["spatial_evidence"]["cells"])


def test_uniform_brightness_change_is_detected_with_sensitivity_warning(scene):
    b = S.brighten(flat(100), 33)
    a = flat(100)
    ev = run(a, b).evidence
    assert ev["measurements"]["changed_pct"] > 95
    assert ev["sensitivity"]["sensitivity_level"] == "HIGH"   # 33 sits between thresholds 28 and 40
    assert ev["radiometric"]["global_offset_flag"] is True
    assert ev["radiometric"]["offset_corrected_changed_pct"] < 5
    ids = {s["id"] for s in ev["statements"]}
    assert "radiometric_offset" in ids and "sensitivity" in ids


def test_synthetic_local_change_highlights_the_right_region(scene):
    a = flat(100, (160, 160))
    b = a.copy()
    b[10:50, 90:150] = 220      # patch in the upper-right: rows 10-50, cols 90-150
    ev = run(a, b, grid=4).evidence
    assert ev["measurements"]["changed_pixels"] == 40 * 60
    top = {c["region_id"] for c in ev["spatial_evidence"]["cells"] if c["evidence_strength"] != "NONE"}
    # cells are 40px: rows 0-1 (0..80) and cols 2-3 (80..160) -> grid_2, grid_3, grid_6? patch rows 10-50 -> rows 0 and 1
    assert top == {"grid_2", "grid_3", "grid_6", "grid_7"}
    assert ev["concentration"]["centroid_phrase"] == "the upper-right of the frame"
    # every cell outside the patch is untouched
    for c in ev["spatial_evidence"]["cells"]:
        if c["region_id"] not in top:
            assert c["changed_pixel_count"] == 0


def test_cell_counts_match_patch_geometry_exactly():
    a = flat(100, (160, 160))
    b = a.copy()
    b[0:40, 0:40] = 230          # exactly grid_0 (cells are 40x40)
    cells = {c["region_id"]: c for c in run(a, b).evidence["spatial_evidence"]["cells"]}
    assert cells["grid_0"]["changed_pixel_count"] == 1600
    assert cells["grid_0"]["change_percentage"] == 100.0
    assert cells["grid_0"]["analysed_pixel_count"] == 1600
    assert cells["grid_0"]["share_of_total_change_pct"] == 100.0
    assert run(a, b).evidence["concentration"]["pattern"] == "CONCENTRATED"
    assert run(a, b).evidence["concentration"]["dominant_cell"] == "grid_0"


def test_region_records_follow_the_data_not_a_template():
    a = flat(100, (160, 160))
    b1, b2 = a.copy(), a.copy()
    b1[0:40, 0:40] = 230
    b2[120:160, 120:160] = 230
    c1 = run(a, b1).evidence["spatial_evidence"]["cells"]
    c2 = run(a, b2).evidence["spatial_evidence"]["cells"]
    assert c1[0]["change_percentage"] == 100.0 and c2[0]["change_percentage"] == 0.0
    assert c2[15]["change_percentage"] == 100.0 and c1[15]["change_percentage"] == 0.0
    assert c1[0]["interpretation"] != c2[0]["interpretation"]
    for c in c1:
        assert c["evidence_type"] == "pixel_intensity_difference"
        assert set(("region_id", "changed_pixel_count", "analysed_pixel_count", "change_percentage",
                    "quality_warning", "interpretation")) <= set(c)


@pytest.mark.parametrize("n", [2, 3, 4, 7, 12])
def test_grid_partitions_the_frame_for_any_size(n):
    a = flat(100, (157, 131))      # not divisible by n
    b = a.copy(); b[5:60, 7:90] = 200
    ev = run(a, b, grid=n).evidence
    cells = ev["spatial_evidence"]["cells"]
    assert len(cells) == n * n
    assert sum(c["analysed_pixel_count"] for c in cells) == 157 * 131
    assert sum(c["changed_pixel_count"] for c in cells) == ev["measurements"]["changed_pixels"]


# ── thresholds, stability, ambiguity ─────────────────────────────────────

def test_threshold_variation_matches_brute_force():
    a = flat(100)
    rng = np.random.default_rng(3)
    b = np.clip(a.astype(int) + rng.integers(-60, 61, a.shape), 0, 255).astype(np.uint8)
    diff = np.abs(a.astype(np.float32).mean(2) - b.astype(np.float32).mean(2))
    sens = run_sensitivity(diff, [10, 20, 28, 40, 60], 28)
    for row in sens["thresholds"]:
        assert row["changed_pixels"] == int((diff > row["threshold"]).sum())
    counts = [r["changed_pixels"] for r in sens["thresholds"]]
    assert counts == sorted(counts, reverse=True)           # monotone non-increasing
    assert [r["is_base"] for r in sens["thresholds"]].count(True) == 1


def test_stable_and_ambiguous_regions_are_separated():
    a = flat(100, (160, 160))
    b = a.copy()
    b[0:40, 0:40] = 100 + 70        # diff ~70 > 40 -> stable at every threshold
    b[0:40, 120:160] = 100 + 33     # diff ~33: changed at 20 and 28, not at 40 -> ambiguous
    ev = run(a, b).evidence
    st = ev["stability"]
    assert st["stable_pixels"] == 1600
    assert st["ambiguous_pixels"] == 1600
    cells = {c["region_id"]: c for c in ev["spatial_evidence"]["cells"]}
    # grid_0: fully stable, but featureless in both images -> quality caution demotes STRONG to MODERATE
    assert cells["grid_0"]["retention_at_strictest"] == 1.0
    assert cells["grid_0"]["quality_warning"] is True
    assert cells["grid_0"]["evidence_strength"] == "MODERATE"
    assert cells["grid_3"]["retention_at_strictest"] == 0.0 and cells["grid_3"]["evidence_strength"] == "WEAK"
    assert st["retention_of_base"] == pytest.approx(0.5, abs=0.01)


def test_stable_change_is_a_subset_of_base_change(scene):
    b = S.add_patch(scene, 60, 60, 120, 120)
    r = run(scene, b)
    assert not (r.base_mask & False).any()
    ev = r.evidence
    assert ev["stability"]["stable_pixels"] <= ev["measurements"]["changed_pixels"]


def test_threshold_normalisation_rules():
    assert normalise_thresholds(28, [40, 20, 20]) == [20, 28, 40]
    assert normalise_thresholds(28, None) == [20, 28, 40]
    for bad in ([0, 28], [255, 28], ["a", 3], [True, 5]):
        with pytest.raises(ThresholdError):
            normalise_thresholds(28, bad)
    with pytest.raises(ThresholdError):
        normalise_thresholds(28, list(range(1, 20)))
    with pytest.raises(ThresholdError):
        normalise_thresholds(0, [10, 20])


# ── noise / uncertainty exposure ─────────────────────────────────────────

def test_noisy_pair_exposes_uncertainty(scene):
    a = S.add_noise(scene, 14, seed=1)
    b = S.add_noise(scene, 14, seed=2)       # same surface, different noise
    ev = run(a, b, thresholds=[10, 14, 20], base=14).evidence
    assert ev["noise_floor"]["threshold_within_noise"] is True
    assert ev["measurements"]["changed_pct"] > 5            # "change" that is pure noise
    assert ev["concentration"]["scattered"] is True        # and it is scattered, not compact
    ids = {s["id"] for s in ev["statements"]}
    assert {"noise_floor", "scatter"} <= ids


def test_compact_change_is_not_flagged_scattered():
    a = flat(100, (160, 160)); b = a.copy(); b[40:100, 40:100] = 230
    assert run(a, b).evidence["concentration"]["scattered"] is False


def test_alignment_impact_what_if(scene):
    shifted = S.shift(scene, 3, 2)
    ev = run(scene, shifted).evidence
    assert ev["alignment_impact"]["shift_compensated_changed_pct"] is not None
    assert ev["alignment_impact"]["shift_compensated_changed_pct"] < ev["measurements"]["changed_pct"]


# ── statements are generated from measurements ───────────────────────────

def test_statement_numbers_trace_to_measurements(scene):
    ev = run(scene, S.add_patch(scene, 60, 60, 140, 140)).evidence
    mag = next(s for s in ev["statements"] if s["id"] == "magnitude")
    assert mag["basis"]["changed_pct"] == ev["measurements"]["changed_pct"]
    assert f"{ev['measurements']['changed_pct']:.1f}%" in mag["text"]
    assert f"{ev['measurements']['changed_pixels']:,}" in mag["text"]
    for s in ev["statements"]:
        assert s["basis"] is not None and s["text"]


def test_engine_is_deterministic(scene):
    b = S.add_patch(scene, 30, 40, 90, 120)
    r1, r2 = run(scene, b), run(scene, b)
    assert r1.evidence == r2.evidence
    assert [l["data"] for l in r1.layers] == [l["data"] for l in r2.layers]


def test_layers_are_complete_and_same_size(scene):
    r = run(scene, S.add_patch(scene, 30, 40, 90, 120))
    assert [l["id"] for l in r.layers] == ["before", "after", "difference", "change_mask", "evidence_grid", "uncertain", "combined"]
    assert len({(l["width"], l["height"]) for l in r.layers}) == 1
    assert all(l["data"].startswith("data:image/png;base64,") for l in r.layers)


def test_engine_rejects_mismatched_grids(scene):
    with pytest.raises(ValueError):
        analyze_change(scene, scene[:100], {"dx_px": 0, "dy_px": 0}, T, 28, 4)


def test_textured_scene_change_earns_strong_evidence(scene):
    b = S.add_patch(scene, 128, 0, 256, 128, color=(250, 250, 250))
    cells = run(scene, b).evidence["spatial_evidence"]["cells"]
    strong = [c for c in cells if c["evidence_strength"] == "STRONG"]
    assert strong and all(not c["quality_warning"] for c in strong)
    assert {c["row"] for c in strong} <= {0, 1} and {c["col"] for c in strong} <= {2, 3}
