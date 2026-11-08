"""Image quality, alignment and suitability diagnostics on synthetic imagery."""
import numpy as np
import pytest

from backend import diagnostics as D
from tests import synthetic as S
from tests.conftest import gray


def quality(a, size=None):
    return D.assess_image_quality(a, size or (a.shape[1], a.shape[0]))


def codes(q):
    return {f["code"] for f in q["flags"]}


# ── quality ──────────────────────────────────────────────────────────────

def test_clean_scene_is_good(scene):
    q = quality(scene)
    assert q["overall"] == "GOOD" and q["flags"] == []


def test_blurred_image_raises_sharpness_warning(scene):
    sharp, blurred = quality(scene), quality(S.blur(scene, 4))
    assert blurred["metrics"]["sharpness_laplacian_variance"] < sharp["metrics"]["sharpness_laplacian_variance"] / 20
    assert blurred["levels"]["sharpness"] in ("LOW", "VERY_LOW")
    assert any(c.startswith("SHARPNESS_") for c in codes(blurred))
    assert "Small surface changes may be difficult to distinguish." in " ".join(blurred["analysis_implications"])


def test_overexposed_image_raises_clipping_warning(scene):
    q = quality(S.brighten(scene, 150))
    assert q["levels"]["clipping"] in ("HIGH", "SEVERE")
    assert any(c.startswith("CLIPPING_") for c in codes(q))
    assert q["metrics"]["clipped_bright_pct"] > 20


def test_dark_and_bright_levels(scene):
    assert quality(S.brighten(scene, -150))["levels"]["brightness"] == "DARK"
    assert quality(np.clip(scene.astype(int) + 190, 0, 255).astype(np.uint8))["levels"]["brightness"] == "BRIGHT"


def test_noise_is_estimated_and_flagged(scene):
    clean, noisy = quality(scene), quality(S.add_noise(scene, 15))
    assert noisy["metrics"]["noise_sigma_estimate"] > clean["metrics"]["noise_sigma_estimate"] + 5
    assert noisy["levels"]["noise"] in ("ELEVATED", "HIGH")


def test_flat_image_is_unusable():
    flat = np.full((128, 128, 3), 120, np.uint8)
    q = quality(flat)
    assert q["overall"] == "UNUSABLE"
    assert D.assess_suitability([q])["decision"] == "REJECT"


def test_low_resolution_flagged(scene):
    small = scene[:100, :100]
    q = D.assess_image_quality(small, (100, 100))
    assert q["levels"]["resolution"] == "VERY_LOW"
    assert D.assess_suitability([q])["decision"] == "REQUEST_BETTER_IMAGERY"
    assert D.assess_image_quality(scene, (200, 200))["levels"]["resolution"] == "LOW"


def test_resolution_uses_original_not_working_size(scene):
    q = D.assess_image_quality(scene, (4000, 3000))
    assert q["levels"]["resolution"] == "ADEQUATE"
    assert q["metrics"]["original_width"] == 4000 and q["metrics"]["working_width"] == 256


def test_grayscale_detection(scene):
    g = np.repeat(scene[..., :1], 3, axis=2)
    assert quality(g)["metrics"]["is_grayscale"] is True
    assert quality(scene)["metrics"]["is_grayscale"] is False


def test_suitability_decisions(scene):
    good = quality(scene)
    assert D.assess_suitability([good])["decision"] == "PROCEED"
    assert D.assess_suitability([quality(S.blur(scene, 4))])["decision"] == "PROCEED_WITH_WARNINGS"
    assert D.assess_suitability([quality(S.brighten(scene, 200))])["decision"] in ("REQUEST_BETTER_IMAGERY", "PROCEED_WITH_WARNINGS")
    dec = D.assess_suitability([good])
    assert "rule" in dec and dec["reasons"] == []


# ── alignment ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dx,dy", [(3, 5), (-4, 2), (8, -6), (0, 7), (-2, -2)])
def test_translation_is_recovered_with_correct_sign(scene, dx, dy):
    r = D.estimate_alignment(gray(scene), gray(S.shift(scene, dx, dy)))
    assert r["dx_px"] == pytest.approx(dx, abs=0.2)
    assert r["dy_px"] == pytest.approx(dy, abs=0.2)
    assert r["limitation"].startswith("This diagnostic estimates visual translation only")


def test_identical_images_are_good(scene):
    r = D.estimate_alignment(gray(scene), gray(scene.copy()))
    assert r["state"] == "GOOD"
    assert r["magnitude_px"] == pytest.approx(0, abs=0.05)
    assert r["valid_overlap_fraction"] == 1.0
    assert r["peak_correlation"] > 0.9


def test_small_translation_warns_large_is_unreliable(scene):
    assert D.estimate_alignment(gray(scene), gray(S.shift(scene, 1, 0)))["state"] == "WARNING"
    big = D.estimate_alignment(gray(scene), gray(S.shift(scene, 12, 9)))
    assert big["state"] == "UNRELIABLE" and big["magnitude_px"] > 10


def test_genuine_local_change_does_not_break_alignment(scene):
    changed = S.add_patch(scene, 100, 100, 150, 150)
    r = D.estimate_alignment(gray(scene), gray(changed))
    assert r["state"] == "GOOD" and r["magnitude_px"] < 0.5


def test_unrelated_scenes_are_unreliable(scene):
    other = S.scene((256, 256), seed=99)
    r = D.estimate_alignment(gray(scene), gray(other))
    assert r["state"] in ("WARNING", "UNRELIABLE")
    assert any("cannot be confirmed" in x or "large" in x or "similarity" in x for x in r["reasons"])


def test_textureless_input_cannot_be_aligned(scene):
    flat = np.full(scene.shape, 100, np.uint8)
    r = D.estimate_alignment(gray(flat), gray(scene))
    assert r["state"] == "UNRELIABLE" and r["dx_px"] is None


def test_mismatched_grid_is_an_error_not_a_guess(scene):
    with pytest.raises(ValueError):
        D.estimate_alignment(gray(scene), gray(scene[:100]))


def test_aspect_check_flags_distortion():
    ok = D.aspect_check((800, 600), (800, 600))
    assert ok["state"] == "GOOD" and ok["same_dimensions"]
    assert D.aspect_check((800, 600), (810, 600))["state"] in ("GOOD", "WARNING")
    assert D.aspect_check((800, 600), (600, 800))["state"] == "UNRELIABLE"


def test_aspect_mismatch_propagates_into_alignment(scene):
    aspect = D.aspect_check((800, 600), (600, 800))
    r = D.estimate_alignment(gray(scene), gray(scene), aspect=aspect)
    assert r["state"] == "UNRELIABLE"
    assert any("aspect ratios differ" in x for x in r["reasons"])


def test_alignment_unreliable_requests_better_imagery_for_change(scene):
    qs = [quality(scene), quality(scene)]
    al = D.estimate_alignment(gray(scene), gray(S.shift(scene, 12, 9)))
    assert D.assess_suitability(qs, al, requires_alignment=True)["decision"] == "REQUEST_BETTER_IMAGERY"
    assert D.assess_suitability(qs, al, requires_alignment=False)["decision"] == "PROCEED"
