# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Wavelet leaders tool: three regularity readings on the dyadic pyramid in one pass, the
h-map and the fitter's table following ``method`` without a recompute, the per-level quantities
on the Scale slider, the refusals and the reading."""
from __future__ import annotations

import sys

import numpy as np
import pytest

pytest.importorskip("pywt")

from dynamix.core import dwt                                                    # noqa: E402
from dynamix.model.contracts import FIELD, HMAP                                # noqa: E402
from dynamix.model.device import get_device, keyed_params, validate_params     # noqa: E402

sys.path.insert(0, "tests")
from conftest import fbm2d, make_field, make_layer                             # noqa: E402


@pytest.fixture
def dev(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return get_device("wavelet_leaders")


@pytest.fixture(scope="module")
def field():
    return make_field(fbm2d(512, 0.5, seed=3))


SMALL = {"scale_min": 4, "scale_max": 32}


def test_the_tool_sits_in_regularity_and_produces_an_h_map(dev):
    assert dev.label == "Wavelet leaders" and dev.families == ("Regularity",)
    assert dev.consumes == ((FIELD,),) and HMAP in dev.produces
    assert [(o.name, o.lazy, o.per_level) for o in dev.outputs] == [
        ("leader", True, True), ("p_leader", True, True), ("wse", True, True), ("oscillation", True, True)]
    p = validate_params(dev, {})
    assert (p["wavelet"], p["scale_min"], p["scale_max"], p["gamma"], p["p"]) == ("sym4", 4, 64, 0.0, 2.0)
    assert (p["wse_rule"], p["wse_beta"], p["wse_depth"], p["wse_width"]) == ("dumeur", 0.25, 1, 1.0)
    assert (p["q_min"], p["q_max"], p["q_step"], p["method"], p["show"]) == (-3.0, 6.0, 0.5, "leaders", "h_map")
    keyed = keyed_params(dev, p)
    assert "method" not in keyed and "show" not in keyed and "q_min" not in keyed      # view / at default
    assert {"wavelet", "scale_min", "scale_max", "gamma", "p", "wse_rule"} <= set(keyed)
    assert dev.roi_margin(p) == dwt.border_cells("sym4") * 64
    assert dev.wavelet_profile(p, 16.0, 3).family == "sym4 ψ, level 4"
    assert dev.derived_reading("scale_min", 4, make_field(np.zeros((16, 16)))) is None


def test_the_run_carries_the_three_readings_their_tables_and_the_h_map(dev, field):
    p = validate_params(dev, dict(SMALL, wse_rule="fixed"))
    res = dev.compute(field, p)
    assert res["h_map"].shape == (512, 512) and res["r2_map"].shape == (512, 512)
    e = dwt.border_cells("sym4")
    assert np.isnan(res["h_map"][: e * 4]).all() and np.isfinite(res["h_map"][192:320, 192:320]).all()
    assert np.allclose(res["h_map"][192:320, 192:320].mean(), 0.5, atol=0.1)
    assert res["hd_kind"] == "structure" and res["hd_std"] is res["hd_cmax"] is res["hd_tables"]["leaders"]
    assert set(res["hd_tables"]) == {"leaders", "p-leaders", "wse", "coefficients"}
    assert list(res["scales"]) == [2.0, 4.0, 8.0, 16.0, 32.0] and res["chains"] == [] and res["extrema"] == []
    assert res["hd_source"] == "wavelet leaders (DWT, sym4)"
    for m in ("leaders", "p-leaders", "wse"):
        assert abs(res["_cumulants"][m]["original"]["c1"] - 0.5) < 0.1, m
        assert res["_admissibility"][m]["admissible"]
    rec = res["record"]
    assert rec["levels"] == {"j_min": 2, "J": 5} and rec["border_cells"] == e and rec["wse"]["rule"] == "fixed"
    assert rec["wse"]["schedule"][0] == [[0, 1.0], [1, 1.0]]
    assert any(r["band"] == "horizontal" and r["level"] == 4 and r["shift"] == [0, 1] for r in rec["registration"])
    assert res["warnings"] == []
    text = dev.reading(res, p)
    assert text.startswith("leaders c1 0.") and "p = 2 c1" in text and "WSE c1" in text and "4–32 px" in text
    assert "inadmissible" not in text and "γ" not in text


def test_switching_the_method_repaints_without_a_recompute_and_keeps_one_object_per_choice(dev, field):
    p = validate_params(dev, dict(SMALL, wse_rule="fixed"))
    res = dev.compute(field, p)
    assert dev.view(res, {"method": "leaders"}) is res
    v = dev.view(res, {"method": "p-leaders"})
    assert v is not res and v is dev.view(res, {"method": "p-leaders"})
    assert v["hd_std"] is res["hd_tables"]["p-leaders"] and v["hd_source"] == "wavelet p-leaders (DWT, sym4)"
    assert v["params"]["method"] == "p-leaders" and v["_quantity"] is res["_quantity"]
    assert not np.array_equal(np.nan_to_num(v["h_map"]), np.nan_to_num(res["h_map"]))
    assert np.isfinite(v["h_map"][192:320, 192:320]).all()
    assert dev.view({"values": 1}, {"method": "wse"}) == {"values": 1}


def test_the_per_level_outputs_paint_the_cells_and_blank_the_border(dev, field):
    p = validate_params(dev, dict(SMALL, wse_rule="fixed"))
    res = dev.compute(field, p)
    for name, key in (("leader", "leaders"), ("p_leader", "p-leaders"), ("wse", "wse"), ("oscillation", "oscillation")):
        out = dev.compute_output(name, field.values, res, p, fetch=None, level=2)
        raster = out["raster"]
        assert raster.shape == (512, 512) and raster.dtype == np.float32
        assert out["diag"]["level"] == 2 and out["diag"]["scale_px"] == 8.0
        assert out["diag"]["valid_cells"] == int(np.sum(res["_valid"][key][2]))
        assert np.isnan(raster[:8]).all() and np.isfinite(raster[256, 256])
        # one cell = one 8 × 8 block
        assert np.all(raster[256:264, 256:264] == raster[256, 256])
    lead = dev.compute_output("leader", field.values, res, p, fetch=None, level=2)["raster"]
    cell = res["_quantity"]["leaders"][2][32, 32]
    assert lead[256, 256] == pytest.approx(np.log2(cell), rel=1e-6)
    osc = dev.compute_output("oscillation", field.values, res, p, fetch=None, level=2)["raster"]
    assert np.nanmin(osc) >= -1e-6                                     # the fixed window lies inside 3λ
    assert dev.compute_output("leader", field.values, res, p, fetch=None, level=99)["diag"]["level"] == 4
    with pytest.raises(ValueError, match="no lazy output"):
        dev.compute_output("edges", field.values, res, p, fetch=None)


def test_the_engine_resolves_the_tool_and_its_levels(dev, field):
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve, resolve_output

    cache = Cache()
    layer = make_layer(("wavelet_leaders", dict(SMALL, wse_rule="fixed")))
    res = resolve(layer, field, cache).result
    assert res["h_map"].shape == (512, 512) and res["hd_kind"] == "structure"
    a = resolve_output(layer, field, cache, "leader", level=1)
    b = resolve_output(layer, field, cache, "leader", level=3)
    assert a["diag"]["scale_px"] == 4.0 and b["diag"]["scale_px"] == 16.0
    assert not np.array_equal(np.nan_to_num(a["raster"]), np.nan_to_num(b["raster"]))
    again = make_layer(("wavelet_leaders", dict(SMALL, wse_rule="fixed", method="wse")))
    v = resolve(again, field, cache).result
    assert v["params"]["method"] == "wse" and v["_quantity"] is res["_quantity"]      # a cache hit, viewed


def test_an_inadmissible_field_is_named_in_the_reading_and_the_warnings(dev):
    f = fbm2d(512, 0.5, seed=2)
    dxf = make_field(np.roll(f, -1, axis=1) - f)
    p = validate_params(dev, dict(SMALL, wse_rule="fixed"))
    res = dev.compute(dxf, p)
    assert len(res["warnings"]) == 2 and all("inadmissible" in w for w in res["warnings"])
    text = dev.reading(res, p)
    assert "leaders c1" in text and "(inadmissible, γ ≥ 1)" in text and "WSE c1 -0." in text
    lifted = dev.compute(dxf, validate_params(dev, dict(SMALL, wse_rule="fixed", gamma=1.0)))
    assert lifted["warnings"] == [] and "γ = 1, original frame" in dev.reading(lifted, p)
    assert abs(lifted["_cumulants"]["leaders"]["original"]["c1"] + 0.5) < 0.12


def test_refusals_name_the_field_the_border_and_the_scale_that_fits(dev):
    with pytest.raises(ValueError, match="power of two"):
        dev.check(validate_params(dev, {"scale_min": 6, "scale_max": 32}))
    with pytest.raises(ValueError, match="at least three fitted levels"):
        dev.check(validate_params(dev, {"scale_min": 16, "scale_max": 32}))
    with pytest.raises(ValueError, match="q max"):
        dev.check(validate_params(dev, {"q_min": 2.0, "q_max": 1.0}))
    small = make_field(fbm2d(128, 0.5, seed=1))
    with pytest.raises(ValueError, match="128 × 128 px field .* fits three levels is 8 px"):
        dev.compute(small, validate_params(dev, {"scale_min": 4, "scale_max": 32}))
    with pytest.raises(ValueError, match="64 × 64 px field .* no scale_max fits"):
        dev.compute(make_field(fbm2d(64, 0.5, seed=1)), validate_params(dev, {"scale_min": 4, "scale_max": 16}))
    with pytest.raises(ValueError, match="place it FIRST"):
        dev.compute({"h_map": None}, validate_params(dev, SMALL))
    with pytest.raises(ValueError, match="scalar 2-D field"):
        dev.compute(make_field(np.zeros((64, 64, 3))), validate_params(dev, SMALL))


def test_non_finite_pixels_blank_their_cells_and_a_hole_is_counted(dev):
    f = fbm2d(512, 0.5, seed=4)
    f[20:30, 20:30] = np.nan
    p = validate_params(dev, dict(SMALL, wse_rule="fixed"))
    res = dev.compute(make_field(f), p)
    assert np.isnan(res["h_map"][20:30, 20:30]).all() and np.isfinite(res["h_map"][240:340, 240:340]).all()
    assert res["_valid_cells"]["leaders"][1] < (512 // 4 - 2 * dwt.border_cells("sym4")) ** 2
    assert res["warnings"] == []


def test_curves_offer_the_cumulant_lines_per_method(dev, field):
    from dynamix.model.curves import curves_of

    p = validate_params(dev, dict(SMALL, wse_rule="fixed"))
    res = dev.compute(field, p)
    curves = curves_of(dev, res, field, p)
    assert [c.name for c in curves][:2] == ["C₁(a) of ln leaders", "C₂(a) of ln leaders"] and len(curves) == 6
    assert curves[0].logx and list(curves[0].x) == [2.0, 4.0, 8.0, 16.0, 32.0]
    assert curves[0].marks == ((4.0, "finest fitted"), (32.0, "coarsest fitted"))
    assert curves_of(dev, {"values": 1}, field, p) == []


def test_dither_perturbs_quantised_data_and_is_keyed(dev):
    f = np.round(fbm2d(512, 0.5, seed=5) * 4) / 4
    p0 = validate_params(dev, dict(SMALL, wse_rule="fixed"))
    p1 = validate_params(dev, dict(SMALL, wse_rule="fixed", dither=True))
    assert keyed_params(dev, p1)["dither"] is True
    a, b = dev.compute(make_field(f), p0), dev.compute(make_field(f), p1)
    assert not np.array_equal(np.nan_to_num(a["h_map"]), np.nan_to_num(b["h_map"]))
    assert b["record"]["dither"] is True
