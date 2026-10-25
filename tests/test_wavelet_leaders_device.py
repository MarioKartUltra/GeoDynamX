# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Wavelet leaders tool: three regularity readings on the dyadic pyramid in one pass, the
h-map following ``method`` without a recompute, the per-level quantities on the Scale slider,
the sample sets it publishes for the fitting tools (nothing is fitted here), the refusals and the
reading."""
from __future__ import annotations

import sys

import numpy as np
import pytest

pytest.importorskip("pywt")

from dynamix.core import dwt, leaders                                           # noqa: E402
from dynamix.engine.resolve import output_raster                                # noqa: E402
from dynamix.model.contracts import FIELD, HMAP, SAMPLES                       # noqa: E402
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


SMALL = {"scale_min": 4, "scale_max": 32, "with_wse": True}          # the three quantities
Q = np.arange(-2.0, 4.01, 1.0)


def _c1(res, method="leaders"):
    """What a fitting tool reads off the published sample sets: c1 over the fitted levels."""
    lev = res["record"]["levels"]
    table = leaders.tables(res["samples"], Q)[method]
    return leaders.cumulants(table, lev["j_min"], lev["J"], res["params"]["gamma"])["original"]["c1"]


def test_the_tool_sits_in_regularity_and_produces_an_h_map(dev):
    assert dev.label == "Wavelet leaders" and dev.families == ("Regularity",)
    assert dev.consumes == ((FIELD,),) and dev.produces == (HMAP, SAMPLES)
    assert [(o.name, o.lazy, o.per_level) for o in dev.outputs] == [
        ("leader", True, True), ("p_leader", True, True), ("wse", True, True), ("oscillation", True, True)]
    p = validate_params(dev, {})
    assert (p["wavelet"], p["scale_min"], p["scale_max"], p["gamma"], p["p"]) == ("sym4", 4, 64, 0.0, 2.0)
    assert (p["wse_rule"], p["wse_beta"], p["wse_depth"], p["wse_width"]) == ("dumeur", 0.25, 1, 1.0)
    assert (p["method"], p["show"]) == ("leaders", "h_map")
    assert "q_min" not in p and "directional" not in p                 # the fitting tools own the moments
    keyed = keyed_params(dev, p)
    assert "method" not in keyed and "show" not in keyed                              # view-only
    assert {"wavelet", "scale_min", "scale_max", "gamma", "p", "wse_rule"} <= set(keyed)
    # A region reads the bank's border in coarsest cells, or the WSE's reach when it is ticked and
    # wider (level j reaches j_r + D cells, j_r = log2 N − j: 5 cells at 64 px on a 1024-px field,
    # beyond the border less one), plus the partial cell its grid may start with.
    big = make_field(np.zeros((1024, 1024)))
    assert dev.roi_margin(p, field=big) == (5 + 1) * 64
    assert dev.roi_margin({**p, "with_wse": True}, field=big) == (4 + 5) * 64 + 64
    assert dev.wavelet_profile(p, 16.0, 3).family == "sym4 ψ, level 4"
    assert dev.derived_reading("scale_min", 4, make_field(np.zeros((16, 16)))) is None


def test_the_run_carries_the_three_readings_their_sample_sets_and_the_h_map(dev, field):
    p = validate_params(dev, dict(SMALL, wse_rule="fixed"))
    res = dev.compute(field, p)
    assert res["h_map"].shape == (512, 512) and res["r2_map"].shape == (512, 512)
    e = dwt.border_cells("sym4")
    assert np.isnan(res["h_map"][: e * 4]).all() and np.isfinite(res["h_map"][192:320, 192:320]).all()
    assert np.allclose(res["h_map"][192:320, 192:320].mean(), 0.5, atol=0.1)
    assert list(res["scales"]) == [2.0, 4.0, 8.0, 16.0, 32.0] and res["chains"] == [] and res["extrema"] == []
    samples = res["samples"]
    assert set(samples["sets"]) == {"leaders", "p-leaders", "wse", "coefficients"}
    assert list(samples["scales"]) == [2.0, 4.0, 8.0, 16.0, 32.0]
    assert all(a == 4 * b for a, b in zip(samples["cells"], samples["cells"][1:]))   # one domain
    for m in ("leaders", "p-leaders", "wse"):
        assert samples["sets"][m]["values"] is res["_quantity"][m]                 # the float32 grids, no copy
        assert all(not (np.asarray(v) & ~w).any()                                 # statistics within the maps
                   for v, w in zip(samples["sets"][m]["valid"], res["_valid"][m]))
        assert abs(_c1(res, m) - 0.5) < 0.1, m
        assert res["_admissibility"][m]["admissible"]
    coef = samples["sets"]["coefficients"]
    assert coef["positive_only"] and coef["values"][0].dtype == np.float32
    assert coef["values"][0].shape == (3,) + coef["valid"][0].shape == (3, 512, 512)    # the mirrored working image
    # The method's set rides as the Partition functions window's structure table; nothing else is fitted here.
    assert res["hd_kind"] == "structure" and res["hd_std"] is res["hd_cmax"] and res["hd_std"]["set"] == "leaders"
    assert "_cumulants" not in res and "_directional" not in res
    rec = res["record"]
    assert (rec["levels"]["j_min"], rec["levels"]["J"]) == (2, 5) and rec["border_cells"] == e and rec["wse"]["rule"] == "fixed"
    assert rec["wse"]["schedule"][0] == [[0, 1.0], [1, 1.0]]
    assert any(r["band"] == "horizontal" and r["level"] == 4 and r["shift"] == [0, 1] for r in rec["registration"])
    assert "cumulants" not in rec and res["warnings"] == []
    text = dev.reading(res, p)
    assert text.startswith("4–32 px · h_min 0.") and "η(2) " in text and text.endswith("leaders, p = 2 and WSE admissible")
    assert "c1" not in text and "γ" not in text


def test_switching_the_method_repaints_without_a_recompute_and_keeps_one_object_per_choice(dev, field):
    p = validate_params(dev, dict(SMALL, wse_rule="fixed"))
    res = dev.compute(field, p)
    assert dev.view(res, {"method": "leaders"}) is res
    v = dev.view(res, {"method": "p-leaders"})
    assert v is not res and v is dev.view(res, {"method": "p-leaders"})
    assert v["params"]["method"] == "p-leaders" and v["_quantity"] is res["_quantity"] and v["samples"] is res["samples"]
    assert not np.array_equal(np.nan_to_num(v["h_map"]), np.nan_to_num(res["h_map"]))
    assert np.isfinite(v["h_map"][192:320, 192:320]).all()
    assert dev.view({"values": 1}, {"method": "wse"}) == {"values": 1}


def test_the_per_level_outputs_paint_the_cells_and_blank_the_border(dev, field):
    p = validate_params(dev, dict(SMALL, wse_rule="fixed"))
    res = dev.compute(field, p)
    for name, key in (("leader", "leaders"), ("p_leader", "p-leaders"), ("wse", "wse"), ("oscillation", "oscillation")):
        out = dev.compute_output(name, field.values, res, p, fetch=None, level=2)
        raster = output_raster(out)
        assert raster.shape == (512, 512) and raster.dtype == np.float32
        assert out["diag"]["level"] == 2 and out["diag"]["scale_px"] == 8.0
        assert out["diag"]["valid_cells"] == int(np.sum(res["_valid"][key][2]))
        assert np.isnan(raster[:8]).all() and np.isfinite(raster[256, 256])
        # one cell = one 8 × 8 block
        assert np.all(raster[256:264, 256:264] == raster[256, 256])
    lead = output_raster(dev.compute_output("leader", field.values, res, p, fetch=None, level=2))
    cell = res["_quantity"]["leaders"][2][32, 32]
    assert lead[256, 256] == pytest.approx(np.log2(cell), rel=1e-6)
    osc = output_raster(dev.compute_output("oscillation", field.values, res, p, fetch=None, level=2))
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
    assert res["h_map"].shape == (512, 512) and set(res["samples"]["sets"]) == {"leaders", "p-leaders", "wse", "coefficients"}
    a = resolve_output(layer, field, cache, "leader", level=1)
    b = resolve_output(layer, field, cache, "leader", level=3)
    assert a["diag"]["scale_px"] == 4.0 and b["diag"]["scale_px"] == 16.0
    assert not np.array_equal(np.nan_to_num(output_raster(a)), np.nan_to_num(output_raster(b)))
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
    assert text.endswith("leaders and p = 2 inadmissible, γ ≥ 1 would clear them") and "h_min -0." in text
    assert abs(_c1(res, "wse") + 0.5) < 0.12                      # the one method that reads it unlifted
    lifted = dev.compute(dxf, validate_params(dev, dict(SMALL, wse_rule="fixed", gamma=1.0)))
    assert lifted["warnings"] == [] and dev.reading(lifted, p).endswith("admissible · γ = 1, original frame")
    assert abs(_c1(lifted) + 0.5) < 0.12


def test_refusals_name_the_field_the_border_and_the_scale_that_fits(dev):
    with pytest.raises(ValueError, match="power of two"):
        dev.check({**validate_params(dev, {"scale_max": 32}), "scale_min": 6})
    with pytest.raises(ValueError, match="at least three fitted levels"):
        dev.check(validate_params(dev, {"scale_min": 16, "scale_max": 32}))
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


def test_the_tool_fits_nothing_itself(dev, field):
    """No tables, no cumulants, no curves: the Structure functions and Log-cumulants tools read
    the sample sets with their own moments and ranges."""
    from dynamix.model.curves import curves_of

    p = validate_params(dev, dict(SMALL, wse_rule="fixed"))
    res = dev.compute(field, p)
    assert curves_of(dev, res, field, p) == [] and not hasattr(dev, "curves")
    assert "tables" not in res and "cumulants" not in res["record"]


def test_dither_perturbs_quantised_data_and_is_keyed(dev):
    f = np.round(fbm2d(512, 0.5, seed=5) * 4) / 4
    p0 = validate_params(dev, dict(SMALL, wse_rule="fixed"))
    p1 = validate_params(dev, dict(SMALL, wse_rule="fixed", dither=True))
    assert keyed_params(dev, p1)["dither"] is True
    a, b = dev.compute(make_field(f), p0), dev.compute(make_field(f), p1)
    assert not np.array_equal(np.nan_to_num(a["h_map"]), np.nan_to_num(b["h_map"]))
    assert b["record"]["dither"] is True


def test_auto_edge_runs_periodic_on_an_extension_output_and_mirrors_otherwise(dev, field):
    from dynamix.devices.periodic_extension import PeriodicExtension
    from dynamix.model.device import defaults_for

    p = validate_params(dev, dict(SMALL, wse_rule="fixed"))
    assert p["edge"] == "auto" and "edge" in keyed_params(dev, p)
    plain = dev.compute(field, p)
    assert plain["record"]["edge"] == {"asked": "auto", "used": "mirror", "border_cells": dwt.border_cells("sym4"),
                                       "block": None, "origin": [0, 0]}
    assert dev.derived_reading("edge", "auto", field).startswith("→ mirror at the data's edge")
    ext = PeriodicExtension().compute(field, defaults_for(PeriodicExtension()))
    stamp = ext.provenance["edges.filled"]
    res = dev.compute(ext, p)
    assert res["record"]["edge"]["used"] == "periodic" and res["record"]["edge"]["border_cells"] == 0
    assert res["record"]["edge"]["block"] == [512, 512] and res["record"]["edge"]["origin"] == [stamp["row0"], stamp["col0"]]
    reading = dev.derived_reading("edge", "auto", ext, p)
    assert reading.startswith("→ periodic (Periodic extension output)")
    assert (f"no multiple of {2 ** 5}" in reading) == (any(n % 32 for n in ext.values.shape))
    assert res["record"]["seam"] == bool(any(n % 32 for n in ext.values.shape))
    # no border: every cell of the block counts, so more cells than the mirrored run kept
    assert all(a > b for a, b in zip(res["_valid_cells"]["leaders"], plain["_valid_cells"]["leaders"]))
    assert res["h_map"].shape == ext.values.shape
    assert abs(_c1(res) - _c1(plain)) < 0.05
    forced = dev.compute(ext, validate_params(dev, dict(SMALL, wse_rule="fixed", edge="mirror")))
    assert forced["record"]["edge"]["used"] == "mirror" and forced["record"]["edge"]["block"] == [512, 512]
    assert dev.derived_reading("edge", "mirror", ext) is None


def test_the_dyadic_result_carries_structure_tables_for_the_partition_functions_window(dev, field):
    """The dyadic leaders open the Partition functions window like the CWT leaders: the set the
    method names becomes one structure-function table at the shared moment orders, marked
    dyadic so the windows label the scale axis j."""
    res = dev.compute(field, validate_params(dev, dict(SMALL, method="p-leaders")))
    assert res["hd_kind"] == "structure" and res["hd_std"] is res["hd_cmax"]
    assert res["samples"]["dyadic"] is True and res["hd_std"]["dyadic"] is True
    assert res["hd_std"]["set"] == "p-leaders" and res["hd_source"] == "dyadic p-leaders"
    np.testing.assert_allclose(res["hd_std"]["q_list"], leaders.TABLE_Q)
    expected = leaders.tables(res["samples"], leaders.TABLE_Q)["p-leaders"]
    np.testing.assert_allclose(res["hd_std"]["tau_qa"], expected["tau_qa"], equal_nan=True)


def test_thin_and_empty_coarse_levels_run_and_warn_instead_of_refusing(dev):
    """The pyramid reaches every level; a fitted level that keeps fewer than 16 cells inside the
    bank's border is thin, one with none carries no statistics, and both are warnings on the
    result and a ⚠ on the scale knob, never a refusal."""
    side = 128
    field = make_field(fbm2d(side, 0.5, seed=1))
    p = validate_params(dev, {"scale_min": 4, "scale_max": 32})
    e = dwt.border_cells(p["wavelet"])
    res = dev.compute(field, p)
    for j in range(2, 6):
        n = max(side // 2 ** j - 2 * e, 0) ** 2
        named = [w for w in res["warnings"] if w.startswith(f"level {j} ")]
        if n == 0:
            assert any("no cells" in w for w in named), (j, res["warnings"])
        elif n < 16:
            assert any(f"{n} cells" in w and "fewer than 16" in w for w in named), (j, res["warnings"])
        else:
            assert not any("fewer than 16" in w or "no cells" in w for w in named)
    assert "scale_max" in dev.knob_warnings(field, p) and "8 px" in dev.knob_warnings(field, p)["scale_max"]
    roomy = make_field(fbm2d(256, 0.5, seed=1))                         # 16 px keeps 36 cells inside the border
    assert dev.knob_warnings(roomy, validate_params(dev, {"scale_min": 4, "scale_max": 16})) == {}
    assert dev.knob_warnings(None, p) == {}
    tiny = dev.compute(make_field(fbm2d(64, 0.5, seed=1)), validate_params(dev, {"scale_min": 4, "scale_max": 16}))
    assert any("no cells" in w for w in tiny["warnings"])


def test_the_h_map_reaches_within_the_finest_levels_band_of_the_edge_and_says_how_many_levels(dev):
    """On a 256² field at the defaults no cell has its whole chain inside sym4's border, yet the
    map is finite wherever three levels are: 96 px square inside an 80-px band, with the result's
    levels map reading 3 everywhere (the 32-px level's band is 160 px)."""
    from conftest import fbm2d
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.model.device import defaults_for

    n = 256
    field = RasterField(name="fbm", values=fbm2d(n, 0.6, seed=2).astype(np.float32), frame=LocalFrame(),
                        x_axis=np.arange(float(n)), y_axis=np.arange(float(n)))
    p = dict(defaults_for(dev))
    res = dev.compute(field, p)
    fin = np.isfinite(res["h_map"])
    rows = np.where(fin.any(axis=1))[0]
    assert rows.min() == 80 and rows.max() == 175 and fin.mean() > 0.12
    assert res["levels_map"].shape == res["h_map"].shape
    assert res["levels_map"][80, 128] == 3 and res["levels_map"][128, 128] == 3
    assert np.isnan(res["levels_map"][79, 128])
    assert res["record"]["levels"]["used"] == {"min": 3, "max": 3}


def test_cycle_spin_averages_the_h_map_over_phases_of_the_grid_and_pools_the_samples(dev):
    """Spin s runs the leaders on s × s phases of the dyadic grid, spread so the finest fitted
    level's cells shift too; the h-map is the pixel-wise mean of the grids' maps, the samples are
    pooled, and spin 1 is the single grid of before."""
    from conftest import fbm2d
    from dynamix.core import leaders as L
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.model.device import defaults_for

    n = 256
    values = fbm2d(n, 0.6, seed=5).astype(np.float32)
    field = RasterField(name="fbm", values=values, frame=LocalFrame(), x_axis=np.arange(float(n)), y_axis=np.arange(float(n)))
    p = {**defaults_for(dev), "scale_max": 32}
    assert p["spin"] == 1
    single = dev.compute(field, p)
    assert "spin" not in single["record"]
    spun = dev.compute(field, {**p, "spin": 2})
    phases = [tuple(ph) for ph in spun["record"]["spin"]["phases"]]
    assert len(phases) == 4 and len(set(phases)) == 4 and phases[0] == tuple(single["record"]["grid"]["phase"])
    assert len({ph[0] % 4 for ph in phases}) > 1                                # the 4-px cells shift too
    maps = []
    for ph in phases:
        r = L.run_leaders(values.astype(np.float64), scale_min=4, scale_max=32, anchor=ph)
        h, _r2, _n = r["local"]["leaders"]
        maps.append(L.paint(h, r["j_min"], (n, n), pad=r["pad"]))
    with np.errstate(all="ignore"):
        expect = np.nanmean(np.stack(maps), axis=0)
    np.testing.assert_allclose(spun["h_map"], expect.astype(np.float32), rtol=1e-5, atol=1e-6, equal_nan=True)
    assert spun["samples"]["sets"]["leaders"]["pooled"] == 4
    assert spun["samples"]["cells"] == single["samples"]["cells"]
    sizes = [int(np.asarray(v).sum()) for v in single["samples"]["sets"]["leaders"]["valid"]]
    pooled = [np.asarray(v).size for v in spun["samples"]["sets"]["leaders"]["values"]]
    assert all(pz >= 3 * s for pz, s in zip(pooled, sizes))


def test_a_per_level_output_travels_as_its_decimated_cell_grid(dev, field):
    """A level of the pyramid is a raster at its own 2^j-px cells: the output carries that grid
    and its placement, so a cached level costs its cells, and :func:`output_raster` repeats them
    over their blocks only when something draws or reads them, giving the field-grid raster of
    before."""
    from dynamix.engine.cache import retained_nbytes

    p = validate_params(dev, dict(SMALL, wse_rule="fixed"))
    res = dev.compute(field, p)
    out = dev.compute_output("leader", field.values, res, p, fetch=None, level=3)
    assert "raster" not in out
    assert out["cells"].shape == res["_quantity"]["leaders"][3].shape and out["cells"].dtype == np.float32
    raster = output_raster(out)
    cells = np.where(res["_valid"]["leaders"][3], np.log2(res["_quantity"]["leaders"][3]), np.nan)
    np.testing.assert_array_equal(raster, leaders.paint(cells, 4, (512, 512), pad=res["_pad"]).astype(np.float32))
    assert retained_nbytes(out) < raster.nbytes / 50


def test_a_region_s_wse_is_complete_at_the_coarsest_level_and_equals_the_whole_field_s(dev):
    """The region's halo covers the WSE's reach at the coarsest level and its resolution index is
    the file's, so an interior region's (θ, ω)-leaders fill it at every level and equal the
    whole field's there."""
    from conftest import fbm2d
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.model.device import defaults_for
    from dynamix.roi.runner import roi_margin_of, run_on_region

    n = 1536
    f = RasterField(name="fbm", values=fbm2d(n, 0.6, seed=8).astype(np.float32), frame=LocalFrame(),
                    x_axis=np.arange(float(n)), y_axis=np.arange(float(n)))
    f.provenance.update({"window": {"row_off": 0, "col_off": 0}, "full_dims": (n, n)})
    p = {**defaults_for(dev), "scale_max": 32, "with_wse": True}
    margin, _ok = roi_margin_of(dev, p, field=f)
    assert margin == (4 + 7 + 1) * 32                                       # sym4: border 5 − 1, reach 7 at 32 px, one cell
    full = dev.compute(f, p)
    r0, c0, h, w = rect = (600, 600, 300, 300)
    roi = run_on_region([(dev, p)], f, rect)
    for level in range(1, 5):
        whole = output_raster(dev.compute_output("wse", None, full, p, fetch=None, level=level))[r0:r0 + h, c0:c0 + w]
        part = output_raster(dev.compute_output("wse", None, roi, p, fetch=None, level=level))
        assert np.isfinite(part).all(), level
        np.testing.assert_allclose(part, whole, rtol=1e-5, atol=1e-5)


def test_the_three_quantities_are_boxes_the_wse_off_by_default_and_the_tool_waits_for_run(dev):
    """Leaders and p-leaders are ticked by default and the WSE, which costs about as much as the
    transform, is not; only the ticked ones are computed, an unticked quantity's output says so,
    a region's halo covers the WSE only when it is ticked, and the tool waits for Run."""
    from conftest import fbm2d
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.model.device import defaults_for

    assert getattr(dev, "waits_for_run", False) is True
    p = dict(defaults_for(dev))
    assert (p["with_leaders"], p["with_p_leaders"], p["with_wse"]) == (True, True, False)
    assert {"with_leaders", "with_p_leaders", "with_wse"} <= set(keyed_params(dev, p))
    n = 256
    f = RasterField(name="fbm", values=fbm2d(n, 0.6, seed=4).astype(np.float32), frame=LocalFrame(),
                    x_axis=np.arange(float(n)), y_axis=np.arange(float(n)))
    p = {**p, "scale_max": 32}
    res = dev.compute(f, p)
    assert set(res["samples"]["sets"]) == {"leaders", "p-leaders", "coefficients"}
    assert dev.output_refusal("wse", p) and dev.output_refusal("oscillation", p) and dev.output_refusal("leader", p) is None
    with pytest.raises(ValueError, match="WSE leaders"):
        dev.compute_output("wse", None, res, p, fetch=None, level=1)
    big = make_field(np.zeros((1024, 1024)))
    assert dev.roi_margin({**p, "scale_max": 64}, field=big) == (5 + 1) * 64
    assert dev.roi_margin({**p, "scale_max": 64, "with_wse": True}, field=big) == (4 + 5) * 64 + 64
    with pytest.raises(ValueError, match="at least one"):
        dev.check({**p, "with_leaders": False, "with_p_leaders": False})
    shown = dev.compute(f, {**p, "with_leaders": False, "method": "leaders"})
    assert shown["params"]["method"] == "leaders" and any("not computed" in w for w in shown["warnings"])


def test_a_chain_saved_before_the_boxes_reopens_computing_all_three(dev):
    """A saved chain without the boxes computed every quantity: it reopens with all three
    ticked, while a chain that names them keeps its choice."""
    from dynamix.model.chain import Chain

    old = Chain.from_payload({"steps": [{"device": "wavelet_leaders", "params": {"scale_max": 32}}]})
    assert (old.steps[0].params["with_leaders"], old.steps[0].params["with_p_leaders"], old.steps[0].params["with_wse"]) == (True, True, True)
    new = Chain.from_payload({"steps": [{"device": "wavelet_leaders", "params": {"scale_max": 32, "with_wse": False}}]})
    assert new.steps[0].params["with_wse"] is False


def test_the_edge_reading_names_the_real_data_a_region_reads_around_it(dev):
    """Beside ``edge`` the reading says where the continuation applies and how much real data a
    region reads around itself first, the WSE's reach included when it is ticked."""
    from dynamix.model.device import defaults_for

    big = make_field(np.zeros((1024, 1024)))
    p = dict(defaults_for(dev))
    text = dev.derived_reading("edge", "auto", big, p)
    assert text.startswith("→ mirror") and "a region reads 384 px of real data around it" in text
    assert "640 px" in dev.derived_reading("edge", "auto", big, {**p, "with_wse": True})
