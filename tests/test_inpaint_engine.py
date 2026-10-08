# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""One inpainting run: regions from every source, the three routes, guides on their own
grids, the veil, the refusals, the readouts and the stage store."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("numba")
pytest.importorskip("pywt")
pytest.importorskip("scipy")

from dynamix.core.frames import LocalFrame  # noqa: E402
from dynamix.core.inpaint import engine as en  # noqa: E402
from dynamix.core.inpaint.grid import georef, relate  # noqa: E402
from dynamix.core.rasterfield import RasterField  # noqa: E402
from dynamix.engine.cache import StageStore  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_microcanonical import _cascade2d  # noqa: E402

UTM = "EPSG:32615"


def _field(values, x0=500015.0, y0=3999985.0, step=30.0, name="t", crs=UTM, **prov):
    ny, nx = values.shape[:2]
    return RasterField(name=name, values=values, frame=LocalFrame(x0=x0, y0=y0, dx=step, dy=step,
                                                                   units="m"),
                       x_axis=x0 + step * np.arange(nx), y_axis=y0 - step * np.arange(ny),
                       provenance={"crs": crs, "source": f"/{name}.tif", **prov})


def _params(**over):
    from dynamix.devices.inpaint import PARAMS

    base = {p.name: p.default for p in PARAMS}
    base.update(over)
    return base


def _box(shape, r0, r1, c0, c1):
    m = np.zeros(shape, dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def _gapped_cascade(box=(48, 80, 48, 80)):
    dens, _ = _cascade2d()
    y = dens.copy()
    y[_box(y.shape, *box)] = np.nan
    return dens, y


def _guide(target, values, role, *, step=30.0, x0=500015.0, y0=3999985.0, name="g", **kw):
    g = _field(values, x0=x0, y0=y0, step=step, name=name)
    return en.Guide(name, role, g, relate(georef(target), georef(g)), **kw)


# ---------------------------------------------------------------- regions

def test_regions_come_from_nodata_threshold_polygons_and_a_sent_mask():
    y = np.random.default_rng(0).uniform(1.0, 2.0, (64, 64))
    y[5:10, 5:10] = np.nan
    y[30:40, 30:40] = 9.0                                    # bright: a cloud
    t = _field(y)
    ids, kinds = en.regions_from(y, _params(mask_source="nodata"))
    assert ids.max() == 1 and kinds == {1: "gap"} and (ids[5:10, 5:10] == 1).all()
    ids, kinds = en.regions_from(y, _params(mask_source="threshold", mask_op=">", mask_value=5.0,
                                            mask_min_px=16, threshold_kind="veil"))
    assert kinds == {1: "gap", 2: "veil"} and (ids[30:40, 30:40] == 2).all()
    assert ids[5, 5] == 1                                    # nodata is always a gap region
    poly = json.dumps([{"kind": "gap", "points": [[50, 50], [60, 50], [60, 60], [50, 60]]}])
    ids, kinds = en.regions_from(y, _params(mask_source="region", _mask_polygons=poly))
    assert kinds == {1: "gap", 2: "gap"} and ids[55, 55] == 2 and ids[49, 55] == 0
    ids, kinds = en.regions_from(y, _params(mask_source="region", _mask_polygons=poly),
                                 window_offset=(40, 24))     # the polygon is in FILE pixels
    assert ids[15, 31] == 2 and ids[31, 15] == 0 and ids[55, 55] == 0
    m = np.zeros((64, 64))
    m[2:6, 50:60] = 1.0
    send = _guide(t, m, "mask", name="cloudmask")
    ids, kinds = en.regions_from(y, _params(mask_source="send", region_kind="veil"), guides=[send],
                                 target_georef=georef(t))
    assert kinds == {1: "gap", 2: "veil"} and ids[3, 55] == 2
    ids, kinds = en.regions_from(y, _params(mask_source="all", mask_op=">", mask_value=5.0,
                                            _mask_polygons=poly, region_kind="veil"),
                                 guides=[send], target_georef=georef(t))
    assert ids.max() == 4 and sorted(kinds.values()) == ["gap", "gap", "gap", "veil"]


def test_nodata_stays_a_gap_under_a_veil_polygon():
    y = np.random.default_rng(4).uniform(1.0, 2.0, (64, 64))
    y[12:16, 12:16] = np.nan
    poly = json.dumps([{"kind": "veil", "points": [[8, 8], [24, 8], [24, 24], [8, 24]]}])
    ids, kinds = en.regions_from(y, _params(mask_source="region", _mask_polygons=poly))
    assert kinds[int(ids[13, 13])] == "gap" and kinds[int(ids[10, 10])] == "veil"
    assert sorted(kinds.values()) == ["gap", "veil"]
    res = en.run_inpaint(_field(y), y, _params(route="sparsity", mask_source="region",
                                               _mask_polygons=poly, n_iter=2, heldout_blocks=0), [])
    assert res["_regions"] == {"gap": 1, "veil": 1}


def test_a_threshold_on_another_band_of_the_stack_and_small_components_dropped():
    rng = np.random.default_rng(1)
    stack = rng.uniform(1.0, 2.0, (32, 32, 3))
    stack[10:20, 10:20, 2] = 5.0
    stack[0:2, 0:2, 2] = 5.0                                 # 4 px: under mask_min_px
    ids, kinds = en.regions_from(stack[..., 0], _params(mask_source="threshold", mask_band="b3",
                                                        mask_op=">", mask_value=4.0, mask_min_px=16),
                                 stack=stack, bands=["b1", "b2", "b3"])
    assert ids.max() == 1 and (ids[10:20, 10:20] == 1).all() and ids[0, 0] == 0


def test_the_mask_send_on_a_nested_grid_is_replicated_by_nearest_neighbour():
    dens, _ = _cascade2d()
    t = _field(dens)
    m = np.zeros((64, 64))
    m[20:30, 20:30] = 1.0
    send = _guide(t, m, "mask", step=60.0, x0=500015.0 + 15.0, y0=3999985.0 - 15.0, name="coarse_mask")
    ids, kinds = en.regions_from(dens, _params(mask_source="send"), guides=[send],
                                 target_georef=georef(t))
    assert (ids[40:60, 40:60] == 1).all() and ids[39, 50] == 0 and ids[60, 50] == 0


# ---------------------------------------------------------------- the routes

def test_no_region_returns_the_data_and_says_so():
    y = np.random.default_rng(2).uniform(1.0, 2.0, (32, 32))
    res = en.run_inpaint(_field(y), y, _params(), [])
    np.testing.assert_array_equal(res["raster_out"], y)
    assert res["_stop"] == "no region" and res["fill_mask"].max() == 0
    assert res["chains"] == [] and res["extrema"] == [] and res["_filled_fraction"] == 0.0


@pytest.mark.parametrize("route", ["sparsity", "cascade", "both"])
def test_each_route_fills_the_gap_keeps_the_data_and_reports(route):
    dens, y = _gapped_cascade()
    t = _field(y)
    p = _params(route=route, heldout_blocks=4, heldout_side=8, seed=3, n_iter=0, level=0,
                wavelet="haar")
    res = en.run_inpaint(t, y, p, [])
    known = np.isfinite(y)
    np.testing.assert_array_equal(res["raster_out"][known], y[known])
    assert np.isfinite(res["raster_out"]).all()
    assert (res["fill_mask"] > 0).sum() == 32 * 32 and res["_filled_fraction"] == pytest.approx(1024 / 16384)
    assert res["_route"] == route and res["_seed"] == 3
    assert res["_stop"] in ("converged", "cap") or res["_stop"].startswith("diverged")
    assert res["_iterations"] >= res["_returned"] >= 1
    assert res["_heldout"]["n_blocks"] == 4 and np.isfinite(res["_heldout"]["rms_rel"])
    assert res["_heldout_mask"].sum() == 4 * 64 and not (res["_heldout_mask"] & ~known).any()
    assert np.isfinite(res["_variance_error"])
    assert set(res["_eta_distance"]) <= set(res["_amp_distance"]) and len(res["_eta_distance"]) >= 6
    assert res["_fill_kind"] == "estimate" and res["_levels"] >= 3
    assert res["_estimate"].dtype == np.float32 and res["_estimate"].shape == y.shape
    inside = res["fill_mask"] > 0
    assert abs(res["_estimate"][inside].mean() - dens[inside].mean()) < 0.5 * dens[inside].std()
    again = en.run_inpaint(t, y, p, [])
    np.testing.assert_array_equal(again["raster_out"], res["raster_out"])   # reproducible


def _cascade_amp(**over):
    dens, y = _gapped_cascade()
    t = _field(y)
    res = en.run_inpaint(t, y, _params(route="cascade", heldout_blocks=0, level=0, wavelet="haar",
                                       **over), [])
    amp = res["_amp_distance"]
    assert amp and all(0.0 <= v <= 1.0 for v in amp.values())   # levels with both sides only
    assert res["_abacus"] is not None and res["_abacus"]["hist"].shape[0] == 3
    assert res["_regions"] == {"gap": 1, "veil": 0}
    return [v for k, v in amp.items() if int(k.split("/")[0]) >= 2]


def test_the_cascade_route_keeps_the_inside_statistics_closer_than_the_fill():
    coarse = _cascade_amp(cascade_synthesis=True)
    assert coarse and np.mean(coarse) < 0.5                 # a physics claim: report the number if it fails


@pytest.mark.xfail(strict=True, reason="plain cascade on Haar: coarse mean amplitude distance 0.609 (synthesis: 0.373) (measured); the plain inference infers parents only as ratios off the fill's children, so the inside level stays anchored to the fill -- the author decides whether the synthesis pass becomes the default")
def test_the_plain_cascade_inference_anchors_the_inside_amplitudes_to_the_fill():
    coarse = _cascade_amp()
    assert coarse and np.mean(coarse) < 0.5


def test_a_value_prior_on_a_coarser_nested_grid_and_the_route_comparison():
    """A second realisation that shares the cascade's coarse levels (the paper's climatology):
    with it the cascade route keeps the fine-level amplitude statistics closer than the sparsity
    route and is not worse than 1.1 x its held-out error."""
    dens, _ = _cascade2d()
    coarse = dens.reshape(8, 16, 8, 16).mean(axis=(1, 3))
    fine, _ = _cascade2d(n=4, seed=11)
    prior_full = np.kron(coarse, np.ones((16, 16))) * np.tile(fine / fine.mean(), (8, 8))
    prior2 = prior_full.reshape(64, 2, 64, 2).mean(axis=(1, 3))      # on a 2x coarser grid
    y = dens.copy()
    y[_box(y.shape, 48, 80, 48, 80)] = np.nan
    t = _field(y)
    g = _guide(t, prior2, "value", step=60.0, x0=500015.0 + 15.0, y0=3999985.0 - 15.0, name="prior")
    assert g.relation.kind == "nested" and g.relation.ratio == 2
    out = {}
    for route in ("sparsity", "cascade"):
        res = en.run_inpaint(t, y, _params(route=route, heldout_blocks=6, heldout_side=8, seed=1,
                                           prior_weight=0.3), [g])
        assert res["_grid_relations"]["prior"]["kind"] == "nested"
        out[route] = res
    assert out["cascade"]["_heldout"]["rms_rel"] <= 1.1 * out["sparsity"]["_heldout"]["rms_rel"]
    fine_amp = {r: np.mean([v for k, v in out[r]["_amp_distance"].items() if k.startswith("1/")])
                for r in out}
    assert fine_amp["cascade"] < fine_amp["sparsity"]


def test_a_value_prior_on_the_same_grid_with_an_offset_and_a_finer_one_are_both_accepted():
    dens, y = _gapped_cascade()
    t = _field(y)
    shifted = np.full(dens.shape, np.nan)                   # the guide's content at its own origin
    shifted[:-2, :-3] = dens[2:, 3:]
    same = _guide(t, shifted * 2.0 + 1.0, "value", x0=500015.0 + 3 * 30.0, y0=3999985.0 - 2 * 30.0, name="s")
    assert same.relation.kind == "same" and same.relation.offset == (2, 3)
    res = en.run_inpaint(t, y, _params(route="sparsity", n_iter=10, heldout_blocks=0), [same])
    assert res["_grid_relations"]["s"]["offset"] == [2, 3]
    assert abs(res["_affine"]["s"][0] - 0.5) < 0.05 and abs(res["_affine"]["s"][1] + 0.5) < 0.1
    finer = _guide(t, np.kron(dens, np.ones((2, 2))), "value", step=15.0, x0=500015.0 - 7.5,
                   y0=3999985.0 + 7.5, name="f")
    assert finer.relation.kind == "nested" and finer.relation.direction == "finer"
    res = en.run_inpaint(t, y, _params(route="sparsity", n_iter=10, heldout_blocks=0), [finer])
    assert res["_grid_relations"]["f"]["direction"] == "finer"
    assert abs(res["_affine"]["f"][0] - 1.0) < 0.05               # an exact aggregation


def test_a_structure_guide_enters_through_its_details_and_weak_bands_are_dropped():
    dens, y = _gapped_cascade()
    t = _field(y)
    g = _guide(t, dens * 3.0, "structure", name="other_band")
    res = en.run_inpaint(t, y, _params(route="sparsity", n_iter=10, heldout_blocks=0,
                                       structure_levels=2), [g])
    assert res["_terms_dropped"] == []
    noise = _guide(t, np.random.default_rng(5).standard_normal(y.shape), "structure", name="noise")
    res = en.run_inpaint(t, y, _params(route="sparsity", n_iter=10, heldout_blocks=0,
                                       structure_levels=2), [noise])
    assert len(res["_terms_dropped"]) == 6                   # every level and orientation


def test_a_dem_structure_guide_is_shaded_under_the_scene_sun():
    dens, y = _gapped_cascade()
    t = _field(y)
    dem = _guide(t, np.cumsum(dens, axis=1) * 10.0, "structure", name="dem", shade=True)
    res = en.run_inpaint(t, y, _params(route="sparsity", n_iter=5, heldout_blocks=0,
                                       sun_azimuth=135.0, sun_elevation=40.0), [dem])
    assert res["_shaded"] == ["dem"]


def test_a_foreign_guide_closes_the_cascade_routes_and_opens_through_a_warp():
    dens, y = _gapped_cascade()
    t = _field(y)
    g = _field(dens, x0=500015.0 + 12.0, step=30.0, name="foreign", crs=UTM)   # 0.4 px off
    guide = en.Guide("foreign", "value", g, relate(georef(t), georef(g)))
    assert guide.relation.kind == "foreign"
    with pytest.raises(ValueError, match="foreign"):
        en.run_inpaint(t, y, _params(route="cascade"), [guide])
    with pytest.raises(ValueError, match="foreign"):
        en.run_inpaint(t, y, _params(route="both"), [guide])
    res = en.run_inpaint(t, y, _params(route="sparsity", n_iter=5, heldout_blocks=0), [guide])
    assert res["_grid_relations"]["foreign"]["kind"] == "foreign"
    seed = en.Guide("seed", "veil_seed", g, guide.relation)
    res = en.run_inpaint(t, y, _params(route="cascade", n_iter=2, heldout_blocks=0), [seed])
    assert res["_route"] == "cascade"                        # a veil seed enters no term


def test_refusals_name_their_cause():
    dens, y = _gapped_cascade()
    t = _field(y)
    everything = np.full(y.shape, np.nan)
    with pytest.raises(ValueError, match="no clear pixel"):
        en.run_inpaint(t, everything, _params(), [])
    with pytest.raises(ValueError, match="3 pyramid levels"):
        en.run_inpaint(t, y, _params(route="cascade", level=2), [])
    with pytest.raises(ValueError, match="abacus_min_pairs"):
        en.run_inpaint(t, y, _params(route="cascade", abacus_min_pairs=10 ** 6, heldout_blocks=0), [])
    with pytest.raises(ValueError, match="held-out blocks caused it"):   # 8 blocks of 16 px on 128^2
        en.run_inpaint(t, y, _params(route="cascade"), [])
    with pytest.raises(ValueError, match="mask_band"):
        en.run_inpaint(t, y, _params(mask_source="threshold", mask_band="b9"), [])
    away = _field(dens, x0=500015.0 + 30.0 * 1000, name="away")
    guide = en.Guide("away", "value", away, relate(georef(t), georef(away)))
    with pytest.raises(ValueError, match="away.*no overlap"):
        en.run_inpaint(t, y, _params(route="sparsity"), [guide])
    signed = y - 1.0
    veil = json.dumps([{"kind": "veil", "points": [[10, 10], [30, 10], [30, 30], [10, 30]]}])
    with pytest.raises(ValueError, match="offset"):
        en.run_inpaint(t, signed, _params(route="sparsity", mask_source="all",
                                          _mask_polygons=veil, veil_model="gain"), [])
    scattered = dens.copy()
    scattered[np.random.default_rng(0).random(dens.shape) < 0.6] = np.nan
    with pytest.raises(ValueError, match="55"):
        en.run_inpaint(t, scattered, _params(route="cascade"), [])
    res = en.run_inpaint(t, scattered, _params(route="sparsity", n_iter=5, heldout_blocks=0), [])
    assert any("55" in w for w in res["_warnings"])
    edge = dens.copy()
    edge[0:20, 0:20] = np.nan
    res = en.run_inpaint(t, edge, _params(route="sparsity", n_iter=5, heldout_blocks=0), [])
    assert any("border" in w for w in res["_warnings"])
    with pytest.raises(ValueError, match="route"):
        en.run_inpaint(t, y, _params(route="magic"), [])


def test_a_veil_region_keeps_its_fine_details_and_recovers_a_smooth_gain():
    from scipy.ndimage import gaussian_filter

    dens, _ = _cascade2d()
    base = dens + 1.0                                        # strictly positive
    veil_mask = _box(base.shape, 40, 88, 40, 88)
    gain = np.ones(base.shape)
    gain[veil_mask] = 0.4 + 0.2 * gaussian_filter(np.random.default_rng(3).random(base.shape), 8)[veil_mask]
    gain = gaussian_filter(gain, 3)
    y = base * gain
    t = _field(y)
    poly = json.dumps([{"kind": "veil", "points": [[40, 40], [88, 40], [88, 88], [40, 88]]}])
    res = en.run_inpaint(t, y, _params(route="sparsity", mask_source="region", _mask_polygons=poly,
                                       veil_model="gain", veil_levels=3, veil_passes=2, n_iter=40,
                                       heldout_blocks=0), [])
    assert res["_veil"]["model"] == "gain" and res["_veil"]["seed"] == "surround"
    est = res["_estimate"]
    inside = veil_mask & (gain < 0.9)
    assert np.corrcoef(est[inside], base[inside])[0, 1] > 0.8
    assert np.corrcoef(np.gradient(est)[0][inside], np.gradient(base)[0][inside])[0, 1] > 0.6
    assert abs(est[inside].mean() - base[inside].mean()) < 0.5 * abs(y[inside].mean() - base[inside].mean())
    log_y = np.log(y)
    res = en.run_inpaint(t, log_y, _params(route="sparsity", mask_source="region", _mask_polygons=poly,
                                           veil_model="offset", veil_levels=3, n_iter=20,
                                           heldout_blocks=0), [])
    assert res["_veil"]["model"] == "offset"
    assert abs(res["_estimate"][inside].mean() - np.log(base)[inside].mean()) < \
        0.5 * abs(log_y[inside].mean() - np.log(base)[inside].mean())


def test_the_stage_store_hits_on_a_route_change_and_misses_on_a_mask_change(monkeypatch):
    from dynamix.core.inpaint import priors as pr

    calls = []
    real = pr.Abacus.from_pyramid

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(pr.Abacus, "from_pyramid", counted)
    dens, y = _gapped_cascade()
    t = _field(y)
    store = StageStore(8)
    en.run_inpaint(t, y, _params(route="cascade", n_iter=2, heldout_blocks=0), [], stage_cache=store)
    assert len(calls) == 1 and len(store) == 1
    en.run_inpaint(t, y, _params(route="both", n_iter=2, heldout_blocks=0), [], stage_cache=store)
    assert len(calls) == 1 and len(store) == 1               # a route change reads the stages
    y2 = y.copy()
    y2[10:14, 10:14] = np.nan
    en.run_inpaint(t, y2, _params(route="cascade", n_iter=2, heldout_blocks=0), [], stage_cache=store)
    assert len(calls) == 2 and len(store) == 2
    en.run_inpaint(t, y, _params(route="cascade", n_iter=2, heldout_blocks=0, seed=5), [],
                   stage_cache=store)
    assert len(calls) == 3                                   # the seed is keyed: it moves the blocks
    fresh = StageStore(8)                                    # the depth the route picks is keyed too
    en.run_inpaint(t, y, _params(route="sparsity", n_iter=2, heldout_blocks=0), [], stage_cache=fresh)
    en.run_inpaint(t, y, _params(route="cascade", n_iter=2, heldout_blocks=0), [], stage_cache=fresh)
    assert len(fresh) == 2
    stack = np.stack([y, y + 1.0], axis=-1)                  # two bands, one NaN pattern, one store
    ts = _field(stack, bands=["b1", "b2"], name="stack")
    en.run_inpaint(ts, stack[..., 0], _params(route="cascade", n_iter=2, heldout_blocks=0, target_band="b1"), [], stage_cache=fresh)
    en.run_inpaint(ts, stack[..., 1], _params(route="cascade", n_iter=2, heldout_blocks=0, target_band="b2"), [], stage_cache=fresh)
    assert len(fresh) == 4                                   # the values are keyed, not the name


def test_a_target_with_non_finite_guide_pixels_inside_the_region_still_fills():
    dens, y = _gapped_cascade()
    t = _field(y)
    holey = dens.copy()
    holey[50:70, 50:70] = np.nan                             # the prior is missing where needed most
    g = _guide(t, holey, "value", name="holey")
    res = en.run_inpaint(t, y, _params(route="sparsity", n_iter=10, heldout_blocks=0), [g])
    assert np.isfinite(res["raster_out"]).all()
    assert any("holey" in w and "coverage" in w for w in res["_warnings"])


def test_a_weights_send_on_the_target_zeroes_what_it_says():
    dens, y = _gapped_cascade()
    t = _field(y)
    w = np.ones(y.shape)
    w[0:20, :] = 0.0                                         # distrust the top rows
    wg = _guide(t, w, "weights", name="coherence")
    res = en.run_inpaint(t, y, _params(route="sparsity", n_iter=5, heldout_blocks=0), [wg])
    known = np.isfinite(y)
    np.testing.assert_array_equal(res["raster_out"][known], y[known])   # the data is never altered
    assert not np.array_equal(res["_estimate"][0:20, :], y[0:20, :].astype(np.float32))   # weight 0: free


def test_a_weights_send_holds_the_estimate_to_the_data_where_the_weight_is_one():
    """The sparsity estimate keeps the data's smooth part only, so the known pixels under
    weight 1 are approached, never matched exactly; where the weight is 0 the fit lets them
    go, and only the correction (which pins every known pixel's smooth part) and the start
    at the data keep the estimate near them, so its deviation there is larger (measured
    ratio 0.55 with five iterations)."""
    dens, y = _gapped_cascade()
    t = _field(y)
    w = np.ones(y.shape)
    w[0:20, :] = 0.0
    wg = _guide(t, w, "weights", name="coherence")
    res = en.run_inpaint(t, y, _params(route="sparsity", n_iter=5, heldout_blocks=0), [wg])
    known = np.isfinite(y)
    est = res["_estimate"].astype(np.float64)
    dev_free = np.sqrt(np.mean((est[0:20, :][known[0:20, :]] - y[0:20, :][known[0:20, :]]) ** 2))
    dev_held = np.sqrt(np.mean((est[30:, :][known[30:, :]] - y[30:, :][known[30:, :]]) ** 2))
    assert dev_held < 0.75 * dev_free
