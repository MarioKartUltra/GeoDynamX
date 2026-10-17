# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The edge knobs (spec 22 §3.2.2–3): reflect by default with a ⚠ that names the scales whose halo
runs past the raster and the mirror lines' false maxima; cut at the cone of influence."""
from __future__ import annotations

import types

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from dynamix.devices.wtmm import WTMM2D  # noqa: E402
from dynamix.devices.wtmm_roi import WTMM2DROI  # noqa: E402
from dynamix.model.device import defaults_for, keyed_params, validate_params  # noqa: E402
from test_roi_halo import _synthetic_parent, _write_tif  # noqa: E402

EDGE_KEYS = {"edge", "cut_scope", "edge_eps"}


def _stub(full=(384, 384)):
    return types.SimpleNamespace(provenance={"full_dims": full})


def _roi_params(**kw):
    dev = WTMM2DROI()
    p = dict(defaults_for(dev), n_oct=2, n_voice=4, a_min=1.0, fracint_alpha=0.0,
             roi_h=64, roi_w=64)
    p.update(kw)
    return validate_params(dev, p)


@pytest.mark.parametrize("dev", [WTMM2D(), WTMM2DROI()], ids=["wtmm2d", "wtmm2d_roi"])
def test_edge_params_are_out_of_the_key_at_their_defaults(dev):
    names = {p.name for p in dev.params}
    assert EDGE_KEYS <= names
    assert not EDGE_KEYS & set(keyed_params(dev, defaults_for(dev)))
    assert defaults_for(dev)["edge"] == "reflect"


def test_reflect_at_a_corner_warns_naming_scales_widths_and_mirror_maxima():
    msg = WTMM2DROI().knob_warnings(_stub(), _roi_params(roi_row=0, roi_col=0))["a_min"]
    assert "mirror" in msg and "reflection" in msg and "N/W" in msg and "8 scale(s)" in msg
    assert "edge = cut" in msg
    assert WTMM2DROI().knob_warnings(_stub(), _roi_params(roi_row=160, roi_col=160)) == {}


def test_a_near_edge_roi_warns_only_for_the_scales_that_reach_past_it():
    msg = WTMM2DROI().knob_warnings(_stub(), _roi_params(roi_row=30, roi_col=160))["a_min"]
    assert "N edge" in msg and "8 scale(s)" not in msg


def test_reflective_boundary_always_warns():
    msg = WTMM2DROI().knob_warnings(_stub(), _roi_params(roi_row=160, roi_col=160,
                                                         boundary="reflective"))["a_min"]
    assert "mirror" in msg and "E/N/S/W" in msg


def test_cut_carries_no_mirror_warning():
    assert WTMM2DROI().knob_warnings(_stub(), _roi_params(roi_row=0, roi_col=0, edge="cut")) == {}


def test_the_corner_warning_rides_the_run_and_the_record(tmp_path):
    from dynamix.core.rasterfield import RasterField

    tif = _write_tif(tmp_path / "p.tif", _synthetic_parent(384, seed=7))
    field = RasterField.from_geotiff_window(tif, row_off=0, col_off=0, height=96, width=96)
    res = WTMM2DROI().compute(field, _roi_params(roi_row=0, roi_col=0))
    assert any("mirror" in w for w in res["_warnings"])
    assert res["_edge"]["mode"] == "reflect" and res["_edge"]["device"] == "wtmm2d_roi"


# ---------------------------------------------------------------------------------- cut (§3.2.3)

def _field(n=128, seed=11):
    """A random-walk surface: it chains well at these settings (about 300 chains)."""
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    v = np.random.default_rng(seed).standard_normal((n, n)).cumsum(0).cumsum(1)
    v = v.astype(np.float32)
    return RasterField(name="f", values=v, frame=LocalFrame(), x_axis=np.arange(float(n)),
                       y_axis=np.arange(float(n)))


def _wtmm_params(**kw):
    dev = WTMM2D()
    p = dict(defaults_for(dev), n_oct=2, n_voice=4, a_min=1.0, fracint_alpha=0.0)
    p.update(kw)
    return validate_params(dev, p)


def _dist(y, x, shape, oy=0, ox=0):
    py, px = np.asarray(y, float) + oy, np.asarray(x, float) + ox
    return np.minimum.reduce([py, px, shape[0] - 1 - py, shape[1] - 1 - px])


def _check_outside(res, shape, oy=0, ox=0):
    borders = [s["border"] for s in res["_edge"]["scales"]]
    for lvl, b in zip(res["extrema"], borders):
        assert (_dist(lvl["y"], lvl["x"], shape, oy, ox) >= b).all()
    scales = np.asarray(res["scales"], float)
    for c in res["chains"]:
        idx = [int(np.argmin(np.abs(scales - 2.0 ** l))) for l in c["log2_scales"]]
        d = _dist(c["y"], c["x"], shape, oy, ox)
        assert all(d[k] >= borders[i] for k, i in enumerate(idx))


@pytest.mark.parametrize("scope", ["largest", "per_scale"])
def test_cut_leaves_nothing_inside_the_recorded_border(scope):
    res = WTMM2D().compute(_field(), _wtmm_params(edge="cut", cut_scope=scope))
    assert res["_edge"]["mode"] == "cut" and res["_edge"]["device"] == "wtmm2d"
    _check_outside(res, (128, 128))
    assert sum(len(l["x"]) for l in res["extrema"]) > 0 and res["chains"]


def test_per_scale_keeps_at_least_as_many_fine_maxima():
    big = WTMM2D().compute(_field(), _wtmm_params(edge="cut", cut_scope="largest"))
    own = WTMM2D().compute(_field(), _wtmm_params(edge="cut", cut_scope="per_scale"))
    n_big = [len(l["x"]) for l in big["extrema"]]
    n_own = [len(l["x"]) for l in own["extrema"]]
    assert all(o >= b for o, b in zip(n_own, n_big)) and n_own[0] > n_big[0]


def test_the_cut_border_is_the_measured_reach():
    from dynamix.core.reach import reach

    res = WTMM2D().compute(_field(), _wtmm_params(edge="cut", cut_scope="per_scale"))
    for s, a in zip(res["_edge"]["scales"], res["scales"]):
        assert s["border"] == int(np.ceil(reach("mexican", None, a).radius))


def test_non_converging_scales_take_the_largest_border():
    res = WTMM2D().compute(_field(), _wtmm_params(edge="cut", cut_scope="per_scale", a_min=0.5))
    rows = res["_edge"]["scales"]
    assert any(not r["converged"] for r in rows)
    assert all(r["border"] == rows[-1]["border"] for r in rows if not r["converged"])


def test_cut_with_coi_drop_drops_the_whole_chain():
    trim = WTMM2D().compute(_field(), _wtmm_params(edge="cut"))
    drop = WTMM2D().compute(_field(), _wtmm_params(edge="cut", border_policy="coi_drop"))
    _check_outside(drop, (128, 128))
    assert len(drop["chains"]) <= len(trim["chains"])
    assert sum(len(c["x"]) for c in drop["chains"]) < sum(len(c["x"]) for c in trim["chains"])


def test_reflect_default_takes_no_edge_path(monkeypatch):
    """The default whole-raster run is the backend's run as before: the cut is never applied
    and nothing new is stamped."""
    import dynamix.devices.wtmm as wtmm

    def _no(*_a, **_k):
        raise AssertionError("cut_whole ran on edge = reflect")

    monkeypatch.setattr(wtmm, "cut_whole", _no)
    res = WTMM2D().compute(_field(), _wtmm_params())
    assert "_edge" not in res and res["chains"]


def test_roi_cut_counts_only_the_parent_edges(tmp_path):
    from dynamix.core.rasterfield import RasterField

    tif = _write_tif(tmp_path / "p.tif", _synthetic_parent(384, seed=7))
    field = RasterField.from_geotiff_window(tif, row_off=0, col_off=0, height=96, width=96)
    res = WTMM2DROI().compute(field, _roi_params(roi_row=0, roi_col=0, edge="cut"))
    _check_outside(res, (384, 384))
    borders = [s["border"] for s in res["_edge"]["scales"]]
    # a maximum close to the ROI's own S or E side, far from the parent's, is kept
    near_inner = [((l["y"] > 63 - b) | (l["x"] > 63 - b)).any()
                  for l, b in zip(res["extrema"], borders)]
    assert any(near_inner)


def test_roi_cut_in_the_region_runner_matches_the_file_path(tmp_path):
    from test_roi_wtmm_window import _KEYS, _picture, _points

    from dynamix.roi.halo import run_wtmm2d_roi
    from dynamix.roi.runner import run_on_region

    tif = tmp_path / "s.tif"
    v = np.random.default_rng(7).standard_normal((96, 96)).cumsum(0).cumsum(1).astype(np.float32)
    _write_tif(tif, v)
    params = _wtmm_params(n_oct=2, n_voice=2, edge="cut", cut_scope="per_scale")
    roi = (0, 0, 24, 28)
    got = run_on_region([(WTMM2D(), params)], _picture(tif), roi)
    want = run_wtmm2d_roi(str(tif), (96, 96), roi, dict({k: params[k] for k in _KEYS},
                                                         edge="cut", cut_scope="per_scale"))
    assert [s["border"] for s in got["_edge"]["scales"]] == \
           [s["border"] for s in want["_edge"]["scales"]]
    for g, w in zip(got["extrema"], want["extrema"]):
        assert _points(g).keys() == _points(w).keys()
    _check_outside(got, (96, 96))
