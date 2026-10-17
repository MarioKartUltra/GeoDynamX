# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Halo widening to the measured reach (spec 22 §3.2.1): a converged reach widens the halo, never
narrowing it below the 2.5 a budget; where the reach does not converge the scale reads the whole
parent when it fits the budget, and otherwise keeps its halo and the honest-halo warning."""
from __future__ import annotations

import math

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from dynamix.core.wtmm_backend import compute_scales2d  # noqa: E402
from dynamix.roi import halo  # noqa: E402
from test_roi_halo import MARGINS, PARAMS, SCALES, _synthetic_parent, _write_tif  # noqa: E402

FINE = {"n_oct": 2, "n_voice": 4, "fracint_alpha": 0.0}


def _plan(a_min, wavelet="mexican", sp=None, parent=(192, 192)):
    scales = compute_scales2d(2, 4, a_min)
    return scales, halo.halo_plan(scales, wavelet, sp, 1e-3, parent)


def test_converged_scales_widen_to_the_reach_and_never_narrow():
    scales, plan = _plan(2.0, "q_gaussian", {"q": 1.5})
    assert all(sh.reach.converged and not sh.whole for sh in plan)
    for sh in plan:
        assert sh.margin == max(halo.halo_margin(sh.a), math.ceil(sh.reach.radius))
    assert plan[0].margin > halo.halo_margin(scales[0])


def test_existing_oracle_margins_are_unchanged():
    plan = halo.halo_plan(SCALES, "mexican", None, 1e-3, (384, 384))
    assert [sh.margin for sh in plan] == MARGINS and not any(sh.whole for sh in plan)


def test_a_small_parent_is_read_whole_where_the_reach_does_not_converge():
    _scales, plan = _plan(0.5)
    whole = [sh.whole for sh in plan]
    assert any(whole) and whole == [not sh.reach.converged for sh in plan]


@pytest.mark.parametrize("a_min", [0.25, 0.5])
def test_whole_parent_roi_moduli_equal_the_whole_raster_moduli(tmp_path, a_min):
    """Spec 22 test 2: a parent small enough to read whole matches the whole-raster moduli."""
    from dynamix.core.wtmm_levels import run_wtmm2d_level

    parent = _synthetic_parent(192, seed=3)
    tif = _write_tif(tmp_path / "p.tif", parent)
    roi = (64, 64, 64, 64)
    params = dict(FINE, a_min=a_min)
    _scales, plan = _plan(a_min)
    levels = [i for i, sh in enumerate(plan) if sh.whole]
    assert levels
    for i in levels:
        got = halo.run_wtmm2d_roi_level(tif, (192, 192), roi, params, i)["mod"]
        want = run_wtmm2d_level(parent, params, i)["mod"][64:128, 64:128]
        assert np.abs(got - want).max() <= 1e-6 * np.abs(want).max()
    res = halo.run_wtmm2d_roi(tif, (192, 192), roi, params)
    assert not any("does not converge" in w for w in res.get("_warnings", []))


def test_over_the_budget_keeps_the_halo_and_warns(tmp_path, monkeypatch):
    monkeypatch.setattr(halo, "WHOLE_PARENT_MAX_PIXELS", 100)
    tif = _write_tif(tmp_path / "p.tif", _synthetic_parent(192, seed=3))
    scales, plan = _plan(0.5)
    assert not any(sh.whole for sh in plan)
    assert [sh.margin for sh in plan[:2]] == [halo.halo_margin(a) for a in scales[:2]]
    res = halo.run_wtmm2d_roi(tif, (192, 192), (64, 64, 64, 64), dict(FINE, a_min=0.5))
    msg = [w for w in res["_warnings"] if "does not converge" in w]
    assert msg and "reach" in msg[0]


def test_the_result_records_the_edge_plan(tmp_path):
    tif = _write_tif(tmp_path / "p.tif", _synthetic_parent(384, seed=7))
    res = halo.run_wtmm2d_roi(tif, (384, 384), (0, 0, 64, 64), dict(PARAMS))
    edge = res["_edge"]
    assert edge["parent_shape"] == [384, 384] and edge["budget_px"] == halo.WHOLE_PARENT_MAX_PIXELS
    first = edge["scales"][0]
    assert first["grids"] == [1024, 2048] and first["ratio"] is not None and first["whole"] is False
    assert first["synthetic_px"]["N"] == first["margin"] and first["synthetic_px"]["S"] == 0
    assert first["real_px"]["S"] == first["margin"]


def test_in_memory_region_path_reads_the_whole_parent_too(tmp_path):
    from test_roi_wtmm_window import _KEYS, _picture, _points

    from dynamix.devices.wtmm import WTMM2D
    from dynamix.model.device import defaults_for, validate_params
    from dynamix.roi.runner import run_on_region

    tif = tmp_path / "s.tif"
    rng = np.random.default_rng(7)
    v = rng.standard_normal((96, 96)).cumsum(0).cumsum(1).astype(np.float32)
    _write_tif(tif, v)
    p = defaults_for(WTMM2D())
    p.update(n_oct=2, n_voice=2, a_min=0.5)
    params = validate_params(WTMM2D(), p)
    roi = (30, 34, 24, 28)
    got = run_on_region([(WTMM2D(), params)], _picture(tif), roi)
    want = halo.run_wtmm2d_roi(str(tif), (96, 96), roi, {k: params[k] for k in _KEYS})
    assert [s["whole"] for s in got["_edge"]["scales"]] == [s["whole"] for s in want["_edge"]["scales"]]
    assert any(s["whole"] for s in want["_edge"]["scales"])
    for g, w in zip(got["extrema"], want["extrema"]):
        pg, pw = _points(g), _points(w)
        assert pg.keys() == pw.keys()
        assert np.allclose([pg[k] for k in pw], [pw[k] for k in pw], rtol=1e-6, atol=1e-9)


def test_the_plan_checks_the_run_grid_until_a_scale_holds_there():
    """Convergence improves with scale: past the first scale that holds at the run's grid, the
    coarser ones are not measured there."""
    scales = compute_scales2d(2, 4, 1.0)
    plan = halo.halo_plan(scales, "mexican", None, 1e-3, (5000, 5000), run_grid=4000)
    assert not plan[0].reach.converged and plan[0].reach.grids == (1024, 2048, 4096)
    first = next(i for i, sh in enumerate(plan) if sh.reach.converged)
    assert plan[first].reach.grids[-1] == 4096
    assert all(sh.reach.grids == (1024, 2048) for sh in plan[first + 1:])


def test_the_roi_run_grid_is_the_coarsest_window():
    scales = compute_scales2d(2, 4, 1.0)
    assert halo.roi_run_grid(64, 48, scales, 32) == 64 + 2 * halo.halo_margin(scales[-1]) + 64


def test_the_record_names_the_run_grid(tmp_path):
    tif = _write_tif(tmp_path / "p.tif", _synthetic_parent(384, seed=7))
    res = halo.run_wtmm2d_roi(tif, (384, 384), (0, 0, 64, 64), dict(PARAMS))
    assert res["_edge"]["run_grid"] == halo.roi_run_grid(64, 64, SCALES, 32)
    assert all(s["ratio_run"] is None for s in res["_edge"]["scales"])
