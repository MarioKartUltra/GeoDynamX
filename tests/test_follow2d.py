# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.follow2d -- the xsmurf ``follow`` (kappa zero-crossing) detector.

Formula provenance: gkapa/gkapap in xsmurf ``interpreter/wt2d_cmds.c:2948/3022`` (verified
byte-identical to upstream pkestene/xsmurf). The detector's contract: same
extrema schema as the NMS plus native ``x_sub``/``y_sub`` and the crossing-interpolated
modulus; the kappa' < 0 gate rejects modulus minima; on a clean single ridge, follow and NMS
must find the same line."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import follow2d


def _rand_derivs(rng, n=12):
    return {k: rng.normal(size=(1, n, n)).astype(np.float32)
            for k in ("dx", "dy", "dxx", "dxy", "dyy", "dxxx", "dxxy", "dxyy", "dyyy")}


# ------------------------------------------------------------------------ the ported formulas

def test_kapa_fields_match_the_gkapa_gkapap_formulas():
    rng = np.random.default_rng(3)
    d = _rand_derivs(rng)
    kapa, kapap = follow2d.kapa_fields(d, 0)
    dx, dy = d["dx"][0].astype(np.float64), d["dy"][0].astype(np.float64)
    dxx, dxy, dyy = (d[k][0].astype(np.float64) for k in ("dxx", "dxy", "dyy"))
    dxxx, dxxy, dxyy, dyyy = (d[k][0].astype(np.float64)
                              for k in ("dxxx", "dxxy", "dxyy", "dyyy"))
    kapa_ref = 2 * dx * dx * dxx + 4 * dxy * dx * dy + 2 * dy * dy * dyy
    kapap_ref = (dx * dx * (4 * (dxx * dxx + dxy * dxy) + 2 * dx * dxxx + 6 * dy * dxxy)
                 + dy * dy * (4 * (dyy * dyy + dxy * dxy) + 2 * dy * dyyy + 6 * dx * dxyy)
                 + 8 * dx * dy * dxy * (dxx + dyy))
    np.testing.assert_allclose(kapa, kapa_ref, rtol=2e-5)
    np.testing.assert_allclose(kapap, kapap_ref, rtol=2e-5)


# --------------------------------------------------------------------------- detection gates

def _ridge_fields(nx=32, ny=8, x0=15.0):
    """An analytic modulus ridge at column x0 with gradient along +x: M = exp(-(x-x0)^2/8),
    kappa = dM/dx-like (sign change at x0), kappa' < 0 there."""
    x = np.arange(nx, dtype=np.float64)
    m1d = np.exp(-((x - x0) ** 2) / 8.0)
    mod = np.broadcast_to(m1d, (ny, nx)).astype(np.float32).copy()
    kapa1d = -(x - x0) * m1d                      # sign-equivalent to dM/du
    kapa = np.broadcast_to(kapa1d, (ny, nx)).astype(np.float32).copy()
    kapap = np.full((ny, nx), -1.0, dtype=np.float32)
    arg = np.zeros((ny, nx), dtype=np.float32)    # u = +x everywhere
    return mod, arg, kapa, kapap


def test_a_clean_ridge_is_claimed_once_per_row_at_the_crossing():
    mod, arg, kapa, kapap = _ridge_fields(x0=15.0)
    e = follow2d.follow_extrema_scale(mod, arg, kapa, kapap)
    ny = mod.shape[0]
    assert e["x"].size == ny                       # exactly one claim per row
    np.testing.assert_array_equal(e["x"], np.full(ny, 15))
    np.testing.assert_allclose(e["x_sub"], 15.0, atol=1e-6)
    np.testing.assert_allclose(e["y_sub"], e["y"], atol=1e-12)
    assert np.all(e["line_id"] >= 0)               # one connected vertical line


def test_a_fractional_ridge_lands_in_the_subpixel_channel():
    mod, arg, kapa, kapap = _ridge_fields(x0=15.37)
    e = follow2d.follow_extrema_scale(mod, arg, kapa, kapap)
    assert set(e["x"].tolist()) == {15}            # integer support = owning pixel
    np.testing.assert_allclose(e["x_sub"], 15.37, atol=1e-2)
    # the value channel: parabola through the bilinear samples AT the crossing >= grid value
    assert np.all(e["mod"] >= mod[e["y"], e["x"]] - 1e-6)


def test_the_kapap_gate_rejects_modulus_minima():
    """Same kappa crossing, kappa' > 0 -- a modulus MINIMUM along u: no claims at all."""
    mod, arg, kapa, kapap = _ridge_fields()
    e = follow2d.follow_extrema_scale(mod, arg, kapa, -kapap)   # flip the gate
    assert e["x"].size == 0


def test_the_threshold_gate_matches_the_nms_convention():
    mod, arg, kapa, kapap = _ridge_fields()
    keep = follow2d.follow_extrema_scale(mod, arg, kapa, kapap, thresh=1e-3)
    drop = follow2d.follow_extrema_scale(mod, arg, kapa, kapap, thresh=1.1)
    assert keep["x"].size > 0 and drop["x"].size == 0


def test_invalid_mask_distrust_matches_the_nms_contract():
    mod, arg, kapa, kapap = _ridge_fields()
    invalid = np.zeros(mod.shape, dtype=bool)
    invalid[0, 15] = True
    e = follow2d.follow_extrema_scale(mod, arg, kapa, kapap, invalid=invalid, radius=1)
    assert 0 not in e["y"] and 1 not in e["y"]     # the dilated distrust band drops both rows


# ------------------------------------------------------------------------ pipeline integration

def _small_field(n=48):
    from conftest import fbm2d
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    vals = fbm2d(n, 0.6, seed=3)
    return RasterField(name="follow-test", values=vals, frame=LocalFrame(),
                       x_axis=np.arange(n, dtype=np.float64),
                       y_axis=np.arange(n, dtype=np.float64))


_SMALL = {"n_oct": 2, "n_voice": 2, "a_min": 1.0, "q_list": np.arange(-1.0, 2.1, 1.0)}


def test_detector_resolves_and_defaults_to_nms():
    from dynamix.core.wtmm_backend import _resolve_wtmm2d_params

    assert _resolve_wtmm2d_params({})["detector"] == "nms"
    assert _resolve_wtmm2d_params({"detector": "follow"})["detector"] == "follow"


def test_run_wtmm2d_follow_end_to_end():
    """detector='follow': every layer carries the native float channels, the interpolate knob
    is inert (follow's channels ARE the interpolation), and the pipeline completes through
    chains and partition tables."""
    from dynamix.core.wtmm_backend import run_wtmm2d

    field = _small_field()
    nms = run_wtmm2d(field, dict(_SMALL), out_dir=None)
    fol = run_wtmm2d(field, dict(_SMALL, detector="follow"), out_dir=None)
    assert all("x_sub" in e for e in fol["extrema"])
    assert any(e["x"].size for e in fol["extrema"])
    assert fol["hd_std"] is not None
    # the two detectors are different discretizations -- sets overlap but need not coincide
    # on an fBm field; sanity: same grid bounds, comparable counts at the finest scale
    n_f, n_n = fol["extrema"][0]["x"].size, nms["extrema"][0]["x"].size
    assert n_f > 0.3 * n_n


def test_follow_and_nms_agree_on_a_clean_gaussian_edge():
    """One blurred step edge: both detectors must find the same column (the continuum object
    is the same; only the discretization differs)."""
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.core.wtmm_backend import run_wtmm2d

    # 31.8, deliberately OFF the half-pixel: at 31.5 the crossing is a claim-convention tie
    # (follow said 31, NMS said 32, both defensible) -- measured; the agreement
    # contract holds where a single pixel unambiguously owns the crossing.
    n = 64
    vals = np.broadcast_to(1.0 / (1.0 + np.exp(-(np.arange(n) - 31.8))), (n, n)).copy()
    field = RasterField(name="edge", values=vals, frame=LocalFrame(),
                        x_axis=np.arange(n, dtype=np.float64),
                        y_axis=np.arange(n, dtype=np.float64))
    params = {"n_oct": 1, "n_voice": 1, "a_min": 1.0, "q_list": np.array([0.0, 2.0])}
    nms = run_wtmm2d(field, dict(params), out_dir=None)
    fol = run_wtmm2d(field, dict(params, detector="follow"), out_dir=None)
    xs_nms = np.unique(nms["extrema"][0]["x"])
    xs_fol = np.unique(fol["extrema"][0]["x"])
    # both sit on the edge columns (31/32), well away from borders
    interior = lambda xs: xs[(xs > 8) & (xs < 55)]
    np.testing.assert_array_equal(interior(xs_fol), interior(xs_nms))


def test_preview_matches_the_full_runs_finest_scale_with_follow():
    from dynamix.core.wtmm_backend import run_wtmm2d, run_wtmm2d_preview

    field = _small_field()
    params = dict(_SMALL, detector="follow")
    full = run_wtmm2d(field, params, out_dir=None)
    prev = run_wtmm2d_preview(field, params)
    for k in ("x", "y", "mod", "x_sub", "y_sub"):
        np.testing.assert_array_equal(prev["extrema"][0][k], full["extrema"][0][k], err_msg=k)


def test_wtmm2d_device_exposes_the_detector_knob():
    from dynamix.devices.wtmm import WTMM2D
    from dynamix.model.device import defaults_for

    dev = WTMM2D()
    assert "detector" in dev.param_groups["PRE"]
    params = dict(defaults_for(dev), n_oct=1, n_voice=1, detector="follow")
    res = dev.compute(_small_field(), params)
    assert all("x_sub" in e for e in res["extrema"])
    assert "chain_product" in res


def test_follow_engine_parity_mlx_vs_numpy():
    """The follow derivative stacks come from either engine; detection must agree on positions
    exactly and on values to the float32-parity floor (the backends-agree law)."""
    pytest.importorskip("mlx.core")
    from dynamix.core import wtmm_backend as wb

    field = _small_field(32)
    params = dict(_SMALL, n_oct=1, detector="follow")
    try:
        wb.set_default_engine("numpy")
        via_np = wb.run_wtmm2d(field, dict(params), out_dir=None)
        wb.set_default_engine("mlx")
        via_mlx = wb.run_wtmm2d(field, dict(params), out_dir=None)
    finally:
        wb.set_default_engine("auto")
    for e_n, e_m in zip(via_np["extrema"], via_mlx["extrema"]):
        np.testing.assert_array_equal(e_m["x"], e_n["x"])
        np.testing.assert_array_equal(e_m["y"], e_n["y"])
        np.testing.assert_allclose(e_m["mod"], e_n["mod"], rtol=2e-4, atol=1e-7)
        np.testing.assert_allclose(e_m["x_sub"], e_n["x_sub"], atol=2e-3)
