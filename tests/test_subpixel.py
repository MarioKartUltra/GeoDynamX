# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.subpixel -- parabolic refinement of NMS modulus maxima along the
gradient direction.

Provenance being pinned here:

- **LastWave 1-D** (``package_extrema1d/src/ext_compute.c``): default-on 3-point parabola at
  detection time, storing the float abscissa AND the refined value ALONGSIDE the integer index,
  with an out-of-bounds clamp falling back to the grid point. This module transplants exactly
  that dual representation along the 2-D gradient direction.
- **xsmurf** (``follow``, ``wt2d_cmds.c``): computes a subpixel offset and keeps only the
  refined MODULUS, never the position. Our ``mod_sub`` is that parity channel; unlike xsmurf we
  keep the offset too.
- **The probes are the NMS's own**: ``wtmm_backend._nms_extrema_scale`` compares each pixel
  against the bilinearly-sampled modulus at ``p ± (cos a, sin a)`` -- the parabola here is fit
  through those SAME three samples, so the refinement interpolates the exact quantity the
  detection compared (no new sampling convention is invented).
- **M-Z/M-H compatibility**: positions stay integer in ``x``/``y`` (reconstruction support --
  pocs2d's constraint model is pixel-exact by the papers' own numerics); the float channel is
  measurement/display only. The split is explicit, never accidental.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import subpixel


def _quad_x(shape=(16, 32), x0=7.3, c=0.01):
    """m(y, x) = 1 - c (x - x0)^2 -- exactly quadratic along +x, constant along y."""
    ny, nx = shape
    x = np.arange(nx, dtype=np.float64)
    m = 1.0 - c * (x[None, :] - x0) ** 2
    return np.broadcast_to(m, shape).copy()


# ------------------------------------------------------------------------------ single scale

def test_axis_aligned_quadratic_recovers_the_exact_offset():
    """A parabola along +x with vertex at x0 = 7.3: the integer max is x = 7, the refined
    position must be x0 EXACTLY (the 3-point parabola is exact on quadratics) and the refined
    modulus must be the vertex value 1.0."""
    m = _quad_x()
    x = np.array([7], dtype=np.int64)
    y = np.array([8], dtype=np.int64)
    arg = np.array([0.0])                      # gradient along +x
    x_sub, y_sub, mod_sub = subpixel.refine_scale(m, arg, x, y)
    np.testing.assert_allclose(x_sub, [7.3], rtol=0, atol=1e-9)
    np.testing.assert_allclose(y_sub, [8.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(mod_sub, [1.0], rtol=0, atol=1e-9)


def test_vertical_gradient_refines_y():
    m = _quad_x().T.copy()                     # parabola along +y now, vertex y0 = 7.3
    x = np.array([4], dtype=np.int64)
    y = np.array([7], dtype=np.int64)
    arg = np.array([np.pi / 2])                # gradient along +y
    x_sub, y_sub, mod_sub = subpixel.refine_scale(m, arg, x, y)
    np.testing.assert_allclose(y_sub, [7.3], atol=1e-9)
    np.testing.assert_allclose(x_sub, [4.0], atol=1e-12)


def test_diagonal_ridge_offset_beats_the_grid():
    """A quadratic ridge whose normal is at 35 degrees, offset 0.37 px from the nearest integer
    point along that normal. Bilinear sampling of the 2-D quadratic is not exact (x^2/y^2 terms),
    so the recovery is approximate -- but it must land within 0.1 px of the true ridge line,
    strictly better than the integer point's own distance. (Measured: ~0.056 px at 35 deg --
    the bilinear cross-term error, curvature-independent to first order; the tolerance below
    is that limit with headroom, not a wish.)"""
    theta = np.deg2rad(35.0)
    ny, nx = 32, 32
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    x_pt, y_pt = 15, 17
    d0 = 0.37
    # signed distance from the ridge line, zero at (x_pt + d0*cos, y_pt + d0*sin)
    d = (xx - x_pt - d0 * np.cos(theta)) * np.cos(theta) \
        + (yy - y_pt - d0 * np.sin(theta)) * np.sin(theta)
    m = 1.0 - 0.005 * d * d
    x_sub, y_sub, _ = subpixel.refine_scale(
        m, np.array([theta]), np.array([x_pt], dtype=np.int64),
        np.array([y_pt], dtype=np.int64))
    d_sub = (x_sub[0] - x_pt - d0 * np.cos(theta)) * np.cos(theta) \
        + (y_sub[0] - y_pt - d0 * np.sin(theta)) * np.sin(theta)
    assert abs(d_sub) < 0.1 < d0               # refined beats the raw grid point


def test_a_tie_with_the_forward_neighbour_gives_exactly_half_a_pixel():
    """NMS keeps ties (m >= fore). m0 == m+ with m- below is the boundary case: the parabola
    vertex is exactly halfway to the tied neighbour -- t* = +0.5, never beyond (the LastWave
    clamp's own boundary)."""
    m = np.zeros((5, 7))
    m[:, 3] = 1.0
    m[:, 4] = 1.0                              # forward tie
    m[:, 2] = 0.5
    x_sub, y_sub, mod_sub = subpixel.refine_scale(
        m, np.array([0.0]), np.array([3], dtype=np.int64), np.array([2], dtype=np.int64))
    np.testing.assert_allclose(x_sub, [3.5], atol=1e-12)
    assert mod_sub[0] >= 1.0


def test_flat_plateau_falls_back_to_the_grid_point():
    m = np.ones((5, 7))
    x_sub, y_sub, mod_sub = subpixel.refine_scale(
        m, np.array([0.3]), np.array([3], dtype=np.int64), np.array([2], dtype=np.int64))
    np.testing.assert_allclose(x_sub, [3.0], atol=1e-12)
    np.testing.assert_allclose(y_sub, [2.0], atol=1e-12)
    np.testing.assert_allclose(mod_sub, [1.0], atol=1e-12)


def test_refined_modulus_is_never_below_the_grid_modulus():
    rng = np.random.default_rng(7)
    m = rng.random((24, 24))
    from scipy.ndimage import gaussian_filter
    m = gaussian_filter(m, 2.0)
    ys, xs = np.mgrid[2:22:4, 2:22:4]
    x = xs.ravel().astype(np.int64)
    y = ys.ravel().astype(np.int64)
    arg = rng.uniform(-np.pi, np.pi, x.size)
    _, _, mod_sub = subpixel.refine_scale(m, arg, x, y)
    assert np.all(mod_sub >= m[y, x] - 1e-12)


def test_offsets_are_clamped_to_half_a_pixel():
    rng = np.random.default_rng(11)
    m = rng.random((24, 24))
    ys, xs = np.mgrid[2:22:3, 2:22:3]
    x = xs.ravel().astype(np.int64)
    y = ys.ravel().astype(np.int64)
    arg = rng.uniform(-np.pi, np.pi, x.size)
    x_sub, y_sub, _ = subpixel.refine_scale(m, arg, x, y)
    # offset along the probe direction never exceeds half a pixel in magnitude
    t = (x_sub - x) * np.cos(arg) + (y_sub - y) * np.sin(arg)
    assert np.all(np.abs(t) <= 0.5 + 1e-12)
    # and the perpendicular displacement is zero -- refinement moves ALONG the gradient only
    p = -(x_sub - x) * np.sin(arg) + (y_sub - y) * np.cos(arg)
    np.testing.assert_allclose(p, 0.0, atol=1e-12)


# ------------------------------------------------------------------------------ stack API

def _stack_fixture():
    m0 = _quad_x((8, 16), x0=5.4)
    m1 = _quad_x((8, 16), x0=9.7)
    mod_stack = np.stack([m0, m1]).astype(np.float32)
    extrema = [
        {"x": np.array([5], dtype=np.int64), "y": np.array([3], dtype=np.int64),
         "mod": np.array([m0[3, 5]]), "arg": np.array([0.0]),
         "line_id": np.array([0], dtype=np.int64)},
        {"x": np.array([10], dtype=np.int64), "y": np.array([6], dtype=np.int64),
         "mod": np.array([m1[6, 10]]), "arg": np.array([0.0]),
         "line_id": np.array([-1], dtype=np.int64)},
    ]
    return extrema, mod_stack


def test_stack_api_adds_float_channels_and_replaces_mod():
    extrema, mod_stack = _stack_fixture()
    out = subpixel.refine_extrema_stack(extrema, mod_stack)
    assert out is not extrema
    for si, (e_in, e_out) in enumerate(zip(extrema, out)):
        np.testing.assert_array_equal(e_out["x"], e_in["x"])
        np.testing.assert_array_equal(e_out["y"], e_in["y"])
        np.testing.assert_array_equal(e_out["arg"], e_in["arg"])
        np.testing.assert_array_equal(e_out["line_id"], e_in["line_id"])
        assert e_out["x_sub"].dtype == np.float64
        assert e_out["y_sub"].dtype == np.float64
    # scale 0: vertex at x = 5.4 (float32 raster -> ~1e-3 tolerance)
    np.testing.assert_allclose(out[0]["x_sub"], [5.4], atol=1e-2)
    np.testing.assert_allclose(out[1]["x_sub"], [9.7], atol=1e-2)
    # the refined modulus is the parity channel: it REPLACES mod (>= the grid value)
    assert out[0]["mod"][0] >= extrema[0]["mod"][0]


def test_stack_api_never_mutates_its_input():
    extrema, mod_stack = _stack_fixture()
    before = [{k: v.copy() for k, v in e.items()} for e in extrema]
    subpixel.refine_extrema_stack(extrema, mod_stack)
    for e_in, e_before in zip(extrema, before):
        assert set(e_in) == set(e_before)
        for k in e_before:
            np.testing.assert_array_equal(e_in[k], e_before[k])


def test_empty_scale_layer_passes_through():
    empty = {"x": np.zeros(0, dtype=np.int64), "y": np.zeros(0, dtype=np.int64),
             "mod": np.zeros(0), "arg": np.zeros(0),
             "line_id": np.zeros(0, dtype=np.int64)}
    out = subpixel.refine_extrema_stack([empty], np.zeros((1, 8, 8), dtype=np.float32))
    assert out[0]["x_sub"].size == 0 and out[0]["y_sub"].size == 0


# ------------------------------------------------------------------- stage-cache round-trip

def test_extrema_arrays_round_trip_the_float_channels():
    from dynamix.core.wtmm_backend import _arrays_to_extrema, _extrema_to_arrays

    extrema, mod_stack = _stack_fixture()
    refined = subpixel.refine_extrema_stack(extrema, mod_stack)
    back = _arrays_to_extrema(_extrema_to_arrays(refined))
    for e_ref, e_back in zip(refined, back):
        for k in ("x", "y", "mod", "arg", "line_id", "x_sub", "y_sub"):
            np.testing.assert_array_equal(e_back[k], e_ref[k], err_msg=k)


def test_extrema_arrays_stay_backward_compatible_without_the_channels():
    """An OLD stage-cache npz (no x_sub/y_sub) must load exactly as before -- and a plain
    (un-refined) extrema list must round-trip without inventing the new keys."""
    from dynamix.core.wtmm_backend import _arrays_to_extrema, _extrema_to_arrays

    extrema, _ = _stack_fixture()
    arrays = _extrema_to_arrays(extrema)
    assert "x_sub" not in arrays
    back = _arrays_to_extrema(arrays)
    assert all("x_sub" not in e for e in back)
    for e_in, e_back in zip(extrema, back):
        for k in ("x", "y", "mod", "arg", "line_id"):
            np.testing.assert_array_equal(e_back[k], e_in[k], err_msg=k)


# ----------------------------------------------------------------------------- pipeline knob

def _small_field(n=48):
    from conftest import fbm2d
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    vals = fbm2d(n, 0.6, seed=3)
    return RasterField(name="subpix-test", values=vals, frame=LocalFrame(),
                       x_axis=np.arange(n, dtype=np.float64),
                       y_axis=np.arange(n, dtype=np.float64))


_SMALL = {"n_oct": 2, "n_voice": 2, "a_min": 1.0, "q_list": np.arange(-1.0, 2.1, 1.0)}


def test_interpolate_defaults_off_and_resolves():
    from dynamix.core.wtmm_backend import _resolve_wtmm2d_params

    assert _resolve_wtmm2d_params({})["interpolate"] is False
    assert _resolve_wtmm2d_params({"interpolate": True})["interpolate"] is True


def test_run_wtmm2d_interpolate_knob_end_to_end():
    """interpolate=True: every extrema layer carries the float channels, integer positions are
    IDENTICAL to the knob-off run (detection unchanged -- only values and the float channels
    move), and the refined moduli flow into the partition tables."""
    from dynamix.core.wtmm_backend import run_wtmm2d

    field = _small_field()
    off = run_wtmm2d(field, dict(_SMALL), out_dir=None)
    on = run_wtmm2d(field, dict(_SMALL, interpolate=True), out_dir=None)

    assert all("x_sub" not in e for e in off["extrema"])
    changed = False
    for e_off, e_on in zip(off["extrema"], on["extrema"]):
        np.testing.assert_array_equal(e_on["x"], e_off["x"])
        np.testing.assert_array_equal(e_on["y"], e_off["y"])
        assert e_on["x_sub"].shape == e_on["x"].shape
        assert np.all(e_on["mod"] >= e_off["mod"] - 1e-12)
        if e_on["mod"].size and not np.array_equal(e_on["mod"], e_off["mod"]):
            changed = True
    assert changed, "refinement changed no modulus at all -- the knob is inert"
    # the values channel reaches Z(q,a): at least one partition entry moves
    assert not np.array_equal(on["hd_std"]["tau_qa"], off["hd_std"]["tau_qa"])


def test_preview_matches_the_full_runs_finest_scale_with_interpolate():
    """The progressive-compute value-identity contract holds THROUGH the refinement: the
    finest-scale preview and the full run agree on the float channels too."""
    from dynamix.core.wtmm_backend import run_wtmm2d, run_wtmm2d_preview

    field = _small_field()
    params = dict(_SMALL, interpolate=True)
    full = run_wtmm2d(field, params, out_dir=None)
    prev = run_wtmm2d_preview(field, params)
    e_full, e_prev = full["extrema"][0], prev["extrema"][0]
    for k in ("x", "y", "mod", "x_sub", "y_sub"):
        np.testing.assert_array_equal(e_prev[k], e_full[k], err_msg=k)


def test_wtmm2d_device_passes_the_knob_through():
    """The device declares ``interpolate`` (PRE group) and forwards it -- the result's extrema
    carry the float channels, and the chain product still attaches over them."""
    from dynamix.devices.wtmm import WTMM2D
    from dynamix.model.device import defaults_for

    dev = WTMM2D()
    assert "interpolate" in dev.param_groups["PRE"]
    params = dict(defaults_for(dev), n_oct=2, n_voice=2, interpolate=True)
    res = dev.compute(_small_field(), params)
    assert all("x_sub" in e for e in res["extrema"])
    assert "chain_product" in res


def test_default_engine_setting_forces_the_numpy_path(monkeypatch):
    """The app-wide master engine (set_default_engine) is honored by cwt2d when no per-call
    _engine is passed -- forcing 'numpy' must produce bit-identical output to an explicit
    _engine='numpy' call, and 'auto'/None must restore the auto-detect."""
    from dynamix.core import wtmm_backend as wb

    field = _small_field(24).values
    scales = [2.0, 4.0]
    backend = wb.get_backend("python")
    try:
        wb.set_default_engine("numpy")
        via_default = backend.cwt2d(field, scales)
        explicit = backend.cwt2d(field, scales, _engine="numpy")
        np.testing.assert_array_equal(via_default["mod"], explicit["mod"])
        np.testing.assert_array_equal(via_default["arg"], explicit["arg"])
        with pytest.raises(ValueError):
            wb.set_default_engine("cuda")
    finally:
        wb.set_default_engine("auto")
    assert wb.get_default_engine() is None
