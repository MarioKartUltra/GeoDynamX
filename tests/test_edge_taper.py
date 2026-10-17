# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Gaussian taper edge (spec 22 §3.3; Jones & Gunaratne 1998 p. 446: the data tapered near the
boundaries so that the field vanishes there): the band inside the real data at the raster's edges,
its maxima dropped; disabled, with its reason, where the band would not fit."""
from __future__ import annotations

import types

import numpy as np
import pytest

from dynamix.core import edges, fft_policy
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices.wtmm import WTMM2D
from dynamix.model.device import defaults_for, validate_params


def _labyrinth(n, k0=1 / 12, seed=3):
    """Annulus-filtered white noise on a 2n grid, cropped to n: band-limited at about 1/k0 px."""
    m = 2 * n
    k = np.hypot(np.fft.fftfreq(m)[:, None], np.fft.fftfreq(m)[None, :])
    spec = fft_policy.double().fft2(np.random.default_rng(seed).standard_normal((m, m)))
    v = fft_policy.double().ifft2(spec * np.exp(-((k - k0) / (0.2 * k0)) ** 2)).real[:n, :n]
    return (10.0 + v / v.std()).astype(np.float32)


def _field(v):
    n0, n1 = v.shape
    return RasterField(name="f", values=v, frame=LocalFrame(), x_axis=np.arange(float(n1)),
                       y_axis=np.arange(float(n0)))


def _params(**kw):
    p = dict(defaults_for(WTMM2D()), n_oct=2, n_voice=4, a_min=1.0, fracint_alpha=0.0,
             edge="gaussian_taper")
    p.update(kw)
    return validate_params(WTMM2D(), p)


def _dist(y, x, shape):
    y, x = np.asarray(y, float), np.asarray(x, float)
    return np.minimum.reduce([y, x, shape[0] - 1 - y, shape[1] - 1 - x])


def test_the_taper_vanishes_at_the_edge_and_leaves_the_interior():
    H = W = 64
    v = np.add.outer(np.arange(H, dtype=float), np.arange(W, dtype=float))
    mu, w_t = float(v.mean()), 12.0
    rows, cols = np.arange(-3, H + 3), np.arange(W)              # three rows past each edge
    block = np.vstack([np.full((3, W), -99.0), v, np.full((3, W), -99.0)])
    out = edges.tapered(block, rows, cols, (H, W), mu, w_t)
    assert np.all(out[:3] == mu) and np.all(out[-3:] == mu)      # beyond the parent: the mean
    inner = out[3 + 12:3 + H - 12, 12:W - 12]
    assert np.array_equal(inner, v[12:H - 12, 12:W - 12])
    edge_px = out[3, 30]                                          # row 0 of the parent
    assert abs(edge_px - mu) <= 3.4e-4 * abs(v[0, 30] - mu)


def test_the_dominant_wavelength_of_a_band_limited_field():
    assert edges.dominant_wavelength(_labyrinth(256, k0=1 / 16)) == pytest.approx(16, rel=0.15)
    x = np.arange(256.0)
    sine = np.sin(2 * np.pi * x / 20.0)[None, :].repeat(8, axis=0)
    assert edges.dominant_wavelength(sine) == pytest.approx(20, rel=0.05)


def test_the_taper_band_is_dropped_and_recorded():
    v = _labyrinth(128)
    res = WTMM2D().compute(_field(v), _params())
    edge = res["_edge"]
    assert edge["mode"] == "gaussian_taper" and edge.get("fallback") is None
    w_t = edge["taper_px"]
    assert w_t >= edge["dominant_px"] and w_t >= edge["scales"][-1]["radius"]
    b = int(np.ceil(w_t))
    assert all(s["border"] == b for s in edge["scales"])
    for lvl in res["extrema"]:
        assert (_dist(lvl["y"], lvl["x"], v.shape) >= b).all()
    assert res["chains"] and all((_dist(c["y"], c["x"], v.shape) >= b).all() for c in res["chains"])


def test_the_taper_is_disabled_when_narrower_than_the_coarsest_reach():
    """Spec 22 test 7: a 64 px raster with three octaves: r(s_max) ~ 58 px > 64/4."""
    v = _labyrinth(64)
    p = _params(n_oct=3)
    res = WTMM2D().compute(_field(v), p)
    assert res["_edge"]["mode"] == "reflect"
    reason = res["_edge"]["fallback"]
    assert "taper" in reason and "r(s_max)" in reason
    assert WTMM2D().knob_warnings(_field(v), p)["edge"] == reason


def test_a_red_spectrum_field_falls_back_with_its_dominant_wavelength():
    rng = np.random.default_rng(5)
    walk = rng.standard_normal((128, 128)).cumsum(0).cumsum(1).astype(np.float32)
    res = WTMM2D().compute(_field(walk), _params())
    assert res["_edge"]["mode"] == "reflect" and "dominant wavelength" in res["_edge"]["fallback"]


def test_the_roi_taper_fills_beyond_the_parent_with_the_mean(tmp_path):
    pytest.importorskip("rasterio")
    from test_roi_halo import _write_tif
    from dynamix.devices.wtmm_roi import WTMM2DROI

    v = _labyrinth(256)
    tif = _write_tif(tmp_path / "p.tif", v)
    field = RasterField.from_geotiff_window(tif, row_off=0, col_off=0, height=96, width=96)
    dev = WTMM2DROI()
    p = validate_params(dev, dict(defaults_for(dev), n_oct=2, n_voice=4, a_min=1.0,
                                  fracint_alpha=0.0, roi_row=0, roi_col=0, roi_h=64, roi_w=64,
                                  edge="gaussian_taper"))
    res = dev.compute(field, p)
    assert res["_edge"]["mode"] == "gaussian_taper"
    b = int(np.ceil(res["_edge"]["taper_px"]))
    for lvl in res["extrema"]:
        assert (_dist(lvl["y"], lvl["x"], (256, 256)) >= b).all()
    assert not any("mirror" in w for w in res.get("_warnings", []))
