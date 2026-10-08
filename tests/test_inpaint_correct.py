# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The pre-fill, the known-pixel correction and the readouts."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")
pytest.importorskip("pywt")
pytest.importorskip("scipy")

from dynamix.core import dwt  # noqa: E402
from dynamix.core.inpaint import correct as co  # noqa: E402
from dynamix.core.inpaint import readouts as ro  # noqa: E402

rng = np.random.default_rng(0)


def _smooth(ny=96, nx=96, sigma=6.0, seed=0):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(np.random.default_rng(seed).standard_normal((ny, nx)), sigma) * 50.0


def _box(shape, r0, r1, c0, c1):
    m = np.zeros(shape, dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def _grad_mag(x):
    gy, gx = np.gradient(x)
    return np.hypot(gy, gx)


def test_boundary_ring_is_one_pixel_each_side_of_the_edge():
    m = _box((20, 20), 5, 10, 5, 12)
    ring = co.boundary_ring(m)
    assert ring[4, 8] and ring[5, 8] and not ring[6, 8] and not ring[3, 8]
    assert ring[7, 4] and ring[7, 5] and ring[7, 11] and ring[7, 12] and not ring[7, 13]
    assert not co.boundary_ring(np.zeros((8, 8), dtype=bool)).any()


def test_prefill_fills_holes_without_a_step_and_keeps_clear_pixels():
    y = _smooth() + 100.0
    holes = _box(y.shape, 30, 60, 40, 70)
    holes[10, 10] = True                                   # a lone non-finite pixel
    y_nan = y.copy()
    y_nan[10, 10] = np.nan
    x = co.prefill(y_nan, holes)
    assert np.isfinite(x).all()
    ring = co.boundary_ring(holes)
    clear = ~holes & ~ring
    np.testing.assert_array_equal(x[clear], y[clear])
    assert _grad_mag(x)[ring].max() <= 1.5 * _grad_mag(y)[clear].max()
    prior = np.full(y.shape, 100.0)
    prior[30:60, 40:70] = y[30:60, 40:70] + 3.0           # a value prior: its shape is used
    xp = co.prefill(y_nan, holes, prior)
    # the rebuild from the gradient keeps the prior's shape and sets its level by the
    # clear-pixel mean match, so the prior's constant offset does not survive (the engine
    # fits the prior's affine (a, b) on the clear overlap before the pre-fill)
    assert abs(xp[45, 55] - y[45, 55]) < 1.0 < abs(x[45, 55] - y[45, 55])
    assert abs(x[45, 55] - 100.0) < 20.0


def test_the_correction_takes_a_smooth_difference_off_the_known_pixels_without_a_step():
    y = dwt.mirror(_smooth())
    region = dwt.mirror(_box((96, 96), 30, 60, 40, 70)) > 0
    known = ~region
    ring = co.boundary_ring(region)
    bump = dwt.mirror(_smooth(sigma=8.0, seed=3)) * 0.2
    x = y + bump
    before = ro.errors(x, y, known)["var_err"]
    s = co.correct(x, y, known, ring, pad=0)
    x2 = x - s
    after = ro.errors(x2, y, known)["var_err"]
    assert after < 0.2 * before                             # measured 0.014-0.133 over sigma 4-16 px, 3 seeds
    assert _grad_mag(x2)[ring].max() <= 1.5 * max(_grad_mag(y)[known].max(), _grad_mag(x)[known].max())
    assert abs(x2[known].mean() - y[known].mean()) < 1e-6 * abs(y).max()


def test_heldout_blocks_are_drawn_on_valid_pixels_apart_and_reproducibly():
    valid = np.ones((128, 128), dtype=bool)
    valid[40:80, 40:80] = False
    a = ro.heldout(valid, 8, 16, 16, np.random.default_rng(0))
    b = ro.heldout(valid, 8, 16, 16, np.random.default_rng(0))
    np.testing.assert_array_equal(a, b)
    assert a.sum() % 256 == 0 and 7 * 256 <= a.sum() <= 8 * 256   # the greedy draw places 7 or 8
    assert not a[40:80, 40:80].any()
    assert ro.heldout(valid, 0, 16, 16, np.random.default_rng(0)).sum() == 0
    few = ro.heldout(np.ones((40, 40), dtype=bool), 8, 16, 16, np.random.default_rng(1))
    assert 0 < few.sum() < 8 * 256                         # as many as fit, never an error


def test_errors_and_the_error_map():
    y = rng.standard_normal((32, 32))
    x = y + 0.1
    where = np.zeros((32, 32), dtype=bool)
    where[:16] = True
    e = ro.errors(x, y, where)
    assert e["n"] == 512 and e["var_err"] == pytest.approx(0.01 / y[where].var())
    assert e["rms_rel"] == pytest.approx(0.1 / (np.quantile(y, 0.995) - np.quantile(y, 0.005)))
    m = ro.error_map(x, y, where)
    assert np.isnan(m[20, 0]) and m[3, 3] == pytest.approx(0.1)
    assert ro.errors(x, y, np.zeros((32, 32), dtype=bool))["n"] == 0


def test_hist_distance_is_the_kolmogorov_distance():
    a = rng.normal(0, 1, 4000)
    b = rng.normal(0.5, 1, 4000)
    d = ro.hist_distance(a, b)
    assert 0.15 < d < 0.25                                  # Phi(0.25) - Phi(-0.25) = 0.197
    assert ro.hist_distance(a, a) == 0.0


def test_band_hists_compare_inside_against_clear_per_level_and_orientation():
    from dynamix.core.inpaint import priors as pr

    y = rng.standard_normal((128, 128)).cumsum(0).cumsum(1)
    region = _box(y.shape, 40, 88, 40, 88)
    coeffs = dwt.pyramid(y, "sym7", 4)
    rho = pr.rho_pyramid(region, "sym7", 4)
    clear = pr.clear_masks(region, "sym7", 4)
    h = ro.band_hists(coeffs, rho, clear)
    assert set(h) == {(j, o) for j in (1, 2, 3) for o in (0, 1, 2)}
    one = h[(1, 0)]
    assert set(one) == {"eta", "amp"}
    assert set(one["eta"]) == {"edges", "inside", "clear", "distance", "n_inside", "n_clear"}
    assert 0.0 <= one["eta"]["distance"] <= 1.0 and one["amp"]["n_clear"] > 0
