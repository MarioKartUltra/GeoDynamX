# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Measurement operators: every one passes <A x, r> = <x, A^T r>; the terms' gradient and
the measured Lipschitz constant bound the cost."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("numba")
pytest.importorskip("pywt")

from dynamix.core import dwt  # noqa: E402
from dynamix.core.inpaint import forward as fw  # noqa: E402

rng = np.random.default_rng(0)


def _adjoint_ok(op, shape_in, tol=1e-10):
    x = rng.standard_normal(shape_in)
    ax = op.apply(x)
    r = rng.standard_normal(ax.shape)
    r[~np.isfinite(ax)] = 0.0
    ax = np.where(np.isfinite(ax), ax, 0.0)
    lhs, rhs = np.vdot(ax, r), np.vdot(x, op.adjoint(r))
    assert abs(lhs - rhs) <= tol * max(abs(lhs), 1.0), (lhs, rhs)


def test_identity_window_blockaverage_and_gain_are_adjoint():
    _adjoint_ok(fw.Identity(), (16, 24))
    _adjoint_ok(fw.Window(3, 5, 10, 12, (16, 24)), (16, 24))
    _adjoint_ok(fw.BlockAverage(6), (36, 48))
    _adjoint_ok(fw.Gain(rng.uniform(0.1, 1.0, (16, 24))), (16, 24))


def test_block_average_six_is_the_mean_and_its_norm_is_one_sixth():
    x = rng.standard_normal((36, 48))
    a = fw.BlockAverage(6).apply(x)
    np.testing.assert_allclose(a, x.reshape(6, 6, 8, 6).mean(axis=(1, 3)))
    L = fw.lipschitz([fw.Term(fw.BlockAverage(6), np.zeros((6, 8)), np.ones((6, 8)))],
                     (36, 48), np.random.default_rng(1))
    assert L == pytest.approx(1.0 / 36, rel=0.15)          # ||A||^2 = (1/f)^2


def test_shift_integer_sub_pixel_and_warp_are_adjoint_and_sample_where_they_say():
    x = rng.standard_normal((20, 30))
    s = fw.shift(2, -3, (20, 30), (20, 30))
    out = s.apply(x)
    np.testing.assert_allclose(out[0:18, 3:30], x[2:20, 0:27])
    assert np.isnan(out[18:, :]).all() and np.isnan(out[:, :3]).all()
    _adjoint_ok(s, (20, 30))
    _adjoint_ok(fw.shift(0.4, -0.3, (20, 30), (20, 30)), (20, 30))
    rows = rng.uniform(0, 19, (7, 9))
    cols = rng.uniform(0, 29, (7, 9))
    _adjoint_ok(fw.Warp(rows, cols, (20, 30)), (20, 30))


def test_detail_is_a_stacked_tight_frame_with_an_exact_adjoint():
    op = fw.Detail("sym7", 3)
    x = rng.standard_normal((64, 64))
    d = op.apply(x)
    assert d.shape == (9, 64, 64)
    _adjoint_ok(op, (64, 64))
    assert op.level_of(0) == 3 and op.level_of(8) == 1 and op.orientation_of(4) == 1
    approx = x - op.adjoint(d)                              # what the details leave: the level-3 approximation
    s = dwt.stationary(x, "sym7", 3)
    only_approx = [s[0]] + [tuple(np.zeros_like(x) for _ in range(3))] * 3
    np.testing.assert_allclose(approx, dwt.stationary_inverse(only_approx, "sym7"), atol=1e-9)


def test_compositions_are_adjoint():
    _adjoint_ok(fw.Compose(fw.BlockAverage(2), fw.shift(1, 1, (32, 32), (32, 32))), (32, 32))
    _adjoint_ok(fw.Compose(fw.Detail("db4", 2), fw.Gain(rng.uniform(0.2, 1.0, (32, 32)))),
                (32, 32))
    rows = rng.uniform(0, 31, (32, 32))
    cols = rng.uniform(0, 31, (32, 32))
    _adjoint_ok(fw.Compose(fw.Detail("db4", 2), fw.Warp(rows, cols, (32, 32))), (32, 32))
    _adjoint_ok(fw.Compose(fw.BlockAverage(2), fw.Window(2, 4, 16, 20, (24, 28))), (24, 28))


def test_a_terms_residual_zeroes_what_is_missing_and_the_gradient_is_the_adjoint_residual():
    x = rng.standard_normal((16, 16))
    data = rng.standard_normal((16, 16))
    data[3, 4] = np.nan
    w = np.ones((16, 16))
    w[5, 5] = 0.0
    term = fw.Term(fw.Identity(), data, w)
    r = term.residual(x)
    assert r[3, 4] == 0.0 and r[5, 5] == 0.0
    g = fw.gradient([term], x)
    expect = np.where(np.isfinite(data), (x - np.nan_to_num(data)) * w, 0.0)
    np.testing.assert_allclose(g, expect)
    assert fw.data_cost([term], x) == pytest.approx(0.5 * (expect ** 2).sum())


def test_the_measured_lipschitz_constant_bounds_the_hessian_on_random_terms():
    terms = [fw.Term(fw.Identity(), np.zeros((32, 32)), rng.uniform(0, 2, (32, 32))),
             fw.Term(fw.BlockAverage(4), np.zeros((8, 8)), rng.uniform(0, 3, (8, 8))),
             fw.Term(fw.Detail("sym7", 2), np.zeros((6, 32, 32)), rng.uniform(0, 1, (6, 32, 32)))]
    L = fw.lipschitz(terms, (32, 32), np.random.default_rng(2))
    zeros = [dataclasses.replace(t, data=np.zeros_like(t.data)) for t in terms]
    basis = np.eye(32 * 32)
    hessian = np.stack([fw.gradient(zeros, e.reshape(32, 32)).ravel() for e in basis], axis=1)
    lam_max = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))[-1]
    assert lam_max <= L <= 1.25 * lam_max


def test_estimate_offset_recovers_integer_and_sub_pixel_shifts():
    own = np.random.default_rng(0)
    base = own.standard_normal((80, 80)).cumsum(0).cumsum(1)
    base -= base.mean()
    for dy0, dx0 in ((2.0, -3.0), (1.5, -2.25), (0.4, 0.7)):
        shifted = fw.shift(dy0, dx0, (80, 80), (80, 80)).apply(base)
        dy, dx = fw.estimate_offset(base, shifted, max_shift=4)
        assert abs(dy - dy0) < 0.2 and abs(dx - dx0) < 0.2, (dy0, dx0, dy, dx)


def test_parallax_warp_is_the_notebooks_displacement():
    dem = np.full((10, 10), 300.0)
    look, pixel = 27.6, 15.0
    op = fw.parallax_warp(dem, look, pixel, (10, 10))
    disp = 300.0 * np.tan(np.radians(look)) / pixel
    np.testing.assert_allclose(op.rows[4, 4], 4 + disp)
    np.testing.assert_allclose(op.cols[4, 4], 4.0)
