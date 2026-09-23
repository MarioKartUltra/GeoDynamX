# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Focus regression (Schadner 2026 generalized focus + Mukli 2015 fixed focus) --
implementation per the verified re-derivation (freddie-friday agent, 2026-09-22): the
reduced closed forms in dynamix.core.spectra.focus_regression, candidates = {feasible
interior root, fixed-focus endpoint, monofractal infinity endpoint}, winner by SSE argmin
(never root classification -- the sign rule fails in ~4% of cases), delta computed in the
additive non-negative form, Vieta for the second root (c/a = -sigma_X^2 exactly).
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.spectra import focus_regression


def _cone(x0=9.0, y0=-3.0, n_t=18, n_q=15, seed=None, noise=0.0):
    X = np.linspace(1.5, 6.5, n_t)
    beta = np.linspace(1.4, 0.4, n_q)                # strictly decreasing h_q
    Y = y0 + (X[:, None] - x0) * beta[None, :]
    if noise:
        Y = Y + np.random.default_rng(seed).normal(0.0, noise, Y.shape)
    return X, Y, beta


def test_exact_cone_recovered_to_machine_precision():
    X, Y, beta = _cone(x0=9.0)
    r = focus_regression(X, Y, x0_min=7.0)
    assert r["branch"] == "interior"
    assert r["x0"] == pytest.approx(9.0, abs=1e-9)
    assert r["y0"] == pytest.approx(-3.0, abs=1e-9)
    np.testing.assert_allclose(r["h_q"], beta, atol=1e-10)
    assert r["sse"] == pytest.approx(0.0, abs=1e-16)


def test_interior_matches_brute_force_sse_minimum():
    X, Y, _ = _cone(x0=8.2, seed=1, noise=0.02)
    r = focus_regression(X, Y, x0_min=7.0)
    # dense brute force over x0 (the alternating-minimization SSE at each x0)
    xb = X.mean(); s2 = X.var(); Xc = X - xb
    b = (Xc @ Y) / (Xc @ Xc); yq = Y.mean(axis=0)

    def sse_at(x0):
        xi = xb - x0
        bb = b.mean(); yb = yq.mean()
        bet = (s2 * b + xi * (yq - yb) + xi * xi * bb) / (s2 + xi * xi)
        y0 = yb - xi * bb
        return float(((Y - (y0 + (X[:, None] - x0) * bet[None, :])) ** 2).sum())

    grid = np.linspace(7.0, 40.0, 20001)
    best = min(sse_at(v) for v in grid)
    assert r["sse"] <= best + 1e-9


def test_fixed_focus_endpoint_when_the_root_is_infeasible():
    # true apex INSIDE the feasible bound -> unconstrained optimum sits at x0 < x0_min,
    # so the box constraint pushes the fit to the fixed-focus endpoint
    X, Y, _ = _cone(x0=7.5, seed=2, noise=0.01)
    r = focus_regression(X, Y, x0_min=10.0)
    assert r["branch"] in ("fixed", "mono")
    if r["branch"] == "fixed":
        assert r["x0"] == pytest.approx(10.0)


def test_monofractal_parallel_lines_take_the_infinity_endpoint():
    X = np.linspace(1.5, 6.5, 18)
    b = 0.62
    intercepts = np.linspace(-1.0, 1.0, 11)
    Y = intercepts[None, :] + X[:, None] * b
    r = focus_regression(X, Y, x0_min=7.0)
    assert r["branch"] == "mono"
    np.testing.assert_allclose(r["h_q"], b, atol=1e-12)
    # the infinity endpoint's common slope IS the mean of the naive per-q slopes
    np.testing.assert_allclose(r["h_q"], r["h_q_naive"].mean(), atol=1e-12)
    assert np.isinf(r["x0"])


def test_near_parallel_is_stable_not_crashy():
    X = np.linspace(1.5, 6.5, 18)
    rng = np.random.default_rng(5)
    Y = 0.1 * X[:, None] + rng.normal(0, 1e-12, (18, 9)) + np.linspace(0, 1, 9)[None, :]
    r = focus_regression(X, Y, x0_min=7.0)
    assert np.all(np.isfinite(r["h_q"]))
    assert r["branch"] in ("interior", "fixed", "mono")


def test_monotone_enforcement_engages_on_disordered_columns():
    X, Y, _ = _cone(x0=9.0, seed=3, noise=0.3)     # heavy noise breaks the q-ordering
    r = focus_regression(X, Y, x0_min=7.0)
    assert np.all(np.diff(r["h_q"]) <= 1e-12)      # the deliverable guarantee


def test_weighted_fit_matches_weighted_brute_force():
    X, Y, _ = _cone(x0=8.5, seed=4, noise=0.05)
    w = np.linspace(2.0, 0.5, Y.shape[1])
    r = focus_regression(X, Y, x0_min=7.0, weights=w, enforce_monotone=False)
    xb = X.mean(); s2 = X.var(); Xc = X - xb
    b = (Xc @ Y) / (Xc @ Xc); yq = Y.mean(axis=0)
    sw = w.sum()

    def sse_at(x0):
        xi = xb - x0
        bb = (w * b).sum() / sw; yb = (w * yq).sum() / sw
        bet = (s2 * b + xi * (yq - yb) + xi * xi * bb) / (s2 + xi * xi)
        y0 = yb - xi * bb
        return float((w[None, :] * (Y - (y0 + (X[:, None] - x0) * bet[None, :])) ** 2).sum())

    grid = np.linspace(7.0, 60.0, 30001)
    best = min(sse_at(v) for v in grid)
    assert r["sse"] <= best + 1e-8


def test_guards_reject_incomplete_input():
    X, Y, _ = _cone()
    Y2 = Y.copy(); Y2[3, 4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        focus_regression(X, Y2, x0_min=7.0)
    with pytest.raises(ValueError, match="scales"):
        focus_regression(X[:3], Y[:3], x0_min=7.0)


def test_mse_q_is_the_per_moment_error_curve():
    X, Y, _ = _cone(x0=9.0, seed=6, noise=0.02)
    r = focus_regression(X, Y, x0_min=7.0, enforce_monotone=False)
    assert r["mse_q"].shape == (Y.shape[1],)
    assert r["sse"] == pytest.approx(float(r["mse_q"].sum() * len(X)), rel=1e-9)
