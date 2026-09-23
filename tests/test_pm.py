# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""pm core: Perona-Malik 1990 anisotropic diffusion, the paper's own discrete scheme.

Scheme (7)+(8)+(10) of Perona & Malik, IEEE PAMI 12(7):629-639 (1990): 4-nearest-neighbor
flux with conduction coefficients g(|neighbor difference|) per arc, lambda <= 1/4,
adiabatic boundaries. The pins below are the paper's OWN claimed properties: the discrete
maximum principle (their eq. 11), total-brightness conservation (their scheme-(10)
property), reduction to linear diffusion at K -> inf, and edge sharpening for slopes past
the flux peak (their section IV-B).
"""
from __future__ import annotations

import numpy as np

from dynamix.core import cdf, pm


def _noise(n=32, seed=7):
    return np.random.default_rng(seed).normal(size=(n, n))


def _mollified_step(n=64, presmooth=3):
    f = np.zeros((n, n))
    f[:, n // 2:] = 1.0
    I, _ = pm.pm_evolve(f, k=1e12, n_iter=presmooth, lam=0.2)  # linear limit = heat eq
    return I


def test_max_principle():
    """Paper eq. (11): no new extrema -- the evolution stays inside the initial range."""
    f = _noise()
    for g in ("exp", "frac"):
        I, _ = pm.pm_evolve(f, k=0.5, n_iter=50, g=g, lam=0.25)
        assert I.min() >= f.min() - 1e-12, g
        assert I.max() <= f.max() + 1e-12, g


def test_total_brightness_is_conserved():
    """Scheme (10) preserves the total amount of brightness (adiabatic boundaries)."""
    f = _noise()
    for g in ("exp", "frac"):
        I, _ = pm.pm_evolve(f, k=0.3, n_iter=40, g=g, lam=0.2)
        np.testing.assert_allclose(I.sum(), f.sum(), rtol=0, atol=1e-9)


def test_linear_limit_is_the_heat_equation():
    """K -> inf makes every conduction coefficient exactly 1.0 in float64, so the scheme
    must match the plain 5-point Laplacian evolution (cdf.lap with edge padding) to
    machine precision (the arc-difference flux associates its sum differently than lap's
    ``sum(neighbors) - 4I``, so bit-equality is one ulp too strict)."""
    f = _noise(n=24)
    lam = 0.2
    expected = f.astype(np.float64)
    for _ in range(5):
        expected = expected + lam * cdf.lap(expected)
    for g in ("exp", "frac"):
        I, _ = pm.pm_evolve(f, k=1e12, n_iter=5, g=g, lam=lam)
        np.testing.assert_allclose(I, expected, rtol=0, atol=1e-12, err_msg=g)


def test_edges_sharpen_past_the_flux_peak():
    """Section IV-B: where the slope exceeds the flux peak the diffusion runs backward and
    the edge steepens -- the maximum neighbor difference must GROW, not decay."""
    I0 = _mollified_step()
    before = np.abs(np.diff(I0, axis=1)).max()
    I, _ = pm.pm_evolve(I0, k=0.05, n_iter=30, g="exp", lam=0.2)
    after = np.abs(np.diff(I, axis=1)).max()
    assert after > before


def test_snapshots_and_exact_continuation():
    """The scheme is autonomous: evolving to n1 and continuing to n2 must equal one run to
    n2 exactly, and snapshots must be copies taken AT the requested iteration counts."""
    f = _noise(n=16)
    I_direct, snaps = pm.pm_evolve(f, k=0.4, n_iter=9, snapshots={3, 9})
    assert set(snaps) == {3, 9}
    np.testing.assert_array_equal(snaps[9], I_direct)
    I_mid, _ = pm.pm_evolve(f, k=0.4, n_iter=3)
    np.testing.assert_array_equal(I_mid, snaps[3])
    I_cont, _ = pm.pm_evolve(I_mid, k=0.4, n_iter=6)
    np.testing.assert_array_equal(I_cont, I_direct)


def test_the_two_paper_g_functions_differ():
    f = _noise()
    I_exp, _ = pm.pm_evolve(f, k=0.3, n_iter=10, g="exp")
    I_frac, _ = pm.pm_evolve(f, k=0.3, n_iter=10, g="frac")
    assert not np.array_equal(I_exp, I_frac)


def test_sigma_iters_is_the_linear_limit_schedule():
    """Nominal-sigma schedule: in the K -> inf limit the flow is the heat equation, where
    n iterations at step lam reach effective Gaussian sigma = sqrt(2*n*lam). Inverting,
    n = max(1, round(sigma^2 / (2*lam))) -- the cdf sigma_iters recipe with cos(theta) = 1."""
    assert pm.sigma_iters((1.0, 2.0, 4.0), lam=0.25) == {1.0: 2, 2.0: 8, 4.0: 32}
    assert pm.sigma_iters((0.5,), lam=0.25)[0.5] == 1  # floor at one iteration
