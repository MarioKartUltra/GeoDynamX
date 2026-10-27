# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The isotropic polyharmonic B-spline in the space domain: the localized Green's-function sum
equals the Fourier definition away from the knots, and its section has no truncation ringing."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import polyharmonic_exact as ph
from dynamix.core import smoothers as sm


def _off_knots(x, margin=0.15):
    return np.abs(x - np.round(x)) > margin


@pytest.mark.parametrize("alpha", [3.0, 4.0])
def test_the_space_domain_sum_is_the_fourier_definition(alpha):
    # alpha 3 has a finite stencil; alpha 4 an infinite one, extrapolated over the truncation.
    # The reference is the σ/64 inverse FFT, whose own truncation error at alpha 3 is 6e-8 for θ
    # and 1.0e-4 for ψ (2.8e-5 at σ/128: it converges toward the space-domain sum); at alpha 4
    # the extrapolated sum keeps a residual of 1.3e-6 in θ's tail whatever the FFT grid.
    gamma, h = alpha + 1.0, np.sqrt(12.0 / (alpha + 1.0))
    for deriv, tol in ((0, 5e-6), (1, 3e-4)):
        u, fft = sm._central_row(sm._frac_hat_factory(alpha), deriv, n=4096, du=1.0 / 64.0, span=4.0)
        exact = ph.evaluate(u / h, 0.0, gamma, deriv)
        ref = np.argmin(np.abs(u - (0.7 if deriv else 0.3)))
        far = _off_knots(u / h)
        assert np.abs(exact / exact[ref] - fft / fft[ref])[far].max() < tol, (alpha, deriv)


def test_the_extrapolated_sum_has_converged():
    x = np.linspace(-2.4, 2.4, 97) + 1e-3
    gamma = 2.5
    p = ph.coefficients(gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = ph._lattice_sum(x, np.zeros_like(x), gamma, 1, 255, p)
    np.testing.assert_allclose(ph.evaluate(x, 0.0, gamma, 1), direct, rtol=0, atol=1e-4 * np.abs(direct).max())


@pytest.mark.parametrize("alpha", [1.5, 2.0, 2.5])
def test_the_section_does_not_ring_between_the_knots(alpha):
    """Between two knots ψ is monotone or has one turning point; the truncated inverse transform
    zig-zags there (its slope changes sign at every grid ripple)."""
    s = sm.smoother("frac_bspline", {"alpha": alpha})
    x_px, y = s.section(8.0, 1)
    h = np.sqrt(12.0 / (alpha + 1.0)) * 8.0 * sm._SIGMA_PER_SCALE
    k = x_px / h
    for m in (-2, -1, 0, 1):
        inside = (k > m + 0.15) & (k < m + 0.85)
        turns = np.count_nonzero(np.diff(np.sign(np.diff(y[inside]))) != 0)
        assert turns <= 1, (alpha, m, turns)


def test_only_an_unbounded_section_is_clipped():
    """ψ below α = 2 is unbounded at the knots: scaled by its main lobe and clipped. A bounded θ or
    ψ (here θ at α = 1.5, whose top sits on the centre knot) keeps its own maximum."""
    x_px, y = sm.smoother("frac_bspline", {"alpha": 1.5}).section(8.0, 1)
    assert np.abs(y).max() == pytest.approx(1.0)
    assert np.count_nonzero(np.abs(y) == 1.0) > 1          # the spikes at the knots are clipped
    x_px, y = sm.smoother("frac_bspline", {"alpha": 1.5}).section(8.0, 0)
    assert np.count_nonzero(np.abs(y) == 1.0) == 1          # one top, nothing flattened
    assert sm.smoother("frac_bspline").psi_name == "∂θ_α"
    assert "polyharmonic" in sm.smoother("frac_bspline").theta_name


def test_absolute_values_and_unit_mass():
    """Not only shapes: β_4 has unit mass and its values are the inverse transform of θ̂ itself."""
    from dynamix.core import fft_policy

    gamma = 4.0
    u = (np.arange(-96, 96) + 0.5) / 8.0                      # knot units, ±12, midpoint rule
    gx, gy = np.meshgrid(u, u)
    assert ph.evaluate(gx, gy, gamma).sum() / 64.0 == pytest.approx(1.0, abs=2e-3)
    n, dx = 2048, 1.0 / 32.0
    w = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    w1, w2 = np.meshgrid(w, w, indexing="ij")
    r2 = w1 * w1 + w2 * w2
    hat = np.where(r2 > 0, (sm._poly_localization(w1, w2) / np.where(r2 > 0, r2, 1.0)) ** (gamma / 2), 1.0)
    field = np.real(fft_policy.double().ifft2(hat)) / dx ** 2
    x = np.array([0.25, 0.5, 1.5])
    np.testing.assert_allclose(ph.evaluate(x, 0.0, gamma), field[(x / dx).astype(int), 0], rtol=1e-4)


def test_an_order_just_off_an_even_integer_is_taken_as_it():
    # Just off γ = 6 the Γ(1 − γ/2) constant and the slowly varying coefficients cancel.
    x = np.linspace(0.05, 2.45, 25)
    np.testing.assert_allclose(ph.evaluate(x, 0.0, 6.0 + 1e-6), ph.evaluate(x, 0.0, 6.0), rtol=0, atol=1e-9)
