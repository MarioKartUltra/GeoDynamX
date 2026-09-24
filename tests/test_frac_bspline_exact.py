# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The symmetric fractional B-spline evaluated exactly in the time domain (Unser & Blu 1999
eq. 11), and its derivative: exact at the classical anchors, a partition of unity with unit mass
at every order, and the same theta(0) as the Fourier route with its analytic tail."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import frac_bspline as creep
from dynamix.core.frac_bspline_exact import beta_star, beta_star_derivative


def test_alpha_1_is_the_hat_and_its_derivative_the_box_pair():
    x = np.linspace(-2.0, 2.0, 81)
    np.testing.assert_allclose(beta_star(x, 1.0), np.maximum(0.0, 1.0 - np.abs(x)), atol=1e-12)
    inside = (np.abs(x) < 1.0) & (x != 0.0)
    np.testing.assert_allclose(beta_star_derivative(x, 1.0)[inside], -np.sign(x[inside]),
                               atol=1e-12)
    assert np.all(beta_star_derivative(x, 1.0)[np.abs(x) > 1.0] == 0.0)


def test_alpha_3_is_the_cubic_b_spline_where_the_creep_copy_is_exact():
    x = np.linspace(-2.5, 2.5, 101)
    np.testing.assert_allclose(beta_star(x, 3.0), creep._bspline_centered(x, 3), atol=1e-12)
    np.testing.assert_allclose(beta_star_derivative(x, 3.0), creep._bspline_derivative(x, 3, 1),
                               atol=1e-12)


@pytest.mark.parametrize("alpha", [1.5, 2.0, 2.5, 4.0])
def test_every_order_is_a_partition_of_unity(alpha):
    """beta^(2 pi n) = delta_n: sum_k beta(x - k) = 1 (hence unit mass), fractional or not
    (alpha = 2 and 4 take the x^2n log|x| branch, whose series needs the log(K)/K terms)."""
    x = np.array([0.0, 0.13, 0.5, 0.77])
    k = np.arange(-60, 61)
    total = beta_star((x[:, None] - k[None, :]).ravel(), alpha).reshape(len(x), -1).sum(1)
    np.testing.assert_allclose(total, 1.0, atol=2e-5)


@pytest.mark.parametrize("alpha", [1.5, 2.0, 2.5])
def test_the_derivative_matches_finite_differences(alpha):
    x = np.array([0.21, 0.63, 1.37, 2.41])              # away from the knots
    h = 1e-5
    fd = (beta_star(x + h, alpha) - beta_star(x - h, alpha)) / (2 * h)
    np.testing.assert_allclose(beta_star_derivative(x, alpha), fd, atol=1e-6)


@pytest.mark.parametrize("alpha", [0.5, 1.5, 2.5])
def test_theta0_agrees_with_the_fourier_route_and_its_analytic_tail(alpha):
    """Two independent routes to theta_alpha(0) = 2 beta_*(0): the time-domain sum here, and
    mz_edges' Fourier integral closed with its analytic tail."""
    from dynamix.core.mz_edges import _theta0_frac

    assert 2.0 * float(beta_star(np.array([0.0]), alpha)[0]) == pytest.approx(
        _theta0_frac(alpha), rel=1e-5)


def test_non_odd_orders_have_small_sign_changing_tails():
    """Not compactly supported unless alpha is odd; the tails change sign at the knot spacing
    (Unser & Blu's fig. 2 shows them at alpha = 0 and 1/2)."""
    x = np.linspace(1.2, 4.0, 561)
    b = beta_star(x, 1.5)
    assert b.min() < -1e-3 and np.sum(np.diff(np.sign(b)) != 0) >= 3
    assert np.all(beta_star(x[x > 2.0], 3.0) == 0.0)
