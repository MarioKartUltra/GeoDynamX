# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The separable fractional B-spline as a WTMM smoother: Unser & Blu's symmetric spline as a tensor
product, matched to the Gaussian's width, drawn from the exact series, and less direction-biased than
the isotropic polyharmonic B-spline at the scales the WTMM reads."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import fft_policy
from dynamix.core import smoothers as sm
from dynamix.core.frac_bspline_exact import beta_star, beta_star_derivative


def test_unit_mass_and_the_gaussians_second_order():
    s = sm.smoother("frac_bspline_sep", {"alpha": 3.0})
    assert s.theta_hat(np.zeros(1), np.zeros(1))[0] == pytest.approx(1.0)
    r = np.array([0.05, 0.1])
    for sx, sy in ((r, 0 * r), (r / np.sqrt(2), r / np.sqrt(2))):
        np.testing.assert_allclose(s.theta_hat(sx, sy), np.exp(-r ** 2), atol=2e-4)
    assert s.vanishing_moments == 1 and s.psi_name == "∂θ_α"
    assert s.theta_name == "separable fractional B-spline α = 3"
    with pytest.raises(ValueError, match="alpha"):
        sm.smoother("frac_bspline_sep", {"alpha": 0.5})


def test_theta_hat_is_the_product_of_its_axes():
    s = sm.smoother("frac_bspline_sep", {"alpha": 2.5})
    sx, sy = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41))
    np.testing.assert_allclose(s.theta_hat(sx, sy), s.theta_hat(sx, 0 * sx) * s.theta_hat(0 * sy, sy), rtol=1e-12)


@pytest.mark.parametrize("alpha", [1.0, 1.5, 2.0, 3.0])
def test_the_section_is_the_exact_series_and_does_not_ring(alpha):
    s = sm.smoother("frac_bspline_sep", {"alpha": alpha})
    h = np.sqrt(12.0 / (alpha + 1.0)) * 8.0 * sm._SIGMA_PER_SCALE        # the knot spacing in px
    for deriv, exact in ((0, beta_star), (1, beta_star_derivative)):
        x, y = s.section(8.0, deriv)
        ref = exact(x / h, alpha)
        np.testing.assert_allclose(y, ref / np.abs(ref).max(), atol=1e-9)
    x, y = s.section(8.0, 1)
    k = x / h
    for m in (-2, -1, 0, 1):
        inside = (k > m + 0.15) & (k < m + 0.85)
        assert np.count_nonzero(np.diff(np.sign(np.diff(y[inside]))) != 0) <= 1, (alpha, m)


def _arg_bias(theta_hat, a, n=512, ring=0.5):
    """The sin 4φ part (degrees) of the gradient-angle error on a ring of radius ring·a around the
    isotropic cusp r^0.5, through the transform's own derivative filters."""
    x = np.arange(n) - n // 2
    gx, gy = np.meshgrid(x, x)
    fft = fft_policy.double()
    f_hat = fft.fft2(np.hypot(gx, gy) ** 0.5)
    k = np.fft.fftfreq(n)
    sx, sy = np.meshgrid(a * k, a * k)
    filt = sm.derivative_filters(sx, sy, theta_hat(sx, sy), ("dx", "dy"), dtype=np.float64)
    tx, ty = (np.real(fft.ifft2(f_hat * filt[d])) for d in ("dx", "dy"))
    on = np.abs(np.hypot(gx, gy) - ring * a) < 0.5
    phi = np.arctan2(gy[on], gx[on])
    err = np.angle(np.exp(1j * (np.arctan2(ty[on], tx[on]) - phi)))
    return float(np.degrees(2 * np.mean(err * np.sin(4 * phi))))


def test_the_separable_spline_biases_gradient_directions_less_than_the_polyharmonic():
    """Measured on the isotropic cusp at half a scale from it (a = 16 px, α = 3): the polyharmonic's
    log θ̂ carries γ|ω|⁴ sin²2φ/1440 against the separable's γ|ω|⁴ sin²2φ/5760, so its sin 4φ
    bias is −0.94° against −0.20°; the Gaussian is unbiased. A straight edge is unbiased for all."""
    sep = _arg_bias(sm.smoother("frac_bspline_sep", {"alpha": 3.0}).theta_hat, 16.0)
    poly = _arg_bias(sm.smoother("frac_bspline", {"alpha": 3.0}).theta_hat, 16.0)
    gauss = _arg_bias(sm.smoother("gaussian").theta_hat, 16.0)
    assert abs(gauss) < 0.01 and abs(sep) < 0.35 and abs(poly) > 0.8 and abs(poly) > 3 * abs(sep)


def test_the_anisotropy_at_the_band_peak_is_the_fourth_order_prediction():
    alpha, gamma = 3.0, 4.0
    ha = np.sqrt(12.0 / gamma) * sm._SIGMA_PER_SCALE
    for name, denominator in (("frac_bspline_sep", 5760.0), ("frac_bspline", 1440.0)):
        s = sm.smoother(name, {"alpha": alpha})
        grid = np.linspace(1e-3, 3.0, 30001)
        peak = grid[np.argmax(grid * s.theta_hat(grid, 0 * grid))]
        d = peak / np.sqrt(2.0)
        ratio = s.theta_hat(np.array([d]), np.array([d]))[0] / s.theta_hat(np.array([peak]), np.zeros(1))[0]
        w = 2.0 * np.pi * peak * ha
        assert ratio == pytest.approx(np.exp(gamma * w ** 4 / denominator), abs=2e-3), name
