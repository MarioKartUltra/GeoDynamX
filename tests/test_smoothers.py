# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The WTMM smoother registry (core/smoothers.py): every smoothing function θ the 2-D WTMM can
take, as its k-space form at a scale, with the derivative filters built from it generically,
its x-axis section and its λ*. The Gaussian pair must be the backend's own filters bit for bit;
the fractional B-spline and the q-Gaussian are matched to the Gaussian's σ at the same scale."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import smoothers as sm
from dynamix.core.wtmm_backend import _build_wavelet_filters_numpy

NAMES = ["dx", "dy", "dxx", "dxy", "dyy", "dxxx", "dxxy", "dxyy", "dyyy"]


def _grid(n=48, scale=5.0):
    k = np.fft.fftfreq(n).astype(np.float32)
    kx, ky = np.meshgrid(k, k)
    return kx, ky, (kx * scale).astype(np.float32), (ky * scale).astype(np.float32)


@pytest.mark.parametrize("name", ["gaussian", "mexican"])
def test_the_gaussian_pair_is_the_backends_own_filters(name):
    kx, ky, sx, sy = _grid()
    s = sm.smoother(name)
    ref = _build_wavelet_filters_numpy(kx, ky, 5.0, wavelet=name)
    got = sm.derivative_filters(sx, sy, s.theta_hat(sx, sy), NAMES)
    for key in NAMES:
        np.testing.assert_allclose(got[key], ref[key], rtol=1e-6, atol=1e-7)
    assert got["dx"].dtype == np.complex64


def test_the_fractional_bspline_matches_the_gaussians_width_at_the_same_scale():
    """The matched-σ convention: θ̂ = 1 at k = 0 and the curvature at 0 is the Gaussian's,
    so to second order in s the two smoothers agree; the derivative needs alpha >= 1."""
    s = sm.smoother("frac_bspline", {"alpha": 3.0})
    assert s.theta_hat(np.zeros(1), np.zeros(1))[0] == pytest.approx(1.0)
    sx = np.array([0.05, 0.1]); sy = np.zeros(2)
    np.testing.assert_allclose(s.theta_hat(sx, sy), np.exp(-sx ** 2), atol=2e-4)
    assert s.theta_name.startswith("fractional B-spline") and "∂θ" in s.psi_name
    with pytest.raises(ValueError, match="alpha"):
        sm.smoother("frac_bspline", {"alpha": 0.5})


def test_the_q_gaussian_at_q_one_is_the_gaussian_and_q_two_is_refused():
    kx, ky, sx, sy = _grid()
    s = sm.smoother("q_gaussian", {"q": 1.0})
    np.testing.assert_allclose(s.theta_hat(sx, sy), np.exp(-(sx ** 2 + sy ** 2)), atol=1e-5)
    wide = sm.smoother("q_gaussian", {"q": 1.5}).theta_hat(sx, sy)
    assert wide.min() >= 0 and wide.max() == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="q"):
        sm.smoother("q_gaussian", {"q": 2.0})
    with pytest.raises(ValueError, match="unknown smoother"):
        sm.smoother("morlet")


def test_sections_are_odd_and_lambda_star_follows_the_gaussian_pair():
    """ψ_x's x-axis section is odd; λ* = a / s* with s* the peak of |s| θ̂(s, 0): √2 a for the
    Gaussian (g1), √(2/3) a for the Mexican (g3), and the matched-σ smoothers near √2 a."""
    a = 8.0
    assert sm.smoother("gaussian").lambda_px(a) == pytest.approx(np.sqrt(2.0) * a, rel=1e-3)
    assert sm.smoother("mexican").lambda_px(a) == pytest.approx(np.sqrt(2.0 / 3.0) * a, rel=1e-3)
    for name, params in (("frac_bspline", {"alpha": 3.0}), ("q_gaussian", {"q": 1.0}),
                         ("q_gaussian", {"q": 0.5}), ("frac_bspline", {"alpha": 1.0})):
        s = sm.smoother(name, params)
        x, y = s.section(a, 1)
        np.testing.assert_allclose(y, -y[::-1], atol=1e-9 * np.abs(y).max())
        assert np.abs(y).max() > 0 and x.size >= 128
        lam = s.lambda_px(a)
        assert np.isfinite(lam) and 0.5 * a < lam < 3.0 * a
    assert sm.smoother("q_gaussian", {"q": 1.0}).lambda_px(a) == pytest.approx(np.sqrt(2.0) * a, rel=2e-2)
    assert sm.smoother("frac_bspline", {"alpha": 3.0}).lambda_px(a) == pytest.approx(np.sqrt(2.0) * a, rel=0.15)
    xs, th = sm.smoother("gaussian").section(a, 0)
    np.testing.assert_allclose(th, th[::-1])                          # θ itself is even


def test_the_polyharmonic_smoother_is_isotropic_to_fourth_order():
    # Van De Ville, Blu & Unser 2005 eq. (12): with the (8/3) correction the expansion of θ̂ is
    # 1 − γ‖ω‖²/24 in every direction, so the axis/diagonal mismatch at radius r is O(r⁴): it
    # shrinks 16-fold when r halves.  (Rabut's elementary localization mismatches at O(r²): 4-fold.)
    s = sm.smoother("frac_bspline", {"alpha": 3.0})

    def mismatch(r):
        axis = s.theta_hat(np.array([r]), np.array([0.0]))[0]
        diag = s.theta_hat(np.array([r / np.sqrt(2)]), np.array([r / np.sqrt(2)]))[0]
        return abs(axis - diag)

    coarse, fine = mismatch(0.2), mismatch(0.1)
    assert coarse > 0
    assert 12.0 < coarse / fine < 20.0
    assert fine < 1e-3


# -- the q-Mexican hat (θ = −Δθ_q, ψ = q-g3) and the q-Meyer smoother -------------------------

def test_the_q_mexican_hat_at_q_one_is_the_mexican_pair_and_its_bounds_hold():
    kx, ky, sx, sy = _grid()
    s = sm.smoother("q_mexican", {"q": 1.0})
    ref = _build_wavelet_filters_numpy(kx, ky, 5.0, wavelet="mexican")
    got = sm.derivative_filters(sx, sy, s.theta_hat(sx, sy), NAMES)
    for key in NAMES:
        np.testing.assert_allclose(got[key], ref[key], rtol=1e-6, atol=1e-7)
    assert "q-g3" in s.psi_name and "q-Mexican" in s.theta_name
    # the smoother is |s|² θ̂_q: the q-Gaussian's transform times the Laplacian's symbol
    g = sm.smoother("q_gaussian", {"q": 1.5}).theta_hat(sx, sy)
    s2 = sx.astype(np.float64) ** 2 + sy.astype(np.float64) ** 2
    np.testing.assert_allclose(sm.smoother("q_mexican", {"q": 1.5}).theta_hat(sx, sy), s2 * g,
                               rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="2/3"):
        sm.smoother("q_mexican", {"q": 0.5})           # q-g3 would jump at the support's edge
    with pytest.raises(ValueError, match="q"):
        sm.smoother("q_mexican", {"q": 2.0})


def test_the_q_mexican_sections_are_the_closed_forms_and_reduce_to_the_mexican_at_q_one():
    a = 8.0
    sigma = a * sm._SIGMA_PER_SCALE
    s1 = sm.smoother("q_mexican", {"q": 1.0})
    x, th = s1.section(a, 0)
    u = x / sigma
    np.testing.assert_allclose(th / np.abs(th).max(), (2.0 - u * u) * np.exp(-u * u / 2.0) / 2.0,
                               atol=1e-9)
    x, psi = s1.section(a, 1)
    g3 = (u ** 3 - 4.0 * u) * np.exp(-u * u / 2.0)
    np.testing.assert_allclose(psi / np.abs(psi).max(), g3 / np.abs(g3).max(), atol=1e-9)
    # q = 1.5: ψ odd, θ even; λ* barely moves (the |s|² symbol fixes the band more than the
    # envelope's tails do: 1.5 % longer than the Mexican's √(2/3) a, measured)
    s = sm.smoother("q_mexican", {"q": 1.5})
    x, psi = s.section(a, 1)
    np.testing.assert_allclose(psi, -psi[::-1], atol=1e-9 * np.abs(psi).max())
    x, th = s.section(a, 0)
    np.testing.assert_allclose(th, th[::-1], atol=1e-9 * np.abs(th).max())
    assert s1.lambda_px(a) == pytest.approx(np.sqrt(2.0 / 3.0) * a, rel=1e-3)
    assert s1.lambda_px(a) < s.lambda_px(a) < 1.1 * s1.lambda_px(a)


def test_the_q_meyer_smoother_is_a_bandlimited_isotropic_taper_matched_to_the_gaussian():
    a = 8.0
    s1 = sm.smoother("q_meyer", {"q": 1.0})
    r = np.linspace(0.0, 4.0, 4001)
    f = s1.theta_hat(r, np.zeros_like(r))
    assert f[0] == pytest.approx(1.0) and f.min() >= 0.0 and f.max() <= 1.0 + 1e-12
    assert np.all(np.diff(f) <= 1e-12)                                   # monotone taper
    assert f[-1] == 0.0 and f[np.searchsorted(r, 0.3)] == pytest.approx(1.0)   # flat band, then 0
    diag = s1.theta_hat(r / np.sqrt(2.0), r / np.sqrt(2.0))
    np.testing.assert_allclose(diag, f, atol=1e-12)                      # radial
    assert s1.lambda_px(a) == pytest.approx(np.sqrt(2.0) * a, rel=2e-3)  # matched at q = 1
    for q in (0.5, 1.5):
        s = sm.smoother("q_meyer", {"q": q})
        g = s.theta_hat(r, np.zeros_like(r))
        assert g[0] == pytest.approx(1.0) and g.min() >= 0.0 and np.all(np.diff(g) <= 1e-12)
        x, psi = s.section(a, 1)
        np.testing.assert_allclose(psi, -psi[::-1], atol=1e-9 * np.abs(psi).max())
        lam = s.lambda_px(a)
        assert np.isfinite(lam) and 0.5 * a < lam < 3.0 * a and lam != pytest.approx(s1.lambda_px(a), rel=1e-3)
    assert "q-Meyer" in s1.theta_name and "∂θ" in s1.psi_name
    for bad in (0.0, 2.0):
        with pytest.raises(ValueError, match="q"):
            sm.smoother("q_meyer", {"q": bad})


def test_the_fractional_sections_below_alpha_three_do_not_ring_between_the_knots():
    # Below α = 3 the spectrum decays as ‖ω‖^(1−γ) and a σ/16 inverse-FFT grid rings between the
    # knots (Gibbs); the section is drawn on a four-times finer grid there. The jumps and spikes AT
    # the knots (ρ^(γ−3)) remain: they are the function.
    for alpha, bound in ((2.5, 0.003), (2.0, 0.006)):
        s = sm.smoother("frac_bspline", {"alpha": alpha})
        x, y = s.section(8.0, 1)
        u = x / (8.0 * sm._SIGMA_PER_SCALE)
        h = np.sqrt(12.0 / (alpha + 1.0))
        between = (u > 0.3 * h) & (u < 0.7 * h)             # away from the knots at 0 and h
        assert np.std(np.diff(y[between])) < bound, alpha
