# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The 2-D Fourier (Hankel) transform of the Tsallis q-Gaussian e_q^(-beta rho^2), in closed form
at every q.

For q > 1 the transform is a Matern function of k (order mu = 1/(q-1) - 1, a K Bessel), for q < 1
a Jahnke-Emde lambda function (order lam = 1/(1-q) + 1, a J Bessel). Both orders grow like
1/|q - 1|, and the literal products k^mu K_mu and Gamma(lam+1) (2/w)^lam J_lam over- or underflow
in double precision there (q = 0.99 and 1.01 already fail at small k). The evaluators are checked
against the same closed forms at 40 digits, and the closed forms against a direct numerical
Hankel transform of the q-Gaussian.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import q_fourier as qf

mpmath = pytest.importorskip("mpmath")

#: Orders from the heavy-tailed end (q -> 2) through the overflow region to |q - 1| = 1e-4.
_MU = (0.3, 1.0, 1.5, 2.0, 7.3, 49.0, 49.9, 50.0, 51.0, 98.0, 99.0, 170.0, 350.0, 999.0,
       9999.0)
_LAM = (1.001, 1.5, 2.0, 3.7, 49.0, 50.0, 101.0, 102.0, 350.0, 499.0, 500.0, 501.0, 1001.0,
        10001.0)


def _args(order, past=0.0):
    """From far inside the overflow region out to where the function has died away (and, for
    the J family, ``past`` its turning point into the oscillating tail)."""
    return np.geomspace(1e-8, past + 12.0 * np.sqrt(order) + 60.0, 40)


def _matern_mp(mu, z):
    with mpmath.workdps(40):
        mu, z = mpmath.mpf(mu), mpmath.mpf(z)
        return float(2 ** (1 - mu) * z ** mu * mpmath.besselk(mu, z) / mpmath.gamma(mu))


def _lambda_mp(lam, w):
    with mpmath.workdps(40):
        lam, w = mpmath.mpf(lam), mpmath.mpf(w)
        return float(mpmath.gamma(lam + 1) * (2 / w) ** lam * mpmath.besselj(lam, w))


@pytest.mark.parametrize("mu", _MU)
def test_matern_matches_the_closed_form_at_40_digits(mu):
    z = _args(mu)
    ref = np.array([_matern_mp(mu, v) for v in z])
    np.testing.assert_allclose(qf._matern(mu, z), ref, rtol=1e-11, atol=1e-300)


@pytest.mark.parametrize("lam", _LAM)
def test_jahnke_emde_lambda_matches_the_closed_form_at_40_digits(lam):
    # Past the turning point (w > lam) the function is below (2/e)^lam; at lam = 10001 that is
    # e^-3000, zero in double precision, and mpmath's J no longer converges there.
    w = _args(lam, past=lam if lam < 5000 else 0.0)
    ref = np.array([_lambda_mp(lam, v) for v in w])
    err = np.abs(qf._je_lambda(lam, w) - ref)
    assert np.all(err <= 1e-11 * np.abs(ref) + 1e-15), (w[np.argmax(err)], err.max())


def test_both_are_one_at_the_origin():
    for mu in (0.3, 99.0, 9999.0):
        assert qf._matern(mu, np.array([0.0]))[0] == 1.0
    for lam in (1.5, 101.0, 10001.0):
        assert qf._je_lambda(lam, np.array([0.0]))[0] == 1.0


def _hankel_mp(q, beta, k):
    """int_0^inf e_q^(-beta rho^2) J0(k rho) rho drho by quadrature (not the closed form)."""
    with mpmath.workdps(25):
        q, beta, k = mpmath.mpf(q), mpmath.mpf(beta), mpmath.mpf(k)
        if q < 1:
            edge = 1 / mpmath.sqrt((1 - q) * beta)                  # compact support
            f = lambda r: (max(1 - (1 - q) * beta * r ** 2, 0) ** (1 / (1 - q))
                           * mpmath.besselj(0, k * r) * r)
            return float(mpmath.quad(f, mpmath.linspace(0, edge, 8)))
        f = lambda r: ((1 + (q - 1) * beta * r ** 2) ** (-1 / (q - 1))
                       * mpmath.besselj(0, k * r) * r)
        return float(mpmath.quadosc(f, [0, mpmath.inf], omega=k))


@pytest.mark.parametrize("q", (-0.5, 0.5, 0.9, 1.5, 1.9, 2.5))
@pytest.mark.parametrize("beta", (0.5, 1.3))
def test_q_gaussian_ft_is_the_hankel_transform(q, beta):
    for k in (0.3, 1.0, 3.0):
        assert qf.q_gaussian_ft(np.array([k]), q, beta)[0] == pytest.approx(
            _hankel_mp(q, beta, k), rel=1e-9, abs=1e-14)


def test_q_equal_one_is_the_gaussian_and_the_family_is_continuous_there():
    k = np.geomspace(1e-6, 30.0, 50)
    gauss = np.exp(-k * k / 2.0)                                     # beta = 1/2: 2 beta = 1
    np.testing.assert_array_equal(qf.q_gaussian_ft(k, 1.0, 0.5), gauss)
    # The true departure from the Gaussian is ~ k^4 |q - 1| / (32 beta^2) relative: 2e-8 at k = 5.
    near = k <= 5.0
    for q in (1.0 - 1e-9, 1.0 + 1e-9):
        np.testing.assert_allclose(qf.q_gaussian_ft(k[near], q, 0.5), gauss[near], rtol=1e-7)


def test_the_value_at_k_zero_is_the_mass():
    for q in (-0.5, 0.5, 0.99, 1.01, 1.5, 1.9):
        assert qf.q_gaussian_ft(np.array([0.0]), q, 0.7)[0] == pytest.approx(
            1.0 / (2 * 0.7 * (2 - q)), rel=1e-15)
    assert qf.q_gaussian_ft(np.array([0.0]), 2.5, 0.7)[0] == np.inf   # no finite mass from q = 2


def test_every_hundredth_q_is_finite_at_every_k():
    k = np.geomspace(1e-6, 200.0, 300)
    for q in np.round(np.arange(-0.99, 2.995, 0.01), 2):
        assert np.all(np.isfinite(qf.q_gaussian_ft(k, float(q), 0.5))), q
