# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The 2-D Fourier transform of the Tsallis q-Gaussian in closed form, stable at every q -- pure
numpy, scipy imported lazily, NO GUI.

For the radial ``f(rho) = e_q^(-beta rho^2)`` the 2-D transform (the 2-D analogue of Borges,
Tsallis, Miranda & Andrade 2004's eq. 19, i.e. the Hankel transform
``int_0^inf f(rho) J0(k rho) rho drho``) is, with the mass ``F(0) = 1/(2 beta (2 - q))``:

* ``q > 1``: ``F(0) M_mu(k/s)``, ``mu = 1/(q-1) - 1``, ``s = sqrt((q-1) beta)``, where M is the
  Matern function ``2^(1-mu) z^mu K_mu(z) / Gamma(mu)`` (Gradshteyn & Ryzhik 6.565.4). From
  q = 2 the mass is infinite (mu <= 0) and F is finite only for k > 0.
* ``q < 1``: ``F(0) Lambda_lam(k b)``, ``lam = 1/(1-q) + 1``, ``b = 1/sqrt((1-q) beta)``, where
  Lambda is the Jahnke-Emde function ``Gamma(lam+1) (2/w)^lam J_lam(w)`` (Sonine's integral).
* ``q = 1``: ``exp(-k^2/(4 beta)) / (2 beta)``.

Both orders grow like 1/|q - 1|. The literal products then over- or underflow in double
precision at small argument (already at q = 0.99 and 1.01), although M lies in [0, 1] and Lambda
in [-1, 1]. Each is evaluated in log space where scipy's Bessel function is representable, by
its small-argument series where it is not, and by Debye's uniform asymptotic expansion (DLMF
10.19, 10.41) at large order, where the Gamma function, the power and the Bessel function cancel
analytically instead of numerically.
"""
from __future__ import annotations

import functools

import numpy as np

__all__ = ["q_gaussian_ft"]

#: The K family (q > 1) uses Debye's expansion from this order up, at every argument; the
#: truncation after U_6 is then below 1e-15.
_DEBYE_K_ORDER = 50.0
#: The J family (q < 1) uses it from this order up, before the turning point, where the terms
#: shrink fast (p^3 <= lam / 500) or J underflows; the small-argument series covers the
#: underflow below this order.
_DEBYE_J_ORDER = 500.0
#: A Bessel value below this is treated as underflowed.
_TINY = 1e-280


@functools.lru_cache(maxsize=1)
def _debye_polys() -> tuple:
    """Debye's polynomials U_0 .. U_6 in p, from DLMF 10.41.10:
    ``U_(k+1) = p^2 (1 - p^2) U_k' / 2 + int_0^p (1 - 5 t^2) U_k dt / 8``."""
    from numpy.polynomial import Polynomial

    p = Polynomial([0.0, 1.0])
    polys = [Polynomial([1.0])]
    for _ in range(6):
        u = polys[-1]
        polys.append(0.5 * p ** 2 * (1 - p ** 2) * u.deriv()
                     + ((1 - 5 * p ** 2) * u).integ() / 8.0)
    return tuple(polys)


def _debye_sum(order: float, p: np.ndarray, sign: float) -> np.ndarray:
    """``sum_k sign^k U_k(p) / order^k``: -1 for K, +1 for J."""
    total = np.zeros_like(p)
    for k, u in enumerate(_debye_polys()):
        total += sign ** k * u(p) / order ** k
    return total


def _stirling_tail(nu: float) -> float:
    """``lnGamma(nu) - [(nu - 1/2) ln nu - nu + ln(2 pi)/2]`` by Stirling's series (nu >= 50:
    the first omitted term is below 1e-18)."""
    r2 = 1.0 / (nu * nu)
    return (1.0 / 12.0 - r2 * (1.0 / 360.0 - r2 * (1.0 / 1260.0 - r2 / 1680.0))) / nu


def _matern(mu: float, z) -> np.ndarray:
    """``M_mu(z) = 2^(1-mu) z^mu K_mu(z) / Gamma(mu)`` for mu > 0, z >= 0: 1 at z = 0,
    decreasing to 0. The normalised 2-D transform of the q > 1 Gaussian."""
    from scipy.special import gammaln, kve

    z = np.asarray(z, dtype=np.float64)
    out = np.ones_like(z)
    pos = z > 0.0
    zp = z[pos]
    if mu >= _DEBYE_K_ORDER:
        # K_mu(mu x) ~ sqrt(pi/(2 mu)) e^(-mu eta) (1 + x^2)^(-1/4) sum (-1)^k U_k(p) / mu^k,
        # p = 1/sqrt(1 + x^2). With Stirling's lnGamma(mu) every mu ln mu cancels, leaving
        # ln M = mu g(x) - ln(1 + x^2)/4 + ln sum - stirling_tail(mu), where
        # g = 1 - w + ln((1 + w)/2), w = sqrt(1 + x^2), is written without cancellation.
        x2 = (zp / mu) ** 2
        w = np.sqrt(1.0 + x2)
        g = -x2 / (1.0 + w) + np.log1p(x2 / (2.0 * (1.0 + w)))
        s = _debye_sum(mu, 1.0 / w, -1.0)
        out[pos] = np.exp(mu * g - 0.25 * np.log1p(x2) + np.log(s) - _stirling_tail(mu))
        return out
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        kz = kve(mu, zp)                                   # K_mu(z) e^z: never underflows
        v = np.exp((1.0 - mu) * np.log(2.0) - gammaln(mu) + mu * np.log(zp) + np.log(kz) - zp)
    over = ~np.isfinite(kz)
    if over.any():
        # K overflows only where z is tiny against sqrt(mu), and there the series
        # 1 - t/(mu-1) + t^2/(2 (mu-1)(mu-2)) is exact to double precision (the z^(2 mu) part is
        # far below it). At mu <= 2 the overflow needs z < 1e-150, so M = 1.
        t = zp[over] ** 2 / 4.0
        v[over] = (1.0 - t / (mu - 1.0) + t * t / (2.0 * (mu - 1.0) * (mu - 2.0))
                   if mu > 2.0 else 1.0)
    out[pos] = v
    return out


def _je_lambda(lam: float, w) -> np.ndarray:
    """``Lambda_lam(w) = Gamma(lam+1) (2/w)^lam J_lam(w)`` (Jahnke-Emde) for lam > 0, w >= 0: 1
    at w = 0. The normalised 2-D transform of the compactly supported q < 1 Gaussian."""
    from scipy.special import gammaln, jv

    w = np.asarray(w, dtype=np.float64)
    out = np.ones_like(w)
    pos = w > 0.0
    wp = w[pos]
    x = wp / lam
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        j = jv(lam, wp)
        v = np.sign(j) * np.exp(gammaln(lam + 1.0) + lam * (np.log(2.0) - np.log(wp))
                                + np.log(np.abs(j)))
    before = x < 1.0                         # before the turning point, where J > 0
    under = before & (np.abs(j) < _TINY)
    if lam >= _DEBYE_J_ORDER:
        one_minus = np.where(before, 1.0 - x * x, 1.0)
        debye = before & (under | (one_minus ** -1.5 <= lam / 500.0))
        if debye.any():
            # J_lam(lam x) ~ e^(lam xi) / (sqrt(2 pi lam) (1 - x^2)^(1/4)) sum U_k(p) / lam^k,
            # p = 1/sqrt(1 - x^2) (DLMF 10.19.3 with x = sech alpha). With Stirling's
            # lnGamma(lam + 1): ln Lambda = lam h(x) + stirling_tail(lam) - ln(1 - x^2)/4
            # + ln sum, h = v - 1 - ln((1 + v)/2), v = sqrt(1 - x^2).
            x2 = x[debye] ** 2
            vv = np.sqrt(1.0 - x2)
            h = -x2 / (1.0 + vv) - np.log1p(-x2 / (2.0 * (1.0 + vv)))
            s = _debye_sum(lam, 1.0 / vv, 1.0)
            v[debye] = np.exp(lam * h + _stirling_tail(lam) - 0.25 * np.log1p(-x2) + np.log(s))
        under &= ~debye
    if under.any():
        # Below the Debye order the underflow sits at t = w^2/4 < ~4 (lam + 1), where the
        # series sum (-t)^m / (m! (lam+1)...(lam+m)) converges fast and loses < 2 digits.
        t = wp[under] ** 2 / 4.0
        term = np.ones_like(t)
        total = np.ones_like(t)
        for m in range(1, 80):
            term *= -t / (m * (lam + m))
            total += term
            if np.all(np.abs(term) < 1e-18 * np.abs(total)):
                break
        v[under] = total
    out[pos] = v
    return out


def q_gaussian_ft(k, q: float, beta: float = 0.5) -> np.ndarray:
    """``F(k) = int_0^inf e_q^(-beta rho^2) J0(k rho) rho drho``: the 2-D Fourier transform of
    the radial q-Gaussian (compactly supported below q = 1, Tsallis's ``[.]_+``), for any real q
    and beta > 0, at every |k|. ``F(0) = 1/(2 beta (2 - q))`` is the mass. From q = 2 the mass is
    infinite and only k > 0 is finite; the 2-D q-Mexican hat's envelope, a q'-Gaussian with
    q' = 1/(2 - q), is such a q-Gaussian once q > 3/2."""
    k = np.abs(np.asarray(k, dtype=np.float64))
    q, beta = float(q), float(beta)
    if q == 1.0:
        return np.exp(-k * k / (4.0 * beta)) / (2.0 * beta)
    if q < 1.0:
        lam = 1.0 / (1.0 - q) + 1.0
        return _je_lambda(lam, k / np.sqrt((1.0 - q) * beta)) / (2.0 * beta * (2.0 - q))
    mu = 1.0 / (q - 1.0) - 1.0
    z = k / np.sqrt((q - 1.0) * beta)
    if mu > 0.0:
        return _matern(mu, z) / (2.0 * beta * (2.0 - q))
    from scipy.special import gamma, kv

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(z > 0.0, z ** mu * kv(mu, z)
                        / (2.0 ** mu * gamma(mu + 1.0) * (q - 1.0) * beta), np.inf)
