# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The symmetric fractional B-spline and its derivative, evaluated exactly in the TIME domain
(Unser & Blu, "Construction of fractional spline wavelet bases", SPIE 3813, 1999, eq. 11;
SIAM Review 42, 2000):

    beta_*^a(x) = 1/Gamma(a+1) sum_{k in Z} (-1)^k |a+1, k| |x - k|_*^a,

with the re-centred binomials ``|r, k| = C(r, k + r/2)`` (their eq. 12) and the symmetric power
``|x|_*^a = |x|^a / (-2 sin(pi a / 2))`` for ``a`` not even, ``x^(2n) log|x| / ((-1)^(1+n) pi)``
for ``a = 2n`` (eq. 7). Nothing is inverted from the Fourier side, so nothing rings: a jump
(the box pair of ``a = 1``) is a jump.

An odd integer ``a`` gives a finite sum -- the classical centred B-spline, compactly supported.
Any other order is not compact: its tails fall off like ``|x|^-(a+2)`` and change sign at the
knot spacing (their fig. 2). The series terms then decay like ``k^-2`` (``k^-3`` for the
derivative), so the partial sums converge like ``1/K`` (``log K / K`` for the even orders'
logarithms); partial sums at ``K, 2K, 4K, ...`` are combined to eliminate those error terms
exactly (Richardson) -- a finite part closed analytically, the device Unser & Blu use for their
autocorrelation sum. Accurate to ~1e-10 near the centre and ~1e-8 out to ``|x| = 60``.

The verbatim Creep evaluators (:mod:`dynamix.core.frac_bspline`) are exact only at integer
orders (a centred causal spline cut to ``[0, a+1]``); this module is the symmetric spline at
every order ``a > -1/2``.
"""
from __future__ import annotations

from math import gamma, pi, sin

import numpy as np

__all__ = ["beta_star", "beta_star_derivative"]

#: Terms K of the first partial sum (then 2K, 4K, ...).
_K = 1000
#: Evaluation points per block (the term matrix is block x (2 K_max + 1)).
_BLOCK = 128


def _is_even(a: float) -> bool:
    return abs(a / 2.0 - round(a / 2.0)) < 1e-12


def _is_odd(a: float) -> bool:
    return abs((a - 1.0) / 2.0 - round((a - 1.0) / 2.0)) < 1e-12


def _power(d, a: float, derivative: bool):
    """``|d|_*^a`` (eq. 7), or its derivative in ``d``."""
    if _is_even(a):
        n = int(round(a / 2.0))
        c = 1.0 / ((-1.0) ** (1 + n) * pi)
        ad = np.abs(d)
        with np.errstate(divide="ignore", invalid="ignore"):
            logd = np.where(ad > 0, np.log(np.where(ad > 0, ad, 1.0)), 0.0 if n else -np.inf)
            if not derivative:
                return c * np.where(ad > 0, d ** (2 * n) * logd, 0.0 if n else -np.inf)
            return c * np.where(ad > 0, 2 * n * d ** (2 * n - 1) * logd + d ** (2 * n - 1), 0.0)
    c = 1.0 / (-2.0 * sin(pi * a / 2.0))
    if not derivative:
        return c * np.abs(d) ** a
    return c * a * np.sign(d) * np.abs(d) ** (a - 1.0)


def _series(x, a: float, derivative: bool) -> np.ndarray:
    from scipy.special import binom

    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    r = a + 1.0
    if _is_odd(a):                                   # finite: the classical centred B-spline
        half = int(round(r / 2.0))
        k = np.arange(-half, half + 1, dtype=np.float64)
    # Partial sums at K, 2K, 4K, ..., their error terms eliminated exactly: 1/K, 1/K^2, 1/K^3;
    # even orders (the x^2n log|x| terms) carry log(K)/K^j terms too, so they take six sums.
    even = _is_even(a)
    Ks = _K * 2.0 ** np.arange(6 if even else 4)
    basis = ([np.log(Ks) / Ks, 1.0 / Ks, np.log(Ks) / Ks ** 2, 1.0 / Ks ** 2,
              np.log(Ks) / Ks ** 3] if even
             else [1.0 / Ks, 1.0 / Ks ** 2, 1.0 / Ks ** 3])
    weights = np.linalg.inv(np.column_stack([np.ones(Ks.size)] + basis))[0]   # K = inf term
    if not _is_odd(a):
        k = np.arange(-Ks[-1], Ks[-1] + 1, dtype=np.float64)
    coef = (-1.0) ** np.abs(k) * binom(r, k + r / 2.0) / gamma(a + 1.0)
    out = np.empty_like(flat)
    for i0 in range(0, flat.size, _BLOCK):
        xb = flat[i0:i0 + _BLOCK]
        terms = _power(xb[:, None] - k[None, :], a, derivative) * coef[None, :]
        if _is_odd(a):                   # compact: exactly 0 outside |x| < (a+1)/2, no round-off
            out[i0:i0 + _BLOCK] = np.where(np.abs(xb) < r / 2.0, terms.sum(axis=1), 0.0)
            continue
        ak = np.abs(k)
        partial = np.stack([terms[:, ak <= m].sum(axis=1) for m in Ks], axis=1)
        out[i0:i0 + _BLOCK] = partial @ weights
    return out.reshape(x.shape)


def beta_star(x, alpha: float) -> np.ndarray:
    """The symmetric fractional B-spline ``beta_*^alpha(x)`` (knots at the integers, unit
    mass), ``alpha > -1/2``."""
    if not alpha > -0.5:
        raise ValueError(f"fractional B-splines need alpha > -1/2, got {alpha!r}")
    return _series(x, float(alpha), derivative=False)


def beta_star_derivative(x, alpha: float) -> np.ndarray:
    """``d/dx beta_*^alpha(x)``, term by term (``alpha >= 1``; at ``alpha = 1`` the box pair,
    0 at its jumps)."""
    if not alpha >= 1.0:
        raise ValueError(f"the derivative is a function only for alpha >= 1, got {alpha!r}")
    return _series(x, float(alpha), derivative=True)
