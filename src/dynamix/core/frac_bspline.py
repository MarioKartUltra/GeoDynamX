# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Fractional B-spline evaluators -- Unser & Blu (SIAM Review 2000), copied VERBATIM from
Creep (``Creep/wavelet/wtmm/wavelets.py``, Parts A/A2; validated there in
``wtmm_cpu.ipynb`` cell 70: fractional orders reproduce the integer B-splines to 0 error).

The provenance chain is Creep -> DynamiX: the four function
bodies below are byte-identical to their Creep originals -- ``tests/test_frac_bspline_port.py``
enforces it (the ``test_mzlib_port`` pattern; skipped when the Creep checkout is absent).
Do not edit them; supersede by adding.

These are the ORACLE for the fractional Mallat-Zhong wavelet (``mz_edges`` §6.4): the
dyadic filter-bank construction (sign-preserving fractional refinement order in
``core/mz_edges.py``) is validated against these real-space forms -- exact match at the
classical integer anchors, |FT| match at fractional orders (the causal-vs-symmetric variant
phase difference; see the mz_edges module).

Only this wrapper block is DynamiX's: Creep imports ``factorial``/``gamma_func`` from scipy
at module top, which would make scipy a hard core dependency (the manifest test forbids it),
so the two names are provided as LAZY shims with identical call semantics -- the recorded
import rewrite, same license as mzlib's one-line rewrite.
"""
from __future__ import annotations

from math import comb

import numpy as np


def factorial(n, exact=False):
    from scipy.special import factorial as _factorial

    return _factorial(n, exact=exact)


def gamma_func(x):
    from scipy.special import gamma as _gamma

    return _gamma(x)


def _bspline_centered(u, m):
    """Centered cardinal B-spline of order m via Cox-de Boor recursion.

    Support: [-(m+1)/2, (m+1)/2].
    Uses the standard recursion B_m(t) with knots at integers 0..m+1,
    then shifts to center at 0.
    """
    # Shift to standard (uncentered) knots: t in [0, m+1]
    t = np.asarray(u, dtype=np.float64) + (m + 1) / 2.0
    result = np.zeros_like(t)

    # Explicit formula: B_m(t) = 1/m! * sum_{k=0}^{m+1} (-1)^k C(m+1,k) * max(t-k,0)^m
    for k in range(m + 2):
        sign = (-1) ** k
        coeff = comb(m + 1, k)
        shifted = np.maximum(t - k, 0.0)
        result = result + sign * coeff * shifted ** m
    result = result / factorial(m, exact=True)
    return result


def _bspline_derivative(u, m, n):
    """n-th derivative of centered B-spline of order m.

    Uses the finite-difference identity:
        B_m^(n)(t) = sum_{k=0}^{n} (-1)^k C(n,k) B_{m-n}(t + n/2 - k)
    """
    result = np.zeros_like(np.asarray(u, dtype=np.float64))
    for k in range(n + 1):
        sign = (-1) ** k
        coeff = comb(n, k)
        result = result + sign * coeff * _bspline_centered(u + n / 2.0 - k, m - n)
    return result


def _frac_bspline_centered(u, alpha):
    """Centered fractional B-spline of continuous order alpha.

    Extends _bspline_centered to real-valued alpha > -1/2.
    Support: [-(alpha+1)/2, (alpha+1)/2].

    Uses the explicit formula:
        beta_alpha(t) = 1/Gamma(alpha+1) * sum_{k=0}^{floor(t)} (-1)^k C_frac(alpha+1, k) * (t-k)^alpha

    where C_frac(alpha+1, k) = Gamma(alpha+2) / (Gamma(k+1) * Gamma(alpha+2-k))
    is the generalized binomial coefficient.

    Reference: Unser & Blu, "Fractional Splines and Wavelets", SIAM Review 2000.
    """
    t = np.asarray(u, dtype=np.float64) + (alpha + 1) / 2.0
    result = np.zeros_like(t)

    # Upper limit for the sum: we need t - k >= 0, so k <= t
    # But also the generalized binomial coeff C_frac(alpha+1, k) decays,
    # and the support is [0, alpha+1], so k goes up to floor(alpha+1)
    k_max = int(np.floor(alpha + 1)) + 1

    for k in range(k_max + 1):
        # Generalized binomial coefficient: C(alpha+1, k) using Gamma
        if k == 0:
            binom_k = 1.0
        else:
            binom_k = gamma_func(alpha + 2) / (gamma_func(k + 1) * gamma_func(alpha + 2 - k))

        sign = (-1.0) ** k
        shifted = np.maximum(t - k, 0.0)
        result = result + sign * binom_k * shifted ** alpha

    result = result / gamma_func(alpha + 1)
    return result


def _frac_bspline_derivative(u, alpha, n_alpha):
    """Fractional derivative of fractional B-spline.

    Computes the n_alpha-th order (possibly non-integer) derivative of
    the fractional B-spline of order alpha, using fractional finite differences:

        psi_alpha^(n)(t) = sum_{k=0}^{ceil(n)} (-1)^k C_frac(n, k) * beta_{alpha-n}(t + n/2 - k)

    For integer n_alpha, this reduces to _bspline_derivative.
    For non-integer n_alpha, uses generalized binomial coefficients.

    The resulting wavelet has n_alpha vanishing moments (fractional).

    Parameters
    ----------
    u : array_like
        Evaluation points (centered coordinates).
    alpha : float
        B-spline order (> -1/2 + n_alpha).
    n_alpha : float
        Derivative order (vanishing moments). Can be non-integer.
    """
    u = np.asarray(u, dtype=np.float64)
    result = np.zeros_like(u)

    # Number of terms: for fractional differences, we need enough terms
    # for convergence. The series C_frac(n, k) * (-1)^k decays for k > n.
    # Use ceil(n) + some extra terms for non-integer n.
    n_terms = int(np.ceil(n_alpha)) + 1

    # Remaining B-spline order
    alpha_rem = alpha - n_alpha

    for k in range(n_terms + 1):
        if k == 0:
            binom_k = 1.0
        else:
            binom_k = gamma_func(n_alpha + 1) / (gamma_func(k + 1) * gamma_func(n_alpha + 1 - k))

        sign = (-1.0) ** k
        result = result + sign * binom_k * _frac_bspline_centered(u + n_alpha / 2.0 - k, alpha_rem)

    return result
