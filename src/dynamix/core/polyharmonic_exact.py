# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The isotropic polyharmonic B-spline of order γ in the plane and its x-derivative, evaluated in
the SPACE domain (Van De Ville, Blu & Unser, "Isotropic polyharmonic B-splines: scaling functions
and wavelets", IEEE Trans. Image Process. 14(11), 1798, 2005), knots on the integer lattice:

    β_γ(x) = Σ_{k ∈ Z²} p_γ[k] ρ_γ(x − k),

ρ_γ the Green's function of (−Δ)^(γ/2) in the plane (ρ̂_γ(ω) = ‖ω‖^(−γ)),

    ρ_γ(x) = Γ(1 − γ/2) / (2^γ π Γ(γ/2)) ‖x‖^(γ−2)                  (γ not even),
    ρ_γ(x) = (−1)^n ‖x‖^(2n−2) log‖x‖ / (2^(2n−1) π ((n−1)!)²)        (γ = 2n),

and p_γ[k] the Fourier coefficients of V(ω)^(γ/2), V the fourth-order isotropic localization of the
WTMM smoother (:func:`dynamix.core.smoothers._poly_localization`), so β̂_γ = (V/‖ω‖²)^(γ/2) is that
smoother's θ̂ exactly. For even γ, V^(γ/2) is a trigonometric polynomial of degree γ/2 and the sum
is finite. Otherwise p_γ[k] falls off like ‖k‖^(−(γ+2)); its moments below order γ vanish, so
truncating the sum to |k|∞ ≤ K shifts it by a constant of order K^(−2) and its shape by O(K^(−4)),
and the sums at K, 2K and 4K are combined to cancel both terms (K = 16, 32, 64 above γ = 6, where
the coefficients reach round-off sooner). A γ within 10⁻⁴ of an even integer is taken as that
integer: just off it, the Γ(1 − γ/2) constant and the slowly varying coefficients cancel
catastrophically. Nothing is inverted from the Fourier side, so nothing rings between the knots.
The singularities at the knots are the function's own (∇ρ_γ ~ ‖x‖^(γ−3)): ∂β/∂x is unbounded for
γ < 3, jumps at γ = 3 and has a cusp for 3 < γ < 4; β itself is singular only at γ = 2 (log).
"""
from __future__ import annotations

import functools
import math

import numpy as np

__all__ = ["coefficients", "evaluate"]

#: The truncations whose sums are extrapolated when γ is not even, by γ ≤ 6 and γ > 6.
_TRUNCATIONS = ((32, 64, 128), (16, 32, 64))

#: How close to an even integer γ is taken as that integer.
_EVEN_SNAP = 1e-4


def _snap(gamma: float) -> float:
    """γ, or the even integer within :data:`_EVEN_SNAP` of it."""
    even = 2.0 * round(gamma / 2.0)
    return even if abs(gamma - even) < _EVEN_SNAP else gamma


def _even(gamma: float) -> bool:
    return gamma == 2.0 * round(gamma / 2.0)


@functools.lru_cache(maxsize=16)
def coefficients(gamma: float, n: int = 512) -> np.ndarray:
    """p_γ[k] for |k₁|, |k₂| < n/2, centred at index n/2: the inverse FFT of V^(γ/2) on an n × n
    grid of [−π, π)², whose aliasing error is O(n^(−(γ+2)))."""
    from dynamix.core import fft_policy
    from dynamix.core.smoothers import _poly_localization

    w = 2.0 * np.pi * np.fft.fftfreq(n)
    w1, w2 = np.meshgrid(w, w, indexing="ij")
    p = np.real(fft_policy.double().ifft2(_poly_localization(w1, w2) ** (0.5 * gamma)))
    return np.fft.fftshift(p)


def _green(dx, r, gamma: float, deriv: int):
    """ρ_γ, or ∂ρ_γ/∂x, at offsets of x-component ``dx`` and norm ``r``; the value at r = 0 is
    the limit where it is finite."""
    if _even(gamma):
        n = round(gamma / 2.0)
        c = (-1) ** n / (2 ** (2 * n - 1) * math.pi * math.factorial(n - 1) ** 2)
        log = np.log(np.where(r > 0, r, 1.0))
        out = c * (r ** (2 * n - 2) * log if not deriv else dx * r ** (2 * n - 4) * ((2 * n - 2) * log + 1.0))
        regular = gamma > (3.0 if deriv else 2.0)
    else:
        c = math.gamma(1.0 - gamma / 2.0) / (2.0 ** gamma * math.pi * math.gamma(gamma / 2.0))
        out = c * (r ** (gamma - 2.0) if not deriv else (gamma - 2.0) * dx * r ** (gamma - 4.0))
        regular = gamma > (3.0 if deriv else 2.0)
    return np.where(r > 0, out, 0.0) if regular else out


def _lattice_sum(x, y, gamma: float, deriv: int, K: int, p) -> np.ndarray:
    """Σ_{|k|∞ ≤ K} p_γ[k] ρ_γ((x, y) − k), or its x-derivative."""
    c = p.shape[0] // 2
    ks = np.arange(-K, K + 1)
    k1, k2 = (a.ravel() for a in np.meshgrid(ks, ks, indexing="ij"))
    weights = p[c - K:c + K + 1, c - K:c + K + 1].ravel()
    out = np.empty(x.size)
    for i in range(0, x.size, 256):
        dx = x[i:i + 256, None] - k1
        out[i:i + 256] = _green(dx, np.hypot(dx, y[i:i + 256, None] - k2), gamma, deriv) @ weights
    return out


def evaluate(x, y, gamma: float, deriv: int = 0) -> np.ndarray:
    """β_γ(x, y), or ∂β_γ/∂x when ``deriv`` is 1, in knot units (unit mass, knot spacing 1)."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    shape = x.shape
    x, y = x.ravel(), y.ravel()
    gamma = _snap(float(gamma))
    p = coefficients(gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        if _even(gamma):
            out = _lattice_sum(x, y, gamma, deriv, round(gamma / 2.0) + 1, p)
        else:
            a, b, c = (_lattice_sum(x, y, gamma, deriv, K, p) for K in _TRUNCATIONS[gamma > 6.0])
            out = (64.0 * c - 20.0 * b + a) / 45.0
    return out.reshape(shape)
