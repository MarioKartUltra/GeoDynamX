# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The 2-D WTMM's smoothing functions, one registry for the transform and for the readings.

The WTMM analyses a field with the gradient of a smoothing function, ψ = ∇θ. Every θ here is
held as its k-space form at a scale, ``theta_hat(sx, sy)`` with ``s = k·a`` in cycles (the
backend's own convention: the Gaussian is exp(−|s|²), so its real-space σ is a/(π√2)), and the
derivative filters the transform convolves with are built from it generically,
``(i s_x)^p (i s_y)^q θ̂``. The same object gives the analyser's x-axis section (the wavelet
scale bar) and its λ* (the bandpass peak wavelength the readings quote), so a smoother is
registered once.

Smoothers:

- ``gaussian`` (ψ = g1) and ``mexican`` (|s|² times the Gaussian; ψ = g3): the backend's pair,
  sections and constants exactly as :mod:`dynamix.core.scale_units` holds them.
- ``frac_bspline``: the isotropic polyharmonic B-spline of order γ = α + 1 (Van De Ville, Blu
  and Unser 2005), θ̂(ω) = (V(ω)/‖ω‖²)^(γ/2) with the fourth-order-isotropic localization
  V(ω) = 4 sin²(ω₁/2) + 4 sin²(ω₂/2) − (8/3) sin²(ω₁/2) sin²(ω₂/2), ω = 2π k h in knot units;
  in one dimension it is Unser and Blu's symmetric fractional B-spline |sinc(ω/2)|^(α+1). A
  separable β(x)β(y) would be anisotropic, and the WTMM reads gradient directions. The knot
  spacing h matches the Gaussian's σ at the same scale (θ̂ ≈ 1 − γ‖hω‖²/24): h = σ √(12/γ). A
  continuous gradient wavelet needs ψ̂ ~ ‖ω‖^(1−γ) integrable in 2-D, γ > 3: α > 2, where one
  dimension needed α > 1.
- ``q_gaussian``: the Tsallis q-Gaussian e_q(−ρ²/2σ²), σ = a/(π√2) (q → 1 the Gaussian exactly),
  its radial transform from :func:`dynamix.core.q_fourier.q_gaussian_ft` tabulated once per q.
  Finite mass in 2-D only for q < 2; a continuous ψ at the compact support's edge (q < 1) needs
  q > 0; the 2-D variance diverges from q = 5/3, so the scale is matched by the parameter σ,
  never by the variance.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Callable

import numpy as np

__all__ = ["NAMES", "Smoother", "derivative_filters", "smoother"]

NAMES = ("gaussian", "mexican", "frac_bspline", "q_gaussian")

#: Real-space Gaussian σ per unit nominal scale, the backend's convention (scale_units).
_SIGMA_PER_SCALE = 1.0 / (np.pi * np.sqrt(2.0))

#: The derivative names the transform asks for and their (p, q) orders.
_ORDERS = {"dx": (1, 0), "dy": (0, 1), "dxx": (2, 0), "dxy": (1, 1), "dyy": (0, 2),
           "dxxx": (3, 0), "dxxy": (2, 1), "dxyy": (1, 2), "dyyy": (0, 3)}


def derivative_filters(sx, sy, theta_hat, names) -> dict:
    """``{name: complex64}`` -- the filters ``(i s_x)^p (i s_y)^q θ̂`` for the derivative names,
    written as explicit real and imaginary parts order by order, the backend's structure:
    first order imaginary +, second real −, third imaginary −."""
    sx = np.asarray(sx, dtype=np.float32)
    sy = np.asarray(sy, dtype=np.float32)
    th = np.asarray(theta_hat, dtype=np.float32)
    zero = np.zeros_like(th)
    out = {}
    for name in names:
        p, q = _ORDERS[name]
        mag = th
        for _ in range(p):
            mag = sx * mag
        for _ in range(q):
            mag = sy * mag
        n = p + q
        if n == 1:
            real, imag = zero, mag
        elif n == 2:
            real, imag = -mag, zero
        else:
            real, imag = zero, -mag
        out[name] = (real + 1j * imag).astype(np.complex64)
    return out


@dataclasses.dataclass(frozen=True)
class Smoother:
    """One smoothing function: its k-space form at a scale, the analyser's section, its λ*."""
    name: str
    theta_name: str
    psi_name: str
    params: dict
    _theta_hat: Callable
    _section: Callable

    def theta_hat(self, sx, sy) -> np.ndarray:
        """θ̂ at ``s = k·a`` (cycles), 1 at the origin."""
        return np.asarray(self._theta_hat(np.asarray(sx, dtype=np.float64),
                                          np.asarray(sy, dtype=np.float64)), dtype=np.float64)

    def section(self, scale: float, deriv_order: int = 1, n_points: int = 256):
        """``(x_px, y)`` of θ (order 0) or of the x-analyser ψ_x = ∂_x θ (order 1) along the x
        axis at nominal scale ``scale``; amplitude unnormalised."""
        return self._section(float(scale), int(deriv_order), int(n_points))

    def lambda_px(self, scale: float, deriv_order: int = 1) -> float:
        """The bandpass peak wavelength in px: a / s* with s* the peak of |s|^m θ̂(s, 0) along
        the analyser's own axis (m the derivative order), parabolically refined."""
        s = np.linspace(1e-4, 8.0, 16384)
        f = s ** int(deriv_order) * np.abs(self.theta_hat(s, np.zeros_like(s)))
        i = int(np.argmax(f))
        s_star = s[i]
        if 0 < i < s.size - 1:
            y0, y1, y2 = f[i - 1], f[i], f[i + 1]
            d = y0 - 2.0 * y1 + y2
            if d < 0:
                s_star = s[i] + 0.5 * (y0 - y2) / d * (s[i + 1] - s[i])
        return float(scale) / float(s_star) if s_star > 0 else float("nan")


# -- the Gaussian pair: the backend's own formulas, scale_units' own sections ------------------

def _gaussian_hat(sx, sy):
    return np.exp(-(sx * sx + sy * sy))


def _mexican_hat(sx, sy):
    k2 = sx * sx + sy * sy
    return k2 * np.exp(-k2)


def _pair_section(name):
    def section(scale, deriv_order, n_points):
        from dynamix.core.scale_units import kernel_section

        return kernel_section(scale, name, deriv_order, n_points)
    return section


# -- the isotropic polyharmonic B-spline ----------------------------------------------------

def _poly_localization(w1, w2):
    """V(ω): the fourth-order-isotropic localization of the polyharmonic B-spline."""
    s1, s2 = np.sin(0.5 * w1) ** 2, np.sin(0.5 * w2) ** 2
    return 4.0 * s1 + 4.0 * s2 - (8.0 / 3.0) * s1 * s2


def _frac_hat_factory(alpha: float):
    gamma = alpha + 1.0
    h_over_a = np.sqrt(12.0 / gamma) * _SIGMA_PER_SCALE     # the matched-σ knot spacing / a

    def theta_hat(sx, sy):
        w1, w2 = 2.0 * np.pi * sx * h_over_a, 2.0 * np.pi * sy * h_over_a
        w2n = w1 * w1 + w2 * w2
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(w2n > 0, _poly_localization(w1, w2) / np.where(w2n > 0, w2n, 1.0), 1.0)
        return np.clip(ratio, 0.0, None) ** (0.5 * gamma)
    return theta_hat


@functools.lru_cache(maxsize=64)
def _frac_section_u(alpha: float, deriv_order: int) -> tuple:
    """The x-axis section in σ units, from the central row of the 2-D inverse transform of
    (i s_x)^m θ̂ on a fine grid, computed once per order (the kernel is not separable)."""
    theta_hat = _frac_hat_factory(alpha)
    n, du = 1024, 1.0 / 16.0                                  # ±32 σ at σ/16
    kappa = np.fft.fftfreq(n, d=du)                          # cycles per σ
    kx, ky = np.meshgrid(kappa, kappa)
    sx, sy = np.pi * np.sqrt(2.0) * kx, np.pi * np.sqrt(2.0) * ky   # s = k a = κ (a/σ)
    filt = theta_hat(sx, sy) * (1j * sx) ** int(deriv_order)
    row = np.fft.ifft2(filt)[0]
    y = np.real(np.fft.fftshift(row))[1:]                    # drop −n/2: a grid symmetric about 0
    u = (np.arange(1, n) - n // 2) * du
    return u, y / np.abs(y).max()


def _frac_section_factory(alpha: float):
    def section(scale, deriv_order, n_points):
        u, y = _frac_section_u(float(alpha), int(deriv_order))
        return u * scale * _SIGMA_PER_SCALE, y
    return section


# -- the q-Gaussian -------------------------------------------------------------------------

def _e_q(u, q: float):
    """Tsallis's q-exponential e_q(u) = [1 + (1 − q) u]_+^(1/(1−q)), the exponential at q = 1."""
    if q == 1.0:
        return np.exp(u)
    base = np.clip(1.0 + (1.0 - q) * u, 0.0, None)
    return base ** (1.0 / (1.0 - q))


@functools.lru_cache(maxsize=64)
def _q_table(q: float) -> tuple:
    """θ̂ of e_q(−ρ²/2σ²) against s = k a on a fine radial grid: F_q(√2 s)/F_q(0), F_q the 2-D
    transform at β = 1/2 (σ = 1 in its units; ωσ = 2π s σ / a = √2 s)."""
    from dynamix.core.q_fourier import q_gaussian_ft

    s = np.linspace(0.0, 48.0, 1 << 16)
    f = q_gaussian_ft(np.sqrt(2.0) * s, q, 0.5)
    return s, np.asarray(f, dtype=np.float64) / float(f[0])


def _q_hat_factory(q: float):
    if q == 1.0:
        return _gaussian_hat

    def theta_hat(sx, sy):
        s_tab, f_tab = _q_table(q)
        r = np.sqrt(sx * sx + sy * sy)
        return np.interp(r, s_tab, f_tab, right=0.0)
    return theta_hat


def _q_section_factory(q: float):
    def section(scale, deriv_order, n_points):
        sigma = float(scale) * _SIGMA_PER_SCALE
        x = np.linspace(-6.0 * sigma, 6.0 * sigma, max(int(n_points), 513))
        u = -x * x / (2.0 * sigma * sigma)
        theta = _e_q(u, q)
        if deriv_order == 0:
            return x, theta
        # d/dx e_q(u) = e_q(u)^q · u′ (the q-derivative), a function for q > 0.
        return x, -(x / (sigma * sigma)) * theta ** q
    return section


# -- the registry ----------------------------------------------------------------------------

def smoother(name: str, params: dict | None = None) -> Smoother:
    """The smoother ``name`` with its parameters: ``alpha`` for ``frac_bspline`` (≥ 1; a
    continuous ψ in 2-D needs > 2), ``q`` for ``q_gaussian`` (0 < q < 2)."""
    params = dict(params or {})
    if name == "gaussian":
        return Smoother("gaussian", "g0", "g1", {}, _gaussian_hat, _pair_section("gaussian"))
    if name == "mexican":
        return Smoother("mexican", "g2", "g3", {}, _mexican_hat, _pair_section("mexican"))
    if name == "frac_bspline":
        alpha = float(params.get("alpha", 3.0))
        if not alpha >= 1.0:
            raise ValueError(f"frac_bspline needs alpha >= 1 (a continuous psi in 2-D needs "
                             f"alpha > 2), got {alpha!r}")
        return Smoother("frac_bspline", f"fractional B-spline α = {alpha:g}", "∂θ_α",
                        {"alpha": alpha}, _frac_hat_factory(alpha), _frac_section_factory(alpha))
    if name == "q_gaussian":
        q = float(params.get("q", 1.5))
        if not 0.0 < q < 2.0:
            raise ValueError(f"q_gaussian needs 0 < q < 2 (finite mass in 2-D below 2, a "
                             f"continuous psi above 0), got {q!r}")
        return Smoother("q_gaussian", f"q-Gaussian q = {q:g}", "∂θ_q", {"q": q},
                        _q_hat_factory(q), _q_section_factory(q))
    raise ValueError(f"unknown smoother {name!r}; expected one of {NAMES}")
