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
  never by the variance. A width β other than ½, e_q(−β ρ²/σ²) = e_q(−½ (ρ √(2β)/σ)²), is the
  same table stretched, θ̂_β(s) = θ̂_½(s/√(2β)); the devices resolve β (fixed, or q-paired so
  the escort variance stays one, ``devices.q_widths``).
- ``q_mexican``: −Δ of that q-Gaussian smoother, |s|² θ̂_q, the q analogue of the Mexican pair
  (ψ = q-g3, in closed form by the chain rule on e_q; Borges's own index is q_B = 2 − 1/q).
  2/3 < q < 2: the q-g3 carries e_q^(3q−2), which must vanish at the support's edge.
- ``q_meyer``: Meyer's radial scaling-function taper with cos(π/2·ν) replaced by the q-cosine
  cos_q(x₀ ν) of Borges et al. 2004 (x₀ its first zero) — the generalization their closing
  remark leaves open. Bandlimited and isotropic; the dilation is fixed at q = 1 so that the
  band peak equals the Gaussian g1's, and q moves it, or q-paired: recomputed per q so the band
  peak stays at the g1's (λ* = √2 a). 0 < q < 2.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Callable

import numpy as np

__all__ = ["NAMES", "Smoother", "derivative_filters", "smoother"]

NAMES = ("gaussian", "mexican", "frac_bspline", "q_gaussian", "q_mexican", "q_meyer")

#: Real-space Gaussian σ per unit nominal scale, the backend's convention (scale_units).
_SIGMA_PER_SCALE = 1.0 / (np.pi * np.sqrt(2.0))

#: The derivative names the transform asks for and their (p, q) orders.
_ORDERS = {"dx": (1, 0), "dy": (0, 1), "dxx": (2, 0), "dxy": (1, 1), "dyy": (0, 2),
           "dxxx": (3, 0), "dxxy": (2, 1), "dxyy": (1, 2), "dyyy": (0, 3)}


def derivative_filters(sx, sy, theta_hat, names, dtype=np.float32) -> dict:
    """``{name: complex}`` -- the filters ``(i s_x)^p (i s_y)^q θ̂`` for the derivative names,
    written as explicit real and imaginary parts order by order, the backend's structure:
    first order imaginary +, second real −, third imaginary −. ``dtype`` is the real precision
    (float32, complex64 filters, the transform's default; float64 at 64-bit FFTs)."""
    sx = np.asarray(sx, dtype=dtype)
    sy = np.asarray(sy, dtype=dtype)
    th = np.asarray(theta_hat, dtype=dtype)
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
        out[name] = (real + 1j * imag).astype(np.result_type(dtype, np.complex64))
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

    @property
    def vanishing_moments(self) -> int:
        """n_ψ, the vanishing moments of ψ = ∇θ: one more than the order of θ̂'s zero at the
        origin (1 for a smoothing θ, θ̂(0) = 1; 3 for the Mexican pairs, θ̂ ~ |s|²). The WTMM
        resolves singularities of Hölder exponent h < n_ψ (Arneodo, Decoster & Roux 2000)."""
        s = np.array([1e-3, 2e-3])
        v = np.abs(self.theta_hat(s, np.zeros_like(s)))
        return 1 + int(round(float(np.log2(v[1] / v[0]))))

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


def _central_row(theta_hat, deriv_order: int, n: int = 1024, du: float = 1.0 / 16.0,
                 span: float = 6.0) -> tuple:
    """The x-axis section in σ units, |u| ≤ ``span``, from the central row of the 2-D inverse
    transform of (i s_x)^m θ̂ on a fine grid (step ``du`` σ over ±n·du/2 σ), for a kernel that
    is not separable or has no closed form."""
    kappa = np.fft.fftfreq(n, d=du)                          # cycles per σ
    kx, ky = np.meshgrid(kappa, kappa)
    sx, sy = np.pi * np.sqrt(2.0) * kx, np.pi * np.sqrt(2.0) * ky   # s = k a = κ (a/σ)
    filt = theta_hat(sx, sy) * (1j * sx) ** int(deriv_order)
    from dynamix.core.fft_policy import double

    row = double().ifft2(filt)[0]
    y = np.real(np.fft.fftshift(row))[1:]                    # drop −n/2: a grid symmetric about 0
    u = (np.arange(1, n) - n // 2) * du
    keep = np.abs(u) <= float(span)
    u, y = u[keep], y[keep]
    return u, y / np.abs(y).max()


@functools.lru_cache(maxsize=64)
def _frac_section_u(alpha: float, deriv_order: int) -> tuple:
    """The fractional B-spline's section, computed once per order (not separable). Below α = 3
    the spectrum decays slowly (ψ̂ ~ ‖ω‖^(1−γ)) and the grid's Nyquist truncation rings between
    the knots, so the grid is four times finer there; the spikes and jumps AT the knots are the
    function itself (its Green's-function derivative ρ^(γ−3)), not the grid."""
    fine = float(alpha) < 3.0
    return _central_row(_frac_hat_factory(alpha), deriv_order, n=4096 if fine else 1024,
                        du=1.0 / 64.0 if fine else 1.0 / 16.0)


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


def _stretched(theta_hat, beta: float):
    """θ̂ at width β from θ̂ at width ½: θ̂_β(s) = θ̂_½(s/√(2β)); width ½ is ``theta_hat`` itself."""
    if beta == 0.5:
        return theta_hat
    c = np.sqrt(2.0 * beta)

    def stretched(sx, sy):
        return theta_hat(sx / c, sy / c)
    return stretched


def _q_hat_factory(q: float):
    if q == 1.0:
        return _gaussian_hat

    def theta_hat(sx, sy):
        s_tab, f_tab = _q_table(q)
        r = np.sqrt(sx * sx + sy * sy)
        return np.interp(r, s_tab, f_tab, right=0.0)
    return theta_hat


def _q_section_factory(q: float, beta: float = 0.5):
    def section(scale, deriv_order, n_points):
        sigma = float(scale) * _SIGMA_PER_SCALE / np.sqrt(2.0 * beta)
        x = np.linspace(-6.0 * sigma, 6.0 * sigma, max(int(n_points), 513))
        u = -x * x / (2.0 * sigma * sigma)
        theta = _e_q(u, q)
        if deriv_order == 0:
            return x, theta
        # d/dx e_q(u) = e_q(u)^q · u′ (the q-derivative), a function for q > 0.
        return x, -(x / (sigma * sigma)) * theta ** q
    return section


# -- the q-Mexican hat: −Δ of the q-Gaussian smoother, its analyser the q-g3 ---------------

def _q_mexican_hat_factory(q: float, beta: float = 0.5):
    envelope = _stretched(_q_hat_factory(q), beta)

    def theta_hat(sx, sy):
        return (sx * sx + sy * sy) * envelope(sx, sy)
    return theta_hat


def _q_mexican_section_factory(q: float, beta: float = 0.5):
    """Closed forms by the chain rule on e_q (each derivative of e_q^a gives a·e_q^(a+q−1)):
    θ = −Δ e_q(−ρ²/2σ²) = (2/σ²) e_q^q − q (ρ²/σ⁴) e_q^(2q−1) and, along x,
    ψ_x = ∂_x θ = (q x/σ⁴) [(2q − 1)(x²/σ²) e_q^(3q−2) − 4 e_q^(2q−1)]; at q = 1 the Mexican
    pair's (2 − u²)e^(−u²/2) and (u³ − 4u)e^(−u²/2). The exponents stay positive for q > 2/3,
    which keeps ψ continuous at the compact support's edge below q = 1. Width β: σ/√(2β)."""
    def section(scale, deriv_order, n_points):
        sigma = float(scale) * _SIGMA_PER_SCALE / np.sqrt(2.0 * beta)
        x = np.linspace(-6.0 * sigma, 6.0 * sigma, max(int(n_points), 513))
        e = _e_q(-x * x / (2.0 * sigma * sigma), q)
        if deriv_order == 0:
            return x, (2.0 / sigma ** 2) * e ** q - q * (x * x / sigma ** 4) * e ** (2.0 * q - 1.0)
        return x, (q * x / sigma ** 4) * ((2.0 * q - 1.0) * (x * x / sigma ** 2) * e ** (3.0 * q - 2.0)
                                          - 4.0 * e ** (2.0 * q - 1.0))
    return section


# -- the q-Meyer smoother: Meyer's radial taper with the q-cosine ---------------------------

def _meyer_nu(x):
    """Meyer's auxiliary polynomial ν(x) = x⁴(35 − 84x + 70x² − 20x³): 0 → 1 on [0, 1], C³ at
    both ends."""
    x = np.clip(x, 0.0, 1.0)
    return x ** 4 * (35.0 - 84.0 * x + 70.0 * x * x - 20.0 * x ** 3)


def _cos_q(x, q: float):
    """Borges, Tsallis, Miranda & Andrade 2004 eq. (31)–(32): cos_q x = Re e_q^(ix) = ρ_q(x)
    cos φ_q(x), ρ_q² = e_q^((1−q)x²), φ_q = arctan((1 − q)x)/(1 − q); the cosine at q = 1."""
    if q == 1.0:
        return np.cos(x)
    t = (1.0 - q) * np.asarray(x, dtype=np.float64)
    return (1.0 + t * t) ** (0.5 / (1.0 - q)) * np.cos(np.arctan(t) / (1.0 - q))


def _cos_q_zero(q: float) -> float:
    """The first zero of cos_q, where φ_q = π/2: x₀ = tan((1 − q)π/2)/(1 − q); π/2 at q = 1,
    finite for 0 < q < 2. cos_q falls monotonically from 1 to 0 on [0, x₀]."""
    if q == 1.0:
        return np.pi / 2.0
    return float(np.tan((1.0 - q) * np.pi / 2.0) / (1.0 - q))


def _meyer_taper(omega, q: float):
    """Meyer's scaling function on the radial frequency with cos(π/2·ν) replaced by cos_q(x₀ ν):
    1 up to 2π/3, the taper to 4π/3, 0 beyond — the q-Meyer the paper's closing remark leaves
    open, generalized along its lines."""
    omega = np.asarray(omega, dtype=np.float64)
    out = _cos_q(_cos_q_zero(q) * _meyer_nu(3.0 * omega / (2.0 * np.pi) - 1.0), q)
    out = np.where(omega <= 2.0 * np.pi / 3.0, 1.0, out)
    return np.where(omega >= 4.0 * np.pi / 3.0, 0.0, out)


@functools.lru_cache(maxsize=64)
def _meyer_scale(q: float = 1.0) -> float:
    """ω per unit s that puts the peak of ω·taper_q(ω) at s* = 1/√2 (λ* = √2 a, the Gaussian
    g1's band peak). Fixed width: the q = 1 value for every q, so the q-deformation moves λ*;
    q-paired: this q's own, so λ* stays put."""
    w = np.linspace(2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0, 1 << 16)
    f = w * _meyer_taper(w, q)
    i = int(np.argmax(f))
    w_star = w[i]
    if 0 < i < w.size - 1:
        y0, y1, y2 = f[i - 1], f[i], f[i + 1]
        d = y0 - 2.0 * y1 + y2
        if d < 0:
            w_star = w[i] + 0.5 * (y0 - y2) / d * (w[1] - w[0])
    return float(w_star * np.sqrt(2.0))


def _meyer_hat_factory(q: float, paired: bool = False):
    c = _meyer_scale(q) if paired else _meyer_scale()

    def theta_hat(sx, sy):
        return _meyer_taper(c * np.sqrt(sx * sx + sy * sy), q)
    return theta_hat


@functools.lru_cache(maxsize=64)
def _meyer_section_u(q: float, deriv_order: int, paired: bool = False) -> tuple:
    """The q-Meyer's section (bandlimited, so no closed form), on a finer grid for its band and
    over a wider window: a bandlimited kernel rings with the period of its cut-off (about 4 σ
    here) and decays as ρ⁻⁴, so ±12 σ shows the rings down to a percent of the main lobe."""
    return _central_row(_meyer_hat_factory(q, paired), deriv_order, n=4096, span=12.0)


def _meyer_section_factory(q: float, paired: bool = False):
    def section(scale, deriv_order, n_points):
        u, y = _meyer_section_u(float(q), int(deriv_order), bool(paired))
        return u * scale * _SIGMA_PER_SCALE, y
    return section


# -- the registry ----------------------------------------------------------------------------

def _beta(params: dict) -> float:
    """A q kernel's width β (default ½, today's kernel); refused unless positive."""
    beta = float(params.get("beta", 0.5))
    if not beta > 0.0:
        raise ValueError(f"a q kernel's width needs beta > 0, got {beta!r}")
    return beta


def _width_label(q: float, beta: float, given: bool) -> str:
    """The β_q a q kernel runs with, for its label: "β_q paired" where it is the q-paired width
    (``microcanonical.paired_q_beta``), else its value."""
    if given and q < 2.0:
        from dynamix.core.microcanonical import paired_q_beta

        if np.isclose(beta, paired_q_beta(q), rtol=1e-12, atol=0.0):
            return "β_q paired"
    return f"β_q {beta:g}"


def smoother(name: str, params: dict | None = None) -> Smoother:
    """The smoother ``name`` with its parameters: ``alpha`` for ``frac_bspline`` (≥ 1; a
    continuous ψ in 2-D needs > 2), ``q`` for ``q_gaussian`` (0 < q < 2), and the q kernels'
    width: ``beta`` (> 0, default ½) for ``q_gaussian`` and ``q_mexican``, ``pairing =
    "q-paired"`` for ``q_meyer`` (its band peak held for every q)."""
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
        beta = _beta(params)
        return Smoother("q_gaussian", f"q-Gaussian q = {q:g}, {_width_label(q, beta, 'beta' in params)}",
                        "∂θ_q", {"q": q, "beta": beta}, _stretched(_q_hat_factory(q), beta),
                        _q_section_factory(q, beta))
    if name == "q_mexican":
        q = float(params.get("q", 1.5))
        if not 2.0 / 3.0 < q < 2.0:
            raise ValueError(f"q_mexican needs 2/3 < q < 2 (finite mass in 2-D below 2, a "
                             f"continuous q-g3 at the support's edge above 2/3), got {q!r}")
        beta = _beta(params)
        return Smoother("q_mexican", f"q-Mexican hat q = {q:g}, {_width_label(q, beta, 'beta' in params)}",
                        "q-g3", {"q": q, "beta": beta}, _q_mexican_hat_factory(q, beta),
                        _q_mexican_section_factory(q, beta))
    if name == "q_meyer":
        q = float(params.get("q", 1.5))
        if not 0.0 < q < 2.0:
            raise ValueError(f"q_meyer needs 0 < q < 2 (the q-cosine's first zero exists "
                             f"there), got {q!r}")
        paired = params.get("pairing", "fixed") == "q-paired"
        return Smoother("q_meyer", f"q-Meyer q = {q:g}, "
                        f"{'band peak held' if paired else 'dilation fixed'}", "∂θ_q",
                        {"q": q, "pairing": "q-paired" if paired else "fixed"},
                        _meyer_hat_factory(q, paired), _meyer_section_factory(q, paired))
    raise ValueError(f"unknown smoother {name!r}; expected one of {NAMES}")
