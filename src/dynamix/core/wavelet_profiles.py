# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Every wavelet's section at the level on show, and the one physical length that labels it:
λ*, the bandpass peak wavelength (1/k*, k* where |ψ̂| peaks along the analysing direction, k in
cycles per pixel). Under the app's normalisation ψ̂_a(k) = ψ̂(ak), so λ* is a constant per
family times the device's own scale parameter: log-log slopes are untouched and "matched
scales" means the same length in every family. The per-family constants are pinned in
``tests/test_wavelet_profiles.py`` against the kernels' own Fourier forms.

A wavelet-bearing device implements ``wavelet_profile(params, scale_px, level) -> Profile``;
:func:`profile_for` dispatches to it. λ* is never read off an FFT of the drawn 1-D section (by
the projection-slice theorem that transform is the 2-D filter integrated over k_y, not its
slice); an isotropic kernel takes the k_x slice of its own 2-D Fourier form, a separable 1-D
filter its own cascade. Pure numpy/scipy/pywt: no Qt.
"""
from __future__ import annotations

import dataclasses
import functools

import numpy as np

__all__ = ["Profile", "format_length", "lambda_px", "profile_for", "wtmm_profile"]


@dataclasses.dataclass(frozen=True)
class Profile:
    x_px: np.ndarray          # pixel offsets along the section, symmetric about 0
    psi: np.ndarray           # amplitude, max |psi| = 1
    scale_px: float           # the device's own scale parameter at this level
    lambda_px: float          # λ* in pixels; nan when the kernel has no band
    family: str
    off_ladder: bool = False  # a fine dyadic level whose own constant is off the asymptotic one
    broad: bool = False       # half-max band wider than 4 octaves
    extra: dict = dataclasses.field(default_factory=dict)
    envelope: np.ndarray | None = None  # optional positive window on the same x grid


def format_length(value: float, unit: str) -> str:
    """The notebook's three tiers: 480 m, 52.0 μm, 3.46 m."""
    v = float(value)
    text = f"{v:.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")
    return f"{text} {unit}"


def _normalised(x, psi) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    peak = float(np.max(np.abs(psi))) if psi.size else 0.0
    return x, psi / peak if peak > 0 else psi


def wtmm_profile(smoothing: str, a: float, params: dict | None = None) -> Profile:
    """The WTMM analysing wavelet ψ (``deriv_order`` 1) at nominal scale ``a``: the true 2-D
    section from :mod:`dynamix.core.scale_units`, and its λ* = 2πσ/√m (the registry's
    smoothers: their own section and numerical peak, with ``params``)."""
    from dynamix.core.scale_units import (analyzing_names, kernel_section, lambda_peak_px,
                                          sigma_px)

    x, psi = _normalised(*kernel_section(float(a), smoothing, 1, params=params))
    theta_name, psi_name = analyzing_names(smoothing, params)
    return Profile(x, psi, float(a), lambda_peak_px(float(a), smoothing, 1, params),
                   f"WTMM ψ {psi_name} = ∇θ (θ {smoothing}, {theta_name})",
                   extra={"a": float(a), "sigma": sigma_px(float(a)), "theta": theta_name})


def profile_for(device_name: str, params: dict, result: dict, level: int):
    """The profile of ``device_name`` at the level on show, or ``None`` when the device has no
    wavelet, the result no scales, or the level no entry. ``result["_scale_idx"]`` (the
    ScaleSelect stamp) is the scale axis's own index; ``level`` the fallback."""
    from dynamix.model.device import get_device

    try:
        device = get_device(device_name)
    except Exception:                                                # noqa: BLE001
        return None
    hook = getattr(device, "wavelet_profile", None)
    scales = result.get("scales") if isinstance(result, dict) else None
    if hook is None or scales is None:
        return None
    scales = np.asarray(scales, dtype=np.float64)
    idx = int(result.get("_scale_idx", level))
    if not 0 <= idx < scales.size:
        return None
    return hook(dict(params or {}), float(scales[idx]), idx)


def lambda_px(device_name: str, params: dict, scale_px: float, level: int = 0) -> float:
    """λ* alone, for an axis: the device's constant times ``scale_px``; nan without a band."""
    from dynamix.model.device import get_device

    hook = getattr(get_device(device_name), "wavelet_profile", None)
    if hook is None:
        return float("nan")
    return float(hook(dict(params or {}), float(scale_px), int(level)).lambda_px)


# -- the measure and multiaffine routes (Turiel's kernels, dynamix.core.microcanonical) --------

#: Angular frequencies on [0, 2π) at r = 2 for the peak search: a step of 0.0008 in κ = r|k|.
_PEAK_N = 16384


def _band(kap, w) -> tuple[float, float]:
    """``(λ*/r, half-max band in octaves)`` from |W(κ)| on κ > 0, κ = r|k| angular: λ*/r is
    2π/κ* at the (parabolically refined) peak; nan when W has no peak."""
    w = np.abs(np.asarray(w, dtype=np.float64))
    ok = (np.asarray(kap) > 0) & np.isfinite(w)
    if not ok.any() or not np.any(w[ok] > 0):
        return float("nan"), 0.0
    kk, ww = np.asarray(kap, dtype=np.float64)[ok], w[ok]
    i = int(np.argmax(ww))
    kstar = kk[i]
    if 0 < i < kk.size - 1:
        y0, y1, y2 = ww[i - 1], ww[i], ww[i + 1]
        d = y0 - 2.0 * y1 + y2
        if d < 0:
            kstar = kk[i] + 0.5 * (y0 - y2) / d * (kk[i + 1] - kk[i])
    above = np.flatnonzero(ww >= 0.5 * ww[i])
    octaves = float(np.log2(kk[above[-1]] / kk[above[0]])) if above.size > 1 else 0.0
    return 2.0 * np.pi / kstar, octaves


def _fourier_slice(route, wavelet, beta, q, frac_n, u0, q_beta, stretch):
    """κ on [0, 2π) and the kernel's own 2-D transform along k_x (the k_y = 0 row of
    ``microcanonical._fourier_kernel`` at r = 2; the form depends on κ = r|k| alone)."""
    from dynamix.core.microcanonical import _fourier_kernel

    n = _PEAK_N
    quad = _fourier_kernel((2, n), 2.0, route, wavelet, beta, q, frac_n, u0=u0, q_beta=q_beta,
                           stretch=stretch)
    kap = 2.0 * 2.0 * np.pi * np.arange(n // 2 + 1) / n
    return kap, np.asarray(quad[0, : n // 2 + 1], dtype=np.float64)


def _hankel_slice(profile, kap):
    """The 2-D transform of a radial kernel on κ: 2π ∫ K(ρ) J0(κρ) ρ dρ (the measure route's
    fractional envelope, which has no closed form)."""
    from scipy.special import j0

    rho = np.linspace(0.0, 80.0, 16001)
    kr = profile(rho) * rho
    return 2.0 * np.pi * np.trapezoid(kr[None, :] * j0(kap[:, None] * rho[None, :]), rho, axis=1)


def _q_width(wavelet, q_family, q, q_pairing, q_beta):
    """The q-family's width β as the device runs it (``holder_methods.q_width``): fixed, or
    paired to q; None for every other kernel."""
    if wavelet != q_family:
        return None
    if q_pairing == "q-paired":
        from dynamix.core.microcanonical import paired_q_beta

        return paired_q_beta(float(q))
    return float(q_beta)


@functools.lru_cache(maxsize=64)
def _frac_measure_band(frac_n: float) -> tuple:
    """The fractional envelope's Littlewood-Paley band at unit scale: a numerical Hankel
    transform that depends on n alone, computed once per order (a cascade asks for it at
    every level on every knob tick)."""
    kap = np.linspace(0.02, 2.0 * np.pi, 3000)
    k_hat = _hankel_slice(lambda rho: np.exp(-0.5 * rho ** float(frac_n)), kap)
    return _band(kap, kap * np.gradient(k_hat, kap))


def measure_profile(wavelet: str, r: float, *, beta: float = 1.0, q_tsallis: float = 1.5,
                    frac_n: float = 2.0, q_pairing: str = "fixed", q_beta: float = 0.5) -> Profile:
    """The measure route's positive kernel at r (its section), with λ* of its Littlewood-Paley
    wavelet r∂_r K_r, the band the slope d log T / d log r measures; no band for a Lorentzian at
    β ≤ 1 or a q-Gaussian at q ≥ 2 (infinite 2-D mass)."""
    from dynamix.core.microcanonical import _radial_kernel

    r, q, beta = float(r), float(q_tsallis), float(beta)
    qb = _q_width(wavelet, "q_gaussian", q, q_pairing, q_beta)
    # The continuous section, sampled at r/16 (the kernel is smooth at every scale; the pixel
    # step would draw a kinked polyline at the smallest ones): the kernel at r/step in units
    # of the step is the same kernel on a finer grid.
    reach = 16.0 if wavelet in ("lorentzian", "q_gaussian") else 8.0
    step = max(r / 16.0, 1.0 / 64.0)
    n = min(2 * int(np.ceil(reach * r / step)) + 1, 8193)
    section = np.fft.fftshift(_radial_kernel((1, n), r / step, wavelet, beta, q, frac_n,
                                             q_beta=qb)[0])
    x = (np.arange(n, dtype=np.float64) - n // 2) * step
    x, psi = _normalised(x, section)
    no_band = (wavelet == "lorentzian" and beta <= 1.0) or (wavelet == "q_gaussian" and q >= 2.0)
    if no_band:
        lam, octaves = float("nan"), 0.0
    else:
        if wavelet == "frac_gaussian":
            lam, octaves = _frac_measure_band(round(float(frac_n), 9))
        else:
            kap, k_hat = _fourier_slice("measure", wavelet, beta, q, frac_n, None, qb, 1.0)
            lp = kap * np.gradient(k_hat, kap)
            lam, octaves = _band(kap, lp)
    extra = {"a": r, "band_octaves": octaves}
    if wavelet == "q_gaussian":
        extra["q"] = q
        extra["q_beta"] = qb
    label = {"gaussian": "Gaussian", "lorentzian": f"Lorentzian β = {beta:g}",
             "q_gaussian": f"q-Gaussian q = {q:g}, β {'paired' if q_pairing == 'q-paired' else 'fixed'}",
             "frac_gaussian": f"fractional envelope n = {float(frac_n):g}"}.get(wavelet, wavelet)
    return Profile(x, psi, r, float(lam) * r if np.isfinite(lam) else float("nan"),
                   f"{label} (measure)", broad=octaves > 4.0, extra=extra)


def multiaffine_profile(wavelet: str, r: float, *, beta: float = 1.0, q_tsallis: float = 1.5,
                        frac_n: float = 2.0, q_pairing: str = "fixed",
                        q_beta: float = 0.5) -> Profile:
    """The multiaffine route's zero-crossing-at-r wavelet (its section) and its λ*: the Ricker
    and its fractional, q-Mexican and Lorentzian-Marr relatives, resolved exactly as
    ``microcanonical.ricker_projections`` resolves them (aliases, Borges' q, the q width)."""
    from dynamix.core.microcanonical import (_MARR_ALIASES, _borges_q, _frac_u0, _marr_kernel,
                                             _q_mexican_stretch)

    r, q, beta, fn = float(r), float(q_tsallis), float(beta), float(frac_n)
    name = wavelet
    qb = _q_width(wavelet, "q_mexican", q, q_pairing, q_beta)
    stretch = 1.0
    if wavelet == "q_mexican":
        stretch = _q_mexican_stretch(q, qb)
        wavelet, q = "q_gaussian", _borges_q(q)
    alias = _MARR_ALIASES.get(wavelet)
    if isinstance(alias, tuple):
        wavelet, fn = alias
    elif alias is not None:
        wavelet = alias
    u0 = _frac_u0(fn) if wavelet == "frac_gaussian" else None
    step = max(r / 16.0, 1.0 / 64.0)
    n = min(2 * int(np.ceil(10.0 * r / step)) + 1, 8193)
    x = (np.arange(n, dtype=np.float64) - n // 2) * step
    x, psi = _normalised(x, _marr_kernel(x * x, r, wavelet, beta, q, fn, u0, stretch))
    kap, k_hat = _fourier_slice("marr", wavelet, beta, q, fn, u0, qb, stretch)
    lam, octaves = _band(kap, k_hat)
    extra = {"a": r, "zero_crossing": r, "band_octaves": octaves}
    if name == "q_mexican":
        extra["q"] = q_tsallis
        extra["q_beta"] = qb
    label = {"g1": "g1", "g2": "Ricker (g2)", "g3": "g3", "gaussian": "Ricker",
             "q_mexican": f"q-Mexican hat q = {float(q_tsallis):g}, "
                          f"β_q {'paired' if q_pairing == 'q-paired' else 'fixed'}",
             "lorentzian_marr": f"Lorentzian-Marr β = {beta:g}",
             "frac_gaussian": f"fractional order n = {fn:g}"}.get(name, name)
    return Profile(x, psi, r, float(lam) * r if np.isfinite(lam) else float("nan"),
                   f"{label} (multiaffine)", broad=octaves > 4.0, extra=extra)


# -- the dyadic families (their own 1-D cascades) and the CDF and Tang-You kernels ------------

#: The Mallat-Zhong quadratic-spline wavelet's continuum constant λ*/s (the peak of
#: |ω sinc⁴(ω/4)| solves tan(ω/4) = ω/3); fine levels fall short of it (level 1 peaks at Nyquist).
_MZ_CONTINUUM = 1.8595228751967214
#: A dyadic level whose own constant is this far off the asymptotic one is off the ladder.
_OFF_LADDER_TOL = 0.05


def _dilate(taps: np.ndarray, step: int) -> np.ndarray:
    """``taps`` with ``step - 1`` zeros between them (the à trous filter at that level)."""
    out = np.zeros((taps.size - 1) * step + 1)
    out[::step] = taps
    return out


def _cascade(h: np.ndarray, g: np.ndarray, level: int) -> np.ndarray:
    """The impulse response of the level-``level`` (1-based) band of an à trous cascade:
    ``g`` dilated 2^(level-1) after ``h`` dilated 2^(level-2) ... down to ``h`` itself."""
    resp = _dilate(np.asarray(g, dtype=np.float64), 2 ** (level - 1))
    for j in range(level - 1, 0, -1):
        resp = np.convolve(resp, _dilate(np.asarray(h, dtype=np.float64), 2 ** (j - 1)))
    return resp


def _peak_wavelength(resp: np.ndarray) -> float:
    """λ in pixels of a 1-D filter's spectral peak (cycles per pixel; a separable filter's own
    transform, which is allowed), parabolically refined."""
    from dynamix.core.fft_policy import double

    n = int(max(8192, 16 * resp.size))
    spec = np.abs(double().rfft(np.pad(np.asarray(resp, dtype=np.float64), (0, n - resp.size))))
    k = np.arange(spec.size) / n
    lam, _octaves = _band(2.0 * np.pi * k, spec)      # _band works in angular κ at r = 1
    return float(lam)


def _cubic_bspline_derivative(t: np.ndarray) -> np.ndarray:
    """d/dt of the centred cubic B-spline (support |t| < 2): piecewise quadratic, odd."""
    t = np.asarray(t, dtype=np.float64)
    u = np.abs(t)
    inner = -2.0 * u + 1.5 * u * u
    outer = -0.5 * (2.0 - u) ** 2
    return np.sign(t) * np.where(u <= 1.0, inner, np.where(u < 2.0, outer, 0.0))


@functools.lru_cache(maxsize=64)
def _mz_continuum_constant(alpha: float) -> float:
    """λ*/2^j of the fractional Mallat-Zhong wavelet in the continuum: the peak of
    w |sinc(w/4)|^(alpha+1) solves tan(w/4) = (alpha+1) w / (4 alpha); alpha = 3 gives
    1.8595 (tan(w/4) = w/3)."""
    lo, hi = 1e-3, 2.0 * np.pi - 1e-3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.tan(mid / 4.0) - (alpha + 1.0) * mid / (4.0 * alpha) < 0.0:
            lo = mid
        else:
            hi = mid
    return float(2.0 * np.pi / (0.5 * (lo + hi)))


def _mz_level_lambda(level: int, alpha: float) -> float:
    """λ* of the fractional cascade's level-``level`` band: the two-tap derivative filter at
    2^(level-1) behind the sign-preserving fractional refinement filters below it (the à trous
    product in frequency), its spectral peak."""
    from dynamix.core.mz_edges import _hf_frac
    from dynamix.core.mzlib import TAPS

    w = np.linspace(1e-3, np.pi, 16384)
    resp = sum(v * np.exp(-1j * k * w * 2.0 ** (level - 1)) for k, v in TAPS["g"].items())
    for j in range(level - 1):
        resp = resp * _hf_frac(w * 2.0 ** j, float(alpha))
    lam, _octaves = _band(w, np.abs(resp))
    return float(lam)


def mz_profile(level: int, alpha: float = 3.0) -> Profile:
    """The Mallat-Zhong wavelet at dyadic level ``level`` (scale 2^level): the quadratic spline
    (``alpha = 3``) as the transform's own à trous cascade applies it (``mzlib.TAPS``), or the
    fractional B-spline of order ``alpha`` (its sign-preserving fractional cascade). The section
    is the wavelet itself, psi_alpha = theta_alpha' with theta_alpha(x) = 2 beta*^alpha(2x) the
    symmetric fractional B-spline, from the exact series (``frac_bspline_exact``: no inverse
    transform, so alpha = 1's box pair carries no ringing), dilated to the level's scale."""
    from dynamix.core.frac_bspline_exact import beta_star_derivative
    from dynamix.core.mzlib import TAPS

    def taps(d: dict) -> np.ndarray:
        lo = min(d)
        out = np.zeros(max(d) - lo + 1)
        for k, v in d.items():
            out[k - lo] = v
        return out

    level, alpha = int(level), float(alpha)
    scale = 2.0 ** level
    if alpha == 3.0:
        lam = _peak_wavelength(_cascade(taps(TAPS["h"]), taps(TAPS["g"]), level))
        c_continuum = _MZ_CONTINUUM
        reach = 1.0                          # theta_3's support is |2x/scale| <= 2
        family = f"Mallat-Zhong quadratic spline, level {level}"
    else:
        lam = _mz_level_lambda(level, alpha)
        c_continuum = _mz_continuum_constant(alpha)
        reach = 1.5                          # the algebraic tails of a fractional order
        family = f"Mallat-Zhong fractional spline α = {alpha:g}, level {level}"
    x = np.linspace(-reach * scale, reach * scale, max(129, int(16 * reach * scale) + 1))
    x, psi = _normalised(x, beta_star_derivative(2.0 * x / scale, alpha))
    c = lam / scale
    return Profile(x, psi, scale, lam, family,
                   off_ladder=abs(c / c_continuum - 1.0) > _OFF_LADDER_TOL,
                   extra={"a": scale, "c_level": c, "c_continuum": c_continuum,
                          "alpha": alpha})


def _bank_filters(bank: str) -> tuple[np.ndarray, np.ndarray]:
    import pywt

    w = pywt.Wavelet(bank)
    return np.asarray(w.dec_lo, dtype=np.float64), np.asarray(w.dec_hi, dtype=np.float64)


def _bank_asymptotic_c(bank: str) -> float:
    h, g = _bank_filters(bank)
    return _peak_wavelength(_cascade(h, g, 8)) / 2.0 ** 8


def dwt_profile(bank: str, level: int) -> Profile:
    """A PyWavelets bank's level-``level`` wavelet (scale 2^level) as its own cascade applies
    it; level 1 of every bank peaks at Nyquist and is off the ladder."""
    import pywt

    h, g = _bank_filters(bank)
    level = int(level)
    resp = _cascade(h, g, level)
    lam = _peak_wavelength(resp)
    scale = 2.0 ** level
    # The section is the cascade's limit function (the bank's continuous wavelet) dilated by
    # 2^level and centred on its peak; the level's taps are its samples, too few to draw at
    # the finest levels.
    _phi, psi_c, x_c = pywt.Wavelet(bank).wavefun(level=8)
    psi_c, x_c = np.asarray(psi_c, dtype=np.float64), np.asarray(x_c, dtype=np.float64)
    x = (x_c - x_c[np.argmax(np.abs(psi_c))]) * scale
    x, psi = _normalised(x, psi_c)
    c, c_inf = lam / scale, _bank_asymptotic_c(bank)
    return Profile(x, psi, scale, lam, f"{bank} ψ, level {level}",
                   off_ladder=abs(c / c_inf - 1.0) > _OFF_LADDER_TOL,
                   extra={"a": scale, "c_level": c, "c_asymptotic": c_inf})


def cdf_profile(edge_mode: str, sigma: float) -> Profile:
    """The linear CDF evolution's equivalent kernel at σ (= 2^k) for small θ: θ t ΔG_σ, the 2-D
    Mexican hat's section, behind v for the ``marr`` mode (its zero crossings), and ∇G_σ, the
    Gaussian first derivative, behind ∇u for the ``mz`` mode (its maxima); λ* = 2πσ/√m with
    m = 2 or 1. Measured on an impulse: within a few per cent of the evolution itself."""
    sigma = float(sigma)
    u = np.linspace(-6.0, 6.0, 481)
    if edge_mode == "marr":
        psi, m, label = ((2.0 - u * u) * np.exp(-u * u / 2.0), 2,
                         "equivalent kernel of v (linear, small θ): θ t ΔG_σ")
    else:
        psi, m, label = (-u * np.exp(-u * u / 2.0), 1,
                         "equivalent kernel of ∇u (linear, small θ): ∇G_σ")
    x, psi = _normalised(u * sigma, psi)
    return Profile(x, psi, sigma, 2.0 * np.pi * sigma / np.sqrt(m), label,
                   extra={"a": sigma, "sigma": sigma})


def tang_you_profile(kernel: str, s: float) -> Profile:
    """The gradient wavelet ∂θ/∂x of the skeleton device's smoothing kernel at support radius
    s (its section), with λ* from the k_x slice of θ's 2-D transform times k."""
    from dynamix.core.wavelet_skeleton import theta_kernel

    s = float(s)
    step = max(s / 16.0, 1.0 / 64.0)             # the kernel's continuous section, finely sampled
    n = min(2 * int(np.ceil(2.5 * s / step)) + 1, 8193)
    theta = np.fft.fftshift(theta_kernel((1, n), s / step, kernel)[0])
    x = (np.arange(n, dtype=np.float64) - n // 2) * step
    x, psi = _normalised(x, np.gradient(theta, x))
    from dynamix.core.fft_policy import double

    big = 1024
    spec = np.abs(double().fft2(theta_kernel((big, big), s, kernel))[0, : big // 2 + 1])
    k = np.arange(spec.size) / big
    lam, _octaves = _band(2.0 * np.pi * k, k * spec)
    return Profile(x, psi, s, float(lam), f"{kernel} gradient wavelet", extra={"a": s})
