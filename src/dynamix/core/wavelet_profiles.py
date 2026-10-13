# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Every wavelet's section at the level on show, and the one physical length that labels it:
λ*, the bandpass peak wavelength (1/k*, k* where |ψ̂| peaks along the analysing direction, k in
cycles per pixel). Under the app's normalisation ψ̂_a(k) = ψ̂(ak), so λ* is a constant per
family times the device's own scale parameter: log-log slopes are untouched and "matched
scales" means the same length in every family. The design and the per-family constants:
``docs/specs/2026-10-13-wavelet-scale-bar-design.md`` §3.

A wavelet-bearing device implements ``wavelet_profile(params, scale_px, level) -> Profile``;
:func:`profile_for` dispatches to it. λ* is never read off an FFT of the drawn 1-D section (by
the projection-slice theorem that transform is the 2-D filter integrated over k_y, not its
slice); an isotropic kernel takes the k_x slice of its own 2-D Fourier form, a separable 1-D
filter its own cascade. Pure numpy/scipy/pywt: no Qt.
"""
from __future__ import annotations

import dataclasses

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


def wtmm_profile(smoothing: str, a: float) -> Profile:
    """The WTMM analysing wavelet ψ (``deriv_order`` 1) at nominal scale ``a``: the true 2-D
    section from :mod:`dynamix.core.scale_units`, and its λ* = 2πσ/√m."""
    from dynamix.core.scale_units import (analyzing_names, kernel_section, lambda_peak_px,
                                          sigma_px)

    x, psi = _normalised(*kernel_section(float(a), smoothing, 1))
    theta_name, psi_name = analyzing_names(smoothing)
    return Profile(x, psi, float(a), lambda_peak_px(float(a), smoothing, 1),
                   f"ψ {psi_name} (WTMM {smoothing})",
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


def measure_profile(wavelet: str, r: float, *, beta: float = 1.0, q_tsallis: float = 1.5,
                    frac_n: float = 2.0, q_pairing: str = "fixed", q_beta: float = 0.5) -> Profile:
    """The measure route's positive kernel at r (its section), with λ* of its Littlewood-Paley
    wavelet r∂_r K_r, the band the slope d log T / d log r measures; no band for a Lorentzian at
    β ≤ 1 or a q-Gaussian at q ≥ 2 (infinite 2-D mass)."""
    from dynamix.core.microcanonical import _radial_kernel

    r, q, beta = float(r), float(q_tsallis), float(beta)
    qb = _q_width(wavelet, "q_gaussian", q, q_pairing, q_beta)
    n = 2 * int(np.ceil((16.0 if wavelet in ("lorentzian", "q_gaussian") else 8.0) * r)) + 1
    n = min(n, 8193)
    section = np.fft.fftshift(_radial_kernel((1, n), r, wavelet, beta, q, frac_n, q_beta=qb)[0])
    x = np.arange(n, dtype=np.float64) - n // 2
    x, psi = _normalised(x, section)
    no_band = (wavelet == "lorentzian" and beta <= 1.0) or (wavelet == "q_gaussian" and q >= 2.0)
    if no_band:
        lam, octaves = float("nan"), 0.0
    else:
        if wavelet == "frac_gaussian":
            kap = np.linspace(0.02, 2.0 * np.pi, 3000)
            k_hat = _hankel_slice(lambda rho: np.exp(-0.5 * rho ** float(frac_n)), kap)
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
    n = min(2 * int(np.ceil(10.0 * r)) + 1, 8193)
    x = np.arange(n, dtype=np.float64) - n // 2
    x, psi = _normalised(x, _marr_kernel(x * x, r, wavelet, beta, q, fn, u0, stretch))
    kap, k_hat = _fourier_slice("marr", wavelet, beta, q, fn, u0, qb, stretch)
    lam, octaves = _band(kap, k_hat)
    extra = {"a": r, "zero_crossing": r, "band_octaves": octaves}
    if name == "q_mexican":
        extra["q"] = q_tsallis
        extra["q_beta"] = qb
    label = {"g1": "g1", "g2": "Ricker (g2)", "g3": "g3", "gaussian": "Ricker",
             "q_mexican": f"q-Mexican hat q = {float(q_tsallis):g}, "
                          f"β {'paired' if q_pairing == 'q-paired' else 'fixed'}",
             "lorentzian_marr": f"Lorentzian-Marr β = {beta:g}",
             "frac_gaussian": f"fractional order n = {fn:g}"}.get(name, name)
    return Profile(x, psi, r, float(lam) * r if np.isfinite(lam) else float("nan"),
                   f"{label} (multiaffine)", broad=octaves > 4.0, extra=extra)
