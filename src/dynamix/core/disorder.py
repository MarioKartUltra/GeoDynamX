# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The disorder function and the Hessian measure of a scalar field.

Two families of global, Euclidean-invariant measures of a 2-D field u(x), built from local
densities of its second derivatives:

- the **disorder function** of a locally striped pattern, in its field form (Gunaratne, Hoffman and
  Kouri 1998, PRE 57:5146, Eq. 2; Hoffman, Gunaratne, Zhang and Kouri 2000, Chaos 10:240, Eq. 1):
  the density D(x) = |(Δ + k₀²)u| / (k₀² ⟨|u|⟩) vanishes on perfect stripes of wavenumber k₀, and its
  moments give the mean disorder δ̄(β) = (2 − β)⟨D^β⟩ (dimensionless), the total δ(β) = |Ω|·δ̄(β) and
  the rooted mean δ_root(β) = δ̄(β)^{1/β} (Hu et al. 2005, Chaos 15:043701, Eq. 2), for 0 < β < 2;
- the **Hessian measure** of a surface without a planform (Nathan and Gunaratne 2005, PRE 71:035101,
  Eq. 1): the density |Λ(x)| = |u_xx u_yy − u_xy²| / Var(u) and μ(β) = ⟨|Λ|^β⟩^{1/(4β)}, an inverse
  length.

β is the moment order throughout (the 1995 PRL writes it μ; μ here always means the Hessian
measure). Derivatives are Fourier multipliers on the FFT policy, each scheme a smoothing gain H(k):
the Hermite DAF of the papers (``daf``, M = 12, σ/Δ = 4, PRE 57 p. 5147), a Gaussian (``gaussian``,
σ_d px) or none (``spectral``, periodic inputs only). The two terms of the Helmholtz operator are
filtered alike, as one multiplier H(k)(k₀² − k²), and the normalising ⟨|u|⟩ carries the same
smoothing, so a perfect stripe reads exactly zero whatever the scheme and the gain cancels to first
order. A non-periodic field is reflected past its edges by the scheme's reach before the transform.
"""
from __future__ import annotations

import math

import numpy as np

from dynamix.core import fft_policy
from dynamix.core.daf import daf_gain

__all__ = ["BANDPASS_WIDTH", "DAF_DEFAULTS", "DAF_REACH", "DELTA_BETAS", "DERIVS", "FLOOR_FRACTION",
           "GOLDMAN_BAND", "K0_METHODS", "MU_BETAS", "ZERO", "bandpass", "daf_filter", "detrend",
           "disorder_moments", "envelope_density", "envelope_halfplane", "filtered",
           "phase_field_density",
           "helmholtz_density", "hessian_density", "hessian_measure", "k0_direction", "k0_estimate",
           "radial_filter", "reach", "transfer"]

#: The derivative schemes: the papers' DAF, a derivative of Gaussian, or plain spectral derivatives.
DERIVS = ("daf", "gaussian", "spectral")
#: The moment orders the disorder is published at (0 < β < 2, PRL 1995 Eq. 8).
DELTA_BETAS = (0.1, 0.25, 0.5, 1.0, 1.5, 1.9)
#: The moment orders the Hessian measure is published at (the papers use 0.1, 1, 3.4 and 5).
MU_BETAS = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 3.4, 5.0)
#: The DAF derivative's settings (PRE 57 p. 5147: "σ/Δ = 4.0, and M = 12").
DAF_DEFAULTS = {"M": 12, "sigma_over_dx": 4.0}
#: The DAF stencil's reach in px ("the sum ran over 20 grid points on each side", PRE 57 p. 5147).
DAF_REACH = 20
#: The density floor for negative moment orders, as a fraction of the mean density.
FLOOR_FRACTION = 1e-6
#: A dimensionless density below this is the operator's round-off, not a measurement, and counts as
#: an exact zero: a small moment order would otherwise read it loudly ((1e-15)^0.1 is 0.03).
ZERO = 1e-12


def transfer(k, deriv: str, *, sigma_d: float = 1.0, M: int = DAF_DEFAULTS["M"],
             sigma_over_dx: float = DAF_DEFAULTS["sigma_over_dx"]):
    """The scheme's smoothing gain H(k) at angular frequency ``k`` (rad/px)."""
    k = np.abs(np.asarray(k, dtype=np.float64))
    if deriv == "daf":
        return daf_gain(k, float(sigma_over_dx), int(M))
    if deriv == "gaussian":
        return np.exp(-0.5 * (k * float(sigma_d)) ** 2)
    if deriv == "spectral":
        return np.ones_like(k)
    raise ValueError(f"unknown derivative scheme {deriv!r}; expected one of {DERIVS}")


_DAF_HALF_WIDTH: dict = {}


def reach(deriv: str, sigma_d: float = 1.0) -> int:
    """How far the scheme's kernel reaches in px: the DAF's full tap extent (37 at σ/Δ = 4, M = 12;
    PRE 57's 20-point stencil cuts its tail at 1e-4), four σ_d for the Gaussian, nothing for the
    spectral scheme. A non-periodic field is reflected by this much before the transform."""
    if deriv == "daf":
        key = (DAF_DEFAULTS["sigma_over_dx"], DAF_DEFAULTS["M"])
        if key not in _DAF_HALF_WIDTH:
            from dynamix.core.daf import hermite_daf

            _DAF_HALF_WIDTH[key] = hermite_daf(*key).size // 2
        return _DAF_HALF_WIDTH[key]
    if deriv == "gaussian":
        return int(math.ceil(4.0 * float(sigma_d))) + 1
    if deriv == "spectral":
        return 0
    raise ValueError(f"unknown derivative scheme {deriv!r}; expected one of {DERIVS}")


def filtered(u, multipliers, *, deriv: str = "daf", sigma_d: float = 1.0,
             periodic: bool = False) -> list:
    """``u`` passed through each Fourier multiplier ``m(kx, ky)`` of ``multipliers`` times the
    scheme's smoothing gain; ``kx``, ``ky`` in rad/px. A non-periodic field is reflected past its
    edges by the scheme's reach first and cropped after."""
    if deriv not in DERIVS:
        raise ValueError(f"unknown derivative scheme {deriv!r}; expected one of {DERIVS}")
    if deriv == "spectral" and not periodic:
        raise ValueError("spectral derivatives need a periodized input: run the Periodic extension "
                         "(DAF) first, or declare the layer periodic; otherwise choose the daf or "
                         "gaussian scheme (see the edge handling)")
    complex_field = np.iscomplexobj(u)
    u = np.asarray(u, dtype=np.complex128 if complex_field else np.float64)
    ny, nx = u.shape
    pad = 0 if periodic else reach(deriv, sigma_d)
    up = np.pad(u, pad, mode="reflect") if pad else u
    ky = 2.0 * math.pi * np.fft.fftfreq(up.shape[0])[:, None]
    kx = 2.0 * math.pi * np.fft.fftfreq(up.shape[1])[None, :]
    gain = transfer(np.hypot(kx, ky), deriv, sigma_d=sigma_d)
    fft = fft_policy.double()
    spectrum = np.asarray(fft.fft2(up))
    out = []
    for m in multipliers:
        arr = np.asarray(fft.ifft2(spectrum * (gain * m(kx, ky))))
        if not complex_field:
            arr = np.real(arr)
        out.append(arr[pad:pad + ny, pad:pad + nx] if pad else arr)
    return out


#: The harmonic filter's band: the full width at half maximum of the Gaussian band-pass, as a
#: fraction of k₀ (spec 19 §3.5, "relative width 0.5").
BANDPASS_WIDTH = 0.5


#: The band kept around the spectral peak by the DAF harmonic filter (Goldman et al. 2002, fn. 18).
GOLDMAN_BAND = (0.7, 1.3)


def radial_filter(u, gain, *, periodic: bool = False, pad: int = 0):
    """``u`` through the radial Fourier multiplier ``gain(|k|)`` (|k| in rad/px), its mean removed;
    reflected by ``pad`` px first when not periodic."""
    u = np.asarray(u, dtype=np.float64)
    ny, nx = u.shape
    pad = 0 if periodic else int(pad)
    up = np.pad(u, pad, mode="reflect") if pad else u
    ky = 2.0 * math.pi * np.fft.fftfreq(up.shape[0])[:, None]
    kx = 2.0 * math.pi * np.fft.fftfreq(up.shape[1])[None, :]
    g = np.asarray(gain(np.hypot(kx, ky)), dtype=np.float64)
    g[0, 0] = 0.0
    fft = fft_policy.double()
    out = np.real(np.asarray(fft.ifft2(np.asarray(fft.fft2(up)) * g)))
    return out[pad:pad + ny, pad:pad + nx] if pad else out


def bandpass(u, k0: float, *, relative_width: float = BANDPASS_WIDTH, periodic: bool = False):
    """A Gaussian band-pass at k₀ of full width at half maximum ``relative_width``·k₀: the cheap
    harmonic filter, two wavelengths of reflection when the field is not periodic."""
    k0 = float(k0)
    sigma_k = relative_width * k0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return radial_filter(u, lambda k: np.exp(-0.5 * ((k - k0) / sigma_k) ** 2),
                         periodic=periodic, pad=math.ceil(4.0 * math.pi / k0))


def daf_filter(u, k_t: float, *, band=GOLDMAN_BAND, periodic: bool = False):
    """Hoffman et al. 2000's harmonic filter: the field made periodic by the self-tuned DAF
    extension (spec 25's rule; skipped when the input is periodic already), then the band
    ``band``·k_t around the spectral peak kept, which drops the harmonics and the drift together.
    Returns ``(filtered, block, tuned)``: the filtered field on the extended grid, the real block
    ``(rows, cols)`` at its origin, and the extension's tuning (None for a periodic input). Where
    the extension does not apply (``tuned.fallback``) the field is reflected by two wavelengths
    instead; the filled gap is never part of a statistic."""
    from dynamix.core import daf

    u = np.asarray(u, dtype=np.float64)
    block = u.shape
    tuned = None
    if not periodic:
        tuned = daf.self_tune(u)
        if tuned.fallback is None:
            u, periodic = daf.extend_2d(u, tuned), True
    lo, hi = (float(b) * float(k_t) for b in band)
    filtered = radial_filter(u, lambda k: (k >= lo) & (k <= hi), periodic=periodic,
                             pad=math.ceil(4.0 * math.pi / k_t))
    return filtered, block, tuned


def detrend(u, mode: str = "mean", mask=None):
    """``(values, valid)``: the field with its mean (or a plane) removed, fitted over the valid
    pixels (finite, and inside ``mask`` when given). Only non-finite pixels are set to zero before
    a transform; a masked pixel keeps its value and is left out of the statistics alone."""
    v = np.array(u, dtype=np.float64, copy=True)
    finite = np.isfinite(v)
    valid = finite.copy()
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if mode == "mean":
        v -= v[valid].mean()
    elif mode == "plane":
        yy, xx = np.nonzero(valid)
        a = np.column_stack([xx, yy, np.ones(xx.size)])
        coef, *_ = np.linalg.lstsq(a, v[valid], rcond=None)
        gy, gx = np.mgrid[0:v.shape[0], 0:v.shape[1]]
        v -= coef[0] * gx + coef[1] * gy + coef[2]
    elif mode != "none":
        raise ValueError(f"unknown detrend {mode!r}; expected mean, plane or none")
    v[~finite] = 0.0
    return v, valid


def helmholtz_density(u, k0: float, *, deriv: str = "daf", sigma_d: float = 1.0,
                      periodic: bool = False, mask=None, detrend_mode: str = "mean"):
    """The field-form disorder density D(x) = |(Δ + k₀²)u| / (k₀² ⟨|u|⟩) (PRE 57 Eq. 2), both
    terms and the normaliser under the scheme's smoothing; NaN where the field is invalid."""
    k0 = float(k0)
    v, valid = detrend(u, detrend_mode, mask)
    operator, smoothed = filtered(
        v, [lambda kx, ky: k0 * k0 - (kx * kx + ky * ky), lambda kx, ky: 1.0],
        deriv=deriv, sigma_d=sigma_d, periodic=periodic)
    density = np.abs(operator) / (k0 * k0 * np.mean(np.abs(smoothed[valid])))
    density[~valid] = np.nan
    return density


def _valid_values(density, mask):
    """The finite, unmasked densities as a flat copy, numerical zeros snapped to 0."""
    d = np.asarray(density, dtype=np.float64)
    valid = np.isfinite(d)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    vals = d[valid].copy()
    vals[vals < ZERO] = 0.0
    return vals


def _raw_moments(vals, b, level):
    """⟨x^β⟩ for each β of ``b``, the density clipped at ``level`` where β < 0."""
    out = np.empty_like(b)
    for i, beta in enumerate(b):
        x = np.maximum(vals, level) if beta < 0 else vals
        out[i] = np.mean(x ** beta)
    return out


def _floored(vals, b, floor, finish):
    """``(values, floor, n_floor, warning)`` for the moments ``finish(raw, b)``: the floor defaults
    to ``FLOOR_FRACTION`` of the mean density; a negative β counts the floored pixels and warns,
    with the change a tenfold higher floor would make."""
    if floor is None:
        floor = FLOOR_FRACTION * float(vals.mean())
    floor = float(floor)
    values = finish(_raw_moments(vals, b, floor), b)
    negative = b < 0
    n_floor, warning = 0, ""
    if negative.any():
        n_floor = int(np.count_nonzero(vals < floor))
        raised = finish(_raw_moments(vals, b, 10.0 * floor), b)
        change = float(np.max(np.abs(raised[negative] / values[negative] - 1.0)))
        area = vals.size
        warning = (f"negative-β moments weight the most ordered parts of the pattern (near-perfect "
                   f"stripes or flat regions) and so mostly amplify noise there; {n_floor} of {area} "
                   f"pixels ({n_floor / max(area, 1):.2%}) sit at the floor {floor:.3g}, and raising "
                   f"the floor tenfold changes them by up to {change:.1%}")
    return values, floor, n_floor, warning


def k0_direction(u, k0: float, *, mask=None) -> tuple:
    """The wavevector of magnitude ``k0`` along the power spectrum's peak, folded into the
    half-plane k_x > 0 (k_y > 0 on its edge): the k of PRL 1995 Eq. 1."""
    v, valid = detrend(u, "mean", mask)
    w = np.where(valid, v, 0.0)
    power = np.abs(np.asarray(fft_policy.double().fft2(w))) ** 2
    power[0, 0] = 0.0
    i, j = np.unravel_index(int(np.argmax(power)), power.shape)
    kx, ky = np.fft.fftfreq(w.shape[1])[j], np.fft.fftfreq(w.shape[0])[i]
    if kx < 0 or (kx == 0 and ky < 0):
        kx, ky = -kx, -ky
    theta = math.atan2(ky, kx)
    return (float(k0) * math.cos(theta), float(k0) * math.sin(theta))


def envelope_halfplane(u, k0_vec, *, periodic: bool = False):
    """The complex envelope A(x) of ``u`` about the wavevector ``k0_vec`` (PRL 1995 Eq. 1): the
    analytic signal from the Fourier half-plane k·k₀ > 0 (half weight on its edge), doubled, then
    demodulated by e^{−ik₀·x} (J. Stat. Phys. 1998 §4). The mean is removed; a non-periodic field
    is reflected by two wavelengths first."""
    k0x, k0y = (float(c) for c in k0_vec)
    u = np.asarray(u, dtype=np.float64)
    ny, nx = u.shape
    pad = 0 if periodic else int(math.ceil(4.0 * math.pi / math.hypot(k0x, k0y)))
    up = np.pad(u, pad, mode="reflect") if pad else u
    ky = 2.0 * math.pi * np.fft.fftfreq(up.shape[0])[:, None]
    kx = 2.0 * math.pi * np.fft.fftfreq(up.shape[1])[None, :]
    along = kx * k0x + ky * k0y
    weight = np.where(along > 0, 2.0, np.where(along == 0, 1.0, 0.0))
    weight[0, 0] = 0.0
    fft = fft_policy.double()
    z = np.asarray(fft.ifft2(np.asarray(fft.fft2(up)) * weight))
    y = np.arange(-pad, ny + pad, dtype=np.float64)[:, None]
    x = np.arange(-pad, nx + pad, dtype=np.float64)[None, :]
    a = z * np.exp(-1j * (k0x * x + k0y * y))
    return a[pad:pad + ny, pad:pad + nx] if pad else a


def envelope_density(a, k0_vec, *, deriv: str = "daf", sigma_d: float = 1.0,
                     periodic: bool = False, mask=None):
    """The envelope disorder density D^env(x) = |⊙A| / (k₀² ⟨|A|⟩), ⊙ = k₀·∇ − (i/2)∇² (PRL 1995
    Eqs. 2–3), one multiplier i(k₀·k + |k|²/2), A and the normaliser under the scheme's smoothing;
    NaN where the envelope is invalid."""
    k0x, k0y = (float(c) for c in k0_vec)
    k0_sq = k0x * k0x + k0y * k0y
    a = np.array(a, dtype=np.complex128, copy=True)
    valid = np.isfinite(a)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    a[~valid] = 0.0
    operator, smoothed = filtered(
        a, [lambda kx, ky: 1j * (k0x * kx + k0y * ky + 0.5 * (kx * kx + ky * ky)),
            lambda kx, ky: 1.0],
        deriv=deriv, sigma_d=sigma_d, periodic=periodic)
    density = np.abs(operator) / (k0_sq * np.mean(np.abs(smoothed[valid])))
    density[~valid] = np.nan
    return density


def phase_field_density(amplitude, kx, ky, div_k, k0: float, *, deriv: str = "daf",
                        sigma_d: float = 1.0, periodic: bool = False, mask=None):
    """The envelope disorder density in its phase-field form: for A = a e^{iφ} with the local
    wavevector K = k₀ + ∇φ, ⊙A = e^{iφ}·½[∇·(a²K)/a + i(a(|K|² − k₀²) − ∇²a)], so
    D^env = |⊙A| / (k₀² ⟨a⟩) needs a, K and ∇·K alone — no ±k partition, and the same under
    K → −K. a, ∇a and ∇²a carry the scheme's smoothing; K and ∇·K come from the ridge, which
    smoothed them already at the wavelet's own width. NaN where the ridge is invalid; an invalid
    pixel (a defect core) holds the valid mean under the transform, so no step is planted at its
    rim."""
    k0_sq = float(k0) ** 2
    a = np.array(amplitude, dtype=np.float64, copy=True)
    valid = np.isfinite(a) & np.isfinite(kx) & np.isfinite(ky) & np.isfinite(div_k)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    a[~valid] = a[valid].mean()
    ax, ay, lap, smoothed = filtered(
        a, [lambda kx_, ky_: 1j * kx_, lambda kx_, ky_: 1j * ky_,
            lambda kx_, ky_: -(kx_ * kx_ + ky_ * ky_), lambda kx_, ky_: 1.0],
        deriv=deriv, sigma_d=sigma_d, periodic=periodic)
    kx, ky, div_k = (np.where(valid, np.asarray(v, dtype=np.float64), 0.0) for v in (kx, ky, div_k))
    real = ax * kx + ay * ky + 0.5 * smoothed * div_k
    imag = 0.5 * (smoothed * (kx * kx + ky * ky - k0_sq) - lap)
    density = np.hypot(real, imag) / (k0_sq * np.mean(np.abs(smoothed[valid])))
    density[~valid] = np.nan
    return density


def disorder_moments(density, betas, *, mask=None, floor=None) -> dict:
    """The disorder at each moment order: ``mean`` δ̄(β) = (2 − β)⟨D^β⟩, ``total`` |Ω|·δ̄(β) in px²
    and ``rooted`` δ̄(β)^{1/β}, over the valid pixels of ``density``. β ≥ 2 and β = 0 are refused;
    a negative β takes the density clipped at ``floor`` (default ``FLOOR_FRACTION`` of the mean),
    with the count of floored pixels and a warning returned beside the values."""
    b = np.atleast_1d(np.asarray(betas, dtype=np.float64))
    if np.any(b >= 2.0):
        raise ValueError("β ≥ 2 is refused for the disorder: PRL 1995 Eq. 8 needs β < 2, the moment "
                         "of a target's center diverging there")
    if np.any(b == 0.0):
        raise ValueError("β = 0 is refused: D⁰ = 1 says nothing about the pattern")
    vals = _valid_values(density, mask)
    area = int(vals.size)
    means, floor, n_floor, warning = _floored(vals, b, floor, lambda raw, bb: (2.0 - bb) * raw)
    return {"beta": b, "mean": means, "total": area * means, "rooted": means ** (1.0 / b),
            "area": area, "floor": floor, "n_floor": n_floor, "warning": warning}


def hessian_density(u, *, deriv: str = "daf", sigma_d: float = 1.0, periodic: bool = False,
                    mask=None, norm: str = "variance", detrend_mode: str = "mean"):
    """The Hessian density |Λ(x)| = |u_xx u_yy − u_xy²| / Var(u) (PRE 71:035101 Eq. 1), the
    variance over the valid pixels; ``norm = "std"`` is the diagnostic variant, which breaks the
    inverse-length dimensions. NaN where the field is invalid."""
    if norm not in ("variance", "std"):
        raise ValueError(f"unknown hessian_norm {norm!r}; expected variance or std")
    v, valid = detrend(u, detrend_mode, mask)
    uxx, uyy, uxy = filtered(
        v, [lambda kx, ky: -kx * kx, lambda kx, ky: -ky * ky, lambda kx, ky: -kx * ky],
        deriv=deriv, sigma_d=sigma_d, periodic=periodic)
    spread = np.var(v[valid]) if norm == "variance" else np.std(v[valid])
    density = np.abs(uxx * uyy - uxy * uxy) / spread
    density[~valid] = np.nan
    return density


def hessian_measure(density, betas, *, mask=None, floor=None) -> dict:
    """The Hessian measure μ(β) = ⟨|Λ|^β⟩^{1/(4β)} (PRE 71:035101 Eq. 1), an inverse length in px⁻¹,
    over the valid pixels of ``density``; β = 0 is refused, a negative β is floored as in
    :func:`disorder_moments`."""
    b = np.atleast_1d(np.asarray(betas, dtype=np.float64))
    if np.any(b == 0.0):
        raise ValueError("β = 0 is refused: the Hessian measure's 1/(4β) root is undefined there")
    vals = _valid_values(density, mask)
    mu, floor, n_floor, warning = _floored(vals, b, floor, lambda raw, bb: raw ** (1.0 / (4.0 * bb)))
    return {"beta": b, "mu": mu, "area": int(vals.size), "floor": floor, "n_floor": n_floor,
            "warning": warning}


#: How k₀ is found: the minimum of δ̄(1) over a bracket around the spectral peak (Gunaratne,
#: Ratnaweera and Tennekone 1999, PRE 59:5058), or the peak of the radially averaged power spectrum.
K0_METHODS = ("min_disorder", "spectral_peak")


def _spectral_peak(v, valid) -> float:
    """The peak of the annulus-mean power spectrum, in rad/px, at the bin's centre."""
    w = np.where(valid, v, 0.0)
    ky = 2.0 * math.pi * np.fft.fftfreq(w.shape[0])[:, None]
    kx = 2.0 * math.pi * np.fft.fftfreq(w.shape[1])[None, :]
    k = np.hypot(kx, ky)
    power = np.abs(np.asarray(fft_policy.double().fft2(w))) ** 2
    dk = 2.0 * math.pi / min(w.shape)
    bins = np.floor(k / dk + 0.5).astype(np.int64)
    counts = np.bincount(bins.ravel())
    sums = np.bincount(bins.ravel(), weights=power.ravel())
    mean_power = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    mean_power[0] = 0.0                                   # the mean, removed already
    return float(np.argmax(mean_power) * dk)


def k0_estimate(u, method: str = "min_disorder", *, deriv: str = "daf", sigma_d: float = 1.0,
                periodic: bool = False, mask=None, bracket=(0.5, 2.0),
                detrend_mode: str = "mean") -> float:
    """The planform wavenumber k₀ in rad/px (:data:`K0_METHODS`). ``min_disorder`` minimises
    δ̄(1; k₀) = ⟨|S u − Δ_S u / k₀²|⟩ / ⟨|S u|⟩ over k₀ in ``bracket`` times the spectral peak,
    which is convex and piecewise linear in 1/k₀² (S the scheme's smoothing, Δ_S the smoothed
    Laplacian), so the minimum is unique."""
    if method not in K0_METHODS:
        raise ValueError(f"unknown k0 method {method!r}; expected one of {K0_METHODS}")
    v, valid = detrend(u, detrend_mode, mask)
    peak = _spectral_peak(v, valid)
    if method == "spectral_peak":
        return peak
    from scipy.optimize import minimize_scalar

    smoothed, laplacian = filtered(
        v, [lambda kx, ky: 1.0, lambda kx, ky: -(kx * kx + ky * ky)],
        deriv=deriv, sigma_d=sigma_d, periodic=periodic)
    s_u, l_u = smoothed[valid], laplacian[valid]
    denominator = np.mean(np.abs(s_u))

    def mean_disorder(s):                                  # s = 1/k₀²
        return float(np.mean(np.abs(s_u + s * l_u)) / denominator)

    lo, hi = sorted((1.0 / (bracket[1] * peak) ** 2, 1.0 / (bracket[0] * peak) ** 2))
    best = minimize_scalar(mean_disorder, bounds=(lo, hi), method="bounded",
                           options={"xatol": 1e-12 * hi})
    return float(1.0 / math.sqrt(best.x))
