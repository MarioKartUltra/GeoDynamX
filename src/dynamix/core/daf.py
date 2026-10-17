# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The DAF periodic extension: a non-periodic field made periodic by filling a gap after it.

The Hermite distributed approximating functional (Frishman, Hoffman, Rakauskas and Kouri 1996,
Eq. 9) is a Gaussian times a truncated Hermite series; as a filter its gain is
e^{-x} Σ_{j ≤ M/2} x^j / j!, x = (kσ)²/2 (Hoffman, Gunaratne, Zhang and Kouri 2000, Eq. 5), flat to
order M in the passband. Unknown samples are chosen so the line agrees with its own DAF smoothing,
min Σ_k W_k |g_k − (δ_M ∗ g)_k|² (Hoffman et al. 2000, Eq. 6): in a gap (Frishman's case), or after
the data on a circle whose period is the data plus the gap (the periodic extension, §III B), line by
line along x and then along y (§III C).

The minimiser is found by least squares on the weighted rows the unknowns touch, never by its
normal equations: the DAF barely constrains the smooth fills of a long gap, and squaring that
system's condition (to 10^15 on Frishman's Fig. 5) loses the answer.

The self-tuning rule (spec 22 §3.4) takes every setting from the data: the band limit f_b holds 99%
of the Hann-windowed power, the gap is two wavelengths at it (at least 8 px), M = 12, σ the largest
with gain 0.99 at f_b, and a Tikhonov term λ‖g − ḡ‖² (λ = 1e-3) steadies the gap values the solve
barely constrains. Below six points per wavelength at f_b, or where the stencil does not fit a line,
it falls back to reflection with the reason. The filled values are synthetic and never enter an
analysis.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = ["SelfTuned", "band_limit", "daf_gain", "extend_2d", "extend_periodic", "fill_gap",
           "hermite_daf", "self_tune", "sigma_for_gain"]

#: The self-tuned extension's Hermite order.
M_SELF = 12
#: Share of the Hann-windowed power below the band limit.
BAND_FRAC = 0.99
#: The DAF gain the self-tuned σ keeps at the band limit.
GAIN = 0.99
#: Tikhonov weight on the gap values' distance from the line's mean.
LAM = 1e-3
#: Fewest points per wavelength at the band limit.
MIN_PPW = 6.0
#: Shortest gap, px.
MIN_GAP = 8
#: Taps smaller than this share of the largest are dropped.
_TAP_FLOOR = 1e-13


def hermite_daf(sigma: float, M: int) -> np.ndarray:
    """Taps δ_M(j | σ, Δ = 1) for j = −w..w (Frishman et al. 1996 Eq. 9): the Gaussian
    exp(−j²/2σ²)/(√(2π) σ) times Σ_{n ≤ M/2} (−¼)ⁿ/n! H_2n(j/(σ√2)), cut where the taps fall below
    1e-13 of the largest."""
    from numpy.polynomial import hermite as H

    w = int(math.ceil(12.0 * sigma))
    j = np.arange(-w, w + 1, dtype=np.float64)
    z = j / (sigma * math.sqrt(2.0))
    c = np.zeros(int(M) + 1)
    for n in range(int(M) // 2 + 1):
        c[2 * n] = (-0.25) ** n / math.factorial(n)
    taps = np.exp(-z * z) / (math.sqrt(2.0 * math.pi) * sigma) * H.hermval(z, c)
    keep = np.flatnonzero(np.abs(taps) > _TAP_FLOOR * np.abs(taps).max())
    half = int(max(abs(j[keep[0]]), abs(j[keep[-1]])))
    return taps[w - half:w + half + 1]


def daf_gain(k, sigma: float, M: int):
    """The Hermite DAF's gain at angular frequency ``k`` (rad/px), Hoffman et al. 2000 Eq. 5."""
    from scipy.special import gammaincc

    return gammaincc(int(M) // 2 + 1, 0.5 * (np.asarray(k, dtype=np.float64) * sigma) ** 2)


def sigma_for_gain(f_b: float, M: int, gain: float = GAIN) -> float:
    """The largest σ (px) whose gain at ``f_b`` cycles/px is still ``gain``."""
    from scipy.special import gammainccinv

    return float(math.sqrt(2.0 * gammainccinv(int(M) // 2 + 1, gain)) / (2.0 * math.pi * f_b))


def _distance_to(unk: np.ndarray, idx: np.ndarray, n: int, periodic: bool) -> np.ndarray:
    d = np.abs(idx[:, None] - unk[None, :])
    if periodic:
        d = np.minimum(d, n - d)
    return d.min(axis=1)


def fill_gap(values, known, sigma: float, M: int, *, periodic: bool, weights=None,
             lam: float = 0.0) -> np.ndarray:
    """The DAF choice of the unknown samples of one line or of a ``(lines, n)`` stack sharing the
    mask ``known``: min Σ_k W_k |g_k − (δ_M ∗ g)_k|² + λ Σ_u |g_u − ḡ|², ḡ the line's known mean.
    Periodic: the line is a circle, and by default W = 1 on the unknowns and exp(−d²/2σ²) at
    distance d from them (Hoffman et al. 2000 §III A). Not periodic: by default W = 1 on the rows
    whose stencil fits the line (Frishman et al. 1996), 0 elsewhere. ``weights`` (length n)
    replaces the default. ``ValueError`` where the stencil does not fit the line."""
    g = np.array(np.atleast_2d(values), dtype=np.float64)
    known = np.asarray(known, dtype=bool)
    n = known.size
    taps = hermite_daf(sigma, M)
    half = taps.size // 2
    if (2 * half >= n) if periodic else (2 * half + 1 > n):
        raise ValueError(f"the DAF stencil (±{half} px) does not fit a {n} px line")
    unk, kn = np.flatnonzero(~known), np.flatnonzero(known)
    idx = np.arange(n)
    rows = idx[_distance_to(unk, idx, n, periodic) <= half]      # the rows the unknowns touch
    offsets = np.arange(-half, half + 1)
    cols = rows[:, None] + offsets[None, :]
    A = np.zeros((rows.size, n))
    A[np.arange(rows.size), rows] = 1.0
    r_i = np.broadcast_to(np.arange(rows.size)[:, None], cols.shape)
    vals = np.broadcast_to(-taps[None, :], cols.shape)
    if periodic:
        np.add.at(A, (r_i, cols % n), vals)
    else:
        ok = (cols >= 0) & (cols < n)
        np.add.at(A, (r_i[ok], cols[ok]), vals[ok])
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)[rows]
    elif periodic:
        w = np.where(~known[rows], 1.0,
                     np.exp(-_distance_to(unk, rows, n, True) ** 2 / (2.0 * sigma ** 2)))
    else:
        w = ((rows >= half) & (rows < n - half)).astype(np.float64)
    sw = np.sqrt(w)[:, None]
    lhs, rhs = sw * A[:, unk], -(sw * A[:, kn]) @ g[:, kn].T
    if lam > 0.0:
        mean = g[:, kn].mean(axis=1)
        lhs = np.vstack([lhs, math.sqrt(lam) * np.eye(unk.size)])
        rhs = np.vstack([rhs, math.sqrt(lam) * np.broadcast_to(mean, (unk.size, mean.size))])
    g[:, unk] = np.linalg.lstsq(lhs, rhs, rcond=None)[0].T
    return g if np.ndim(values) == 2 else g[0]


def extend_periodic(lines, m: int, sigma: float, M: int, lam: float = LAM) -> np.ndarray:
    """Each line of ``lines`` (``(lines, n)``) followed by ``m`` DAF values that make it periodic
    with period ``n + m``."""
    a = np.atleast_2d(np.asarray(lines, dtype=np.float64))
    g = np.concatenate([a, np.zeros((a.shape[0], int(m)))], axis=1)
    known = np.r_[np.ones(a.shape[1], dtype=bool), np.zeros(int(m), dtype=bool)]
    return fill_gap(g, known, sigma, M, periodic=True, lam=lam)


def band_limit(values, frac: float = BAND_FRAC) -> float:
    """The radial frequency (cycles/px) below which ``frac`` of the field's Hann-windowed power
    (mean removed) lies; a line or a stack of lines reads along its last axis."""
    from dynamix.core import fft_policy

    a = np.atleast_2d(np.asarray(values, dtype=np.float64))
    a = (a - a.mean()) * np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    power = np.abs(fft_policy.double().rfft2(a)) ** 2
    kx = np.fft.rfftfreq(a.shape[1])
    power[:, 1:(a.shape[1] + 1) // 2] *= 2.0              # the negative kx the half-plane folds in
    k = np.hypot(np.fft.fftfreq(a.shape[0])[:, None], kx[None, :]).ravel()
    order = np.argsort(k)
    cum = np.cumsum(power.ravel()[order])
    if not cum[-1] > 0.0:
        return 0.0
    return float(k[order][np.searchsorted(cum, frac * cum[-1])])


@dataclasses.dataclass(frozen=True)
class SelfTuned:
    """The self-tuned extension's settings and, where it does not apply, why (``fallback``)."""

    f_b: float
    gap: int
    M: int
    sigma: float
    lam: float
    ppw: float
    fallback: "str | None"


def self_tune(values) -> SelfTuned:
    """Every setting of the periodic extension from the data (spec 22 §3.4)."""
    a = np.atleast_2d(np.asarray(values, dtype=np.float64))
    f_b = band_limit(a)
    if not f_b > 0.0:
        return SelfTuned(0.0, MIN_GAP, M_SELF, 0.0, LAM, math.inf,
                         "the field has no band limit (it is constant)")
    ppw = 1.0 / f_b
    gap = max(MIN_GAP, int(round(2.0 / f_b)))
    sigma = sigma_for_gain(f_b, M_SELF)
    fallback = None
    if ppw < MIN_PPW:
        fallback = (f"fewer than {MIN_PPW:g} points per wavelength at the band limit "
                    f"({ppw:.1f} px at f_b = {f_b:.3f} cycles/px): the data are not band-limited "
                    "enough for the DAF extension; reflection used")
    else:
        half = hermite_daf(sigma, M_SELF).size // 2
        shortest = min(a.shape) if a.shape[0] > 1 else a.shape[1]
        if 2 * half >= shortest + gap:
            fallback = (f"the DAF stencil (±{half} px at σ = {sigma:.1f} px) does not fit a "
                        f"{shortest} px line with its {gap} px gap; reflection used")
    return SelfTuned(f_b, gap, M_SELF, sigma, LAM, ppw, fallback)


def extend_2d(values, tuned: SelfTuned) -> np.ndarray:
    """The field made periodic: each row extended by the gap along x, then each column of that
    along y (Hoffman et al. 2000 §III C); shape ``(ny + gap, nx + gap)``, the data at the origin."""
    a = np.asarray(values, dtype=np.float64)
    ex = extend_periodic(a, tuned.gap, tuned.sigma, tuned.M, tuned.lam)
    return extend_periodic(ex.T, tuned.gap, tuned.sigma, tuned.M, tuned.lam).T
