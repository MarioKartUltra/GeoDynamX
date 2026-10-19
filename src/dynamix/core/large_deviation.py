# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The large-deviation (coarse-grain) spectrum of a per-scale set of moduli: wavelet leaders,
p-leaders, (θ, ω)-leaders, bare coefficients or the WTMM's maxima, whatever carries one value
per cell (or line) per scale.

Seuret 2016, Definition 2.13 (the ε → 0 and resolution → ∞ limits made explicit): with the
local exponent of a cell h_λ = log2 M_λ / log2(a/L) (a the scale, L the field's extent, so a/L
plays 2^{−j} of the dyadic literature), N^ε(a, h) = #{λ : |h_λ − h| ≤ ε} counts the cells at
scale a within ε of h, and D_LD(h) is minus the slope of log2 N^ε(a, h) against log2(a/L) over
the fit window: the count grows as (a/L)^{−D} towards fine scales. Two bin rules:

- ``fixed``: a sweep of ε on a fixed h grid (the spread across ε is the convergence check);
- ``adaptive`` (Barral and Gonçalves 2011, as restated in Gonçalves' arXiv 1209.4806 eq. 8:
  h_j = Λ_j′(q), ε_j = (|Λ_j″(q)| / j)^{1/2} with Λ_j the log-Laplace functional of the local
  exponents): per scale and per q the bin centre is the q-tilted mean of h_λ and its half-width
  (ln 2)^{1/2} times the q-tilted standard deviation, so the bin follows the population a
  Boltzmann weight M^q picks out; the count's scaling gives D(q) and the tilted mean's slope
  gives h(q) (the canonical h(q) of the partition tables, so the pair is prefactor-free). A
  finite bin overestimates D where the spectrum falls steeply (its count follows the most
  numerous cells inside it), by a few tenths in the tails over four octaves; a q-tilt cannot
  enter a non-concave dip (the tilted population is bimodal there): the adaptive route reads
  the concave hull, the fixed route reaches the dip.

The chain D_H(h) ≤ D_LD(h) ≤ L(h) (Leonarduzzi et al. 2018, Cor. 1; Seuret §2.4.4) is checked
against the Legendre spectrum of the same input by :func:`hull_excess`. The prefactor of the
moduli (their amplitude at a = L) shifts every fixed-rule h_λ by log2 C / log2(a/L), less at
finer scales; it is the fixed route's known bias, reported through :func:`prefactor_shift`,
not corrected.
"""
from __future__ import annotations

import numpy as np

__all__ = ["adaptive", "counts", "fixed", "hull_excess", "local_exponents", "prefactor_shift"]

#: The fixed route's default sweep of half-widths in h.
EPS_SWEEP = (0.10, 0.05, 0.025)


def _scales(scales_px, length_px, log2_a_min, log2_a_max) -> tuple:
    a = np.asarray(scales_px, dtype=np.float64)
    L = float(length_px)
    if np.any(a >= L):
        raise ValueError(f"every scale must lie below the field extent L = {L} px")
    keep = [k for k, s in enumerate(a) if log2_a_min - 1e-9 <= np.log2(s) <= log2_a_max + 1e-9]
    return a, L, keep


def _samples(mods, counts_of, weight: float = 1.0) -> tuple:
    m = np.asarray(mods, dtype=np.float64).ravel()
    c = np.ones_like(m) if counts_of is None else np.asarray(counts_of, dtype=np.float64).ravel()
    ok = np.isfinite(m) & (m > 0) & (c > 0)
    return m[ok], c[ok] * float(weight)


def _weights(scale_weight, n: int) -> np.ndarray:
    """One multiplicative factor on the counts per scale: the ratio of the dyadic grid's cell
    count to the cells kept, so a border that thins coarse levels faster leaves D(0) = d."""
    return np.ones(n) if scale_weight is None else np.asarray(scale_weight, dtype=np.float64)


def local_exponents(mods, scale_px: float, length_px: float) -> np.ndarray:
    """h_λ = log2 M_λ / log2(a/L) for the positive finite moduli at one scale."""
    m, _ = _samples(mods, None)
    return np.log2(m) / np.log2(float(scale_px) / float(length_px))


def _window_counts(h, c, centres, eps) -> np.ndarray:
    """Σ of ``c`` over |h − centre| ≤ eps for every centre (``eps`` a scalar or one per centre)."""
    order = np.argsort(h)
    hs, cum = h[order], np.concatenate([[0.0], np.cumsum(c[order])])
    lo = np.searchsorted(hs, np.asarray(centres) - eps - 1e-12, side="left")
    hi = np.searchsorted(hs, np.asarray(centres) + eps + 1e-12, side="right")
    return cum[hi] - cum[lo]


def counts(mods_per_scale, scales_px, h_grid, eps: float, *, length_px: float,
           counts_per_scale=None, scale_weight=None) -> np.ndarray:
    """N^ε(a, h): ``(n_scales, n_h)`` cells (or their multiplicities, times the scale's
    weight) within ``eps`` of each grid value, at every scale."""
    a = np.asarray(scales_px, dtype=np.float64)
    wts = _weights(scale_weight, a.size)
    out = np.zeros((a.size, len(h_grid)))
    for k in range(a.size):
        m, c = _samples(mods_per_scale[k], None if counts_per_scale is None else counts_per_scale[k], wts[k])
        if m.size:
            h = np.log2(m) / np.log2(a[k] / float(length_px))
            out[k] = _window_counts(h, c, np.asarray(h_grid, dtype=np.float64), float(eps))
    return out


def _minus_slopes(x, Y) -> np.ndarray:
    """−slope of each row of ``Y`` against ``x``; NaN where a row has a non-finite entry."""
    x = np.asarray(x, dtype=np.float64)
    xm = x - x.mean()
    ok = np.all(np.isfinite(Y), axis=-1)
    s = np.full(Y.shape[:-1], np.nan)
    if ok.any():
        Ym = Y[ok] - Y[ok].mean(axis=-1, keepdims=True)
        s[ok] = -(Ym @ xm) / float(xm @ xm)
    return s


def fixed(mods_per_scale, scales_px, *, h_grid, log2_a_min: float, log2_a_max: float,
          length_px: float, eps_list=EPS_SWEEP, counts_per_scale=None, scale_weight=None) -> dict:
    """D_LD on ``h_grid`` for each half-width of ``eps_list`` over the scales inside the
    window: ``D`` is ``(n_eps, n_h)`` (NaN where a bin is empty at some scale), ``spread`` the
    range of D across ε per h (the convergence check), ``log2_counts`` ``(n_eps, n_kept, n_h)``."""
    a, L, keep = _scales(scales_px, length_px, log2_a_min, log2_a_max)
    h_grid = np.asarray(h_grid, dtype=np.float64)
    eps_list = tuple(float(e) for e in eps_list)
    if not eps_list or not all(np.isfinite(e) and e > 0 for e in eps_list):
        raise ValueError("epsilon half-widths must be finite and positive")
    nan = np.full((len(eps_list), h_grid.size), np.nan)
    if len(keep) < 2:
        return {"h": h_grid, "D": nan, "eps": eps_list, "spread": np.full(h_grid.size, np.nan),
                "epsilon_count": np.zeros(h_grid.size, dtype=int),
                "log2_counts": np.full((len(eps_list), len(keep), h_grid.size), np.nan),
                "scales": a[keep]}
    x = np.log2(a[keep] / L)
    sub_mods = [mods_per_scale[k] for k in keep]
    sub_counts = None if counts_per_scale is None else [counts_per_scale[k] for k in keep]
    sub_w = _weights(scale_weight, a.size)[keep]
    log2_n = np.empty((len(eps_list), len(keep), h_grid.size))
    for i, eps in enumerate(eps_list):
        n = counts(sub_mods, a[keep], h_grid, eps, length_px=L, counts_per_scale=sub_counts,
                   scale_weight=sub_w)
        with np.errstate(divide="ignore"):
            log2_n[i] = np.where(n > 0, np.log2(np.where(n > 0, n, 1.0)), np.nan)
    D = _minus_slopes(x, np.swapaxes(log2_n, 1, 2))          # (n_eps, n_h) rows over the scales
    epsilon_count = np.isfinite(D).sum(axis=0)
    common = (epsilon_count == len(eps_list)) & (len(eps_list) >= 2)
    spread = np.full(h_grid.size, np.nan)
    spread[common] = np.ptp(D[:, common], axis=0)
    return {"h": h_grid, "D": D, "eps": eps_list, "spread": spread, "log2_counts": log2_n,
            "epsilon_count": epsilon_count, "scales": a[keep]}


def adaptive(mods_per_scale, scales_px, *, q_list, log2_a_min: float, log2_a_max: float,
             length_px: float, counts_per_scale=None, scale_weight=None) -> dict:
    """The (h(q), D(q)) curve of the adaptive bins: per kept scale and q the centre is the
    q-tilted mean of h_λ and the half-width (ln 2)^{1/2} times its q-tilted standard deviation
    (a finite-scale variance-bandwidth adaptation, not a verified implementation of the
    original Barral–Gonçalves estimator); ``D`` is minus the
    count's slope against log2(a/L), ``h`` the slope of the tilted mean of log2 M against the
    same (the partition tables' h(q)); ``centres`` and ``eps`` are ``(n_kept, n_q)``."""
    a, L, keep = _scales(scales_px, length_px, log2_a_min, log2_a_max)
    q = np.asarray(q_list, dtype=np.float64)
    if len(keep) < 2:
        nan = np.full(q.size, np.nan)
        return {"q": q, "h": nan, "D": nan.copy(), "centres": np.full((len(keep), q.size), np.nan),
                "eps": np.full((len(keep), q.size), np.nan),
                "log2_counts": np.full((len(keep), q.size), np.nan), "scales": a[keep]}
    x = np.log2(a[keep] / L)
    wts = _weights(scale_weight, a.size)
    centres, eps, log2_n, mean_log2 = (np.full((len(keep), q.size), np.nan) for _ in range(4))
    for i, k in enumerate(keep):
        m, c = _samples(mods_per_scale[k], None if counts_per_scale is None else counts_per_scale[k], wts[k])
        if m.size == 0:
            continue
        lm = np.log2(m)
        h = lm / x[i]
        lw = np.log(c)[None, :] + q[:, None] * np.log(m)[None, :]
        w = np.exp(lw - lw.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        centre = w @ h
        sd = np.sqrt(np.log(2.0) * np.maximum((w * (h[None, :] - centre[:, None]) ** 2).sum(axis=1), 0.0))
        n = np.array([_window_counts(h, c, np.array([cc]), float(ee))[0] for cc, ee in zip(centre, sd)])
        centres[i], eps[i], mean_log2[i] = centre, sd, w @ lm
        with np.errstate(divide="ignore"):
            log2_n[i] = np.where(n > 0, np.log2(np.where(n > 0, n, 1.0)), np.nan)
    D = _minus_slopes(x, log2_n.T)
    h = -_minus_slopes(x, mean_log2.T)
    return {"q": q, "h": h, "D": D, "centres": centres, "eps": eps, "log2_counts": log2_n,
            "scales": a[keep]}


def prefactor_shift(mods_per_scale, scales_px, *, log2_a_min: float, log2_a_max: float,
                    length_px: float, counts_per_scale=None) -> np.ndarray:
    """Per kept scale, the mean of h_λ less the slope-based h(0) of the same cells: what the
    moduli's amplitude at a = L adds to every fixed-rule exponent there, log2 C / log2(a/L)."""
    a, L, keep = _scales(scales_px, length_px, log2_a_min, log2_a_max)
    x = np.log2(a[keep] / L)
    means = np.full(len(keep), np.nan)
    for i, k in enumerate(keep):
        m, c = _samples(mods_per_scale[k], None if counts_per_scale is None else counts_per_scale[k])
        if m.size:
            means[i] = float((c * np.log2(m)).sum() / c.sum())
    if len(keep) < 2 or not np.all(np.isfinite(means)):
        return np.full(len(keep), np.nan)
    h0 = float(np.polyfit(x, means, 1)[0])
    return means / x - h0


def hull_excess(h, D, q_list, tau) -> np.ndarray:
    """D − L(h): how far each point stands above the Legendre spectrum of the fitted τ(q)
    (``spectra.legendre_dh``), the bound D_LD ≤ L; NaN outside the hull's h range."""
    from dynamix.core.spectra import legendre_dh

    hull = legendre_dh(np.asarray(q_list, dtype=np.float64), np.asarray(tau, dtype=np.float64))
    hh, dd = np.asarray(hull["h"], dtype=np.float64), np.asarray(hull["D"], dtype=np.float64)
    ok = np.isfinite(hh) & np.isfinite(dd)
    if ok.sum() < 2:
        return np.full(np.asarray(h).shape, np.nan)
    order = np.argsort(hh[ok])
    hs, ds = hh[ok][order], dd[ok][order]
    h = np.asarray(h, dtype=np.float64)
    inside = (h >= hs[0]) & (h <= hs[-1])
    out = np.full(h.shape, np.nan)
    out[inside] = np.asarray(D, dtype=np.float64)[inside] - np.interp(h[inside], hs, ds)
    return out
