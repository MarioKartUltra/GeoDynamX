# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Marginal multiplier kernels and spatial covariance of wavelet log magnitudes.

    Arnéodo et al., PRL 80:708 (1998); Decoster et al., EPJB 15:739 (2000).
    Skeleton pairs use their recorded positions, without zero-filled missing samples.
"""
from __future__ import annotations

import numpy as np


def self_similarity(mods, scales, pairs, p, *, denominator_min=0.05, min_count=20):
    p, scales = np.asarray(p, float), np.asarray(scales, float)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any(np.diff(p)<=0) or not 0 < denominator_min < 1:
        raise ValueError("finite p grid and denominator threshold in (0,1) required")
    phi, counts = [], []
    for values in mods:
        v = np.asarray(values, float).ravel()
        v = np.log(v[np.isfinite(v) & (v > 0)])
        counts.append(v.size)
        # Keep memory proportional to the sample count, not count times the p grid.
        phi.append(np.array([np.mean(np.exp(1j*t*v)) if v.size else np.nan+0j for t in p]))
    from dynamix.core.spectra import ln_moments
    cumulants = ln_moments(mods)
    rows = []
    for fine, coarse in pairs:
        if not 0 <= fine < len(mods) or not 0 <= coarse < len(mods) or scales[fine] >= scales[coarse]:
            raise ValueError("each pair requires a fine scale smaller than its coarse scale")
        good = (np.abs(phi[coarse]) >= denominator_min) & np.isfinite(phi[fine])
        if min(counts[fine], counts[coarse]) < min_count:
            good[:] = False
        ratio = np.full(p.shape, np.nan+1j*np.nan)
        ratio[good] = phi[fine][good]/phi[coarse][good]
        phase = np.full(p.shape, np.nan)
        # Phase is determined only on the reliable interval connected to p=0.
        if p.size:
            zero = int(np.argmin(np.abs(p)))
            if np.isclose(p[zero], 0) and good[zero]:
                left = zero
                right = zero
                while left > 0 and good[left-1]:
                    left -= 1
                while right+1 < p.size and good[right+1]:
                    right += 1
                ph = np.unwrap(np.angle(ratio[left:right+1]))
                phase[left:right+1] = ph-ph[zero-left]
        with np.errstate(divide="ignore", invalid="ignore"):
            logabs = np.log(np.abs(ratio))
        rows.append({"pair": [int(fine), int(coarse)], "logabs": logabs, "phase": phase,
                     "ratio": ratio, "valid": good, "depth": float(np.log(scales[coarse]/scales[fine])),
                     "cumulant_difference": cumulants[:,fine]-cumulants[:,coarse]})
    return {"p": p, "counts": counts, "rows": rows, "definition": "characteristic function of ln positive modulus"}


def local_energy(modulus, scale, *, width=1., truncate=3.):
    """Half-log local energy with a normalized Gaussian bump and full-support validity.

    In 2-D the paper's a^-4 integral is a^-2 times a normalized local average.
    Pixels whose bump touches a hole or the frame are invalid, not padded observations.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter
    a = float(scale)
    if a <= 0 or width <= 0 or truncate <= 0:
        raise ValueError("scale and bump widths must be positive")
    m = np.asarray(modulus, float)
    sigma = a*width
    radius = int(truncate*sigma+0.5)
    good = np.isfinite(m)
    energy = gaussian_filter(np.where(good, m*m, 0), sigma, mode="constant", truncate=truncate)/a**2
    invalid = maximum_filter(~good, size=2*radius+1, mode="constant", cval=1)
    with np.errstate(divide="ignore"):
        return np.where(~invalid & (energy > 0), .5*np.log(energy), np.nan)


def _bin_lags(dx, dy, product, valid, edges, *, pitch, direction=None, half_width=15.):
    x, y = dx*pitch[0], dy*pitch[1]
    r = np.hypot(x, y)
    good = (valid > 0) & (r > 0)
    if direction is not None:
        from dynamix.core.orientation import Frame
        if direction.frame is not Frame.PIXEL:
            raise ValueError("lag sector requires native pixel-axis angle; convert the frame explicitly")
        delta = (np.degrees(np.arctan2(y, x))-float(direction.degrees)+90) % 180-90
        good &= np.abs(delta) <= half_width
    bins = np.searchsorted(edges, r[good], side="right")-1
    use = (bins >= 0) & (bins < len(edges)-1)
    count = np.bincount(bins[use], weights=valid[good][use], minlength=len(edges)-1)
    total = np.bincount(bins[use], weights=product[good][use], minlength=len(edges)-1)
    return total, count


def _result(total, counts, edges, min_pairs):
    covariance = np.full(total.shape, np.nan)
    np.divide(total, counts, out=covariance, where=counts >= min_pairs)
    return {"r": np.sqrt(np.asarray(edges[:-1])*np.asarray(edges[1:])),
            "covariance": covariance, "counts": counts, "edges": np.asarray(edges),
            "centering": "per-scale valid-population mean", "normalization": "covariance, not Pearson"}


def dense_covariance(first, second, edges, *, pitch=(1., 1.), direction=None, half_width=15.,
                     min_pairs=20, fft=None):
    """Nonwrapping masked covariance, with exact overlap counts at each displacement."""
    from dynamix.core.fft_policy import double
    a, b = np.asarray(first, float), np.asarray(second, float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("dense magnitudes must share a 2-D native grid")
    edges = _edges(edges)
    ma, mb = np.isfinite(a), np.isfinite(b)
    a = np.where(ma, a-(a[ma].mean() if ma.any() else 0), 0)
    b = np.where(mb, b-(b[mb].mean() if mb.any() else 0), 0)
    fft = double() if fft is None else fft
    shape = tuple(2*n-1 for n in a.shape)
    def padded(v):
        out = np.zeros(shape)
        out[:v.shape[0], :v.shape[1]] = v
        return out
    def correlate(x, y):
        out = fft.ifft2(np.conj(np.asarray(fft.fft2(padded(x))))*np.asarray(fft.fft2(padded(y))))
        return np.fft.fftshift(np.asarray(out).real)
    product = correlate(a, b)
    counts = np.maximum(0, np.rint(correlate(ma.astype(float), mb.astype(float))))
    ny, nx = a.shape
    dy, dx = np.mgrid[-ny+1:ny, -nx+1:nx]
    total, n = _bin_lags(dx, dy, product, counts, edges, pitch=pitch,
                         direction=direction, half_width=half_width)
    out = _result(total, n, edges, min_pairs)
    common = ma & mb
    out["zero"] = float((a[common]*b[common]).mean()) if common.any() else np.nan
    return out


def _edges(edges):
    edges = np.asarray(edges, float)
    if edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all() or edges[0] <= 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("lag edges must be positive, finite and strictly increasing")
    return edges


def skeleton_covariance(first, second, edges, *, pitch=(1., 1.), direction=None,
                        half_width=15., min_pairs=20, max_pairs=5_000_000, cancelled=None):
    """Exact native-coordinate pairs in radial/axial-sector bins, with a bounded pair budget."""
    from scipy.spatial import cKDTree
    edges = _edges(edges)
    def data(s):
        m = np.asarray(s["mod"], float)
        xy = np.column_stack((s["x"], s["y"]))*np.asarray(pitch)
        good = np.isfinite(m) & (m > 0) & np.isfinite(xy).all(axis=1)
        v = np.log(m[good])
        return xy[good], v-v.mean() if v.size else v
    a, u = data(first)
    b, v = data(second)
    total, counts = np.zeros(edges.size-1), np.zeros(edges.size-1)
    tree = cKDTree(b)
    seen = 0
    for i, point in enumerate(a):
        if cancelled is not None and cancelled():
            raise InterruptedError("correlation cancelled")
        ids = tree.query_ball_point(point, edges[-1])
        seen += len(ids)
        if seen > max_pairs:
            raise ValueError("Pair budget exceeded; reduce maximum separation or the scale-pair selection")
        if not ids:
            continue
        delta = b[ids]-point
        t, n = _bin_lags(delta[:, 0], delta[:, 1], u[i]*v[ids], np.ones(len(ids)), edges,
                          pitch=(1, 1), direction=direction, half_width=half_width)
        total += t
        counts += n
    return _result(total, counts, edges, min_pairs)


def cascade_curve(r, variance, integral_scale):
    """Finite-size W-cascade comparison, valid only for 0 < r < L."""
    r = np.asarray(r, float)
    if variance < 0 or integral_scale <= 0:
        raise ValueError("nonnegative variance and positive integral scale required")
    good = (r > 0) & (r < integral_scale)
    out = np.full(r.shape, np.nan)
    out[good] = variance*(np.log2(integral_scale/r[good])-2+2*r[good]/integral_scale)
    return out
