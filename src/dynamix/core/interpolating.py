# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Interpolating subdivision and an independent biorthogonal lifting transform.

DD uses Deslauriers–Dubuc's classical four-point predictor. GLDAF uses the
Gaussian–Lagrange shell of US6847737 (47–48), restricted to its polynomial
stencil and normalized at midpoints. The update is one half the predictor's
transpose, a local lifting design. Neither bank is an orthogonal DWT bank.
"""
from __future__ import annotations

import numpy as np


def predictor(bank="dd4", order=3, sigma=1.0):
    if bank == "dd4":
        return np.arange(-1, 3), np.array([-1., 9., 9., -1.]) / 16
    if bank != "gldaf":
        raise ValueError("interpolating bank must be dd4 or gldaf")
    M, s = int(order), float(sigma)
    if M < 1 or M > 15 or M % 2 != 1 or not np.isfinite(s) or s <= 0:
        raise ValueError("GLDAF needs odd shell order 1..15 and positive finite sigma")
    offsets = np.arange(-M + 1, M + 1)
    x = offsets - .5
    shell = np.prod(1 - (x[:, None] / np.arange(1, M + 1)) ** 2, axis=1)
    w = np.exp(-.5 * (x / s) ** 2) * shell
    if abs(w.sum()) < 1e-12:
        raise ValueError("GLDAF midpoint stencil has vanishing DC response")
    return offsets, w / w.sum()


def _indices(k, n, edge):
    if edge == "periodic":
        return k % n
    if edge != "reflect":
        raise ValueError("boundary must be reflect or periodic")
    if n == 1:
        return np.zeros_like(k)
    z = k % (2 * n - 2)
    return np.minimum(z, 2 * n - 2 - z)


def _predict(even, count, offsets, weights, edge):
    k = np.arange(count)
    out = np.zeros(even.shape[:-1] + (count,))
    for offset, w in zip(offsets, weights):
        out += w * even[..., _indices(k + offset, even.shape[-1], edge)]
    return out


def _update(detail, n, offsets, weights, edge):
    out = np.zeros(detail.shape[:-1] + (n,))
    rows = out.reshape(-1, n)
    d = detail.reshape(-1, detail.shape[-1])
    k = np.arange(detail.shape[-1])
    for offset, w in zip(offsets, weights):
        idx = _indices(k + offset, n, edge)
        np.add.at(rows, (np.arange(rows.shape[0])[:, None], idx[None, :]), .5 * w * d)
    return out


def split(values, axis=-1, *, bank="dd4", order=3, sigma=1., edge="reflect"):
    x = np.moveaxis(np.asarray(values, dtype=np.float64), axis, -1)
    if x.shape[-1] < 2:
        raise ValueError("lifting axis needs at least two samples")
    off, w = predictor(bank, order, sigma)
    even, odd = x[..., ::2], x[..., 1::2]
    detail = odd - _predict(even, odd.shape[-1], off, w, edge)
    low = even + _update(detail, even.shape[-1], off, w, edge)
    return np.moveaxis(low, -1, axis), np.moveaxis(detail, -1, axis)


def merge(low, detail, axis=-1, *, bank="dd4", order=3, sigma=1., edge="reflect"):
    lo = np.moveaxis(np.asarray(low, dtype=np.float64), axis, -1)
    d = np.moveaxis(np.asarray(detail, dtype=np.float64), axis, -1)
    off, w = predictor(bank, order, sigma)
    even = lo - _update(d, lo.shape[-1], off, w, edge)
    odd = d + _predict(even, d.shape[-1], off, w, edge)
    out = np.empty(lo.shape[:-1] + (lo.shape[-1] + d.shape[-1],))
    out[..., ::2], out[..., 1::2] = even, odd
    return np.moveaxis(out, -1, axis)


def pyramid(values, levels=3, **settings):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("interpolating wavelets need a finite 2D band; fill nodata explicitly first")
    if int(levels) < 1 or min(x.shape) < 2 ** int(levels):
        raise ValueError("each lifting level needs at least two samples on both axes")
    bands = []
    for _ in range(int(levels)):
        lx, hx = split(x, axis=1, **settings)
        x, vertical = split(lx, axis=0, **settings)
        horizontal, diagonal = split(hx, axis=0, **settings)
        bands.append((horizontal, vertical, diagonal))
    return {"low": x, "bands": bands, "settings": dict(settings)}


def inverse(coef, *, threshold=0., keep=None):
    """Reconstruct; ``keep=(level,band)`` isolates one component (levels start at 1)."""
    t = float(threshold)
    if not np.isfinite(t) or t < 0:
        raise ValueError("threshold must be finite and nonnegative")
    x = coef["low"].copy() if keep is None or keep == (len(coef["bands"]), "low") else np.zeros_like(coef["low"])
    for j in range(len(coef["bands"]), 0, -1):
        selected = []
        for name, band in zip(("horizontal", "vertical", "diagonal"), coef["bands"][j - 1]):
            b = np.sign(band) * np.maximum(np.abs(band) - t, 0)
            selected.append(b if keep is None or keep == (j, name) else np.zeros_like(b))
        h, v, d = selected
        lx = merge(x, v, axis=0, **coef["settings"])
        hx = merge(h, d, axis=0, **coef["settings"])
        x = merge(lx, hx, axis=1, **coef["settings"])
    return x


def refine(values, levels=1, *, edge="reflect"):
    """Classical DD4 refinement over the existing closed domain, with old nodes retained."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim not in (1, 2) or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("DD interpolation needs finite 1D or 2D samples, at least two per axis")
    if int(levels) != levels or not 1 <= levels <= 8:
        raise ValueError("refinement levels must be 1..8")
    shape = tuple((n - 1) * 2 ** int(levels) + 1 for n in x.shape)
    if np.prod(shape, dtype=np.float64) > 32_000_000:
        raise ValueError("refined grid exceeds 32 million samples; reduce levels or crop the source")
    off, w = predictor()
    for _ in range(int(levels)):
        for axis in range(x.ndim):
            a = np.moveaxis(x, axis, -1)
            out = np.empty(a.shape[:-1] + (2 * a.shape[-1] - 1,))
            out[..., ::2] = a
            out[..., 1::2] = _predict(a, a.shape[-1] - 1, off, w, edge)
            x = np.moveaxis(out, -1, axis)
    return x
