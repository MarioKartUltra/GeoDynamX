# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Measurement operators and terms of the inpainting cost
``F(x) = 1/2 sum_k || W_k^(1/2) (A_k x - d_k) ||^2``.

Each operator maps the working image (or a crop of it) onto a data grid and has an exact
adjoint; a sample that falls outside the input is ``NaN`` and the term zeroes it, so a guide
that overlaps the target only in part contributes where it exists. The step size is ``1/L``
with ``L`` measured by power iteration, because a ``Warp`` can exceed norm 1 where its mapping
compresses.
"""
from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["BlockAverage", "Compose", "Detail", "Gain", "Identity", "Term", "Warp", "Window",
           "data_cost", "estimate_offset", "gradient", "lipschitz", "parallax_warp", "shift"]


class Identity:
    def apply(self, x):
        return np.asarray(x, dtype=np.float64)

    def adjoint(self, r):
        return np.asarray(r, dtype=np.float64)


class Window:
    """Rows ``[r0, r0 + h)`` and columns ``[c0, c0 + w)`` of the input; the adjoint zero-pads."""

    def __init__(self, r0: int, c0: int, h: int, w: int, shape_in):
        self.r0, self.c0, self.h, self.w = int(r0), int(c0), int(h), int(w)
        self.shape_in = tuple(int(n) for n in shape_in)

    def apply(self, x):
        return np.asarray(x, dtype=np.float64)[self.r0:self.r0 + self.h, self.c0:self.c0 + self.w]

    def adjoint(self, r):
        out = np.zeros(self.shape_in)
        out[self.r0:self.r0 + self.h, self.c0:self.c0 + self.w] = r
        return out


class BlockAverage:
    """The mean over ``f x f`` blocks; the input's sides are multiples of ``f``."""

    def __init__(self, f: int):
        self.f = int(f)

    def apply(self, x):
        from dynamix.core.inpaint._kernels import kernels

        x = np.ascontiguousarray(x, dtype=np.float64)
        out = np.empty((x.shape[0] // self.f, x.shape[1] // self.f))
        kernels()["block_mean"](x, self.f, out)
        return out

    def adjoint(self, r):
        from dynamix.core.inpaint._kernels import kernels

        r = np.ascontiguousarray(r, dtype=np.float64)
        out = np.empty((r.shape[0] * self.f, r.shape[1] * self.f))
        kernels()["block_spread"](r, self.f, out)
        return out


class Warp:
    """Bilinear samples of the input at ``(rows, cols)`` (input pixel coordinates, one pair
    per output pixel); the adjoint is the transposed splat."""

    def __init__(self, rows, cols, shape_in):
        self.rows = np.ascontiguousarray(rows, dtype=np.float64)
        self.cols = np.ascontiguousarray(cols, dtype=np.float64)
        self.shape_in = tuple(int(n) for n in shape_in)

    def apply(self, x):
        from dynamix.core.inpaint._kernels import kernels

        out = np.empty(self.rows.size)
        kernels()["bilinear_sample"](np.ascontiguousarray(x, dtype=np.float64),
                                     self.rows.ravel(), self.cols.ravel(), out)
        return out.reshape(self.rows.shape)

    def adjoint(self, r):
        from dynamix.core.inpaint._kernels import kernels

        out = np.zeros(self.shape_in)
        kernels()["bilinear_splat"](np.ascontiguousarray(r, dtype=np.float64).ravel(),
                                    self.rows.ravel(), self.cols.ravel(), out)
        return out


def shift(dy: float, dx: float, shape_out, shape_in) -> Warp:
    """Output pixel ``(i, j)`` reads input ``(i + dy, j + dx)``: integer or sub-pixel."""
    i, j = np.indices(tuple(int(n) for n in shape_out), dtype=np.float64)
    return Warp(i + float(dy), j + float(dx), shape_in)


class Detail:
    """The detail bands of the stationary transform, stacked ``(3 levels, ny, nx)`` coarsest
    level first and horizontal, vertical, diagonal within a level; the adjoint is ``iswt2``
    with a zero approximation. A tight frame: norm 1."""

    def __init__(self, name: str, levels: int):
        self.name, self.levels = str(name), int(levels)

    def level_of(self, band: int) -> int:
        return self.levels - int(band) // 3

    def orientation_of(self, band: int) -> int:
        return int(band) % 3

    def apply(self, x):
        from dynamix.core.dwt import stationary

        s = stationary(x, self.name, self.levels)
        return np.stack([d for lev in s[1:] for d in lev])

    def adjoint(self, r):
        from dynamix.core.dwt import stationary_inverse

        r = np.asarray(r, dtype=np.float64)
        coeffs = [np.zeros(r.shape[1:])] + [tuple(r[3 * i:3 * i + 3]) for i in range(self.levels)]
        return stationary_inverse(coeffs, self.name)


class Gain:
    def __init__(self, s):
        self.s = np.asarray(s, dtype=np.float64)

    def apply(self, x):
        return self.s * x

    def adjoint(self, r):
        return self.s * r


class Compose:
    """``outer(inner(x))``."""

    def __init__(self, outer, inner):
        self.outer, self.inner = outer, inner

    def apply(self, x):
        return self.outer.apply(self.inner.apply(x))

    def adjoint(self, r):
        return self.inner.adjoint(self.outer.adjoint(r))


@dataclasses.dataclass(frozen=True)
class Term:
    """``(A, d, W)``: ``W`` on ``d``'s grid, non-negative; non-finite data or samples weigh 0."""

    op: object
    data: np.ndarray
    weights: np.ndarray

    def residual(self, x) -> np.ndarray:
        ax = self.op.apply(x)
        r = ax - self.data
        ok = np.isfinite(r) & (self.weights > 0)
        return np.where(ok, r, 0.0)


def gradient(terms, x) -> np.ndarray:
    out = None
    for t in terms:
        g = t.op.adjoint(np.where(np.isfinite(t.weights), t.weights, 0.0) * t.residual(x))
        out = g if out is None else out + g
    return np.zeros(np.shape(x)) if out is None else out


def data_cost(terms, x) -> float:
    return 0.5 * float(sum((np.where(np.isfinite(t.weights), t.weights, 0.0)
                            * t.residual(x) ** 2).sum() for t in terms))


def lipschitz(terms, shape, rng, n_iter: int = 20) -> float:
    """The largest eigenvalue of ``sum_k A_k^T W_k A_k`` by power iteration, times 1.05."""
    zeros = [dataclasses.replace(t, data=np.zeros_like(t.data)) for t in terms]
    v = rng.standard_normal(tuple(int(n) for n in shape))
    v /= np.linalg.norm(v)
    L = 1.0
    for _ in range(int(n_iter)):
        hv = gradient(zeros, v)
        L = float(np.linalg.norm(hv))
        if L == 0.0:
            return 1.0
        v = hv / L
    return 1.05 * L


def estimate_offset(a, b, max_shift: int = 8) -> tuple:
    """``(dy, dx)`` such that ``b[i, j] ~ a[i + dy, j + dx]`` on the finite overlap: the
    normalised correlation over every integer shift within ``max_shift`` (a periodic FFT
    correlation of a non-periodic field peaks at zero shift), refined by a parabola."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ny, nx = a.shape
    m = int(max_shift)
    score = np.full((2 * m + 1, 2 * m + 1), -np.inf)
    for dy in range(-m, m + 1):
        i0, i1 = max(0, -dy), min(ny, ny - dy)
        for dx in range(-m, m + 1):
            j0, j1 = max(0, -dx), min(nx, nx - dx)
            if i1 <= i0 or j1 <= j0:
                continue
            bb = b[i0:i1, j0:j1]
            aa = a[i0 + dy:i1 + dy, j0 + dx:j1 + dx]
            ok = np.isfinite(aa) & np.isfinite(bb)
            if ok.sum() < 4:
                continue
            u, v = aa[ok] - aa[ok].mean(), bb[ok] - bb[ok].mean()
            den = np.sqrt((u * u).sum() * (v * v).sum())
            if den > 0:
                score[dy + m, dx + m] = (u * v).sum() / den
    k, l = np.unravel_index(int(np.argmax(score)), score.shape)

    def refine(y0, y1, y2):
        den = y0 - 2.0 * y1 + y2
        return 0.0 if not np.isfinite(den) or den == 0.0 else 0.5 * (y0 - y2) / den

    dy = k - m + (refine(score[k - 1, l], score[k, l], score[k + 1, l]) if 0 < k < 2 * m else 0.0)
    dx = l - m + (refine(score[k, l - 1], score[k, l], score[k, l + 1]) if 0 < l < 2 * m else 0.0)
    return float(dy), float(dx)


def parallax_warp(dem, look_deg: float, pixel: float, shape) -> Warp:
    """ASTER 3B onto 3N: the along-track (row) displacement ``dem tan(look) / pixel``."""
    disp = np.asarray(dem, dtype=np.float64) * np.tan(np.radians(float(look_deg))) / float(pixel)
    i, j = np.indices(tuple(int(n) for n in shape), dtype=np.float64)
    return Warp(i + np.where(np.isfinite(disp), disp, 0.0), j, shape)
