# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Wavelet leaders on the continuous wavelet transform (spec 30, §3.1 and §3.3).

The leader at position x and scale a_j is the supremum of the lifted CWT modulus M over the
scale-space cylinder: every finer scale a_i ≤ a_j, every position within the disc |x' − x|² ≤ R_j².
That is Jaffard's two-microlocal bound taken as a quantity (Jaffard 1994, Lemma 1): the sup over
the cylinder of a C^h point is O(a^h), and the disc, not a cone, is what makes it true for
oscillating singularities (Meyer 1998, Theorem 1.7). Computed as the running maximum over the
finer scales and then one exact disc dilation per scale.

The disc maximum is decomposed into chords. For a row offset dy the chord half-width is
w = ⌊√(R² − dy²)⌋ and the maximum over the chord is a running maximum of window 2w + 1 (van Herk
and Gil–Werman: a prefix pass and a suffix pass over blocks of the window's length, whatever w),
so a scale costs pixels × 2R. The same decomposition with prefix sums gives the disc sum of the
p-leaders. The kernels are numba, compiled lazily with ``cache=True`` (the ``_line_kernels``
idiom of ``wtmm_backend``); the pure-Python bodies stand in when numba is absent.
"""
from __future__ import annotations

import math

import numpy as np

_KERNELS: dict = {}
_NEG = -np.inf


def _disc_max_py(src, R, out):
    """``out[y, x]`` = max of ``src`` over |Δ|² ≤ R² around (y, x), rows outside the image left
    out. ``src`` holds −inf where a pixel is missing; ``out`` arrives filled with −inf."""
    ny, nx = src.shape
    r = int(math.floor(R))
    for dy in range(-r, r + 1):
        w = int(math.floor(math.sqrt(R * R - dy * dy)))
        L = 2 * w + 1
        n = nx + 2 * w
        pad = np.empty(n, np.float32)
        g = np.empty(n, np.float32)
        h = np.empty(n, np.float32)
        for y in range(ny):
            ys = y + dy
            if ys < 0 or ys >= ny:
                continue
            for i in range(w):
                pad[i] = _NEG
                pad[n - 1 - i] = _NEG
            for i in range(nx):
                pad[w + i] = src[ys, i]
            for i in range(n):                       # prefix maximum within each block of L
                if i % L == 0 or pad[i] > g[i - 1]:
                    g[i] = pad[i]
                else:
                    g[i] = g[i - 1]
            for i in range(n - 1, -1, -1):           # suffix maximum within each block of L
                if i % L == L - 1 or i == n - 1 or pad[i] > h[i + 1]:
                    h[i] = pad[i]
                else:
                    h[i] = h[i + 1]
            for x in range(nx):                      # the window [x, x + L − 1] of the padded row
                e = x + L - 1
                v = g[e] if g[e] > h[x] else h[x]
                if v > out[y, x]:
                    out[y, x] = v


def _disc_sum_py(src, finite, R, total, count):
    """``total[y, x]`` = sum of ``src`` and ``count[y, x]`` = number of finite pixels over the disc
    |Δ|² ≤ R² around (y, x); ``src`` holds 0 where a pixel is missing and ``finite`` 1 or 0."""
    ny, nx = src.shape
    r = int(math.floor(R))
    for dy in range(-r, r + 1):
        w = int(math.floor(math.sqrt(R * R - dy * dy)))
        L = 2 * w + 1
        n = nx + 2 * w
        cs = np.zeros(n + 1, np.float64)
        cc = np.zeros(n + 1, np.int64)
        for y in range(ny):
            ys = y + dy
            if ys < 0 or ys >= ny:
                continue
            for i in range(n):
                inside = w <= i < w + nx
                cs[i + 1] = cs[i] + (src[ys, i - w] if inside else 0.0)
                cc[i + 1] = cc[i] + (finite[ys, i - w] if inside else 0)
            for x in range(nx):
                total[y, x] += cs[x + L] - cs[x]
                count[y, x] += cc[x + L] - cc[x]


def _kernels() -> dict:
    """Lazily njit(cache=True)-compile the disc kernels; the Python bodies when numba is absent."""
    if not _KERNELS:
        try:
            from numba import njit
        except Exception:                                   # pragma: no cover
            _KERNELS["max"] = _disc_max_py
            _KERNELS["sum"] = _disc_sum_py
        else:
            _KERNELS["max"] = njit(cache=True)(_disc_max_py)
            _KERNELS["sum"] = njit(cache=True)(_disc_sum_py)
    return _KERNELS


def disc_dilate(src, R: float) -> np.ndarray:
    """The maximum of ``src`` over the disc of radius ``R`` (|Δ|² ≤ R², a float radius) around
    every pixel, float32; a non-finite pixel is left out, and a pixel whose disc holds no finite
    value is NaN. Rows and columns beyond the image are not mirrored: the validity mask of the
    caller decides where a leader may be read."""
    src = np.asarray(src, dtype=np.float32)
    filled = np.where(np.isfinite(src), src, np.float32(_NEG)).astype(np.float32)
    out = np.full(src.shape, _NEG, dtype=np.float32)
    _kernels()["max"](filled, float(R), out)
    out[~np.isfinite(out)] = np.nan
    return out


def disc_boxsum(src, R: float) -> tuple[np.ndarray, np.ndarray]:
    """``(sum, count)`` of the finite pixels of ``src`` over the disc of radius ``R`` around every
    pixel: the sum float32, the count int64 (0 where the disc holds no finite pixel)."""
    src = np.asarray(src, dtype=np.float32)
    finite = np.isfinite(src)
    total = np.zeros(src.shape, dtype=np.float64)
    count = np.zeros(src.shape, dtype=np.int64)
    _kernels()["sum"](np.where(finite, src, 0.0).astype(np.float64), finite.astype(np.int64),
                      float(R), total, count)
    return total.astype(np.float32), count


#: Measured energy radii per (smoother, operator, the smoother's own knobs, energy fraction).
_RADII: dict = {}
_SMOOTHER_KNOBS = ("alpha", "q", "beta")


def energy_radius(wavelet: str, operator: str, params: dict | None, energy: float = 0.99) -> float:
    """``c`` such that the disc of radius ``c × scale`` holds the fraction ``energy`` of ∫|ψ|² for
    the analysing kernel of the smoother ``wavelet`` under ``operator``: ``"gradient"``, the
    vector kernel ∇θ whose modulus the WTMM stage takes, or ``"none"``, θ itself. Measured once
    from the backend's own Fourier kernels (``_build_wavelet_filters_numpy``'s convention:
    wavenumbers in cycles per pixel times the stage's scale) on a 1024-square grid at a reference
    scale, inverse-transformed by the FFT policy's engine, and cached. Spec 30 §2: the constant is
    tied to the kernel, not re-measured on the pixel grid at every scale."""
    knobs = tuple(sorted((k, float(v)) for k, v in (params or {}).items() if k in _SMOOTHER_KNOBS))
    key = (wavelet, operator, knobs, float(energy))
    if key in _RADII:
        return _RADII[key]
    if operator not in ("gradient", "none"):
        raise ValueError(f"unknown operator {operator!r}; expected 'gradient' or 'none'")
    from dynamix.core import fft_policy
    from dynamix.core.smoothers import theta_hat
    from dynamix.core.wtmm_backend import _build_wavelet_filters_numpy

    n, ref = 1024, 64.0                              # σ ≈ 14 px: resolved, and far from the edge
    f = np.fft.fftfreq(n)
    kx, ky = np.meshgrid(f, f)
    fft = fft_policy.active()
    if operator == "gradient":
        filters = _build_wavelet_filters_numpy(kx, ky, ref, wavelet=wavelet, params=params,
                                               dtype=np.float64)
        density = sum(np.abs(np.asarray(fft.ifft2(filters[name]))) ** 2 for name in ("dx", "dy"))
    else:
        theta = theta_hat(wavelet, dict(params or {}), kx * ref, ky * ref).astype(np.complex128)
        density = np.abs(np.asarray(fft.ifft2(theta))) ** 2
    x = f * n                                        # the kernel sits at index 0 and wraps
    r = np.hypot(*np.meshgrid(x, x)).ravel()
    order = np.argsort(r, kind="stable")
    cumulative = np.cumsum(density.ravel()[order])
    radius = float(r[order][np.searchsorted(cumulative, energy * cumulative[-1])])
    if radius > n / 4:
        raise ValueError(f"{wavelet}/{operator}: the kernel's energy radius {radius:.0f} px at scale "
                         f"{ref} exceeds the measuring grid; the smoother's support is too wide")
    _RADII[key] = radius / ref
    return _RADII[key]


def cylinder_leaders(mod, radii, sup_from: int = 0) -> np.ndarray:
    """``L[j]`` = the disc dilation, with radius ``radii[j]``, of the running maximum of ``mod[i]``
    over ``sup_from ≤ i ≤ j``: the cylinder supremum of spec 30 §3.1, scale by scale from the finest
    admitted one up. ``mod`` is ``(scales, rows, cols)``, finest scale first; a non-finite modulus
    is left out of the running maximum. Scales below ``sup_from`` are NaN (not computed)."""
    mod = np.asarray(mod, dtype=np.float32)
    n = mod.shape[0]
    if not 0 <= int(sup_from) < n:
        raise ValueError(f"sup_from {sup_from}: nothing finer to take the supremum over "
                         f"({n} scales on the ladder)")
    out = np.full(mod.shape, np.nan, dtype=np.float32)
    cum = None
    for j in range(int(sup_from), n):
        cum = mod[j].copy() if cum is None else np.fmax(cum, mod[j])
        out[j] = disc_dilate(cum, float(radii[j]))
    return out
