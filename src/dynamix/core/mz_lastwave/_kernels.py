# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""numba kernels of the LastWave ``dwtrans2d`` port, compiled on first use.

Each kernel transliterates one of the authors' C loops in the same arithmetic order, which is what
makes the port bit for bit with the C: the FIR convolutions of ``convol2.c`` (periodic, rows and
columns), the polar / cartesian conversions of ``coordinate.c``, ``W2_interp`` and the two
projection passes of ``ext2_proj.c``. They are compiled with ``fastmath=False`` and
``cache=True``; the row loops run in parallel, one row (or one block of columns) per thread, which
leaves every output value's arithmetic unchanged. numba is imported only when :func:`kernels` is
first called.
"""
from __future__ import annotations

import math
import types

import numpy as np

#: The compiled kernels, built by the first :func:`kernels` call.
_KERNELS = None

_NEEDS_NUMBA = "the M–Z (LastWave) engine needs numba; install it with pip install numba"


def kernels():
    """The compiled kernels as a namespace: ``conv_rows, conv_cols, scale_into, add_into, polar,
    cartesian, interp, proj1_rows, proj1_cols, clip_rows, clip_cols``.

    Raises ``RuntimeError`` naming numba when it is not installed.
    """
    global _KERNELS
    if _KERNELS is None:
        try:
            import numba
        except ImportError as exc:
            raise RuntimeError(_NEEDS_NUMBA) from exc
        _KERNELS = _build(numba)
    return _KERNELS


def _build(numba):
    njit = numba.njit
    prange = numba.prange
    PI = math.pi

    @njit(parallel=True, cache=True, fastmath=False)
    def conv_rows(x, out, f, sym, l1, r1, scale, size):
        """``W2_convper`` along each row (periodic):
        ``out[j] = f0·x[j] + Σ_k f_k·(sym·x[j−l1−(k−1)s] + x[j+r1+(k−1)s])``, the row read once
        into a wrapped buffer."""
        ny, n = x.shape
        P = max(l1, r1) + (size - 2) * scale
        for i in prange(ny):
            buf = np.empty(n + 2 * P)
            for j in range(n + 2 * P):
                buf[j] = x[i, (j - P) % n]
            for j in range(n):
                c = P + j
                s = f[0] * buf[c]
                left = c - l1
                right = c + r1
                for k in range(1, size):
                    s += f[k] * (sym * buf[left] + buf[right])
                    right += scale
                    left -= scale
                out[i, j] = s

    @njit(parallel=True, cache=True, fastmath=False)
    def conv_cols(x, out, f, sym, l1, r1, scale, size, mul):
        """``W2_convper`` along each column (periodic), then ``out *= mul`` when ``mul != 1``."""
        n, nx = x.shape
        for i in prange(n):
            for j in range(nx):
                out[i, j] = f[0] * x[i, j]
            left = i - l1
            right = i + r1
            for k in range(1, size):
                il = left % n
                ir = right % n
                fk = f[k]
                for j in range(nx):
                    out[i, j] += fk * (sym * x[il, j] + x[ir, j])
                right += scale
                left -= scale
            if mul != 1.0:
                for j in range(nx):
                    out[i, j] = out[i, j] * mul

    @njit(parallel=True, cache=True, fastmath=False)
    def scale_into(x, out, mul):
        ny, nx = x.shape
        for i in prange(ny):
            for j in range(nx):
                out[i, j] = x[i, j] * mul

    @njit(parallel=True, cache=True, fastmath=False)
    def add_into(a, b):
        ny, nx = a.shape
        for i in prange(ny):
            for j in range(nx):
                a[i, j] = a[i, j] + b[i, j]

    @njit(parallel=True, cache=True, fastmath=False)
    def polar(h, v, m, a):
        """``W2_magnitude`` / ``W2_argument``: ``m = sqrt(h² + v²)``, ``a = atan2(v, h)``, 0 at
        ``h = v = 0``."""
        ny, nx = h.shape
        for i in prange(ny):
            for j in range(nx):
                x = h[i, j]
                y = v[i, j]
                m[i, j] = math.sqrt(x * x + y * y)
                a[i, j] = 0.0 if (x == 0.0 and y == 0.0) else math.atan2(y, x)

    @njit(parallel=True, cache=True, fastmath=False)
    def cartesian(m, a, h, v):
        ny, nx = m.shape
        for i in prange(ny):
            for j in range(nx):
                h[i, j] = m[i, j] * math.cos(a[i, j])
                v[i, j] = m[i, j] * math.sin(a[i, j])

    @njit(cache=True, fastmath=False)
    def interp(u0, un, n, u, r1):
        """``W2_interp`` (``ext2_proj.c``): the sinh interpolation of the errors ``u0`` at 0 and
        ``un`` at ``n`` into ``u[0:n]``, the C recurrence verbatim."""
        rn_1 = r1
        for i in range(1, n - 1):
            rn_1 *= r1
        r2n_2 = rn_1 * rn_1
        r_1 = 1.0 / r1
        r_2 = r_1 * r_1
        r2 = r1 * r1
        r2n = r2n_2 * r2
        a0 = u0 / (1.0 - r2n)
        an = un / (1.0 - r2n)
        u[0] = u0
        r2n_2i = r2n_2
        r2i = r2
        ri = r1
        rn_i = rn_1
        for i in range(1, n):
            u[i] = a0 * ri * (1 - r2n_2i) + an * rn_i * (1 - r2i)
            r2n_2i *= r_2
            r2i *= r2
            ri *= r1
            rn_i *= r_1

    @njit(parallel=True, cache=True, fastmath=False)
    def proj1_rows(h, ext, tgt, a):
        """``W2_pt_level_proj_1st`` along each row, in place on ``h``: anchors at column 0 (target
        0), the extrema strictly inside, and column nx−1 (target 0); ``h`` plus the interpolated
        error between consecutive anchors; the last column set to 0."""
        ny, nx = h.shape
        for i in prange(ny):
            u = np.empty(nx)
            t0 = 0
            e0 = 0.0 - h[i, t0]
            while t0 < nx - 1:
                t1 = t0 + 1
                found = False
                while t1 < nx - 1:
                    if ext[i, t1]:
                        found = True
                        break
                    t1 += 1
                if found:
                    e1 = tgt[i, t1] - h[i, t1]
                else:
                    t1 = nx - 1
                    e1 = 0.0 - h[i, t1]
                n = t1 - t0
                interp(e0, e1, n, u, a)
                for k in range(n):
                    h[i, t0 + k] += u[k]
                t0 = t1
                e0 = e1
            h[i, nx - 1] = 0.0

    @njit(parallel=True, cache=True, fastmath=False)
    def proj1_cols(v, ext, tgt, a):
        """:func:`proj1_rows` along each column, in blocks of 16 columns per thread."""
        ny, nx = v.shape
        nb = (nx + 15) // 16
        for b in prange(nb):
            u = np.empty(ny)
            for j in range(b * 16, min(nx, b * 16 + 16)):
                t0 = 0
                e0 = 0.0 - v[t0, j]
                while t0 < ny - 1:
                    t1 = t0 + 1
                    found = False
                    while t1 < ny - 1:
                        if ext[t1, j]:
                            found = True
                            break
                        t1 += 1
                    if found:
                        e1 = tgt[t1, j] - v[t1, j]
                    else:
                        t1 = ny - 1
                        e1 = 0.0 - v[t1, j]
                    n = t1 - t0
                    interp(e0, e1, n, u, a)
                    for k in range(n):
                        v[t0 + k, j] += u[k]
                    t0 = t1
                    e0 = e1
                v[ny - 1, j] = 0.0

    @njit(parallel=True, cache=True, fastmath=False)
    def clip_rows(m, ang, ext, emag):
        """``W2_pt_level_proj_2nd`` along each interior row, in place on the modulus ``m``: between
        consecutive anchors the modulus is clipped to a running minimum from each anchor towards
        the segment minimum, while the argument stays horizontal."""
        ny, nx = m.shape
        for i in prange(1, ny - 1):
            t0 = 0
            t1 = 0
            m0 = m[i, t0]
            while t0 < nx - 1:
                m1 = 0.0
                t1 += 1
                while t1 < nx - 1:
                    if ext[i, t1]:
                        m1 = emag[i, t1]
                        break
                    t1 += 1
                if t1 == nx - 1:
                    if t0 == 0:
                        break
                    else:
                        m1 = m[i, t1]
                t_min = t0
                m_min = m[i, t0]
                for j in range(t0 + 1, t1 + 1):
                    if m[i, j] < m_min:
                        m_min = m[i, j]
                        t_min = j
                if m0 > m_min:
                    for j in range(t0 + 1, t_min):
                        if abs(abs(ang[i, j]) / PI - 0.5) >= 0.375:
                            if m[i, j] > m[i, j - 1]:
                                m[i, j] = m[i, j - 1]
                        else:
                            break
                if m1 > m_min:
                    for j in range(t1 - 1, t_min, -1):
                        if abs(abs(ang[i, j]) / PI - 0.5) >= 0.375:
                            if m[i, j] > m[i, j + 1]:
                                m[i, j] = m[i, j + 1]
                        else:
                            break
                t0 = t1
                m0 = m1

    @njit(parallel=True, cache=True, fastmath=False)
    def clip_cols(m, ang, ext, emag):
        """:func:`clip_rows` along each interior column while the argument stays vertical, in
        blocks of 16 columns per thread."""
        ny, nx = m.shape
        nb = (nx - 2 + 15) // 16
        for b in prange(nb):
            for j in range(1 + b * 16, min(nx - 1, 1 + b * 16 + 16)):
                t0 = 0
                t1 = 0
                m0 = m[t0, j]
                while t0 < ny - 1:
                    m1 = 0.0
                    t1 += 1
                    while t1 < ny - 1:
                        if ext[t1, j]:
                            m1 = emag[t1, j]
                            break
                        t1 += 1
                    if t1 == ny - 1:
                        if t0 == 0:
                            break
                        else:
                            m1 = m[t1, j]
                    t_min = t0
                    m_min = m[t0, j]
                    for i in range(t0 + 1, t1 + 1):
                        if m[i, j] < m_min:
                            m_min = m[i, j]
                            t_min = i
                    if m0 > m_min:
                        for i in range(t0 + 1, t_min):
                            if abs(abs(ang[i, j]) / PI - 0.5) <= 0.125:
                                if m[i, j] > m[i - 1, j]:
                                    m[i, j] = m[i - 1, j]
                            else:
                                break
                    if m1 > m_min:
                        for i in range(t1 - 1, t_min, -1):
                            if abs(abs(ang[i, j]) / PI - 0.5) <= 0.125:
                                if m[i, j] > m[i + 1, j]:
                                    m[i, j] = m[i + 1, j]
                            else:
                                break
                    t0 = t1
                    m0 = m1

    return types.SimpleNamespace(
        conv_rows=conv_rows, conv_cols=conv_cols, scale_into=scale_into, add_into=add_into,
        polar=polar, cartesian=cartesian, interp=interp, proj1_rows=proj1_rows,
        proj1_cols=proj1_cols, clip_rows=clip_rows, clip_cols=clip_cols)
