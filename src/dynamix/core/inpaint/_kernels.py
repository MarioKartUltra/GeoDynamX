# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The inpainting engine's inner loops, ``njit(cache=True)``-compiled on first use.

numba is required: a plain-Python scan of a 4096-pixel pyramid would run for minutes, so the
engine refuses to run without it (the ``mz_lastwave`` rule) instead of running slowly. Every
kernel writes into a preallocated output; those that select return a count.
"""
from __future__ import annotations

import math

import numpy as np

_NEEDS_NUMBA = "the inpainting engine needs numba; install it with pip install numba"
_KERNELS = None


def kernels() -> dict:
    """The compiled kernels, built once per process."""
    global _KERNELS
    if _KERNELS is None:
        try:
            import numba
        except ImportError as exc:
            raise RuntimeError(_NEEDS_NUMBA) from exc
        _KERNELS = _build(numba)
    return _KERNELS


def _build(numba) -> dict:
    njit = numba.njit

    @njit(cache=True)
    def block_mean(x, f, out):
        ny, nx = out.shape
        for i in range(ny):
            for j in range(nx):
                s = 0.0
                for a in range(f):
                    for b in range(f):
                        s += x[i * f + a, j * f + b]
                out[i, j] = s / (f * f)

    @njit(cache=True)
    def block_spread(r, f, out):
        ny, nx = r.shape
        for i in range(ny):
            for j in range(nx):
                v = r[i, j] / (f * f)
                for a in range(f):
                    for b in range(f):
                        out[i * f + a, j * f + b] = v

    @njit(cache=True)
    def bilinear_sample(x, rows, cols, out):
        ny, nx = x.shape
        for n in range(rows.shape[0]):
            r, c = rows[n], cols[n]
            if not (r >= 0.0 and r <= ny - 1.0 and c >= 0.0 and c <= nx - 1.0):
                out[n] = np.nan
                continue
            i0, j0 = int(math.floor(r)), int(math.floor(c))
            i1, j1 = min(i0 + 1, ny - 1), min(j0 + 1, nx - 1)
            fr, fc = r - i0, c - j0
            out[n] = ((1.0 - fr) * (1.0 - fc) * x[i0, j0] + (1.0 - fr) * fc * x[i0, j1]
                      + fr * (1.0 - fc) * x[i1, j0] + fr * fc * x[i1, j1])

    @njit(cache=True)
    def bilinear_splat(r, rows, cols, out):
        ny, nx = out.shape
        for n in range(rows.shape[0]):
            rr, cc = rows[n], cols[n]
            if not (rr >= 0.0 and rr <= ny - 1.0 and cc >= 0.0 and cc <= nx - 1.0):
                continue
            v = r[n]
            if v != v:
                continue
            i0, j0 = int(math.floor(rr)), int(math.floor(cc))
            i1, j1 = min(i0 + 1, ny - 1), min(j0 + 1, nx - 1)
            fr, fc = rr - i0, cc - j0
            out[i0, j0] += (1.0 - fr) * (1.0 - fc) * v
            out[i0, j1] += (1.0 - fr) * fc * v
            out[i1, j0] += fr * (1.0 - fc) * v
            out[i1, j1] += fr * fc * v

    @njit(cache=True)
    def integral(mask, out):
        """``out[i + 1, j + 1]`` = the count of set pixels in ``mask[:i + 1, :j + 1]``."""
        ny, nx = mask.shape
        for i in range(ny):
            row = 0
            for j in range(nx):
                row += 1 if mask[i, j] else 0
                out[i + 1, j + 1] = out[i, j + 1] + row

    @njit(cache=True)
    def window_any(integ, half, step, offset, out):
        """``out[i, j]`` = 1 when any set pixel lies in the window
        ``[step i + offset - half, step i + offset + step + half)`` on each axis, clamped."""
        ny, nx = integ.shape[0] - 1, integ.shape[1] - 1
        for i in range(out.shape[0]):
            r0 = max(0, step * i + offset - half)
            r1 = min(ny, step * i + offset + step + half)
            for j in range(out.shape[1]):
                c0 = max(0, step * j + offset - half)
                c1 = min(nx, step * j + offset + step + half)
                if r1 <= r0 or c1 <= c0:
                    out[i, j] = 0
                    continue
                s = integ[r1, c1] - integ[r0, c1] - integ[r1, c0] + integ[r0, c0]
                out[i, j] = 1 if s > 0 else 0

    @njit(cache=True)
    def clear_pairs(parent, child, clear_p, clear_c, floor_p, floor_c, out):
        """``ln|c| - ln|p|`` for every clear parent-child pair whose magnitudes reach the
        floors (an exact cascade's cancelled coefficients are 1e-17, not 0); returns the count
        written into ``out``."""
        n = 0
        ny, nx = parent.shape
        for k in range(ny):
            for l in range(nx):
                p = parent[k, l]
                if clear_p[k, l] == 0 or abs(p) < floor_p or p == 0.0:
                    continue
                lp = math.log(abs(p))
                for a in range(2):
                    for b in range(2):
                        c = child[2 * k + a, 2 * l + b]
                        if clear_c[2 * k + a, 2 * l + b] == 0 or abs(c) < floor_c or c == 0.0:
                            continue
                        out[n] = math.log(abs(c)) - lp
                        n += 1
        return n

    @njit(cache=True)
    def _log_density(log_hist, edges, o, v):
        n = log_hist.shape[1]
        dv = edges[1] - edges[0]
        t = (v - edges[0]) / dv - 0.5
        if t <= 0.0:
            return log_hist[o, 0]
        if t >= n - 1:
            return log_hist[o, n - 1]
        i = int(t)
        f = t - i
        return (1.0 - f) * log_hist[o, i] + f * log_hist[o, i + 1]

    @njit(cache=True)
    def cascade_level(parent, child, rho, log_hist, edges, lo, hi, cmax, n_theta):
        """Pottier et al. 2008 eqs. 13-16 for one level: every parent ``(o, k, l)`` inferred by
        maximum likelihood over its children ``(2k, 2k+1) x (2l, 2l+1)`` of the (already
        shift-aligned) child band and blended by ``rho[k, l]``; the parent's sign is kept (its
        children's sum decides when it is 0). A parent whose eq. 15 interval is empty is left
        alone: scanning the hull instead inflates the parents level after level."""
        n_o, ny, nx = parent.shape
        lnc = np.empty(4)
        for o in range(n_o):
            floor = 1e-8 * cmax[o]
            for k in range(ny):
                for l in range(nx):
                    r = rho[k, l]
                    if r <= 0.0:
                        continue
                    m = 0
                    csum = 0.0
                    for a in range(2):
                        for b in range(2):
                            c = child[o, 2 * k + a, 2 * l + b]
                            csum += c
                            if abs(c) >= floor and c != 0.0:
                                lnc[m] = math.log(abs(c))
                                m += 1
                    if m < 2:
                        continue
                    mx, mn = lnc[0], lnc[0]
                    for i in range(1, m):
                        mx = max(mx, lnc[i])
                        mn = min(mn, lnc[i])
                    t0, t1 = mx - hi[o], mn - lo[o]
                    if t1 <= t0:
                        continue                            # no theta fits every child: leave it
                    best, theta0 = -1e300, t0
                    for s in range(n_theta):
                        th = t0 + (t1 - t0) * s / max(n_theta - 1, 1)
                        d = 0.0
                        for i in range(m):
                            d += _log_density(log_hist, edges, o, lnc[i] - th)
                        if d > best:
                            best, theta0 = d, th
                    p0 = parent[o, k, l]
                    sign = 1.0 if p0 > 0.0 else (-1.0 if p0 < 0.0 else (1.0 if csum >= 0.0 else -1.0))
                    parent[o, k, l] = (1.0 - r) * p0 + r * sign * math.exp(theta0)

    @njit(cache=True)
    def synth_level(parent, child, rho, cdf, edges, sign_freq, u1, u2):
        """Beyond the paper: every child with ``rho[o] > 0.9`` under a nonzero parent is drawn
        as ``|c| = |p| exp(eta)``, ``eta`` from the abacus's CDF by inverse sampling (``u1``)
        and the parent's sign kept with probability ``sign_freq`` (``u2``). ``rho`` is per
        orientation ``(3, ny, nx)``, rolled like the child band."""
        n_o, ny, nx = child.shape
        nb = cdf.shape[1]
        for o in range(n_o):
            for i in range(ny):
                for j in range(nx):
                    if rho[o, i, j] <= 0.9:
                        continue
                    p = parent[o, i // 2, j // 2]
                    if p == 0.0:
                        continue
                    u = u1[o, i, j]
                    b = 0
                    while b < nb - 1 and cdf[o, b] < u:
                        b += 1
                    eta = 0.5 * (edges[b] + edges[b + 1])
                    s = 1.0 if u2[o, i, j] < sign_freq[o] else -1.0
                    child[o, i, j] = s * (1.0 if p > 0.0 else -1.0) * abs(p) * math.exp(eta)

    @njit(cache=True)
    def heldout_blocks(valid, candidates, side, margin, n, out):
        """Up to ``n`` ``side x side`` blocks at the first candidate top-left corners (flat
        indices, in the caller's random order) whose block plus ``margin`` is valid and clear
        of every block already taken; ``out`` marks the blocks; returns how many."""
        ny, nx = valid.shape
        taken = 0
        for idx in range(candidates.shape[0]):
            if taken >= n:
                break
            r = candidates[idx] // nx
            c = candidates[idx] % nx
            if r + side > ny or c + side > nx:
                continue
            ok = True
            for i in range(max(0, r - margin), min(ny, r + side + margin)):
                for j in range(max(0, c - margin), min(nx, c + side + margin)):
                    if valid[i, j] == 0 or out[i, j] != 0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            for i in range(r, r + side):
                for j in range(c, c + side):
                    out[i, j] = 1
            taken += 1
        return taken

    @njit(cache=True)
    def fill_polygon(points, out, value):
        """Even-odd rasterisation of the closed polygon ``points`` (``(col, row)`` vertices)
        into ``out`` at pixel centres."""
        ny, nx = out.shape
        n = points.shape[0]
        for i in range(ny):
            y = float(i)
            for j in range(nx):
                x = float(j)
                inside = False
                b = n - 1
                for a in range(n):
                    xa, ya = points[a, 0], points[a, 1]
                    xb, yb = points[b, 0], points[b, 1]
                    if (ya > y) != (yb > y):
                        xc = xa + (y - ya) * (xb - xa) / (yb - ya)
                        if x < xc:
                            inside = not inside
                    b = a
                if inside:
                    out[i, j] = value

    @njit(cache=True)
    def mask_edges(mask, xs, ys):
        """The pixel-edge segments where ``mask`` changes, as NaN-separated polylines in
        pixel-centre coordinates (``x = col - 0.5`` lines, ``y = row - 0.5`` lines); returns
        the number of values written (three per segment)."""
        ny, nx = mask.shape
        n = 0
        for i in range(ny + 1):
            for j in range(nx):
                above = mask[i - 1, j] if i > 0 else False
                below = mask[i, j] if i < ny else False
                if above != below:
                    xs[n], ys[n] = j - 0.5, i - 0.5
                    xs[n + 1], ys[n + 1] = j + 0.5, i - 0.5
                    xs[n + 2], ys[n + 2] = np.nan, np.nan
                    n += 3
        for j in range(nx + 1):
            for i in range(ny):
                left = mask[i, j - 1] if j > 0 else False
                right = mask[i, j] if j < nx else False
                if left != right:
                    xs[n], ys[n] = j - 0.5, i - 0.5
                    xs[n + 1], ys[n + 1] = j - 0.5, i + 0.5
                    xs[n + 2], ys[n + 2] = np.nan, np.nan
                    n += 3
        return n

    return {"block_mean": block_mean, "block_spread": block_spread,
            "bilinear_sample": bilinear_sample, "bilinear_splat": bilinear_splat,
            "integral": integral, "window_any": window_any, "clear_pairs": clear_pairs,
            "cascade_level": cascade_level, "synth_level": synth_level,
            "heldout_blocks": heldout_blocks, "fill_polygon": fill_polygon,
            "mask_edges": mask_edges}
