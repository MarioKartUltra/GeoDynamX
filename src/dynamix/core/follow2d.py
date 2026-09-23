# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
# Portions translated from xsmurf, Copyright (C) 1999 Centre de Recherche Paul Pascal,
# Bordeaux, France (N. Decoster, P. Kestener, S. Roux, A. Arneodo), GPL-2.0 -- see NOTICE.
"""The xsmurf ``follow`` detector: WTMM edges as zero-crossings of kappa, ported for DynamiX.

xsmurf's historical scalar-2D default (``imStudy.tcl``: ``useNMaxSup`` defaults to 0) detects
an edge point not by comparing modulus values (NMS) but by the analytic condition: kappa = 0
with kappa' < 0, where kappa is the derivative of the (squared) modulus along the gradient
direction and kappa' its next directional derivative -- the sign gate that rejects modulus
MINIMA. The kappa/kappa' formulas here are EXACT ports of xsmurf's ``gkapa``/``gkapap``
(``interpreter/wt2d_cmds.c:2948/3022``, verified byte-identical to upstream
``pkestene/xsmurf``):

    kappa  = 2 dx^2 dxx + 4 dxy dx dy + 2 dy^2 dyy
    kappa' = dx^2 (4(dxx^2 + dxy^2) + 2 dx dxxx + 6 dy dxxy)
           + dy^2 (4(dyy^2 + dxy^2) + 2 dy dyyy + 6 dx dxyy)
           + 8 dx dy dxy (dxx + dyy)

(xsmurf's own unnormalized convention: kappa is grad(M^2) . grad(s) -- sign-equivalent to
(grad M) . u for crossings and gates, which is all detection consumes; the commented-out
normalization in the C is dropped there too.)

**Detection** (this module's discretization -- xsmurf walks the kappa = 0 level contour; we
claim, per pixel, the crossing of kappa along the unit gradient u = (cos a, sin a) nearest to
it): pixel p is an edge iff kappa changes sign between p and p +/- u (bilinear samples, the
NMS sampling convention), the crossing offset |t| <= 1/2 (each crossing claimed by the pixel
that owns it), kappa'(p) < 0, and the modulus passes the same fraction-of-max threshold the
NMS applies. The crossing offset is kept as the native subpixel channel (``x_sub``/``y_sub``
-- the schema the interpolate knob introduced), and the stored modulus is the 3-point parabola
through the bilinear modulus samples EVALUATED AT the crossing -- the follow value channel
(xsmurf default ``followVersion=1`` interpolates a cubic through 4 samples at the crossing;
the 3-point parabola is this codebase's one sampling convention, a recorded divergence).

Per the user ("at least use numba"): the kappa/kappa' combination and the per-pixel crossing
scan are single fused numba kernels (lazy ``njit(cache=True)``, plain-python fallback -- the
``_line_kernels`` idiom), so the nine derivative stacks never materialize eight temporaries;
the FFT-heavy derivative stacks themselves come from the cwt engine (mlx when available,
``derivs='all'``), never from here.
"""
from __future__ import annotations

import numpy as np

__all__ = ["kapa_fields", "follow_extrema_scale"]

_KERNELS: dict = {}


# These two are written in the numba-friendly subset of Python and compiled on first use by
# _follow_kernels() -- numba stays a lazy dependency (the wtmm_backend._line_kernels idiom).

def _kapa_kernels_py(dx, dy, dxx, dxy, dyy, dxxx, dxxy, dxyy, dyyy, kapa, kapap):
    ny, nx = dx.shape
    for i in range(ny):
        for j in range(nx):
            gx = dx[i, j]
            gy = dy[i, j]
            gxx = dxx[i, j]
            gxy = dxy[i, j]
            gyy = dyy[i, j]
            kapa[i, j] = 2.0 * gx * gx * gxx + 4.0 * gxy * gx * gy + 2.0 * gy * gy * gyy
            kapap[i, j] = (
                gx * gx * (4.0 * (gxx * gxx + gxy * gxy)
                           + 2.0 * gx * dxxx[i, j] + 6.0 * gy * dxxy[i, j])
                + gy * gy * (4.0 * (gyy * gyy + gxy * gxy)
                             + 2.0 * gy * dyyy[i, j] + 6.0 * gx * dxyy[i, j])
                + 8.0 * gx * gy * gxy * (gxx + gyy))


def _crossing_scan_py(kapa, kapap, mod, arg, thresh_abs, claimed, t_out, mod_out):
    ny, nx = kapa.shape
    for i in range(ny):
        for j in range(nx):
            if kapap[i, j] >= 0.0 or mod[i, j] < thresh_abs:
                continue
            a = arg[i, j]
            cx = np.cos(a)
            sy = np.sin(a)
            k0 = kapa[i, j]
            # bilinear sample of kapa (and mod) one step fore/back along u -- the NMS's own
            # probe convention, clamped to the grid (map_coordinates mode="nearest").
            k_f = _bilin(kapa, i + sy, j + cx)
            k_b = _bilin(kapa, i - sy, j - cx)
            t = 2.0                                     # sentinel: no claim
            if k0 == 0.0:
                t = 0.0
            elif (k0 > 0.0) != (k_f > 0.0):
                tf = k0 / (k0 - k_f)
                if tf <= 0.5:
                    t = tf
            if t == 2.0 and (k0 > 0.0) != (k_b > 0.0):
                tb = k0 / (k0 - k_b)
                if tb < 0.5:
                    t = -tb
            if t == 2.0:
                continue
            m0 = mod[i, j]
            m_f = _bilin(mod, i + sy, j + cx)
            m_b = _bilin(mod, i - sy, j - cx)
            b = 0.5 * (m_f - m_b)
            a2 = 0.5 * (m_f + m_b) - m0
            claimed[i, j] = True
            t_out[i, j] = t
            mod_out[i, j] = m0 + b * t + a2 * t * t


def _bilin_py(img, y, x):
    ny, nx = img.shape
    if y < 0.0:
        y = 0.0
    if y > ny - 1.0:
        y = ny - 1.0
    if x < 0.0:
        x = 0.0
    if x > nx - 1.0:
        x = nx - 1.0
    y0 = int(y)
    x0 = int(x)
    y1 = y0 + 1 if y0 + 1 < ny else y0
    x1 = x0 + 1 if x0 + 1 < nx else x0
    fy = y - y0
    fx = x - x0
    return ((1.0 - fy) * ((1.0 - fx) * img[y0, x0] + fx * img[y0, x1])
            + fy * ((1.0 - fx) * img[y1, x0] + fx * img[y1, x1]))


_bilin = _bilin_py                                       # rebound to the njit version on compile


def _follow_kernels() -> dict:
    """Lazily njit(cache=True)-compile the kappa and crossing kernels (plain-python fallback)."""
    global _bilin
    if not _KERNELS:
        try:
            from numba import njit
        except Exception:                                # pragma: no cover
            _KERNELS["kapa"] = _kapa_kernels_py
            _KERNELS["scan"] = _crossing_scan_py
        else:
            _bilin = njit(cache=True, inline="always")(_bilin_py)
            _KERNELS["kapa"] = njit(cache=True)(_kapa_kernels_py)
            _KERNELS["scan"] = njit(cache=True)(_crossing_scan_py)
    return _KERNELS


def kapa_fields(derivs: dict, si: int) -> tuple:
    """``(kapa, kapap)`` float32 rasters for scale index ``si`` of an engine ``derivs='all'``
    result (keys dx..dyyy, each ``(n_sc, ny, nx)``). One fused pass, no temporaries."""
    args = [np.ascontiguousarray(np.asarray(derivs[k][si], dtype=np.float32))
            for k in ("dx", "dy", "dxx", "dxy", "dyy", "dxxx", "dxxy", "dxyy", "dyyy")]
    kapa = np.empty_like(args[0])
    kapap = np.empty_like(args[0])
    _follow_kernels()["kapa"](*args, kapa, kapap)
    return kapa, kapap


def follow_extrema_scale(mod, arg, kapa, kapap, *, thresh: float = 1e-3,
                         invalid=None, radius: int = 0) -> dict:
    """Single-scale follow detection -- same output schema as ``_nms_extrema_scale`` plus the
    native ``x_sub``/``y_sub`` channels and the crossing-interpolated modulus.

    ``invalid``/``radius``: the NaN-distrust contract the NMS applies (originally-NaN mask
    dilated by ``ceil(scale)`` px; covered extrema dropped), reproduced here so the two
    detectors share the honesty rule.
    """
    from scipy.ndimage import binary_dilation, label

    m = np.ascontiguousarray(np.asarray(mod, dtype=np.float32))
    a = np.ascontiguousarray(np.asarray(arg, dtype=np.float32))
    k = np.ascontiguousarray(np.asarray(kapa, dtype=np.float32))
    kp = np.ascontiguousarray(np.asarray(kapap, dtype=np.float32))
    ny, nx = m.shape
    claimed = np.zeros((ny, nx), dtype=np.bool_)
    t_out = np.zeros((ny, nx), dtype=np.float32)
    mod_out = np.zeros((ny, nx), dtype=np.float32)
    thresh_abs = np.float32(float(thresh) * float(m.max()) if m.size else 0.0)
    _follow_kernels()["scan"](k, kp, m, a, thresh_abs, claimed, t_out, mod_out)

    if invalid is not None and invalid.any():
        dilated = (binary_dilation(invalid, structure=np.ones((3, 3)), iterations=radius)
                   if radius > 0 else invalid)
        claimed &= ~dilated

    labels, n_labels = label(claimed, structure=np.ones((3, 3)))
    y_idx, x_idx = np.nonzero(claimed)
    line_id = labels[y_idx, x_idx].astype(np.int64)
    if n_labels > 0:
        counts = np.bincount(labels.reshape(-1), minlength=n_labels + 1)
        line_id[counts[line_id] <= 1] = -1
    t = t_out[y_idx, x_idx].astype(np.float64)
    ang = a[y_idx, x_idx].astype(np.float64)
    return {
        "x": x_idx.astype(np.int64), "y": y_idx.astype(np.int64),
        "mod": mod_out[y_idx, x_idx].astype(np.float64), "arg": ang,
        "line_id": line_id,
        "x_sub": x_idx + t * np.cos(ang),
        "y_sub": y_idx + t * np.sin(ang),
    }
