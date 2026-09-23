# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Subpixel refinement of NMS modulus maxima -- position AND value, along the gradient.

Written for DynamiX after the 2026-09-20 reference-implementation census;
a REIMPLEMENTATION against documented
reference semantics, never a copy (the ``chain_stats``/``spectra`` license). The provenance,
precisely:

- **LastWave 1-D** (``package_extrema1d/src/ext_compute.c``, default-on): a 3-point parabola
  through the extremum and its two neighbours refines BOTH the position and the value, stored
  ALONGSIDE the integer index, with an out-of-bounds clamp back to the grid point. That dual
  representation is what this module transplants to 2-D.
- **xsmurf** (``follow``/``eiset``, ``interpreter/wt2d_cmds.c``): interpolates the modulus at a
  subpixel offset along the gradient and keeps only the refined MODULUS -- the value it stores
  feeds the partition function, exactly as LastWave's 1-D WTMM consumes the parabola-refined
  ordinate (``pf_functions.c``). ``mod_sub`` here is that parity channel. xsmurf computes the
  offset and throws it away (``wt2d_cmds.c:5222``); we keep it, because the display staircase
  is a position problem.
- **The probes are the NMS's own.** ``wtmm_backend._nms_extrema_scale`` keeps a pixel when its
  modulus is >= the bilinearly-sampled modulus at ``p ± (cos a, sin a)``. The parabola here is
  fit through THOSE three samples (same ``map_coordinates(order=1, mode="nearest")`` call), so
  the refinement interpolates the exact quantity the detection compared -- no second sampling
  convention exists to disagree with the first.
- **Mallat-Zhong stays pixel-exact.** The papers' reconstruction numerics pin constraints at
  integer support (verified with quotes in the research doc); refined positions are a
  measurement/display channel that must NEVER migrate into ``pocs2d``'s constraint support.
  That is why ``x``/``y`` stay integer and the float channel is a separate pair of keys.

The math, per extremum at integer ``p`` with unit gradient ``u = (cos a, sin a)`` and probe
values ``m- = m(p - u)``, ``m0 = m(p)``, ``m+ = m(p + u)``:

    a2 = (m+ + m-)/2 - m0      (parabola curvature; <= 0 at a genuine maximum)
    b  = (m+ - m-)/2
    t* = -b / (2 a2)  if a2 < 0 else 0,   clamped to [-1/2, +1/2]
    p_sub   = p + t* u
    mod_sub = m0 + b*t* + a2*t*^2        (the vertex value; >= m0 inside the clamp)

An NMS survivor has ``m0 >= max(m-, m+)``, which already bounds ``|t*| <= 1/2`` (equality at a
tie with one neighbour); the clamp guards float degeneracies only. ``a2 >= 0`` (flat plateau or
non-concave probes) falls back to the grid point and value -- LastWave's own fallback.
"""
from __future__ import annotations

import numpy as np

__all__ = ["refine_scale", "refine_extrema_stack"]


def refine_scale(mod, arg, x, y):
    """Refine one scale's extrema along their gradient directions.

    Parameters
    ----------
    mod : (ny, nx) array
        Wavelet-gradient modulus raster at this scale (the SAME raster the NMS ran on --
        fracint-lifted when the pipeline lifts).
    arg : (m,) array
        Per-extremum gradient angle in radians (the ``arg`` the extrema already carry).
    x, y : (m,) int arrays
        Integer extrema positions (column, row).

    Returns
    -------
    (x_sub, y_sub, mod_sub) : float64 arrays, shape (m,)
    """
    from scipy.ndimage import map_coordinates

    m = np.asarray(mod, dtype=np.float64)
    a = np.asarray(arg, dtype=np.float64)
    xi = np.asarray(x, dtype=np.float64)
    yi = np.asarray(y, dtype=np.float64)
    if xi.size == 0:
        z = np.zeros(0, dtype=np.float64)
        return z, z.copy(), z.copy()

    cx = np.cos(a)
    sy = np.sin(a)
    # The NMS's exact probes: bilinear, mode="nearest" (_nms_extrema_scale).
    fore = map_coordinates(m, [yi + sy, xi + cx], order=1, mode="nearest")
    back = map_coordinates(m, [yi - sy, xi - cx], order=1, mode="nearest")
    m0 = m[np.asarray(y, dtype=np.int64), np.asarray(x, dtype=np.int64)]

    a2 = 0.5 * (fore + back) - m0
    b = 0.5 * (fore - back)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(a2 < 0.0, -b / (2.0 * a2), 0.0)
    t = np.clip(np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0), -0.5, 0.5)
    mod_sub = m0 + b * t + a2 * t * t
    # Never below the grid value: the vertex of a downward parabola dominates m0, but a clamped
    # t on a degenerate fit could dip under it by float dust -- honesty floor, not physics.
    mod_sub = np.maximum(mod_sub, m0)
    return xi + t * cx, yi + t * sy, mod_sub


def refine_extrema_stack(extrema: list, mod_stack) -> list:
    """Refine every scale of an ``extrema2d`` result against its modulus stack.

    Returns a NEW list of NEW dicts -- the input is never mutated (the cached extrema stage may
    own it; the ``_apply_fracint2d`` law). Each output dict keeps ``x``/``y``/``arg``/
    ``line_id`` as the same objects, REPLACES ``mod`` with the refined modulus (the value
    channel that feeds chaining and the partition function -- xsmurf/LastWave parity), and adds
    ``x_sub``/``y_sub`` float64 positions.
    """
    mod_stack = np.asarray(mod_stack)
    out = []
    for si, e in enumerate(extrema):
        x_sub, y_sub, mod_sub = refine_scale(mod_stack[si], e["arg"], e["x"], e["y"])
        refined = dict(e)
        refined["mod"] = mod_sub
        refined["x_sub"] = x_sub
        refined["y_sub"] = y_sub
        out.append(refined)
    return out
