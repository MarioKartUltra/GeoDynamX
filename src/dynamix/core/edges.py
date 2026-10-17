# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Edge handling for the scale-space tools: what fills a scale's reach where real data runs out.

Real parent data comes first (:mod:`dynamix.roi.halo` reads each scale's halo, or the whole parent
where a reach does not converge). Where the halo still runs past the raster, ``edge`` decides:
``"reflect"`` mirrors the data (the default; the mirror lines can carry false maxima, which the
warning says), ``"cut"`` keeps the padding out of the result by dropping the cone of influence,
``"gaussian_taper"`` rolls the real data off to their mean in a band at the raster's edges (Jones and
Gunaratne 1998, p. 446: the data tapered near the boundaries so that the field vanishes there) and
drops the band's maxima, since its values are no longer the data's, and ``"daf_periodic"`` makes the
real data periodic with the self-tuned DAF extension (:mod:`dynamix.core.daf`) and transforms it
without padding.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["CUT_SCOPES", "EDGES", "corner_warning", "cut_borders", "deficits",
           "dominant_wavelength", "drop_in_cone", "taper_check", "taper_profile", "tapered"]

#: The edge modes, in the order the knob lists them.
EDGES = ("reflect", "cut", "gaussian_taper", "daf_periodic")
#: Longest side the dominant-wavelength estimate looks at; a larger field is decimated to it.
_DOMINANT_SIDE = 512
#: Whose cone of influence a cut uses: the coarsest scale's for every scale, or each scale's own.
CUT_SCOPES = ("largest", "per_scale")

_SIDES = ("E", "N", "S", "W")


def deficits(roi, parent_shape, margin: int) -> dict:
    """Pixels of a ``margin`` px halo around ``roi`` = ``(row, col, h, w)`` that fall outside a
    parent of ``parent_shape`` = ``(H, W)``, per side."""
    r0, c0, h, w = (int(v) for v in roi)
    H, W = int(parent_shape[0]), int(parent_shape[1])
    m = int(margin)
    return {"N": max(0, m - r0), "S": max(0, r0 + h + m - H),
            "W": max(0, m - c0), "E": max(0, c0 + w + m - W)}


def cut_borders(plan, scope: str) -> list:
    """Per-scale cut border (px, finest first) from a halo plan's reaches: the coarsest scale's
    reach at every scale for ``"largest"``; each scale's own for ``"per_scale"``, never wider than
    the coarsest's, and the coarsest's where a scale's reach does not converge. Where the coarsest
    scale's reach does not converge either, its larger grid's radius stands in."""
    top = plan[-1].reach
    b_max = int(math.ceil(top.radius if top.converged else top.measured))
    if scope == "largest":
        return [b_max] * len(plan)
    return [min(b_max, int(math.ceil(sh.reach.radius))) if sh.reach.converged else b_max
            for sh in plan]


def drop_in_cone(levels, borders, offset, parent_shape) -> list:
    """Each level's maxima at least its border from every edge of the parent: positions on a grid
    whose origin sits at ``offset`` = ``(row, col)`` of a parent of ``parent_shape`` = ``(H, W)``;
    every per-point array of a level is subset in step."""
    oy, ox = (int(v) for v in offset)
    H, W = int(parent_shape[0]), int(parent_shape[1])
    out = []
    for level, b in zip(levels, borders):
        x = np.asarray(level["x"], dtype=np.float64) + ox
        y = np.asarray(level["y"], dtype=np.float64) + oy
        keep = np.minimum.reduce([x, y, W - 1 - x, H - 1 - y]) >= b
        n = keep.size
        out.append({k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n
                        else v) for k, v in level.items()})
    return out


def chain_reach(scales, borders):
    """The ``reach_at`` callable :func:`dynamix.core.chaining.border_trimmed` takes: a chain's
    ``log2_scales`` to the border of the nearest scale."""
    log2 = np.log2(np.asarray(scales, dtype=np.float64))
    b = np.asarray(borders, dtype=np.float64)

    def reach_at(log2_scales):
        ls = np.asarray(log2_scales, dtype=np.float64)
        return b[np.abs(ls[:, None] - log2[None, :]).argmin(axis=1)]
    return reach_at


def dominant_wavelength(values) -> float:
    """The field's dominant wavelength in px: one over the radial frequency where its
    Hann-windowed power (mean removed, summed over rings of one frequency step) peaks, the zero
    frequency left out. A field longer than 512 px is decimated first. A red spectrum peaks at the
    first ring, so its dominant wavelength is the field's own size."""
    from dynamix.core import fft_policy

    a = np.asarray(values, dtype=np.float64)
    a = a.reshape(-1, a.shape[-1]) if a.ndim == 1 else a
    step = max(1, int(math.ceil(max(a.shape) / _DOMINANT_SIDE)))
    a = a[::step, ::step]
    finite = np.isfinite(a)
    a = np.where(finite, a, a[finite].mean() if finite.any() else 0.0)
    a = (a - a.mean()) * np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    power = np.abs(fft_policy.double().rfft2(a)) ** 2
    dk = 1.0 / max(a.shape)
    k = np.hypot(np.fft.fftfreq(a.shape[0])[:, None], np.fft.rfftfreq(a.shape[1])[None, :])
    rings = np.bincount(np.floor(k / dk + 0.5).astype(np.int64).ravel(), weights=power.ravel())
    rings[0] = 0.0
    return step / (int(np.argmax(rings)) * dk)


def taper_profile(t, w_t: float):
    """1 from ``w_t`` px inside the edge on, exp(−½((w_t − t)/(w_t/4))²) nearer, ``t`` px from
    it (e⁻⁸, 3.4e-4, at the edge itself)."""
    t = np.asarray(t, dtype=np.float64)
    return np.where(t >= w_t, 1.0, np.exp(-0.5 * ((w_t - t) / (0.25 * w_t)) ** 2))


def tapered(block, rows, cols, parent_shape, mu: float, w_t: float):
    """``block`` with the taper applied: ``rows`` and ``cols`` are its rows' and columns' indices
    in a parent of ``parent_shape``; inside it, values within ``w_t`` px of the parent's edge are
    rolled off towards ``mu`` (the rest untouched), and outside it the field is ``mu``."""
    H, W = int(parent_shape[0]), int(parent_shape[1])
    r = np.asarray(rows)[:, None]
    c = np.asarray(cols)[None, :]
    inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    t = np.minimum(np.minimum(r, H - 1 - r), np.minimum(c, W - 1 - c))
    block = np.asarray(block, dtype=np.float64)
    rolled = mu + (block - mu) * taper_profile(t, w_t)
    return np.where(inside, np.where(t >= w_t, block, rolled), mu)


def taper_check(parent_shape, dominant: float, reach_max: float):
    """``(w_t, reason)``: the taper width, the field's dominant wavelength or the coarsest
    scale's reach r(s_max), whichever is longer, and None, or the reason the taper is off (it
    would take more than a quarter of the raster's shorter side)."""
    w_t = max(float(dominant), float(reach_max))
    limit = min(int(parent_shape[0]), int(parent_shape[1])) / 4.0
    if w_t <= limit:
        return w_t, None
    return w_t, (f"Gaussian taper off, reflect used: the taper must span the field's dominant "
                 f"wavelength ({dominant:.1f} px) and the coarsest scale's reach r(s_max) "
                 f"({reach_max:.1f} px), {w_t:.1f} px, more than a quarter of the raster's "
                 f"shorter side ({limit:.0f} px).")


def corner_warning(rows, mode: str) -> "str | None":
    """The ⚠ text where halos run past the raster under ``mode``: ``rows`` are ``(scale_px,
    synthetic px per side)`` per scale, finest first. None when nothing was padded or the mode
    pads nothing that reaches the result."""
    if mode != "reflect":
        return None
    hit = [(float(a), d) for a, d in rows if any(d.values())]
    if not hit:
        return None
    sides = "/".join(s for s in _SIDES if any(d[s] for _a, d in hit))
    worst = max(v for _a, d in hit for v in d.values())
    span = f"{hit[0][0]:.2f} px" if len(hit) == 1 else f"{hit[0][0]:.2f}–{hit[-1][0]:.2f} px"
    return (f"{len(hit)} scale(s) at {span} reach past the raster's {sides} edge: up to {worst} px "
            "of reflection fill the halo there. Reflection can create false maxima at the mirror "
            "lines; edge = cut drops the cone of influence instead.")
