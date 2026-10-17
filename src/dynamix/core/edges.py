# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Edge handling for the scale-space tools: what fills a scale's reach where real data runs out.

Real parent data comes first (:mod:`dynamix.roi.halo` reads each scale's halo, or the whole parent
where a reach does not converge). Where the halo still runs past the raster, ``edge`` decides:
``"reflect"`` mirrors the data (the default; the mirror lines can carry false maxima, which the
warning says), ``"cut"`` keeps the padding out of the result by dropping the cone of influence.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["CUT_SCOPES", "EDGES", "corner_warning", "cut_borders", "deficits", "drop_in_cone"]

#: The edge modes, in the order the knob lists them.
EDGES = ("reflect", "cut")
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
