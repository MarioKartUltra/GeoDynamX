# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Ordered point runs of each horizontal maxima line (H-line) at one scale.

One ordering for every surface that draws H-lines as lines rather than dots: the 2-D canvas
(``shell/canvas.py::hline_polylines``, which predates this module and keeps its own NaN-separated
form) and the 3-D scene (``shell/arrangement/scene.py``). Both follow the topology's own walk,
:func:`dynamix.core.wtmm_backend._order_lines` -- a PRIVATE name in a verbatim EQSelect copy, so
this is the third call site carrying that coupling.
"""
from __future__ import annotations

import numpy as np

from dynamix.core.wtmm_backend import _order_lines


def hline_runs(ext: dict, shape) -> list[np.ndarray]:
    """Index arrays into ``ext["x"]``/``ext["y"]``, one per contiguous run, each ordered along
    its H-line. Singletons (``line_id == -1``) are not lines and never appear; a branching line
    yields one run per walk restart (``seg`` changes), exactly as the canvas splits them."""
    x = np.asarray(ext.get("x", ()), dtype=np.int64)
    y = np.asarray(ext.get("y", ()), dtype=np.int64)
    line_id = np.asarray(ext.get("line_id", ()), dtype=np.int64)
    if x.size == 0 or not np.any(line_id >= 0):
        return []
    ny, nx = int(shape[0]), int(shape[1])
    grid = np.full(ny * nx, -1, dtype=np.int64)
    grid[y * nx + x] = np.arange(x.size, dtype=np.int64)
    order, seg, starts = _order_lines(x, y, line_id, grid, nx, ny)
    order = np.asarray(order, dtype=np.int64)
    seg = np.asarray(seg)
    starts = np.asarray(starts, dtype=np.int64)
    if order.size == 0:
        return []
    # One pass over the whole ordering: a run breaks where the walk restarts (``seg`` changes)
    # and at every line's start; ``np.split`` once on the sorted union, then keep the walks
    # with an edge. Per line this was three numpy calls times tens of thousands of lines.
    breaks = np.nonzero(seg[1:] != seg[:-1])[0] + 1
    cuts = np.unique(np.concatenate([breaks, starts[(starts > 0) & (starts < order.size)]]))
    bounds = np.concatenate([[0], cuts, [order.size]])
    return [order[i:j] for i, j in zip(bounds[:-1], bounds[1:]) if j - i >= 2]      # a line needs an edge
