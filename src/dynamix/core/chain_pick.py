# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Screen/data-space picking over chain point sets. Pure numpy; the shell converts
click coordinates and pixel radii to data units before calling in."""
from __future__ import annotations

import numpy as np

from dynamix.core.selection import nearest_point, points_in_box, points_in_polygon


def _chain_points(chains):
    """(pts2d (N,2) float array, owner chain-index (N,) int array) over all chains."""
    xs, ys, owner = [], [], []
    for i, chain in enumerate(chains):
        x = np.asarray(chain.get("x", ()), dtype=np.float64).ravel()
        y = np.asarray(chain.get("y", ()), dtype=np.float64).ravel()
        k = min(x.size, y.size)
        if k == 0:
            continue
        xs.append(x[:k]); ys.append(y[:k])
        owner.append(np.full(k, i, dtype=np.int64))
    if not xs:
        return np.empty((0, 2)), np.empty(0, dtype=np.int64)
    return np.column_stack([np.concatenate(xs), np.concatenate(ys)]), np.concatenate(owner)


def pick_chain(chains, xy, max_dist) -> int | None:
    pts2d, owner = _chain_points(chains)
    if not len(pts2d):
        return None
    idx = nearest_point(pts2d, xy, max_dist=max_dist)
    if idx < 0:
        return None
    return int(owner[idx])


def chains_in_polygon(chains, polygon) -> list[int]:
    pts2d, owner = _chain_points(chains)
    if not len(pts2d) or len(polygon) < 3:
        return []
    inside = points_in_polygon(pts2d, polygon)
    return sorted(set(int(i) for i in owner[inside]))


def chains_in_box(chains, xmin, xmax, ymin, ymax) -> list[int]:
    """The box-mode sibling of :func:`chains_in_polygon` -- same
    "any point of a chain lands inside -> the whole chain is picked" shape, over
    :func:`dynamix.core.selection.points_in_box`'s axis-aligned rectangle test instead of a
    polygon. Wires ``core.selection.points_in_box`` (pure-numpy, pre-existing, unused by any
    GUI before this task) into the same owner-index bookkeeping :func:`chains_in_polygon`
    already uses."""
    pts2d, owner = _chain_points(chains)
    if not len(pts2d):
        return []
    inside = points_in_box(pts2d, xmin, xmax, ymin, ymax)
    return sorted(set(int(i) for i in owner[inside]))
