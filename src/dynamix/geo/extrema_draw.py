# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Extrema drawn as xsmurf's ``econvert`` draws them: one block per maximum, never joined, an
argument tick from each block's centre of length ``3 + 30 * M / M_max`` pixels
(``widgets/ConvExt.c`` ``_line2_``). Pure geometry in the pixel frame; ``to_axes`` carries it
onto a grid's own axes, so a pixel-frame angle never reaches a map as a map angle."""
from __future__ import annotations

import numpy as np


def block_positions(ext: dict, *, subpixel: bool):
    use_sub = subpixel and "x_sub" in ext and "y_sub" in ext
    cols = np.asarray(ext["x_sub"] if use_sub else ext["x"], dtype=np.float64)
    rows = np.asarray(ext["y_sub"] if use_sub else ext["y"], dtype=np.float64)
    return cols, rows


def tick_ends(cols, rows, arg, mod, m_max: float):
    mod = np.abs(np.asarray(mod, dtype=np.float64))
    length = 3.0 + (30.0 * mod / m_max if m_max > 0 else 0.0 * mod)
    a = np.asarray(arg, dtype=np.float64)
    return cols + length * np.cos(a), rows + length * np.sin(a)


def modulus_scalars(mod, mode: str = "linear", q: float = 1.0):
    m = np.abs(np.asarray(mod, dtype=np.float64))
    if m.size == 0 or not np.any(m > 0):
        return np.zeros(m.shape)
    if mode == "log":
        lm = np.log(np.where(m > 0, m, np.nan))
        lo, hi = np.nanmin(lm), np.nanmax(lm)
        return np.nan_to_num((lm - lo) / (hi - lo) if hi > lo else lm * 0.0)
    p = q if mode == "q" else 1.0
    return m ** p / np.max(m) ** p


def group_rgb(chains, scale_idx: int, cols, rows):
    out = np.full((len(cols), 3), -1, dtype=np.int64)
    where = {}
    for chain in chains or ():
        if not any(str(t).startswith("group:") for t in chain.get("tags") or ()):
            continue
        x, y = np.asarray(chain["x"]), np.asarray(chain["y"])
        if scale_idx < len(x):
            where[(int(x[scale_idx]), int(y[scale_idx]))] = chain.get("group_color") or (128, 128, 128)
    for i, (c, r) in enumerate(zip(np.rint(cols).astype(int), np.rint(rows).astype(int))):
        if (c, r) in where:
            out[i] = where[(c, r)]
    return out


def to_axes(cols, rows, x_axis, y_axis):
    from dynamix.geo.mapping import axis_at

    return axis_at(x_axis, cols), axis_at(y_axis, rows)
