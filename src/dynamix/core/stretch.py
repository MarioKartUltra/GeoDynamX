# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Contrast stretches for DISPLAY: map a field onto [0, 1] for colouring.

Under a linear (min/max) stretch a single outlier crushes everything else to black. These
are the standard remote-sensing choices (ENVI's menu): percent clip, standard
deviation, logarithmic, histogram equalisation. NaN stays NaN so masked cells stay
transparent. Nothing here is analysis -- the WTMM never sees a stretched value.
"""
from __future__ import annotations

import numpy as np

STRETCHES = ("linear", "percent", "stddev", "log", "histogram", "bipolar")


def _scale(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = np.zeros_like(v, dtype=np.float64)
        out[~np.isfinite(v)] = np.nan
        return out
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def stretch(values, mode: str = "linear", *, percent: float = 2.0, k: float = 2.0) -> np.ndarray:
    """``values`` → floats in [0, 1] (NaN preserved) under ``mode``:

    - ``linear``: min..max.
    - ``percent``: the ``percent``-th .. ``100 - percent``-th percentiles, clipped.
    - ``stddev``: mean ± ``k``·σ, clipped.
    - ``log``: ``log1p`` over the range shifted to start at 0, then min..max -- lifts the dark end.
    - ``histogram``: equalisation -- each cell's rank among the finite cells, so every grey
      level is used equally.
    - ``bipolar``: symmetric about ZERO for signed data (PCs, residuals, anomalies): −m..+m
      with m the ``100 - percent``-th percentile of ``|v|``, clipped -- zero lands mid-scale,
      so positive and negative read as opposites.
    """
    if mode not in STRETCHES:
        raise ValueError(f"unknown stretch {mode!r}; one of {', '.join(STRETCHES)}")
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    if not finite.any():
        return np.full(v.shape, np.nan)
    f = v[finite]
    if mode == "linear":
        out = _scale(v, float(f.min()), float(f.max()))
    elif mode == "percent":
        lo, hi = np.percentile(f, [percent, 100.0 - percent])
        out = _scale(v, float(lo), float(hi))
    elif mode == "stddev":
        m, s = float(f.mean()), float(f.std())
        out = _scale(v, m - k * s, m + k * s)
    elif mode == "log":
        shifted = v - float(f.min())
        lg = np.log1p(np.where(finite, shifted, 0.0))
        out = _scale(lg, 0.0, float(np.log1p(float(f.max()) - float(f.min()))))
    elif mode == "bipolar":
        m = float(np.percentile(np.abs(f), 100.0 - percent))
        if m <= 0.0:
            m = float(np.abs(f).max())
        out = _scale(v, -m, m) if m > 0.0 else np.full(v.shape, 0.5)   # all zero: mid-scale
    else:  # histogram
        order = np.argsort(f, kind="stable")
        ranks = np.empty(f.size, dtype=np.float64)
        ranks[order] = np.arange(f.size, dtype=np.float64)
        out = np.zeros_like(v)
        out[finite] = ranks / max(f.size - 1, 1)
    out = np.asarray(out, dtype=np.float64)
    out[~finite] = np.nan
    return out


def parse_levels(spec) -> "int | list | None":
    """Density-slice spec text -> ``None`` (off), an int class COUNT, or data-unit BREAKS.

    ``""`` -> None. A single non-negative integer token ("5") -> that many QUANTILE classes.
    Anything else -> comma/semicolon-separated floats, the class boundaries in DATA units
    (ENVI's density-slice ranges -- e.g. h breaks ``"-0.5, 0, 0.8"``). Raises ``ValueError``
    on unparseable text so a UI can refuse honestly.
    """
    s = str(spec or "").strip()
    if not s:
        return None
    parts = [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]
    if len(parts) == 1 and parts[0].isdigit():
        return int(parts[0])
    try:
        return [float(x) for x in parts]
    except ValueError as exc:
        raise ValueError(f"levels: {spec!r} is neither a class count nor "
                         f"comma-separated breaks") from exc


def classify(values, levels) -> np.ndarray:
    """Density slice (ENVI): values -> class index / (n_classes - 1) in [0, 1], NaN kept.

    ``levels``: an int N -> N QUANTILE classes over the finite values (equal-population --
    the right default for an h map); or ascending DATA-UNIT breaks -> ``len + 1`` classes,
    open-ended at both ends. Replaces the continuous stretch when active, so the colormap
    paints piecewise-constant slices; pair with a text spec via :func:`parse_levels`.
    """
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    if not finite.any():
        return np.full(v.shape, np.nan)
    breaks = resolve_breaks(v, levels)
    idx = np.searchsorted(breaks, v, side="right").astype(np.float64)
    out = idx / max(breaks.size, 1)
    out[~finite] = np.nan
    return out


def resolve_breaks(values, levels) -> np.ndarray:
    """The actual class boundaries for a levels spec: an int N -> the N-quantile interior
    breaks of ``values``'s finite cells; a break list -> itself, sorted/deduped. The shared
    resolution ``classify`` uses -- exposed so the slice editor can seed its draggable lines
    from the same numbers the display is using."""
    v = np.asarray(values, dtype=np.float64)
    if isinstance(levels, (int, np.integer)):
        n = max(int(levels), 2)
        return np.unique(np.nanquantile(v, np.linspace(0.0, 1.0, n + 1)[1:-1]))
    breaks = np.unique(np.asarray(list(levels), dtype=np.float64))
    if breaks.size == 0 or not np.isfinite(breaks).all():
        raise ValueError("breaks must be finite and non-empty")
    return breaks


def slice_indices(values, breaks) -> "tuple[np.ndarray, int]":
    """``(class index int array, n_classes)`` for explicit ascending breaks -- the integer
    sibling of :func:`classify`, for callers painting explicit per-class colors. NaN cells get
    index -1."""
    v = np.asarray(values, dtype=np.float64)
    breaks = np.asarray(breaks, dtype=np.float64)
    idx = np.searchsorted(breaks, v, side="right").astype(np.int64)
    idx[~np.isfinite(v)] = -1
    return idx, int(breaks.size) + 1


def parse_class_colors(text) -> "list | None":
    """``"#rrggbb,none,..."`` -> a list of (r, g, b, a) ints, or ``None`` for empty/blank.

    ``none`` (or ``-``) is a TRANSPARENT class -- alpha 0 -- which is what turns a density
    slice into Turiel's set display: one band colored, the rest invisible (the MSM-figure
    workflow). Raises ``ValueError`` on malformed entries -- user-typed/persisted
    display data, parsed here (no Qt) so the canvas and the pyvista scene share one reading."""
    s = str(text or "").strip()
    if not s:
        return None
    out = []
    for part in s.split(","):
        token = part.strip().lower()
        if token in ("none", "-"):
            out.append((0, 0, 0, 0))
            continue
        h = token.lstrip("#")
        if len(h) != 6:
            raise ValueError(f"class color {part.strip()!r} is not #rrggbb or 'none'")
        out.append(tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,))
    return out
