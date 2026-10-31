# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Contrast stretches for DISPLAY: map a field onto [0, 1] for colouring.

Under a linear (min/max) stretch a single outlier crushes everything else to black. These
are the standard remote-sensing choices (ENVI's menu): percent clip, standard
deviation, logarithmic, histogram equalisation. NaN stays NaN so masked cells stay
transparent. A diverging colormap's middle colour marks a value (zero for h and for any signed
field): :func:`centered` puts it there, two slopes meeting at the centre, in the modes
:func:`center_mode` names. Nothing here is analysis -- the WTMM never sees a stretched value.
"""
from __future__ import annotations

import numpy as np

STRETCHES = ("linear", "percent", "stddev", "log", "histogram", "bipolar")

#: The stretches that scale linearly between two data-unit limits (:func:`stretch_limits`), the
#: ones :func:`centered` can split at a centre value.
LIMIT_STRETCHES = ("linear", "percent", "stddev", "bipolar")

#: matplotlib's diverging colormaps: its Diverging family and the dark-centred berlin, managua and
#: vanimo. Their ``_r`` reverses and pyqtgraph's colorcet ``CET-D`` maps diverge too
#: (:func:`diverging_name`).
DIVERGING = ("PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn", "Spectral",
             "coolwarm", "bwr", "seismic", "berlin", "managua", "vanimo")


def diverging_name(name) -> bool:
    """Whether the built-in colormap ``name`` is diverging: its middle colour marks a neutral
    value and its two ends departures from it of opposite sign."""
    name = str(name)
    return name.removesuffix("_r") in DIVERGING or name.startswith("CET-D")


def _scale(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = np.zeros_like(v, dtype=np.float64)
        out[~np.isfinite(v)] = np.nan
        return out
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def stretch(values, mode: str = "linear", *, percent: float = 2.0, k: float = 2.0,
            limits=None) -> np.ndarray:
    """``values`` → floats in [0, 1] (NaN preserved) under ``mode``:

    - ``linear``: min..max, or ``limits`` = (lo, hi) when given (a byte image's own 0..255, so
      the picture reads as the file shows it).
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
    if mode in LIMIT_STRETCHES:
        lo, hi = stretch_limits(v, mode, percent=percent, k=k, limits=limits)
        if mode == "bipolar" and hi <= 0.0:
            out = np.full(v.shape, 0.5)                                # all zero: mid-scale
        else:
            out = _scale(v, lo, hi)
    elif mode == "log":
        shifted = v - float(f.min())
        lg = np.log1p(np.where(finite, shifted, 0.0))
        out = _scale(lg, 0.0, float(np.log1p(float(f.max()) - float(f.min()))))
    else:  # histogram
        order = np.argsort(f, kind="stable")
        ranks = np.empty(f.size, dtype=np.float64)
        ranks[order] = np.arange(f.size, dtype=np.float64)
        out = np.zeros_like(v)
        out[finite] = ranks / max(f.size - 1, 1)
    out = np.asarray(out, dtype=np.float64)
    out[~finite] = np.nan
    return out


def stretch_limits(values, mode: str = "linear", *, percent: float = 2.0, k: float = 2.0,
                   limits=None) -> tuple[float, float]:
    """The data-unit ``(lo, hi)`` a limit stretch (:data:`LIMIT_STRETCHES`) maps onto 0 and 1, as
    :func:`stretch` describes each: min..max or ``limits``, the percentiles, mean ± kσ, ±m.
    ``(nan, nan)`` for a field with no finite cell."""
    if mode not in LIMIT_STRETCHES:
        raise ValueError(f"{mode!r} has no data-unit limits; one of {', '.join(LIMIT_STRETCHES)}")
    v = np.asarray(values, dtype=np.float64)
    f = v[np.isfinite(v)]
    if mode == "linear" and limits is not None:
        return float(limits[0]), float(limits[1])
    if not f.size:
        return np.nan, np.nan
    if mode == "linear":
        return float(f.min()), float(f.max())
    if mode == "percent":
        lo, hi = np.percentile(f, [percent, 100.0 - percent])
        return float(lo), float(hi)
    if mode == "stddev":
        m, s = float(f.mean()), float(f.std())
        return m - k * s, m + k * s
    m = float(np.percentile(np.abs(f), 100.0 - percent))
    if m <= 0.0:
        m = float(np.abs(f).max())
    return -m, m


def stretch_at(values, at, mode: str = "linear", *, percent: float = 2.0, k: float = 2.0,
               limits=None) -> np.ndarray:
    """Where the data values ``at`` sit on the [0, 1] scale :func:`stretch` maps ``values`` onto
    under ``mode``: the positions of a colour bar's ticks. A limit stretch scales ``at`` between
    its limits, ``log`` takes ``log1p`` of its offset from the minimum, ``histogram`` its rank
    among the finite cells."""
    a = np.asarray(at, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    f = v[np.isfinite(v)]
    if not f.size:
        return np.full(a.shape, np.nan)
    if mode in LIMIT_STRETCHES:
        lo, hi = stretch_limits(f, mode, percent=percent, k=k, limits=limits)
        return np.full(a.shape, 0.5) if mode == "bipolar" and hi <= 0.0 else _scale(a, lo, hi)
    if mode == "log":
        low = float(f.min())
        return _scale(np.log1p(np.maximum(a - low, 0.0)), 0.0, float(np.log1p(float(f.max()) - low)))
    if mode != "histogram":
        raise ValueError(f"unknown stretch {mode!r}; one of {', '.join(STRETCHES)}")
    rank = np.clip(np.searchsorted(np.sort(f), a, side="left") / max(f.size - 1, 1), 0.0, 1.0)
    return np.where(np.isfinite(a), rank, np.nan)


#: Where a diverging ramp's middle colour goes (:func:`center_mode`): ``none`` keeps the plain
#: mapping between the stretch's limits, ``value`` puts it on the centre value with each side
#: reaching its own limit, ``symmetric`` on the centre value with the limits centre ± the larger
#: departure; ``auto`` is ``value`` for a field on both sides of the centre, ``none`` otherwise.
CENTER_MODES = ("auto", "none", "value", "symmetric")


def center_mode(values, center: float = 0.0, mode: str = "auto") -> str:
    """``mode`` with ``auto`` resolved on ``values``: ``value`` when finite cells lie on both
    sides of ``center`` (a signed field about zero), ``none`` otherwise, so a field on one side
    of it (KAM, a modulus) keeps the whole ramp."""
    if mode != "auto":
        return mode
    v = np.asarray(values, dtype=np.float64)
    f = v[np.isfinite(v)]
    return "value" if f.size and f.min() < center < f.max() else "none"


def centered_limits(lo: float, hi: float, center: float = 0.0,
                    symmetric: bool = False) -> tuple[float, float]:
    """The limits :func:`centered` maps onto 0 and 1: ``lo``..``hi`` widened to reach ``center``
    when it lies outside them, or with ``symmetric`` ``center`` ∓ max(|lo − center|,
    |hi − center|), one slope on both sides. A limit that is not finite reads as the centre."""
    c = float(center)
    lo = float(lo) if np.isfinite(lo) else c
    hi = float(hi) if np.isfinite(hi) else c
    lo, hi = min(lo, c), max(hi, c)
    if symmetric:
        m = max(c - lo, hi - c)
        lo, hi = c - m, c + m
    return lo, hi


def centered(values, lo: float, hi: float, center: float = 0.0,
             symmetric: bool = False) -> np.ndarray:
    """Two slopes meeting at ``center``: lo..center onto 0..0.5 and center..hi onto 0.5..1, each
    linear and clipped, so a diverging ramp's middle colour lands on ``center`` whatever the
    limits are (:func:`centered_limits` widens them to reach it, or makes them symmetric). A side
    of zero width sends what lies beyond the centre to its end. NaN preserved."""
    v = np.asarray(values, dtype=np.float64)
    lo, hi = centered_limits(lo, hi, center, symmetric)
    c = float(center)
    with np.errstate(invalid="ignore", divide="ignore"):
        below = (0.5 * np.clip((v - lo) / (c - lo), 0.0, 1.0) if c > lo
                 else np.where(v < c, 0.0, 0.5))
        above = (0.5 + 0.5 * np.clip((v - c) / (hi - c), 0.0, 1.0) if hi > c
                 else np.where(v > c, 1.0, 0.5))
    out = np.asarray(np.where(v <= c, below, above), dtype=np.float64)
    out[~np.isfinite(v)] = np.nan
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
    breaks of ``values``'s finite cells; a break list -> itself, sorted. A repeated break is an
    empty class, so the classes either side keep their numbers and per-class colours stay
    aligned. The shared resolution ``classify`` uses -- exposed so the slice editor can seed
    its draggable lines from the same numbers the display is using."""
    v = np.asarray(values, dtype=np.float64)
    if isinstance(levels, (int, np.integer)):
        n = max(int(levels), 2)
        return np.unique(np.nanquantile(v, np.linspace(0.0, 1.0, n + 1)[1:-1]))
    breaks = np.sort(np.asarray(list(levels), dtype=np.float64))
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


#: Periodic colourings, the escape-time fractal schemes (Sekino, "Fractal coloring"): ``off``
#: leaves the stretch in charge.
BAND_MODES = ("off", "alternate", "wave", "cycle")


def bands(values, mode: str, period: float, *, phase: float = 0.0, k: int = 6, q: float = 1.0):
    """``values`` in units of ``period`` from ``phase``, u = (v − phase)/period, as [0, 1]
    colormap positions: ``alternate`` flips between 0 and 1 each period (zebra bands along level
    sets); ``wave`` rises and falls once per period, t = (1 − |2 frac(u) − 1|)^q; ``cycle``
    steps through ``k`` colours, t = (⌊u⌋ mod k)/(k − 1). Non-finite cells stay NaN."""
    if mode not in BAND_MODES[1:]:
        raise ValueError(f"bands mode must be one of {BAND_MODES[1:]}")
    if not period > 0:
        raise ValueError("bands period must be positive")
    u = (np.asarray(values, dtype=np.float64) - phase) / period
    n = np.floor(u)
    if mode == "alternate":
        return np.mod(n, 2.0)
    if mode == "wave":
        return (1.0 - np.abs(2.0 * (u - n) - 1.0)) ** q
    k = max(int(k), 2)
    return np.mod(n, k) / (k - 1)


def auto_band_period(values, n: int = 16) -> float:
    """A ``n``-th of the finite range of ``values``; 1 when it is flat or empty."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    span = float(v.max() - v.min()) if v.size else 0.0
    return span / n if span > 0 else 1.0


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
