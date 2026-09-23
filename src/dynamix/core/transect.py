# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Transect v1: pure numpy/scipy sampling, smoothing, and
swath-selection math for an A->A' profile drawn on the raster canvas.

Qt-free, per the project's ("Nothing outside dynamix/shell/ imports PySide6...") -- the shell
(dynamix.shell.canvas/transect_panel/profile_dialog/main_window) is the only place any of this
ever touches a pixel on screen or a mouse click. Everything here works in PLAIN PIXEL coordinates
("Scales are stored in pixels, converted for display, never the reverse") -- there is no
lon/lat, no haversine, no CRS anywhere in this module. EQSelect's own ``eqselect/transect.py`` (a
read-only reference) solves the
geographic version of the same problem; the algorithms below REIMPLEMENT its documented semantics
(bilinear sampling, NaN-off-grid, the NaN-bridge-then-filter-then-restore smoothing shape) for a
plain pixel grid -- nothing here is copied from it (the project rules forbid importing or modifying
EQSelect at all, verbatim-copy or otherwise; this is DynamiX's own code to the same, documented
algorithm).
"""
from __future__ import annotations

import numpy as np

__all__ = ["SMOOTHINGS", "orient_endpoints", "sample_profile", "smooth_profile",
          "chains_in_buffer"]

#: The four smoothing kinds EQSelect itself offers (its own ``transect.py``'s ``SMOOTHINGS``) -- kept as the identical tuple/order/spelling so a future combo box population needs no
#: translation table.
SMOOTHINGS = ("none", "gaussian", "median", "savgol")


def orient_endpoints(a, b):
    """Auto-orient a transect's two endpoints: LEFT->RIGHT when the segment is more
    horizontal than vertical (``|dx| >= |dy|``), else BOTTOM->TOP -- the pixel-space analog of
    EQSelect's own W->E / S->N rule ("A->A' orientation is automatic... regardless
    of click order... List header says so").

    **Y-axis convention -- read this carefully.** Canvas/image pixel space grows DOWNWARD: row 0
    is the TOP of the raster and y increases toward the BOTTOM (``dynamix.shell.canvas.Canvas``'s
    own ``view.invertY(True)`` is what makes the on-screen picture agree with this). So
    "BOTTOM -> TOP" here means the START point is the one with the LARGER y (closer to the bottom
    of the screen / the raster's last row) and the END point has the SMALLER y (closer to the top
    / row 0) -- the OPPOSITE of what "increasing y" suggests to anyone thinking in ordinary
    Cartesian (y-grows-up) terms. This is the one place a sign error here would be invisible on a
    square, symmetric test case and wrong on every real screenshot.

    ``a``/``b`` are ``(x, y)`` pairs (any numeric type); returns a ``((ax, ay), (bx, by))`` pair
    of plain float tuples, possibly SWAPPED from the input order -- never any other geometry
    (never a projection, a rotation, or a new point). A degenerate ``a == b`` returns the pair
    unchanged: it falls into the ``|dx| >= |dy|`` branch (``0 >= 0``), and ``ax <= bx`` is true
    when they are equal, so no separate case is needed.
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    if abs(dx) >= abs(dy):
        return ((ax, ay), (bx, by)) if ax <= bx else ((bx, by), (ax, ay))
    return ((ax, ay), (bx, by)) if ay >= by else ((bx, by), (ax, ay))


def sample_profile(values2d, a_px, b_px, n: int = 400, spacing: float = 1.0):
    """Bilinear sample of ``values2d`` along the straight A->B line, ``n`` evenly spaced points.

    ``a_px``/``b_px`` are ``(x, y)`` PIXEL coordinates (column, row) -- the same convention every
    other pixel-space helper in this codebase already uses (``dynamix.shell.canvas.
    roi_from_corners``'s own docstring: "x is a COLUMN and y is a ROW"). ``values2d`` is indexed
    ``(row, col)``, so :class:`scipy.interpolate.RegularGridInterpolator` is built and queried
    with points as ``(y, x)`` pairs, not ``(x, y)``.

    Returns ``(dist, z)``. ``z`` is the bilinearly interpolated value at each of the ``n``
    samples (``bounds_error=False, fill_value=np.nan``) -- a sample that lands off the grid is
    NaN, never clamped or extrapolated, mirroring EQSelect's own ``extract_profile``/
    ``sample_along``. ``dist`` is the straight-line distance from A, in PIXEL units, scaled by
    ``spacing`` -- ``spacing`` is frame units per pixel (``1.0``, the default, leaves ``dist`` in
    bare pixels). This module never converts a physical unit itself; the shell is the one place
    that knows a frame's own unit (mirrors ``dynamix.shell.main_window``'s existing
    ``_skeleton_px_size`` -- CW§1's identical "frame units per pixel, or None for bare pixels"
    reading, reused here rather than re-derived a second way).
    """
    z = np.asarray(values2d, dtype=np.float64)
    ny, nx = z.shape[0], z.shape[1]
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (np.arange(ny, dtype=np.float64), np.arange(nx, dtype=np.float64)), z,
        bounds_error=False, fill_value=np.nan)
    t = np.linspace(0.0, 1.0, int(n))
    ax, ay = float(a_px[0]), float(a_px[1])
    bx, by = float(b_px[0]), float(b_px[1])
    px = ax + (bx - ax) * t
    py = ay + (by - ay) * t
    zvals = interp(np.column_stack([py, px]))
    dist = np.hypot(px - ax, py - ay) * float(spacing)
    return dist, zvals


def smooth_profile(z, smoothing: str = "none", sigma: float = 3.0, window: int = 11):
    """1-D filter along a sampled profile -- EQSelect's own ``smooth_profile`` ALGORITHM
    (``eqselect/transect.py``), reimplemented (not copied -- see the module docstring).

    ``smoothing`` must be one of :data:`SMOOTHINGS`; anything else raises ``ValueError`` (EQSelect
    itself raises identically for an unknown kind). ``"none"`` (or ``""``/``None``, both fold to
    ``"none"``) returns an unfiltered copy.

    **NaN-bridging -- read this before assuming the filter "just works" on a profile with
    off-grid samples.** :func:`sample_profile`'s own NaN (a sample outside the raster) is first
    BRIDGED: linearly interpolated over, BY INDEX, from its nearest finite neighbors
    (``np.interp`` against the finite subset) -- so a scipy filter kernel never sees, and
    therefore never SPREADS, a NaN into nearby genuinely-sampled values (a single missing sample
    in the middle of a gaussian/median/savgol window would otherwise turn a whole neighborhood of
    real data into NaN too). The filter then runs on the fully-bridged array. Finally, every
    ORIGINALLY-missing sample is restored to NaN in the OUTPUT -- the bridge exists ONLY to keep
    the filter's kernel from tripping over a hole; it is never allowed to fabricate a value the
    profile did not actually sample. This is the exact mechanism EQSelect's own docstring
    describes ("bridge NaNs... so filters don't spread them" / "keep originally-missing samples
    missing") -- interp-over-NaN, filter, restore, not filter-then-mask or drop-and-resample.

    A profile with fewer than 2 finite samples skips the bridge (nothing to interpolate FROM) and
    filters the raw, mostly-NaN array as-is -- scipy's own behavior for degenerate input, not a
    special case added here.
    """
    z = np.asarray(z, dtype=np.float64)
    kind = (smoothing or "none").lower()
    if kind not in SMOOTHINGS:
        raise ValueError(f"unknown smoothing {smoothing!r}; choose from {SMOOTHINGS}")
    if kind == "none":
        return z.copy()

    x = z.copy()
    nan = ~np.isfinite(x)
    if nan.any() and (~nan).sum() >= 2:
        idx = np.arange(x.size)
        x[nan] = np.interp(idx[nan], idx[~nan], x[~nan])

    if kind == "gaussian":
        from scipy.ndimage import gaussian_filter1d
        out = gaussian_filter1d(x, sigma=float(sigma), mode="nearest")
    elif kind == "median":
        from scipy.ndimage import median_filter
        out = median_filter(x, size=max(3, int(window)), mode="nearest")
    else:                                                     # "savgol"
        from scipy.signal import savgol_filter
        w = max(5, int(window) | 1)                            # force odd, >= 5
        w = min(w, x.size - (x.size + 1) % 2)                  # not longer than the data (odd)
        out = savgol_filter(x, w, polyorder=2) if x.size > w else x

    out = np.asarray(out, dtype=np.float64)
    out[nan] = np.nan                                          # keep originally-missing missing
    return out


def chains_in_buffer(chains, a, b, buffer_px) -> list[int]:
    """Point-to-segment distance from every chain point to the transect segment A-B; a chain's
    index is appended to the result the instant ANY ONE of its own points lies within
    ``buffer_px`` of the segment -- the same "any point of a chain -> the whole chain is picked"
    doctrine every other picking helper in this codebase already applies
    (``dynamix.core.chain_pick.chains_in_box``/``chains_in_polygon``).

    Distance to the SEGMENT, not the infinite line through it: a point beyond A or beyond A'
    clamps to its nearest ENDPOINT (the standard point-to-segment formula), rather than being
    excluded outright -- a round end-cap, the simplest honest reading of "point-to-segment
    distance". EQSelect's own clean-RECTANGLE variant (``project_events_to_segment``'s
    ``on_segment`` flag, which excludes a clamped-past-the-end foot) is a documented v1 non-goal
    (the design's "Deferred" list) -- round end-caps are a strict superset (never fewer chains
    than the rectangle would find), which is the honest direction to simplify in for a first cut.

    ``a``/``b`` and every chain's own ``x``/``y`` must already be in the SAME coordinate space --
    the caller's job. ``dynamix.shell.main_window``'s swath-select handler passes the SAME
    display-shifted chains ``Canvas.set_pick_chains`` already holds (via ``Canvas.pick_chains()``),
    so a swath selection and an ordinary click pick never disagree about where a chain actually is
    ("your swath select must use the SAME shifted coordinates for chain distance,
    consistent with picking").
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    abx, aby = bx - ax, by - ay
    length2 = abx * abx + aby * aby
    buffer_px = float(buffer_px)
    hits: list[int] = []
    for i, chain in enumerate(chains):
        x = np.asarray(chain.get("x", ()), dtype=np.float64).ravel()
        y = np.asarray(chain.get("y", ()), dtype=np.float64).ravel()
        k = min(x.size, y.size)
        if k == 0:
            continue
        x, y = x[:k], y[:k]
        if length2 <= 0:
            dist = np.hypot(x - ax, y - ay)
        else:
            t = np.clip(((x - ax) * abx + (y - ay) * aby) / length2, 0.0, 1.0)
            dist = np.hypot(x - (ax + t * abx), y - (ay + t * aby))
        if np.any(dist <= buffer_px):
            hits.append(i)
    return hits
