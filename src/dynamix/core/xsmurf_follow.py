# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
# Portions translated from xsmurf, Copyright (C) 1999 Centre de Recherche Paul Pascal,
# Bordeaux, France (N. Decoster, P. Kestener, S. Roux, A. Arneodo), GPL-2.0 -- see NOTICE.
"""EXACT ports of xsmurf's live follow pipeline -- pure numpy/python, NO GUI.

Ported from the tree the XSmurfWrapper oracle actually runs, ``xSmurfMacPorts/xsmurf``:

- ``follow_contour``/``near_contour_line``: ``wt2d/Extrema.c:1105/1083`` -- the DETECTION the
  ``follow`` Tcl command always uses (every ``followVersion`` routes here): a pixel is an
  edge iff kapap < 0 AND kapa changes sign against a 4-connected neighbor with
  |kapa(p)| < |kapa(neighbor)| (the nearer side of the kapa = 0 level line). Borders
  (1-px frame) excluded, as the C's loop bounds do. Closure of shapes is a SET property:
  wherever the level line passes between two 4-neighbors exactly one pixel registers, so a
  closed level line yields a gap-free ring -- no chain heuristic involved.
- ``interpolated_modulus``/``get_m2``: ``interpreter/wt2d_cmds.c:5170/5103`` -- the
  ``followVersion = 1`` DEFAULT value channel (``_get_interpolated_modulus_``): pick the
  cardinal of the strongest opposite-sign kapa (``_get_near_pos_``), linear crossing ratio
  ``s = |kapa(p)/(kapa(p) - kapa(p+u))|``, cubic through the four axis samples evaluated at
  ``s``, with the C's own sentinels (negative cubic -> -4 -> grid fallback; result outside
  ``[0, 2*mod]`` -> grid fallback; border -> grid). ``-old`` (followVersion 0) is the
  DEPRECATED no-interpolation path and is deliberately not ported; ``-v2`` is not ported yet
  (recorded gap). One recorded divergence: a non-finite crossing ratio (impossible for a
  registered maximum, which always has a genuine sign change) falls back to the grid value
  instead of propagating NaN as the C would under NDEBUG.
- ``search_lines``/``is_line_end``: ``wt2d/chain.c:605/…`` -- the chaining (``hsearch``):
  phase 1 seeds ONLY line ends (repeat until exhausted) so open lines are walked end-to-end;
  phase 2 seeds whatever remains, which is by then only CLOSED rings; the walk is
  ``_seek_neighbors_`` -- all 8 neighbors in the VERBATIM ``iPosArray``/``jPosArray``
  priority (cardinals before diagonals; the C comment's compass names mislabel the arrays --
  the arrays are the truth), single successor, PREPENDING each found extremum (so the seed
  ends the list), consuming as it goes; a line is CLOSED iff size > 3 and its two ends are
  Chebyshev-adjacent (``_are_near_``). An end is a pixel with <= 1 neighbor, or exactly 2
  neighbors that are 4-adjacent to EACH OTHER (a kink hugging the line). The C recursion is
  made iterative here (a long line would blow Python's stack); the visit order is identical.

``tests/test_xsmurf_follow_port.py`` pins these unit semantics; the dataset-level oracle is
``xsmurf_wrapper.find_extrema_follow`` in the parity file.
"""
from __future__ import annotations

import numpy as np

__all__ = ["near_contour_line", "follow_contour", "get_m2", "interpolated_modulus",
           "search_lines", "is_line_end", "I_POS_ARRAY", "J_POS_ARRAY"]

#: wt2d/Extrema.c pos_incr ring as (drow, dcol): [NW, N, NE, E, SE, S, SW, W, NW].
_RING = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))

#: wt2d/chain.c neighbor priority, VERBATIM: iPosArray adds to the COLUMN, jPosArray to the
#: ROW. Effective order: W, N, S, E, NW, SW, NE, SE -- cardinals before diagonals.
I_POS_ARRAY = [-1, 0, 0, 1, -1, -1, 1, 1]
J_POS_ARRAY = [0, -1, 1, 0, -1, 1, -1, 1]


def near_contour_line(kapa, y: int, x: int) -> int:
    """C ``near_contour_line``: the pos_incr index (1|3|5|7, truthy) of a cardinal whose
    kapa has the opposite sign AND larger magnitude, else 0. Caller keeps points inside the
    1-px frame, as the C's loop bounds do."""
    k = np.asarray(kapa)
    for i in (1, 3, 5, 7):
        dy, dx = _RING[i]
        if k[y, x] * k[y + dy, x + dx] < 0 and abs(k[y, x]) < abs(k[y + dy, x + dx]):
            return i
    return 0


def follow_contour(kapa, kapap):
    """C ``w2_folow_contour`` (min_or_max = 1), vectorized: ``(ys, xs)`` of every interior
    pixel with kapap < 0 that sits on the nearer side of a 4-neighbor kapa crossing."""
    k = np.asarray(kapa, dtype=np.float64)
    kp = np.asarray(kapap, dtype=np.float64)
    near = np.zeros(k.shape, dtype=bool)
    for i in (1, 3, 5, 7):
        dy, dx = _RING[i]
        n = np.roll(np.roll(k, -dy, axis=0), -dx, axis=1)
        near |= (k * n < 0) & (np.abs(k) < np.abs(n))
    keep = (kp < 0) & near
    keep[0, :] = keep[-1, :] = False          # the C loops y in [1, ly-2], x in [1, lx-2]
    keep[:, 0] = keep[:, -1] = False
    ys, xs = np.nonzero(keep)
    return ys.astype(np.int64), xs.astype(np.int64)


def get_m2(y_1: float, y0: float, y1: float, y2: float, s: float) -> float:
    """C ``_get_m2_`` verbatim: cubic through f(-1..2) = (y_1, y0, y1, y2) evaluated at
    ``s`` in [0, 1]; a negative value returns the C's -4 sentinel."""
    k0 = y0
    k1 = (-2.0 * y_1 + 6.0 * y1 - 3.0 * y0 - y2) / 6.0
    k2 = (y_1 + y1 - 2.0 * y0) / 2.0
    k3 = -k1 + k2 - y_1 + k0
    val = k3 * s * s * s + k2 * s * s + k1 * s + k0
    if val < 0:
        return -4.0
    return float(val)


def interpolated_modulus(mod, kapa, x: int, y: int) -> float:
    """C ``_get_interpolated_modulus_`` (followVersion = 1, the live default)."""
    m = np.asarray(mod, dtype=np.float64)
    k = np.asarray(kapa, dtype=np.float64)
    ly, lx = m.shape
    if x < 2 or x >= lx - 2 or y < 2 or y >= ly - 2:
        return float(m[y, x])
    # _get_near_pos_: among the firing cardinals, the one with the LARGEST |kapa(neighbor)|.
    best_i, best = 0, 0.0
    for i in (1, 3, 5, 7):
        dy, dx = _RING[i]
        if k[y, x] * k[y + dy, x + dx] < 0 and abs(k[y, x]) < abs(k[y + dy, x + dx]):
            if best_i == 0 or abs(k[y + dy, x + dx]) > best:
                best_i, best = i, abs(k[y + dy, x + dx])
    dy, dx = _RING[best_i]                    # best_i == 0 -> the C's pos_incr[0] NW quirk
    denom = k[y, x] - k[y + dy, x + dx]
    if denom == 0.0 or not np.isfinite(denom):
        return float(m[y, x])                 # recorded divergence: unreachable for a
    ratio = abs(k[y, x] / denom)              # registered maximum; guards NaN, not the C
    if not (0.0 <= ratio <= 1.0):
        return float(m[y, x])
    result = get_m2(m[y - dy, x - dx], m[y, x], m[y + dy, x + dx],
                    m[y + 2 * dy, x + 2 * dx], ratio)
    if result < 0 or result > 2.0 * m[y, x]:
        return float(m[y, x])
    return result


def _neighbors_of(y: int, x: int):
    """The walk's neighbor coordinates in the verbatim priority order."""
    for k in range(8):
        yield y + J_POS_ARRAY[k], x + I_POS_ARRAY[k]


def is_line_end(mask, y: int, x: int) -> bool:
    """C ``_is_line_end_``: <= 1 neighbor -> end; > 2 -> interior; exactly 2 -> an end iff
    the two neighbors are 4-adjacent to EACH OTHER (the kink rule). ``mask`` is the CURRENT
    unconsumed-extrema state, exactly as the C consults ``cur_ext_array``."""
    mk = np.asarray(mask)
    ly, lx = mk.shape
    found = []
    for ny, nx in _neighbors_of(y, x):
        if 0 <= ny < ly and 0 <= nx < lx and mk[ny, nx]:
            found.append((ny, nx))
            if len(found) > 2:
                return False
    if len(found) <= 1:
        return True
    (j1, i1), (j2, i2) = found
    return (abs(i1 - i2) == 0 and abs(j1 - j2) == 1) or \
           (abs(i1 - i2) == 1 and abs(j1 - j2) == 0)


def _walk(ext, seed_y: int, seed_x: int):
    """C ``_seek_neighbors_`` made iterative: single successor per step in the verbatim
    priority, PREPENDING (the seed ends up last), consuming from ``ext`` as it goes."""
    ly, lx = ext.shape
    pts = [(seed_y, seed_x)]
    ext[seed_y, seed_x] = False
    cy, cx = seed_y, seed_x
    while True:
        for ny, nx in _neighbors_of(cy, cx):
            if 0 <= ny < ly and 0 <= nx < lx and ext[ny, nx]:
                ext[ny, nx] = False
                pts.insert(0, (ny, nx))
                cy, cx = ny, nx
                break
        else:
            return pts


def search_lines(mask):
    """C ``search_lines``: ``[(points, closed), ...]`` over the extrema in ``mask``.

    ``points`` are (row, col) in the walk's own order (seed last -- the prepend);
    ``closed`` is the C's LINE_CLOSED: size > 3 with Chebyshev-adjacent ends."""
    ext = np.array(mask, dtype=bool, copy=True)
    lines = []

    def _find(ends_only: bool) -> int:
        found = 0
        for y, x in zip(*np.nonzero(ext)):     # raster order, the C's pos loop
            y, x = int(y), int(x)
            if not ext[y, x]:
                continue                        # consumed by an earlier walk this pass
            if ends_only and not is_line_end(ext, y, x):
                continue
            pts = _walk(ext, y, x)
            (fy, fx), (ty, tx) = pts[0], pts[-1]
            closed = (len(pts) > 3 and abs(fy - ty) <= 1 and abs(fx - tx) <= 1)
            lines.append((pts, closed))
            found += 1
        return found

    while _find(True):
        pass
    _find(False)
    return lines


def _near_pos_many(kapa, ys, xs):
    """Vectorized ``_get_near_pos_`` + crossing ratio for arrays of points: returns
    ``(dy, dx, ratio, fired)`` where (dy, dx) is the chosen cardinal (the strongest
    opposite-|kapa| crossing, the C's tie rule) and ``ratio`` the linear zero-crossing
    fraction toward it. Points that fired no cardinal get the C's pos_incr[0] NW quirk with
    ``fired = False`` (unreachable for registered maxima; kept for faithfulness)."""
    k = np.asarray(kapa, dtype=np.float64)
    k0 = k[ys, xs]
    best = np.full(k0.shape, -1.0)
    bi = np.zeros(k0.shape, dtype=np.int64)          # 0 = the NW quirk
    for i in (1, 3, 5, 7):
        dy, dx = _RING[i]
        kn = k[ys + dy, xs + dx]
        fire = (k0 * kn < 0) & (np.abs(k0) < np.abs(kn)) & (np.abs(kn) > best)
        best = np.where(fire, np.abs(kn), best)
        bi = np.where(fire, i, bi)
    ring = np.asarray(_RING[:8])
    dy = ring[bi, 0]
    dx = ring[bi, 1]
    kn = k[ys + dy, xs + dx]
    denom = k0 - kn
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(np.where(denom != 0, k0 / np.where(denom == 0, 1.0, denom), np.nan))
    return dy, dx, ratio, bi != 0


def interpolated_modulus_many(mod, kapa, ys, xs):
    """Vectorized ``_get_interpolated_modulus_`` v1 over point arrays -- identical fallbacks
    to the scalar port (border, non-finite/out-of-range ratio, cubic sentinel, 2*mod guard)."""
    m = np.asarray(mod, dtype=np.float64)
    ly, lx = m.shape
    ys = np.asarray(ys, dtype=np.int64)
    xs = np.asarray(xs, dtype=np.int64)
    out = m[ys, xs].copy()
    interior = (xs >= 2) & (xs < lx - 2) & (ys >= 2) & (ys < ly - 2)
    if not interior.any():
        return out
    yi, xi = ys[interior], xs[interior]
    dy, dx, s, _fired = _near_pos_many(kapa, yi, xi)
    ok = np.isfinite(s) & (s >= 0.0) & (s <= 1.0)
    y_1 = m[yi - dy, xi - dx]
    y0 = m[yi, xi]
    y1 = m[yi + dy, xi + dx]
    y2 = m[yi + 2 * dy, xi + 2 * dx]
    k0 = y0
    k1 = (-2.0 * y_1 + 6.0 * y1 - 3.0 * y0 - y2) / 6.0
    k2 = (y_1 + y1 - 2.0 * y0) / 2.0
    k3 = -k1 + k2 - y_1 + k0
    val = k3 * s ** 3 + k2 * s ** 2 + k1 * s + k0
    good = ok & (val >= 0) & (val <= 2.0 * y0)
    res = np.where(good, val, y0)
    out[interior] = res
    return out


def follow_extrema_scale_exact(mod, arg, kapa, kapap, *, thresh: float = 1e-3,
                               invalid=None, radius: int = 0) -> dict:
    """The exact-port follow detector in ``_nms_extrema_scale``'s output schema.

    Detection/value = the parity-proven C ports above; ``line_id`` AND the display ordering
    come from the C's own ``search_lines`` walk (``_xs_runs`` index arrays in walk order,
    ``_xs_closed`` the LINE_CLOSED flags -- the backend lifts them into ``_hline_runs`` /
    ``_hline_closed`` so both views draw xsmurf's own lines, closed rings closing).
    ``x_sub``/``y_sub`` are the kapa crossing point along the chosen cardinal (offset =
    ratio) -- a DynamiX display channel (the C stores integer pos only), used by the vector
    view; the raster view stays on the integer support. ``invalid``/``radius`` reproduce the
    NaN-distrust contract the NMS applies."""
    m = np.asarray(mod, dtype=np.float64)
    a = np.asarray(arg, dtype=np.float64)
    k = np.asarray(kapa, dtype=np.float64)
    kp = np.asarray(kapap, dtype=np.float64)
    ys, xs = follow_contour(k, kp)
    if thresh and ys.size:
        keep = m[ys, xs] >= float(thresh) * float(m.max())
        ys, xs = ys[keep], xs[keep]
    if invalid is not None and np.asarray(invalid).any() and ys.size:
        from scipy.ndimage import binary_dilation
        inv = np.asarray(invalid, dtype=bool)
        dilated = (binary_dilation(inv, structure=np.ones((3, 3)), iterations=radius)
                   if radius > 0 else inv)
        keep = ~dilated[ys, xs]
        ys, xs = ys[keep], xs[keep]

    if not ys.size:
        z = np.zeros(0)
        return {"x": z.astype(np.int64), "y": z.astype(np.int64), "mod": z, "arg": z,
                "line_id": z.astype(np.int64), "x_sub": z, "y_sub": z,
                "_xs_runs": [], "_xs_closed": []}

    dy, dx, ratio, _fired = _near_pos_many(k, ys, xs)
    off = np.where(np.isfinite(ratio), np.clip(ratio, 0.0, 1.0), 0.0)
    mods = interpolated_modulus_many(m, k, ys, xs)

    mask = np.zeros(m.shape, dtype=bool)
    mask[ys, xs] = True
    lines = search_lines(mask)
    idx_of = {(int(yy), int(xx)): i for i, (yy, xx) in enumerate(zip(ys, xs))}
    line_id = np.full(ys.shape, -1, dtype=np.int64)
    runs, closed = [], []
    lid = 0
    for pts, is_closed in lines:
        if len(pts) >= 2:
            run = np.array([idx_of[p] for p in pts], dtype=np.int64)
            runs.append(run)
            closed.append(bool(is_closed))
            line_id[run] = lid
            lid += 1
    return {
        "x": xs.astype(np.int64), "y": ys.astype(np.int64),
        "mod": mods, "arg": a[ys, xs],
        "line_id": line_id,
        "x_sub": xs + off * dx, "y_sub": ys + off * dy,
        "_xs_runs": runs, "_xs_closed": closed,
    }


def runs_for_layer(layer, shape):
    """``(runs, closed)`` for an extrema layer, via the C's own ``search_lines`` walk.

    Uses the layer's own ``_xs_runs``/``_xs_closed`` when present (a fresh
    ``follow_extrema_scale_exact`` result); rebuilds them from the integer support when the
    layer came back through a stage-cache npz round-trip (the CSR serialization keeps only
    the fixed array keys). Rebuilding is bit-identical: the walk is a pure function of the
    support mask."""
    if "_xs_runs" in layer:
        return layer["_xs_runs"], layer["_xs_closed"]
    ys = np.asarray(layer["y"], dtype=np.int64)
    xs = np.asarray(layer["x"], dtype=np.int64)
    mask = np.zeros(tuple(shape)[:2], dtype=bool)
    mask[ys, xs] = True
    idx_of = {(int(yy), int(xx)): i for i, (yy, xx) in enumerate(zip(ys, xs))}
    runs, closed = [], []
    for pts, is_closed in search_lines(mask):
        if len(pts) >= 2:
            runs.append(np.array([idx_of[p] for p in pts], dtype=np.int64))
            closed.append(bool(is_closed))
    return runs, closed


def kapa_from_field(snap):
    """``(mod, arg, kapa, kapap)`` for one SMOOTHED snapshot (a cdf Re channel, a pm
    diffused field), from finite-difference derivative stacks.

    The DynamiX extension feeding the exact detector: xsmurf's ``follow`` command takes four
    IMAGES and never asks where they came from ("Needs 4 images…"), so running it over a
    diffusion snapshot's own derivatives is the command's design, not a new method. The
    kapa/kapap combinations are gkapa/gkapap verbatim (dynamix.core.follow2d's ported
    formulas); only the derivative SOURCE differs -- np.gradient chains instead of the CWT
    engine's spectral stacks (a Gaussian-CWT modulus at the snapshot's own scale IS this
    field's gradient modulus, by the cdf Ricker identity)."""
    I = np.asarray(snap, dtype=np.float64)
    dy, dx = np.gradient(I)
    dyy, dxy = np.gradient(dy)
    _dxy2, dxx = np.gradient(dx)
    dyyy, dxyy = np.gradient(dyy)[0], np.gradient(dxy)[0]
    dxxy, dxxx = np.gradient(dxy)[1], np.gradient(dxx)[1]
    kapa = 2.0 * dx * dx * dxx + 4.0 * dxy * dx * dy + 2.0 * dy * dy * dyy
    kapap = (dx * dx * (4.0 * (dxx ** 2 + dxy ** 2) + 2.0 * dx * dxxx + 6.0 * dy * dxxy)
             + dy * dy * (4.0 * (dyy ** 2 + dxy ** 2) + 2.0 * dy * dyyy + 6.0 * dx * dxyy)
             + 8.0 * dx * dy * dxy * (dxx + dyy))
    return np.hypot(dx, dy), np.arctan2(dy, dx), kapa, kapap


def follow_layer_from_snapshot(snap, floor: float):
    """``(layer, runs, closed)`` -- the exact follow detector over one diffusion snapshot.

    ``floor`` is the devices' fraction-of-max modulus cut (grad_nms's own semantic), which
    is also what the exact adapter's ``thresh`` implements. The layer carries the native
    subpixel channels; the runs/closed pair is the search_lines walk for the result-level
    ``_hline_runs``/``_hline_closed`` stamps."""
    mod, arg, kapa, kapap = kapa_from_field(snap)
    layer = follow_extrema_scale_exact(mod, arg, kapa, kapap, thresh=float(floor))
    runs = layer.pop("_xs_runs")
    closed = layer.pop("_xs_closed")
    return layer, runs, closed
