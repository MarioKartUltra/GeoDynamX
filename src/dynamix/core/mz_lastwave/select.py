# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Reconstruction from selected maxima: which extrema of each level constrain ``e2recons``.

Each level l takes a state: ``all`` (its maxima), ``own`` (the maxima the rack's filters keep at
l), ``near`` (its own maxima within r px of a maximum kept at a source level s; with the α check,
only those whose modulus fits the source chain's decay), ``coder`` (the kept positions of s carrying
level l's own transform values there, Mallat & Zhong's coding mode, §IX), ``predict`` (the kept
positions and arguments of s with the modulus the decay predicts) or ``open`` (no constraints). A
source is an ``all``, ``own`` or ``near`` level.

Selections are made on the field's own grid, where the display draws the extrema, and reach the
working field as removals: a working-field maximum takes the verdict of its primary twin (itself,
or in mirror mode its reflection i -> -i mod 2N, under which LastWave's maxima are whole-sample
symmetric from level 2 up; at level 1, whose two gradient components are staggered, the nearest
primary maximum within 1 px answers), and a maximum without a twin is kept, so an all-pass
selection gives back the unselected extrema exactly.

Moduli are ``extrema2``'s stored normalisation, in which a singularity of Hölder exponent α gives
maxima growing as 2**(l α) from level 2 up (a step 0, a line -1, a point -2). Level 1 departs from
that law through its staggered two-tap gradient and its detection, by a factor that depends on the
full direction of the edge's normal (0.86 to 1.37 on straight binary steps; a normal and its
opposite differ), so a prediction at or from level 1 carries that factor and the α check's
tolerance has to cover it.
"""
from __future__ import annotations

import json

import numpy as np

STATES = ("all", "own", "near", "coder", "predict", "open")
_MASKED = ("all", "own", "near", "open")
_BORROW = ("coder", "predict")

def default_source(J: int) -> int:
    return 2 if J >= 2 else 1


def parse_levels(text: str, J: int) -> dict:
    """``recon_levels`` as ``{"levels": {l: {"state", "source"}}, "filters": {l: {key: params}}}``
    for l = 1..J. An empty text is every level ``own``; a level the text leaves out is ``own``
    with the default source. Raises ValueError on an unknown state or a source outside 1..J."""
    data = json.loads(text) if text else {}
    raw = data.get("levels", {})
    levels = {}
    for l in range(1, J + 1):
        entry = raw.get(str(l), {})
        state = entry.get("state", "own")
        if state not in STATES:
            raise ValueError(f"level {l}: unknown state {state!r}; choices: {STATES}")
        source = int(entry.get("source", default_source(J)))
        if state in ("near",) + _BORROW and not 1 <= source <= J:
            raise ValueError(f"level {l} {state}s level {source}; the levels are 1..{J}")
        levels[l] = {"state": state, "source": source}
    filters = {int(k): dict(v) for k, v in data.get("filters", {}).items() if 1 <= int(k) <= J}
    return {"levels": levels, "filters": filters}


def dump_levels(table: dict) -> str:
    """The JSON text of a table :func:`parse_levels` reads back (sorted, compact)."""
    return json.dumps({"levels": {str(l): e for l, e in table["levels"].items()},
                       "filters": {str(l): f for l, f in table.get("filters", {}).items()}},
                      sort_keys=True, separators=(",", ":"))


def _offsets(r: int) -> list:
    """``(dy, dx)`` within Chebyshev radius ``r``, nearest first, the 4-neighbours of a ring before
    its diagonals."""
    offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    return sorted(offs, key=lambda o: (max(abs(o[0]), abs(o[1])), abs(o[0]) + abs(o[1])))


def _index_grid(y, x, shape) -> np.ndarray:
    grid = np.full(shape, -1, np.int64)
    grid[np.asarray(y, np.int64), np.asarray(x, np.int64)] = np.arange(np.size(y))
    return grid


def nearest(y, x, grid, r: int, accept=None) -> np.ndarray:
    """For each point ``(y, x)`` the index ``grid`` holds at the nearest cell within Chebyshev
    radius ``r`` (:func:`_offsets` order), or -1 where every such cell holds -1. ``accept(points,
    hits)`` may refuse a hit (a bool per pair), and the search goes on past it."""
    y = np.asarray(y, np.int64)
    x = np.asarray(x, np.int64)
    ny, nx = grid.shape
    out = np.full(y.size, -1, np.int64)
    for dy, dx in _offsets(int(r)):
        todo = np.flatnonzero(out < 0)
        if not todo.size:
            break
        yy, xx = y[todo] + dy, x[todo] + dx
        ok = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
        hit = np.full(todo.size, -1, np.int64)
        hit[ok] = grid[yy[ok], xx[ok]]
        if accept is not None:
            got = np.flatnonzero(hit >= 0)
            hit[got[~accept(todo[got], hit[got])]] = -1
        out[todo] = hit
    return out


def _line_median(lid: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Every point of a line gets the median α of the line's fitted points (a line with none stays
    NaN); a singleton (``line_id == -1``) keeps its own."""
    out = alpha.copy()
    lab = np.flatnonzero(lid >= 0)
    if not lab.size:
        return out
    order = np.argsort(lid[lab], kind="stable")
    ids, vals = lid[lab][order], alpha[lab][order]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    meds = [np.nanmedian(v) if np.isfinite(v).any() else np.nan
            for v in np.split(vals, starts[1:])]
    per = np.repeat(np.asarray(meds, float), np.diff(np.r_[starts, ids.size]))
    back = np.empty(ids.size)
    back[order] = per
    out[lab] = back
    return out


def _link_up(extrema: list, shape, i: int) -> np.ndarray:
    """The maximum of level i + 2 each maximum of level i + 1 propagates to (Mallat & Zhong's rule:
    the nearest with a gradient of the same sign, here within 2**i px, as far as the maxima of a
    line or a point move out between the two scales), or -1."""
    src, dst = extrema[i], extrema[i + 1]
    grid = _index_grid(dst["y"], dst["x"], shape)
    a_src = np.asarray(src["arg"], float)
    a_dst = np.asarray(dst["arg"], float)

    def same_sign(points, hits):
        return np.cos(a_src[points] - a_dst[hits]) > 0

    return nearest(src["y"], src["x"], grid, 2 ** i, accept=same_sign)


def chain_alpha(extrema: list, shape) -> list:
    """α of every maximum, per level: each maximum propagates up level by level (:func:`_link_up`)
    to level J; α is the least-squares slope of log2 of the stored modulus against the level over
    the linked levels from 2 up (NaN with fewer than two), then the median over the maximum's
    line."""
    J = len(extrema)
    ys = [np.asarray(e["y"], np.int64) for e in extrema]
    up = [_link_up(extrema, shape, i) for i in range(J - 1)]
    tiny = np.finfo(float).tiny
    logm = [np.log2(np.maximum(np.asarray(e["mod"], float), tiny)) for e in extrema]
    level = np.arange(1, J + 1, dtype=float)
    out = []
    for i, e in enumerate(extrema):
        n = ys[i].size
        vals = np.full((n, J), np.nan)
        if i >= 1:
            vals[:, i] = logm[i]
        cur = np.arange(n)
        for k in range(i, J - 1):
            ok = cur >= 0
            nxt = np.full(n, -1, np.int64)
            nxt[ok] = up[k][cur[ok]]
            cur = nxt
            ok = cur >= 0
            vals[ok, k + 1] = logm[k + 1][cur[ok]]
        vals[:, 0] = np.nan                                   # level 1 never enters the fit
        w = np.isfinite(vals)
        cnt = w.sum(axis=1)
        safe = np.maximum(cnt, 1)
        jm = (w * level).sum(axis=1) / safe
        vm = np.where(w, vals, 0.0).sum(axis=1) / safe
        dj = np.where(w, level - jm[:, None], 0.0)
        dv = np.where(w, vals - vm[:, None], 0.0)
        den = (dj * dj).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            alpha = np.where((cnt >= 2) & (den > 0), (dj * dv).sum(axis=1) / den, np.nan)
        if i >= 1:
            alpha = _inherit(alpha, out[i - 1], up[i - 1])
        lid = np.asarray(e.get("line_id", np.full(n, -1)), np.int64)
        out.append(_line_median(lid, alpha))
    return out


def _inherit(alpha: np.ndarray, below: np.ndarray, link: np.ndarray) -> np.ndarray:
    """An unfitted maximum (the coarsest level has no level above it) takes the median α of the
    maxima one level down that propagate to it."""
    miss = ~np.isfinite(alpha)
    ok = (link >= 0) & np.isfinite(below)
    if not (miss.any() and ok.any()):
        return alpha
    out = alpha.copy()
    tgt, val = link[ok], below[ok]
    order = np.argsort(tgt, kind="stable")
    tgt, val = tgt[order], val[order]
    starts = np.flatnonzero(np.r_[True, tgt[1:] != tgt[:-1]])
    meds = np.asarray([np.median(v) for v in np.split(val, starts[1:])])
    fill = np.full(alpha.size, np.nan)
    fill[tgt[starts]] = meds
    out[miss] = fill[miss]
    return out


def _decay_log2(l: int, s: int, alpha) -> np.ndarray:
    """log2 of the modulus ratio level l over level s along a chain of exponent ``alpha``."""
    return alpha * (l - s)


def _alpha_of(e: dict, index, fallback: float) -> np.ndarray:
    a = np.asarray(e.get("alpha", np.full(np.size(e["x"]), np.nan)), float)[index]
    return np.where(np.isfinite(a), a, fallback)


def _near(dst: dict, src: dict, src_keep, shape, l: int, s: int, *, radius, alpha_check,
          alpha_tol, alpha_fallback) -> np.ndarray:
    kept = np.flatnonzero(src_keep)
    grid = _index_grid(np.asarray(src["y"])[kept], np.asarray(src["x"])[kept], shape)
    q = nearest(dst["y"], dst["x"], grid, radius)
    keep = q >= 0
    if alpha_check and keep.any():
        qi = kept[q[keep]]
        tiny = np.finfo(float).tiny
        pred = (np.log2(np.maximum(np.asarray(src["mod"], float)[qi], tiny))
                + _decay_log2(l, s, _alpha_of(src, qi, alpha_fallback)))
        got = np.log2(np.maximum(np.asarray(dst["mod"], float)[keep], tiny))
        keep[np.flatnonzero(keep)] = np.abs(got - pred) <= alpha_tol
    return keep


def primary_selection(extrema: list, shape, table: dict, keep_of, *, radius: int = 1,
                      alpha_check: bool = False, alpha_tol: float = 0.5,
                      alpha_fallback: float = 0.0, only=None) -> list:
    """Per level (index l - 1): a bool mask over the level's extrema for the ``all``, ``own``,
    ``near`` and ``open`` levels, ``None`` for the ``coder`` and ``predict`` ones. ``keep_of(l)``
    gives what the rack's filters keep at level l, called only for the ``own`` levels resolved (and
    for a ``near`` level's source when that is the level itself). ``only`` limits the work to those
    levels and their sources. Raises ValueError for a borrowing source, a level borrowing from
    itself, or a cycle of ``near`` levels."""
    J = len(extrema)
    levels = table["levels"]
    out: list = [None] * J
    done: set = set()

    def resolve(l, stack):
        if l in done:
            return out[l - 1]
        state, s = levels[l]["state"], levels[l]["source"]
        n = np.size(extrema[l - 1]["x"])
        if state == "all":
            m = np.ones(n, bool)
        elif state == "open":
            m = np.zeros(n, bool)
        elif state == "own":
            m = np.asarray(keep_of(l), bool)
        elif state == "near":
            if s == l:
                src_keep = np.asarray(keep_of(l), bool)
            else:
                if l in stack:
                    raise ValueError(f"levels {sorted(stack | {l})} are near each other in a cycle")
                if levels[s]["state"] in _BORROW:
                    raise ValueError(f"level {l} is near level {s}, which borrows its positions; "
                                     f"choose a level with a selection of its own")
                src_keep = resolve(s, stack | {l})
            m = _near(extrema[l - 1], extrema[s - 1], src_keep, shape, l, s, radius=radius,
                      alpha_check=alpha_check, alpha_tol=alpha_tol,
                      alpha_fallback=alpha_fallback)
        else:
            if s == l:
                raise ValueError(f"level {l} cannot take its {state} positions from itself")
            if levels[s]["state"] in _BORROW:
                raise ValueError(f"level {l} borrows from level {s}, which borrows itself; "
                                 f"choose a level with a selection of its own")
            resolve(s, stack | {l})
            m = None
        out[l - 1] = m
        done.add(l)
        return m

    for l in (range(1, J + 1) if only is None else only):
        resolve(l, frozenset())
    return out


def _mask_layer(e: dict, keep) -> dict:
    n = np.size(e["x"])
    return {k: (v[keep] if hasattr(v, "shape") and v.shape[:1] == (n,) else v)
            for k, v in e.items()}


def shown_level(extrema: list, table: dict, primary: list, l: int, *,
                alpha_fallback: float = 0.0) -> dict:
    """The extrema layer drawn at level l: the kept maxima of a masked level; for ``coder`` and
    ``predict`` the source's kept maxima at their positions, carrying the source's modulus
    (``coder``, whose own values need the transform) or the predicted one (``predict``)."""
    if primary[l - 1] is not None:
        return _mask_layer(extrema[l - 1], primary[l - 1])
    state, s = table["levels"][l]["state"], table["levels"][l]["source"]
    src = extrema[s - 1]
    idx = np.flatnonzero(primary[s - 1])
    lay = _mask_layer(src, primary[s - 1])
    if state == "predict":
        lay["mod"] = (np.asarray(src["mod"], float)[idx]
                      * 2.0 ** _decay_log2(l, s, _alpha_of(src, idx, alpha_fallback)))
    return lay


def _twins(wy, wx, shape, full_shape):
    """``(ty, tx, has)``: the primary twin of each working-field point (the reflection
    i -> -i mod 2N past the primary quadrant) and whether it lies on the field's grid."""
    ny, nx = shape
    Y, X = full_shape
    ty = np.where(wy < ny, wy, (-wy) % Y)
    tx = np.where(wx < nx, wx, (-wx) % X)
    return ty, tx, (ty < ny) & (tx < nx)


def _twin_index(wy, wx, prim: dict, shape, full_shape) -> np.ndarray:
    """Per working-field point, the index of its primary twin among ``prim``'s maxima (the
    nearest within 1 px when the twin itself is not one), or -1."""
    ty, tx, has = _twins(wy, wx, shape, full_shape)
    grid = _index_grid(prim["y"], prim["x"], shape)
    out = np.full(wy.size, -1, np.int64)
    out[has] = grid[ty[has], tx[has]]
    miss = np.flatnonzero(has & (out < 0))
    if miss.size:
        out[miss] = nearest(ty[miss], tx[miss], grid, 1)
    return out


def working_mask(wmask: np.ndarray, prim: dict, keep, shape) -> np.ndarray:
    """The maxima of the working-field mask ``wmask`` the primary verdict ``keep`` (over
    ``prim``, the level's extrema on the field's grid) keeps: removals only, a maximum without a
    primary twin stays."""
    wy, wx = np.nonzero(wmask)
    twin = _twin_index(wy, wx, prim, shape, wmask.shape)
    drop = np.zeros(wy.size, bool)
    has = twin >= 0
    drop[has] = ~np.asarray(keep, bool)[twin[has]]
    out = wmask.copy()
    out[wy[drop], wx[drop]] = False
    return out


def working_extrep(ex, extrema: list, shape, table: dict, primary: list, *, transform=None,
                   alpha_fallback: float = 0.0):
    """The ``Extrep`` ``e2recons`` reconstructs from: every masked level's working-field maxima the
    selection keeps (:func:`working_mask`), and at a ``coder`` or ``predict`` level the working-field
    positions its source keeps, carrying the level's own transform values there (``transform``, the
    working field's ``Transform``) or the predicted modulus with the source's argument."""
    from dynamix.core.mz_lastwave.detect import Extrep, _g1_factor, _polar
    from dynamix.core.mz_lastwave.transform import fact

    J = ex.J
    mask, mag, arg = [None] * (J + 1), [None] * (J + 1), [None] * (J + 1)
    for l in range(1, J + 1):
        if primary[l - 1] is not None:
            m = working_mask(ex.mask[l], extrema[l - 1], primary[l - 1], shape)
            mask[l], mag[l], arg[l] = m, np.where(m, ex.mag[l], 0.0), np.where(m, ex.arg[l], 0.0)
    for l in range(1, J + 1):
        if primary[l - 1] is not None:
            continue
        state, s = table["levels"][l]["state"], table["levels"][l]["source"]
        P = mask[s].copy()
        if state == "coder":
            m, a = _polar(np.ascontiguousarray(transform.Wx_full[l]),
                          np.ascontiguousarray(transform.Wy_full[l]))
            mag[l] = np.where(P, (m * _g1_factor(l)) / fact(l), 0.0)
            arg[l] = np.where(P, a, 0.0)
        else:
            wy, wx = np.nonzero(P)
            twin = _twin_index(wy, wx, extrema[s - 1], shape, P.shape)
            alpha = np.full(wy.size, float(alpha_fallback))
            has = twin >= 0
            alpha[has] = _alpha_of(extrema[s - 1], twin[has], alpha_fallback)
            m = np.zeros(P.shape)
            m[wy, wx] = ex.mag[s][wy, wx] * 2.0 ** _decay_log2(l, s, alpha)
            mag[l] = m
            arg[l] = np.where(P, ex.arg[s], 0.0)
        mask[l] = P
    return Extrep(J, mask, mag, arg)


def level_filters(text: str, level: int) -> dict:
    """``{step key: params}`` stored for ``level`` in ``recon_levels``' text (none: ``{}``)."""
    data = json.loads(text) if text else {}
    return {k: dict(v) for k, v in data.get("filters", {}).get(str(level), {}).items()}
