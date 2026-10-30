# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Morse/Cerf tracking across scale: the merge tree of the modulus at each scale and the mutual
steepest-ascent links between neighbouring scales.

The WTMM modulus M(x, y; a) is a one-parameter family of functions of (x, y). In a generic
one-parameter family of Morse functions the critical points move on smooth paths in (x, y, log a)
and change only at folds, where a maximum and a saddle are born together or die together (Cerf 1970,
Publ. Math. IHES 39, 5; for Gaussian scale space, Damon 1995, J. Differential Equations 115, 368,
and Lifshitz and Pizer 1990, IEEE PAMI 12, 529).

Per scale, on the domain D = {M finite, M > 0, M >= thresh max M} (:func:`domain`), ties in M
broken by pixel index:

* the maxima are the 8-neighbour local maxima of M on D;
* the merge tree of the superlevel sets {M >= t} (Carr, Snoeyink and Axen 2003, Comput. Geom. 24,
  75): pixels taken in decreasing M and joined by 8-connected union-find; where a pixel joins
  components, every one but the component holding the highest maximum dies (the elder rule), its
  maximum paired with that pixel, its merge saddle s, with persistence M(max) - M(s) and parent
  peak the highest maximum of the merged component (Edelsbrunner, Letscher and Zomorodian 2002,
  Discrete Comput. Geom. 28, 511); a component's highest maximum pairs with the component's lowest
  pixel. Invariant: the number of components of {M >= t} on D is the number of maxima with
  M >= t minus the number of merge saddles with M(s) >= t;
* the basin of a maximum is the set of pixels whose steepest 8-neighbour ascent ends on it.

Across scales (vineyards: Cohen-Steiner, Edelsbrunner and Morozov 2006, Proc. SoCG, 119), for
tracked points P_i at a_i (pixels: the maxima themselves, or the WTMM single maxima) with values
v_i (:func:`track`):

* a basin's owner is its point with the largest value, the lowest index on a tie;
* a point p at a_i climbs M(., a_{i+1}) to a maximum, whose owner q is p's candidate;
* the link p -> q holds when q climbs M(., a_i) into p's basin (mutual) and p has the largest value
  among the mutual claimants of q;
* a point without a link ends its line: an exit when its pixel leaves D at a_{i+1}, an
  annihilation when its candidate continues another line, an orphan death otherwise;
* a point at a_{i+1} without a link starts a line: an entry when its pixel lies outside D at a_i,
  an orphan birth when some point chose it, a creation otherwise. Per step
  n_{i+1} = n_i - deaths + births.

When the points are the maxima themselves every basin holds one point, and the link is the mutual
steepest ascent of the maxima. A line's basin successor is the line of the candidate its last point
chose; its elder-rule peak the line that owns, at its last scale, the basin of the parent peak of
its own basin's maximum (of its own basin, for a point that does not own it).

Every pixel and point loop is an ``njit(cache=True)`` kernel compiled by :func:`kernels` on first
use; each has a plain-numpy twin (``*_twin``) the tests hold it to.
"""
from __future__ import annotations

import numpy as np

#: Event codes: a line starts at the finest scale or by an entry, a creation or an orphan birth,
#: and survives to the coarsest scale or ends by an annihilation, an orphan death or an exit.
BIRTHS = ("finest", "entry", "creation", "orphan")
DEATHS = ("survivor", "annihilation", "orphan", "exit")

_KERNELS: dict = {}


def _union_find_py(order, R, ny, nx):
    """Elder-rule merge tree of the pixels ``order`` (decreasing M) on the rank image ``R``
    (higher M, higher rank; -1 off the domain), 8-connected. Per pixel: whether it is a maximum;
    for a maximum that merges, its merge saddle and the parent peak it merges into (-1 otherwise);
    for every maximum, the pixel it pairs with (the saddle, or its component's lowest pixel)."""
    n = ny * nx
    uf = np.full(n, -1, np.int64)
    top = np.full(n, -1, np.int64)
    low = np.full(n, -1, np.int64)
    is_max = np.zeros(n, np.bool_)
    saddle = np.full(n, -1, np.int64)
    peak = np.full(n, -1, np.int64)
    base = np.full(n, -1, np.int64)
    roots = np.empty(8, np.int64)
    for k in range(order.shape[0]):
        p = order[k]
        y = p // nx
        x = p % nx
        uf[p] = p
        top[p] = p
        nr = 0
        for dy in range(-1, 2):
            yy = y + dy
            if yy < 0 or yy >= ny:
                continue
            for dx in range(-1, 2):
                xx = x + dx
                if (dy == 0 and dx == 0) or xx < 0 or xx >= nx:
                    continue
                q = yy * nx + xx
                if uf[q] < 0:
                    continue
                r = q
                while uf[r] != r:
                    r = uf[r]
                while uf[q] != r:
                    t = uf[q]
                    uf[q] = r
                    q = t
                seen = False
                for t in range(nr):
                    if roots[t] == r:
                        seen = True
                if not seen:
                    roots[nr] = r
                    nr += 1
        if nr == 0:
            is_max[p] = True
            low[p] = p
            continue
        w = roots[0]
        for t in range(1, nr):
            if R[top[roots[t]]] > R[top[w]]:
                w = roots[t]
        for t in range(nr):
            r = roots[t]
            if r != w:
                saddle[top[r]] = p
                peak[top[r]] = top[w]
                base[top[r]] = p
                uf[r] = w
        uf[p] = w
        low[w] = p
    for k in range(order.shape[0]):
        p = order[k]
        if is_max[p] and saddle[p] < 0:
            r = p
            while uf[r] != r:
                r = uf[r]
            base[p] = low[r]
    return is_max, saddle, peak, base


def _climb_py(R, ny, nx, starts):
    """Steepest 8-neighbour ascent on the rank image ``R`` (-1 off the domain): the maximum each
    start pixel reaches, -1 for a start off the domain or a start of -1."""
    out = np.empty(starts.shape[0], np.int64)
    for k in range(starts.shape[0]):
        p = starts[k]
        if p < 0 or R[p] < 0:
            out[k] = -1
            continue
        while True:
            y = p // nx
            x = p % nx
            best = p
            for dy in range(-1, 2):
                yy = y + dy
                if yy < 0 or yy >= ny:
                    continue
                for dx in range(-1, 2):
                    xx = x + dx
                    if xx >= 0 and xx < nx and R[yy * nx + xx] > R[best]:
                        best = yy * nx + xx
            if best == p:
                break
            p = best
        out[k] = p
    return out


def _owners_py(basin, values, n_pix):
    """Per pixel, the point owning the basin of the maximum at that pixel: the largest value among
    the points whose ``basin`` it is, the lowest index on a tie; -1 elsewhere."""
    owner = np.full(n_pix, -1, np.int64)
    for j in range(basin.shape[0]):
        m = basin[j]
        if m >= 0 and (owner[m] < 0 or values[j] > values[owner[m]]):
            owner[m] = j
    return owner


def _compete_py(cand, basin_a, back, values, n_b):
    """The link of each finer point (-1 for none): its candidate, when the candidate climbs back
    into its basin and no other such claimant of that candidate has a larger value (the lower
    index on a tie)."""
    winner = np.full(n_b, -1, np.int64)
    for i in range(cand.shape[0]):
        j = cand[i]
        if j < 0 or basin_a[i] < 0 or back[j] != basin_a[i]:
            continue
        w = winner[j]
        if w < 0 or values[i] > values[w]:
            winner[j] = i
    link = np.full(cand.shape[0], -1, np.int64)
    for j in range(n_b):
        if winner[j] >= 0:
            link[winner[j]] = j
    return link


def kernels() -> dict:
    """The kernels, ``njit(cache=True)``-compiled on first use."""
    if not _KERNELS:
        from numba import njit

        for name, fn in (("union_find", _union_find_py), ("climb", _climb_py),
                         ("owners", _owners_py), ("compete", _compete_py)):
            _KERNELS[name] = njit(cache=True)(fn)
    return _KERNELS


def domain(M, thresh: float):
    """D = {M finite, M > 0, M >= thresh max M}: the pixels the WTMM's own threshold keeps."""
    M = np.asarray(M)
    finite = np.isfinite(M)
    top = float(M[finite].max()) if finite.any() else 0.0
    with np.errstate(invalid="ignore"):
        return finite & (M > 0) & (M >= float(thresh) * top)


def merge_tree(M, dom) -> dict:
    """The merge tree of ``M`` on ``dom``: ``R`` (the rank image, -1 off the domain), and per
    maximum, in decreasing M: ``pix``, ``val``, ``sad`` (merge saddle pixel, -1 for a component's
    highest maximum), ``sadv`` (the paired value: the saddle's, or the component's lowest),
    ``pers`` (``val - sadv``) and ``par`` (parent peak pixel, -1 without a saddle)."""
    M = np.asarray(M, dtype=np.float64)
    flat = np.flatnonzero(dom)
    order = flat[np.argsort(-M.ravel()[flat], kind="stable")]
    R = np.full(M.size, -1, np.int64)
    R[order] = np.arange(order.size - 1, -1, -1)
    is_max, saddle, peak, base = kernels()["union_find"](order, R, *M.shape)
    pix = order[is_max[order]]
    Mf = M.ravel()
    sadv = Mf[base[pix]]
    return dict(R=R, shape=M.shape, pix=pix, val=Mf[pix], sad=saddle[pix], sadv=sadv,
                pers=Mf[pix] - sadv, par=peak[pix])


def _index_of(lv: dict, pixels):
    """Each pixel's index among ``lv``'s maxima, -1 for a pixel that is none (or -1)."""
    pixels = np.asarray(pixels, dtype=np.int64)
    out = np.full(pixels.size, -1, np.int64)
    if lv["pix"].size:
        srt = np.argsort(lv["pix"])
        k = np.minimum(np.searchsorted(lv["pix"], pixels, sorter=srt), srt.size - 1)
        hit = (pixels >= 0) & (lv["pix"][srt[k]] == pixels)
        out[hit] = srt[k[hit]]
    return out


def _ended(A: dict, la, ended, dk, succ) -> dict:
    """The record of the finer points ``ended`` (a mask over ``A``'s points) whose lines end at
    ``A``'s scale: line ids, death codes ``dk``, successor lines ``succ``, elder-rule peak lines
    and merge saddles."""
    e = np.flatnonzero(ended)
    lv = A["lv"]
    nx = lv["shape"][1]
    basin, idx = A["basin"][e], A["idx"][e]
    own = np.full(e.size, -1, np.int64)
    own[basin >= 0] = A["owner"][basin[basin >= 0]]
    top = (own == e) & (idx >= 0)                    # the points that own their basin
    elder = np.where(own == e, -1, own)
    par = lv["par"][idx[top]]
    elder[top] = np.where(par >= 0, A["owner"][np.maximum(par, 0)], -1)
    sad = np.full(e.size, -1, np.int64)
    sad[top] = lv["sad"][idx[top]]
    has = sad >= 0
    saddle = np.full((e.size, 3), np.nan)
    saddle[has] = np.column_stack([sad[has] % nx, sad[has] // nx, lv["sadv"][idx[has]]])
    peak = np.full(e.size, -1, np.int64)
    peak[elder >= 0] = la[elder[elder >= 0]]
    return dict(ids=la[e], kind=dk, succ=succ, peak=peak, saddle=saddle)


def track(levels, points, values) -> dict:
    """Lines of the ``points`` (per scale, finest first: flat pixel indices) with ``values`` across
    the ``levels`` (:func:`merge_tree` per scale, any iterable), by the module's rule.

    Returns ``line`` (per scale, each point's line id; the finest scale's points are lines
    ``0 .. n_0 - 1``), ``n`` (lines), per line ``last`` (its last scale), ``birth`` and ``death``
    (codes into :data:`BIRTHS` and :data:`DEATHS`), ``succ`` (the basin successor's line),
    ``peak`` (the elder-rule peak's line; both -1 for none) and ``saddle`` (x, y, M of the merge
    saddle its last point's basin dies at; NaN for none, or for a point that does not own its
    basin), per scale ``pers`` (each point's basin persistence, NaN off the domain) and ``basin``,
    and per step ``cand``, ``back``, ``link`` and ``events`` (n_i, links, annihilations, orphan
    deaths, exits, entries, creations, orphan births, n_(i+1))."""
    k = kernels()
    climb, owners, compete = k["climb"], k["owners"], k["compete"]
    out = dict(line=[], pers=[], basin=[], cand=[], back=[], link=[], events=[])
    births, records = [], []
    prev, n_lines = None, 0
    for s, (lv, pts, val) in enumerate(zip(levels, points, values)):
        pts = np.asarray(pts, dtype=np.int64)
        val = np.asarray(val, dtype=np.float64)
        ny, nx = lv["shape"]
        basin = climb(lv["R"], ny, nx, pts)
        idx = _index_of(lv, basin)
        cur = dict(lv=lv, pts=pts, val=val, basin=basin, idx=idx,
                   owner=owners(basin, val, ny * nx))
        if prev is None:
            line = np.arange(pts.size, dtype=np.int64)
            n_lines = pts.size
            births.append(np.zeros(pts.size, np.int8))
        else:
            la = out["line"][-1]
            fwd = climb(lv["R"], ny, nx, prev["pts"])
            cand = np.where(fwd >= 0, cur["owner"][np.maximum(fwd, 0)], -1)
            back = climb(prev["lv"]["R"], ny, nx, pts)
            lk = compete(cand, prev["basin"], back, prev["val"], pts.size)
            line = np.full(pts.size, -1, np.int64)
            line[lk[lk >= 0]] = la[lk >= 0]
            new = line < 0
            line[new] = n_lines + np.arange(int(new.sum()))
            n_lines += int(new.sum())
            chosen = np.zeros(pts.size, bool)
            chosen[cand[cand >= 0]] = True
            bk = np.where(back < 0, 1, np.where(chosen, 3, 2))[new].astype(np.int8)
            births.append(bk)
            ended = lk < 0
            c = cand[ended]
            dk = np.full(c.size, 2, np.int8)            # an orphan death, unless
            dk[c >= 0] = np.where(new[c[c >= 0]], 2, 1)  # the candidate continues another line
            dk[fwd[ended] < 0] = 3                       # or the pixel leaves the domain
            succ = np.full(c.size, -1, np.int64)
            succ[c >= 0] = line[c[c >= 0]]
            rec = _ended(prev, la, ended, dk, succ)
            rec["last"] = np.full(rec["ids"].size, s - 1, np.int64)
            records.append(rec)
            out["cand"].append(cand)
            out["back"].append(back)
            out["link"].append(lk)
            out["events"].append((prev["pts"].size, int((~ended).sum()),
                                  *np.bincount(dk, minlength=4)[1:].tolist(),
                                  *np.bincount(bk, minlength=4)[1:].tolist(), pts.size))
        out["line"].append(line)
        pers = np.full(idx.size, np.nan)
        pers[idx >= 0] = lv["pers"][idx[idx >= 0]]
        out["pers"].append(pers)
        out["basin"].append(basin)
        prev = cur
    n_sc = len(out["line"])
    out["n"] = n_lines
    out["birth"] = np.concatenate(births) if births else np.zeros(0, np.int8)
    out["last"] = np.full(n_lines, n_sc - 1, np.int64)
    out["death"] = np.zeros(n_lines, np.int8)
    out["succ"] = np.full(n_lines, -1, np.int64)
    out["peak"] = np.full(n_lines, -1, np.int64)
    out["saddle"] = np.full((n_lines, 3), np.nan)
    for rec in records:
        ids = rec["ids"]
        out["last"][ids] = rec["last"]
        out["death"][ids] = rec["kind"]
        out["succ"][ids] = rec["succ"]
        out["peak"][ids] = rec["peak"]
        out["saddle"][ids] = rec["saddle"]
    out["events"] = np.asarray(out["events"], dtype=np.int64).reshape(-1, 9)
    return out


# ---------------------------------------------------------------------------------------------
# The plain-numpy twins: the oracles the kernels are tested against.
# ---------------------------------------------------------------------------------------------

def merge_tree_twin(M, dom):
    """The merge tree by a level sweep (the ``_union_find_py`` oracle): at each pixel's level t,
    label {M >= t} on the domain (8-connected, ``scipy.ndimage.label``); a maximum dies at the first
    level where its component holds a higher one, paired with that level's pixel, its parent peak
    that component's highest pixel. Returns ``is_max``, ``saddle`` and ``peak`` per pixel."""
    from scipy import ndimage

    M = np.asarray(M, dtype=np.float64)
    flat = np.flatnonzero(dom)
    order = flat[np.argsort(-M.ravel()[flat], kind="stable")]
    R = np.full(M.shape, -1, np.int64)
    R.ravel()[order] = np.arange(order.size - 1, -1, -1)
    ny, nx = M.shape
    pad = np.pad(R, 1, constant_values=-2)
    nb = np.stack([pad[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx] for dy in (-1, 0, 1)
                   for dx in (-1, 0, 1) if dy or dx])
    is_max = (R >= 0) & np.all(R > nb, axis=0)
    maxima = np.flatnonzero(is_max)
    pixel_of = order[::-1]
    alive = np.ones(maxima.size, bool)
    saddle = np.full(M.size, -1, np.int64)
    peak = np.full(M.size, -1, np.int64)
    for p in order:
        lab, n = ndimage.label(R >= R.ravel()[p], np.ones((3, 3)))
        tops = np.asarray(ndimage.maximum(R, lab, np.arange(1, n + 1))).astype(np.int64)
        ml = lab.ravel()[maxima]
        t = tops[np.maximum(ml, 1) - 1]
        die = alive & (ml > 0) & (t != R.ravel()[maxima])
        saddle[maxima[die]] = p
        peak[maxima[die]] = pixel_of[t[die]]
        alive &= ~die
    return is_max.ravel(), saddle, peak


def climb_twin(R, shape, starts):
    """Steepest ascent by whole arrays (the ``_climb_py`` oracle): every pixel's highest pixel in
    its 3 x 3 neighbourhood, followed until it stops."""
    ny, nx = shape
    Rp = np.pad(np.asarray(R).reshape(shape).astype(np.int64), 1, constant_values=-2)
    ip = np.pad(np.arange(ny * nx).reshape(shape), 1, constant_values=-1)
    sh = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    best = np.argmax(np.stack([Rp[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx] for dy, dx in sh]), 0)
    idx = np.stack([ip[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx] for dy, dx in sh])
    nxt = np.take_along_axis(idx, best[None], 0)[0].ravel()
    starts = np.asarray(starts, dtype=np.int64)
    p = np.maximum(starts, 0)
    while not np.array_equal(nxt[p], p):
        p = nxt[p]
    return np.where((starts < 0) | (np.asarray(R).ravel()[np.maximum(starts, 0)] < 0), -1, p)


def owners_twin(basin, values, n_pix):
    """The ``_owners_py`` oracle by one sort: per basin, the largest value, the lowest index on a
    tie."""
    basin = np.asarray(basin, dtype=np.int64)
    j = np.flatnonzero(basin >= 0)
    o = j[np.lexsort((j, -np.asarray(values)[j], basin[j]))]
    first = np.ones(o.size, bool)
    first[1:] = basin[o][1:] != basin[o][:-1]
    owner = np.full(n_pix, -1, np.int64)
    owner[basin[o[first]]] = o[first]
    return owner


def compete_twin(cand, basin_a, back, values, n_b):
    """The ``_compete_py`` oracle by one sort over the mutual claimants."""
    cand = np.asarray(cand, dtype=np.int64)
    i = np.flatnonzero((cand >= 0) & (basin_a >= 0))
    i = i[back[cand[i]] == basin_a[i]]
    o = i[np.lexsort((i, -np.asarray(values)[i], cand[i]))]
    first = np.ones(o.size, bool)
    first[1:] = cand[o][1:] != cand[o][:-1]
    link = np.full(cand.size, -1, np.int64)
    link[o[first]] = cand[o[first]]
    return link
