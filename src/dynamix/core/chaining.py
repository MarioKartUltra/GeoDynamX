# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Cross-scale chaining as a pluggable interface.

A chainer links the maxima of one scale to the next and hands back V-chains in the one schema the
rest of the app reads (``PythonWTMMBackend.chains2d``: a list of dicts with ``x``, ``y``, ``mod``,
``log2_mod`` and ``log2_scales``, index 0 the finest scale). :class:`XsmurfBench` is today's chaining,
the exact port of xsmurf's ``vchain``, and the benchmark every other chainer is scored against on
synthetic cusps of known Hölder exponent (``tests/test_chaining_benchmark.py``). The WTMM device's
``chainer`` param names one; its default is the benchmark, so the keys of every existing recipe are
what they were.

The two other chainers share one numba linking pass (:func:`_link_pair`): the coarser single maxima
are bucketed into cells the size of the benchmark's Chebyshev box, each eligible finer point scans
the nine cells around it, and a coarser point keeps its closest claimant, as the benchmark does.
They differ in what a finer point prefers: :class:`GradientArgHorizontal` takes the candidate
nearest the ray along its own gradient argument, :class:`VerticalExperimental` the nearest
candidate with no similitude band. Chains are anchored at the finest scale, as in the benchmark.

:class:`MorseCerf` links the same single maxima through the topology of the modulus itself
(:mod:`dynamix.core.morse`): the merge tree at each scale and mutual steepest-ascent climbs between
neighbouring scales. It reads the per-scale modulus stack, which :func:`link` takes as ``fields``
and the other chainers ignore, and its chains carry the line's events beside the one schema.
"""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np

#: The chainer a recipe that names none runs: the benchmark.
DEFAULT = "xsmurf_bench"

_KERNELS = None


#: Spec 14's chaining options, off by default (xsmurf's behaviour): the angle test and its
#: threshold, the search-radius rule and its constant, the tie-break between claimants, and the
#: border policy at the cone of influence (``coi_k`` times the scale in pixels from the edge).
OPTION_DEFAULTS = {"angle_test": False, "angle_threshold": 0.8, "radius_rule": "xsmurf",
                   "radius_c": 1.0, "tie_break": "distance", "border_policy": "none",
                   "coi_k": 2.0}
BORDER_POLICIES = ("none", "coi_trim", "coi_drop")

#: The cascade tree's parent of a line that ends (the WTMM device's ``tree_parent``, recorded on
#: the result for the tree pass): the line its basin drains into one scale coarser, or the line
#: holding its elder-rule parent peak. :class:`MorseCerf` chains carry both, as ``succ`` and
#: ``peak``.
TREE_PARENTS = ("basin_successor", "elder_rule")

#: The keys a :class:`MorseCerf` chain carries beside the one schema: per point, and per chain.
MORSE_POINT_KEYS = ("pers",)
MORSE_CHAIN_KEYS = ("birth", "death", "line", "succ", "peak", "saddle")


class Chainer(Protocol):
    name: str
    #: Whether :meth:`link` reads ``fields``, the per-scale modulus stack: a caller that does not
    #: hold the stack keeps one for such a chainer only.
    reads_fields: bool

    def link(self, extrema: list, scales, params: dict, backend, *, fields=None,
             floors=None) -> list:
        """V-chains from the per-scale ``extrema`` (``extrema2d``'s schema) over ``scales``
        (finest first); ``params`` is the resolved WTMM parameter set, ``backend`` the
        :class:`~dynamix.core.wtmm_backend.PythonWTMMBackend` whose single-maxima and chaining
        passes a chainer may reuse, ``fields`` the modulus the extrema were found on,
        ``(n_scales, ny, nx)``, and ``floors`` the modulus each scale's extrema were kept above
        (the detector's floor; ``thresh`` times the scale's maximum when None)."""


class XsmurfBench:
    """The benchmark: ``PythonWTMMBackend.chains2d``, the exact port of xsmurf's ``vchain``
    (single maxima, the Chebyshev box, the similitude band, the squared-distance cap, the
    nearest-claimant competition)."""

    name = "xsmurf_bench"
    reads_fields = False

    def link(self, extrema: list, scales, params: dict, backend, *, fields=None,
             floors=None) -> list:
        opt = {k: params.get(k, v) for k, v in OPTION_DEFAULTS.items()}
        return backend.chains2d(extrema, scales, similitude=params["similitude"],
                                box_ratio=params["box_ratio"], dist2_max=params["dist2_max"],
                                min_len=params["min_chain_len"], smooth=params["smooth"],
                                band_per=params.get("band_per", "voice"),
                                angle_test=bool(opt["angle_test"]),
                                angle_threshold=float(opt["angle_threshold"]),
                                radius_rule=opt["radius_rule"], radius_c=float(opt["radius_c"]),
                                tie_break=opt["tie_break"])


def _kernels():
    """The linking kernel, compiled on first use (``cache=True``)."""
    global _KERNELS
    if _KERNELS is None:
        import numba

        @numba.njit(cache=True)
        def link_pair(fx, fy, fmod, farg, elig, cx, cy, cmod, box, lo, hi, r2max, along_arg):
            """For the eligible finer points (``elig``, indices into the finer arrays) the index
            of the coarser point each one wins, -1 for none: every finer point claims its
            preferred admissible candidate in the box (inside the modulus band ``lo < ratio <
            hi`` and under the squared distance ``r2max``); a coarser point then keeps its
            closest claimant, the earlier finer index on a tie."""
            n_f = fx.shape[0]
            n_c = cx.shape[0]
            claim = np.full(n_f, -1, dtype=np.int64)
            if n_c == 0 or elig.shape[0] == 0:
                return claim
            # cells of the box's size: a candidate within the box lies in the 3 x 3 cells around
            cell = float(box)
            x0 = cx.min()
            y0 = cy.min()
            ncx = int((cx.max() - x0) // cell) + 1
            ncy = int((cy.max() - y0) // cell) + 1
            cell_of = np.empty(n_c, dtype=np.int64)
            for j in range(n_c):
                cell_of[j] = int((cy[j] - y0) // cell) * ncx + int((cx[j] - x0) // cell)
            order = np.argsort(cell_of)
            starts = np.zeros(ncx * ncy + 1, dtype=np.int64)
            for j in range(n_c):
                starts[cell_of[j] + 1] += 1
            for c in range(ncx * ncy):
                starts[c + 1] += starts[c]
            best_d2 = np.full(n_f, np.inf)
            for e in range(elig.shape[0]):
                i = elig[e]
                px = fx[i]
                py = fy[i]
                ux = math.cos(farg[i])
                uy = math.sin(farg[i])
                ix = int((px - x0) // cell)
                iy = int((py - y0) // cell)
                best_j = -1
                best_score = np.inf
                best_tie = np.inf
                for dy in range(-1, 2):
                    cy_i = iy + dy
                    if cy_i < 0 or cy_i >= ncy:
                        continue
                    for dx in range(-1, 2):
                        cx_i = ix + dx
                        if cx_i < 0 or cx_i >= ncx:
                            continue
                        c = cy_i * ncx + cx_i
                        for o in range(starts[c], starts[c + 1]):
                            j = order[o]
                            ddx = cx[j] - px
                            ddy = cy[j] - py
                            if abs(ddx) > box or abs(ddy) > box:
                                continue
                            d2 = ddx * ddx + ddy * ddy
                            if d2 >= r2max:
                                continue
                            if lo > 0.0:
                                ratio = cmod[j] / fmod[i]
                                if ratio <= lo or ratio >= hi:
                                    continue
                            if along_arg:
                                score = abs(ddx * uy - ddy * ux)
                                tie = d2
                            else:
                                score = d2
                                tie = float(j)
                            if score < best_score or (score == best_score and tie < best_tie):
                                best_score = score
                                best_tie = tie
                                best_j = j
                                best_d2[i] = d2
                claim[i] = best_j
            # competition: a coarser point keeps its closest claimant
            winner = np.full(n_c, -1, dtype=np.int64)
            winner_d2 = np.full(n_c, np.inf)
            for e in range(elig.shape[0]):
                i = elig[e]
                j = claim[i]
                if j < 0:
                    continue
                if best_d2[i] < winner_d2[j]:
                    winner_d2[j] = best_d2[i]
                    winner[j] = i
            for e in range(elig.shape[0]):
                i = elig[e]
                j = claim[i]
                if j >= 0 and winner[j] != i:
                    claim[i] = -1
            return claim

        _KERNELS = link_pair
    return _KERNELS


def _assemble(sm: list, links: list, scales, min_len: int) -> list:
    """Chains from the per-scale single maxima ``sm`` and the links ``links[s][i]`` (the index
    at scale ``s + 1`` that point ``i`` of scale ``s`` links to, -1 for none): every finest-scale
    point starts one; those shorter than ``min_len`` are dropped."""
    log2_scales = np.log2(np.asarray(scales, dtype=np.float64))
    chains = []
    n_sc = len(sm)
    for i0 in range(len(sm[0]["x"])):
        xs, ys, mods = [], [], []
        s, i = 0, i0
        while True:
            xs.append(int(round(float(sm[s]["x"][i]))))
            ys.append(int(round(float(sm[s]["y"][i]))))
            mods.append(float(sm[s]["mod"][i]))
            if s + 1 >= n_sc or links[s][i] < 0:
                break
            s, i = s + 1, int(links[s][i])
        k = len(xs)
        if k < min_len:
            continue
        mod = np.asarray(mods, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            log2_mod = np.log2(np.abs(mod))
        chains.append({"x": np.asarray(xs, dtype=np.int64), "y": np.asarray(ys, dtype=np.int64),
                       "mod": mod, "log2_mod": log2_mod, "log2_scales": log2_scales[:k].copy()})
    return chains


def _link_all(extrema: list, scales, params: dict, backend, *, band: bool, along_arg: bool) -> list:
    """The shared pass: single maxima, then :func:`_link_pair` scale pair by scale pair with the
    benchmark's box, bands (when ``band``) and distance cap, then the chains."""
    sm = backend.single_maxima2d(extrema, smooth=params["smooth"])
    n_sc = len(sm)
    xs = [np.asarray(e["x"], dtype=np.float64) for e in sm]
    ys = [np.asarray(e["y"], dtype=np.float64) for e in sm]
    mods = [np.asarray(e["mod"], dtype=np.float64) for e in sm]
    args = [np.asarray(e["arg"], dtype=np.float64) for e in sm]
    similitude = float(params["similitude"]) if band else 0.0
    per_octave = params.get("band_per", "voice") == "octave"
    r2max = float(params["dist2_max"])
    kernel = _kernels()
    links = []
    elig = np.arange(len(xs[0]), dtype=np.int64)
    for s in range(n_sc - 1):
        box = max(1, int(params["box_ratio"] * np.log2(scales[s + 1]) * 2))
        # The band per voice, or per octave: each step takes similitude^(log2 of its ratio).
        lo = (similitude ** float(np.log2(scales[s + 1] / scales[s]))
              if per_octave and similitude > 0.0 else similitude)
        hi = (1.0 / lo) if lo > 0.0 else np.inf
        claim = kernel(xs[s], ys[s], mods[s], args[s], elig, xs[s + 1], ys[s + 1], mods[s + 1],
                       float(box), lo, hi, r2max, along_arg)
        links.append(claim)
        elig = np.unique(claim[claim >= 0])
    return _assemble(sm, links, scales, int(params["min_chain_len"]))


class GradientArgHorizontal:
    """A placeholder beside the benchmark: the same box, bands and competition, but a finer point
    prefers the coarser candidate nearest the ray along its own gradient argument (the direction
    the maxima line crosses), the nearer one on a tie."""

    name = "gradient_arg_horizontal"
    reads_fields = False

    def link(self, extrema: list, scales, params: dict, backend, *, fields=None,
             floors=None) -> list:
        return _link_all(extrema, scales, params, backend, band=True, along_arg=True)


class VerticalExperimental:
    """Labs: the nearest coarser candidate in the box under the distance cap, with no similitude
    band, so a maximum whose modulus jumps across scales still chains."""

    name = "vertical_experimental"
    reads_fields = False

    def link(self, extrema: list, scales, params: dict, backend, *, fields=None,
             floors=None) -> list:
        return _link_all(extrema, scales, params, backend, band=False, along_arg=False)


class MorseCerf:
    """The single maxima linked by Morse/Cerf tracking of the modulus (:func:`dynamix.core.morse.
    track`): per scale the merge tree of ``fields`` on the WTMM's thresholded domain, above the
    detector's own floor (``floors``), so every single maximum lies on it; a finer
    single maximum climbs the next scale's modulus to a maximum and takes the single maximum that
    owns that basin (the largest modulus in it); the link holds when that point climbs the finer
    modulus back into the finer point's basin, and a coarser point keeps the largest such
    claimant. The WTMMM set is xsmurf's; only the linking differs from the benchmark's.

    Chains are rooted at the finest scale, as the benchmark's; the lines born above it take part
    in the links and the successors but are not chains. Beside the one schema each chain carries
    ``pers`` (per point, the persistence of the maximum whose basin holds it), ``birth`` and
    ``death`` (:data:`~dynamix.core.morse.BIRTHS`, :data:`~dynamix.core.morse.DEATHS`), ``line``
    (its index in this list), ``succ`` (the ``line`` of its basin successor) and ``peak`` (of its
    elder-rule peak), -1 for none or for a line that is not a chain, and ``saddle`` (x, y,
    modulus of the merge saddle where its last basin dies, NaN for none)."""

    name = "morse_cerf"
    reads_fields = True

    def link(self, extrema: list, scales, params: dict, backend, *, fields=None,
             floors=None) -> list:
        from dynamix.core import morse

        if fields is None:
            raise ValueError("the morse_cerf chainer climbs the modulus: pass its per-scale stack "
                             "as fields")
        sm = backend.single_maxima2d(extrema, smooth=params["smooth"])
        nx = np.asarray(fields[0]).shape[1]
        thresh = float(params.get("thresh", 0.0))
        floors = [None] * len(fields) if floors is None else floors
        levels = (morse.merge_tree(M, morse.domain(M, thresh, f)) for M, f in zip(fields, floors))
        points = [np.asarray(e["y"], np.int64) * nx + np.asarray(e["x"], np.int64) for e in sm]
        tr = morse.track(levels, points, [e["mod"] for e in sm])
        return _morse_chains(sm, tr, scales, int(params["min_chain_len"]))


def _morse_chains(sm: list, tr: dict, scales, min_len: int) -> list:
    """The chains of :class:`MorseCerf`: the lines of ``tr`` rooted at the finest scale and
    spanning ``min_len`` scales or more, their points the single maxima ``sm``."""
    from dynamix.core.morse import BIRTHS, DEATHS

    n_sc = len(sm)
    n0 = len(sm[0]["x"]) if n_sc else 0
    length = tr["last"][:n0] + 1
    keep = np.flatnonzero(length >= min_len)
    chain_of = np.full(tr["n"], -1, np.int64)
    chain_of[keep] = np.arange(keep.size)

    def ref(line):
        return int(chain_of[line]) if line >= 0 else -1

    cols = {k: np.zeros((n0, n_sc)) for k in ("x", "y", "mod", "pers")}
    for s in range(n_sc):
        ln = tr["line"][s]
        sel = np.flatnonzero(ln < n0)
        for k in ("x", "y", "mod"):
            cols[k][ln[sel], s] = np.asarray(sm[s][k])[sel]
        cols["pers"][ln[sel], s] = tr["pers"][s][sel]
    log2_scales = np.log2(np.asarray(scales, dtype=np.float64))
    chains = []
    for c, L in enumerate(keep):
        k = int(length[L])
        mod = cols["mod"][L, :k].copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            log2_mod = np.log2(np.abs(mod))
        chains.append({"x": cols["x"][L, :k].astype(np.int64),
                       "y": cols["y"][L, :k].astype(np.int64), "mod": mod, "log2_mod": log2_mod, "log2_scales": log2_scales[:k].copy(),
                       "pers": cols["pers"][L, :k].copy(),
                       "birth": BIRTHS[tr["birth"][L]], "death": DEATHS[tr["death"][L]],
                       "line": c, "succ": ref(tr["succ"][L]), "peak": ref(tr["peak"][L]),
                       "saddle": tuple(float(v) for v in tr["saddle"][L])})
    return chains


#: Every chainer by name.
CHAINERS: dict[str, Chainer] = {c.name: c for c in (XsmurfBench(), GradientArgHorizontal(),
                                                     VerticalExperimental(), MorseCerf())}


def chainer(name: str) -> Chainer:
    """The chainer called ``name``; ``KeyError`` naming the known ones otherwise."""
    try:
        return CHAINERS[name]
    except KeyError:
        raise KeyError(f"no chainer named {name!r}; known: {sorted(CHAINERS)}") from None


def border_trimmed(chains: list, shape, policy: str, coi_k: float, min_len: int, *,
                   reach_at=None, offset=(0, 0)) -> list:
    """The chains under a border policy: ``"coi_trim"`` cuts a chain at the first scale where it
    comes within ``coi_k`` scales (the normalized scale, in pixels) of the image edge, and drops it
    when fewer than ``min_len`` scales remain; ``"coi_drop"`` drops any chain that enters that
    zone; ``"none"`` keeps them all (xsmurf). ``reach_at`` maps a chain's ``log2_scales`` to the
    zone's width in place of ``coi_k`` scales (spec 22's measured reach); ``offset`` places the
    chains' grid at ``(row, col)`` of the image ``shape`` (an ROI inside its parent)."""
    if policy == "none" or shape is None:
        return chains
    if policy not in BORDER_POLICIES:
        raise ValueError(f"border_policy {policy!r}; known: {BORDER_POLICIES}")
    ny, nx = int(shape[0]), int(shape[1])
    oy, ox = (float(v) for v in offset)
    out = []
    for c in chains:
        x = np.asarray(c["x"], dtype=np.float64) + ox
        y = np.asarray(c["y"], dtype=np.float64) + oy
        log2_scales = np.asarray(c["log2_scales"], dtype=np.float64)
        reach = coi_k * 2.0 ** log2_scales if reach_at is None else reach_at(log2_scales)
        inside = np.minimum.reduce([x, y, nx - 1 - x, ny - 1 - y]) >= reach
        if inside.all():
            out.append(c)
        elif policy == "coi_trim":
            keep = int(np.argmin(inside))           # the first scale inside the zone
            if keep >= min_len:
                out.append({k: (v[:keep] if isinstance(v, np.ndarray) and len(v) == len(x) else v)
                            for k, v in c.items()})
    return out


def link(extrema: list, scales, params: dict, backend, *, shape=None, fields=None,
         floors=None) -> list:
    """V-chains by the chainer ``params["chainer"]`` names (:data:`DEFAULT` when absent), under
    the border policy when ``shape`` (the image's ``(ny, nx)``) is given; ``fields``, the
    per-scale modulus stack, and ``floors``, the detector's floor per scale, go to the chainer
    (:attr:`Chainer.reads_fields`)."""
    chains = chainer(params.get("chainer", DEFAULT)).link(extrema, scales, params, backend,
                                                           fields=fields, floors=floors)
    return border_trimmed(chains, shape, params.get("border_policy", "none"),
                          float(params.get("coi_k", OPTION_DEFAULTS["coi_k"])),
                          int(params.get("min_chain_len", 2)))
