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
"""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np

#: The chainer a recipe that names none runs: the benchmark.
DEFAULT = "xsmurf_bench"

_KERNELS = None


class Chainer(Protocol):
    name: str

    def link(self, extrema: list, scales, params: dict, backend) -> list:
        """V-chains from the per-scale ``extrema`` (``extrema2d``'s schema) over ``scales``
        (finest first); ``params`` is the resolved WTMM parameter set, ``backend`` the
        :class:`~dynamix.core.wtmm_backend.PythonWTMMBackend` whose single-maxima and chaining
        passes a chainer may reuse."""


class XsmurfBench:
    """The benchmark: ``PythonWTMMBackend.chains2d``, the exact port of xsmurf's ``vchain``
    (single maxima, the Chebyshev box, the similitude band, the squared-distance cap, the
    nearest-claimant competition)."""

    name = "xsmurf_bench"

    def link(self, extrema: list, scales, params: dict, backend) -> list:
        return backend.chains2d(extrema, scales, similitude=params["similitude"],
                                box_ratio=params["box_ratio"], dist2_max=params["dist2_max"],
                                min_len=params["min_chain_len"], smooth=params["smooth"],
                                band_per=params.get("band_per", "voice"))


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

    def link(self, extrema: list, scales, params: dict, backend) -> list:
        return _link_all(extrema, scales, params, backend, band=True, along_arg=True)


class VerticalExperimental:
    """Labs: the nearest coarser candidate in the box under the distance cap, with no similitude
    band, so a maximum whose modulus jumps across scales still chains."""

    name = "vertical_experimental"

    def link(self, extrema: list, scales, params: dict, backend) -> list:
        return _link_all(extrema, scales, params, backend, band=False, along_arg=False)


#: Every chainer by name.
CHAINERS: dict[str, Chainer] = {c.name: c for c in (XsmurfBench(), GradientArgHorizontal(),
                                                     VerticalExperimental())}


def chainer(name: str) -> Chainer:
    """The chainer called ``name``; ``KeyError`` naming the known ones otherwise."""
    try:
        return CHAINERS[name]
    except KeyError:
        raise KeyError(f"no chainer named {name!r}; known: {sorted(CHAINERS)}") from None


def link(extrema: list, scales, params: dict, backend) -> list:
    """V-chains by the chainer ``params["chainer"]`` names (:data:`DEFAULT` when absent)."""
    return chainer(params.get("chainer", DEFAULT)).link(extrema, scales, params, backend)
