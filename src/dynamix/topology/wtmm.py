# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The WTMM binding: H-chain segmentation and the incidence graph.

Splitter reference: wtmm_ebsd/viewer.py:467 ("Split between adjacent WTMMM at the LOCAL MIN of
|T|") -- reimplemented here fresh, per the design decision, so topology has no wtmm_ebsd
dependency. Ordering reuses core's _order_lines rather than re-deriving a neighbour walk.

Memory rule: nodes anchor INDICES into the result arrays (scale, row, line, span); no coordinate
or modulus arrays are copied into the model.

Anchor invariant
----------------
Anchors index the UNFILTERED transform result -- the arrays ``build_chain_topology`` was handed.
A filter that rewrites ``result["extrema"]`` (MinVChains itself, OrientationWedge,
ModulusThreshold) re-indexes those arrays and the anchors no longer address the layer on screen.
Consumers must resolve an anchor against the transform's cached result, never against a
post-filter layer. MinVChains stamps ``_topology_stale`` when it has rewritten the layers.

Approximations, stated plainly
------------------------------
The model is exact for a simple, unbranched line. Where the data is not that, three honest
approximations stand in, and a reader of the graph should know all three:

1. **Run endpoints stand in for a branching line's boundary.** ``_order_lines`` returns a
   branching line as several contiguous *runs*, each ordered on its own; "the boundary of the
   line" is then a topological question this binding does not answer. Each run's own endpoints
   are used instead, so a segment reaching the end of ITS run is called ``touches_end`` even
   where the parent line continues down another arm.
2. **A single-point span reports its owner as interior.** ``at="boundary"`` needs a span longer
   than one point to distinguish an end from the middle (``b - a > 1``); a one-point span is all
   boundary and all interior at once, and interior is the reading chosen.
3. **``equal`` (R400) is asserted only for single-run lines.** A branching line has several
   full-span runs and they cannot each *be* the whole line; every run of such a line degrades to
   ``covered_by`` (R476). The cost is that a branching line never carries an ``equal`` edge even
   when one arm arguably is the line.
"""
from __future__ import annotations

import numpy as np

# Private-symbol coupling: `_order_lines` is a PRIVATE name in core, and core is a VERBATIM copy
# of EQSelect (never modify EQSelect, copies stay verbatim). A future re-copy that
# renames this helper breaks topology here, and nothing in core's public surface warns of it.
from dynamix.core.wtmm_backend import _order_lines
from dynamix.topology.codes import LINE, POINT, line_in_line_code, point_code
from dynamix.topology.model import TopologyModel


def split_walk(mod_walk: np.ndarray, owner_pos: np.ndarray) -> list[tuple[int, int]]:
    """Spans [start, stop) along one ordered walk, one per owner, cut at the local min of
    ``mod_walk`` strictly between adjacent owners (leftmost min on ties, per argmin).

    NaN is a hole in the walk, not its deepest point: ``np.argmin`` would return the NaN's index
    and put the cut where the data says nothing, so the minimum is taken with ``nanargmin``."""
    m = int(len(mod_walk))
    owners = np.sort(np.asarray(owner_pos, dtype=np.int64))
    if owners.size == 0:
        return [(0, m)]
    cuts = []
    for lo, hi in zip(owners[:-1], owners[1:]):
        if hi - lo == 1:
            cuts.append(int(hi))
        else:
            cuts.append(int(lo) + 1 + int(np.nanargmin(mod_walk[lo + 1:hi])))
    edges = [0, *cuts, m]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def build_chain_topology(result: dict) -> TopologyModel:
    if "extrema" not in result or "_shape" not in result:
        raise ValueError(
            "build_chain_topology needs a WTMM2D result dict with 'extrema' and '_shape'; "
            "put a ChainTopology device after WTMM2D in the chain"
        )
    ny, nx = result["_shape"]
    chains = result.get("chains") or []
    model = TopologyModel()

    for ci in range(len(chains)):
        model.add_node(f"v{ci}", LINE, space_dim=3, anchor=("vchain", ci))

    for si, ext in enumerate(result["extrema"]):
        if "line_id" not in ext:
            raise ValueError(
                f"scale {si}: extrema carry no 'line_id'; the upstream result predates "
                f"line labelling and cannot be segmented"
            )
        x, y, mod, line_id = ext["x"], ext["y"], ext["mod"], ext["line_id"]
        if x.size == 0:
            continue
        # chain index by position at this scale (chain index k <-> scale index k)
        pos2ci = {(int(c["x"][si]), int(c["y"][si])): ci
                  for ci, c in enumerate(chains) if len(c["x"]) > si}

        grid = np.full(ny * nx, -1, dtype=np.int64)
        grid[y * nx + x] = np.arange(x.size, dtype=np.int64)
        order, seg, starts = _order_lines(x, y, line_id, grid, nx, ny)

        for li in range(len(starts) - 1):
            sl = order[starts[li]:starts[li + 1]]
            if sl.size == 0:
                continue
            lid = int(line_id[sl[0]])
            hid = f"h{si}.{lid}"
            if hid not in model.nodes:
                model.add_node(hid, LINE, space_dim=2, anchor=("hchain", si, lid))
            # contiguous runs of the seg id are individually-ordered polylines
            run_breaks = np.nonzero(np.diff(seg[starts[li]:starts[li + 1]]))[0] + 1
            walks = np.split(sl, run_breaks)
            # A branching line arrives as several runs, and several full-span runs cannot each BE
            # the whole line -- R400 `equal` would then be asserted against itself. Gate it on the
            # unbranched case; a full-span run of a branching line degrades to R476 covered_by.
            single_run = len(walks) == 1
            for ri, walk in enumerate(walks):
                mod_walk = np.abs(mod[walk])
                owner_pos = np.array(
                    [k for k, row in enumerate(walk)
                     if (int(x[row]), int(y[row])) in pos2ci], dtype=np.int64)
                spans = split_walk(mod_walk, owner_pos)
                whole = single_run and len(spans) == 1 and spans[0] == (0, len(walk))
                for j, (a, b) in enumerate(spans):
                    sid = f"h{si}.{lid}.r{ri}.s{j}"
                    model.add_node(sid, LINE, space_dim=1,
                                   anchor=("hseg", si, lid, ri, a, b))
                    model.assert_spatial(
                        line_in_line_code(equal=whole,
                                          touches_end=(not whole and (a == 0 or b == len(walk)))),
                        sid, hid, space_dim=1)
                    for k in owner_pos:
                        if not (a <= k < b):
                            continue
                        row = int(walk[k])
                        eid = f"e{si}.{row}"
                        if eid not in model.nodes:
                            model.add_node(eid, POINT, space_dim=2,
                                           anchor=("extremum", si, row))
                        at_seg = "boundary" if k in (a, b - 1) and b - a > 1 else "interior"
                        model.assert_spatial(point_code(at=at_seg), eid, sid, space_dim=1)
                        ci = pos2ci[(int(x[row]), int(y[row]))]
                        n_ch = len(chains[ci]["x"])
                        at_v = "boundary" if si in (0, n_ch - 1) else "interior"
                        model.assert_spatial(point_code(at=at_v), eid, f"v{ci}",
                                             space_dim=3)
    return model


def vchain_index(model) -> dict[str, frozenset[str]]:
    """H-chain node id -> the ids of the V-chains crossing it, in one pass over the edges.

    A pure function of the model, so it is built once by the ChainTopology transform (where it
    caches with the graph) rather than per redraw inside MinVChains -- rebuilding it on every
    slider move cost 37.8 ms at 128x128 against a 16 ms filter budget.

    One pass over ``model.edges``, classified by the two endpoints' anchor kinds, rather than a
    ``neighbors()`` walk per H-chain -- ``neighbors()``/``edges_of()`` are unindexed linear scans,
    so the nested version was O(H-chains x edges). Classification goes by anchor KIND on both
    endpoints rather than by position (e.a vs e.b), so it stays correct regardless of which side
    the builder puts the child/parent on. Nodes with no anchor -- a hand-asserted fault names
    itself, not an array row -- are unclassifiable here and their edges are skipped.
    """
    seg2hchain: dict[str, str] = {}
    ext2segs: dict[str, list[str]] = {}
    ext2vchains: dict[str, list[str]] = {}
    for e in model.edges:
        if e.family != "spatial":
            continue
        na, nb = model.nodes[e.a], model.nodes[e.b]
        if not (na.anchor and nb.anchor):
            continue
        ka, kb = na.anchor[0], nb.anchor[0]
        if {ka, kb} == {"hseg", "hchain"}:
            seg2hchain[e.a if ka == "hseg" else e.b] = e.a if ka == "hchain" else e.b
        elif {ka, kb} == {"extremum", "hseg"}:
            ext2segs.setdefault(e.a if ka == "extremum" else e.b, []).append(
                e.a if ka == "hseg" else e.b)
        elif {ka, kb} == {"extremum", "vchain"}:
            ext2vchains.setdefault(e.a if ka == "extremum" else e.b, []).append(
                e.a if ka == "vchain" else e.b)

    vids: dict[str, set[str]] = {}
    for ext_id, vchain_ids in ext2vchains.items():
        for seg_id in ext2segs.get(ext_id, ()):
            hchain_id = seg2hchain.get(seg_id)
            if hchain_id is not None:
                vids.setdefault(hchain_id, set()).update(vchain_ids)
    return {hid: frozenset(v) for hid, v in vids.items()}
