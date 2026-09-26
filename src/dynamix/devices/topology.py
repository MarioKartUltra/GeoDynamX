# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Topology devices: build the incidence graph (Transform), query it (Filters).

ChainTopology is deliberately its own transform rather than part of WTMM2D: the graph caches
separately from the (much larger) scale stack, and the same device shape will serve every future
binding (map, drillhole) -- topology-as-a-transform is the pattern, WTMM is its first instance.
"""
from __future__ import annotations

from dynamix.model.param import Param, ParamKind


class ChainTopology:
    """Build the H-chain/segment/extremum/V-chain incidence graph from a WTMM result."""

    name = "chain_topology"
    params = ()

    def compute(self, field, params: dict, *, progress=None) -> dict:
        from dynamix.topology.wtmm import build_chain_topology, vchain_index

        out = dict(field)                    # `field` is the upstream WTMM result dict
        model = build_chain_topology(field)
        out["topology"] = model
        # The H-chain -> V-chain incidence index is a pure function of the graph, so it belongs
        # here rather than in MinVChains: built once, it rides the transform cache and costs a
        # redraw nothing. Rebuilding it per redraw measured 37.8 ms at 128x128, against the 16 ms
        # a Filter is documented to fit inside.
        out["_topology_vids"] = vchain_index(model)
        return out

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)


class MinVChains:
    """Keep H-chains crossed by at least N distinct V-chains.

    The subject is the H-chain; the V-chain list is evidence and passes through untouched.
    EQSelect could not express this filter -- its only addressable collection was the V-chains --
    and pruning the evidence instead of the subject is the exact bug this device exists to end.

    Anchor invariant (see :mod:`dynamix.topology.wtmm`): topology anchors index the UNFILTERED
    transform result. This filter rewrites ``result["extrema"]``, and so do the other extrema
    filters, so after any of them an anchor's ``row`` no longer addresses the layer on screen.
    Consumers must resolve anchors against the transform's cached result, never against a
    post-filter layer; ``_topology_stale`` is stamped on the result whenever the layers have been
    rewritten, so a consumer can tell without guessing.
    """

    name = "min_vchains"
    params = (
        Param("min_vchains", ParamKind.INT, default=2, min=1, max=64,
              soft_min=1, soft_max=8, label="Min V-chains"),
    )

    def apply(self, result: dict, params: dict) -> dict:
        import numpy as np

        model = result.get("topology")
        if model is None:
            return result
        need = int(params["min_vchains"])

        # ChainTopology prebuilds the incidence index (it is a pure function of the graph and
        # caches with it). Rebuild inline only when it is absent -- an older cached result, or a
        # model asserted by hand and dropped into a result dict -- so this device keeps working
        # off any topology, just without the free lunch.
        # Perf: the drop computation is a Python walk over EVERY h-chain node (~187k on a 2048
        # DEM -> ~120 ms, profiled) and depends ONLY on (this topology, need). resolve() re-runs
        # the whole filter chain on every tweak, so unmemoized, scrubbing scale or min |W| pays
        # that 120 ms each time for an identical result. Memoize it on the model, keyed by need:
        # the model is the cached transform's own object (stable across resolves, replaced only
        # when the transform recomputes), so this is exactly EQSelect's "cache per-layer filter
        # state" -- a downstream tweak skips the walk entirely. The memo holds only the small
        # (dropped, n_dropped) summary, never a copy of any point array.
        memo = getattr(model, "_minv_drop_memo", None)
        if memo is None:
            memo = {}
            try:
                model._minv_drop_memo = memo
            except (AttributeError, TypeError):
                memo = None                        # a frozen model: fall back to recomputing
        cached = memo.get(need) if memo is not None else None
        if cached is not None:
            dropped, n_dropped = cached
        else:
            vids_by_hchain = result.get("_topology_vids")
            if vids_by_hchain is None:
                from dynamix.topology.wtmm import vchain_index

                vids_by_hchain = vchain_index(model)

            dropped = {}                           # scale index -> {line_id}
            n_dropped = 0
            for h in model.nodes_where(anchor0="hchain"):
                if len(vids_by_hchain.get(h.node_id, ())) < need:
                    _, si, lid = h.anchor
                    dropped.setdefault(si, set()).add(lid)
                    n_dropped += 1
            if memo is not None:
                memo[need] = (dropped, n_dropped)

        out = dict(result)
        out["_hchains_dropped"] = n_dropped
        if n_dropped:
            # `dropped` is keyed by ABSOLUTE scale index, so the displayed layers must be paired
            # with their true scale indices, not with their positions. After ScaleSelect the list
            # is one layer long and every position is 0, so enumerate() would prune the selected
            # scale using scale 0's line ids -- and both scales label their lines from 0, so the
            # wrong lines vanish with nothing raised anywhere.
            # The `len == 1` guard keeps the zip below length-preserving: a stamped _scale_idx
            # always comes with a one-layer list, and if that ever stopped being true, dropping
            # layers on the floor would be a worse bug than pruning by position.
            base = result.get("_scale_idx")
            n_layers = len(result["extrema"])
            sis = [int(base)] if (base is not None and n_layers == 1) else range(n_layers)
            new_ext = []
            for si, ext in zip(sis, result["extrema"]):
                bad = dropped.get(si)
                if not bad:
                    new_ext.append(ext)
                    continue
                keep = ~np.isin(ext["line_id"], list(bad))
                new_ext.append({k: v[keep] for k, v in ext.items()})
            out["extrema"] = new_ext
            # The rows moved: every ("extremum", si, row) anchor in the model now points at a
            # different point of this layer. Say so, rather than leaving a consumer to discover
            # it by drawing the wrong extremum.
            out["_topology_stale"] = True
        return out
