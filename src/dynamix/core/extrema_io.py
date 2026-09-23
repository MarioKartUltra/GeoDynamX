# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""WTMM extrema .npz I/O — loading exported chains and flattening them for picking.

EXTRACTION NOTE (DynamiX): these two functions were lifted VERBATIM from EQSelect's
``eqselect/reflayers.py`` (lines 302-382). They were misplaced there: ``reflayers`` is
otherwise about tectonic reference layers — Slab2 surfaces, plate boundaries, seafloor
age, Syracuse thermal — none of which DynamiX copies. These two are pure-numpy WTMM
extrema I/O with no earthquake content, and their own docstrings state they are GUI-free.

The function bodies are otherwise unmodified. **One deliberate change:** the original converted
``elev_m -> depth_km`` positive-DOWN (``-s[:, 2] / 1000.0``) in two places; both now produce a
signed ``height_km`` positive-UP, matching the same flip in :mod:`dynamix.core.projection`. A
loader disagreeing with the projection would mirror every chain through the ellipsoid.

VERTICAL DATUM: ``height_km`` is *ellipsoidal* height — zero at the WGS84 reference surface,
which is what the geodetic→ECEF transform in ``projection.project`` consumes. Source rasters
almost always carry *orthometric* height (above the geoid / mean sea level; SRTM uses EGM96), and
the two differ by the geoid undulation, roughly −106 m to +85 m globally. DynamiX currently treats
them as interchangeable. That is invisible at globe scale and irrelevant to the WTMM itself, which
runs on the raster grid and never sees z — but it is a real datum conflation, recorded here rather
than left implicit. Correcting it means a geoid model, which is not warranted yet.
"""
from __future__ import annotations

import numpy as np

def load_topo_extrema(path):
    """Load a ``topo_wtmm.export_extrema(...)`` .npz of WTMM chains for the GUI's
    Topo-extrema overlay. Returns the raw polylines (lon/lat/DEPTH_km, with
    ``depth_km = -elev_m/1000`` so they sit on the relief surface) + per-chain
    attributes the filters key on. The GUI builds the actual ``RefLayer(kind='lines')``
    from a filtered subset, so this stays GUI-free (numpy only).

    Returns dict: ``h_segments`` (list of (M,3) lon/lat/depth), ``h_scale``, ``h_len``,
    ``v_segments``, ``v_persist``, ``n_scales``, ``scales``, ``region``.
    """
    d = np.load(path, allow_pickle=True)

    def _chains(xyz, off, attrs, ptattr=None):
        segs, kept = [], [[] for _ in attrs]
        ptsegs = [] if ptattr is not None else None
        for i in range(len(off) - 1):
            a, b = int(off[i]), int(off[i + 1])
            s = xyz[a:b]
            if len(s) < 2:                                # need >= 2 points for a polyline
                continue
            segs.append(np.column_stack([s[:, 0], s[:, 1], s[:, 2] / 1000.0]))
            for j, at in enumerate(attrs):
                kept[j].append(at[i])
            if ptattr is not None:
                ptsegs.append(np.asarray(ptattr[a:b]))       # per-POINT attribute sliced to this chain
        return segs, [np.asarray(k) for k in kept], ptsegs

    h_segs, (h_scale, h_len), _ = _chains(d["h_xyz"], d["h_off"], [d["h_scale"], d["h_len"]])
    v_pt_scale = d["v_scale"] if "v_scale" in d.files else None   # TRUE per-point scale index (additive)
    v_segs, (v_persist,), v_scale_segs = _chains(d["v_xyz"], d["v_off"], [d["v_persist"]], ptattr=v_pt_scale)
    out = dict(h_segments=h_segs, h_scale=h_scale, h_len=h_len,
               v_segments=v_segs, v_persist=v_persist,
               n_scales=int(d["n_scales"]), scales=np.asarray(d["scales"]),
               region=str(d["region"]))
    if v_scale_segs is not None:                              # per-chain true scale index (aligned to v_segments)
        out["v_scale_segments"] = v_scale_segs
    # schema v2: the branching cascade tree + H<->V incidence node table (see topo_wtmm.export_extrema)
    if "node_xyz" in d.files and "schema_version" in d.files and int(d["schema_version"]) >= 2:
        nx = np.asarray(d["node_xyz"], float)                       # (N,3) lon/lat/elev_m
        out.update(
            node_pts=np.column_stack([nx[:, 0], nx[:, 1], nx[:, 2] / 1000.0]),  # elev_m -> height_km
            node_scale=np.asarray(d["node_scale"]),
            node_mod=np.asarray(d["node_mod"]) if "node_mod" in d.files else None,  # |W| for the log-log plot
            node_hchain=np.asarray(d["node_hchain"]),               # -> index into h_segments (H<->V incidence)
            node_parent=np.asarray(d["node_parent"]),
            node_root=np.asarray(d["node_root"]),                   # tree/lineage id (coarsest ancestor)
            node_depth=np.asarray(d["node_depth"]),
            n_nodes=int(d["n_nodes"]), n_trees=int(d["n_trees"]))
    return out


def flatten_extrema(te):
    """Flatten a loaded topo-extrema dict (:func:`load_topo_extrema`) into selectable point clouds
    with a per-point chain index, for lasso/box/click picking of extrema.

    For each kind ``k`` in ``('h', 'v')`` returns ``{k}_pts`` (N, 3) lon/lat/depth and ``{k}_chain``
    (N,) int giving the chain each point belongs to (index into ``te[f'{k}_segments']``). Also
    ``n_h_chains`` / ``n_v_chains``. GUI-free (numpy only): the app projects the points through the
    live camera and maps a hit -> its chain (by-V = maxima line, by-H = within-scale contour).
    """
    def _flat(segs):
        if not segs:
            return np.zeros((0, 3), float), np.zeros(0, np.int64)
        pts = np.vstack([np.asarray(s, float) for s in segs])
        chain = np.concatenate([np.full(len(s), i, np.int64) for i, s in enumerate(segs)])
        return pts, chain

    h_pts, h_chain = _flat(te["h_segments"])
    v_pts, v_chain = _flat(te["v_segments"])
    out = dict(h_pts=h_pts, h_chain=h_chain, v_pts=v_pts, v_chain=v_chain,
               n_h_chains=len(te["h_segments"]), n_v_chains=len(te["v_segments"]))
    if "node_pts" in te:                                    # schema v2: branching cascade tree + H<->V incidence
        out.update(node_pts=np.asarray(te["node_pts"], float),
                   node_scale=np.asarray(te["node_scale"]),
                   node_mod=(np.asarray(te["node_mod"]) if te.get("node_mod") is not None else None),
                   node_hchain=np.asarray(te["node_hchain"]),
                   node_parent=np.asarray(te["node_parent"]),
                   node_root=np.asarray(te["node_root"]),
                   node_depth=np.asarray(te["node_depth"]),
                   n_nodes=int(te["n_nodes"]), n_trees=int(te["n_trees"]))
    return out
