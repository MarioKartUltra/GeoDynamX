# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Cheap filters over a computed WTMM stack.

Every one of these is a lookup or a predicate. None may recompute anything, and none may show a
progress bar -- if one ever needs to, it is a Transform wearing the wrong protocol.
"""
from __future__ import annotations

import numpy as np

from dynamix.core.chain_product import (materialize_selection, narrow_selection,
                                        selection_of)
from dynamix.model.param import Param, ParamKind

#: Orientation is axial: an NE-SW lineament is one orientation regardless of gradient polarity, so
#: 175 deg and 10 deg are 15 deg apart, not 165.
_AXIAL_PERIOD = 180.0


def _wrapped_delta(a, b, period: float = _AXIAL_PERIOD):
    """Smallest axial separation between orientations, elementwise."""
    d = np.abs(np.asarray(a, float) - float(b)) % period
    return np.minimum(d, period - d)


def _layers(result: dict) -> list:
    return result.get("extrema") or []


def _mask_layer(layer: dict, keep: np.ndarray) -> dict:
    """Apply a boolean mask to every equal-length array in one extrema layer."""
    n = len(keep)
    return {k: (v[keep] if hasattr(v, "shape") and v.shape[:1] == (n,) else v)
            for k, v in layer.items()}


class ScaleSelect:
    """Pick one scale out of the computed stack.

    This is the device that makes the whole precompute-then-filter design worth it: it is a list
    index, so scrubbing it redraws without touching the transform.
    """

    name = "scale_select"
    selection_aware = True
    params = (
        Param("scale_idx", ParamKind.INT, default=0, min=0, max=1023,
              soft_min=0, soft_max=15, label="Scale"),
    )

    def apply(self, result: dict, params: dict) -> dict:
        layers = _layers(result)
        if not layers:
            return result
        idx = int(np.clip(params["scale_idx"], 0, len(layers) - 1))
        out = dict(result)
        out["extrema"] = [layers[idx]]
        out["_scale_idx"] = idx
        # Geometry-reuse anchor: downstream filters mint a NEW extrema dict on every tweak, so
        # identity-keyed geometry caches always miss. This is the id-STABLE unfiltered layer (the
        # cached transform's own object) -- the canvas orders its H-lines once under this key and
        # masks per tweak instead of re-walking.
        out["_ext_base"] = layers[idx]
        runs = result.get("_hline_runs")
        if runs is not None and idx < len(runs):
            out["_ext_base_runs"] = runs[idx]     # the transform's worker-side ordering walk
        closed = result.get("_hline_closed")
        if closed is not None and idx < len(closed):
            # The follow detector's LINE_CLOSED flags (xsmurf search_lines) ride beside the
            # runs so the vector view can close closed rings' polylines.
            out["_ext_base_closed"] = closed[idx]
        scales = result.get("scales")
        if scales is not None and len(scales) > idx:
            out["_scale_px"] = float(np.asarray(scales)[idx])
        # Selection path: the scale pick is an index selection over
        # the product's own scale columns. The dict work above stays -- it is O(1) references,
        # and every legacy consumer keys on it.
        sel = selection_of(result)
        if sel is not None:
            prod = result["chain_product"]
            si = idx if len(layers) > 1 else int(result.get("_scale_idx", idx))
            narrow_selection(out, sel, h=np.asarray(prod["h_scale"]) == si,
                             iso_pts=np.asarray(prod["iso_scale"]) == si)
        return out


class OrientationWedge:
    """Keep extrema whose argument lies within a wedge of orientations.

    "Show me lineaments striking 040 +/- 15" is the question this answers, and it is the one
    structural geologists actually ask.
    """

    name = "orientation_wedge"
    selection_aware = True
    params = (
        Param("centre", ParamKind.ANGLE, default=0.0, wrap=_AXIAL_PERIOD,
              units="deg", label="Strike"),
        Param("half_width", ParamKind.FLOAT, default=90.0, min=0.0, max=90.0,
              soft_min=5.0, soft_max=90.0, units="deg", label="± width"),
        Param("north", ParamKind.CHOICE, default="grid",
              choices=("grid", "true", "magnetic"), label="North"),
    )

    def apply(self, result: dict, params: dict) -> dict:
        half = float(params["half_width"])
        if half >= 90.0:
            return result                     # full wedge: nothing to do, skip the work
        centre = float(params["centre"])
        # Selection path: a pure per-point wedge over the product's static arg columns.
        # A NaN argument (a layer that never carried one) passes, as the dict path passes
        # whole layers without ``arg``.
        sel = selection_of(result)
        if sel is not None:
            prod = result["chain_product"]

            def _inside(arg):
                arg = np.asarray(arg, dtype=float)
                deg = np.degrees(arg) % _AXIAL_PERIOD
                with np.errstate(invalid="ignore"):
                    keep = _wrapped_delta(deg, centre) <= half
                return keep | np.isnan(arg)

            out = dict(result)
            out["_wedge"] = (centre, half, params["north"])
            return narrow_selection(out, sel, h_pts=_inside(prod["h_arg"]),
                                    iso_pts=_inside(prod["iso_arg"]))
        out = dict(result)
        kept = []
        for layer in _layers(result):
            arg = layer.get("arg")
            if arg is None:
                kept.append(layer)
                continue
            deg = np.degrees(np.asarray(arg, float)) % _AXIAL_PERIOD
            kept.append(_mask_layer(layer, _wrapped_delta(deg, centre) <= half))
        out["extrema"] = kept
        out["_wedge"] = (centre, half, params["north"])
        return out


class ModulusThreshold:
    """Keep extrema above a fraction of the layer's peak modulus.

    Expressed as a fraction rather than an absolute so one setting is meaningful across scales --
    |W| falls off with scale, and an absolute floor would silently empty the coarse layers.
    """

    name = "modulus_threshold"
    selection_aware = True
    params = (
        Param("frac", ParamKind.FLOAT, default=0.0, min=0.0, max=1.0,
              soft_min=0.0, soft_max=0.5, label="Min |W| (frac of peak)"),
    )

    def apply(self, result: dict, params: dict) -> dict:
        frac = float(params["frac"])
        if frac <= 0.0:
            return result
        # Selection path: per-point compare against the product's STATIC per-scale peak
        # (``scale_peak`` -- the transform's own layer maximum), so the knob means the same
        # thing whatever ran before it. Value-identical to the dict path in any chain that
        # has not removed points upstream.
        sel = selection_of(result)
        if sel is not None:
            prod = result["chain_product"]
            peak = np.asarray(prod["scale_peak"], dtype=float)
            pt_scale = np.repeat(np.asarray(prod["h_scale"], np.int64),
                                 np.diff(prod["h_off"]))
            with np.errstate(invalid="ignore"):
                h_keep = np.asarray(prod["h_mod"], float) >= frac * peak[pt_scale]
                iso_keep = (np.asarray(prod["iso_mod"], float)
                            >= frac * peak[np.asarray(prod["iso_scale"], np.int64)])
            return narrow_selection(dict(result), sel, h_pts=h_keep, iso_pts=iso_keep)
        out = dict(result)
        kept = []
        for layer in _layers(result):
            mod = layer.get("mod")
            if mod is None or len(mod) == 0:
                kept.append(layer)
                continue
            mod = np.asarray(mod, float)
            kept.append(_mask_layer(layer, mod >= frac * float(np.nanmax(mod))))
        out["extrema"] = kept
        return out


class HLineLength:
    """Keep extrema on H-lines whose point count is within a range.

    Part of EQSelect's fast filter stack: short H-lines are usually linking coincidences, and very
    long ones on mosaic data are often stitching seams -- both ends of the range earn their knob.
    Length here is the POINT COUNT of the line at this scale (the H-line family the chain filters'
    cross-scale lengths do not measure).

    Isolated extrema (``line_id == -1``) pass through untouched: they are not lines, and their
    display is the orphan-dots toggle's business, not a length predicate's.
    """

    name = "hline_length"
    selection_aware = True
    params = (
        Param("min_len", ParamKind.INT, default=1, min=1, max=100000,
              soft_min=1, soft_max=64, label="Min points"),
        Param("max_len", ParamKind.INT, default=0, min=0, max=1000000,
              soft_min=0, soft_max=4096, label="Max points (0 = no cap)"),
    )

    def apply(self, result: dict, params: dict) -> dict:
        lo = int(params["min_len"])
        hi = int(params["max_len"])
        if lo <= 1 and hi == 0:
            return result
        # Selection path: a pure range over the ORIGINAL per-chain length column (EQSelect's
        # static-metric doctrine -- the line's own length, not what upstream filters left of
        # it). Singletons are not lines and pass untouched, exactly as below.
        sel = selection_of(result)
        if sel is not None:
            prod = result["chain_product"]
            h_len = np.asarray(prod["h_len"])
            keep = h_len >= lo
            if hi > 0:
                keep = keep & (h_len <= hi)
            return narrow_selection(dict(result), sel, h=keep)
        out = dict(result)
        kept = []
        for layer in _layers(result):
            lid = np.asarray(layer.get("line_id", ()))
            if lid.size == 0:
                kept.append(layer)
                continue
            labelled = lid >= 0
            n_of = np.zeros(lid.shape, dtype=np.int64)
            if labelled.any():
                counts = np.bincount(lid[labelled])
                n_of[labelled] = counts[lid[labelled]]
            keep = ~labelled | ((n_of >= lo) & ((n_of <= hi) if hi > 0 else True))
            kept.append(_mask_layer(layer, keep))
        out["extrema"] = kept
        return out

    def reading(self, result: dict, params: dict) -> str:
        """Live feedback so the H-line LENGTH filter reads as clearly present as the V-chain ones
        (it exists, this makes
        it show its own data). Point-count distribution of the transform's H-lines, from the
        chain product's static ``h_len`` column."""
        prod = result.get("chain_product")
        if prod is None or not len(prod.get("h_len", ())):
            return "no H-lines — run the transform"
        h_len = np.asarray(prod["h_len"])
        return f"{h_len.size} H-lines · points ∈ [{int(h_len.min())}, {int(h_len.max())}]"

    def data_hints(self, result: dict, params: dict) -> dict:
        """Auto-range ``min_len``'s slider to the observed H-line point counts (snap to the
        floor, pass-all, like the chain filters)."""
        prod = result.get("chain_product")
        if prod is None or not len(prod.get("h_len", ())):
            return {}
        h_len = np.asarray(prod["h_len"])
        lo, hi = float(h_len.min()), float(h_len.max())
        return {"min_len": (lo, hi, lo)}


def _per_line(lid: np.ndarray, val: np.ndarray, how: str) -> np.ndarray:
    """Per-POINT value of its H-line's aggregate (``sup`` or ``mean``) of ``val``; a singleton
    (``line_id == -1``) is a one-point line, so it gets its own value back."""
    out = val.copy()
    labelled = lid >= 0
    if not labelled.any():
        return out
    l = lid[labelled]
    n = int(l.max()) + 1
    if how == "mean":
        agg = np.bincount(l, weights=val[labelled], minlength=n) / np.maximum(np.bincount(l, minlength=n), 1)
    else:
        agg = np.full(n, -np.inf)
        np.maximum.at(agg, l, val[labelled])
    out[labelled] = agg[l]
    return out


def _chain_sup_at(layer: dict, si: int, chains: list, nx: int) -> np.ndarray:
    """Per point: the max |W| over ALL scales of the chain it stands on at scale ``si``, or its
    own |W| when no chain passes through it. Chains are finest-anchored (``chains2d``: no chain
    is born at mid-scale), so chain point ``j`` sits at scale index ``j``."""
    mod = np.asarray(layer["mod"], float)
    out = mod.copy()
    pos = np.asarray(layer["y"], np.int64) * nx + np.asarray(layer["x"], np.int64)
    keys, vals = [], []
    for c in chains:
        cx, cy, cm = np.asarray(c["x"], np.int64), np.asarray(c["y"], np.int64), np.asarray(c["mod"], float)
        if si < cx.size:
            keys.append(cy[si] * nx + cx[si])
            vals.append(np.nanmax(cm))
    if not keys:
        return out
    keys, vals = np.asarray(keys), np.asarray(vals)
    order = np.argsort(keys, kind="stable")
    keys, vals = keys[order], vals[order]
    idx = np.searchsorted(keys, pos)
    hit = (idx < keys.size) & (keys[np.minimum(idx, keys.size - 1)] == pos)
    out[hit] = np.maximum(out[hit], vals[idx[hit]])
    return out


def _segments_keep(layer: dict, shape, floor: float) -> np.ndarray:
    """Keep mask for the ``segments`` mode: every ordered H-line run is cut at the local minima
    of |W| along it into local-max-to-local-min sections; a section survives when its own peak
    clears ``floor``; a cutting minimum is a node shared by its two sections and stays while
    either survives. Points on no run (singletons, one-point lines) are judged alone."""
    from dynamix.core.hlines import hline_runs

    mod = np.asarray(layer["mod"], float)
    keep = mod >= floor
    for run in hline_runs(layer, shape):
        m = mod[run]
        n = m.size
        interior = np.arange(1, n - 1)
        is_min = (m[interior] < m[interior - 1]) & (m[interior] <= m[interior + 1])
        cuts = [0, *interior[is_min].tolist(), n - 1]
        run_keep = np.zeros(n, dtype=bool)
        for a, b in zip(cuts[:-1], cuts[1:]):
            if m[a:b + 1].max() >= floor:
                run_keep[a:b + 1] = True
        keep[run] = run_keep
    return keep


class HLineModulus:
    """Keep H-lines -- not points -- above a fraction of the layer's peak modulus.

    ``modulus_threshold`` judges every extremum alone, so a strong lineament loses its weak
    flanks and breaks into pieces. This one judges the line:

    * ``sup`` -- the line's max |W| at this scale.
    * ``sup_all_scales`` -- the max |W| over every scale of the chains threading the line's points
      (a line that is weak here but strong at another scale is kept); needs ``chains``. Its
      "peak" is the STACK's peak |W| -- both sides over all scales -- so at a coarse scale it
      is stricter than ``sup``, whose peak is that scale's own.
    * ``mean`` -- the mean |W| along the line at this scale.
    * ``segments`` -- the line is cut at the local minima of |W| along it into local-max-to-
      local-min sections; each section is judged by its own peak, and the minima stay as the
      nodes joining the survivors, so one weak stretch is removed without breaking the H-chain
      into two (``line_id`` is untouched).

    A singleton is a one-point line and gets the point-wise rule, so on isolated extrema this
    device IS ``modulus_threshold``.
    """

    name = "hline_modulus"
    selection_aware = True
    params = (
        Param("frac", ParamKind.FLOAT, default=0.0, min=0.0, max=1.0,
              soft_min=0.0, soft_max=0.5, label="Min |W| (frac of peak)"),
        Param("mode", ParamKind.CHOICE, default="sup",
              choices=("sup", "sup_all_scales", "mean", "segments"), label="Per line"),
    )

    def apply(self, result: dict, params: dict) -> dict:
        frac = float(params["frac"])
        if frac <= 0.0:
            return result
        mode = str(params["mode"])
        sel = selection_of(result)
        if sel is not None and mode in ("sup", "mean"):
            # Selection path: the line verdict is a range over the static h_mod_sup/h_mod_mean
            # columns against the per-scale peak; a singleton is a one-point line and gets the
            # point-wise rule, exactly as the dict path gives it.
            prod = result["chain_product"]
            col = np.asarray(prod["h_mod_sup" if mode == "sup" else "h_mod_mean"], float)
            peak = np.asarray(prod["scale_peak"], dtype=float)
            with np.errstate(invalid="ignore"):
                h_keep = col >= frac * peak[np.asarray(prod["h_scale"], np.int64)]
                iso_keep = (np.asarray(prod["iso_mod"], float)
                            >= frac * peak[np.asarray(prod["iso_scale"], np.int64)])
            return narrow_selection(dict(result), sel, h=h_keep, iso_pts=iso_keep)
        if sel is not None:
            # segments / sup_all_scales are genuinely path-shaped cuts, not metric ranges:
            # materialize the honest dicts once and run the dict path over them. Replacing
            # ``extrema`` below invalidates the selection, so every downstream consumer falls
            # back with us rather than drawing a stale subset.
            result = materialize_selection(result)
        layers = _layers(result)
        chains = result.get("chains") or []
        # Absolute scale index per layer -- the same pairing MinVChains documents: after
        # ScaleSelect the list is one layer long at `_scale_idx`, otherwise position is index.
        base = result.get("_scale_idx")
        sis = [int(base)] if (base is not None and len(layers) == 1) else range(len(layers))
        stack_peak = max([float(np.nanmax(np.asarray(l["mod"], float)))
                          for l in layers if l.get("mod") is not None and len(l["mod"])]
                         + [float(np.nanmax(np.asarray(c["mod"], float))) for c in chains if len(c["mod"])],
                         default=0.0)
        out = dict(result)
        kept = []
        for si, layer in zip(sis, layers):
            mod = layer.get("mod")
            if mod is None or len(mod) == 0:
                kept.append(layer)
                continue
            mod = np.asarray(mod, float)
            lid = np.asarray(layer.get("line_id", np.full(mod.size, -1)), np.int64)
            shape = result.get("_shape") or (int(layer["y"].max()) + 1, int(layer["x"].max()) + 1)
            if mode == "segments":
                keep = _segments_keep(layer, shape, frac * float(np.nanmax(mod)))
            elif mode == "sup_all_scales":
                val = _chain_sup_at(layer, si, chains, int(shape[1]))
                keep = _per_line(lid, val, "sup") >= frac * stack_peak
            else:
                keep = _per_line(lid, mod, mode) >= frac * float(np.nanmax(mod))
            kept.append(_mask_layer(layer, keep))
        out["extrema"] = kept
        return out

    def reading(self, result: dict, params: dict) -> str:
        """Live feedback for the per-LINE H modulus filter: the per-line sup |W|
        distribution from the chain product's static ``h_mod_sup`` column, so this reads as
        present and data-driven like the V-chain modulus filter."""
        prod = result.get("chain_product")
        col = prod.get("h_mod_sup") if prod is not None else None
        if col is None or not len(col):
            return "no H-lines — run the transform"
        col = np.asarray(col, float)
        return f"{col.size} H-lines · sup|W| ∈ [{float(col.min()):.2f}, {float(col.max()):.2f}]"


class HLineHolder:
    """Keep H-lines whose Hölder exponent α lies within a range.

    α is the analysis' own per-point ``alpha`` (``mz_edges``: the decay of the maxima across the
    levels along each chain, the median over the line, ``core.mz_lastwave.select.chain_alpha``),
    so the knob selects edges by singularity type -- 0 a step, -1 a line, -2 a point -- as
    Mallat & Zhong's §VII discriminates them. A layer without ``alpha`` passes whole; a point
    whose chain could not be fitted (NaN) passes while ``keep_unfitted`` is on.
    """

    name = "hline_holder"
    params = (
        Param("min_alpha", ParamKind.FLOAT, default=-3.0, min=-3.0, max=3.0,
              soft_min=-3.0, soft_max=2.0, label="Min α"),
        Param("max_alpha", ParamKind.FLOAT, default=3.0, min=-3.0, max=3.0,
              soft_min=-2.0, soft_max=3.0, label="Max α"),
        Param("keep_unfitted", ParamKind.BOOL, default=True, label="Keep unfitted"),
    )

    def apply(self, result: dict, params: dict) -> dict:
        lo, hi = float(params["min_alpha"]), float(params["max_alpha"])
        keep_nan = bool(params["keep_unfitted"])
        if lo <= -3.0 and hi >= 3.0 and keep_nan:
            return result
        out = dict(result)
        kept = []
        for layer in _layers(result):
            alpha = layer.get("alpha")
            if alpha is None:
                kept.append(layer)
                continue
            alpha = np.asarray(alpha, float)
            with np.errstate(invalid="ignore"):
                keep = (alpha >= lo) & (alpha <= hi)
            kept.append(_mask_layer(layer, keep | (np.isnan(alpha) & keep_nan)))
        out["extrema"] = kept
        return out
