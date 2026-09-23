# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The draw-ready CSR chain product: EQSelect's fast extrema representation, as a first-class form.

EQSelect scrubbed
2M-point extrema reps in real time because its representation was ordered ONCE (at export), its
filters were index selections over a per-chain metric table, and its draws were bounded. DynamiX
grafted pieces of that onto the device pipeline (``_ext_base``/``_hline_runs`` stamps); this module
makes the representation canonical instead of a retrofit.

One dict, built worker-side by the transform, immutable afterwards (the engine cache's own law):

* **H chains** -- each scale's ordered maxima-line runs (``dynamix.core.hlines.hline_runs`` is the
  ordering authority -- the same ``_order_lines`` walk EQSelect's export runs once), flattened CSR:
  ``h_x``/``h_y``/``h_mod``/``h_arg``/``h_src`` per point (``h_src`` maps back into the layer's
  own arrays), ``h_off`` boundaries, and per-chain ``h_scale``/``h_len``/``h_mod_sup``/
  ``h_mod_mean``. Points on no run -- singletons (``line_id == -1``) and stranded labelled
  points alike -- are not lines and ride a separate flat ``iso_*`` side-table instead
  (``iso_x``/``iso_y``/``iso_mod``/``iso_arg``/``iso_scale``/``iso_src``), so per-point filters
  and the orphan-dots display cover them without faking one-point chains. Every layer point
  lands in exactly one of ``h_src``/``iso_src`` -- the partition selection materialization
  relies on. ``scale_peak`` is each layer's own max |W| (lines and singletons together), the
  static reference the fraction-of-peak filters select against.
* **V chains** -- ``result["chains"]`` flattened the same way: ``v_x``/``v_y``/``v_mod``/
  ``v_scale``/``v_arg`` per point, ``v_off``, and per-chain ``v_persist``/``v_mod_finest``/
  ``v_mod_sup``/``v_holder_ols``/``v_holder_max``/``v_max_log2_mod``/``v_arg_finest``. The chain
  dicts drop ``arg`` (the verbatim backend's ``chains2d`` never carried it -- EQSelect's
  2026-08-01 spec, Phase B), so the argument is JOINED here from the extrema layers: chain point
  ``j`` sits at scale index ``j`` (chains are finest-anchored), and its ``(x, y)`` is looked up in
  ``extrema[j]``. NaN where the join finds nothing.

Metric columns must agree with the estimators the filter devices already trust
(``dynamix.core.chain_stats``) to float tolerance: a filter switching from the dict path to this
table must keep and drop the SAME chains. ``tests/test_chain_product.py`` pins that agreement.

Coordinates are PIXEL indices -- the form every view masks and projects from. Frame-unit
conversion happens only at npz export, exactly as EQSelect's ``export_chains_npz`` does it.
"""
from __future__ import annotations

import json

import numpy as np

from dynamix.core.hlines import hline_runs


def _per_chain_ols(chain_id, n_chains, ls, lm):
    """Vectorized per-chain OLS slope of ``lm`` against ``ls`` over jointly-finite pairs --
    ``chain_stats._chain_ols_holder``'s semantics (NaN below 2 pairs), one ``bincount`` pass."""
    finite = np.isfinite(ls) & np.isfinite(lm)
    w = finite.astype(np.float64)
    n = np.bincount(chain_id, weights=w, minlength=n_chains)
    sx = np.bincount(chain_id, weights=np.where(finite, ls, 0.0), minlength=n_chains)
    sy = np.bincount(chain_id, weights=np.where(finite, lm, 0.0), minlength=n_chains)
    sxx = np.bincount(chain_id, weights=np.where(finite, ls * ls, 0.0), minlength=n_chains)
    sxy = np.bincount(chain_id, weights=np.where(finite, ls * lm, 0.0), minlength=n_chains)
    denom = n * sxx - sx * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n * sxy - sx * sy) / denom
    slope[(n < 2) | ~np.isfinite(slope)] = np.nan
    return slope


def _per_chain_max(chain_id, n_chains, values):
    """Per-chain max of the FINITE entries of ``values``; NaN for a chain with none."""
    out = np.full(n_chains, -np.inf)
    finite = np.isfinite(values)
    if finite.any():
        np.maximum.at(out, chain_id[finite], values[finite])
    out[np.isneginf(out)] = np.nan
    return out


def _derive_h_metrics(h_mod, h_off, h_len):
    """``(h_mod_sup, h_mod_mean)`` from the flat H arrays -- one ``reduceat`` pass each."""
    if h_mod.size:
        starts = h_off[:-1]
        return np.maximum.reduceat(h_mod, starts), np.add.reduceat(h_mod, starts) / h_len
    return np.zeros(0), np.zeros(0)


def _derive_v_metrics(v_mod, v_off, chain_id, n_v, ls, lm):
    """``(v_mod_finest, v_mod_sup, v_holder_ols, v_holder_max, v_max_log2_mod)`` from the flat
    V arrays plus the per-point ``log2_scales``/``log2_mod`` columns. Shared by the builder and
    the npz loader so both derive the identical metric table from the identical CSR."""
    if v_mod.size:
        v_mod_finest = v_mod[v_off[:-1]]
        v_mod_sup = np.maximum.reduceat(v_mod, v_off[:-1])
    else:
        v_mod_finest = np.zeros(0)
        v_mod_sup = np.zeros(0)
    v_holder_ols = _per_chain_ols(chain_id, n_v, ls, lm) if n_v else np.zeros(0)
    v_max_log2_mod = _per_chain_max(chain_id, n_v, lm) if n_v else np.zeros(0)
    # Max LOCAL slope (chain_stats._chain_max_slope_holder): diffs over the raw arrays, cut at
    # chain boundaries; only non-finite step slopes are dropped, NaN when none survive.
    if v_mod.size and ls.size >= 2:
        within = chain_id[:-1] == chain_id[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            slopes = np.diff(lm) / np.diff(ls)
        v_holder_max = _per_chain_max(chain_id[:-1][within], n_v, slopes[within])
    else:
        v_holder_max = np.full(n_v, np.nan)
    return v_mod_finest, v_mod_sup, v_holder_ols, v_holder_max, v_max_log2_mod


def build_chain_product(extrema, chains, scales, shape, *, runs=None) -> dict:
    """Build the CSR chain product from a transform's ``extrema``/``chains``/``scales``.

    ``runs`` -- the per-scale ``hline_runs`` lists the transform already computed
    (``result["_hline_runs"]``); passing them makes the H side a pure concatenation. Without
    them the ordering walk runs here, so call this on the worker, never at a landing.
    """
    ny, nx = int(shape[0]), int(shape[1])

    # --- H: ordered runs per scale, flattened; the leftovers become the iso side -------------
    # ``has_sub``: the 2026-09-20 interpolate knob stamped float subpixel positions on every
    # extrema layer (all-or-nothing -- the refinement is a whole-stack pass); the product then
    # carries parallel FLOAT display columns (h_xf/h_yf, iso_xf/iso_yf) alongside the integer
    # identity columns, which every mask/pick/projection keeps using untouched.
    has_sub = bool(extrema) and "x_sub" in extrema[0]
    h_xs, h_ys, h_mods, h_args, h_srcs, h_lens, h_scales = [], [], [], [], [], [], []
    h_xfs, h_yfs, iso_xfs, iso_yfs = [], [], [], []
    iso_xs, iso_ys, iso_mods, iso_args, iso_scs, iso_srcs = [], [], [], [], [], []
    scale_peak = np.full(len(extrema), np.nan)
    for si, layer in enumerate(extrema):
        lx = np.asarray(layer["x"], dtype=np.int64)
        ly = np.asarray(layer["y"], dtype=np.int64)
        lmod = np.asarray(layer["mod"], dtype=np.float64)
        larg = layer.get("arg")
        larg = np.asarray(larg, dtype=np.float64) if larg is not None \
            else np.full(lx.size, np.nan)
        if has_sub:
            lxf = np.asarray(layer["x_sub"], dtype=np.float64)
            lyf = np.asarray(layer["y_sub"], dtype=np.float64)
        if lmod.size and np.isfinite(lmod).any():
            scale_peak[si] = float(np.nanmax(lmod))
        layer_runs = runs[si] if runs is not None and si < len(runs) \
            else hline_runs(layer, shape)
        covered = np.zeros(lx.size, dtype=bool)
        for run in layer_runs:
            covered[run] = True
            h_xs.append(lx[run])
            h_ys.append(ly[run])
            h_mods.append(lmod[run])
            h_args.append(larg[run])
            h_srcs.append(run.astype(np.int64))
            h_lens.append(run.size)
            h_scales.append(si)
            if has_sub:
                h_xfs.append(lxf[run])
                h_yfs.append(lyf[run])
        iso = np.flatnonzero(~covered)
        if iso.size:
            iso_xs.append(lx[iso])
            iso_ys.append(ly[iso])
            iso_mods.append(lmod[iso])
            iso_args.append(larg[iso])
            iso_scs.append(np.full(iso.size, si, dtype=np.int64))
            iso_srcs.append(iso)
            if has_sub:
                iso_xfs.append(lxf[iso])
                iso_yfs.append(lyf[iso])

    h_len = np.asarray(h_lens, dtype=np.int64)
    h_off = np.concatenate([[0], np.cumsum(h_len)]).astype(np.int64)
    h_x = np.concatenate(h_xs) if h_xs else np.zeros(0, dtype=np.int64)
    h_y = np.concatenate(h_ys) if h_ys else np.zeros(0, dtype=np.int64)
    h_mod = np.concatenate(h_mods) if h_mods else np.zeros(0)
    h_arg = np.concatenate(h_args) if h_args else np.zeros(0)
    h_src = np.concatenate(h_srcs) if h_srcs else np.zeros(0, dtype=np.int64)
    h_mod_sup, h_mod_mean = _derive_h_metrics(h_mod, h_off, h_len)
    iso_x = np.concatenate(iso_xs) if iso_xs else np.zeros(0, dtype=np.int64)
    iso_y = np.concatenate(iso_ys) if iso_ys else np.zeros(0, dtype=np.int64)
    iso_mod = np.concatenate(iso_mods) if iso_mods else np.zeros(0)
    iso_arg = np.concatenate(iso_args) if iso_args else np.zeros(0)
    iso_scale = np.concatenate(iso_scs) if iso_scs else np.zeros(0, dtype=np.int64)
    iso_src = np.concatenate(iso_srcs) if iso_srcs else np.zeros(0, dtype=np.int64)

    # --- V: chains flattened ------------------------------------------------------------------
    n_v = len(chains)
    v_counts = np.asarray([len(np.asarray(c["x"])) for c in chains], dtype=np.int64)
    v_off = np.concatenate([[0], np.cumsum(v_counts)]).astype(np.int64)

    def _vcat(key, dtype):
        if n_v:
            return np.concatenate([np.asarray(c[key], dtype=dtype) for c in chains])
        return np.zeros(0, dtype=dtype)

    v_x = _vcat("x", np.int64)
    v_y = _vcat("y", np.int64)
    v_mod = _vcat("mod", np.float64)
    v_scale = (np.concatenate([np.arange(k, dtype=np.int64) for k in v_counts])
               if n_v else np.zeros(0, dtype=np.int64))
    chain_id = np.repeat(np.arange(n_v, dtype=np.int64), v_counts)

    v_persist = np.asarray([len(c["mod"]) for c in chains], dtype=np.int64)
    ls = _vcat("log2_scales", np.float64)
    lm = _vcat("log2_mod", np.float64)
    v_mod_finest, v_mod_sup, v_holder_ols, v_holder_max, v_max_log2_mod = \
        _derive_v_metrics(v_mod, v_off, chain_id, n_v, ls, lm)

    # Argument join: per scale, look each chain point up in its own extrema layer.
    v_arg = np.full(v_x.size, np.nan)
    pos = v_y * nx + v_x
    for si in np.unique(v_scale) if v_x.size else ():
        if si >= len(extrema):
            continue
        layer = extrema[int(si)]
        larg = layer.get("arg")
        if larg is None:
            continue
        lpos = np.asarray(layer["y"], np.int64) * nx + np.asarray(layer["x"], np.int64)
        if lpos.size == 0:
            continue
        order = np.argsort(lpos, kind="stable")
        lpos_sorted = lpos[order]
        larg_sorted = np.asarray(larg, dtype=np.float64)[order]
        sel = np.flatnonzero(v_scale == si)
        idx = np.searchsorted(lpos_sorted, pos[sel])
        ok = (idx < lpos_sorted.size)
        ok[ok] &= lpos_sorted[idx[ok]] == pos[sel][ok]
        v_arg[sel[ok]] = larg_sorted[idx[ok]]
    if v_arg.size and n_v:
        v_arg_finest = np.where(v_counts > 0,
                                v_arg[np.minimum(v_off[:-1], v_arg.size - 1)], np.nan)
    else:
        v_arg_finest = np.full(n_v, np.nan)

    out_sub = {}
    if has_sub:
        out_sub = {
            "h_xf": np.concatenate(h_xfs) if h_xfs else np.zeros(0),
            "h_yf": np.concatenate(h_yfs) if h_yfs else np.zeros(0),
            "iso_xf": np.concatenate(iso_xfs) if iso_xfs else np.zeros(0),
            "iso_yf": np.concatenate(iso_yfs) if iso_yfs else np.zeros(0),
        }
    return {
        **out_sub,
        "h_x": h_x, "h_y": h_y, "h_off": h_off, "h_mod": h_mod, "h_arg": h_arg,
        "h_src": h_src,
        "h_scale": np.asarray(h_scales, dtype=np.int64), "h_len": h_len,
        "h_mod_sup": h_mod_sup, "h_mod_mean": h_mod_mean,
        "iso_x": iso_x, "iso_y": iso_y, "iso_mod": iso_mod, "iso_arg": iso_arg,
        "iso_scale": iso_scale, "iso_src": iso_src, "scale_peak": scale_peak,
        "v_x": v_x, "v_y": v_y, "v_off": v_off, "v_mod": v_mod, "v_scale": v_scale,
        "v_arg": v_arg, "v_persist": v_persist, "v_mod_finest": v_mod_finest,
        "v_mod_sup": v_mod_sup, "v_holder_ols": v_holder_ols, "v_holder_max": v_holder_max,
        "v_max_log2_mod": v_max_log2_mod, "v_arg_finest": v_arg_finest,
        "scales": np.asarray(scales, dtype=np.float64).copy(),
        "n_scales": len(np.asarray(scales).reshape(-1)),
        "shape": (ny, nx),
    }


def attach_chain_product(result: dict) -> dict:
    """Stamp ``_hline_runs`` (when absent) and ``chain_product`` onto a transform result,
    in place, and return it.

    Called by a Transform's own ``compute`` -- i.e. on the worker, the one place the ordering
    walk and the metric table are allowed to cost anything. The stamped product is immutable
    afterwards by the engine cache's own law, which is what lets every filter tweak reuse it
    by identity.
    """
    shape = tuple(result["_shape"])[:2]
    runs = result.get("_hline_runs")
    if runs is None:
        runs = [hline_runs(layer, shape) for layer in result.get("extrema") or []]
        result["_hline_runs"] = runs
    scales = result.get("scales")
    result["chain_product"] = build_chain_product(
        result.get("extrema") or [], result.get("chains") or [],
        scales if scales is not None else [], shape, runs=runs)
    result["_selection"] = {
        "keep_h": None, "keep_v": None, "keep_h_pts": None, "keep_iso_pts": None,
        "extrema_ref": id(result.get("extrema")), "chains_ref": id(result.get("chains")),
    }
    return result


# ---------------------------------------------------------------------------------------------
# Index selections over the product.
#
# The state is EQSelect's filter model: ``keep_h``/``keep_v`` are sorted index arrays into the
# chain tables (None = all), ``keep_h_pts``/``keep_iso_pts`` boolean masks over the flat point
# arrays (None = all). Metrics are STATIC columns of the immutable product, so composition is
# pure intersection and filter order cannot change what a knob value means.
#
# The state is identity-bound to the result's own ``extrema``/``chains`` objects. Any device
# that rewrites those outside this model (groups, topology, classify, the segments cut) makes
# ``selection_of`` return None from then on -- stale selections are structurally impossible, and
# consumers that see None simply take the legacy dict path.
# ---------------------------------------------------------------------------------------------

#: EQSelect's ``_TE_MAX`` (app_window.py:4915): the drawn-chain ceiling that keeps dense/global
#: exports responsive. Views cap at this and SAY SO -- never a silent truncation.
DRAW_CAP = 6000


def _capped(idx, metric, cap):
    if idx is None:
        idx = np.arange(metric.size, dtype=np.int64)
    total = int(idx.size)
    if total <= int(cap):
        return idx, total
    keep = np.argsort(metric[idx], kind="stable")[::-1][: int(cap)]
    return np.sort(idx[keep]), total


def capped_h(product, keep_h, cap=DRAW_CAP):
    """``(indices, total)`` -- the current H selection bounded to ``cap`` chains, keeping the
    LONGEST (EQSelect's own rule). ``total`` is the uncapped selected count, for the honest
    "drawing X of Y" reading."""
    return _capped(keep_h, np.asarray(product["h_len"]), cap)


def capped_v(product, keep_v, cap=DRAW_CAP):
    """``(indices, total)`` -- the V selection bounded to ``cap``, keeping the MOST PERSISTENT."""
    return _capped(keep_v, np.asarray(product["v_persist"]), cap)


#: The one shared stub ``stub_capped_chains`` substitutes -- read-only by every consumer
#: (``chain.get(...)`` on an empty dict), so sharing one object is safe and keeps the capped
#: list allocation-free beyond the list itself.
_CHAIN_STUB: dict = {}


def stub_capped_chains(chains, cap=DRAW_CAP):
    """``(chains, note)`` with the least persistent chains replaced by empty stubs once the
    list exceeds ``cap`` -- bounded draw for a consumer whose chain identity is LIST POSITION
    (the scene's picking lookup), so nothing may be renumbered or removed. Under the cap the
    ORIGINAL list comes back untouched with ``note`` None."""
    if len(chains) <= int(cap):
        return chains, None
    persist = np.fromiter((len(c.get("mod", ())) for c in chains), np.int64, len(chains))
    keep = np.argsort(persist, kind="stable")[::-1][: int(cap)]
    mask = np.zeros(len(chains), dtype=bool)
    mask[keep] = True
    capped = [c if mask[i] else _CHAIN_STUB for i, c in enumerate(chains)]
    note = f"drawing {int(cap)} of {len(chains)} V chains — tighten filters to see the rest"
    return capped, note


#: DISABLED 2026-09-14 — do NOT flip back to False without a redesign. The
#: filters-as-index-selections mechanism this gates (``narrow_selection`` in the filter devices,
#: ``materialize_selection`` in ``engine.resolve``, the canvas/scene fast paths) was unsound when
#: a selection-aware filter interleaves with a non-aware one that rewrites ``extrema`` (min_vchains):
#: the ``h_src`` back-references then index an already-shrunk layer and raise IndexError inside a
#: Qt signal handler, silently wedging the canvas — this IS the min |W| / scale-bar freeze. With it disabled every filter takes
#: its honest dict path (correct, ~3 ms/tweak, measured). The chain_product build + schema-v4 npz
#: export + draw-cap helpers are all independent of this gate and stay live. Re-enabling it is the
#: progressive (finest-first) compute redesign's job, which must first solve the lifecycle problem.
_SELECTION_ENABLED = False


def selection_of(result) -> dict | None:
    """The live selection state, or None when there is none or it cannot be trusted.

    Returns None unconditionally while :data:`_SELECTION_ENABLED` is False (see that flag) — so
    every filter device's ``if sel is not None:`` fast path is dormant and the honest dict path
    runs instead. The retained logic below is what a future redesign re-enables.

    Tolerates any ``result`` shape: a chain can end field-tail (the noise device hands the next
    transform a FIELD clone, and a noise-only chain lands that clone directly), so a non-dict
    simply has no selection -- it must never raise inside a worker resolve."""
    if not _SELECTION_ENABLED:
        return None
    if not isinstance(result, dict):
        return None
    sel = result.get("_selection")
    if sel is None or result.get("chain_product") is None:
        return None
    if sel.get("extrema_ref") != id(result.get("extrema")):
        return None
    if sel.get("chains_ref") != id(result.get("chains")):
        return None
    return sel


def narrow_selection(out, sel, *, h=None, v=None, h_pts=None, iso_pts=None) -> dict:
    """Compose a filter's own narrowing (each a boolean mask over the FULL table or point
    array) with the incoming state ``sel``, stamping the result onto ``out`` bound to ``out``'s
    own ``extrema``/``chains`` identities. Returns ``out``."""
    def _idx(prev, mask):
        if mask is None:
            return prev
        mask = np.asarray(mask, dtype=bool)
        if prev is None:
            return np.flatnonzero(mask)
        return prev[mask[prev]]

    def _pts(prev, mask):
        if mask is None:
            return prev
        mask = np.asarray(mask, dtype=bool)
        return mask if prev is None else (prev & mask)

    out["_selection"] = {
        "keep_h": _idx(sel["keep_h"], h),
        "keep_v": _idx(sel["keep_v"], v),
        "keep_h_pts": _pts(sel["keep_h_pts"], h_pts),
        "keep_iso_pts": _pts(sel["keep_iso_pts"], iso_pts),
        "extrema_ref": id(out.get("extrema")),
        "chains_ref": id(out.get("chains")),
    }
    return out


def _mask_layer_arrays(layer: dict, keep: np.ndarray) -> dict:
    """A boolean mask over every equal-length array in one extrema layer -- the same rule as
    ``devices.filters._mask_layer``, repeated here because core never imports devices."""
    n = len(keep)
    return {k: (v[keep] if hasattr(v, "shape") and getattr(v, "shape", ())[:1] == (n,) else v)
            for k, v in layer.items()}


def layer_point_keep(product, sel, si, n_points) -> np.ndarray:
    """Boolean keep over ONE extrema layer's own point arrays: the live selection scattered
    back through ``h_src``/``iso_src`` for scale ``si``. The materializer and the scene's
    fast path share this -- both must land the identical mask."""
    counts = np.diff(product["h_off"])
    chain_ok = np.zeros(len(product["h_len"]), dtype=bool)
    if sel["keep_h"] is None:
        chain_ok[:] = True
    else:
        chain_ok[sel["keep_h"]] = True
    vert_ok = np.repeat(chain_ok, counts)
    if sel["keep_h_pts"] is not None:
        vert_ok = vert_ok & sel["keep_h_pts"]
    iso_ok = sel["keep_iso_pts"]
    if iso_ok is None:
        iso_ok = np.ones(len(product["iso_src"]), dtype=bool)
    pt_scale = np.repeat(product["h_scale"], counts)
    keep = np.zeros(int(n_points), dtype=bool)
    keep[product["h_src"][vert_ok & (pt_scale == si)]] = True
    keep[product["iso_src"][iso_ok & (np.asarray(product["iso_scale"]) == si)]] = True
    return keep


def materialize_selection(result) -> dict:
    """Filtered ``extrema``/``chains`` dicts from the product + live selection, as a NEW result.

    The ONE honest gather per resolve: chains by list-index (the original dict objects, never
    copied), extrema layers by a single boolean scatter through ``h_src``/``iso_src``. Legacy
    consumers then see exactly the selected subset; the product and the (re-bound) selection
    stay live for the views' own index path. A result with nothing narrowed, or no live
    selection, comes back unchanged.
    """
    if not isinstance(result, dict):
        return result          # a field-tail landing (e.g. a noise-only chain) has no selection
    sel = selection_of(result)
    prod = result.get("chain_product")
    if sel is None or prod is None:
        return result
    h_narrowed = (sel["keep_h"] is not None or sel["keep_h_pts"] is not None
                  or sel["keep_iso_pts"] is not None)
    v_narrowed = sel["keep_v"] is not None
    if not h_narrowed and not v_narrowed:
        return result

    out = dict(result)
    if v_narrowed:
        chains = result.get("chains") or []
        out["chains"] = [chains[i] for i in sel["keep_v"]]
    if h_narrowed:
        layers = result.get("extrema") or []
        base_idx = result.get("_scale_idx")
        sis = [int(base_idx)] if (base_idx is not None and len(layers) == 1) \
            else list(range(len(layers)))
        new_layers = []
        for pos, si in zip(range(len(layers)), sis):
            layer = layers[pos]
            keep = layer_point_keep(prod, sel, si, len(np.asarray(layer["x"])))
            new_layers.append(_mask_layer_arrays(layer, keep))
        out["extrema"] = new_layers
    return narrow_selection(out, sel)      # no further narrowing -- re-bind the identities


def export_chain_product_npz(path, product, *, field, params) -> dict:
    """Write ``product`` as an EQSelect schema-v4 chain ``.npz`` (the design A6: the SAME bundle,
    in memory and on disk, so labels attach to the saved dataset).

    v4 is v3's exact CSR layout (``wtmm_backend.export_chains_npz``, the verbatim EQSelect copy)
    plus per-point ``h_arg``/``v_arg`` -- EQSelect's own 2026-08-01 extrema-layers spec, Phase B.
    Coordinates convert to frame units through ``field.x_axis``/``y_axis`` at THIS boundary only
    (z always 0.0, the v3 convention). The ``px_*`` keys are DynamiX-additive pixel columns for
    :func:`load_chain_product_npz`'s exact reconstruction; every EQSelect loader reads keys by
    name, so they are invisible to it -- ``tests/test_chain_product.py`` pins loadability through
    the verbatim ``load_chains_npz`` itself.
    """
    from dynamix.core.frames import frame_to_meta

    x_axis = np.asarray(field.x_axis, dtype=np.float64)
    y_axis = np.asarray(field.y_axis, dtype=np.float64)
    h_x = np.asarray(product["h_x"], dtype=np.int64)
    h_y = np.asarray(product["h_y"], dtype=np.int64)
    v_x = np.asarray(product["v_x"], dtype=np.int64)
    v_y = np.asarray(product["v_y"], dtype=np.int64)

    def _xyz(xs, ys):
        return np.column_stack(
            [x_axis[xs], y_axis[ys], np.zeros(xs.size)]).astype(np.float32)

    np.savez_compressed(
        path,
        h_xyz=_xyz(h_x, h_y), h_off=np.asarray(product["h_off"], np.int64),
        h_scale=np.asarray(product["h_scale"], np.int32),
        h_len=np.asarray(product["h_len"], np.int32),
        h_mod=np.asarray(product["h_mod"], np.float32),
        h_arg=np.asarray(product["h_arg"], np.float32),
        v_xyz=_xyz(v_x, v_y), v_off=np.asarray(product["v_off"], np.int64),
        v_persist=np.asarray(product["v_persist"], np.int32),
        v_scale=np.asarray(product["v_scale"], np.int32),
        v_mod=np.asarray(product["v_mod"], np.float32),
        v_arg=np.asarray(product["v_arg"], np.float32),
        schema_version=4,
        frame_kind=field.frame.kind,
        frame_meta=json.dumps(frame_to_meta(field.frame)),
        params=json.dumps(params),
        scales=np.asarray(product["scales"], np.float64),
        n_scales=int(product["n_scales"]),
        region=str(field.name),
        px_h_x=h_x.astype(np.int32), px_h_y=h_y.astype(np.int32),
        px_v_x=v_x.astype(np.int32), px_v_y=v_y.astype(np.int32),
        px_shape=np.asarray(product["shape"], np.int64),
        # The iso side (points on no run) -- additive keys with no EQSelect counterpart; its
        # loaders read keys by name and never see them.
        iso_xyz=_xyz(np.asarray(product["iso_x"], np.int64),
                     np.asarray(product["iso_y"], np.int64)),
        iso_mod=np.asarray(product["iso_mod"], np.float32),
        iso_arg=np.asarray(product["iso_arg"], np.float32),
        iso_scale=np.asarray(product["iso_scale"], np.int32),
        px_iso_x=np.asarray(product["iso_x"], np.int32),
        px_iso_y=np.asarray(product["iso_y"], np.int32),
    )
    return dict(path=path, n_horizontal=len(product["h_len"]),
                n_vertical=len(product["v_persist"]), n_scales=int(product["n_scales"]))


def load_chain_product_npz(path) -> dict:
    """Rebuild the in-memory product from a DynamiX v4 file (its ``px_*`` pixel columns).

    Metric columns are re-derived from the CSR through the SAME helpers the builder uses, so a
    loaded product filters identically to a freshly built one (to the file's float32 tolerance).
    A file without pixel columns is an EQSelect export, not ours -- refused by name so nobody
    silently draws frame-unit coordinates as pixels; those files load via the verbatim
    ``dynamix.core.wtmm_backend.load_chains_npz`` instead.
    """
    d = np.load(path, allow_pickle=True)
    if "px_h_x" not in d.files:
        raise ValueError(
            "not a DynamiX chain-product npz (no px_* pixel columns); EQSelect v2-v4 files "
            "load via dynamix.core.wtmm_backend.load_chains_npz")

    h_off = np.asarray(d["h_off"], np.int64)
    h_len = np.asarray(d["h_len"], np.int64)
    h_mod = np.asarray(d["h_mod"], np.float64)
    v_off = np.asarray(d["v_off"], np.int64)
    v_mod = np.asarray(d["v_mod"], np.float64)
    v_scale = np.asarray(d["v_scale"], np.int64)
    v_arg = np.asarray(d["v_arg"], np.float64)
    scales = np.asarray(d["scales"], np.float64)
    n_v = len(v_off) - 1
    v_counts = np.diff(v_off)
    chain_id = np.repeat(np.arange(n_v, dtype=np.int64), v_counts)

    with np.errstate(divide="ignore", invalid="ignore"):
        lm = np.log2(np.abs(v_mod))
    ls = np.log2(scales)[v_scale] if v_mod.size else np.zeros(0)

    h_mod_sup, h_mod_mean = _derive_h_metrics(h_mod, h_off, h_len)
    v_mod_finest, v_mod_sup, v_holder_ols, v_holder_max, v_max_log2_mod = \
        _derive_v_metrics(v_mod, v_off, chain_id, n_v, ls, lm)
    if v_arg.size and n_v:
        v_arg_finest = np.where(v_counts > 0,
                                v_arg[np.minimum(v_off[:-1], v_arg.size - 1)], np.nan)
    else:
        v_arg_finest = np.full(n_v, np.nan)

    # scale_peak re-derived from the file's own points -- the same lines-plus-singletons max
    # the builder records (every layer point is in exactly one of the two sides).
    h_scale_chain = np.asarray(d["h_scale"], np.int64)
    iso_mod = np.asarray(d["iso_mod"], np.float64)
    iso_scale = np.asarray(d["iso_scale"], np.int64)
    n_scales = int(d["n_scales"])
    peak = np.full(n_scales, -np.inf)
    if h_mod.size:
        h_pt_scale = np.repeat(h_scale_chain, np.diff(h_off))
        finite = np.isfinite(h_mod)
        np.maximum.at(peak, h_pt_scale[finite], h_mod[finite])
    if iso_mod.size:
        finite = np.isfinite(iso_mod)
        np.maximum.at(peak, iso_scale[finite], iso_mod[finite])
    peak[np.isneginf(peak)] = np.nan

    n_h_pts = int(h_off[-1]) if len(h_off) else 0
    return {
        "h_x": np.asarray(d["px_h_x"], np.int64), "h_y": np.asarray(d["px_h_y"], np.int64),
        "h_off": h_off, "h_mod": h_mod, "h_arg": np.asarray(d["h_arg"], np.float64),
        "h_src": np.full(n_h_pts, -1, dtype=np.int64),   # no backing layers on load
        "h_scale": h_scale_chain, "h_len": h_len,
        "h_mod_sup": h_mod_sup, "h_mod_mean": h_mod_mean,
        "iso_x": np.asarray(d["px_iso_x"], np.int64),
        "iso_y": np.asarray(d["px_iso_y"], np.int64),
        "iso_mod": iso_mod, "iso_arg": np.asarray(d["iso_arg"], np.float64),
        "iso_scale": iso_scale,
        "iso_src": np.full(iso_mod.size, -1, dtype=np.int64),
        "scale_peak": peak,
        "v_x": np.asarray(d["px_v_x"], np.int64), "v_y": np.asarray(d["px_v_y"], np.int64),
        "v_off": v_off, "v_mod": v_mod, "v_scale": v_scale, "v_arg": v_arg,
        "v_persist": np.asarray(d["v_persist"], np.int64),
        "v_mod_finest": v_mod_finest, "v_mod_sup": v_mod_sup,
        "v_holder_ols": v_holder_ols, "v_holder_max": v_holder_max,
        "v_max_log2_mod": v_max_log2_mod, "v_arg_finest": v_arg_finest,
        "scales": scales, "n_scales": int(d["n_scales"]),
        "shape": tuple(int(v) for v in np.asarray(d["px_shape"])),
    }
