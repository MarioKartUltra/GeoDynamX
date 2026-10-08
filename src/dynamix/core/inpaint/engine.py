# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""One inpainting run: the regions of a target band, its guides each on their own grid, the
route, the readouts and the stamps (the design's sections 4, 6, 11 and 13).

The unknown lives on the working image ``W`` (the target mirror-extended once in each axis
and rounded up to a multiple of ``2**J``); the target term is the mirrored data under the
mirrored weights, and every guide term reaches ``W`` through ``Quarter``. A value prior keeps
its own grid (a shift, a block average over its full blocks inside the target, or a warp); a
structure guide, a mask send and a veil seed are resampled onto the target's grid, the one
resampling the design allows, since it happens to a guide inside its measurement and never
to the target. Held-out blocks are gaps for the solver and data for the returned raster.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

from dynamix.core import dwt
from dynamix.core.inpaint import correct as co
from dynamix.core.inpaint import forward as fw
from dynamix.core.inpaint import priors as pr
from dynamix.core.inpaint import readouts as ro
from dynamix.core.inpaint import solver as so
from dynamix.core.inpaint.grid import GridRelation, georef, georef_from, warp_rows_cols

__all__ = ["Guide", "ROLES", "ROUTES", "empty_result", "regions_from", "run_inpaint"]

ROUTES = ("sparsity", "cascade", "both")
ROLES = ("value", "structure", "weights", "mask", "veil_seed")
DEFAULT_ITER = {"sparsity": 200, "cascade": 20}
PAPER_MISSING_LIMIT = 0.55
STRUCTURE_FLOOR = 0.3
GAIN_FLOOR = 0.05
#: Bytes the structure and veil terms may hold on the working image (a Detail term keeps
#: 3 J full-size bands twice): a full VNIR scene at three levels is 12 GB, an ROI is not.
TERM_BUDGET = 2 ** 31


@dataclasses.dataclass
class Guide:
    """A send resolved onto its own grid: ``field`` holds the values, ``weights`` (same grid)
    an optional weights raster, ``shade`` marks a DEM to be hillshaded, ``stamp`` identifies
    the content for the stage key, ``weights_for`` names what a weights send weights."""

    label: str
    role: str
    field: object
    relation: GridRelation
    weights: object = None
    shade: bool = False
    stamp: str = ""
    weights_for: str = "target"
    target_georef: dict = dataclasses.field(default_factory=dict)


# --- regions ------------------------------------------------------------------------------

def _label(mask, min_px: int = 1):
    from scipy.ndimage import label

    ids, n = label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3)))
    if min_px > 1 and n:
        counts = np.bincount(ids.ravel())
        small = np.nonzero(counts < min_px)[0]
        ids[np.isin(ids, small[small > 0])] = 0
    return ids


def _nearest_onto(values, src: dict, dst: dict) -> np.ndarray:
    rows, cols = warp_rows_cols(dst, src)
    r, c = np.rint(rows).astype(int), np.rint(cols).astype(int)
    H, W = np.shape(values)[:2]
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    out = np.full(rows.shape, np.nan)
    out[ok] = np.asarray(values, dtype=np.float64)[r[ok], c[ok]]
    return out


def _bilinear_onto(values, src: dict, dst: dict) -> np.ndarray:
    rows, cols = warp_rows_cols(dst, src)
    return fw.Warp(rows, cols, np.shape(values)[:2]).apply(values)


def regions_from(values, params: dict, *, window_offset=(0, 0), bands=None, stack=None,
                 guides=(), target_georef=None) -> tuple:
    """``(ids, kinds)``: region ids on the target's grid (0 outside) and ``{id: kind}``: the
    non-finite pixels first (always gaps), then what ``mask_source`` names in the order
    threshold, drawn polygons, sent masks (a later source overwrites where they overlap),
    relabelled 1..n."""
    y = np.asarray(values, dtype=np.float64)
    source = params.get("mask_source", "nodata")
    ids = np.zeros(y.shape, dtype=np.int32)
    kinds: dict = {}
    nxt = 1

    def add(mask, kind):
        nonlocal nxt
        comp = _label(mask)
        n = int(comp.max())
        if n:
            sel = comp > 0
            ids[sel] = comp[sel] + (nxt - 1)
            kinds.update({nxt + i: kind for i in range(n)})
            nxt += n

    add(~np.isfinite(y), "gap")
    if source in ("threshold", "all"):
        band = y
        band_id = str(params.get("mask_band", "") or "")
        if band_id:
            names = [str(b) for b in (bands or ())]
            if stack is None or band_id not in names:
                raise ValueError(f"inpaint: no band {band_id!r} for the threshold (mask_band is "
                                 f"one of {', '.join(names) or 'the target band only'})")
            band = np.asarray(stack, dtype=np.float64)[..., names.index(band_id)]
        value = float(params.get("mask_value", 0.0))
        hit = (band > value) if params.get("mask_op", ">") == ">" else (band < value)
        hit &= np.isfinite(band)
        add(_label(hit, int(params.get("mask_min_px", 16))) > 0, str(params.get("threshold_kind", "gap")))
    if source in ("region", "all"):
        from dynamix.core.inpaint._kernels import kernels

        try:
            polys = json.loads(params.get("_mask_polygons", "[]") or "[]")
        except ValueError as exc:
            raise ValueError(f"inpaint: the drawn regions are not readable ({exc})") from exc
        off_r, off_c = (float(v) for v in window_offset)
        for poly in polys:
            pts = np.asarray(poly.get("points", ()), dtype=np.float64).reshape(-1, 2)
            if len(pts) < 3:
                continue
            mask = np.zeros(y.shape, dtype=np.int32)
            kernels()["fill_polygon"](np.ascontiguousarray(pts - np.array([off_c, off_r])), mask, 1)
            add(mask > 0, str(poly.get("kind", "gap")))
    if source in ("send", "all"):
        if target_georef is None and any(g.role == "mask" for g in guides):
            raise ValueError("inpaint: a sent mask needs the target's georeference")
        for g in guides:
            if g.role != "mask":
                continue
            m = _nearest_onto(g.field.values, georef(g.field), target_georef)
            add(np.isfinite(m) & (m != 0), str(params.get("region_kind", "gap")))
    # relabel 1..n (ids are dense again after overwrites), one lookup over the raster
    present = np.unique(ids[ids > 0])
    lut = np.zeros(nxt, dtype=np.int32)
    lut[present] = np.arange(1, present.size + 1, dtype=np.int32)
    return lut[ids], {i + 1: kinds[int(k)] for i, k in enumerate(present)}


# --- helpers --------------------------------------------------------------------------------

def _key(target, values, ids, kinds, params: dict, guides, held, depths) -> str:
    """The stage key: the target's VALUES (the pre-fill and the abacus depend on them, and a
    stack's bands share one name), the regions, the held-out blocks, the bank and the depths
    the route picked (``depths = (J, Jmax, shape_w)``), the seed and the sends."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(ids).tobytes())
    h.update(np.ascontiguousarray(np.asarray(values, dtype=np.float64)).tobytes())
    h.update(np.ascontiguousarray(held).tobytes())
    payload = {"name": getattr(target, "name", ""), "shape": list(np.shape(values)),
               "kinds": {str(k): v for k, v in kinds.items()},
               "wavelet": params.get("wavelet"), "depths": [int(depths[0]), int(depths[1]),
                                                           [int(n) for n in depths[2]]],
               "structure_levels": params.get("structure_levels"),
               "veil_levels": params.get("veil_levels"), "seed": params.get("seed"),
               "heldout_blocks": params.get("heldout_blocks"),
               "heldout_side": params.get("heldout_side"), "abacus": params.get("abacus"),
               "abacus_file": params.get("abacus_file"),
               "guides": [[g.label, g.role, g.stamp, g.relation.stamp()] for g in guides]}
    h.update(json.dumps(payload, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _affine(ry, g) -> tuple:
    ok = np.isfinite(ry) & np.isfinite(g)
    if ok.sum() < 16:
        return 1.0, 0.0, int(ok.sum())
    a = np.cov(g[ok], ry[ok])[0, 1] / max(g[ok].var(), 1e-30)
    return float(a), float(ry[ok].mean() - a * g[ok].mean()), int(ok.sum())


def _full_blocks(o: int, f: int, n_guide: int, n_target: int):
    """Guide indices ``[k0, k1)`` whose ``f``-blocks (starting at target index ``o + k f``)
    lie inside ``[0, n_target)``."""
    return max(0, -(o // f)), min(n_guide, (n_target - o) // f)


def _aggregate_finer(values, rel: GridRelation, ny: int, nx: int) -> np.ndarray:
    f, (oy, ox) = rel.ratio, rel.offset
    H, W = np.shape(values)[:2]
    out = np.full((ny, nx), np.nan)
    i0, i1 = _full_blocks(oy, f, ny, H)
    j0, j1 = _full_blocks(ox, f, nx, W)
    if i1 > i0 and j1 > j0:
        block = np.asarray(values, dtype=np.float64)[oy + i0 * f:oy + i1 * f, ox + j0 * f:ox + j1 * f]
        out[i0:i1, j0:j1] = block.reshape(i1 - i0, f, j1 - j0, f).mean(axis=(1, 3))
    return out


def _onto_target(g: Guide, tg: dict, *, nearest=False, values=None) -> np.ndarray:
    """A guide's values on the target's grid: nested finer by exact aggregation, else by a
    nearest or bilinear warp of the guide."""
    vals = np.asarray(g.field.values if values is None else values, dtype=np.float64)
    if g.relation.kind == "nested" and g.relation.direction == "finer" and not nearest:
        return _aggregate_finer(vals, g.relation, *tg["shape"])
    src = georef(g.field)
    return _nearest_onto(vals, src, tg) if nearest else _bilinear_onto(vals, src, tg)


def _shaded(g: Guide, params: dict) -> np.ndarray:
    from dynamix.core.hillshade import hillshade

    f = g.field
    z_factor = 1.0 / 111320.0 if getattr(f.frame, "units", "") == "deg" else 1.0
    dx = abs(float(f.x_axis[1] - f.x_axis[0])) if f.x_axis.size > 1 else 1.0
    dy = abs(float(f.y_axis[1] - f.y_axis[0])) if f.y_axis.size > 1 else 1.0
    return hillshade(f.values, dx, dy, azimuth=float(params.get("sun_azimuth", 0.0)),
                     altitude=float(params.get("sun_elevation", 45.0)), z_factor=z_factor)


def _value_operator(g: Guide, ny: int, nx: int):
    """``(operator G -> the guide's grid, data, weights)`` of a value prior on its own grid."""
    rel = g.relation
    vals = np.asarray(g.field.values, dtype=np.float64)
    w = np.ones(vals.shape) if g.weights is None else np.asarray(g.weights.values, dtype=np.float64)
    if rel.kind == "same":
        op = fw.shift(rel.offset[0] + rel.subpixel[0], rel.offset[1] + rel.subpixel[1],
                      vals.shape, (ny, nx))
        return op, vals, w
    if rel.kind == "nested" and rel.direction == "coarser":
        f = rel.ratio
        k0, k1 = _full_blocks(rel.offset[0], f, vals.shape[0], ny)
        l0, l1 = _full_blocks(rel.offset[1], f, vals.shape[1], nx)
        if k1 <= k0 or l1 <= l0:
            raise ValueError(f"{g.label}: no overlap with the target")
        op = fw.Compose(fw.BlockAverage(f), fw.Window(rel.offset[0] + k0 * f, rel.offset[1] + l0 * f,
                                                     (k1 - k0) * f, (l1 - l0) * f, (ny, nx)))
        return op, vals[k0:k1, l0:l1], w[k0:k1, l0:l1]
    if rel.kind == "nested":                                 # finer: aggregated onto G first
        agg = _aggregate_finer(vals, rel, ny, nx)
        wa = _aggregate_finer(w, rel, ny, nx)
        return fw.Identity(), agg, np.where(np.isfinite(wa), wa, 0.0)
    rows, cols = warp_rows_cols(georef(g.field), g.target_georef)
    return fw.Warp(rows, cols, (ny, nx)), vals, w


def _scattered(gap) -> bool:
    """More than half of the gap's pixels border a clear pixel."""
    inner = gap & co.boundary_ring(gap)
    return bool(gap.any()) and inner.sum() > 0.5 * gap.sum()


def _warn_border(mask, L: int, J: int, ny: int, nx: int, warnings: list) -> None:
    reach = min((L - 1) * 2 ** J, min(ny, nx) // 4)
    r, c = np.nonzero(mask)
    if r.size and (r.min() < reach or c.min() < reach or r.max() >= ny - reach
                   or c.max() >= nx - reach):
        warnings.append("a region lies within the coarsest atom's support of the raster border: "
                        "values near the image border may be extreme (Pottier et al. 2008)")


def empty_result(values, params: dict, reason: str) -> dict:
    y = np.asarray(values, dtype=np.float64)
    return {"raster_out": y, "fill_mask": np.zeros(y.shape, dtype=np.int32), "chains": [],
            "extrema": [], "params": dict(params), "_route": params.get("route", "sparsity"),
            "_grid_relations": {}, "_seed": int(params.get("seed", 0)), "_iterations": 0,
            "_returned": 0, "_stop": reason, "_variance_error": 0.0,
            "_heldout": {"rms_rel": float("nan"), "var_err": float("nan"), "n_blocks": 0},
            "_eta_distance": {}, "_amp_distance": {}, "_terms_dropped": [], "_veil": None,
            "_regions": {"gap": 0, "veil": 0},
            "_fill_kind": "estimate", "_abacus": None, "_hists": {}, "_warnings": [],
            "_filled_fraction": 0.0, "_missing_fraction": float((~np.isfinite(y)).mean()),
            "_levels": 0, "_estimate": y.astype(np.float32),
            "_heldout_mask": np.zeros(y.shape, dtype=bool), "_affine": {}, "_shaded": [],
            "_history": {"cost": [], "score": []}}


# --- the run ----------------------------------------------------------------------------------

def run_inpaint(target, values, params: dict, guides, *, stage_cache=None, cancel=None,
                progress=None) -> dict:
    route = str(params.get("route", "sparsity"))
    if route not in ROUTES:
        raise ValueError(f"inpaint: unknown route {route!r}; one of {ROUTES}")
    y = np.asarray(values, dtype=np.float64)
    ny, nx = y.shape
    tg = georef(target)
    guides = list(guides)
    for g in guides:
        g.target_georef = tg
    name = str(params.get("wavelet", "sym7"))
    seed = int(params.get("seed", 0))
    warnings: list = []
    prov = getattr(target, "provenance", None) or {}
    win = prov.get("window") or {}
    offset = (int(win.get("row_off", 0) or 0), int(win.get("col_off", 0) or 0))
    tvals = getattr(target, "values", None)
    stack = tvals if tvals is not None and np.ndim(tvals) == 3 else None
    ids, kinds = regions_from(y, params, window_offset=offset, bands=prov.get("bands"),
                              stack=stack, guides=guides, target_georef=tg)
    gap = np.isin(ids, [k for k, v in kinds.items() if v == "gap"])
    veil = np.isin(ids, [k for k, v in kinds.items() if v == "veil"])
    finite = np.isfinite(y)
    if not (gap | veil).any():
        return empty_result(y, params, "no region")
    if not (finite & ~gap & ~veil).any():
        raise ValueError("inpaint: no clear pixel at all -- every pixel is declared or missing")
    if veil.any() and params.get("veil_model", "gain") == "gain" and np.nanmin(y) <= 0.0:
        raise ValueError("inpaint: a gain veil needs strictly positive data; choose the offset "
                         "model (veil_model = offset) for signed or log-domain data")
    for g in guides:
        if g.role in ("value", "structure") and g.relation.kind == "foreign" and route != "sparsity":
            raise ValueError(f"{g.label}: a foreign {g.role} send closes the cascade routes -- "
                             "use the sparsity route, or have the guide's producer resample it")
        if g.role in ("value", "structure") and g.relation.mapping_source == "none":
            raise ValueError(f"{g.label}: no mapping to the target (neither raster is "
                             "georeferenced and they come from different files)")

    # the target's weights, the held-out blocks
    w_t = np.ones(y.shape)
    for g in guides:
        if g.role == "weights" and g.weights_for == "target":
            wt = _onto_target(g, tg)
            w_t = np.where(np.isfinite(wt), wt, 0.0)
    w_t[~finite | gap | veil] = 0.0
    rng = np.random.default_rng(seed)
    n_blocks, side = int(params.get("heldout_blocks", 8)), int(params.get("heldout_side", 16))
    held = ro.heldout(w_t > 0, n_blocks, side, side, rng) if n_blocks > 0 else np.zeros(y.shape, bool)
    w_t[held] = 0.0
    gap_solver = gap | held
    region_g = gap_solver | veil
    missing = float(region_g.mean())
    if missing > PAPER_MISSING_LIMIT:
        text = (f"{100 * missing:.0f} % of the raster is missing or declared: Pottier et al. 2008 "
                f"(section 3.7) reach 55 % of random pixels")
        if _scattered(gap_solver) and route != "sparsity":
            raise ValueError(f"inpaint: {text}; the cascade route is refused on a scattered mask "
                             "(more than half of its pixels border clear ones) -- the sparsity "
                             "route runs with a warning")
        warnings.append(text)

    # depths and the working image
    L = dwt.filter_length(name)
    J = int(params.get("level", 0))
    if J <= 0:
        J = (max(1, min(dwt.max_level((2 * ny, 2 * nx), name), 8)) if route == "sparsity"
             else dwt.full_depth((2 * ny, 2 * nx)))
    Js, Jv = int(params.get("structure_levels", 3)), int(params.get("veil_levels", 3))
    has_structure = any(g.role == "structure" for g in guides)
    Jmax = max(J, Js if has_structure else 1, Jv if veil.any() else 1)
    shape_w = dwt.working_shape((ny, nx), Jmax)
    if route != "sparsity" and J < 3:
        raise ValueError(f"inpaint: the cascade route needs at least 3 pyramid levels; the "
                         f"working image is {shape_w[0]} x {shape_w[1]} at level {J}")
    _warn_border(gap | veil, L, J, ny, nx, warnings)

    def mirror(a):
        return dwt.mirror(np.asarray(a, dtype=np.float64), shape_w)

    quarter = fw.Window(0, 0, ny, nx, shape_w)

    # the value priors' affine fits on the clear overlap; the first one also pre-fills
    y_nan = np.where(w_t > 0, y, np.nan)
    affine: dict = {}
    value_ops: dict = {}
    prior_g = None
    for g in guides:
        if g.role != "value":
            continue
        op, data, w = _value_operator(g, ny, nx)
        probe = y_nan if isinstance(op, fw.Identity) else op.apply(y_nan)
        if not (np.isfinite(probe) & np.isfinite(data)).any():
            raise ValueError(f"{g.label}: no overlap with the target")
        a, b, n = _affine(probe, data)
        affine[g.label] = (a, b, n)
        value_ops[g.label] = (op, data, w)
        if n < 16:
            warnings.append(f"{g.label}: fewer than 16 clear overlap pixels; the prior "
                            "enters unscaled")
        if prior_g is None:
            raw = _onto_target(g, tg)
            prior_g = a * raw + b                            # the fitted prior fills the holes
            cov = float(np.isfinite(prior_g)[region_g].mean())
            if cov < 0.99:
                warnings.append(f"{g.label}: the value prior's coverage of the regions is "
                                f"{100 * cov:.0f} %; the clear mean fills the rest")

    # stages: the pre-fill, the masks on the pyramid, the abacus
    key = _key(target, y, ids, kinds, params, guides, held, (J, Jmax, shape_w))
    stages = stage_cache.get(key) if stage_cache is not None else None
    if stages is None:
        x0_g = co.prefill(np.where(gap_solver, np.nan, y), gap_solver, prior_g)
        rho_gap = pr.rho_pyramid(mirror(gap_solver) > 0.5, name, J)
        rho_all = pr.rho_pyramid(mirror(region_g) > 0.5, name, J)
        rho = [rho_gap[j - 1] if j <= Jv else rho_all[j - 1] for j in range(1, J + 1)]
        clear = pr.clear_masks(mirror(region_g) > 0.5, name, J)
        stages = {"x0": x0_g, "rho": rho, "clear": clear, "abacus": None}
        if stage_cache is not None:
            stage_cache[key] = stages
    x0_g, rho, clear = stages["x0"], stages["rho"], stages["clear"]
    y_w = mirror(np.where(finite, y, x0_g))
    x0_w = mirror(x0_g)
    known_w = mirror(w_t) > 0.0
    region_w = mirror(region_g) > 0.5
    ring_w = co.boundary_ring(region_w)
    held_w = mirror(held) > 0.5 if held.any() else None

    # terms
    terms = [fw.Term(fw.Identity(), y_w, mirror(w_t))]
    relations, dropped, shaded = {}, [], []
    n_structure = sum(1 for g in guides if g.role == "structure")
    bands_held = 2 * (3 * Js * n_structure + (3 * Jv if veil.any() else 0))   # data + weights per band
    if bands_held * 8 * int(np.prod(shape_w)) > TERM_BUDGET:
        raise ValueError(f"inpaint: the structure and veil terms would hold {bands_held} bands "
                         f"of {shape_w[0]} x {shape_w[1]} float64 "
                         f"({bands_held * 8 * int(np.prod(shape_w)) / 2 ** 30:.1f} GB, over the "
                         f"{TERM_BUDGET / 2 ** 30:.0f} GB budget) -- run on an ROI, or fewer levels")
    for g in guides:
        relations[g.label] = g.relation.stamp()
        if g.role == "value":
            op, data, w = value_ops[g.label]
            a, b, _n = affine[g.label]
            terms.append(fw.Term(fw.Compose(op, quarter), a * data + b,
                                 float(params.get("prior_weight", 0.3)) * np.where(np.isfinite(w), w, 0.0)))
        elif g.role == "structure":
            vals = _shaded(g, params) if g.shade else None
            if g.shade:
                shaded.append(g.label)
            on_g = _onto_target(g, tg, values=vals)
            if not np.isfinite(on_g).any():
                raise ValueError(f"{g.label}: no overlap with the target")
            D = fw.Detail(name, Js)
            d_y = D.apply(y_w)
            d_g = D.apply(mirror(np.where(np.isfinite(on_g), on_g, np.nanmean(on_g))))
            gains = np.zeros(d_g.shape[0])
            weights = np.zeros(d_g.shape)
            lam_s = float(params.get("structure_weight", 0.1))
            for band in range(d_g.shape[0]):
                j = D.level_of(band)
                clear_px = (pr.clear_pixels(region_g, name, j) == 1) & np.isfinite(on_g)
                sel = mirror(clear_px) > 0.5
                a_b, b_b = d_y[band][sel], d_g[band][sel]
                corr = float(np.corrcoef(a_b, b_b)[0, 1]) if sel.sum() > 16 and b_b.std() > 0 else 0.0
                if not np.isfinite(corr) or corr < STRUCTURE_FLOOR:
                    dropped.append([g.label, int(j), int(D.orientation_of(band))])
                    continue
                gains[band] = float(np.dot(a_b, b_b) / max(np.dot(b_b, b_b), 1e-30))
                weights[band] = lam_s * mirror(1 - pr.clear_pixels(region_g, name, j))
            terms.append(fw.Term(D, gains[:, None, None] * d_g, weights))

    # the veil: a gain or an offset on the fine levels, refitted between passes
    veil_info, veil_terms, s_w = None, [], None
    if veil.any():
        model = str(params.get("veil_model", "gain"))
        veil_w = mirror(veil) > 0.5
        Dv = fw.Detail(name, Jv)
        seeds = [g for g in guides if g.role == "veil_seed"]
        seed_src = "surround"
        near = (pr.clear_pixels(veil, name, 1) == 0) & finite & ~veil & ~co.boundary_ring(veil)
        if model == "gain":
            if seeds:
                sh = _onto_target(seeds[0], tg, values=_shaded(seeds[0], params))
                s_g = np.where(np.isfinite(sh), np.maximum(sh, GAIN_FLOOR), 1.0)
                seed_src = seeds[0].label
            else:
                ratio = float(y[veil & finite].mean() / max(y[near].mean(), 1e-30)) if near.any() else 1.0
                s_g = np.where(veil, min(max(ratio, GAIN_FLOOR), 1.0), 1.0)
            s_w = mirror(s_g)
            x0_w = np.where(veil_w, y_w / s_w, x0_w)            # the start: the data unveiled by the seed
        else:
            o = float(y[veil & finite].mean() - y[near].mean()) if near.any() else 0.0
            s_w = mirror(np.where(veil, o, 0.0))
            x0_w = np.where(veil_w, y_w - s_w, x0_w)
        veil_info = {"model": model, "seed": seed_src, "levels": Jv}

        def veil_terms_for(s):
            wv = np.broadcast_to(veil_w.astype(np.float64), (3 * Jv,) + tuple(shape_w)).copy()
            if model == "gain":
                return [fw.Term(fw.Compose(Dv, fw.Gain(s)), Dv.apply(y_w), wv)]
            return [fw.Term(Dv, Dv.apply(y_w - s), wv)]

        veil_terms = veil_terms_for(s_w)

    def problem_for(extra):
        all_terms = terms + extra
        Lip = fw.lipschitz(all_terms, shape_w, np.random.default_rng(seed + 7))
        return so.Problem(all_terms, y_w, known_w, ring_w, region_w, Lip, heldout=held_w)

    problem = problem_for(veil_terms)
    n_iter = int(params.get("n_iter", 0))
    controls = so.Controls(n_iter=n_iter or DEFAULT_ITER["sparsity" if route == "sparsity" else "cascade"],
                           tol=float(params.get("tol", 1e-3)),
                           stop_var_err=float(params.get("stop_var_err", 0.01)),
                           batch=float(params.get("batch", 1.0)),
                           correct=bool(params.get("correct", True)), seed=seed)

    # the abacus (cascade routes)
    abacus = None
    if route != "sparsity":
        if params.get("abacus", "self") == "file":
            abacus = pr.Abacus.load(str(params.get("abacus_file", "")), wavelet=name)
        else:
            abacus = stages.get("abacus")
            if abacus is None:
                coeffs = dwt.pyramid(y_w, name, J)

                def pairs_of(clear_masks):
                    try:
                        return pr.Abacus.from_pyramid(coeffs, clear_masks, name)
                    except ValueError:                      # no clear pair at all: count 0
                        return None

                abacus = pairs_of(clear)
                count = 0 if abacus is None else int(abacus.n_pairs.min())
                need = int(params.get("abacus_min_pairs", 1000))
                if count < need:
                    cause = ""
                    if held.any():
                        without = pairs_of(pr.clear_masks(mirror(gap | veil) > 0.5, name, J))
                        if without is not None and int(without.n_pairs.min()) >= need:
                            cause = " -- the held-out blocks caused it (set heldout_blocks to 0)"
                    raise ValueError(f"inpaint: the self abacus has {count} clear pairs in an "
                                     f"orientation, under abacus_min_pairs = {need}{cause}; a "
                                     "corpus abacus file is the way out")
                stages["abacus"] = abacus

    def run_route(x_start, problem):
        x, hist = x_start, None
        if route in ("sparsity", "both"):
            lam = pr.auto_lambda(y_w, clear[0], name, float(params.get("lam_rel", 0.2)))
            prior = pr.Sparsity(name, J, lam, np.random.default_rng(seed))
            c = controls if route == "sparsity" else dataclasses.replace(
                controls, n_iter=n_iter or DEFAULT_ITER["sparsity"])
            x, hist = so.fista(problem, prior, x, c, cancel=cancel, progress=progress)
        if route in ("cascade", "both"):
            prior = pr.Cascade(name, J, abacus, rho,
                               synthesis=bool(params.get("cascade_synthesis", False)),
                               rng=np.random.default_rng(seed + 3))
            x, hist = so.cascade_loop(problem, prior, x, controls, extra_terms=problem.terms[1:],
                                      cancel=cancel, progress=progress)
        return x, hist

    passes = int(params.get("veil_passes", 2)) if veil.any() else 1
    x_w, hist = x0_w, None
    for p in range(passes):
        x_w, hist = run_route(x_w, problem)
        if veil.any() and p + 1 < passes:
            from scipy.ndimage import gaussian_filter

            sig, vm = float(2 ** Jv), veil_w.astype(np.float64)
            if veil_info["model"] == "gain":
                num = gaussian_filter(x_w * y_w * vm, sig, mode="wrap")
                den = gaussian_filter(x_w * x_w * vm, sig, mode="wrap")
                s_w = np.where(veil_w, np.clip(num / np.maximum(den, 1e-30), GAIN_FLOOR, 1.0), 1.0)
            else:
                s_w = np.where(veil_w, gaussian_filter((y_w - x_w) * vm, sig, mode="wrap"), 0.0)
            problem = problem_for(veil_terms_for(s_w))

    # readouts on the estimate, then the final step
    x_g = np.asarray(dwt.quarter(x_w, ny, nx), dtype=np.float64)
    var_err = ro.errors(x_g, y, w_t > 0)["var_err"]
    held_err = (ro.errors(x_g, y, held) if held.any()
                else {"rms_rel": float("nan"), "var_err": float("nan"), "n": 0})
    hists = ro.band_hists(dwt.pyramid(x_w, name, J), rho, clear, abacus,
                          shift=dwt.child_shift(name))
    data_pixels = finite & ~gap & ~veil
    n_regions = {"gap": sum(1 for v in kinds.values() if v == "gap"),
                 "veil": sum(1 for v in kinds.values() if v == "veil")}
    return {
        "raster_out": np.where(data_pixels, y, x_g),
        "fill_mask": np.where(gap | veil, ids, 0).astype(np.int32), "chains": [], "extrema": [],
        "params": dict(params), "_route": route, "_grid_relations": relations, "_seed": seed,
        "_iterations": int(hist.iterations), "_returned": int(hist.returned), "_stop": hist.stop,
        "_variance_error": float(var_err), "_regions": n_regions,
        "_heldout": {"rms_rel": float(held_err["rms_rel"]), "var_err": float(held_err["var_err"]),
                     "n_blocks": int(held.sum() // (side * side)) if held.any() else 0},
        "_eta_distance": {f"{j}/{o}": v["eta"]["distance"] for (j, o), v in hists.items()
                          if np.isfinite(v["eta"]["distance"])},
        "_amp_distance": {f"{j}/{o}": v["amp"]["distance"] for (j, o), v in hists.items()
                          if np.isfinite(v["amp"]["distance"])},
        "_terms_dropped": dropped, "_veil": veil_info, "_fill_kind": "estimate",
        "_abacus": None if abacus is None else {
            "edges": abacus.edges, "hist": abacus.hist, "lo": abacus.lo, "hi": abacus.hi,
            "n_pairs": abacus.n_pairs, "wavelet": abacus.wavelet},
        "_hists": hists, "_warnings": warnings, "_filled_fraction": float((gap | veil).mean()),
        "_missing_fraction": missing, "_levels": int(J), "_estimate": x_g.astype(np.float32),
        "_heldout_mask": held, "_affine": affine, "_shaded": shaded,
        "_history": {"cost": list(hist.cost), "score": list(hist.score)},
    }
