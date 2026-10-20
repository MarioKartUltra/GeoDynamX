# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Wavelet leaders on the WTMM's continuous transform (spec 30): the device.

The leader at x and scale a is the supremum of the lifted CWT modulus over the scale-space
cylinder, every finer scale within the disc of radius R(a) = k·c·a (``core/cwt_leaders``). The
front end is the WTMM's own, the smoother, the ladder, η and the edge handling, so the ``cwt``
stage is shared with ``wtmm2d``; the measure is either every pixel (Jaffard's leaders) or the
WTMM's maxima only (his repair of the wavelet-maxima method). No fitting here: the sample sets
go to the fitting windows, the h-map is the per-pixel slope over the fit range.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from dynamix.core import cwt_leaders as core
from dynamix.devices.wtmm import BOUNDARY_PARAMS, SMOOTHER_PARAMS, WTMM2D, resolved_boundary, smoother_params
from dynamix.model.contracts import FIELD, HMAP, SAMPLES
from dynamix.model.device import Output, keyed_params
from dynamix.model.origin import FIRST_PARTY
from dynamix.model.param import Param, ParamKind

_WTMM = {p.name: p for p in WTMM2D.params}
_LADDER = ("n_oct", "n_voice", "a_min", "fracint_alpha")
_MAXIMA = ("thresh", "detector", "tie_rule", "interpolate")
_FRONT_KEYS = ("n_oct", "n_voice", "a_min", "wavelet", "fracint_alpha")


def _reused(name: str, **changes) -> Param:
    p = _WTMM[name]
    return dataclasses.replace(p, **changes) if changes else p


class CwtLeaders:
    provenance_precision = "float32"

    @staticmethod
    def provenance_outcome(result, params):
        from dynamix.engine.execution import resolved_outcome
        return resolved_outcome(result, params)

    name = "cwt_leaders"
    label = "CWT leaders"
    origin = dataclasses.replace(FIRST_PARTY, citation="Jaffard 1994 Lemma 1; Jaffard, Lashermes & Abry 2006; "
                                 "the cylinder supremum on the WTMM's transform (spec 30)")
    families = ("Regularity",)
    role = "primary"
    consumes = ((FIELD,),)
    produces = (HMAP, SAMPLES)
    capabilities = frozenset({"SelfReporting", "RegionAware", "Viewable", "OutputProducer", "StripExtras"})
    uses_stage_cache = True
    params = (
        *SMOOTHER_PARAMS,
        _reused("n_oct"), _reused("n_voice", default=1), _reused("a_min"), _reused("fracint_alpha"),
        *BOUNDARY_PARAMS,
        Param("operator", ParamKind.CHOICE, default="gradient", choices=("gradient", "none"),
              choice_labels=("gradient of θ (the WTMM's modulus)", "θ itself (g2 / q-g2; the mass kernel)"),
              label="Operator"),
        Param("measure", ParamKind.CHOICE, default="leaders", choices=("leaders", "wtmm"),
              choice_labels=("every pixel (leaders)", "the WTMM's maxima (Jaffard's repair)"), label="Measure"),
        *(dataclasses.replace(_WTMM[n], active_when=("measure", ("wtmm",))) for n in _MAXIMA),
        Param("p", ParamKind.FLOAT, default=2.0, min=0.0, max=8.0, soft_min=0.5, soft_max=4.0,
              label="p (p-leaders; 0 = off)"),
        Param("energy", ParamKind.FLOAT, default=0.99, min=0.9, max=0.999, label="Energy fraction",
              level="advanced"),
        Param("radius_k", ParamKind.FLOAT, default=1.0, min=0.5, max=2.0, label="Radius × R99", level="advanced"),
        Param("sup_from", ParamKind.INT, default=0, min=0, max=9, label="Finest octave in the sup",
              level="advanced"),
        Param("sample_stride", ParamKind.FLOAT, default=0.0, min=0.0, max=1.0,
              label="Sample lattice (× R; 0 = every pixel)", level="advanced"),
        Param("fit_min", ParamKind.FLOAT, default=0.0, min=0.0, max=4096.0, units="px",
              label="Fit from (scale; 0 = auto)", level="advanced"),
        Param("fit_max", ParamKind.FLOAT, default=0.0, min=0.0, max=4096.0, units="px",
              label="Fit to (scale; 0 = auto)", level="advanced"),
        Param("show", ParamKind.CHOICE, default="h_map", choices=("h_map", "leader", "p_leader"),
              choice_labels=("h-map", "log2 leader", "log2 p-leader"), label="Show", view=True),
    )
    outputs = (
        Output("leader", "raster", lazy=True, label="log2 leader", per_level=True),
        Output("p_leader", "raster", lazy=True, label="log2 p-leader", per_level=True),
    )
    param_groups = {
        "SMOOTHER": tuple(p.name for p in SMOOTHER_PARAMS),
        "LADDER": _LADDER,
        "EDGE": tuple(p.name for p in BOUNDARY_PARAMS),
        "LEADERS": ("operator", "measure", "p", "energy", "radius_k", "sup_from", "sample_stride"),
        "MAXIMA": _MAXIMA,
        "FIT": ("fit_min", "fit_max"),
    }

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key

        return cache_key(self.name, source_id, keyed_params(self, params))

    def check(self, params: dict) -> None:
        if int(params["sup_from"]) >= int(params["n_oct"]):
            raise ValueError(f"cwt_leaders: sup_from {params['sup_from']} octaves leaves nothing finer to take "
                             f"the supremum over ({params['n_oct']} octaves)")

    # ------------------------------------------------------------------------------ the radius

    @staticmethod
    def _c(params: dict) -> float:
        return core.energy_radius(params["wavelet"], params["operator"], smoother_params(params) or {},
                                  float(params["energy"]))

    def roi_margin(self, params: dict) -> int:
        """What an ROI must read beyond its window: the disc plus the kernel's reach at the
        coarsest scale, 2 R(a_max) with k = 1 (spec 30 §3.4)."""
        from dynamix.core.wtmm_backend import compute_scales2d

        a_max = float(compute_scales2d(params["n_oct"], params["n_voice"], params["a_min"])[-1])
        return int(math.ceil((1.0 + float(params["radius_k"])) * self._c(params) * a_max))

    # ------------------------------------------------------------------------------ compute

    @staticmethod
    def _front(field, params: dict) -> tuple:
        """``(backend params, padding record, edge used)``: the WTMM's own resolve of the
        smoother, ladder, lift and edge, so the stage key matches `wtmm2d`'s."""
        edge, border = resolved_boundary(params, field)
        if edge not in ("reflect", "periodic", "edge_pad", "zero_pad"):
            raise ValueError(f"cwt_leaders: Edge {edge!r} runs the WTMM scale by scale through the halo; "
                             "this tool takes reflect, periodic, edge-value or zero padding")
        params = dict(params, edge=edge, border_cut=border)
        backend = {"n_oct": params["n_oct"], "n_voice": params["n_voice"], "a_min": params["a_min"],
                   "wavelet": params["wavelet"], "smoother_params": smoother_params(params),
                   "fracint_alpha": params["fracint_alpha"]}
        backend, padding = WTMM2D._boundary(field, params, backend)
        return backend, padding, edge

    def compute(self, field, params: dict, *, progress=None, cancel=None, stage_cache=None) -> dict:
        from dynamix.core.wtmm_backend import run_cwt_stage, run_wtmm2d

        values = np.asarray(field.values)
        if values.ndim != 2:
            raise ValueError(f"cwt_leaders: a single-band field is required, got shape {values.shape}")
        self.check(params)
        backend, padding, edge = self._front(field, params)
        stage = run_cwt_stage(field, backend, operator=params["operator"], progress=progress,
                              cancel=cancel, stage_cache=stage_cache)
        mod, scales = np.asarray(stage["mod"], dtype=np.float32), np.asarray(stage["scales"], dtype=np.float64)
        c = self._c(params)
        radii = float(params["radius_k"]) * c * scales
        warnings = []
        if radii[0] < 3.0:
            warnings.append(f"R(a_min) = {radii[0]:.1f} px: discs under 3 px are lumpy; those scales are outside "
                            "the default fit range")

        sup_from = int(params["sup_from"]) * int(params["n_voice"])
        measured = mod
        maxima_hits: set = set()
        if params["measure"] == "wtmm":
            res = run_wtmm2d(field, dict(backend, **{k: params[k] for k in _MAXIMA}), progress=progress,
                             cancel=cancel, stage_cache=stage_cache)
            maxima_hits = set(res.get("cache_hits", ()))
            measured = np.full(mod.shape, np.nan, dtype=np.float32)
            for i, ext in enumerate(res["extrema"][:len(scales)]):
                y, x = np.asarray(ext["y"], dtype=np.int64), np.asarray(ext["x"], dtype=np.int64)
                measured[i, y, x] = mod[i, y, x]
        leaders = core.cylinder_leaders(measured, radii, sup_from=sup_from)

        p = float(params["p"])
        p_leaders = None
        if p > 0:
            p_leaders = np.full(mod.shape, np.nan, dtype=np.float32)
            acc = None
            for j in range(sup_from, len(scales)):
                term = np.where(np.isfinite(measured[j]), measured[j].astype(np.float64) ** p, np.nan)
                acc = term if acc is None else np.where(np.isfinite(term), np.nan_to_num(acc) + term, acc)
                total, count = core.disc_boxsum(acc, float(radii[j]))
                with np.errstate(divide="ignore", invalid="ignore"):
                    mean = total / (count * (j - sup_from + 1))
                p_leaders[j] = np.where(count > 0, mean ** (1.0 / p), np.nan).astype(np.float32)

        # The valid mask is the WTMM's own edge group: the disc sup over the frame is what the
        # kernel computes, a mirrored coefficient is attenuated (the gradient's normal component
        # suppressed, the tangential one boosted) and a sup forgives it, so reflect padding with
        # no Border cut keeps every pixel; the Border cut's band at each scale is the same one
        # that drops the WTMM's maxima (spec 30 §3.4).
        ny, nx = values.shape
        borders = [0.0] * len(scales)
        if backend["border_cut"] != "none":
            from dynamix.core.edges import border_widths
            borders = [float(b) for b in border_widths(
                list(scales), backend["border_cut"], wavelet=backend["wavelet"],
                smoother_params=backend["smoother_params"], eps=backend["edge_eps"],
                pct=backend["border_pct"], k=backend["border_k"], scope=backend["border_scope"],
                run_grid=max(ny, nx) + 2 * int(backend["pad"]))]
        y, x = np.ogrid[:ny, :nx]
        inside = np.minimum(np.minimum(y, ny - 1 - y), np.minimum(x, nx - 1 - x))
        valid = []
        for j in range(len(scales)):
            m = np.isfinite(leaders[j])
            if borders[j] > 0:
                m &= inside >= borders[j]
            if float(params["sample_stride"]) > 0:
                stride = max(1, int(round(float(params["sample_stride"]) * radii[j])))
                lattice = np.zeros_like(m)
                lattice[::stride, ::stride] = True
                m &= lattice
            valid.append(m)
        # The cell count is the independent discs over the valid area, A_valid / πR²: on a full
        # field the whole domain's discs, scaling as a^-2 as the dyadic device's cube count does;
        # on the maxima measure the occupied discs, Jaffard's sum over occupied cells.
        cells = n_eff = [float(v.sum()) / max(math.pi * r * r, 1.0) for v, r in zip(valid, radii)]

        lo, hi = float(params["fit_min"]), float(params["fit_max"])
        # The default range: discs of at least 3 px and at least 16 independent discs (the dyadic
        # device's floor); an explicit range is taken as given.
        fit = [j for j in range(sup_from, len(scales))
               if (radii[j] >= 3.0 and n_eff[j] >= 16.0 if lo <= 0 else scales[j] >= lo)
               and (hi <= 0 or scales[j] <= hi)]
        if lo <= 0 and any(n_eff[j] < 16.0 for j in range(sup_from, len(scales))):
            warnings.append("scales with fewer than 16 independent discs are outside the default fit range")
        if len(fit) < 2:
            warnings.append("fewer than two scales in the fit range: no h-map")
            h_map = np.full(values.shape, np.nan, dtype=np.float32)
            r2_map = h_map.copy()
        else:
            h_map, r2_map = _slopes(leaders[fit], scales[fit], np.logical_and.reduce([valid[j] for j in fit]))
        fit_range = (float(scales[fit[0]]), float(scales[fit[-1]])) if fit else (float("nan"), float("nan"))

        sets = {"cwt-leaders": {"values": [leaders[j] for j in range(len(scales))], "valid": valid}}
        if p_leaders is not None:
            sets["cwt-p-leaders"] = {"values": [p_leaders[j] for j in range(len(scales))], "valid": valid}
        record = {
            "method": self.name, "operator": params["operator"], "measure": params["measure"],
            "wavelet": params["wavelet"], "smoother_params": smoother_params(params),
            "ladder": {k: params[k] for k in ("n_oct", "n_voice", "a_min")}, "scales": scales.tolist(),
            "energy_radius_per_scale": c, "radius_k": float(params["radius_k"]), "radii_px": radii.tolist(),
            "energy": float(params["energy"]), "sup_from_octaves": int(params["sup_from"]),
            "eta": float(params["fracint_alpha"]), "p": p,
            "p_leader": "(mean over the cylinder of M^p)^(1/p), the mean over the admitted scales "
                        "of the disc mean" if p > 0 else None,
            "n_eff": n_eff, "cells": cells, "edge": {"used": edge, "padding": padding, "borders_px": borders},
            "fit_range": fit_range, "sample_stride": float(params["sample_stride"]),
            "cwt_cache_hits": sorted(stage["cache_hits"]), "maxima_cache_hits": sorted(maxima_hits),
            "warnings": warnings,
        }
        return {
            "h_map": h_map, "r2_map": r2_map, "chains": [], "extrema": [], "scales": scales,
            "params": dict(params),
            "samples": {"scales": scales, "cells": cells, "d": 2, "fit_range": fit_range, "sets": sets},
            "_leaders": leaders, "_p_leaders": p_leaders, "_valid": valid, "_radii": radii,
            "_frame": getattr(field, "frame", None), "_shape": tuple(values.shape),
            "record": record, "warnings": warnings,
        }

    def compute_roi(self, window, core_rect, params: dict, *, info=None, progress=None, cancel=None,
                    stage_cache=None) -> dict:
        """The region's own result: the whole window computed, then every per-scale array and map
        cut to the core and the independent-disc counts recounted there."""
        m, _m, h, w = (int(v) for v in core_rect)
        res = self.compute(window, params, progress=progress, cancel=cancel, stage_cache=stage_cache)
        cut = (slice(m, m + h), slice(m, m + w))
        res["_leaders"] = res["_leaders"][:, cut[0], cut[1]]
        if res["_p_leaders"] is not None:
            res["_p_leaders"] = res["_p_leaders"][:, cut[0], cut[1]]
        res["_valid"] = [v[cut] for v in res["_valid"]]
        for key in ("h_map", "r2_map"):
            res[key] = res[key][cut]
        cells = n_eff = [float(v.sum()) / max(math.pi * r * r, 1.0) for v, r in zip(res["_valid"], res["_radii"])]
        sets = {"cwt-leaders": {"values": [res["_leaders"][j] for j in range(len(res["scales"]))],
                                "valid": res["_valid"]}}
        if res["_p_leaders"] is not None:
            sets["cwt-p-leaders"] = {"values": [res["_p_leaders"][j] for j in range(len(res["scales"]))],
                                     "valid": res["_valid"]}
        res["samples"] = dict(res["samples"], cells=cells, sets=sets)
        res["record"] = dict(res["record"], n_eff=n_eff, cells=cells)
        res["_shape"] = (h, w)
        return res

    # ------------------------------------------------------------------------------ products

    def compute_output(self, name: str, values, result: dict, params: dict, *, fetch,
                       progress=None, cancel=None, level=None) -> dict:
        """log2 of the leader or p-leader at the level shown; NaN outside the valid region."""
        stack = {"leader": result["_leaders"], "p_leader": result["_p_leaders"]}.get(name)
        if stack is None:
            raise ValueError(f"{self.name}: no lazy output named {name!r}"
                             + (" (p = 0)" if name == "p_leader" else ""))
        idx = int(np.clip(0 if level is None else int(level), 0, len(result["scales"]) - 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            raster = np.where(result["_valid"][idx], np.log2(stack[idx]), np.nan).astype(np.float32)
        return {"raster": raster, "diag": {"level": idx, "scale_px": float(result["scales"][idx]),
                                           "radius_px": float(result["_radii"][idx]),
                                           "n_eff": float(result["record"]["n_eff"][idx])}}

    def reading(self, result: dict, params: dict) -> str:
        rec = result["record"]
        scales, radii, cells = rec["scales"], rec["radii_px"], rec["n_eff"]
        lo, hi = rec["fit_range"]
        parts = [f"{rec['operator']} of {rec['wavelet']}" + (" · WTMM maxima" if rec["measure"] == "wtmm" else ""),
                 f"{len(scales)} scales {scales[0]:.3g}–{scales[-1]:.3g} px",
                 f"R {radii[0]:.1f}–{radii[-1]:.1f} px", f"fit {lo:.3g}–{hi:.3g} px",
                 f"N_eff {min(cells):.0f}–{max(cells):.0f}"]
        if rec["eta"]:
            parts.append(f"η {rec['eta']:g}")
        if rec["p"] > 0:
            parts.append(f"p = {rec['p']:g}")
        return " · ".join(parts + rec["warnings"])

    def view(self, result: dict, params: dict) -> dict:
        return result


def _slopes(stack, scales, valid) -> tuple:
    """Per-pixel least-squares slope of log2 L against log2 a over the stack's scales, and r²;
    NaN where any leader is invalid."""
    x = np.log2(np.asarray(scales, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log2(np.asarray(stack, dtype=np.float64))
    xm = x - x.mean()
    ym = y - y.mean(axis=0)
    sxx = float((xm * xm).sum())
    sxy = np.tensordot(xm, ym, axes=(0, 0))
    syy = (ym * ym).sum(axis=0)
    slope = sxy / sxx
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(syy > 0, (sxy * sxy) / (sxx * syy), np.nan)
    ok = valid & np.isfinite(y).all(axis=0)
    return (np.where(ok, slope, np.nan).astype(np.float32), np.where(ok, r2, np.nan).astype(np.float32))
