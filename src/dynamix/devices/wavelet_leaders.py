# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Wavelet leaders: the regularity of a band read three ways on the dyadic pyramid.

The leaders, p-leaders and (θ, ω)-leaders of :mod:`dynamix.core.leaders` on the critically
sampled pyramid of :mod:`dynamix.core.dwt` (each level half the resolution of the last), in one
pass: a local exponent per cell painted on the native grid as the h-map, the per-level quantities
on the Scale slider, and the quantities over their valid cells as per-level sample sets (the
``Samples`` contract) for the tools that fit them with their own moment orders and scale ranges.
Nothing is fitted here. Each method's admissibility on the data at hand is stamped (h_min > 0 for
the leaders, η(p) > 0 for the p-leaders, nothing for the (θ, ω)-leaders) with the
pseudo-fractional integration γ that would clear it. ``method`` chooses which of the three the
h-map shows; switching it recomputes nothing.
"""
from __future__ import annotations

import math

import numpy as np

from dynamix.core import dwt, leaders
from dynamix.devices.holder_map import field_values_2d
from dynamix.devices.wtmm import filled_block, filled_origin
from dynamix.model.contracts import FIELD, HMAP, SAMPLES
from dynamix.model.device import Output, keyed_params
from dynamix.model.param import Param, ParamKind

__all__ = ["WaveletLeaders"]

#: Lazy output name -> the quantity it paints.
_OUTPUT_OF = {"leader": "leaders", "p_leader": "p-leaders", "wse": "wse", "oscillation": "oscillation"}
_METHOD_LABEL = {"leaders": "leaders", "p-leaders": "p-leaders", "wse": "WSE leaders"}


class WaveletLeaders:
    """Leaders, p-leaders and (θ, ω)-leaders of a band, with the h-map and the sample sets."""

    provenance_precision = "float64"

    @staticmethod
    def provenance_outcome(result, params):
        from dynamix.engine.execution import resolved_outcome
        return resolved_outcome(result, params)

    provenance_libraries = (('pywt', 'PyWavelets'),)

    name = "wavelet_leaders"
    label = "Wavelet leaders"
    families = ("Regularity",)
    role = "primary"                    # a drop on a layer forks a sibling over the same field
    consumes = ((FIELD,),)
    produces = (HMAP, SAMPLES)
    capabilities = frozenset({"SelfReporting", "RegionAware", "Viewable", "OutputProducer",
                              "StripExtras"})
    params = (
        Param("wavelet", ParamKind.CHOICE, default="sym4", choices=dwt.BANKS, label="Wavelet"),
        # How the field is continued to the pyramid's working image: auto reads a Periodic
        # extension's stamp and runs periodic, with no border lost; else the mirror.
        Param("edge", ParamKind.CHOICE, default="auto",
              choices=("auto", "mirror", "reflect", "periodic", "edge_pad", "zero_pad"),
              choice_labels=("auto", "mirror", "reflect", "periodic", "edge-value padding", "zero padding"),
              label="Edge"),
        Param("scale_min", ParamKind.INT, default=4, min=2, max=512, soft_min=2, soft_max=16,
              units="px", label="Finest fitted cell"),
        Param("scale_max", ParamKind.INT, default=64, min=4, max=1024, soft_min=16, soft_max=128,
              units="px", label="Coarsest fitted cell"),
        Param("gamma", ParamKind.FLOAT, default=0.0, min=-2.0, max=4.0, soft_min=0.0, soft_max=2.0,
              label="γ"),
        Param("dither", ParamKind.BOOL, default=False, label="Dither"),
        Param("spin", ParamKind.INT, default=1, min=1, max=4, soft_max=4, label="Cycle spin (grids per axis)"),
        Param("p", ParamKind.FLOAT, default=2.0, min=0.25, max=16.0, soft_min=0.5, soft_max=4.0,
              label="p"),
        Param("wse_rule", ParamKind.CHOICE, default="dumeur", choices=("dumeur", "fixed"),
              choice_labels=("resolution-indexed (Dumeur)", "fixed depth and width"), label="Window"),
        Param("wse_beta", ParamKind.FLOAT, default=0.25, min=0.05, max=0.95, soft_min=0.1, soft_max=0.5,
              label="θ exponent β", active_when=("wse_rule", ("dumeur",))),
        Param("wse_depth", ParamKind.INT, default=1, min=0, max=4, label="Depth (octaves)",
              active_when=("wse_rule", ("fixed",))),
        Param("wse_width", ParamKind.FLOAT, default=1.0, min=0.0, max=8.0, soft_min=0.0, soft_max=3.0,
              label="Half-width (cells)", active_when=("wse_rule", ("fixed",))),
        # View-only: what the h-map shows, and the raster drawn.
        Param("method", ParamKind.CHOICE, default="leaders", choices=tuple(leaders.METHODS),
              choice_labels=tuple(_METHOD_LABEL[m] for m in leaders.METHODS), label="Method", view=True),
        Param("show", ParamKind.CHOICE, default="h_map",
              choices=("h_map", "leader", "p_leader", "wse", "oscillation"),
              choice_labels=("h-map", "log2 leader", "log2 p-leader", "log2 WSE leader",
                             "oscillation ratio"),
              label="Show", view=True),
    )
    outputs = (
        Output("leader", "raster", lazy=True, label="log2 leader", per_level=True),
        Output("p_leader", "raster", lazy=True, label="log2 p-leader", per_level=True),
        Output("wse", "raster", lazy=True, label="log2 WSE leader", per_level=True),
        Output("oscillation", "raster", lazy=True, label="oscillation ratio", per_level=True),
    )
    param_groups = {
        "PYRAMID": ("wavelet", "edge", "scale_min", "scale_max", "gamma", "dither"),
        "P-LEADERS": ("p",),
        "WSE": ("wse_rule", "wse_beta", "wse_depth", "wse_width"),
    }

    # ------------------------------------------------------------------------------ the knobs

    @staticmethod
    def _levels(params: dict) -> tuple:
        """``(j_min, J)``: the finest and coarsest fitted levels, both powers of two."""
        out = []
        for key in ("scale_min", "scale_max"):
            v = int(params[key])
            if v < 2 or v & (v - 1):
                raise ValueError(f"wavelet_leaders: {key} {v} px is not a power of two")
            out.append(int(round(math.log2(v))))
        return tuple(out)

    @staticmethod
    def _edge(field, params: dict) -> str:
        """The continuation used: ``auto`` is periodic on a Periodic extension's output (its
        ``edges.filled`` stamp), the mirror otherwise."""
        asked = params.get("edge", "auto")
        if asked != "auto":
            return asked
        return "periodic" if filled_block(field) is not None else "mirror"

    def check(self, params: dict) -> None:
        j_min, J = self._levels(params)
        if J - j_min < 2:
            raise ValueError(f"wavelet_leaders: at least three fitted levels: scale_min {params['scale_min']} "
                             f"px to scale_max {params['scale_max']} px spans {J - j_min + 1}")

    def roi_margin(self, params: dict) -> int:
        """What an ROI must read beyond its window: the bank's border in cells of the coarsest
        level, plus one cell, the partial cell the window's grid may start with when it keeps the
        field's cells."""
        return (dwt.border_cells(params["wavelet"]) + 1) * int(params["scale_max"])

    @staticmethod
    def grid_anchor(field, scale_max: int):
        """Where the coarsest cell's boundary falls on this field's pixels, so a region keeps the
        whole field's cells: the field's centred phase shifted by the region's origin in it, from
        the provenance a processing window carries; None for a field that is the whole file."""
        prov = getattr(field, "provenance", None) or {}
        win, full = prov.get("window"), prov.get("full_dims")
        if not win or not full:
            return None
        cell = int(scale_max)
        return tuple(((int(n) % cell) // 2 - int(off)) % cell
                     for n, off in zip(full, (win.get("row_off", 0), win.get("col_off", 0))))

    def derived_reading(self, name: str, value, field, params: dict | None = None) -> str | None:
        """The fitted cells in the field's units beside ``scale_min`` and ``scale_max``; what
        ``auto`` resolves to beside ``edge``."""
        if name == "wse_beta":
            return f"θ(j) = j + j^{float(value):g}; ω(j′) = j′ · resolution index j"
        if name == "edge":
            if field is None:
                return None
            used = self._edge(field, dict(params or {}, edge=value))
            head = (f"→ {used}" + (" (Periodic extension output)" if used == "periodic" else "")
                    if value == "auto" else "")
            if used == "periodic":
                # The working image is the field wrapped to a multiple of 2^J: a size that is not
                # one leaves a seam at the wrap, a border again.
                J = self._levels(params or {})[1] if params else None
                shape = tuple(int(n) for n in np.shape(getattr(field, "values", ()))[:2])
                if J is not None and shape and any(n % 2 ** J for n in shape):
                    seam = f"{shape[0]} × {shape[1]} px is no multiple of {2 ** J}: the wrap is a seam, bordered"
                    return f"{head} · {seam}" if head else seam
            return head or None
        if name not in ("scale_min", "scale_max"):
            return None
        from dynamix.core.units import px_to_metres

        px, unit = px_to_metres(field)
        if px is None or unit == "px":
            return None
        return f"{int(value)} px ≈ {int(value) * px:.3g} {unit}"

    def wavelet_profile(self, params: dict, scale_px: float, level: int):
        """The bank's wavelet at the level whose cell is ``scale_px`` (the wavelet scale bar)."""
        from dynamix.core.wavelet_profiles import dwt_profile

        return dwt_profile(params["wavelet"], int(round(math.log2(float(scale_px)))))

    # ------------------------------------------------------------------------------- compute

    def _fits(self, shape, params: dict, edge: str) -> tuple:
        """Whether every fitted level keeps at least 16 cells inside the border (4 × 4), the
        border (none on a torus), and the largest ``scale_max`` whose three coarsest levels all
        do (0 when none)."""
        j_min, J = self._levels(params)
        e = 0 if edge == "periodic" else dwt.border_cells(params["wavelet"])
        side = min(int(n) for n in shape)
        fits = all(side // 2 ** j - 2 * e >= 4 for j in range(j_min, J + 1))
        largest = max((2 ** j for j in range(3, 12) if side // 2 ** j - 2 * e >= 4), default=0)
        return fits, largest, e

    def knob_warnings(self, field, params: dict) -> dict:
        """The strip's ⚠ on the coarsest cell when a fitted level would keep fewer than 16 cells
        inside the bank's border on this field; nothing before a field is loaded."""
        values = getattr(field, "values", None)
        if values is None or np.ndim(values) < 2 or not params:
            return {}
        try:
            self.check(params)
        except ValueError:
            return {}
        fits, largest, e = self._fits(filled_block(field) or np.shape(values)[:2], params, self._edge(field, params))
        if fits:
            return {}
        where = (f"levels above {largest} px keep fewer than 16 cells inside {params['wavelet']}'s border of "
                 f"{e} cells on this field; their moments are thin or empty" if largest else
                 f"no level keeps 16 cells inside {params['wavelet']}'s border of {e} cells on this field")
        return {"scale_max": where}

    def compute(self, field, params: dict, *, progress=None) -> dict:
        vals = field_values_2d(field, self.name)
        self.check(params)
        edge = self._edge(field, params)
        block, origin = filled_block(field), filled_origin(field)
        _fits, _largest, e = self._fits(block or vals.shape, params, edge)
        if params["dither"]:
            from dynamix.core.mz_edges import _dithered, measure_lsb

            vals = _dithered(vals, measure_lsb(vals))
        tick = (lambda frac: progress("wavelet leaders", frac)) if progress is not None else None
        kw = dict(wavelet=params["wavelet"], scale_min=int(params["scale_min"]),
                  scale_max=int(params["scale_max"]), gamma=float(params["gamma"]), p=float(params["p"]),
                  wse_rule=params["wse_rule"], wse_depth=int(params["wse_depth"]),
                  wse_width=float(params["wse_width"]), wse_beta=float(params["wse_beta"]),
                  edge=edge, origin=origin, block=block)
        run = leaders.run_leaders(vals, **kw, anchor=self.grid_anchor(field, int(params["scale_max"])), progress=tick)
        j_min, J = run["j_min"], run["levels"]
        warnings = []
        truncated = [j for j, flag in enumerate(run["wse_truncated"], 1) if flag and j >= j_min]
        if truncated:
            warnings.append(f"WSE finer-scale neighborhoods truncated at levels {truncated}; compare a coarser fit range")
        adm = run["admissibility"]
        for m in leaders.METHODS:
            if not adm[m]["admissible"]:
                warnings.append(f"{_METHOD_LABEL[m]} inadmissible on this field (h_min "
                                f"{adm['h_min']['integrated']:.2f}, η({params['p']:g}) "
                                f"{adm['eta']['integrated']:.2f}); γ ≥ {adm[m]['suggested_gamma']:g} would clear it")
        rows, cols = block or vals.shape
        # The pyramid reaches every level; a fitted level that keeps fewer than 16 cells inside
        # the border is thin and one with none carries no statistics: warnings, never a refusal.
        for j in range(j_min, J + 1):
            n_j = int(run["valid_cells"]["leaders"][j - 1])
            if n_j == 0:
                warnings.append(f"level {j} ({2 ** j} px): no cells inside {params['wavelet']}'s border of {e} "
                                "cells; the level carries no statistics")
            elif n_j < 16:
                warnings.append(f"level {j} ({2 ** j} px): {n_j} cells inside the border, fewer than 16; "
                                "its moments are thin")
        for j in range(j_min, J + 1):
            inside = (rows // 2 ** j - 2 * e) * (cols // 2 ** j - 2 * e)
            if inside > 0 and run["valid_cells"]["leaders"][j - 1] < inside / 2:
                warnings.append(f"level {j} ({2 ** j} px): more than half its cells touch non-finite pixels")
            if run["zero_cells"][j - 1] > 0.01 * max(run["valid_cells"]["leaders"][j - 1], 1):
                warnings.append(f"level {j} ({2 ** j} px): {run['zero_cells'][j - 1]} leaders are exactly zero "
                                "(quantised data); the dither knob perturbs by half a least significant bit")
                break
        h_cells = {m: (h.astype(np.float32), r2.astype(np.float32), n.astype(np.float32)) for m, (h, r2, n) in run["local"].items()}
        quantity = {m: [np.asarray(a, dtype=np.float32) for a in run["quantity"][m]] for m in leaders.METHODS}
        quantity["oscillation"] = [np.asarray(a, dtype=np.float32) for a in run["oscillation"]]
        mods = [np.asarray(a, dtype=np.float32) for a in run["coefficients"]]
        valid = {m: list(run["valid"][m]) for m in leaders.METHODS + ("coefficients",)}
        valid["oscillation"] = [a & b for a, b in zip(valid["leaders"], valid["wse"])]
        method = params["method"]
        h_map, r2_map, levels_map = self._paint(h_cells[method], j_min, vals.shape, run["pad"])
        used = levels_map[np.isfinite(levels_map)]
        record = {
            "method": self.name, "wavelet": params["wavelet"],
            "levels": {"j_min": j_min, "J": J,
                       "used": {"min": int(used.min()), "max": int(used.max())} if used.size else None},
            "grid": {"cell": 2 ** J, "phase": [int(v) for v in run["phase"]], "pad": [int(v) for v in run["pad"]]},
            "edge": {"asked": params["edge"], "used": edge, "border_cells": e,
                     "block": list(block) if block else None, "origin": list(origin)},
            "seam": bool(run["seam"]),
            "gamma": float(params["gamma"]), "p": float(params["p"]), "dither": bool(params["dither"]),
            "wse": {"rule": params["wse_rule"], "beta": float(params["wse_beta"]),
                    "schedule": [[list(pair) for pair in lev] for lev in run["wse_schedule"]],
                    "truncated_levels": truncated,
                    "decimation": run["samples"]["sets"]["wse"].get("decimation", []),
                    "normalization": run["samples"]["sets"]["wse"].get("normalization", "dense fixed window")},
            "border_cells": run["border_cells"],
            "registration": [{"level": j, "band": band, "shift": list(v[:2]) if band != "detail" else [v[0]],
                              "residual_px": list(v[2:]) if band != "detail" else [v[1]]}
                             for (j, band), v in run["registration"].items()],
            "valid_cells": run["valid_cells"], "zero_cells": run["zero_cells"],
            "admissibility": adm, "warnings": warnings,
        }
        samples = leaders.sample_sets(quantity, mods, valid, run["scales"], run["data_shape"],
                                      wse_schedule=run["wse_schedule"] if params["wse_rule"] == "dumeur" else None)
        spun_maps = None
        if int(params["spin"]) > 1:
            samples, spun_maps, phases = self._spin(vals, params, kw, run, samples, h_cells)
            h_map, r2_map, levels_map = spun_maps[method]
            used = levels_map[np.isfinite(levels_map)]
            record["levels"]["used"] = {"min": int(used.min()), "max": int(used.max())} if used.size else None
            record["spin"] = {"grids": len(phases), "phases": [[int(v) for v in ph] for ph in phases]}
        # The method's set as one structure-function table in the partition schema: the
        # Partition functions window opens on a leaders result like the CWT leaders'.
        hd = leaders.structure_table(samples, params["method"])
        return {
            "h_map": h_map, "r2_map": r2_map, "levels_map": levels_map, "chains": [], "extrema": [],
            "scales": run["scales"], "params": dict(params),
            "samples": samples, "hd_std": hd, "hd_cmax": hd, "hd_kind": "structure",
            "hd_source": f"dyadic {params['method']}",
            "_h_cells": h_cells, "_quantity": quantity, "_valid": valid, "_admissibility": adm,
            "_border_cells": run["border_cells"], "_valid_cells": run["valid_cells"],
            "_dyadic_origin": tuple(int(v) for v in run["phase"]), "_pad": tuple(int(v) for v in run["pad"]),
            "_frame": getattr(field, "frame", None), "_shape": tuple(vals.shape),
            "record": record, "warnings": warnings,
            **({"_spun_maps": spun_maps} if spun_maps is not None else {}),
        }

    def _spin(self, vals, params, kw, run, samples, h_cells):
        """The leaders on s × s phases of the dyadic grid (cycle spinning): the phase steps by
        2^J / s + 1 pixels per axis, so every level's cells shift, the finest included. Every
        phase is a dyadic grid of its own, so the formalism holds on each: the grids' samples are
        pooled (one grid's cell counts and σ's, :func:`leaders.pool_sample_sets`) and each
        method's h-map and R² are averaged pixel by pixel, the levels map keeping the fewest
        levels any grid used there. Returns the pooled samples, the maps per method and the
        phases, the first being ``run``'s own."""
        import warnings as _warnings

        s, cell = int(params["spin"]), 2 ** run["levels"]
        step = cell // s + 1
        base = tuple(int(v) for v in run["phase"])
        phases = [((base[0] + a * step) % cell, (base[1] + b * step) % cell) for a in range(s) for b in range(s)]
        shape = np.shape(vals)
        schedule = run["wse_schedule"] if params["wse_rule"] == "dumeur" else None
        grids = [samples]
        maps = {m: [self._paint(h_cells[m], run["j_min"], shape, run["pad"])] for m in leaders.METHODS}
        for phase in phases[1:]:
            r = leaders.run_leaders(vals, **kw, anchor=phase)
            quantity = {m: [np.asarray(a, dtype=np.float32) for a in r["quantity"][m]] for m in leaders.METHODS}
            mods = [np.asarray(a, dtype=np.float32) for a in r["coefficients"]]
            valid = {m: list(r["valid"][m]) for m in leaders.METHODS + ("coefficients",)}
            grids.append(leaders.sample_sets(quantity, mods, valid, r["scales"], r["data_shape"],
                                             wse_schedule=r["wse_schedule"] if schedule is not None else None))
            for m in leaders.METHODS:
                cells = tuple(np.asarray(a, dtype=np.float32) for a in r["local"][m])
                maps[m].append(self._paint(cells, r["j_min"], shape, r["pad"]))
        with np.errstate(all="ignore"), _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)          # a pixel no grid covers stays NaN
            averaged = {m: tuple(f(np.stack([g[i] for g in stack]), axis=0).astype(np.float32)
                                 for i, f in enumerate((np.nanmean, np.nanmean, np.nanmin)))
                        for m, stack in maps.items()}
        return leaders.pool_sample_sets(grids), averaged, phases

    @staticmethod
    def _paint(cells, j_min: int, shape, pad=None) -> tuple:
        """The h-map, its R² and the levels fitted per cell, painted over the cells; the levels
        map is NaN where the h-map is."""
        h, r2, n = cells
        h_map = leaders.paint(h, j_min, shape, pad).astype(np.float32)
        levels = np.where(np.isfinite(h_map), leaders.paint(n, j_min, shape, pad), np.nan).astype(np.float32)
        return h_map, leaders.paint(r2, j_min, shape, pad).astype(np.float32), levels

    @staticmethod
    def _painted(cells, level: int, result: dict) -> np.ndarray:
        """``cells`` of ``level`` painted over the ROI's own pixels: an ROI result's cells were
        computed on its processing window, ``margin`` px wider on every side, so the window is
        painted and the margin cut; a whole-field result's window is the field."""
        h, w = (int(v) for v in result["_shape"])
        m = int((result.get("_roi") or {}).get("margin") or 0)
        pad = result.get("_pad")
        return leaders.paint(cells, level, (h + 2 * m, w + 2 * m), pad)[m:m + h, m:m + w].astype(np.float32)

    def view(self, result: dict, params: dict) -> dict:
        """The h-map of ``method``: the cached result itself for the method it was computed
        with, else a view built once per method and kept with the result, so the shell sees the
        same object for the same choice (the 3-D scene rebuilds on a new one)."""
        if "_h_cells" not in result:
            return result
        method = params.get("method", result["params"].get("method"))
        if method == result["params"].get("method") or method not in result["_h_cells"]:
            return result
        views = result.setdefault("_views", {})
        if method not in views and "_spun_maps" in result:
            h_map, r2_map, levels_map = result["_spun_maps"][method]
            views[method] = {**result, "h_map": h_map, "r2_map": r2_map, "levels_map": levels_map,
                             "params": {**result["params"], "method": method}}
        if method not in views:
            j_min = int(result["record"]["levels"]["j_min"])
            h_cells, r2_cells, n_cells = result["_h_cells"][method]
            h_map = self._painted(h_cells, j_min, result)
            r2_map = self._painted(r2_cells, j_min, result)
            levels_map = np.where(np.isfinite(h_map), self._painted(n_cells, j_min, result), np.nan).astype(np.float32)
            views[method] = {**result, "h_map": h_map, "r2_map": r2_map, "levels_map": levels_map,
                             "params": {**result["params"], "method": method}}
        return views[method]

    def compute_output(self, name: str, values, result: dict, params: dict, *, fetch,
                       progress=None, cancel=None, level=None) -> dict:
        """One quantity at the level shown on its own decimated grid: log2 of the leader,
        p-leader or (θ, ω)-leader, or the oscillation ratio, NaN on the cells the border or
        non-finite pixels exclude, with the placement :func:`~dynamix.engine.resolve.output_raster`
        repeats them by over the layer's pixels (the level, the grid's offset, the ROI's crop of
        the window)."""
        if name not in _OUTPUT_OF:
            raise ValueError(f"{self.name}: no lazy output named {name!r}")
        key = _OUTPUT_OF[name]
        J = len(result["_quantity"][key])
        idx = int(np.clip(0 if level is None else int(level), 0, J - 1))
        cells = np.asarray(result["_quantity"][key][idx], dtype=np.float64)
        valid = result["_valid"][key][idx]
        if key != "oscillation":
            with np.errstate(divide="ignore", invalid="ignore"):
                cells = np.where(cells > 0, np.log2(np.where(cells > 0, cells, 1.0)), np.nan)
        cells = np.where(valid, cells, np.nan)
        h, w = (int(v) for v in result["_shape"])
        m = int((result.get("_roi") or {}).get("margin") or 0)
        pad = tuple(int(v) for v in (result.get("_pad") or (0, 0)))
        return {"cells": np.asarray(cells, dtype=np.float32),
                "paint": {"level": idx + 1, "pad": pad, "crop": (m, m, h, w)},
                "diag": {"level": idx, "scale_px": 2.0 ** (idx + 1), "valid_cells": int(np.sum(valid))}}

    # ------------------------------------------------------------------------------- reading

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, keyed_params(self, params))

    def reading(self, result: dict, params: dict) -> str:
        """The fitted range, the two regularity verdicts and which methods they admit; the
        frame when γ lifted the coefficients. No exponent: the fitting tools read those."""
        if "_admissibility" not in result:
            return "no leaders — run the transform"
        adm, p = result["_admissibility"], float(result["params"]["p"])
        lev = result["record"]["levels"]
        parts = [f"{2 ** lev['j_min']}–{2 ** lev['J']} px", f"h_min {adm['h_min']['original']:.2f}",
                 f"η({p:g}) {adm['eta']['original']:.2f}"]
        bad = [(label, adm[m]["suggested_gamma"])
               for m, label in (("leaders", "leaders"), ("p-leaders", f"p = {p:g}")) if not adm[m]["admissible"]]
        if not bad:
            parts.append(f"leaders, p = {p:g} and WSE admissible")
        elif len(bad) == 1:
            parts.append(f"{bad[0][0]} inadmissible, γ ≥ {bad[0][1]:g} would clear it")
        else:
            parts.append(f"{bad[0][0]} and {bad[1][0]} inadmissible, γ ≥ {max(g for _, g in bad):g} would clear them")
        if adm["gamma"]:
            parts.append(f"γ = {adm['gamma']:g}, original frame")
        return " · ".join(parts)
