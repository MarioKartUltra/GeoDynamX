# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Wavelet leaders: the regularity of a band read three ways on the dyadic pyramid.

The leaders, p-leaders and (θ, ω)-leaders of :mod:`dynamix.core.leaders` on the critically
sampled pyramid of :mod:`dynamix.core.dwt` (each level half the resolution of the last), in one
pass: for each the structure-function tables the fitter reads (τ(q) = ζ(q) − 2, h(q), D(q)), the
log-cumulants, a local exponent per cell painted on the native grid as the h-map, and the
per-level quantities on the Scale slider. Each method's admissibility on the data at hand is
stamped (h_min > 0 for the leaders, η(p) > 0 for the p-leaders, nothing for the (θ, ω)-leaders)
with the pseudo-fractional integration γ that would clear it. ``method`` chooses which of the
three the h-map and the fitter show; switching it recomputes nothing.
"""
from __future__ import annotations

import math

import numpy as np

from dynamix.core import dwt, leaders
from dynamix.devices.holder_map import field_values_2d
from dynamix.devices.wtmm import MOMENT_PARAMS, q_list
from dynamix.model.contracts import FIELD, HMAP
from dynamix.model.device import Output, keyed_params
from dynamix.model.param import Param, ParamKind

__all__ = ["WaveletLeaders"]

#: Lazy output name -> the quantity it paints.
_OUTPUT_OF = {"leader": "leaders", "p_leader": "p-leaders", "wse": "wse", "oscillation": "oscillation"}
_METHOD_LABEL = {"leaders": "leaders", "p-leaders": "p-leaders", "wse": "WSE leaders"}


class WaveletLeaders:
    """Leaders, p-leaders and (θ, ω)-leaders of a band, with their readouts."""

    name = "wavelet_leaders"
    label = "Wavelet leaders"
    families = ("Regularity",)
    role = "primary"                    # a drop on a layer forks a sibling over the same field
    consumes = ((FIELD,),)
    produces = frozenset({HMAP})
    capabilities = frozenset({"SelfReporting", "RegionAware", "Viewable", "OutputProducer",
                              "StripExtras"})
    params = (
        Param("wavelet", ParamKind.CHOICE, default="sym4", choices=dwt.BANKS, label="Wavelet"),
        Param("scale_min", ParamKind.INT, default=4, min=2, max=512, soft_min=2, soft_max=16,
              units="px", label="Finest fitted cell"),
        Param("scale_max", ParamKind.INT, default=64, min=4, max=1024, soft_min=16, soft_max=128,
              units="px", label="Coarsest fitted cell"),
        Param("gamma", ParamKind.FLOAT, default=0.0, min=-2.0, max=4.0, soft_min=0.0, soft_max=2.0,
              label="γ"),
        Param("dither", ParamKind.BOOL, default=False, label="Dither"),
        Param("p", ParamKind.FLOAT, default=2.0, min=0.25, max=16.0, soft_min=0.5, soft_max=4.0,
              label="p"),
        Param("wse_rule", ParamKind.CHOICE, default="dumeur", choices=("dumeur", "fixed"),
              choice_labels=("resolution-indexed (Dumeur)", "fixed depth and width"), label="Window"),
        Param("wse_beta", ParamKind.FLOAT, default=0.25, min=0.05, max=0.95, soft_min=0.1, soft_max=0.5,
              label="β", active_when=("wse_rule", ("dumeur",))),
        Param("wse_depth", ParamKind.INT, default=1, min=0, max=4, label="Depth (octaves)",
              active_when=("wse_rule", ("fixed",))),
        Param("wse_width", ParamKind.FLOAT, default=1.0, min=0.0, max=8.0, soft_min=0.0, soft_max=3.0,
              label="Half-width (cells)", active_when=("wse_rule", ("fixed",))),
        *MOMENT_PARAMS,
        # The directional formalism on the hyperbolic coefficients: a second transform.
        Param("directional", ParamKind.BOOL, default=False, label="Directional"),
        # View-only: what the h-map and the fitter show first, and the raster drawn.
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
        "PYRAMID": ("wavelet", "scale_min", "scale_max", "gamma", "dither"),
        "P-LEADERS": ("p",),
        "WSE": ("wse_rule", "wse_beta", "wse_depth", "wse_width"),
        "MOMENTS": ("q_min", "q_max", "q_step"),
        "DIRECTIONAL": ("directional",),
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

    def check(self, params: dict) -> None:
        j_min, J = self._levels(params)
        if J - j_min < 2:
            raise ValueError(f"wavelet_leaders: at least three fitted levels: scale_min {params['scale_min']} "
                             f"px to scale_max {params['scale_max']} px spans {J - j_min + 1}")
        q_list(params)

    def roi_margin(self, params: dict) -> int:
        """What an ROI must read beyond its window: the bank's border in cells of the coarsest level."""
        return dwt.border_cells(params["wavelet"]) * int(params["scale_max"])

    def derived_reading(self, name: str, value, field, params: dict | None = None) -> str | None:
        """The fitted cells in the field's units beside ``scale_min`` and ``scale_max``."""
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

    def _fits(self, shape, params: dict) -> tuple:
        """Whether every fitted level keeps at least 16 cells inside the border (4 × 4), the
        border, and the largest ``scale_max`` whose three coarsest levels all do (0 when none)."""
        j_min, J = self._levels(params)
        e = dwt.border_cells(params["wavelet"])
        side = min(int(n) for n in shape)
        fits = all(side // 2 ** j - 2 * e >= 4 for j in range(j_min, J + 1))
        largest = max((2 ** j for j in range(3, 12) if side // 2 ** j - 2 * e >= 4), default=0)
        return fits, largest, e

    def compute(self, field, params: dict, *, progress=None) -> dict:
        vals = field_values_2d(field, self.name)
        self.check(params)
        fits, largest, e = self._fits(vals.shape, params)
        if not fits:
            fit_note = (f"the largest scale_max that fits three levels is {largest} px" if largest
                        else "no scale_max fits three levels")
            raise ValueError(
                f"wavelet_leaders: a {vals.shape[0]} × {vals.shape[1]} px field with {params['wavelet']}'s "
                f"border of {e} cells leaves fewer than 16 cells on a level between {params['scale_min']} "
                f"and {params['scale_max']} px; {fit_note}")
        if params["dither"]:
            from dynamix.core.mz_edges import _dithered, measure_lsb

            vals = _dithered(vals, measure_lsb(vals))
        tick = (lambda frac: progress("wavelet leaders", frac)) if progress is not None else None
        run = leaders.run_leaders(
            vals, wavelet=params["wavelet"], scale_min=int(params["scale_min"]),
            scale_max=int(params["scale_max"]), gamma=float(params["gamma"]), p=float(params["p"]),
            wse_rule=params["wse_rule"], wse_depth=int(params["wse_depth"]),
            wse_width=float(params["wse_width"]), wse_beta=float(params["wse_beta"]),
            q_list=q_list(params), progress=tick)
        j_min, J = run["j_min"], run["levels"]
        warnings = []
        adm = run["admissibility"]
        for m in leaders.METHODS:
            if not adm[m]["admissible"]:
                warnings.append(f"{_METHOD_LABEL[m]} inadmissible on this field (h_min "
                                f"{adm['h_min']['integrated']:.2f}, η({params['p']:g}) "
                                f"{adm['eta']['integrated']:.2f}); γ ≥ {adm[m]['suggested_gamma']:g} would clear it")
        for j in range(j_min, J + 1):
            inside = (vals.shape[0] // 2 ** j - 2 * e) * (vals.shape[1] // 2 ** j - 2 * e)
            if run["valid_cells"]["leaders"][j - 1] < inside / 2:
                warnings.append(f"level {j} ({2 ** j} px): more than half its cells touch non-finite pixels")
            if run["zero_cells"][j - 1] > 0.01 * max(run["valid_cells"]["leaders"][j - 1], 1):
                warnings.append(f"level {j} ({2 ** j} px): {run['zero_cells'][j - 1]} leaders are exactly zero "
                                "(quantised data); the dither knob perturbs by half a least significant bit")
                break
        directional = None
        if params["directional"]:
            from dynamix.core import directional as dr

            q = q_list(params)
            directional = dr.analyse(vals, params["wavelet"], J, j_min=j_min, p_list=q[q > 0],
                                     gamma=float(params["gamma"]))
            if not np.isfinite(directional["eta_alpha"]).any():
                warnings.append("directional: no anisotropy keeps three unclipped levels between "
                                f"{params['scale_min']} and {params['scale_max']} px; deepen scale_max")
        h_cells = {m: (h.astype(np.float32), r2.astype(np.float32)) for m, (h, r2) in run["local"].items()}
        quantity = {m: [np.asarray(a, dtype=np.float32) for a in run["quantity"][m]] for m in leaders.METHODS}
        quantity["oscillation"] = [np.asarray(a, dtype=np.float32) for a in run["oscillation"]]
        valid = {m: list(run["valid"][m]) for m in leaders.METHODS}
        valid["oscillation"] = [a & b for a, b in zip(valid["leaders"], valid["wse"])]
        method = params["method"]
        h_map, r2_map = self._paint(h_cells[method], j_min, vals.shape)
        record = {
            "method": self.name, "wavelet": params["wavelet"], "levels": {"j_min": j_min, "J": J},
            "gamma": float(params["gamma"]), "p": float(params["p"]), "dither": bool(params["dither"]),
            "wse": {"rule": params["wse_rule"], "schedule": [[list(pair) for pair in lev] for lev in run["wse_schedule"]]},
            "border_cells": run["border_cells"],
            "registration": [{"level": j, "band": band, "shift": list(v[:2]) if band != "detail" else [v[0]],
                              "residual_px": list(v[2:]) if band != "detail" else [v[1]]}
                             for (j, band), v in run["registration"].items()],
            "valid_cells": run["valid_cells"], "zero_cells": run["zero_cells"],
            "admissibility": adm, "cumulants": run["cumulants"], "warnings": warnings,
        }
        if directional is not None:
            record["directional"] = {
                "alpha_step": dr.ALPHA_STEP, "p": [float(v) for v in directional["p"]],
                "levels_per_alpha": [int(v) for v in directional["n_levels"]],
                "eta_e1": [float(v) for v in directional["eta_e1"]],
                "eta_e2": [float(v) for v in directional["eta_e2"]],
                "alpha_star_e1": [float(v) for v in directional["alpha_star_e1"]],
            }
        return {
            "h_map": h_map, "r2_map": r2_map, "chains": [], "extrema": [],
            "scales": run["scales"], "params": dict(params),
            "hd_std": run["tables"][method], "hd_cmax": run["tables"][method],
            "hd_tables": run["tables"], "hd_source": self._source(method, params), "hd_kind": "structure",
            "_h_cells": h_cells, "_quantity": quantity, "_valid": valid,
            "_cumulants": run["cumulants"], "_admissibility": adm,
            "_border_cells": run["border_cells"], "_valid_cells": run["valid_cells"],
            "_dyadic_origin": (0, 0), "_directional": directional,
            "_frame": getattr(field, "frame", None), "_shape": tuple(vals.shape),
            "record": record, "warnings": warnings,
        }

    @staticmethod
    def _source(method: str, params: dict) -> str:
        return f"wavelet {_METHOD_LABEL[method]} (DWT, {params['wavelet']})"

    @staticmethod
    def _paint(cells, j_min: int, shape) -> tuple:
        h, r2 = cells
        return (leaders.paint(h, j_min, shape).astype(np.float32),
                leaders.paint(r2, j_min, shape).astype(np.float32))

    def view(self, result: dict, params: dict) -> dict:
        """The h-map and the fitter's table of ``method``: the cached result itself for the method
        it was computed with, else a view built once per method and kept with the result, so the
        shell sees the same object for the same choice (the 3-D scene rebuilds on a new one)."""
        if "_h_cells" not in result:
            return result
        method = params.get("method", result["params"].get("method"))
        if method == result["params"].get("method") or method not in result["_h_cells"]:
            return result
        views = result.setdefault("_views", {})
        if method not in views:
            j_min = int(result["record"]["levels"]["j_min"])
            h_map, r2_map = self._paint(result["_h_cells"][method], j_min, result["_shape"])
            views[method] = {**result, "h_map": h_map, "r2_map": r2_map,
                             "hd_std": result["hd_tables"][method], "hd_cmax": result["hd_tables"][method],
                             "hd_source": self._source(method, result["params"]),
                             "params": {**result["params"], "method": method}}
        return views[method]

    def compute_output(self, name: str, values, result: dict, params: dict, *, fetch,
                       progress=None, cancel=None, level=None) -> dict:
        """One quantity at the level shown, painted over its cells: log2 of the leader, p-leader
        or (θ, ω)-leader, or the oscillation ratio; NaN on the cells the border or non-finite
        pixels exclude."""
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
        raster = leaders.paint(cells, idx + 1, result["_shape"]).astype(np.float32)
        return {"raster": raster, "diag": {"level": idx, "scale_px": 2.0 ** (idx + 1),
                                           "valid_cells": int(np.sum(valid))}}

    # ------------------------------------------------------------------------------ readouts

    def curves(self, result, field, params: dict) -> list:
        """For the Curve window: the first two cumulants of ln ℓ per level for each method, and
        the two regularity lines the verdicts come from, log₂ max |d| and log₂ mean |d|^p; the
        fitted range marked."""
        from dynamix.model.curves import Curve

        if not isinstance(result, dict) or "hd_tables" not in result:
            return []
        lev = result["record"]["levels"]
        marks = ((2.0 ** lev["j_min"], "finest fitted"), (2.0 ** lev["J"], "coarsest fitted"))
        out = []
        for m in leaders.METHODS:
            t = result["hd_tables"][m]
            mom = np.asarray(t["lnT_moments"])
            label = _METHOD_LABEL[m]
            c = result["_cumulants"][m]["original"]
            out.append(Curve(f"C₁(a) of ln {label}", np.asarray(t["scales"]), mom[0], "a (px)",
                             "mean ln ℓ", logx=True, marks=marks, note=f"slope c1 = {c['c1']:.3f}"))
            out.append(Curve(f"C₂(a) of ln {label}", np.asarray(t["scales"]), mom[1], "a (px)",
                             "variance of ln ℓ", logx=True, marks=marks,
                             note=f"c2 = {c['c2']:+.4f} (the app's cascade sign)"))
        d = result.get("_directional")
        if d is not None:
            p = np.asarray(d["p"])
            fitted = int(np.sum(d["n_levels"] >= 3))
            note = f"{fitted} of {d['alpha'].size} anisotropies keep three unclipped levels"
            for key, name in (("eta_e1", "η(p, e₁), along x"), ("eta_e2", "η(p, e₂), along y"),
                              ("eta_iso", "η(p), the diagonal α = (1, 1)")):
                out.append(Curve(name, p, np.asarray(d[key]), "p", "η", note=note))
            out.append(Curve("α₁*(p), the anisotropy profile of e₁", p, np.asarray(d["alpha_star_e1"]),
                             "p", "α₁*", note="the largest maximiser of η_α(p)/α₁; 1 is isotropic"))
            for key, name in (("L_e1", "L(H, e₁), the directional Legendre bound"),
                              ("L_e2", "L(H, e₂), the directional Legendre bound")):
                out.append(Curve(name, np.asarray(d[key]["h"]), np.asarray(d[key]["L"]), "H", "L",
                                 note="Theorem 6's upper bound over p ≥ p_α, capped at d = 2"))
        return out

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, keyed_params(self, params))

    def reading(self, result: dict, params: dict) -> str:
        if "_admissibility" not in result:
            return "no leaders — run the transform"
        adm, cums, p = result["_admissibility"], result["_cumulants"], float(result["params"]["p"])
        parts = []
        for m, label in (("leaders", "leaders"), ("p-leaders", f"p = {p:g}"), ("wse", "WSE")):
            c = cums[m]["original"]
            text = f"{label} c1 {c['c1']:.2f} c2 {c['c2']:+.3f}"
            if not adm[m]["admissible"]:
                text += f" (inadmissible, γ ≥ {adm[m]['suggested_gamma']:g})"
            parts.append(text)
        lev = result["record"]["levels"]
        parts.append(f"{2 ** lev['j_min']}–{2 ** lev['J']} px · h_min {adm['h_min']['original']:.2f} · "
                     f"η({p:g}) {adm['eta']['original']:.2f}")
        if adm["gamma"]:
            parts.append(f"γ = {adm['gamma']:g}, original frame")
        return " · ".join(parts)
