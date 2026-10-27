# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Directional scaling functions: the directional thermodynamic formalism of Ben Slimane,
Ben Abid, Ben Omrane and Halouani (Symmetry 11:825, 2019) on the hyperbolic (fully separable)
wavelet coefficients of a band (:mod:`dynamix.core.directional`).

The anisotropic scaling function η_α(p) on an α₁ grid of step ⅛ (α₂ = 2 − α₁), read along the
index families (j₁, j₂) = ([jα₁], [jα₂]) as the slowest family's envelope slope over the levels
no family clips (at least three); η(p, e₁) = sup_α η_α(p)/α₁ with the maximising α₁ as the
anisotropy profile α₁*(p), η(p, e₂) by symmetry, η(p) the diagonal α = (1, 1), and the
directional Legendre bound of their Theorem 6, L(H, e), capped at d. The readings are curves
against p and H for the Curve window, and the two directional exponents η(2, e)/2 stand beside
the WTMMM's H_x and H_y in the Directional roughness window. Anisotropy needs depth: at the
finest levels [jα] cannot resolve a fractional α.
"""
from __future__ import annotations

import math
import dataclasses
import hashlib

import numpy as np

from dynamix.core import directional as dr
from dynamix.core import dwt
from dynamix.devices.holder_map import field_values_2d
from dynamix.devices.wtmm import filled_block, filled_origin
from dynamix.model.contracts import CURVE, FIELD
from dynamix.model.device import Output, keyed_params
from dynamix.model.param import Param, ParamKind
from dynamix.model.origin import FIRST_PARTY

__all__ = ["DirectionalScaling"]


class DirectionalScaling:
    """η_α(p), η(p, e₁), η(p, e₂), α₁*(p) and the directional Legendre bounds of a band."""

    provenance_precision = "float64"

    @staticmethod
    def provenance_outcome(result, params):
        from dynamix.engine.execution import resolved_outcome
        return resolved_outcome(result, params)

    provenance_libraries = (('pywt', 'PyWavelets'),)

    name = "directional_scaling"
    label = "Directional scaling functions"
    origin = dataclasses.replace(FIRST_PARTY, citation="Ben Slimane, Ben Abid, Ben Omrane & "
                                 "Halouani (2019), Symmetry 11, 825: the directional thermodynamic "
                                 "formalism on the hyperbolic wavelet basis")
    families = ("Regularity",)
    role = "primary"                    # a drop on a layer forks a sibling over the same field
    consumes = ((FIELD,),)
    produces = frozenset({CURVE})
    capabilities = frozenset({"SelfReporting", "RegionAware", "StripExtras", "OutputProducer"})
    #: Local directional exponents from the hyperbolic leaders' strips (Abry et al. 2015, Thm 3.4).
    outputs = (
        Output("h_x", "raster", lazy=True, label="H_x, strip leaders"),
        Output("h_y", "raster", lazy=True, label="H_y, strip leaders"),
        Output("anisotropy", "raster", lazy=True, label="anisotropy α₁* = 2H_y/(H_x + H_y)"),
    )
    uses_stage_cache = True
    params = (
        Param("wavelet", ParamKind.CHOICE, default="sym4", choices=dwt.BANKS, label="Wavelet"),
        Param("edge", ParamKind.CHOICE, default="auto",
              choices=("auto", "mirror", "reflect", "periodic", "edge_pad", "zero_pad"),
              choice_labels=("auto", "mirror", "reflect", "periodic", "edge-value padding", "zero padding"),
              label="Edge"),
        Param("scale_min", ParamKind.INT, default=4, min=2, max=512, soft_min=2, soft_max=16,
              units="px", label="Finest fitted cell", dyadic=True),
        Param("scale_max", ParamKind.INT, default=64, min=8, max=1024, soft_min=32, soft_max=256,
              units="px", label="Coarsest cell", dyadic=True),
        Param("gamma", ParamKind.FLOAT, default=0.0, min=-2.0, max=4.0, soft_min=0.0, soft_max=2.0,
              label="γ"),
        Param("estimator", ParamKind.CHOICE, default="level_line", choices=("level_line", "gamma", "legacy"),
              choice_labels=("level line (Abry et al. 2015, Prop. 4.7)", "Γ union, read literally (BS19 44–47)",
                             "legacy level-family approximation"), label="Estimator"),
        Param("coarse_partner", ParamKind.CHOICE, default="fallback", choices=dr.COARSE_PARTNER,
              choice_labels=("where details are absent", "always", "never"), label="φ as coarse partner",
              active_when=("estimator", ("level_line",))),
        Param("alpha_step", ParamKind.CHOICE, default="0.125", choices=("0.25", "0.125", "0.0625"),
              label="α₁ increment"),
        Param("p_min", ParamKind.FLOAT, default=0.5, min=0.05, max=20.0, soft_min=0.5, soft_max=4.0, label="p min"),
        Param("p_max", ParamKind.FLOAT, default=6.0, min=0.05, max=20.0, soft_min=2.0, soft_max=10.0, label="p max"),
        Param("p_step", ParamKind.FLOAT, default=0.5, min=0.05, max=5.0, soft_min=0.25, soft_max=1.0, label="p step"),
    )
    inspector_actions = dict.fromkeys(("p_min", "p_max", "p_step", "alpha_step", "scale_min"),
                                      "Refit cached hyperbolic coefficients")
    params = tuple(dataclasses.replace(p, level="advanced") if p.name in
                   ("p_min", "p_max", "p_step", "alpha_step", "scale_min") else p for p in params)
    param_groups = {"PYRAMID": ("wavelet", "edge", "scale_min", "scale_max", "gamma"),
                    "MOMENTS": ("p_min", "p_max", "p_step")}

    @staticmethod
    def _levels(params: dict) -> tuple:
        out = []
        for key in ("scale_min", "scale_max"):
            v = int(params[key])
            if v < 2 or v & (v - 1):
                raise ValueError(f"directional_scaling: {key} {v} px is not a power of two")
            out.append(int(round(math.log2(v))))
        return tuple(out)

    @staticmethod
    def p_list(params: dict) -> np.ndarray:
        lo, hi, step = float(params["p_min"]), float(params["p_max"]), float(params["p_step"])
        if hi < lo:
            raise ValueError(f"directional_scaling: p max {hi:g} below p min {lo:g}")
        from dynamix.core.partition_tables import moment_grid

        return moment_grid(lo, hi, step)

    @staticmethod
    def _edge(field, params: dict) -> str:
        asked = params.get("edge", "auto")
        if asked != "auto":
            return asked
        return "periodic" if filled_block(field) is not None else "mirror"

    def check(self, params: dict) -> None:
        j_min, J = self._levels(params)
        if J - j_min < 2:
            raise ValueError(f"directional_scaling: at least three levels: scale_min {params['scale_min']} px "
                             f"to scale_max {params['scale_max']} px spans {J - j_min + 1}")
        self.p_list(params)

    def roi_margin(self, params: dict) -> int:
        return dwt.border_cells(params["wavelet"]) * int(params["scale_max"])

    footprint_knobs = frozenset({"wavelet", "edge", "scale_min", "scale_max"})

    def footprint(self, field, params: dict, level: int, stride: int = 1) -> dict:
        """A whole-field run at ``level`` (0 = the finest): the cells within the bank's border of
        each edge are read through the mirrored edge and dropped, a band (border_cells − 1)·2^j px
        wide (none under a Periodic extension)."""
        from dynamix.roi.picture import display_stride

        v = np.asarray(field.values, dtype=np.float64)
        v = v[..., 0] if v.ndim == 3 else v
        k = display_stride(field)
        j = min(max(int(level) + 1, 1), self._levels(params)[1])
        edge = self._edge(field, params)
        band = 0 if edge == "periodic" else (dwt.border_cells(params["wavelet"]) - 1) * 2 ** j
        s = max(1, int(stride))
        q = -(-band // (s * k))
        padded = np.pad(v[::s, ::s], q, mode="wrap" if edge == "periodic" else "symmetric")
        return {"edge": edge, "pad": band, "shown": band, "padded": padded, "stride": s * k,
                "offset": -q * s * k, "border": band, "band": "edge exclusion",
                "shape": (v.shape[0] * k, v.shape[1] * k)}

    def roi_footprint(self, field, params: dict, rect, level: int, stride: int = 1) -> dict:
        """An ROI run reads a halo of ``roi_margin`` px around the ROI for every level's support
        (the dataset where it has pixels, mirrored past its edges as the runner reads it) and
        counts only the ROI's own cells: the halo is outlined, nothing inside the ROI is dropped."""
        from dynamix.roi.picture import display_stride

        v = np.asarray(field.values, dtype=np.float64)
        v = v[..., 0] if v.ndim == 3 else v
        k = display_stride(field)
        H, W = v.shape[0] * k, v.shape[1] * k
        r, c, h, w = (int(x) for x in rect)
        m = self.roi_margin(params)
        s = max(1, int(stride)) * k

        def axis(start, length, n):
            px = start - m + s * np.arange(-(-(length + 2 * m) // s))
            period = 2 * (n - 1)
            px = np.abs(px) % period if period > 0 else np.zeros_like(px)
            px = np.where(px >= n, period - px, px)
            return np.minimum(px // k, n // k - 1)

        return {"edge": "reflect", "pad": m, "shown": m, "padded": v[np.ix_(axis(r, h, H), axis(c, w, W))],
                "stride": s, "offset": -m, "border": 0, "band": "halo", "shape": (h, w), "origin": (r, c)}

    def knob_warnings(self, field, params: dict) -> dict:
        """The strip's ⚠ on the coarsest cell when a level would keep fewer than 16 cells inside
        the bank's border on this field (the same rule as the wavelet leaders)."""
        values = getattr(field, "values", None)
        if values is None or np.ndim(values) < 2 or not params:
            return {}
        try:
            j_min, J = self._levels(params)
        except ValueError:
            return {}
        e = 0 if self._edge(field, params) == "periodic" else dwt.border_cells(params["wavelet"]) - 1
        side = min(int(n) for n in (filled_block(field) or np.shape(values)[:2]))
        if all(side // 2 ** j - 2 * e >= 4 for j in range(j_min, J + 1)):
            return {}
        largest = max((2 ** j for j in range(3, 12) if side // 2 ** j - 2 * e >= 4), default=0)
        return {"scale_max": (f"levels above {largest} px keep fewer than 16 cells inside {params['wavelet']}'s "
                              f"border of {e} cells on this field" if largest else
                              f"no level keeps 16 cells inside {params['wavelet']}'s border of {e} cells on this field")}

    def derived_reading(self, name: str, value, field, params: dict | None = None) -> str | None:
        if name == "edge" and field is not None and value == "auto":
            used = self._edge(field, dict(params or {}, edge=value))
            return f"→ {used}" + (" (Periodic extension output)" if used == "periodic" else "")
        return None

    def compute_roi(self, window, core, params: dict, *, info=None, progress=None, stage_cache=None) -> dict:
        """An ROI run: the coefficients on the processing window, whose real halo carries each
        level's support and border, and the statistics over the cells wholly inside the ROI
        ``core``. The halo reflected past the dataset's edge counts as missing, so an ROI cell
        whose wavelet reaches it drops as a whole-field run drops it."""
        r, c, h, w = (int(v) for v in core)
        return self.compute(window, params, progress=progress, stage_cache=stage_cache,
                            region=((r, c), (h, w)), real=(info or {}).get("real"))

    def compute(self, field, params: dict, *, progress=None, stage_cache=None, region=None,
                real=None) -> dict:
        raw = field_values_2d(field, self.name)
        vals = raw if real is None else np.where(real, raw, np.nan)
        self.check(params)
        j_min, J = self._levels(params)
        size = vals.shape if region is None else region[1]
        if min(size) < 2 ** J:
            raise ValueError(f"directional_scaling: a {size[0]} × {size[1]} px "
                             f"{'ROI' if region else 'field'} is below scale_max {2 ** J} px")
        edge = self._edge(field, params)
        p = self.p_list(params)
        from dynamix.core.wtmm_backend import _field_data_hash
        from dynamix.engine.cache import cache_key

        origin, block = filled_origin(field), filled_block(field)
        key = cache_key("directional-coefficients", _field_data_hash(field),
                        {"wavelet": params["wavelet"], "J": J, "gamma": params["gamma"], "edge": edge,
                         "origin": origin, "block": block, "complete": params["estimator"] != "legacy",
                         "real": None if real is None else hashlib.sha1(np.packbits(real)).hexdigest()})
        if stage_cache is not None and key in stage_cache:
            coef = stage_cache[key]
        else:
            coef = dr.coefficients(vals, params["wavelet"], J, gamma=float(params["gamma"]), edge=edge,
                                   origin=origin, block=block, complete=params["estimator"] != "legacy")
            if stage_cache is not None:
                stage_cache[key] = coef
        if region is not None:
            coef = dr.restrict(coef, *region)
            (r, c), (h, w) = region
            vals = vals[r:r + h, c:c + w]
        d = dr.analyse(vals, params["wavelet"], J, j_min=j_min, p_list=p, gamma=float(params["gamma"]),
                       edge=edge, estimator=params["estimator"], step=float(params["alpha_step"]),
                       origin=origin, block=block, coefficient_cache=coef,
                       coarse_partner=params.get("coarse_partner", "fallback"))
        warnings = _fit_warnings(raw if real is None else raw[real], coef, d, params,
                                 roi=None if region is None else region[1])
        fitted = int(np.isfinite(d["eta_alpha"]).any(axis=0).sum())
        record = {
            "method": self.name, "wavelet": params["wavelet"], "edge": {"asked": params["edge"], "used": edge},
            "levels": {"j_min": j_min, "J": J}, "gamma": float(params["gamma"]), "alpha_step": float(params["alpha_step"]),
            "estimator": params["estimator"], "estimator_description": d["estimator_description"],
            "coarse_partner": d.get("coarse_partner"), "floor": d.get("floor"),
            "no_structure": d.get("no_structure"), "H_e1": d.get("H_e1"), "H_e2": d.get("H_e2"),
            "alpha_star_e1_interval": np.asarray(d["alpha_star_e1_interval"]).tolist(),
            "complete_basis": d["complete_basis"],
            "resolution_origin": J, "origin": list(origin), "block": list(block) if block else None,
            "seam": d["seam"], "finite_scale_estimate": True,
            "p": [float(v) for v in p], "levels_per_alpha": [int(v) for v in d["n_levels"]],
            "eta_e1": [float(v) for v in d["eta_e1"]], "eta_e2": [float(v) for v in d["eta_e2"]],
            "alpha_star_e1": [float(v) for v in d["alpha_star_e1"]], "warnings": warnings,
        }
        crop = None if region is None else (*region[0], *region[1])
        return {"_directional": d, "fitted_anisotropies": fitted, "params": dict(params),
                "_strips": _strips(coef), "_map_crop": crop, "_map_shape": tuple(raw.shape),
                "_shape": tuple(vals.shape), "_frame": getattr(field, "frame", None),
                "record": record, "warnings": warnings}

    def curves(self, result, field, params: dict) -> list:
        """For the Curve window: η(p, e₁), η(p, e₂), η(p) and α₁*(p) against p; L(H, e₁) and
        L(H, e₂) against H."""
        from dynamix.model.curves import Curve

        d = result.get("_directional") if isinstance(result, dict) else None
        if d is None:
            return []
        p = np.asarray(d["p"])
        fitted = int(np.isfinite(np.asarray(d["eta_alpha"])).any(axis=0).sum())
        note = f"{fitted} of {d['alpha'].size} anisotropies fitted ({d.get('estimator_description', d['estimator'])})"
        if float(params.get("gamma", 0.0) or 0.0):
            note += f"; coefficient integration γ = {float(params['gamma']):g}, order γ/2 per axis"
        out = [Curve(name, p, np.asarray(d[key]), "p", "η", note=note)
               for key, name in (("eta_e1", "η(p, e₁), along x"), ("eta_e2", "η(p, e₂), along y"),
                                 ("eta_iso", "η(p), the diagonal α = (1, 1)"))]
        out.append(Curve("α₁*(p), the anisotropy profile of e₁", p, np.asarray(d["alpha_star_e1"]), "p", "α₁*",
                         note="the largest α₁ within one standard error of the maximum of η_α(p)/α₁ "
                              "(the window shows the whole interval); 1 is isotropic"))
        for key, name in (("L_e1", "upper bound on D(H, e₁)"), ("L_e2", "upper bound on D(H, e₂)")):
            leg = d[key]
            knee = leg.get("knee", float("nan"))
            out.append(Curve(name, np.asarray(leg["h"]), np.asarray(leg["L"]), "H", "D",
                             note="Theorem 6 (Ben Slimane et al. 2019) bounds the upper spectrum "
                                  "dim{h(·, e) ≤ H}, which never decreases; empty where no value is drawn; "
                                  "its rising stretch is set by p max"
                                  + (f"; knee at H = {knee:.2f}, below H_e by about d(1 − α_min)/(p_max α_e)"
                                     if np.isfinite(knee) else "")))
        return out

    def compute_output(self, name: str, values, result: dict, params: dict, *, fetch=None,
                       progress=None, cancel=None, level=None) -> dict:
        """H_x, H_y or the anisotropy at every pixel (of the ROI on an ROI run), NaN where a strip
        reaches the border or a missing pixel. Their resolution is anisotropic: the coarsest fitted
        cell along the fine axis, 3·scale_max px along the coarse one."""
        if name not in ("h_x", "h_y", "anisotropy"):
            raise ValueError(f"{self.name}: no lazy output named {name!r}")
        maps = result.get("_maps")
        if maps is None:
            j_min, _J = self._levels(params)
            maps = result["_maps"] = dr.strip_maps(result["_strips"], j_min, result["_map_shape"],
                                                   crop=result.get("_map_crop"))
        return {"raster": np.asarray(maps[name], dtype=np.float32),
                "diag": {"levels": maps["levels"].tolist(), "coarse_level": maps["coarse_level"],
                         "valid_fraction": float(np.isfinite(maps[name]).mean())}}

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, keyed_params(self, params))

    def reading(self, result: dict, params: dict) -> str:
        d = result.get("_directional") if isinstance(result, dict) else None
        if d is None:
            return "no scaling functions — run the transform"
        p = np.asarray(d["p"], dtype=np.float64)
        at = np.flatnonzero(np.isclose(p, 2.0))
        empty = d.get("no_structure") or {}
        parts = []
        if "H_e1" in d:
            span = f"p {p.min():g}–{p.max():g}"
            hx = "no structure along x" if empty.get("x") else f"H_x {d['H_e1']:.2f}"
            hy = "no structure along y" if empty.get("y") else f"H_y {d['H_e2']:.2f}"
            parts.append(f"{hx} · {hy} (slopes of η(p, e) over {span})")
        if at.size:
            i = at[0]
            lo, hi = np.asarray(d["alpha_star_e1_interval"])[:, i]
            if np.isfinite(lo):
                parts.append(f"α₁* {lo:.3g}–{hi:.3g} at p = 2" if hi > lo else f"α₁* {lo:.3g} at p = 2")
        parts.append(f"{result['fitted_anisotropies']} of {d['alpha'].size} anisotropies fitted")
        return " · ".join(parts)


def _strips(coef: dict) -> dict:
    """The leaders the local maps read: fine along one axis and coarsest along the other."""
    lead = dr.hyperbolic_leaders({"levels": coef["levels"],
                                  "mods": {k: v for k, v in coef["mods"].items() if min(k) > 0},
                                  "valid": {k: v for k, v in coef["valid"].items() if min(k) > 0}})
    J = lead["levels"]
    keep = {(l, J) for l in range(1, J + 1)} | {(J, l) for l in range(1, J + 1)}
    return {"levels": J, "leaders": {k: lead["leaders"][k] for k in keep},
            "valid": {k: lead["valid"][k] for k in keep}}


def _fit_warnings(observed, coef: dict, d: dict, params: dict, roi=None) -> list:
    """What kept the fit or the bound from existing, in the order to fix it: an anisotropy needs
    three resolutions whose every Γ member has valid cells; a bound needs η_α(p) to reach d = 2
    inside the p range; an α* at the end of the fitted anisotropies is only a bound.
    ``observed``: the real pixels read, halo included; ``roi``: an ROI run's (h, w)."""
    J, e = int(d["levels"]), int(coef["border_cells"])
    out = []
    if roi is not None:
        largest = max((2 ** j for j in range(1, J + 1) if min(roi) // 2 ** j >= 4), default=0)
        if largest < 2 ** J:
            out.append(f"Levels above {largest} px keep fewer than 16 cells inside the {roi[0]} × {roi[1]} "
                       "px ROI: draw it wider or lower scale max.")
    fitted = np.flatnonzero(np.isfinite(d["eta_alpha"]).any(axis=0))
    for axis, name in (("x", "e₁"), ("y", "e₂")):
        if (d.get("no_structure") or {}).get(axis):
            out.append(f"No structure along {axis}: every {axis}-detail pair is rounding (a field "
                       f"layered along {axis}), so η(p, {name}) is not read.")
    iso = np.asarray(d.get("eta_iso", []), dtype=np.float64)
    bad = np.asarray(d["p"])[np.isfinite(iso) & (iso <= 0)]
    if bad.size:
        out.append(f"η(p) ≤ 0 at p = {', '.join(f'{v:g}' for v in bad)}: the directional sup (Ben Slimane "
                   "et al.'s eq. 50) holds only where η(p) > 0, so those p are not read; raise γ.")
    if fitted.size == 0:
        what = ("three frontier points" if d.get("estimator") == "level_line"
                else "three resolutions")
        out.insert(0, f"No anisotropy has {what} with valid cells between {params['scale_min']} "
                      f"and {params['scale_max']} px.")
        missing = float(np.mean(~np.isfinite(observed))) if np.size(observed) else 0.0
        if missing > 0:
            out.append(f"{100 * missing:.1f}% of the pixels are missing and every cell whose wavelet reaches "
                       "one is dropped" + (f", a square of (2·{e}+1)·2^j px around each at level j" if e else "")
                       + ": put Fill before Directional scaling.")
        diagonal = [j for j in range(1, J + 1) if d["valid_cells"][j - 1][j - 1] > 0]
        if roi is None and e and diagonal and diagonal[-1] < J:
            out.append(f"Cells survive only up to {2 ** diagonal[-1]} px: {params['wavelet']} drops {e} cells "
                       "per side at every level"
                       + ("" if params["wavelet"] == "db2" else
                          "; db2 drops the fewest but has two vanishing moments, so it reads exponents "
                          "below 2 and a quadratic trend leaks into it")
                       + ". An ROI inside a larger dataset reads those cells from its surroundings.")
        return out
    alpha = np.asarray(d["alpha"], dtype=np.float64)
    lo, hi = alpha[fitted].min(), alpha[fitted].max()
    pinned = []
    for key, axis, ends in (("alpha_star_e1", "e₁", (lo, hi)), ("alpha_star_e2", "e₂", (2 - hi, 2 - lo))):
        grid = (alpha.min(), alpha.max()) if axis == "e₁" else (2 - alpha.max(), 2 - alpha.min())
        star = np.asarray(d[key], dtype=np.float64)
        star = star[np.isfinite(star)]
        if any(np.isclose(star, end).any() and not np.isclose(end, edge)
               for end, edge in zip(ends, grid)):
            pinned.append(axis)
    if pinned:
        out.append(f"α* along {' and '.join(pinned)} sits at an end of the fitted anisotropies (α₁ {lo:g} to "
                   f"{hi:g}), so η(p, e) there is a lower bound: widen the scale range to fit more of them.")
    largest = float(np.nanmax(d["eta_alpha"]))
    if largest < 2.0:
        out.append(f"η stays below d = 2 at every fitted anisotropy in this p range (largest {largest:.2f}), "
                   "so there is no directional bound in it: raise γ; raise p max only if η is still "
                   "rising at p max.")
    return out
