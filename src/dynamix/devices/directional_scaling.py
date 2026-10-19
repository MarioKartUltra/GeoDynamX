# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Directional scaling functions: the directional thermodynamic formalism of Ben Slimane,
Ben Braiek, Ben Abid, Ben Omrane and Mabrouk (2019) on the hyperbolic (fully separable) wavelet
coefficients of a band (:mod:`dynamix.core.directional`).

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

import numpy as np

from dynamix.core import directional as dr
from dynamix.core import dwt
from dynamix.devices.holder_map import field_values_2d
from dynamix.devices.wtmm import filled_block, filled_origin
from dynamix.model.contracts import CURVE, FIELD
from dynamix.model.device import keyed_params
from dynamix.model.param import Param, ParamKind

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
    families = ("Regularity",)
    role = "primary"                    # a drop on a layer forks a sibling over the same field
    consumes = ((FIELD,),)
    produces = frozenset({CURVE})
    capabilities = frozenset({"SelfReporting", "RegionAware", "StripExtras"})
    uses_stage_cache = True
    params = (
        Param("wavelet", ParamKind.CHOICE, default="sym4", choices=dwt.BANKS, label="Wavelet"),
        Param("edge", ParamKind.CHOICE, default="auto",
              choices=("auto", "mirror", "reflect", "periodic", "edge_pad", "zero_pad"),
              choice_labels=("auto", "mirror", "reflect", "periodic", "edge-value padding", "zero padding"),
              label="Edge"),
        Param("scale_min", ParamKind.INT, default=4, min=2, max=512, soft_min=2, soft_max=16,
              units="px", label="Finest fitted cell"),
        Param("scale_max", ParamKind.INT, default=64, min=8, max=1024, soft_min=32, soft_max=256,
              units="px", label="Coarsest cell"),
        Param("gamma", ParamKind.FLOAT, default=0.0, min=-2.0, max=4.0, soft_min=0.0, soft_max=2.0,
              label="γ"),
        Param("estimator", ParamKind.CHOICE, default="gamma", choices=("gamma", "legacy"),
              choice_labels=("resolution-index Γ union", "legacy level-family approximation"), label="Estimator"),
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

    def derived_reading(self, name: str, value, field, params: dict | None = None) -> str | None:
        if name == "edge" and field is not None and value == "auto":
            used = self._edge(field, dict(params or {}, edge=value))
            return f"→ {used}" + (" (Periodic extension output)" if used == "periodic" else "")
        return None

    def compute(self, field, params: dict, *, progress=None, stage_cache=None) -> dict:
        vals = field_values_2d(field, self.name)
        self.check(params)
        j_min, J = self._levels(params)
        if min(vals.shape) < 2 ** J:
            raise ValueError(f"directional_scaling: a {vals.shape[0]} × {vals.shape[1]} px field is below "
                             f"scale_max {2 ** J} px")
        edge = self._edge(field, params)
        p = self.p_list(params)
        from dynamix.core.wtmm_backend import _field_data_hash
        from dynamix.engine.cache import cache_key

        origin, block = filled_origin(field), filled_block(field)
        key = cache_key("directional-coefficients", _field_data_hash(field),
                        {"wavelet": params["wavelet"], "J": J, "gamma": params["gamma"], "edge": edge,
                         "origin": origin, "block": block, "estimator": params["estimator"]})
        if stage_cache is not None and key in stage_cache:
            coef = stage_cache[key]
        else:
            coef = dr.coefficients(vals, params["wavelet"], J, gamma=float(params["gamma"]), edge=edge,
                                   origin=origin, block=block, complete=params["estimator"] == "gamma")
            if stage_cache is not None:
                stage_cache[key] = coef
        d = dr.analyse(vals, params["wavelet"], J, j_min=j_min, p_list=p, gamma=float(params["gamma"]),
                       edge=edge, estimator=params["estimator"], step=float(params["alpha_step"]),
                       origin=origin, block=block, coefficient_cache=coef)
        warnings = []
        fitted = int(np.sum(d["n_levels"] >= 3))
        if fitted == 0:
            warnings.append(f"no anisotropy keeps three unclipped levels between {params['scale_min']} and "
                            f"{params['scale_max']} px; deepen scale_max")
        record = {
            "method": self.name, "wavelet": params["wavelet"], "edge": {"asked": params["edge"], "used": edge},
            "levels": {"j_min": j_min, "J": J}, "gamma": float(params["gamma"]), "alpha_step": float(params["alpha_step"]),
            "estimator": params["estimator"], "complete_basis": d["complete_basis"],
            "resolution_origin": J, "origin": list(origin), "block": list(block) if block else None,
            "seam": d["seam"], "finite_scale_estimate": True,
            "p": [float(v) for v in p], "levels_per_alpha": [int(v) for v in d["n_levels"]],
            "eta_e1": [float(v) for v in d["eta_e1"]], "eta_e2": [float(v) for v in d["eta_e2"]],
            "alpha_star_e1": [float(v) for v in d["alpha_star_e1"]], "warnings": warnings,
        }
        return {"_directional": d, "fitted_anisotropies": fitted, "params": dict(params),
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
        note = f"{int(np.sum(d['n_levels'] >= 3))} of {d['alpha'].size} anisotropies keep three unclipped levels"
        out = [Curve(name, p, np.asarray(d[key]), "p", "η", note=note)
               for key, name in (("eta_e1", "η(p, e₁), along x"), ("eta_e2", "η(p, e₂), along y"),
                                 ("eta_iso", "η(p), the diagonal α = (1, 1)"))]
        out.append(Curve("α₁*(p), the anisotropy profile of e₁", p, np.asarray(d["alpha_star_e1"]), "p", "α₁*",
                         note="the largest maximiser of η_α(p)/α₁; 1 is isotropic"))
        out += [Curve(name, np.asarray(d[key]["h"]), np.asarray(d[key]["L"]), "H", "L",
                      note="Theorem 6's upper bound over p ≥ p_α, capped at d = 2")
                for key, name in (("L_e1", "L(H, e₁), the directional Legendre bound"),
                                  ("L_e2", "L(H, e₂), the directional Legendre bound"))]
        return out

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, keyed_params(self, params))

    def reading(self, result: dict, params: dict) -> str:
        d = result.get("_directional") if isinstance(result, dict) else None
        if d is None:
            return "no scaling functions — run the transform"
        p = np.asarray(d["p"], dtype=np.float64)
        at = np.flatnonzero(np.isclose(p, 2.0))
        parts = []
        if at.size:
            i = at[0]
            parts.append(f"η(2, e₁)/2 {d['eta_e1'][i] / 2:.2f} · η(2, e₂)/2 {d['eta_e2'][i] / 2:.2f} · "
                         f"α₁*(2) {d['alpha_star_e1'][i]:.3g}")
        parts.append(f"{result['fitted_anisotropies']} of {d['alpha'].size} anisotropies keep three unclipped levels")
        return " · ".join(parts)
