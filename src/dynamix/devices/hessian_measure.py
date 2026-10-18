# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Hessian measure (Nathan–Gunaratne): μ(β) of one field, a planform not required.

The Hessian measure of Nathan and Gunaratne 2005 (PRE 71:035101, Eq. 1): the density
|Λ(x)| = |u_xx u_yy − u_xy²| / Var(u) and μ(β) = ⟨|Λ|^β⟩^{1/(4β)}, an inverse length, as a curve
against the moment order β (:mod:`dynamix.core.disorder`), with the density as a raster.
Derivatives are the papers' DAF by default; spectral derivatives need a periodized input, which a
Periodic extension's stamp or the ``periodic`` knob declares.
"""
from __future__ import annotations

import numpy as np

from dynamix.core import disorder
from dynamix.devices.measures_common import (DETREND_PARAM, FLOOR_PARAM, SCHEME_PARAMS,
                                             deriv_record, periodized, scheme)
from dynamix.model.contracts import CURVE, FIELD
from dynamix.model.param import Param, ParamKind

__all__ = ["HessianMeasure"]


class HessianMeasure:
    """μ(β) of a field, with |Λ(x)| as a raster."""

    name = "hessian_measure"
    label = "Hessian measure (Nathan–Gunaratne)"
    families = ("Measures",)
    role = "primary"                    # a drop on a layer forks a sibling over the same field
    consumes = ((FIELD,),)
    produces = frozenset({CURVE})
    capabilities = frozenset({"SelfReporting", "RegionAware", "Viewable"})
    params = (
        *SCHEME_PARAMS,
        DETREND_PARAM,
        Param("hessian_norm", ParamKind.CHOICE, default="variance", choices=("variance", "std"),
              label="Norm"),
        FLOOR_PARAM,
    )

    def roi_margin(self, params: dict) -> int:
        """The derivative kernel's reach: what an ROI must read beyond its window."""
        return disorder.reach(**scheme(params))

    def compute(self, field, params: dict, *, progress=None) -> dict:
        values = np.asarray(field.values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("hessian_measure: a single-band field")
        sch = scheme(params)
        periodized_by = periodized(field, params)
        v, valid = disorder.detrend(values, params["detrend"])
        density = disorder.hessian_density(v, norm=params["hessian_norm"], detrend_mode="none",
                                           periodic=periodized_by is not None, mask=valid, **sch)
        mu = disorder.hessian_measure(density, disorder.MU_BETAS,
                                      floor=float(params["beta_floor"]) or None)
        warnings = [mu["warning"]] if mu["warning"] else []
        record = {
            "deriv": deriv_record(sch, periodized_by), "detrend": params["detrend"],
            "hessian_norm": params["hessian_norm"],
            "beta_floor": {"value": mu["floor"], "n_floor": mu["n_floor"]},
            "valid_pixels": int(mu["area"]), "warnings": warnings,
        }
        return {
            "beta_mu": np.asarray(disorder.MU_BETAS, dtype=np.float64),
            "hessian_measure": mu["mu"],
            "hessian_density": density, "raster_out": density,
            "record": record, "warnings": warnings, "_shape": tuple(values.shape),
        }

    def curves(self, result, field, params: dict) -> list:
        """For the Curve window: μ(β) against β, log y."""
        from dynamix.model.curves import Curve

        if not isinstance(result, dict) or "hessian_measure" not in result:
            return []
        return [Curve("Hessian measure μ(β)", np.asarray(result["beta_mu"]),
                      np.asarray(result["hessian_measure"]), "β", "μ(β) (px⁻¹)", logy=True,
                      note=f"{result['record']['hessian_norm']} norm; "
                           f"{result['record']['valid_pixels']} valid pixels")]

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)

    def reading(self, result: dict, params: dict) -> str:
        if "hessian_measure" not in result:
            return "no measure — run the transform"
        mu = np.asarray(result["hessian_measure"])
        betas = list(disorder.MU_BETAS)
        return f"μ(1) {mu[betas.index(1.0)]:.3g} px⁻¹ · μ(5) {mu[betas.index(5.0)]:.3g} px⁻¹"
