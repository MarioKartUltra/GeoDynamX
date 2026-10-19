# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""An explicit field-preparation step for Fourier fractional integration."""
from __future__ import annotations

import dataclasses

import numpy as np

from dynamix.model.contracts import FIELD, RECEIVED
from dynamix.model.param import Param, ParamKind
from dynamix.model.origin import FIRST_PARTY

INTEGRATION_PARAMS = (
    Param("integration_order", ParamKind.FLOAT, default=0., min=-6., max=6., soft_min=0., soft_max=2.,
          label="Input integration s", keyed_at_default=False),
    Param("integration_operator", ParamKind.CHOICE, default="bessel", choices=("bessel", "riesz"),
          label="Integration operator", level="advanced", keyed_at_default=False),
    Param("integration_length", ParamKind.FLOAT, default=0., min=0., max=1e9, soft_max=1000.,
          label="Reference length (px; 0 = domain / 2π)", level="advanced", keyed_at_default=False),
)


def applied_order(field):
    history = (getattr(field, "provenance", None) or {}).get("fractional_integration", [])
    return sum(float(item["order"]) for item in history)


def integrated_field(field, params, *, edge, stage_cache=None):
    from dynamix.core.fractional import integrate

    order = float(params.get("integration_order", 0.))
    if order == 0:
        return field
    if not hasattr(field, "values"):
        raise ValueError("fractional integration runs on a field before analysis")
    length = float(params.get("integration_length", 0.)) or min(field.values.shape[:2])/(2*np.pi)
    operator = params.get("integration_operator", "bessel")
    from dynamix.core import fft_policy
    from dynamix.core.wtmm_backend import _field_data_hash
    from dynamix.engine.cache import cache_key

    policy = fft_policy.active()
    key = cache_key("fractional-field", _field_data_hash(field), {
        "order": order, "operator": operator, "length": length, "edge": edge,
        "fft": policy.name, "precision": policy.precision})
    if stage_cache is not None and key in stage_cache:
        values = stage_cache[key]
    else:
        values = integrate(field.values, order, operator=operator, length_px=length, edge=edge)
        if stage_cache is not None:
            stage_cache[key] = values
    provenance = dict(field.provenance)
    history = list(provenance.get("fractional_integration", []))
    history.append({"order": order, "operator": operator, "length_px": length, "edge": edge,
                    "dc": "preserved", "shift_convention": "h_original = h_integrated - order"})
    provenance["fractional_integration"] = history
    return dataclasses.replace(field, values=values, provenance=provenance, name=f"{field.name}+I{order:g}")


class FractionalIntegration:
    @staticmethod
    def provenance_outcome(result, params):
        from dynamix.engine.execution import resolved_outcome
        return resolved_outcome(result, params) if params["integration_order"] else {}

    name = "fractional_integration"
    label = "Fractional integration (Fourier)"
    origin = dataclasses.replace(FIRST_PARTY, citation="Bessel potential, Dumeur et al. (2025), Eq. 12; explicit reference length and mean-preserving Riesz option")
    families = ("Data preparation",)
    consumes, produces = ((FIELD,),), (RECEIVED,)
    capabilities = frozenset({"StripExtras"})
    uses_stage_cache = True
    params = INTEGRATION_PARAMS + (
        Param("edge", ParamKind.CHOICE, default="mirror", choices=("mirror", "periodic"), label="Boundary"),)

    def compute(self, field, params, *, progress=None, stage_cache=None):
        return integrated_field(field, params, edge=params["edge"], stage_cache=stage_cache)

    def derived_reading(self, name, value, field, params=None):
        if name == "integration_order":
            return "global Fourier integration; positive s smooths, negative s differentiates; mean preserved"
        return None

    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key
        from dynamix.model.device import keyed_params

        return cache_key(self.name, source_id, keyed_params(self, params))
