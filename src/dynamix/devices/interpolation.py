# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Independent interpolation, smoothing and interpolating-wavelet tools."""
from __future__ import annotations

import dataclasses

import numpy as np

from dynamix.model.contracts import COMPONENTS, FIELD, RECEIVED, STACK
from dynamix.model.device import Output, keyed_params
from dynamix.model.param import Param, ParamKind
from dynamix.model.origin import FIRST_PARTY


def _values(field, *, stack=False):
    """The field's finite values: 2-D, or ``(ny, nx, nc)`` too when ``stack`` (a tool that
    works band by band)."""
    if isinstance(field, dict) or not hasattr(field, "values"):
        raise ValueError("this preparation tool needs a field before the analysing transform")
    x = np.asarray(field.values, dtype=np.float64)
    if x.ndim not in ((2, 3) if stack else (2,)) or not np.isfinite(x).all():
        raise ValueError(("a finite field or band stack" if stack else "a finite single-band field")
                         + " is required; fill nodata explicitly first")
    return x


def _per_band(x, fn):
    """``fn`` over a 2-D array, or over each band of ``(ny, nx, nc)`` with the results stacked."""
    if x.ndim == 2:
        return fn(x)
    return np.stack([fn(x[..., b]) for b in range(x.shape[-1])], axis=-1)


def _clone(field, values, method, record):
    provenance = dict(field.provenance)
    history = list(provenance.get("preprocessing", []))
    history.append({"method": method, **record})
    provenance["preprocessing"] = history
    return dataclasses.replace(field, values=values, name=f"{field.name}+{method}", provenance=provenance)


class _Key:
    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key

        return cache_key(self.name, source_id, keyed_params(self, params))


class DyadicInterpolation(_Key):
    provenance_precision = "float64"

    @staticmethod
    def provenance_outcome(result, params):
        from dynamix.engine.execution import resolved_outcome
        return resolved_outcome(result, params)

    name = "dyadic_interpolation"
    label = "Dyadic interpolation (DD4)"
    origin = dataclasses.replace(FIRST_PARTY, citation="Deslauriers & Dubuc, Interpolation dyadique, EPM/RT-87/22 (1987), classical four-point scheme")
    families = ("Data preparation",)
    consumes, produces = ((FIELD,), (STACK,)), (RECEIVED,)
    capabilities = frozenset({"StripExtras"})
    params = (
        Param("levels", ParamKind.INT, default=1, min=1, max=8, soft_max=3, label="Refinements"),
        Param("edge", ParamKind.CHOICE, default="reflect", choices=("reflect", "periodic"), label="Boundary"),
    )

    def compute(self, field, params, *, progress=None):
        from dynamix.core.interpolating import refine

        x = _values(field, stack=True)
        if field.provenance.get("edges.filled"):
            raise ValueError("interpolate the native field before Periodic extension")
        axes = [np.asarray(field.x_axis, dtype=float), np.asarray(field.y_axis, dtype=float)]
        for a in axes:
            if a.size < 2 or not np.allclose(np.diff(a), a[1] - a[0], rtol=1e-8, atol=1e-12) or a[1] == a[0]:
                raise ValueError("DD interpolation requires uniformly spaced source axes")
        y = _per_band(x, lambda band: refine(band, params["levels"], edge=params["edge"]))
        factor = 2 ** int(params["levels"])
        record = {**params, "factor": factor, "source_shape": list(x.shape[:2]),
                  "source_spacing": [float(a[1] - a[0]) for a in axes],
                  "retained_input_nodes": "row and column indices divisible by factor",
                  "derived_sampling": True}
        out = _clone(field, y, self.name, record)
        out.provenance["grid.requires_fork"] = True
        return dataclasses.replace(out, x_axis=np.linspace(axes[0][0], axes[0][-1], y.shape[1]),
                                   y_axis=np.linspace(axes[1][0], axes[1][-1], y.shape[0]))

    def derived_reading(self, name, value, field, params=None):
        if name != "levels" or getattr(field, "values", None) is None:
            return None
        factor = 2 ** int(value)
        ny, nx = field.provenance.get("_reading_shape", field.values.shape[:2])
        return (f"{(ny-1)*factor+1} × {(nx-1)*factor+1} · input {ny} × {nx}; "
                f"{int(value)} refinement(s), spacing ÷{factor}; endpoints and original nodes retained; "
                "fork derivative dataset for analysis")


class DAFSmoothing(_Key):
    provenance_precision = "float64"

    @staticmethod
    def provenance_outcome(result, params):
        from dynamix.engine.execution import resolved_outcome
        return resolved_outcome(result, params)

    name = "daf_smoothing"
    label = "DAF low-pass"
    origin = dataclasses.replace(FIRST_PARTY, citation="Hermite DAF gain; Hoffman et al. (2000), Eq. 5; US6847737")
    families = ("Data preparation",)
    consumes, produces = ((FIELD,), (STACK,)), (RECEIVED,)
    capabilities = frozenset({"FieldStage", "RegionAware", "StripExtras"})
    field_stage = True
    params = (
        Param("sigma", ParamKind.FLOAT, default=3., min=.25, max=100., soft_min=1., soft_max=10.,
              units="px", label="σ (px)"),
        Param("order", ParamKind.INT, default=12, min=0, max=40, label="Hermite order M (even)"),
        Param("edge", ParamKind.CHOICE, default="reflect", choices=("reflect", "periodic"), label="Boundary"),
    )

    def check(self, params):
        if params["order"] % 2:
            raise ValueError("Hermite DAF order M must be even")

    def roi_margin(self, params):
        from dynamix.core.disorder import reach

        return reach("daf", M=params["order"], sigma_over_dx=params["sigma"])

    def compute(self, field, params, *, progress=None):
        from dynamix.core.disorder import filtered

        x = _values(field, stack=True)
        if params["order"] % 2:
            raise ValueError("Hermite DAF order M must be even")
        y = _per_band(x, lambda band: filtered(band, [lambda kx, ky: 1.], deriv="daf", M=params["order"],
                                               sigma_over_dx=params["sigma"], periodic=params["edge"] == "periodic")[0])
        return _clone(field, y, self.name, {**params, "residual_rms": float(np.sqrt(np.mean((x-y)**2)))})

    def curves(self, result, field, params):
        from dynamix.core.daf import daf_gain
        from dynamix.model.curves import Curve

        f = np.linspace(0, .5, 501)
        return [Curve("DAF gain", f, daf_gain(2*np.pi*f, params["sigma"], params["order"]),
                      "frequency (cycles/input pixel)", "gain", note="Explicit Hermite low-pass; no automatic tuning")]


WAVELET_PARAMS = (
    Param("bank", ParamKind.CHOICE, default="dd4", choices=("dd4", "gldaf"),
          choice_labels=("Deslauriers–Dubuc 4-point", "Gaussian–Lagrange DAF"), label="Predictor"),
    Param("order", ParamKind.INT, default=3, min=1, max=15, label="Shell order M (odd)", active_when=("bank", ("gldaf",))),
    Param("sigma", ParamKind.FLOAT, default=1., min=.25, max=5., soft_min=.5, soft_max=2.,
          label="Gaussian σ (coarse cells)", active_when=("bank", ("gldaf",))),
    Param("levels", ParamKind.INT, default=3, min=1, max=10, soft_max=6, label="Levels"),
    Param("edge", ParamKind.CHOICE, default="reflect", choices=("reflect", "periodic"), label="Boundary"),
)


def _coef(field, params, stage_cache):
    from dynamix.core.interpolating import pyramid
    from dynamix.core.wtmm_backend import _field_data_hash
    from dynamix.engine.cache import cache_key

    x = _values(field)
    settings = {k: params[k] for k in ("bank", "order", "sigma", "edge")}
    key = cache_key("interpolating-pyramid", _field_data_hash(field), {**settings, "levels": params["levels"]})
    if stage_cache is not None and key in stage_cache:
        return stage_cache[key]
    c = pyramid(x, params["levels"], **settings)
    if stage_cache is not None:
        stage_cache[key] = c
    return c


class InterpolatingWavelets(_Key):
    provenance_precision = "float64"

    @staticmethod
    def provenance_outcome(result, params):
        from dynamix.engine.execution import resolved_outcome
        return resolved_outcome(result, params)

    name = "interpolating_wavelets"
    label = "Interpolating wavelets (DD / DAF)"
    origin = dataclasses.replace(FIRST_PARTY, citation="DD predictor; US6847737 Eqs. 47–48 GLDAF shell; local half-transpose lifting update and DC normalization")
    families = ("Decomposition",)
    consumes, produces = ((FIELD,),), (COMPONENTS,)
    capabilities = frozenset({"OutputProducer", "Viewable", "StripExtras"})
    uses_stage_cache = True
    params = WAVELET_PARAMS + (Param("show", ParamKind.CHOICE, default="low",
                                    choices=("low", "horizontal", "vertical", "diagonal"), view=True, label="Show"),)
    outputs = tuple(Output(n, "raster", lazy=True, label=n, per_level=n != "low")
                    for n in ("low", "horizontal", "vertical", "diagonal"))
    param_groups = {"PYRAMID": ("bank", "order", "sigma", "levels", "edge")}

    def compute(self, field, params, *, progress=None, stage_cache=None):
        from dynamix.core.interpolating import inverse
        from dynamix.devices.decompose import _result

        c = _coef(field, params, stage_cache)
        y = inverse(c, keep=(params["levels"], "low"))
        return _result(field, y.astype(np.float32), {"_interpolating": c,
                       "scales": 2. ** np.arange(1, params["levels"]+1),
                       "record": {"method": self.name, "lifting_update": "half predictor transpose",
                                  "midpoint_dc_normalized": True, "orthogonal": False}}, params)

    def compute_output(self, name, values, result, params, *, fetch, progress=None, cancel=None, level=None):
        from dynamix.core.interpolating import inverse

        if name not in ("low", "horizontal", "vertical", "diagonal"):
            raise ValueError(f"unknown interpolating band {name}")
        j = params["levels"] if name == "low" else int(np.clip(0 if level is None else level, 0, params["levels"]-1))+1
        return {"raster": inverse(result["_interpolating"], keep=(j, name)).astype(np.float32)}

    def view(self, result, params):
        return result


class InterpolatingWaveletFilter(_Key):
    provenance_precision = "float64"

    @staticmethod
    def provenance_outcome(result, params):
        from dynamix.engine.execution import resolved_outcome
        return resolved_outcome(result, params)

    name = "interpolating_wavelet_filter"
    label = "Interpolating wavelet filter (DD / DAF)"
    origin = InterpolatingWavelets.origin
    families = ("Data preparation",)
    consumes, produces = ((FIELD,),), (RECEIVED,)
    capabilities = frozenset({"StripExtras"})
    uses_stage_cache = True
    params = WAVELET_PARAMS + (Param("threshold", ParamKind.FLOAT, default=0., min=0., max=1e9,
                                    soft_max=1., label="Soft threshold (field units)"),)
    inspector_actions = {"threshold": "Reconstruct cached interpolating coefficients"}
    params = tuple(dataclasses.replace(p, level="advanced") if p.name == "threshold" else p for p in params)
    param_groups = {"PYRAMID": ("bank", "order", "sigma", "levels", "edge"), "RESTORATION": ("threshold",)}

    def compute(self, field, params, *, progress=None, stage_cache=None):
        from dynamix.core.interpolating import inverse

        c = _coef(field, params, stage_cache)
        y = inverse(c, threshold=params["threshold"])
        return _clone(field, y, self.name, {**params, "orthogonal": False,
                      "lifting_update": "half predictor transpose", "midpoint_dc_normalized": True,
                      "residual_rms": float(np.sqrt(np.mean((field.values-y)**2)))})
