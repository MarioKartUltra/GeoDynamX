# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Invariant measures (Gunaratne): the disorder function and the Hessian measure of one field.

The field-form disorder of Gunaratne, Hoffman and Kouri 1998 (PRE 57:5146, Eq. 2) — the mean
δ̄(β), the total |Ω|·δ̄(β) and the rooted mean δ̄(β)^{1/β} — and the Hessian measure μ(β) of Nathan
and Gunaratne 2005 (PRE 71:035101, Eq. 1), each a curve against the moment order β, with the two
local densities as rasters (:mod:`dynamix.core.disorder`). k₀ is found by the minimum of δ̄(1)
(Gunaratne, Ratnaweera and Tennekone 1999), by the spectral peak, or given as a wavelength;
derivatives are the papers' DAF by default; spectral derivatives need a periodized input, which a
Periodic extension's stamp or the ``periodic`` knob declares.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from dynamix.core import disorder
from dynamix.model.contracts import CURVE, FIELD
from dynamix.model.param import Param, ParamKind

__all__ = ["InvariantMeasures"]


class InvariantMeasures:
    """δ̄(β), δ(β), δ_root(β) and μ(β) of a field, with D(x) and |Λ(x)| as rasters."""

    name = "invariant_measures"
    label = "Invariant measures (Gunaratne)"
    families = ("Microstructure",)
    role = "primary"                    # a drop on a layer forks a sibling over the same field
    consumes = ((FIELD,),)
    produces = frozenset({CURVE})
    capabilities = frozenset({"SelfReporting", "RegionAware", "Viewable"})
    params = (
        Param("deriv", ParamKind.CHOICE, default="daf", choices=disorder.DERIVS,
              choice_labels=("DAF (M = 12, σ/Δ = 4)", "Gaussian (σ_d)", "spectral (periodic input)"),
              label="Derivatives"),
        Param("sigma_d", ParamKind.FLOAT, default=1.0, min=0.3, max=16.0, soft_min=0.5, soft_max=4.0,
              units="px", label="σ_d (px)", active_when=("deriv", ("gaussian",))),
        # A field that is periodic as it stands (model output on a periodic domain): no padding,
        # and spectral derivatives allowed. A Periodic extension's stamp declares it by itself.
        Param("periodic", ParamKind.BOOL, default=False, label="Periodic input"),
        Param("k0", ParamKind.CHOICE, default="min_disorder",
              choices=(*disorder.K0_METHODS, "manual"),
              choice_labels=("minimum of δ̄(1)", "spectral peak", "given λ₀"), label="k₀"),
        Param("lambda0", ParamKind.FLOAT, default=20.0, min=2.0, max=1e6, soft_min=4.0, soft_max=200.0,
              units="px", label="λ₀ (px)", active_when=("k0", ("manual",))),
        Param("disorder_form", ParamKind.CHOICE, default="field", choices=("field", "envelope"),
              choice_labels=("field (PRE 57 Eq. 2)", "envelope (PRL 1995 Eq. 8)"), label="Disorder of"),
        Param("harmonic_filter", ParamKind.CHOICE, default="daf", choices=("daf", "bandpass", "none"),
              choice_labels=("DAF extension + band (Hoffman 2000)", "Gaussian band-pass at k₀", "none"),
              label="Harmonic filter"),
        Param("detrend", ParamKind.CHOICE, default="mean", choices=("mean", "plane", "none"),
              label="Detrend"),
        Param("hessian_norm", ParamKind.CHOICE, default="variance", choices=("variance", "std"),
              choice_labels=("variance (PRE 2005 Eq. 1)", "std (diagnostic)"), label="Hessian norm"),
        # The density floor for negative moment orders; 0 takes the core's default fraction.
        Param("beta_floor", ParamKind.FLOAT, default=0.0, min=0.0, max=1.0, soft_min=0.0,
              soft_max=1e-3, label="β < 0 floor (0 = auto)", level="advanced"),
        # View-only: the rasters are computed and cached once; the pick is a cache hit. A raster
        # the chosen form did not make falls back to the form's own density.
        Param("show", ParamKind.CHOICE, default="field_disorder_density",
              choices=("field_disorder_density", "envelope_disorder_density", "envelope_amplitude",
                       "hessian_density"),
              choice_labels=("disorder density D(x)", "envelope disorder density", "envelope |A(x)|",
                             "Hessian density |Λ(x)|"), label="Show", view=True),
    )

    @staticmethod
    def _shown(result: dict, show):
        return result[show] if show in result else result[result["record"]["disorder_density"]]

    def view(self, result: dict, params: dict) -> dict:
        """``show`` picks which raster replaces the field on the canvas, from the cached result."""
        if "record" not in result:
            return result
        return {**result, "raster_out": self._shown(result, params.get("show")),
                "params": {**result.get("params", {}), "show": params.get("show")}}

    def roi_margin(self, params: dict) -> int:
        """The derivative kernel's reach: what an ROI must read beyond its window."""
        return disorder.reach(params["deriv"], float(params["sigma_d"]))

    def compute(self, field, params: dict, *, progress=None) -> dict:
        values = np.asarray(field.values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("invariant_measures: a single-band field")
        provenance = getattr(field, "provenance", None) or {}
        periodized_by = ("declared periodic" if params["periodic"]
                         else "edges.filled" if provenance.get("edges.filled") else None)
        periodic = periodized_by is not None
        deriv, sigma_d = params["deriv"], float(params["sigma_d"])
        v, valid = disorder.detrend(values, params["detrend"])
        # The Chaos filter first: it may extend the field, so the statistics' mask grows with it
        # (the filled gap stays out) and the derivatives run on the periodic extension.
        stats = valid
        if params["harmonic_filter"] == "daf":
            k_t = disorder.k0_estimate(v, "spectral_peak", mask=valid, detrend_mode="none")
            v, block, tuned = disorder.daf_filter(v, k_t, periodic=periodic)
            if tuned is not None and tuned.fallback is None:
                periodic, periodized_by = True, "daf extension"
                stats = np.zeros(v.shape, dtype=bool)
                stats[:block[0], :block[1]] = valid
            filter_record = {"method": "daf", "band": list(disorder.GOLDMAN_BAND), "k_t": k_t,
                             "extension": None if tuned is None else dataclasses.asdict(tuned)}

        if params["k0"] == "manual":
            k0 = 2.0 * math.pi / float(params["lambda0"])
            k0_record = {"method": "manual", "value": k0, "lambda_px": float(params["lambda0"])}
        else:
            k0 = disorder.k0_estimate(v, params["k0"], deriv=deriv, sigma_d=sigma_d,
                                      periodic=periodic, mask=stats, detrend_mode="none")
            k0_record = {"method": params["k0"], "value": k0, "lambda_px": 2.0 * math.pi / k0}

        if params["harmonic_filter"] == "bandpass":
            v = disorder.bandpass(v, k0, periodic=periodic)
            filter_record = {"method": "bandpass", "relative_width": disorder.BANDPASS_WIDTH}
        elif params["harmonic_filter"] == "none":
            filter_record = {"method": "none"}

        ny, nx = values.shape
        form = params["disorder_form"]
        kw = dict(deriv=deriv, sigma_d=sigma_d, periodic=periodic, mask=stats)
        rasters = {}
        if form == "envelope":
            kvec = disorder.k0_direction(v, k0, mask=stats)
            a = disorder.envelope_halfplane(v, kvec, periodic=periodic)
            density = disorder.envelope_density(a, kvec, **kw)[:ny, :nx]
            rasters["envelope_amplitude"] = np.where(stats, np.abs(a), np.nan)[:ny, :nx]
            envelope_record = {"method": "fft_halfplane", "k0_vector": [kvec[0], kvec[1]],
                               "direction_deg": math.degrees(math.atan2(kvec[1], kvec[0]))}
        else:
            density = disorder.helmholtz_density(v, k0, detrend_mode="none", **kw)[:ny, :nx]
            envelope_record = None
        hessian = disorder.hessian_density(v, norm=params["hessian_norm"], detrend_mode="none",
                                           **kw)[:ny, :nx]
        floor = float(params["beta_floor"]) or None
        delta = disorder.disorder_moments(density, disorder.DELTA_BETAS, floor=floor)
        mu = disorder.hessian_measure(hessian, disorder.MU_BETAS, floor=floor)
        names = {kind: f"{kind}_{form}_disorder" for kind in ("mean", "total", "rooted_mean")}
        rasters.update({f"{form}_disorder_density": density, "hessian_density": hessian})
        deriv_record = {"scheme": deriv, "reach_px": disorder.reach(deriv, sigma_d)}
        if deriv == "daf":
            deriv_record.update(disorder.DAF_DEFAULTS)
        elif deriv == "gaussian":
            deriv_record["sigma_d"] = sigma_d
        if periodic:
            deriv_record["periodized_by"] = periodized_by
        warnings = [w for w in (delta["warning"], mu["warning"]) if w]
        record = {
            "deriv": deriv_record, "k0": k0_record, "harmonic_filter": filter_record,
            "disorder_form": form, "envelope": envelope_record,
            "disorder_density": f"{form}_disorder_density",
            "detrend": params["detrend"], "hessian_norm": params["hessian_norm"],
            "disorder_normalization": {names["mean"]: "mean", names["total"]: "total",
                                       names["rooted_mean"]: "rooted_mean"},
            "beta_floor": {"value": delta["floor"], "n_floor_disorder": delta["n_floor"],
                           "n_floor_hessian": mu["n_floor"]},
            "valid_pixels": int(delta["area"]), "warnings": warnings,
        }
        result = {
            "beta_delta": np.asarray(disorder.DELTA_BETAS, dtype=np.float64),
            names["mean"]: delta["mean"],
            names["total"]: delta["total"],
            names["rooted_mean"]: delta["rooted"],
            "beta_mu": np.asarray(disorder.MU_BETAS, dtype=np.float64),
            "hessian_measure": mu["mu"],
            **rasters,
            "k0": k0, "record": record, "warnings": warnings,
            "_shape": tuple(values.shape),
        }
        result["raster_out"] = self._shown(result, params["show"])
        return result

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)

    def reading(self, result: dict, params: dict) -> str:
        if "k0" not in result:
            return "no measures — run the transform"
        k0 = float(result["k0"])
        how = {"manual": "given", "min_disorder": "min δ̄(1)", "spectral_peak": "spectral peak"}
        method = how.get(result.get("record", {}).get("k0", {}).get("method", ""), "")
        form = result["record"]["disorder_form"]
        d1 = float(np.asarray(result[f"mean_{form}_disorder"])[list(disorder.DELTA_BETAS).index(1.0)])
        m1 = float(np.asarray(result["hessian_measure"])[list(disorder.MU_BETAS).index(1.0)])
        return (f"λ₀ {2.0 * math.pi / k0:.1f} px ({method}) · δ̄^{form[:3]}(1) {d1:.3g} · "
                f"μ(1) {m1:.3g} px⁻¹")
