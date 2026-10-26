# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Disorder function: the disorder of a locally striped field.

The field-form disorder of Gunaratne, Hoffman and Kouri 1998 (PRE 57:5146, Eq. 2) — the mean
δ̄(β), the total |Ω|·δ̄(β) and the rooted mean δ̄(β)^{1/β} — or the envelope form of the 1995 PRL
(Eq. 8), each a curve against the moment order β, with the density as a raster
(:mod:`dynamix.core.disorder`). k₀ is found by the minimum of δ̄(1) (Gunaratne, Ratnaweera and
Tennekone 1999), by the spectral peak, or given as a wavelength; the envelope comes from the
Fourier half-plane (Jones and Gunaratne 1998) or from Passot and Newell's wavelet ridge
(:mod:`dynamix.core.ridge`), with the phase-field density by default; derivatives are the papers'
DAF by default; spectral derivatives need a periodized input, which a Periodic extension's stamp or
the ``periodic`` knob declares.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from dynamix.core import analytic_cwt, disorder, ridge
from dynamix.devices.measures_common import (DETREND_PARAM, FLOOR_PARAM, SCHEME_PARAMS,
                                             deriv_record, periodized, scheme)
from dynamix.model.contracts import CURVE, FIELD
from dynamix.model.param import Param, ParamKind

__all__ = ["DisorderFunction"]

#: The knobs of the wavelet ridge apply under the envelope form by the ridge.
_RIDGE = (("disorder_form", ("envelope",)), ("envelope", ("wavelet_ridge",)))


class DisorderFunction:
    """δ̄(β), δ(β) and δ_root(β) of a field, with D(x) and the envelope's rasters."""

    name = "disorder_function"
    label = "Disorder function"
    families = ("Measures",)
    role = "primary"                    # a drop on a layer forks a sibling over the same field
    consumes = ((FIELD,),)
    produces = frozenset({CURVE})
    capabilities = frozenset({"SelfReporting", "RegionAware", "Viewable"})
    params = (
        *SCHEME_PARAMS,
        Param("k0", ParamKind.CHOICE, default="min_disorder",
              choices=(*disorder.K0_METHODS, "manual"),
              choice_labels=("min δ̄(1)", "spectral peak", "given λ₀"), label="k₀"),
        Param("lambda0", ParamKind.FLOAT, default=20.0, min=2.0, max=1e6, soft_min=4.0, soft_max=200.0,
              units="px", label="λ₀ (px)", active_when=("k0", ("manual",))),
        Param("disorder_form", ParamKind.CHOICE, default="field", choices=("field", "envelope"),
              label="Disorder of"),
        Param("envelope", ParamKind.CHOICE, default="fft_halfplane",
              choices=("fft_halfplane", "wavelet_ridge"),
              choice_labels=("half-plane", "wavelet ridge"), label="Envelope",
              active_when=("disorder_form", ("envelope",))),
        # The ridge's density: the phase-field form needs no ±k partition (the default); the
        # PRL's operator on A e^{i(Ψ − k₀·x)} is the comparison, blind within a node of the fold.
        Param("density", ParamKind.CHOICE, default="phase_field",
              choices=("phase_field", "envelope_operator"),
              choice_labels=("phase field", "operator"), label="Density", active_when=_RIDGE),
        Param("omega", ParamKind.FLOAT, default=analytic_cwt.OMEGA, min=2.0, max=12.0, soft_min=2.0,
              soft_max=6.0, label="ω", active_when=_RIDGE),
        Param("anisotropy", ParamKind.FLOAT, default=1.0, min=0.25, max=4.0, soft_min=0.25, soft_max=4.0,
              label="α₂/α₁", active_when=_RIDGE),
        Param("ridge_q", ParamKind.FLOAT, default=1.0, min=-1.0, max=1.95, soft_min=0.5, soft_max=1.5,
              label="q", active_when=_RIDGE),
        Param("q_pairing", ParamKind.CHOICE, default="fixed", choices=("fixed", "q-paired"),
              label="q width", active_when=_RIDGE),
        Param("q_beta", ParamKind.FLOAT, default=0.5, min=0.05, max=5.0, soft_min=0.1, soft_max=2.0,
              label="β_q", active_when=(*_RIDGE, ("q_pairing", ("fixed",)))),
        Param("harmonic_filter", ParamKind.CHOICE, default="daf", choices=("daf", "bandpass", "none"),
              choice_labels=("DAF + band", "band-pass", "none"), label="Harmonic filter"),
        DETREND_PARAM,
        FLOOR_PARAM,
        # View-only: the rasters are computed and cached once; the pick is a cache hit. A raster
        # the chosen form did not make falls back to the form's own density.
        Param("show", ParamKind.CHOICE, default="field_disorder_density",
              choices=("field_disorder_density", "envelope_disorder_density", "envelope_amplitude",
                       "local_wavenumber"),
              choice_labels=("D(x)", "D^env(x)", "|A(x)|", "|K(x)|"), label="Show", view=True),
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
        return disorder.reach(**scheme(params))

    def compute(self, field, params: dict, *, progress=None) -> dict:
        values = np.asarray(field.values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("disorder_function: a single-band field")
        sch = scheme(params)
        periodized_by = periodized(field, params)
        periodic = periodized_by is not None
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
            k0 = disorder.k0_estimate(v, params["k0"], periodic=periodic, mask=stats,
                                      detrend_mode="none", **sch)
            k0_record = {"method": params["k0"], "value": k0, "lambda_px": 2.0 * math.pi / k0}

        if params["harmonic_filter"] == "bandpass":
            v = disorder.bandpass(v, k0, periodic=periodic)
            filter_record = {"method": "bandpass", "relative_width": disorder.BANDPASS_WIDTH}
        elif params["harmonic_filter"] == "none":
            filter_record = {"method": "none"}

        ny, nx = values.shape
        form = params["disorder_form"]
        kw = dict(periodic=periodic, **sch)
        rasters = {}
        warnings = []
        if form == "envelope" and params["envelope"] == "wavelet_ridge":
            kvec = disorder.k0_direction(v, k0, mask=stats)
            q = float(params["ridge_q"])
            if params["q_pairing"] == "q-paired":
                from dynamix.core.microcanonical import paired_q_beta

                q_beta = paired_q_beta(q)
            else:
                q_beta = float(params["q_beta"])
            r = ridge.extract_ridge(v, kvec, omega=float(params["omega"]),
                                    alpha=(1.0, float(params["anisotropy"])), q=q, q_beta=q_beta,
                                    periodic=periodic)
            good = stats & r.converged
            if params["density"] == "phase_field":
                density = disorder.phase_field_density(r.amplitude, r.kx, r.ky, r.div_k, k0,
                                                       mask=good, **kw)
            else:
                yy, xx = np.mgrid[0:v.shape[0], 0:v.shape[1]].astype(np.float64)
                a = r.amplitude * np.exp(1j * (r.phase - kvec[0] * xx - kvec[1] * yy))
                density = disorder.envelope_density(a, kvec, mask=good & ~r.boundary, **kw)
            density = density[:ny, :nx]
            rasters["envelope_amplitude"] = np.where(good, r.amplitude, np.nan)[:ny, :nx]
            rasters["local_wavenumber"] = np.where(good, np.hypot(r.kx, r.ky), np.nan)[:ny, :nx]
            envelope_record = {**r.record, "density": params["density"],
                               "direction_deg": math.degrees(math.atan2(kvec[1], kvec[0]))}
            if r.record["n_unconverged"]:
                warnings.append(f"the wavelet ridge did not converge at {r.record['n_unconverged']} "
                                f"pixels (defect cores, or a local wavenumber outside "
                                f"{ridge.BRACKET[0]:g}–{ridge.BRACKET[1]:g} k₀); they are left out")
        elif form == "envelope":
            kvec = disorder.k0_direction(v, k0, mask=stats)
            a = disorder.envelope_halfplane(v, kvec, periodic=periodic)
            density = disorder.envelope_density(a, kvec, mask=stats, **kw)[:ny, :nx]
            rasters["envelope_amplitude"] = np.where(stats, np.abs(a), np.nan)[:ny, :nx]
            envelope_record = {"method": "fft_halfplane", "k0_vector": [kvec[0], kvec[1]],
                               "direction_deg": math.degrees(math.atan2(kvec[1], kvec[0]))}
        else:
            density = disorder.helmholtz_density(v, k0, detrend_mode="none", mask=stats, **kw)[:ny, :nx]
            envelope_record = None
        delta = disorder.disorder_moments(density, disorder.DELTA_BETAS,
                                          floor=float(params["beta_floor"]) or None)
        names = {kind: f"{kind}_{form}_disorder" for kind in ("mean", "total", "rooted_mean")}
        rasters[f"{form}_disorder_density"] = density
        warnings += [delta["warning"]] if delta["warning"] else []
        record = {
            "deriv": deriv_record(sch, periodized_by), "k0": k0_record,
            "harmonic_filter": filter_record, "disorder_form": form, "envelope": envelope_record,
            "disorder_density": f"{form}_disorder_density", "detrend": params["detrend"],
            "disorder_normalization": {names["mean"]: "mean", names["total"]: "total",
                                       names["rooted_mean"]: "rooted_mean"},
            "beta_floor": {"value": delta["floor"], "n_floor": delta["n_floor"]},
            "valid_pixels": int(delta["area"]), "warnings": warnings,
        }
        result = {
            "beta_delta": np.asarray(disorder.DELTA_BETAS, dtype=np.float64),
            names["mean"]: delta["mean"],
            names["total"]: delta["total"],
            names["rooted_mean"]: delta["rooted"],
            **rasters,
            "k0": k0, "record": record, "warnings": warnings,
            "_shape": tuple(values.shape),
        }
        result["raster_out"] = self._shown(result, params["show"])
        return result

    def curves(self, result, field, params: dict) -> list:
        """For the Curve window: the mean, total and rooted mean disorder against β, log y."""
        from dynamix.model.curves import Curve

        if not isinstance(result, dict) or "k0" not in result:
            return []
        form = result["record"]["disorder_form"]
        beta = np.asarray(result["beta_delta"])
        note = (f"{form} form; λ₀ {2.0 * math.pi / float(result['k0']):.1f} px; "
                f"{result['record']['valid_pixels']} valid pixels")
        return [Curve(f"mean {form} disorder δ̄(β)", beta, np.asarray(result[f"mean_{form}_disorder"]),
                      "β", "δ̄(β)", logy=True, note=note),
                Curve(f"total {form} disorder δ(β)", beta, np.asarray(result[f"total_{form}_disorder"]),
                      "β", "δ(β) (px²)", logy=True, note=note),
                Curve(f"rooted mean {form} disorder", beta,
                      np.asarray(result[f"rooted_mean_{form}_disorder"]), "β", "δ̄(β)^{1/β}",
                      logy=True, note=note)]

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)

    def reading(self, result: dict, params: dict) -> str:
        if "k0" not in result:
            return "no measure — run the transform"
        k0 = float(result["k0"])
        how = {"manual": "given", "min_disorder": "min δ̄(1)", "spectral_peak": "spectral peak"}
        method = how.get(result.get("record", {}).get("k0", {}).get("method", ""), "")
        form = result["record"]["disorder_form"]
        d1 = float(np.asarray(result[f"mean_{form}_disorder"])[list(disorder.DELTA_BETAS).index(1.0)])
        return f"λ₀ {2.0 * math.pi / k0:.1f} px ({method}) · δ̄^{form[:3]}(1) {d1:.3g}"
