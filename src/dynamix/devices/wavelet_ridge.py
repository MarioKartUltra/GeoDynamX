# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Wavelet ridge (Passot–Newell): the amplitude and the local wavevector of a striped field.

Passot and Newell 1994 (Physica D 74:301, Appendix A) read a pattern's slowly varying amplitude
and its wavevector field off the ridge of a 2-D analytic wavelet transform
(:mod:`dynamix.core.ridge`, the q-Morlet of :mod:`dynamix.core.analytic_cwt`): at each pixel the
scale and orientation where the transform's phase is stationary. The ridge rasters are scale-free
— |A(x)|, |K(x)|, the roll direction, the ridge scale, the phase — and the transform the ridge is
read from is on the Scale slider: at the level shown, |T| maximised over the orientations, the
orientation that wins, and the phase there, one entry per level of the usual ladder, computed on
request. The planform wavenumber k₀ that centres the ridge's search comes from the spectral peak or
a given wavelength; a non-periodic field is reflected, or periodised first by the Periodic
extension, whose stamp this tool reads.
"""
from __future__ import annotations

import math

import numpy as np

from dynamix.core import analytic_cwt, disorder, ridge
from dynamix.model.contracts import COMPONENTS, FIELD
from dynamix.model.device import Output
from dynamix.model.param import Param, ParamKind

__all__ = ["WaveletRidge"]


def _e_q(u, q: float):
    """Tsallis's q-exponential e_q(u), the exponential at q = 1."""
    if q == 1.0:
        return np.exp(u)
    return np.clip(1.0 + (1.0 - q) * u, 0.0, None) ** (1.0 / (1.0 - q))


class WaveletRidge:
    """The ridge's rasters, and the directional transform under the Scale slider."""

    name = "wavelet_ridge"
    label = "Wavelet ridge (Passot–Newell)"
    families = ("Patterns",)
    role = "primary"                    # a drop on a layer forks a sibling over the same field
    consumes = ((FIELD,),)
    produces = frozenset({COMPONENTS})
    capabilities = frozenset({"SelfReporting", "RegionAware", "Viewable", "OutputProducer",
                              "StripExtras"})
    params = (
        Param("k0", ParamKind.CHOICE, default="spectral_peak", choices=("spectral_peak", "manual"),
              choice_labels=("spectral peak", "given λ₀"), label="k₀"),
        Param("lambda0", ParamKind.FLOAT, default=20.0, min=2.0, max=1e6, soft_min=4.0, soft_max=200.0,
              units="px", label="λ₀ (px)", active_when=("k0", ("manual",))),
        Param("omega", ParamKind.FLOAT, default=analytic_cwt.OMEGA, min=2.0, max=12.0, soft_min=2.0,
              soft_max=6.0, label="ω"),
        Param("anisotropy", ParamKind.FLOAT, default=1.0, min=0.25, max=4.0, soft_min=0.25, soft_max=4.0,
              label="α₂/α₁"),
        Param("ridge_q", ParamKind.FLOAT, default=1.0, min=-1.0, max=1.95, soft_min=0.5, soft_max=1.5,
              label="q"),
        Param("q_pairing", ParamKind.CHOICE, default="fixed", choices=("fixed", "q-paired"), label="q width"),
        Param("q_beta", ParamKind.FLOAT, default=0.5, min=0.05, max=5.0, soft_min=0.1, soft_max=2.0,
              label="β_q", active_when=("q_pairing", ("fixed",))),
        # A field that is periodic as it stands; a Periodic extension's stamp declares it by itself.
        Param("periodic", ParamKind.BOOL, default=False, label="Periodic input"),
        # The ladder the Scale slider walks for the transform's own rasters.
        Param("n_oct", ParamKind.INT, default=3, min=1, max=12, label="Octaves"),
        Param("n_voice", ParamKind.INT, default=4, min=1, max=64, label="Voices/octave"),
        Param("a_min", ParamKind.FLOAT, default=2.0, min=0.5, max=64.0, soft_min=1.0, soft_max=8.0,
              label="aₘᵢₙ (px)"),
        Param("orientations", ParamKind.INT, default=12, min=4, max=72, soft_min=8, soft_max=36,
              label="Orientations"),
        # View-only: the ridge rasters are in the result; the three per-level ones are lazy.
        Param("show", ParamKind.CHOICE, default="ridge_amplitude",
              choices=("ridge_amplitude", "ridge_wavenumber", "ridge_direction", "ridge_scale",
                       "ridge_phase", "modulus", "orientation", "phase"),
              choice_labels=("|A(x)|", "|K(x)|", "direction(x)", "a_r(x)", "phase(x)",
                             "|T| over orientations", "orientation of max |T|", "phase at max |T|"),
              label="Show", view=True),
    )
    outputs = (
        Output("modulus", "raster", lazy=True, label="|T| over orientations", per_level=True),
        Output("orientation", "raster", lazy=True, label="orientation of max |T|", per_level=True),
        Output("phase", "raster", lazy=True, label="phase at max |T|", per_level=True),
    )
    param_groups = {
        "RIDGE": ("k0", "lambda0", "omega", "anisotropy", "ridge_q", "q_pairing", "q_beta", "periodic"),
        "LADDER": ("a_min", "n_oct", "n_voice", "orientations"),
    }

    _RIDGE_RASTERS = ("ridge_amplitude", "ridge_wavenumber", "ridge_direction", "ridge_scale",
                      "ridge_phase")

    @staticmethod
    def _wavelet(params: dict) -> dict:
        """:func:`analytic_cwt.wavelet_hat`'s keywords from the knobs."""
        q = float(params["ridge_q"])
        if params["q_pairing"] == "q-paired":
            from dynamix.core.microcanonical import paired_q_beta

            q_beta = paired_q_beta(q)
        else:
            q_beta = float(params["q_beta"])
        return dict(omega=float(params["omega"]), alpha=(1.0, float(params["anisotropy"])), q=q,
                    q_beta=q_beta)

    @staticmethod
    def _periodic(field, params: dict) -> bool:
        provenance = getattr(field, "provenance", None) or {}
        return bool(params["periodic"]) or bool(provenance.get("edges.filled"))

    @staticmethod
    def _scales(params: dict) -> np.ndarray:
        """The ladder in pixels: a = a_min·2^{i/n_voice} over n_oct octaves, both ends in (the
        WTMM ladder's a is xsmurf's nominal unit, not the envelope's width)."""
        n_oct, n_voice = int(params["n_oct"]), int(params["n_voice"])
        return float(params["a_min"]) * 2.0 ** (np.arange(n_oct * n_voice + 1) / n_voice)

    def _shown(self, result: dict, show):
        return result[show] if show in self._RIDGE_RASTERS else result["ridge_amplitude"]

    def view(self, result: dict, params: dict) -> dict:
        """A ridge raster on ``show`` replaces the field; a per-level pick is the engine's lazy
        output, the amplitude standing in until it lands."""
        if "record" not in result:
            return result
        return {**result, "raster_out": self._shown(result, params.get("show")),
                "params": {**result.get("params", {}), "show": params.get("show")}}

    def roi_margin(self, params: dict) -> int:
        """What an ROI must read beyond its window: the coarsest wavelet's reach."""
        a_max = float(params["a_min"]) * 2.0 ** int(params["n_oct"])
        return int(math.ceil(4.0 * a_max / math.sqrt(min(1.0, float(params["anisotropy"])))))

    def compute(self, field, params: dict, *, progress=None) -> dict:
        values = np.asarray(field.values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("wavelet_ridge: a single-band field")
        periodic = self._periodic(field, params)
        v, valid = disorder.detrend(values, "mean")
        if params["k0"] == "manual":
            k0 = 2.0 * math.pi / float(params["lambda0"])
            k0_record = {"method": "manual", "value": k0, "lambda_px": float(params["lambda0"])}
        else:
            k0 = disorder.k0_estimate(v, "spectral_peak", mask=valid, detrend_mode="none")
            k0_record = {"method": "spectral_peak", "value": k0, "lambda_px": 2.0 * math.pi / k0}
        kvec = disorder.k0_direction(v, k0, mask=valid)
        if progress is not None:
            progress("wavelet ridge", 0.0)
        r = ridge.extract_ridge(v, kvec, periodic=periodic, **self._wavelet(params))
        if progress is not None:
            progress("wavelet ridge", 1.0)
        good = valid & r.converged

        def masked(a):
            return np.where(good, a, np.nan)

        direction = np.degrees(np.arctan2(r.ky, r.kx)) % 180.0       # a director: 0 ≤ ϑ < 180°
        warnings = []
        if r.record["n_unconverged"]:
            warnings.append(f"the ridge did not converge at {r.record['n_unconverged']} pixels "
                            f"(defect cores, or a local wavenumber outside "
                            f"{ridge.BRACKET[0]:g}–{ridge.BRACKET[1]:g} k₀); they are blank")
        scales = self._scales(params)
        record = {**r.record, "k0": k0_record,
                  "direction_deg": math.degrees(math.atan2(kvec[1], kvec[0])),
                  "ladder": {"n_oct": int(params["n_oct"]), "n_voice": int(params["n_voice"]),
                             "a_min": float(params["a_min"]), "n_scales": int(scales.size),
                             "orientations": int(params["orientations"])},
                  "warnings": warnings}
        result = {
            "ridge_amplitude": masked(r.amplitude), "ridge_wavenumber": masked(np.hypot(r.kx, r.ky)),
            "ridge_direction": masked(direction), "ridge_scale": masked(r.scale),
            "ridge_phase": masked(r.phase), "converged": good,
            "scales": scales, "k0": k0, "record": record, "warnings": warnings,
            "_shape": tuple(values.shape),
        }
        result["raster_out"] = self._shown(result, params["show"])
        return result

    def compute_output(self, name: str, values, result: dict, params: dict, *, fetch,
                       progress=None, cancel=None, level=None) -> dict:
        """The directional transform at the level shown: over ``orientations`` wavelets centred
        on |κ| = ω/a around the half-plane, |T| at its maximum, the orientation that wins and the
        phase there."""
        if name not in ("modulus", "orientation", "phase"):
            raise ValueError(f"{self.name}: no lazy output named {name!r}")
        scales = np.asarray(result["scales"], dtype=np.float64)
        idx = int(np.clip(0 if level is None else int(level), 0, scales.size - 1))
        a = float(scales[idx])
        wavelet = self._wavelet(params)
        periodic = bool(params["periodic"]) or bool(result.get("record", {}).get("periodized"))
        v, _valid = disorder.detrend(np.asarray(values, dtype=np.float64), "mean")
        n_o = int(params["orientations"])
        best = np.full(v.shape, -1.0)
        theta = np.zeros(v.shape)
        phase = np.zeros(v.shape)
        if progress is not None:
            progress(f"{name} level {idx + 1}/{scales.size}", 0.0)
        for j in range(n_o):
            th = math.pi * j / n_o
            kappa = (wavelet["omega"] / a * math.cos(th), wavelet["omega"] / a * math.sin(th))
            t = analytic_cwt.transform(v, kappa, periodic=periodic, **wavelet)
            mod = np.abs(t)
            stronger = mod > best
            best[stronger] = mod[stronger]
            theta[stronger] = math.degrees(th)
            phase[stronger] = np.angle(t)[stronger]
        if progress is not None:
            progress(f"{name} level {idx + 1}/{scales.size}", 1.0)
        raster = {"modulus": best, "orientation": theta, "phase": phase}[name]
        return {"raster": raster.astype(np.float32),
                "diag": {"level": idx, "scale_px": a, "lambda_px": 2.0 * math.pi * a / wavelet["omega"]}}

    def wavelet_profile(self, params: dict, scale_px: float, level: int):
        """The q-Morlet's section along its carrier at the scale shown, and λ* = 2πa/ω, the
        carrier's wavelength: the one length the scale bar labels."""
        from dynamix.core.wavelet_profiles import Profile

        w = self._wavelet(params)
        a = float(scale_px)
        span = 4.0 * a / math.sqrt(2.0 * w["q_beta"])
        x = np.linspace(-span, span, 513)
        u = x / a
        psi = _e_q(-w["q_beta"] * u * u, w["q"]) * np.cos(w["omega"] * u)
        return Profile(x, psi / np.max(np.abs(psi)), a, 2.0 * math.pi * a / w["omega"],
                       f"q-Morlet (ω = {w['omega']:g}, q = {w['q']:g})",
                       extra={"a": a, "omega": w["omega"], "q": w["q"], "q_beta": w["q_beta"]})

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k
        from dynamix.model.device import keyed_params

        return _k(self.name, source_id, keyed_params(self, params))

    def reading(self, result: dict, params: dict) -> str:
        if "k0" not in result:
            return "no ridge — run the transform"
        amp = np.asarray(result["ridge_amplitude"])
        good = np.isfinite(amp)
        median = float(np.median(amp[good])) if good.any() else float("nan")
        return (f"λ₀ {2.0 * math.pi / float(result['k0']):.1f} px · |A| median {median:.3g} · "
                f"converged {good.mean():.0%}")
