# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Periodic extension (DAF), spec 25 §3.1: a non-periodic field made periodic, so a scale-space tool
after it can transform it with periodic boundaries and no padding.

The field is extended by a gap after its last row and column, filled by the self-tuned Hermite DAF
(:mod:`dynamix.core.daf`; Frishman, Hoffman, Rakauskas and Kouri 1996; Hoffman, Gunaratne, Zhang and
Kouri 2000), so that the whole is periodic. The grid keeps the field's pixel size and origin, so the
extended raster reaches past the original bounds by the gap. The output carries the stamp
``edges.filled = {"rows": H, "cols": W}``, the real block, which WTMM reads: no maxima in the gap, the
threshold from real pixels, and its Edge ``auto`` resolving to ``periodic``. Under the ``centred``
layout, the default, the same field is rolled so the data sit in the middle with the gap split
around them, and the stamp adds the block's offset, ``row0`` and ``col0``; the engine stamps a
result computed on the output with that frame (``_stage``), so the shell draws its overlays and
rasters back on the data. The band limit is read off the data, where a fraction of the power
lies, or given as a wavelength λ_b; the gap is two wavelengths and σ follows from the DAF's gain
there. Fewer than 6 points per wavelength at the limit makes a rough fill, which the strip's ⚠
says while the extension runs (WTMM's automatic edge handling reflects instead at that point);
only a stencil that cannot fit the field with its gap passes the field through unchanged.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from dynamix.model.contracts import FIELD, RECEIVED
from dynamix.model.param import Param, ParamKind

__all__ = ["FILLED", "PeriodicExtension"]

#: The provenance stamp naming a Periodic extension output's real block.
FILLED = "edges.filled"


def _extended_axis(axis, n: int, lead: int = 0) -> np.ndarray:
    """The axis continued by ``n`` samples, ``lead`` of them before the data's first sample."""
    axis = np.asarray(axis, dtype=np.float64)
    step = float(axis[1] - axis[0]) if axis.size > 1 else 1.0
    return axis[0] + step * (np.arange(axis.size + int(n), dtype=np.float64) - int(lead))


class PeriodicExtension:
    """The field plus a DAF-filled gap that makes it periodic, its real block stamped."""

    name = "periodic_extension"
    label = "Periodic extension (DAF)"
    families = ("Data preparation",)
    #: Not a field stage: over an ROI's processing window the gap would change the window's size,
    #: so the tool works on whole rasters (fork the output to analyse it on its own).
    capabilities = frozenset({"StripExtras"})
    #: A single-band field: the DAF fills one band's gap.
    consumes = ((FIELD,),)
    #: Input-preserving: an extended field stays a field.
    produces = (RECEIVED,)

    #: The DAF's knobs sit on the tool's box in the rack.
    params = (
        Param("band", ParamKind.CHOICE, default="auto", choices=("auto", "given"),
              choice_labels=("fraction of the power", "given λ_b"), label="Band limit"),
        Param("band_frac", ParamKind.FLOAT, default=0.99, min=0.5, max=0.9999, soft_min=0.9,
              soft_max=0.999, label="Power fraction", active_when=("band", ("auto",))),
        Param("lambda_b", ParamKind.FLOAT, default=20.0, min=4.0, max=1e5, soft_min=6.0,
              soft_max=200.0, units="px", label="λ_b (px)", active_when=("band", ("given",))),
        Param("M", ParamKind.INT, default=12, min=2, max=40, soft_min=4, soft_max=24,
              label="DAF order M"),
        Param("lam", ParamKind.FLOAT, default=1e-3, min=0.0, max=1.0, soft_min=0.0, soft_max=0.01,
              label="Regularization λ"),
        # Where the data sit in the extended raster: at the origin with the gap after the last row
        # and column, or in the middle with the gap split around them -- the same periodic field,
        # rolled, the stamp carrying the block's offset.
        Param("layout", ParamKind.CHOICE, default="centred", choices=("origin", "centred"),
              choice_labels=("data at the origin", "centred"), label="Layout",
              keyed_at_default=False),
    )

    @staticmethod
    def _tuned(values, params: dict):
        from dynamix.core.daf import self_tune

        given = 1.0 / float(params["lambda_b"]) if params.get("band") == "given" else None
        return self_tune(values, frac=params["band_frac"], M=params["M"], lam=params["lam"],
                         f_b=given)

    def compute(self, field, params: dict, *, progress=None):
        if isinstance(field, dict):
            raise ValueError("Periodic extension runs on the FIELD, before the analysing transform — "
                             "drag it ahead of the scale-space tool in the chain")
        from dynamix.core.daf import extend_2d

        values = np.asarray(field.values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Periodic extension takes a single-band field")
        tuned = self._tuned(values, params)
        if not tuned.fits:
            return field                             # nothing to compute; the ⚠ says why
        if progress is not None:
            progress("periodic extension", 0.0)
        extended = extend_2d(values, tuned)
        if progress is not None:
            progress("periodic extension", 1.0)
        H, W = values.shape
        gy, gx = extended.shape[0] - H, extended.shape[1] - W
        # One gap per axis, after the data: on the torus it bridges both edges of that axis.
        # Centred, the same field is rolled so the data sit in the middle, the gap around them.
        layout = params.get("layout", "origin")
        r0, c0 = (gy // 2, gx // 2) if layout == "centred" else (0, 0)
        if r0 or c0:
            extended = np.roll(np.roll(extended, r0, axis=0), c0, axis=1)
        provenance = dict(field.provenance)
        provenance[FILLED] = {"rows": int(H), "cols": int(W)}
        if r0 or c0:
            provenance[FILLED].update(row0=int(r0), col0=int(c0))
        provenance["periodic_extension"] = {"f_b": float(tuned.f_b), "gap": int(tuned.gap),
                                            "M": int(tuned.M), "sigma": float(tuned.sigma),
                                            "lam": float(tuned.lam), "layout": layout,
                                            "band": params.get("band", "auto"),
                                            "points_per_wavelength": float(tuned.ppw),
                                            "band_limited": bool(tuned.band_limited)}
        return dataclasses.replace(field, values=extended, name=f"{field.name}+daf{tuned.gap}",
                                   x_axis=_extended_axis(field.x_axis, gx, c0),
                                   y_axis=_extended_axis(field.y_axis, gy, r0),
                                   provenance=provenance)

    def knob_warnings(self, field, params: dict) -> dict:
        """The ⚠ beside the band-limit knob: a rough band limit (the extension runs), or a stencil
        that does not fit (the field passes through)."""
        values = getattr(field, "values", None)
        if values is None or np.asarray(values).ndim != 2:
            return {}
        from dynamix.model.device import validate_params

        tuned = self._tuned(np.asarray(values, dtype=np.float64), validate_params(self, params or {}))
        if tuned.fits and tuned.band_limited:
            return {}
        reason = tuned.fallback.replace("; reflection used", "")
        if tuned.fits:
            reason += "; the fill is rough (lower the power fraction or give a longer λ_b)"
        return {"band": reason}

    def curves(self, result, field, params: dict) -> list:
        """For the Curve window: the input field's radially averaged power spectrum with the
        band limit and the spectral peak marked, and the fraction of the power below each k with
        the band limit at the fraction asked for. What the tuning saw, so the fit can be read."""
        from dynamix.core.daf import radial_spectrum
        from dynamix.model.curves import Curve
        from dynamix.model.device import validate_params

        values = getattr(field, "values", None)
        if values is None or np.asarray(values).ndim != 2:
            return []
        p = validate_params(self, params or {})
        t = self._tuned(np.asarray(values, dtype=np.float64), p)
        k, power, fraction = radial_spectrum(values)
        peak = float(k[1:][np.argmax(power[1:])]) if k.size > 1 else float("nan")
        marks = [(t.f_b, f"f_b = {t.f_b:.3g} (λ_b {1 / t.f_b:.1f} px)" if t.f_b > 0 else "f_b"),
                 (peak, f"peak (λ {1 / peak:.1f} px)" if peak > 0 else "peak")]
        how = (f"given λ_b = {p['lambda_b']:g} px" if p["band"] == "given"
               else f"{p['band_frac']:.4g} of the power")
        note = (f"band limit f_b = {t.f_b:.4g} cycles/px ({how}); gap {t.gap} px, σ {t.sigma:.2f} px, "
                f"M {t.M}; {t.ppw:.1f} points per wavelength at f_b"
                + ("" if t.band_limited else " — fewer than 6: the fill is rough"))
        return [Curve("power spectrum", k, power, "k (cycles/px)", "power (annulus mean)",
                      logx=True, logy=True, marks=tuple(marks), note=note),
                Curve("power fraction below k", k, fraction, "k (cycles/px)", "fraction of the power",
                      logx=True, marks=(marks[0],), note=note)]

    def derived_reading(self, name: str, value, field, params: dict | None = None) -> str | None:
        """Under the band-limit knob: what the tuning settled on for this field."""
        if name != "band" or getattr(field, "values", None) is None:
            return None
        values = np.asarray(field.values)
        if values.ndim != 2:
            return None
        from dynamix.model.device import validate_params

        t = self._tuned(values.astype(np.float64), validate_params(self, params or {}))
        if not t.f_b > 0.0:
            return "no band limit"
        return f"λ_b {1.0 / t.f_b:.1f} px · gap {t.gap} px · σ {t.sigma:.1f} px"

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)
