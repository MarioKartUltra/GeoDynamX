# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Periodic extension (DAF), spec 25 §3.1: a non-periodic field made periodic, so a scale-space tool
after it can transform it with periodic boundaries and no padding.

The field is extended by a gap after its last row and column, filled by the self-tuned Hermite DAF
(:mod:`dynamix.core.daf`; Frishman, Hoffman, Rakauskas and Kouri 1996; Hoffman, Gunaratne, Zhang and
Kouri 2000), so that the whole is periodic. The grid keeps the field's pixel size and origin, so the
extended raster reaches past the original bounds by the gap. The output carries the stamp
``edges.filled = {"rows": H, "cols": W}``, the real block, which WTMM reads: no maxima in the gap, the
threshold from real pixels, and its Edge ``auto`` resolving to ``periodic``. Below the DAF's band
limit (fewer than 6 points per wavelength) the extension does not apply: the field passes through
unchanged and the strip's ⚠ gives the reason.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from dynamix.model.contracts import FIELD, RECEIVED
from dynamix.model.param import Param, ParamKind

__all__ = ["FILLED", "PeriodicExtension"]

#: The provenance stamp naming a Periodic extension output's real block.
FILLED = "edges.filled"


def _extended_axis(axis, n: int) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    step = float(axis[1] - axis[0]) if axis.size > 1 else 1.0
    return axis[0] + step * np.arange(axis.size + int(n), dtype=np.float64)


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

    params = (
        Param("band_frac", ParamKind.FLOAT, default=0.99, min=0.5, max=0.9999, soft_min=0.9,
              soft_max=0.999, label="Band limit (fraction of the power)", level="advanced"),
        Param("M", ParamKind.INT, default=12, min=2, max=40, soft_min=4, soft_max=24,
              label="DAF order M", level="advanced"),
        Param("lam", ParamKind.FLOAT, default=1e-3, min=0.0, max=1.0, soft_min=0.0, soft_max=0.01,
              label="Regularization λ", level="advanced"),
    )

    @staticmethod
    def _tuned(values, params: dict):
        from dynamix.core.daf import self_tune

        return self_tune(values, frac=params["band_frac"], M=params["M"], lam=params["lam"])

    def compute(self, field, params: dict, *, progress=None):
        if isinstance(field, dict):
            raise ValueError("Periodic extension runs on the FIELD, before the analysing transform — "
                             "drag it ahead of the scale-space tool in the chain")
        from dynamix.core.daf import extend_2d

        values = np.asarray(field.values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Periodic extension takes a single-band field")
        tuned = self._tuned(values, params)
        if tuned.fallback:
            return field                             # nothing honest to add; the ⚠ says why
        if progress is not None:
            progress("periodic extension", 0.0)
        extended = extend_2d(values, tuned)
        if progress is not None:
            progress("periodic extension", 1.0)
        H, W = values.shape
        gy, gx = extended.shape[0] - H, extended.shape[1] - W
        provenance = dict(field.provenance)
        provenance[FILLED] = {"rows": int(H), "cols": int(W)}
        provenance["periodic_extension"] = {"f_b": float(tuned.f_b), "gap": int(tuned.gap),
                                            "M": int(tuned.M), "sigma": float(tuned.sigma),
                                            "lam": float(tuned.lam)}
        return dataclasses.replace(field, values=extended, name=f"{field.name}+daf{tuned.gap}",
                                   x_axis=_extended_axis(field.x_axis, gx),
                                   y_axis=_extended_axis(field.y_axis, gy),
                                   provenance=provenance)

    def knob_warnings(self, field, params: dict) -> dict:
        """The ⚠ beside the band-limit knob where the DAF extension does not apply to this field."""
        values = getattr(field, "values", None)
        if values is None or np.asarray(values).ndim != 2:
            return {}
        from dynamix.model.device import validate_params

        tuned = self._tuned(np.asarray(values, dtype=np.float64), validate_params(self, params or {}))
        return {"band_frac": tuned.fallback} if tuned.fallback else {}

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)
