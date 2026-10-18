# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""What the two invariant-measure tools share: the derivative scheme's knobs and how they resolve
to :func:`dynamix.core.disorder.filtered`'s keywords, and how a field counts as periodic."""
from __future__ import annotations

from dynamix.core import disorder
from dynamix.model.param import Param, ParamKind

__all__ = ["DETREND_PARAM", "FLOOR_PARAM", "SCHEME_PARAMS", "deriv_record", "periodized", "scheme"]

#: The derivative scheme: the papers' DAF with its order and width, a Gaussian with its width, or
#: spectral derivatives; and whether the field is periodic as it stands.
SCHEME_PARAMS = (
    Param("deriv", ParamKind.CHOICE, default="daf", choices=disorder.DERIVS,
          choice_labels=("DAF", "Gaussian", "spectral"), label="Derivatives"),
    Param("daf_M", ParamKind.INT, default=disorder.DAF_DEFAULTS["M"], min=2, max=40,
          soft_min=4, soft_max=20, label="M", active_when=("deriv", ("daf",))),
    Param("daf_sigma", ParamKind.FLOAT, default=disorder.DAF_DEFAULTS["sigma_over_dx"], min=1.0,
          max=12.0, soft_min=2.0, soft_max=6.0, label="σ/Δ", active_when=("deriv", ("daf",))),
    Param("sigma_d", ParamKind.FLOAT, default=1.0, min=0.3, max=16.0, soft_min=0.5, soft_max=4.0,
          units="px", label="σ_d (px)", active_when=("deriv", ("gaussian",))),
    # A field that is periodic as it stands (model output on a periodic domain): no padding, and
    # spectral derivatives allowed. A Periodic extension's stamp declares it by itself.
    Param("periodic", ParamKind.BOOL, default=False, label="Periodic input"),
)
DETREND_PARAM = Param("detrend", ParamKind.CHOICE, default="mean", choices=("mean", "plane", "none"),
                      label="Detrend")
#: The density floor for negative moment orders; 0 takes the core's default fraction.
FLOOR_PARAM = Param("beta_floor", ParamKind.FLOAT, default=0.0, min=0.0, max=1.0, soft_min=0.0,
                    soft_max=1e-3, label="β<0 floor", level="advanced")


def scheme(params: dict) -> dict:
    """:func:`disorder.filtered`'s keywords from the knobs."""
    return dict(deriv=params["deriv"], sigma_d=float(params["sigma_d"]),
                M=int(params["daf_M"]), sigma_over_dx=float(params["daf_sigma"]))


def periodized(field, params: dict) -> "str | None":
    """Why the field counts as periodic — declared by the knob, or stamped by a Periodic
    extension — or None when it does not."""
    if params["periodic"]:
        return "declared periodic"
    provenance = getattr(field, "provenance", None) or {}
    return "edges.filled" if provenance.get("edges.filled") else None


def deriv_record(sch: dict, periodized_by: "str | None") -> dict:
    """The record's account of the scheme: its name, reach and settings, and what periodized."""
    record = {"scheme": sch["deriv"], "reach_px": disorder.reach(**sch)}
    if sch["deriv"] == "daf":
        record.update({"M": sch["M"], "sigma_over_dx": sch["sigma_over_dx"]})
    elif sch["deriv"] == "gaussian":
        record["sigma_d"] = sch["sigma_d"]
    if periodized_by is not None:
        record["periodized_by"] = periodized_by
    return record
