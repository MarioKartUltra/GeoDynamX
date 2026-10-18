# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The two widths of a q-wavelet, resolved for every device that offers one.

A q-wavelet's width is set two ways: FIXED, one β of Borges et al. 2004's ``e_q^(-beta x^2)``
for every q, so q changes the shape and the width together; or Q-PAIRED, ``beta(q) = 1/(2(2 -
q))`` (``microcanonical.paired_q_beta``), which holds the escort (q-)variance at one per
component in 2-D, so q changes the shape alone. Both agree at q = 1. The devices resolve the
knobs to a number here; the core takes the number. A kernel without a β (the q-Meyer taper)
declares the pairing knob alone and resolves what it holds itself.
"""
from __future__ import annotations

from dynamix.model.param import Param, ParamKind

__all__ = ["q_width", "q_width_check", "width_knobs"]


def width_knobs(q_wavelets: tuple, *, beta_wavelets: "tuple | None" = None,
                keyed_at_default: bool = True) -> tuple:
    """``q_pairing`` ("fixed" / "q-paired", label "q width"), applying while ``wavelet`` is one
    of ``q_wavelets``, and ``q_beta`` (the fixed width, default 1/2), applying while
    ``wavelet`` is one of ``beta_wavelets`` (default ``q_wavelets``) and the width is fixed."""
    q_wavelets = tuple(q_wavelets)
    beta_wavelets = q_wavelets if beta_wavelets is None else tuple(beta_wavelets)
    return (
        Param("q_pairing", ParamKind.CHOICE, default="fixed", choices=("fixed", "q-paired"),
              label="q width", active_when=("wavelet", q_wavelets),
              keyed_at_default=keyed_at_default),
        Param("q_beta", ParamKind.FLOAT, default=0.5, min=0.05, max=5.0, soft_min=0.1,
              soft_max=2.0, units="", label="β_q",
              active_when=(("wavelet", beta_wavelets), ("q_pairing", ("fixed",))),
              keyed_at_default=keyed_at_default),
    )


def q_width(params: dict, q_wavelets: tuple) -> "float | None":
    """The width β the chosen q kernel runs with (None unless ``wavelet`` is one of
    ``q_wavelets``): the knob's value when fixed, ``microcanonical.paired_q_beta(q)`` when
    q-paired."""
    if params.get("wavelet") not in q_wavelets:
        return None
    if params.get("q_pairing", "fixed") == "q-paired":
        from dynamix.core.microcanonical import paired_q_beta

        return paired_q_beta(float(params["q_tsallis"]))
    return float(params.get("q_beta", 0.5))


def q_width_check(params: dict, q_wavelets: tuple) -> None:
    """A q-paired width needs q < 2: the 2-D escort variance it holds at one is infinite from
    q = 2."""
    if (params.get("wavelet") in q_wavelets and params.get("q_pairing") == "q-paired"
            and float(params.get("q_tsallis", 1.5)) >= 2.0):
        raise ValueError(f"a q-paired width needs q < 2 (got q = {params['q_tsallis']}): the "
                         "escort variance it holds at one is infinite from q = 2 -- use a fixed "
                         "width there")
