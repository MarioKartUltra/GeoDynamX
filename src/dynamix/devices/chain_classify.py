# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Seam classification: tag V-chains that are survey-stitch artifacts, not features.

Physical basis (grounded in the literature extraction of the design): the Hölder exponent
along a chain is the slope of log2|W| vs log2 a, and the Mallat-Hwang exponent table (via Liew
1997 §4.5) gives the two seam signatures this device looks for -- a step edge has h ~= 0 (modulus
flat across scale), a one-pixel stitch line (delta ridge) has h ~= -1 (modulus proportional to
a^-1). Neither alone is sufficient: a real geological step edge also has h ~= 0. What survey seams
add is straightness -- a constant WTMMM argument along the H-line, the signature Decoster et al.
(2000) tie to modulus-argument independence. A chain is tagged only when both the exponent band
and the straightness cutoff agree.

Only pure numpy over the chain/extrema dict schema -- no wtmm/wtmm_ebsd import, ever (this module
lives under devices/, which stays headless and free of the optional analysis-core dependency).

**The fit-floor doctrine.** ``h`` is a slope over ``log2(scale)``, and not every scale a
chain touches is honest evidence for that slope: the fit-floor doctrine (grounded in the wavelet's own real-space support) says a point only belongs
in an exponent fit when its scale ``a`` satisfies ``σ(a) ≥ 3 px`` (``σ ≈ 1.57·a``, so ``a ≳ 1.9``)
-- below that the analyzing kernel is barely wider than a pixel and the "slope" it reports is
discretization noise, not physics -- AND ``a ≤ L/8`` where ``L`` is the raster's own shorter side,
past which a chain is reporting on a scale comparable to the image itself. ``fit_a_min`` is the
device's own knob for the lower edge (a user-facing quantity, tuned per survey, in the SAME units
as ``wtmm2d``'s own ``a_min``); the ``/8`` fraction is NOT a Param -- ``L`` is a property of the
raster, not something a user retunes per run, so a second knob for it would be speculative
flexibility with no use site. The window is applied ONCE, here,
before any estimator ever sees a chain's points -- the estimators in ``estimators.py`` stay pure
functions of whatever chain dict they are handed, fitting every finite point in it, exactly as
before.
"""
from __future__ import annotations

import numpy as np

from dynamix.devices.estimators import HOLDER_ESTIMATORS
from dynamix.model.param import Param, ParamKind

#: The fit-floor doctrine's upper edge, the design's "a ≤ L/8" -- a doctrine constant, not a Param
#: (see the module docstring). ``L`` is ``min(result["_shape"])``, read per-result in :meth:`
#: ChainClassify.apply`, never per-chain (every chain in one result shares the same raster).
_FIT_A_MAX_FRACTION = 1.0 / 8.0


def _fit_masked(chain: dict, fit_a_min: float, fit_a_max: float | None) -> dict:
    """A COPY of ``chain`` with ``log2_scales``/``log2_mod`` restricted to the fit-floor doctrine's
    window, ``a = 2**log2_scales`` -- everything else in ``chain`` rides through untouched.

    ``fit_a_max`` is ``None`` when the result carries no ``_shape`` to derive ``L`` from (the
    upper edge is then simply not enforced, rather than guessed at). Points failing the window are
    dropped, not merely masked to NaN, so the estimator's own "need at least 2 points" guard
    (``estimators.py``) is what turns an empty or single-point survival into an honest NaN --
    the window logic here does not have to duplicate that count itself.
    """
    log2_scales = np.asarray(chain.get("log2_scales", ()), dtype=np.float64)
    log2_mod = np.asarray(chain.get("log2_mod", ()), dtype=np.float64)
    a = np.exp2(log2_scales)
    keep = a >= fit_a_min
    if fit_a_max is not None:
        keep &= a <= fit_a_max
    out = dict(chain)
    out["log2_scales"] = log2_scales[keep]
    out["log2_mod"] = log2_mod[keep]
    return out


def chain_straightness(chain: dict, extrema0: dict) -> float:
    """Axial resultant of WTMMM argument along the H-line under a chain's finest-scale point.

    Locates the chain's finest-scale point ``(x[0], y[0])`` in ``extrema0``
    (``result["extrema"][0]``, index 0 = finest scale), then gathers every point sharing that
    point's ``line_id`` and computes ``|mean(exp(2j * arg))|``. The argument is doubled before
    averaging so a constant *gradient direction* reads as straight even across the +-pi flip a
    two-sided edge produces (the doubling maps antipodal directions onto the same axial value, so
    a mean over an axis rather than a full circle does not cancel to zero).

    Any lookup failure -- no ``"arg"``/``"line_id"`` key, the chain's point not present in
    ``extrema0``, or a negative (unassigned) ``line_id`` -- returns NaN. That is deliberate: NaN
    fails every threshold comparison, so a chain we cannot honestly measure as straight is never
    tagged a seam.
    """
    arg = extrema0.get("arg")
    line_id = extrema0.get("line_id")
    if arg is None or line_id is None:
        return float("nan")

    ex = np.asarray(extrema0.get("x", ()))
    ey = np.asarray(extrema0.get("y", ()))
    cx = chain.get("x")
    cy = chain.get("y")
    if cx is None or cy is None or len(cx) == 0 or len(cy) == 0:
        return float("nan")
    if len(ex) == 0 or len(ey) == 0:
        return float("nan")

    match = np.flatnonzero((ex == cx[0]) & (ey == cy[0]))
    if match.size == 0:
        return float("nan")

    line_id = np.asarray(line_id)
    lid = int(line_id[match[0]])
    if lid < 0:
        return float("nan")

    on_line = line_id == lid
    if not on_line.any():
        return float("nan")

    arg_line = np.asarray(arg, dtype=np.float64)[on_line]
    return float(np.abs(np.mean(np.exp(2j * arg_line))))


class ChainClassify:
    """Tag chains matching the step-seam or delta-seam signature; flag or exclude them.

    ``flag`` (default) leaves ``chains`` untouched in length and order and only adds ``tags`` /
    ``tag_origin`` to the matched ones, so downstream devices keep seeing every chain. ``exclude``
    moves tagged chains into ``chains_excluded`` -- evidence a consumer can still render as ghosts
    -- and records how many were dropped in ``_chains_dropped``.
    """

    name = "chain_classify"
    params = (
        Param("estimator", ParamKind.CHOICE, default="ols", choices=("ols", "max"),
              label="Hölder estimator"),
        # Fit-floor doctrine, lower edge: σ ≈ 1.57·a must be ≥ 3 px before a scale is
        # honest evidence for h. Default 2.0 -> σ ≈ 3.1 px, just past the floor.
        Param("fit_a_min", ParamKind.FLOAT, default=2.0, min=0.01, max=64.0,
              soft_min=1.0, soft_max=4.0, label="fit floor aₘᵢₙ (σ ≥ 3 px)"),
        Param("h_step_lo", ParamKind.FLOAT, default=-0.15, min=-3.0, max=3.0,
              soft_min=-0.5, soft_max=0.5, label="step h ≥"),
        Param("h_step_hi", ParamKind.FLOAT, default=0.15, min=-3.0, max=3.0,
              soft_min=-0.5, soft_max=0.5, label="step h ≤"),
        Param("h_delta_lo", ParamKind.FLOAT, default=-1.2, min=-3.0, max=3.0,
              soft_min=-1.5, soft_max=-0.5, label="delta h ≥"),
        Param("h_delta_hi", ParamKind.FLOAT, default=-0.8, min=-3.0, max=3.0,
              soft_min=-1.5, soft_max=-0.5, label="delta h ≤"),
        Param("straightness_min", ParamKind.FLOAT, default=0.85, min=0.0, max=1.0,
              label="straightness ≥"),
        Param("action", ParamKind.CHOICE, default="flag", choices=("flag", "exclude"),
              label="Action"),
        Param("show_ghosts", ParamKind.BOOL, default=True, label="Show ghosts"),
    )

    def apply(self, result: dict, params: dict) -> dict:
        chains = result.get("chains") or []
        extrema = result.get("extrema") or []
        extrema0 = extrema[0] if extrema else {}
        estimator = HOLDER_ESTIMATORS[str(params["estimator"])]
        fit_a_min = float(params["fit_a_min"])
        shape = result.get("_shape")
        fit_a_max = min(shape) * _FIT_A_MAX_FRACTION if shape else None

        classified = []          # [(chain, tagged_bool), ...], order preserved
        for chain in chains:
            h = estimator(_fit_masked(chain, fit_a_min, fit_a_max))
            straightness = chain_straightness(chain, extrema0)
            tags = []
            if straightness >= float(params["straightness_min"]):
                if params["h_step_lo"] <= h <= params["h_step_hi"]:
                    tags.append("seam_step")
                if params["h_delta_lo"] <= h <= params["h_delta_hi"]:
                    tags.append("seam_delta")
            if tags:
                tagged_chain = dict(chain)
                tagged_chain["tags"] = tags
                tagged_chain["tag_origin"] = "device:chain_classify"
                classified.append((tagged_chain, True))
            else:
                classified.append((chain, False))

        out = dict(result)
        if params["action"] == "exclude":
            out["chains"] = [c for c, tagged in classified if not tagged]
            out["chains_excluded"] = [c for c, tagged in classified if tagged]
            out["_chains_dropped"] = len(out["chains_excluded"])
        else:
            out["chains"] = [c for c, _tagged in classified]
        out["_show_ghosts"] = bool(params["show_ghosts"])
        return out
