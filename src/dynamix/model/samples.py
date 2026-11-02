# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The per-level sample sets a result carries, in one schema whatever produced them, for the
tools that read a multiresolution quantity's distribution scale by scale (the large-deviation
spectrum, the per-scale histograms, the Gaussian ensemble)::

    {label: {"mods": [1-D array per scale], "scales": px per scale, "weights": per-scale count
             factors or None}}

A Wavelet leaders result publishes the ``Samples`` contract (:mod:`dynamix.model.contracts`):
each set over its valid cells, weighted by the field's cell count over the valid count, the cells
the border took. A WTMM result offers the WTMMM chained across scales, without and with the sup
rule (one modulus per chain per scale, the moduli its two partition tables count), the WTMMM of
each scale unchained (the maxima along each scale's maxima lines, the points the chainer links),
and, experimental, the complete horizontal chains: every pixel of each scale's maxima lines with
its modulus (xsmurf's ``efct`` default, every maxima pixel), the extrema the result keeps.
"""
from __future__ import annotations

import numpy as np

__all__ = ["CHAINS", "CHAINS_SUP", "DESCRIPTIONS", "H_CHAINS", "RAW", "has_sample_sets", "histogram_h_map", "sample_sets"]

#: The WTMM's sets by their names on a menu.
CHAINS, CHAINS_SUP, RAW = "WTMMM (chains; no sup)", "WTMMM (chains w/sup)", "WTMMM (raw; no chains)"
H_CHAINS = "WTMM (complete h-chains) (experimental)"

#: Set name in the ``Samples`` contract -> its label on a menu.
LABELS = {"leaders": "leaders", "p-leaders": "p-leaders", "wse": "WSE leaders",
          "coefficients": "coefficients", "cwt-leaders": "CWT leaders", "cwt-p-leaders": "CWT p-leaders"}

#: What each set holds, in one sentence, by its name on a menu (a chooser's tooltips).
DESCRIPTIONS = {
    CHAINS: "The WTMMM linked across scales into chains, one modulus per chain per scale: the set Z(q, a) "
            "runs over, as the no-sup (std) table counts it.",
    CHAINS_SUP: "The WTMMM chains with the sup rule, each modulus the largest along its chain up to that "
                "scale: the set the sup (cmax) table counts.",
    RAW: "Every WTMMM at each scale (the maxima of the modulus along that scale's maxima lines), "
         "unchained: chained or not, every one counts.",
    H_CHAINS: "Experimental: every pixel of each scale's maxima lines (the complete horizontal chains) "
              "with its modulus, xsmurf's efct default.",
    "leaders": "The wavelet leaders: at each level the largest coefficient over the 3×3 block of cells "
               "around each cell and all their finer descendants.",
    "p-leaders": "The p-leaders: at each level the ℓᵖ sum of the coefficients over the same block and "
                 "descendants as a leader's.",
    "WSE leaders": "The (θ, ω)-leaders of the weak scaling exponent: the largest coefficient over the window "
                   "of finer levels and nearby cells its schedule sets.",
    "coefficients": "The wavelet coefficients themselves, every band at each level.",
    "CWT leaders": "The leaders of the continuous transform: at each scale the largest modulus over the "
                   "finer scales within a disc around each pixel.",
    "CWT p-leaders": "The p-leaders of the continuous transform: the ℓᵖ disc sums over the finer scales "
                     "in place of the largest modulus."}


def _flat(values, valid) -> list:
    return [np.asarray(m)[..., v].ravel() if np.ndim(m) > np.ndim(v) else np.asarray(m)[v]
            for m, v in zip(values, valid)]


#: The chains list :func:`_lines` judged last and its verdict. Every scale tick refreshes the
#: buttons with a new result over the same list, and the judgement walks every chain.
_judged: tuple = (None, False)


def _lines(result: dict) -> list:
    """The chains whose moduli are recorded (a chain built by a stub carries none)."""
    global _judged
    chains = result.get("chains") or []
    seen, verdict = _judged
    if chains is not seen:
        verdict = bool(chains) and all(isinstance(ch, dict) and "mod" in ch for ch in chains)
        _judged = (chains, verdict)
    return chains if verdict else []


def _maxima(result: dict) -> list:
    extrema = result.get("extrema") or []
    mods = [e["mod"] for e in extrema if isinstance(e, dict) and "mod" in e]
    return mods if mods and len(mods) == len(extrema) == len(result["scales"]) else []


#: The extrema :func:`_single_maxima` reduced last and their reduction: a reduction walks every
#: scale's maxima, and each window opened on one result asks again.
_reduced: tuple = (None, [])


def _single_maxima(result: dict) -> list:
    """The WTMMM of each scale, unchained: the maxima of the modulus along each scale's maxima
    lines (xsmurf's ``ssm``, with the result's own ``smooth``), the points the chainer links;
    ``[]`` when the extrema lack their positions and lines."""
    global _reduced
    extrema = result.get("extrema") or []
    if extrema is not _reduced[0]:
        keys = ("x", "y", "mod", "arg", "line_id")
        ok = (len(extrema) == len(result["scales"])
              and all(isinstance(e, dict) and all(k in e for k in keys) for e in extrema))
        mods = []
        if ok:
            from dynamix.core.wtmm_backend import get_backend

            smooth = bool((result.get("params") or {}).get("smooth", False))
            mods = [e["mod"] for e in get_backend("python").single_maxima2d(extrema, smooth=smooth)]
        _reduced = (extrema, mods)
    return _reduced[1]


def histogram_h_map(result):
    """The h-map the microcanonical histogram and the box counting read: a leaders result's
    cells whose chain spans every fitted level (``levels_map`` at ``J − j_min + 1``), each point's
    exponent being the regression across the dyadic levels there (Cardo and Corvalán 2009, the
    GMWLP); the edge cells fitted over the finer levels alone stay on the canvas only. Another
    tool's map, without a levels map, passes through unchanged; None without an h-map."""
    if not isinstance(result, dict) or result.get("h_map") is None:
        return None
    h_map, levels = result["h_map"], result.get("levels_map")
    window = ((result.get("record") or {}).get("levels") or {})
    if levels is None or "J" not in window or "j_min" not in window:
        return h_map
    full = int(window["J"]) - int(window["j_min"]) + 1
    return np.where(np.asarray(levels) == full, np.asarray(h_map), np.nan).astype(np.asarray(h_map).dtype)


def has_sample_sets(result) -> bool:
    """Whether :func:`sample_sets` would offer anything, without building it (a button refresh
    on every landing)."""
    if not isinstance(result, dict) or result.get("scales") is None:
        return False
    return bool(_lines(result) or _maxima(result) or isinstance(result.get("samples"), dict))


def sample_sets(result: dict) -> dict:
    """The sets ``result`` carries, in the order a menu lists them; ``{}`` for a result with
    none (a field tail, a points layer, a finest-scale preview)."""
    out = {}
    scales = result.get("scales") if isinstance(result, dict) else None
    if scales is None:
        return out
    n_sc = len(scales)
    chains = _lines(result)
    if chains:
        from dynamix.core.partition_tables import _modulus_matrix, _sup_matrix

        M = _modulus_matrix(chains, n_sc)
        for name, table in ((CHAINS, M), (CHAINS_SUP, _sup_matrix(M))):
            out[name] = {"mods": [table[np.isfinite(table[:, k]), k] for k in range(n_sc)],
                         "scales": scales, "weights": None}
    raw = _single_maxima(result)
    if raw:
        out[RAW] = {"mods": raw, "scales": scales, "weights": None}
    mods = _maxima(result)
    if mods:
        out[H_CHAINS] = {"mods": mods, "scales": scales, "weights": None}
    samples = result.get("samples")
    if isinstance(samples, dict):
        cells = samples["cells"]
        for name, s in samples["sets"].items():
            flat = _flat(s["values"], s["valid"])
            out[LABELS.get(name, name)] = {
                "mods": flat, "scales": samples["scales"],
                "target": ("coefficient large deviations; wavelet-dependent, not a pointwise spectrum"
                           if name == "coefficients" else "weak-scaling leader large deviations"
                           if name == "wse" else "leader large deviations"),
                "weights": [float(n) / max(x.size, 1) for n, x in zip(s.get("cells", cells), flat)]}
    return out
