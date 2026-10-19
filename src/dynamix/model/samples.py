# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The per-level sample sets a result carries, in one schema whatever produced them, for the
tools that read a multiresolution quantity's distribution scale by scale (the large-deviation
spectrum, the per-scale histograms, the Gaussian ensemble)::

    {label: {"mods": [1-D array per scale], "scales": px per scale, "weights": per-scale count
             factors or None}}

A Wavelet leaders result publishes the ``Samples`` contract (:mod:`dynamix.model.contracts`):
each set over its valid cells, weighted by the field's cell count over the valid count, the cells
the border took. A WTMM result offers its maxima lines (one modulus per chain per scale, what its
partition functions count) and every maximum.
"""
from __future__ import annotations

import numpy as np

__all__ = ["sample_sets"]

#: Set name in the ``Samples`` contract -> its label on a menu.
LABELS = {"leaders": "leaders", "p-leaders": "p-leaders", "wse": "WSE leaders",
          "coefficients": "coefficients"}


def _flat(values, valid) -> list:
    return [np.asarray(m)[..., v].ravel() if np.ndim(m) > np.ndim(v) else np.asarray(m)[v]
            for m, v in zip(values, valid)]


def sample_sets(result: dict) -> dict:
    """The sets ``result`` carries, in the order a menu lists them; ``{}`` for a result with
    none (a field tail, a points layer, a finest-scale preview)."""
    out = {}
    scales = result.get("scales") if isinstance(result, dict) else None
    if scales is None:
        return out
    n_sc = len(scales)
    chains = result.get("chains") or []
    if chains:
        from dynamix.core.partition_tables import _modulus_matrix

        M = _modulus_matrix(chains, n_sc)
        out["maxima lines"] = {"mods": [M[np.isfinite(M[:, k]), k] for k in range(n_sc)],
                               "scales": scales, "weights": None}
    extrema = result.get("extrema") or []
    mods = [e["mod"] for e in extrema if isinstance(e, dict) and "mod" in e]
    if mods and len(mods) == len(extrema) == n_sc:
        out["every maximum"] = {"mods": mods, "scales": scales, "weights": None}
    samples = result.get("samples")
    if isinstance(samples, dict):
        cells = samples["cells"]
        for name, s in samples["sets"].items():
            flat = _flat(s["values"], s["valid"])
            out[LABELS.get(name, name)] = {
                "mods": flat, "scales": samples["scales"],
                "weights": [float(n) / max(x.size, 1) for n, x in zip(cells, flat)]}
    return out
