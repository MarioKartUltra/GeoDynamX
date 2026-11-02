# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The per-level sample sets of a result (:mod:`dynamix.model.samples`): the WTMM's maxima
lines and every maximum, the ``Samples`` contract's sets flattened over their valid cells with
the count factor the border costs."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.model.samples import has_sample_sets, sample_sets


def _chains(scales):
    def chain(mods):
        mod = np.asarray(mods, dtype=np.float64)
        k = mod.size
        with np.errstate(divide="ignore"):
            lm = np.log2(np.abs(mod))
        return {"x": np.zeros(k, dtype=np.int64), "y": np.arange(k, dtype=np.int64),
                "mod": mod, "log2_mod": lm, "log2_scales": np.log2(scales)[:k]}
    return [chain([8.0, 6.0, 9.0, 7.0]), chain([0.5, 0.4, 0.3, 0.2]),
            chain([2.0, 1.5]), chain([3.0, 2.5, 2.0, 1.5])]


def _line(mods, row=0):
    """One scale's extrema: a straight maxima line along ``row`` with the moduli ``mods``."""
    n = len(mods)
    return {"x": np.arange(n, dtype=np.int64), "y": np.full(n, row, dtype=np.int64),
            "mod": np.asarray(mods, dtype=np.float64), "arg": np.zeros(n), "line_id": np.zeros(n, dtype=np.int64)}


def test_a_wtmm_result_offers_its_chains_without_and_with_the_sup_rule_its_raw_wtmmm_and_every_maximum():
    """The chains as the two partition tables count them (no sup: the modulus along each chain;
    sup: the running maximum from the finest scale), the WTMMM of each scale unchained (the
    maxima along each maxima line, interior points only), every maximum."""
    from dynamix.core.partition_tables import _modulus_matrix, _sup_matrix
    from dynamix.model.samples import CHAINS, CHAINS_SUP, DESCRIPTIONS, RAW

    scales = np.array([2.0, 4.0, 8.0, 16.0])
    extrema = [_line([1.0, 3.0, 2.0, 2.5, 1.0, 0.5]) for _ in scales]
    sets = sample_sets({"scales": scales, "chains": _chains(scales), "extrema": extrema, "params": {}})
    assert list(sets) == [CHAINS, CHAINS_SUP, RAW, "every maximum"]
    assert (CHAINS, CHAINS_SUP, RAW) == ("WTMMM (chains; no sup)", "WTMMM (chains w/ sup)", "WTMMM (raw; no chains)")
    assert [len(m) for m in sets[CHAINS]["mods"]] == [4, 4, 3, 3]              # chains alive per scale
    assert sets[CHAINS]["mods"][0].tolist() == [8.0, 0.5, 2.0, 3.0]
    assert sets[CHAINS]["mods"][2].tolist() == [9.0, 0.3, 2.0]
    assert [len(m) for m in sets[CHAINS_SUP]["mods"]] == [4, 4, 3, 3]          # the same chains
    assert sets[CHAINS_SUP]["mods"][1].tolist() == [8.0, 0.5, 2.0, 3.0]        # the sup from the finest scale
    assert sets[CHAINS_SUP]["mods"][2].tolist() == [9.0, 0.5, 3.0]
    M = _modulus_matrix(_chains(scales), 4)                                     # what the cmax table counts
    np.testing.assert_array_equal(np.sort(sets[CHAINS_SUP]["mods"][3]), np.sort(_sup_matrix(M)[:, 3][np.isfinite(M[:, 3])]))
    assert [m.tolist() for m in sets[RAW]["mods"]] == [[3.0, 2.5]] * 4          # the line's two interior maxima
    assert [len(m) for m in sets["every maximum"]["mods"]] == [6, 6, 6, 6]
    assert all(sets[k]["weights"] is None for k in sets)
    assert all(DESCRIPTIONS[k] for k in sets)
    assert has_sample_sets({"scales": scales, "chains": _chains(scales), "extrema": extrema})
    no_lines = [{"mod": np.array([1.0, 2.0, 3.0])} for _ in scales]             # moduli without positions
    assert list(sample_sets({"scales": scales, "chains": _chains(scales), "extrema": no_lines})) == \
        [CHAINS, CHAINS_SUP, "every maximum"]
    bare = [{"x": np.zeros(2), "y": np.zeros(2)}]                                # a stub's chains: no modulus
    assert sample_sets({"scales": scales, "chains": bare, "extrema": []}) == {}
    assert not has_sample_sets({"scales": scales, "chains": bare}) and not has_sample_sets({"h_map": None})


def test_the_raw_wtmmm_are_the_points_the_chainer_links(clean_registry):
    """On a real WTMM result the raw set holds, per scale, the moduli of xsmurf's single maxima
    of the result's own extrema, the set every chain is drawn from."""
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.core.wtmm_backend import get_backend
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device
    from dynamix.model.samples import CHAINS, RAW
    from tests.conftest import FIXED_EDGES, fbm2d

    register_builtin_devices()
    dev = get_device("wtmm2d")
    field = RasterField(name="f", values=fbm2d(64, 0.6, seed=2), frame=LocalFrame(units="px"),
                        x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    res = dev.compute(field, {**defaults_for(dev), **FIXED_EDGES, "n_oct": 3})
    sets = sample_sets(res)
    single = get_backend("python").single_maxima2d(res["extrema"], smooth=bool(res["params"]["smooth"]))
    assert [m.tolist() for m in sets[RAW]["mods"]] == [s["mod"].tolist() for s in single]
    assert all(len(r) <= len(e["mod"]) for r, e in zip(sets[RAW]["mods"], res["extrema"]))
    assert len(sets[CHAINS]["mods"][0]) <= len(sets[RAW]["mods"][0])           # the chains start at single maxima


def test_the_samples_contract_flattens_over_the_valid_cells_with_the_count_factor():
    valid = [np.ones((8, 8), bool), np.ones((4, 4), bool)]
    valid[1][0, 0] = False
    coef_valid = [np.ones((8, 8), bool), np.ones((4, 4), bool)]
    samples = {"scales": np.array([2.0, 4.0]), "cells": [64, 16], "d": 2,
               "sets": {"leaders": {"values": [np.full((8, 8), 2.0), np.full((4, 4), 4.0)], "valid": valid},
                        "coefficients": {"values": [np.ones((3, 8, 8)), np.ones((3, 4, 4))],
                                         "valid": coef_valid, "positive_only": True}}}
    sets = sample_sets({"scales": samples["scales"], "samples": samples, "params": {"gamma": 1.0}})
    assert list(sets) == ["leaders", "coefficients"]
    assert [len(m) for m in sets["leaders"]["mods"]] == [64, 15]
    assert sets["leaders"]["weights"] == pytest.approx([1.0, 16 / 15])
    assert [len(m) for m in sets["coefficients"]["mods"]] == [192, 48]        # three bands per cell
    assert sets["coefficients"]["weights"] == pytest.approx([64 / 192, 16 / 48])
    assert sample_sets({"params": {}}) == {} and sample_sets(None) == {}
    assert has_sample_sets({"scales": samples["scales"], "samples": samples})


def test_the_chains_are_judged_once_however_often_a_scale_tick_asks():
    """A scale tick hands the button refresh a new result dict over the same chains list; the walk
    over every chain runs once per list."""
    looked = []

    class Chain(dict):
        def __contains__(self, key):
            looked.append(key)
            return super().__contains__(key)

    scales = np.array([2.0, 4.0, 8.0, 16.0])
    result = {"scales": scales, "chains": [Chain(c) for c in _chains(scales)], "extrema": []}
    for level in range(4):
        assert has_sample_sets({**result, "_scale_idx": level})
    assert len(looked) == len(result["chains"])
    assert has_sample_sets({**result, "chains": list(result["chains"])})    # another result
    assert len(looked) == 2 * len(result["chains"])


def test_the_histogram_map_keeps_the_cells_whose_chain_spans_every_fitted_level():
    """Cardo & Corvalán read each point's exponent off the regression across the dyadic levels:
    the GMWLP histogram and the box counting take the cells fitted over every level of the
    window, the h-map's edge cells fitted over the finer levels alone staying on the canvas
    only. A map without a levels map (another tool's) passes through unchanged."""
    from dynamix.model.samples import histogram_h_map

    h = np.arange(16.0, dtype=np.float32).reshape(4, 4)
    levels = np.array([[3, 3, 3, 3], [3, 5, 5, 3], [3, 5, 5, 3], [np.nan] * 4], dtype=np.float32)
    result = {"h_map": h, "levels_map": levels, "record": {"levels": {"j_min": 2, "J": 6}}}
    out = histogram_h_map(result)
    assert np.isfinite(out).sum() == 4 and np.array_equal(out[1:3, 1:3], h[1:3, 1:3])
    assert out is not h and np.array_equal(result["h_map"], h)                # the canvas's map untouched
    plain = {"h_map": h}
    assert histogram_h_map(plain) is h
    assert histogram_h_map({}) is None
    thin = {**result, "levels_map": np.full((4, 4), 3.0, dtype=np.float32)}
    assert not np.isfinite(histogram_h_map(thin)).any()                      # no full chain: nothing to count
