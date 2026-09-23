# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.devices.filters.HLineModulus -- the H-line-aware modulus filter.

"min |W| frac of peak is treating the horizontal extrema as individual points
instead of lists of points ... filter based on the sup of the horizontal chain"; then the four
options: sup at a scale, sup at all scales, mean at a scale, and segments (a chain cut at the
local minima of |W| along it into local-max-to-local-min sections, joined by the minima nodes)."""
from __future__ import annotations

import numpy as np

from dynamix.devices.filters import HLineModulus
from dynamix.model.device import defaults_for


def _layer(mods, line_ids, y=0):
    """Points laid out along x in the given order (so the walk orders them the same way)."""
    mod = np.asarray(mods, dtype=np.float64)
    n = mod.size
    return {"x": np.arange(n, dtype=np.int64), "y": np.full(n, y, dtype=np.int64),
            "mod": mod, "arg": np.zeros(n), "line_id": np.asarray(line_ids, dtype=np.int64)}


def _result(*layers, **extra):
    d = {"extrema": list(layers), "_shape": (8, 64)}
    d.update(extra)
    return d


def _params(**over):
    d = defaults_for(HLineModulus())
    d.update(over)
    return d


def test_defaults_are_a_passthrough():
    res = _result(_layer([1.0, 0.1], [0, 0]))
    assert HLineModulus().apply(res, _params()) is res


def test_sup_keeps_the_whole_line_when_its_peak_clears_the_floor():
    # line 0: peak 1.0 with a weak tail; line 1: never above 0.3; two singletons judged alone
    mods = [1.0, 0.2, 0.1,   0.3, 0.2,   0.9, 0.1]
    lids = [0, 0, 0,         1, 1,       -1, -1]
    out = HLineModulus().apply(_result(_layer(mods, lids)), _params(frac=0.5, mode="sup"))
    kept = out["extrema"][0]
    assert list(kept["mod"]) == [1.0, 0.2, 0.1, 0.9]          # weak points of line 0 survive
    assert list(kept["line_id"]) == [0, 0, 0, -1]


def test_mean_uses_the_average_along_the_line():
    mods = [1.0, 0.1, 0.1,   0.6, 0.6, 0.6]
    lids = [0, 0, 0,         1, 1, 1]
    out = HLineModulus().apply(_result(_layer(mods, lids)), _params(frac=0.5, mode="mean"))
    assert set(out["extrema"][0]["line_id"]) == {1}          # line 0's mean is 0.4


def test_segments_drop_a_weak_middle_but_keep_the_joining_minima():
    # one line: two strong lobes joined through a weak bump; peak of the layer is 1.0 elsewhere
    mods = [0.9, 0.5, 0.1, 0.3, 0.1, 0.5, 0.9,   1.0]
    lids = [0, 0, 0, 0, 0, 0, 0,                 -1]
    out = HLineModulus().apply(_result(_layer(mods, lids)), _params(frac=0.6, mode="segments"))
    kept = out["extrema"][0]
    assert list(kept["mod"]) == [0.9, 0.5, 0.1, 0.1, 0.5, 0.9, 1.0]   # the 0.3 section is gone
    assert list(kept["line_id"]) == [0] * 6 + [-1]                    # still ONE h-chain


def test_segments_drop_a_line_that_is_weak_everywhere_including_its_minima():
    mods = [0.2, 0.3, 0.2, 0.3, 0.2,   1.0]
    lids = [0, 0, 0, 0, 0,             -1]
    out = HLineModulus().apply(_result(_layer(mods, lids)), _params(frac=0.6, mode="segments"))
    assert list(out["extrema"][0]["mod"]) == [1.0]


def test_sup_all_scales_reads_the_chain_through_the_point():
    # scale 0: line 0 is weak here but its point (1, 0) is the foot of a chain that reaches 1.0
    # at scale 1; line 1 is weak and on no chain.
    mods = [0.1, 0.1,   0.1, 0.1]
    lids = [0, 0,       1, 1]
    chains = [{"x": np.array([1, 1]), "y": np.array([0, 0]), "mod": np.array([0.1, 1.0])}]
    res = _result(_layer(mods, lids), chains=chains)
    out = HLineModulus().apply(res, _params(frac=0.5, mode="sup_all_scales"))
    assert set(out["extrema"][0]["line_id"]) == {0}


def test_every_layer_uses_its_own_peak():
    a = _layer([1.0, 0.1], [0, 0])
    b = _layer([0.05, 0.01,   0.001, 0.001], [0, 0, 1, 1], y=1)   # coarse: everything is smaller
    out = HLineModulus().apply(_result(a, b), _params(frac=0.5, mode="sup"))
    assert set(out["extrema"][0]["line_id"]) == {0}
    assert set(out["extrema"][1]["line_id"]) == {0}                 # 0.05 is this layer's peak


def test_input_result_is_not_mutated():
    res = _result(_layer([1.0, 0.1, 0.3, 0.2], [0, 0, 1, 1]))
    before = {k: v.copy() for k, v in res["extrema"][0].items()}
    HLineModulus().apply(res, _params(frac=0.5, mode="sup"))
    for k, v in before.items():
        np.testing.assert_array_equal(res["extrema"][0][k], v)
