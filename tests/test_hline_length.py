# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.devices.filters.HLineLength -- the H-line point-count range filter."""
from __future__ import annotations

import numpy as np

from dynamix.devices.filters import HLineLength
from dynamix.model.device import defaults_for


def _layer(line_ids):
    lid = np.asarray(line_ids, dtype=np.int64)
    n = lid.size
    return {"x": np.arange(n, dtype=np.int64), "y": np.zeros(n, dtype=np.int64),
            "mod": np.ones(n), "arg": np.zeros(n), "line_id": lid}


def _result(*layers):
    return {"extrema": list(layers)}


def _params(**over):
    d = defaults_for(HLineLength())
    d.update(over)
    return d


def test_defaults_are_a_passthrough():
    res = _result(_layer([0, 0, 0, 1, 1, -1]))
    out = HLineLength().apply(res, _params())
    assert out is res                      # min 1, no cap: nothing to do, no copy


def test_min_len_drops_short_lines_and_keeps_orphans():
    # line 0 has 3 points, line 1 has 10, plus two orphans
    lid = [0] * 3 + [1] * 10 + [-1, -1]
    out = HLineLength().apply(_result(_layer(lid)), _params(min_len=5))
    kept = out["extrema"][0]["line_id"]
    assert set(kept[kept >= 0]) == {1} and np.count_nonzero(kept == 1) == 10
    assert np.count_nonzero(kept == -1) == 2


def test_max_len_caps_long_lines():
    lid = [0] * 3 + [1] * 10 + [-1]
    out = HLineLength().apply(_result(_layer(lid)), _params(max_len=5))
    kept = out["extrema"][0]["line_id"]
    assert set(kept[kept >= 0]) == {0} and np.count_nonzero(kept == -1) == 1


def test_range_is_inclusive_on_both_ends():
    lid = [0] * 4 + [1] * 6 + [2] * 8
    out = HLineLength().apply(_result(_layer(lid)), _params(min_len=6, max_len=6))
    assert set(out["extrema"][0]["line_id"]) == {1}


def test_every_layer_of_the_stack_is_filtered():
    out = HLineLength().apply(
        _result(_layer([0] * 2), _layer([0] * 9)), _params(min_len=5))
    assert out["extrema"][0]["line_id"].size == 0
    assert out["extrema"][1]["line_id"].size == 9


def test_input_result_is_not_mutated():
    res = _result(_layer([0] * 3 + [1] * 10))
    before = res["extrema"][0]["line_id"].copy()
    HLineLength().apply(res, _params(min_len=5))
    np.testing.assert_array_equal(res["extrema"][0]["line_id"], before)


def test_scale_select_stamps_the_unfiltered_layer_for_geometry_reuse():
    # 2026-08-30: downstream filters mint a NEW extrema dict per tweak, so
    # geometry caches keyed on identity always miss. ScaleSelect stamps the id-stable UNFILTERED
    # layer so the canvas can order H-lines once and mask per tweak.
    import numpy as np
    from dynamix.devices.filters import ScaleSelect
    from dynamix.model.device import defaults_for
    layer = {"x": np.arange(4), "y": np.zeros(4, np.int64), "mod": np.ones(4),
             "arg": np.zeros(4), "line_id": np.array([0, 0, 1, 1])}
    out = ScaleSelect().apply({"extrema": [layer, dict(layer)]}, {**defaults_for(ScaleSelect()), "scale_idx": 1})
    assert out["_ext_base"] is out["extrema"][0]


def test_wtmm_stamps_the_hline_ordering_and_scale_select_picks_it():
    # 2026-08-30: the H-line ordering walk ran on the MAIN thread at every
    # landing -- with noise in the chain every run mints a new result, so it ran every time. The
    # transform now pays the walk ONCE, on the worker, cached with the result; ScaleSelect hands
    # the selected scale's runs to the views.
    import numpy as np
    from pathlib import Path
    from dynamix.core.rasterfield import RasterField
    from dynamix.devices.wtmm import WTMM2D
    from dynamix.devices.filters import ScaleSelect
    from dynamix.model.device import defaults_for
    rng = np.random.default_rng(1)
    field = RasterField._from_bare_array(rng.random((48, 48)), Path("st.npy"))
    params = {**defaults_for(WTMM2D()), "n_oct": 2, "n_voice": 2}
    res = WTMM2D().compute(field, params)
    assert "_hline_runs" in res and len(res["_hline_runs"]) == len(res["extrema"])
    out = ScaleSelect().apply(res, {**defaults_for(ScaleSelect()), "scale_idx": 1})
    assert out["_ext_base_runs"] is res["_hline_runs"][1]


def test_hline_length_reads_its_line_count_and_range():
    """2026-09-15: the H-line LENGTH filter now shows live feedback (it exists as 'Min points';
    the missing reading made it feel absent vs the V-chain filters)."""
    import numpy as np
    from dynamix.core.chain_product import attach_chain_product
    from dynamix.devices.filters import HLineLength
    l0 = {"x": np.arange(6, dtype=np.int64), "y": np.zeros(6, dtype=np.int64),
          "mod": np.ones(6), "arg": np.zeros(6),
          "line_id": np.array([0, 0, 0, 1, 1, -1], np.int64)}
    res = attach_chain_product({"extrema": [l0], "chains": [],
                                "scales": np.asarray([1.0]), "_shape": (8, 64)})
    r = HLineLength().reading(res, {"min_len": 1, "max_len": 0})
    assert "H-lines" in r and "points ∈ [2, 3]" in r        # lines of 3 and 2 points
    assert HLineLength().data_hints(res, {"min_len": 1, "max_len": 0})["min_len"] == (2.0, 3.0, 2.0)
    # no product / no lines -> honest "run the transform", never a crash
    assert "run the transform" in HLineLength().reading({}, {"min_len": 1, "max_len": 0})
