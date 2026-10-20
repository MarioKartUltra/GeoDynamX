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
    # Downstream filters mint a NEW extrema dict per tweak, so geometry caches keyed on identity
    # always miss. ScaleSelect stamps the id-stable UNFILTERED layer so the canvas can order
    # H-lines once and mask per tweak.
    import numpy as np
    from dynamix.devices.filters import ScaleSelect
    from dynamix.model.device import defaults_for
    layer = {"x": np.arange(4), "y": np.zeros(4, np.int64), "mod": np.ones(4),
             "arg": np.zeros(4), "line_id": np.array([0, 0, 1, 1])}
    out = ScaleSelect().apply({"extrema": [layer, dict(layer)]}, {**defaults_for(ScaleSelect()), "scale_idx": 1})
    assert out["_ext_base"] is out["extrema"][0]


def test_wtmm_stamps_the_hline_ordering_and_scale_select_picks_it():
    # The transform pays the H-line ordering walk ONCE, on the worker, cached with the result;
    # ScaleSelect hands the selected scale's runs to the views. On the MAIN thread the walk would
    # run at every landing -- with noise in the chain every run mints a new result.
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
    """The H-line LENGTH filter ('Min points') shows live feedback as the V-chain filters do: a
    reading of its line count and point range."""
    import numpy as np
    from dynamix.core.chain_product import attach_chain_product
    from dynamix.devices.filters import HLineLength
    l0 = {"x": np.arange(6, dtype=np.int64), "y": np.zeros(6, dtype=np.int64),
          "mod": np.ones(6), "arg": np.zeros(6),
          "line_id": np.array([0, 0, 0, 1, 1, -1], np.int64)}
    res = attach_chain_product({"extrema": [l0], "chains": [],
                                "scales": np.asarray([1.0]), "_shape": (8, 64)})
    r = HLineLength().reading(res, {"min_len": 1, "max_len": 0})
    assert "H-chains" in r and "points ∈ [2, 3]" in r        # lines of 3 and 2 points
    assert HLineLength().data_hints(res, {"min_len": 1, "max_len": 0})["min_len"] == (2.0, 3.0, 2.0)
    # no product / no lines -> honest "run the transform", never a crash
    assert "run the transform" in HLineLength().reading({}, {"min_len": 1, "max_len": 0})


def test_hline_runs_equal_the_per_line_walk_and_cost_one_pass():
    """The ordered runs of every H-line at a scale, split where the walk restarts, computed in
    one pass over the ordering instead of one numpy split per line (the per-line loop cost a
    second per run on a 1024² field with 58 000 lines). The reference is the per-line form."""
    import numpy as np
    from dynamix.core.hlines import hline_runs
    from dynamix.core.wtmm_backend import _order_lines

    def reference(ext, shape):
        x = np.asarray(ext["x"], dtype=np.int64); y = np.asarray(ext["y"], dtype=np.int64)
        line_id = np.asarray(ext["line_id"], dtype=np.int64)
        ny, nx = shape
        grid = np.full(ny * nx, -1, dtype=np.int64); grid[y * nx + x] = np.arange(x.size)
        order, seg, starts = _order_lines(x, y, line_id, grid, nx, ny)
        runs = []
        for li in range(len(starts) - 1):
            sl = order[starts[li]:starts[li + 1]]
            if sl.size == 0:
                continue
            breaks = np.nonzero(np.diff(seg[starts[li]:starts[li + 1]]))[0] + 1
            runs.extend(w for w in np.split(sl, breaks) if w.size >= 2)
        return runs

    rng = np.random.default_rng(3)
    ny, nx = 96, 128
    # lines: random walks on the grid, plus singletons and a branching line
    xs, ys, ids = [], [], []
    lid = 0
    for _ in range(40):
        x0, y0 = rng.integers(2, nx - 2), rng.integers(2, ny - 2)
        n = rng.integers(2, 30)
        px, py = [x0], [y0]
        for _k in range(n - 1):
            dx, dy = rng.integers(-1, 2), rng.integers(-1, 2)
            nxp, nyp = min(max(px[-1] + dx, 0), nx - 1), min(max(py[-1] + dy, 0), ny - 1)
            if (nxp, nyp) != (px[-1], py[-1]):
                px.append(nxp); py.append(nyp)
        xs += px; ys += py; ids += [lid] * len(px); lid += 1
    for _ in range(15):
        xs.append(int(rng.integers(0, nx))); ys.append(int(rng.integers(0, ny))); ids.append(-1)
    ext = {"x": np.asarray(xs), "y": np.asarray(ys), "line_id": np.asarray(ids)}
    # duplicates on the grid: keep the first occurrence of a pixel
    _, keep = np.unique(ext["y"] * nx + ext["x"], return_index=True)
    ext = {k: v[np.sort(keep)] for k, v in ext.items()}
    got, want = hline_runs(ext, (ny, nx)), reference(ext, (ny, nx))
    assert len(got) == len(want) > 10
    for a, b in zip(got, want):
        np.testing.assert_array_equal(a, b)
    assert hline_runs({"x": [], "y": [], "line_id": []}, (ny, nx)) == []
