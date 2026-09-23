# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""dynamix.core.hlines — ordered point runs of each H-line at one scale (2026-08-28).

The 2-D canvas joins an H-line's extrema into a polyline (``canvas.hline_polylines``); the 3-D
scene drew them as loose points. This pure helper gives both surfaces one ordering, from the
same ``_order_lines`` walk the topology uses."""
from __future__ import annotations

import numpy as np

from dynamix.core.hlines import hline_runs


def _ext(x, y, line_id):
    n = len(x)
    return {"x": np.asarray(x, np.int64), "y": np.asarray(y, np.int64),
            "mod": np.ones(n), "arg": np.zeros(n), "line_id": np.asarray(line_id, np.int64)}


def test_each_labelled_line_becomes_one_ordered_run_and_singletons_are_excluded():
    # line 0: three points in a row (given out of order); line 1: two points; one isolated
    ext = _ext(x=[5, 3, 4, 10, 11, 20], y=[2, 2, 2, 7, 7, 9], line_id=[0, 0, 0, 1, 1, -1])
    runs = hline_runs(ext, shape=(16, 32))
    assert len(runs) == 2
    assert sorted(len(r) for r in runs) == [2, 3]
    three = next(r for r in runs if len(r) == 3)
    assert list(ext["x"][three]) in ([3, 4, 5], [5, 4, 3])          # ordered along the line
    assert all(5 not in r for r in runs)                             # index 5 is the singleton


def test_empty_or_unlabelled_extrema_give_no_runs():
    assert hline_runs(_ext([], [], []), shape=(4, 4)) == []
    assert hline_runs(_ext([1, 2], [1, 1], [-1, -1]), shape=(4, 4)) == []


def test_a_one_point_line_has_no_edge_and_yields_no_run():
    ext = _ext(x=[1, 3, 4], y=[1, 2, 2], line_id=[0, 1, 1])
    runs = hline_runs(ext, shape=(4, 8))
    assert len(runs) == 1 and len(runs[0]) == 2
