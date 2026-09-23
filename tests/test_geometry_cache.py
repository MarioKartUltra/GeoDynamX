# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Canvas overlay-geometry memoization -- scrub cost must not rebuild unchanged polylines."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.shell import canvas as canvas_mod
from dynamix.shell.canvas import Canvas


@pytest.fixture
def cv(qtbot):
    c = Canvas()
    qtbot.addWidget(c)
    c.set_field(np.zeros((64, 64)))
    return c


def _ext(n_lines=40, pts=25):
    lid = np.repeat(np.arange(n_lines), pts)
    n = lid.size
    rng = np.random.default_rng(0)
    return {"x": (np.arange(n) % 64).astype(np.int64),
            "y": (np.arange(n) // 64 % 64).astype(np.int64),
            "mod": rng.random(n), "arg": rng.uniform(-np.pi, np.pi, n),
            "line_id": lid.astype(np.int64)}


def _result(ext, chains=None):
    return {"extrema": [ext], "_shape": (64, 64), "chains": chains or [],
            "params": {}, "scales": np.array([1.0])}


def test_same_result_objects_do_not_rebuild_geometry(cv, monkeypatch):
    calls = {"hline": 0, "trails": 0}
    real_h, real_v = canvas_mod.hline_polylines, canvas_mod.vchain_trails

    def counting_h(*a, **k):
        calls["hline"] += 1
        return real_h(*a, **k)

    def counting_v(*a, **k):
        calls["trails"] += 1
        return real_v(*a, **k)

    monkeypatch.setattr(canvas_mod, "hline_polylines", counting_h)
    monkeypatch.setattr(canvas_mod, "vchain_trails", counting_v)
    ext = _ext()
    chains = [{"x": np.arange(5), "y": np.arange(5), "mod": np.ones(5),
               "log2_mod": np.zeros(5), "log2_scales": np.arange(5.0)}]
    res = _result(ext, chains)
    cv.set_result(res, 0)
    first = dict(calls)
    for _ in range(10):                      # ten scrub ticks over the same cached objects
        cv.set_result(_result(ext, chains), 0)
    assert calls == first                    # zero additional geometry builds


def test_new_objects_do_rebuild(cv):
    a, b = _ext(), _ext()
    cv.set_result(_result(a), 0)
    xa = cv.hchain_item.getData()[0].copy()
    b["x"] = (b["x"] + 1) % 64               # different geometry, different object
    cv.set_result(_result(b), 0)
    assert not np.array_equal(xa, cv.hchain_item.getData()[0])


def test_cache_is_capped_and_keeps_strong_refs(cv):
    for i in range(80):
        cv.set_result(_result(_ext(n_lines=2, pts=3)), 0)
    assert len(cv._geom_cache) <= Canvas._GEOMETRY_CACHE_MAX * 3   # three kinds share the cap sweep
    assert set(cv._geom_cache) == set(cv._geom_refs)


def test_group_paint_noop_preserves_chains_identity_for_the_scrub_cache(cv, monkeypatch):
    """``GroupPaint.apply`` used to copy ``result["chains"]``
    unconditionally, even when nothing was actually painted this redraw -- breaking exactly this
    cache. A committed layer with nothing new to paint (every referenced group empty here) must
    still be a zero-extra-geometry-build scrub, same as any other unchanged result."""
    from dynamix.devices.groups import GroupPaint, encode_groups

    calls = {"trails": 0}
    real_v = canvas_mod.vchain_trails

    def counting_v(*a, **k):
        calls["trails"] += 1
        return real_v(*a, **k)

    monkeypatch.setattr(canvas_mod, "vchain_trails", counting_v)
    ext = _ext()
    chains = [{"x": np.arange(5), "y": np.arange(5), "mod": np.ones(5),
               "log2_mod": np.zeros(5), "log2_scales": np.arange(5.0)}]
    base = _result(ext, chains)
    spec_json = encode_groups({"g": {"signature": "s", "chains": [], "color": [0, 0, 0]}})
    params = {"spec_json": spec_json, "signature": "s"}

    cv.set_result(GroupPaint().apply(base, params), 0)
    first = dict(calls)
    for _ in range(10):                      # ten scrub ticks, a FRESH apply() call every time
        cv.set_result(GroupPaint().apply(base, params), 0)
    assert calls == first                    # zero additional geometry builds


def test_offsets_stay_outside_the_cache(cv):
    """An ROI result re-rendered under a different display offset must move, not reuse stale
    translated arrays -- the cache stores result-space geometry only."""
    ext = _ext(n_lines=3, pts=4)
    res = _result(ext)
    cv.set_result(res, 0)
    base_x = cv.hchain_item.getData()[0].copy()
    shifted = dict(res)
    shifted["_roi"] = {"roi": (7, 11, 64, 64)}      # display_offset now (7, 11)
    cv.set_result(shifted, 0)
    np.testing.assert_allclose(cv.hchain_item.getData()[0], base_x + 11)
