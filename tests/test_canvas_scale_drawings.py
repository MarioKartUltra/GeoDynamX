# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A scale scrub draws a scale it has drawn before from the canvas's table of the result's
drawings: the capped H-lines, the maxima masks and the capped V-trails are built once per scale
of a resolved result, and a filter change or a new result starts the table again."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.rasterfield import RasterField

FIXTURE = "tests/fixtures/kam_64.npz"


@pytest.fixture
def scrubbed(qtbot, clean_registry, monkeypatch):
    """A computed demo window with a draw cap low enough to withhold lines, every scale visited
    once, and a counter on ``cap_polylines``."""
    from dynamix.devices import register_builtin_devices
    from dynamix.shell import canvas as canvas_mod
    from dynamix.shell.main_window import DEMO_CHAIN, MainWindow

    register_builtin_devices()
    win = MainWindow(steps=DEMO_CHAIN)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.load_field(RasterField.load_npz(FIXTURE), FIXTURE)
    qtbot.waitUntil(lambda: win._thread is None, timeout=60000)
    calls = []
    real = canvas_mod.cap_polylines
    monkeypatch.setattr(canvas_mod, "cap_polylines", lambda *a: (calls.append(a), real(*a))[1])
    win.canvas.draw_cap = 3
    n = win.transport.slider.maximum() + 1
    assert n > 1
    for i in range(n):
        win.transport._set_index(i)
    assert calls
    return win, n, calls


def _drawn(canvas):
    items = (canvas.extrema_item, canvas.hchain_item, canvas.hchain_neg_item, canvas.vtrail_item)
    arrays = [np.empty(0) if a is None else a for item in items for a in item.getData()]
    arrays += [canvas.extrema_raster_item.image, canvas.isolated_raster_item.image]
    return arrays, canvas.cap_note


def _fresh(win):
    """What the canvas draws for the result on show with nothing memoized to read from."""
    win.canvas._geom_cache.clear()
    win.canvas._geom_refs.clear()
    unstamped = {k: v for k, v in win._active_result.items() if k != "_landed"}
    win.canvas.set_result(unstamped, 0)
    return _drawn(win.canvas)


def _same(a, b):
    (arrays_a, note_a), (arrays_b, note_b) = a, b
    return note_a == note_b and len(arrays_a) == len(arrays_b) and all(
        np.array_equal(x, y, equal_nan=x.dtype.kind == "f") for x, y in zip(arrays_a, arrays_b))


def test_a_second_visit_to_a_scale_caps_no_polylines(scrubbed):
    win, n, calls = scrubbed
    first = len(calls)
    for i in reversed(range(n)):
        win.transport._set_index(i)
    assert len(calls) == first


def test_a_revisited_scale_draws_what_a_fresh_build_draws(scrubbed):
    win, n, calls = scrubbed
    assert win.canvas.cap_note                        # the cap withheld lines on this field
    for i in reversed(range(n)):
        win.transport._set_index(i)
        assert _same(_drawn(win.canvas), _fresh(win)), f"scale {i}"


def test_a_filter_change_or_a_new_result_starts_the_table_again(scrubbed):
    win, n, calls = scrubbed
    before = len(calls)
    win._on_param_changed(win._index_of("modulus_threshold"), "frac", 0.3)
    assert len(calls) > before                        # the new filter state is built
    assert _same(_drawn(win.canvas), _fresh(win))
    before = len(calls)
    win.transport._set_index(0)                       # drawn before the change: built again
    assert len(calls) > before
    assert _same(_drawn(win.canvas), _fresh(win))
    before = len(calls)
    result = win._active_result
    win.canvas.set_result({**result, "_ext_base": dict(result["_ext_base"])}, 0)
    assert len(calls) > before                        # the same recipe recomputed: another result
