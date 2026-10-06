# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Per-level output rows in the window: wtmm2d's modulus and argument rows are off by default and
request nothing; an on row shows the level the Transport is on, each level computed once; a
marked level is pinned in the cache (it survives the byte budget), unmarking or turning the row
off releases it, and the derived-raster export names the level and the marked levels. The window's
cache takes its byte budget from Settings."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.resolve import resolve_output
from dynamix.model.device import defaults_for, get_device

FAST = {"n_oct": 1, "n_voice": 3}


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


@pytest.fixture
def win(qtbot, builtins):
    from dynamix.shell.main_window import MainWindow

    v = np.random.default_rng(4).standard_normal((40, 50)).cumsum(0).cumsum(1)
    f = RasterField(name="plain", values=v, frame=LocalFrame(), x_axis=np.arange(50.0),
                    y_axis=np.arange(40.0))
    w = MainWindow(steps=())
    qtbot.addWidget(w)
    w.load_field(f, "/nonexistent/plain.npz")
    qtbot.waitUntil(lambda: not w.is_computing, timeout=30000)
    return w


@pytest.fixture
def dispatches(monkeypatch):
    """The names of the output jobs the window starts, in order."""
    import dynamix.shell.main_window as mw

    seen: list = []
    real = mw.OutputWorker

    class _Counted(real):
        def __init__(self, layer, field, cache, source_id, name, **kw):
            seen.append(name)
            super().__init__(layer, field, cache, source_id, name, **kw)

    monkeypatch.setattr(mw, "OutputWorker", _Counted)
    return seen


def _wtmm_layer(win, qtbot):
    """Drop wtmm2d on the dataset (it spawns a child one tick later) and let it land."""
    desc = [{"device": "wtmm2d", "params": {**defaults_for(get_device("wtmm2d")), **FAST}}]
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.strips.set_steps(desc, field=win.field)
        win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    return win.layer


def _idle(win, qtbot):
    qtbot.waitUntil(lambda: not win.is_computing and win._out_request is None, timeout=60000)


def _key(win, layer, name, level):
    return win._output_keys(layer, level=level)[name]


def _show(win, qtbot, layer, name):
    win.layer_list.outputHideToggled.emit(layer.layer_id, name, False)
    _idle(win, qtbot)


def _level(win, qtbot, k):
    """The Transport moves to level ``k`` on the active layer."""
    win._on_scale_changed(k)
    _idle(win, qtbot)


def _row_text(win, layer, name):
    return win.layer_list._output_items[layer.layer_id][name][0].text(0)


def test_the_rows_are_off_and_request_nothing(win, qtbot, dispatches):
    layer = _wtmm_layer(win, qtbot)
    rows = win.layer_list._output_rows[layer.layer_id]
    assert list(rows) == ["modulus", "argument"]
    assert all(row.hide_button.isChecked() for row in rows.values())
    _idle(win, qtbot)
    assert dispatches == []
    assert _key(win, layer, "modulus", 0) not in win.cache


def test_an_on_row_shows_the_transport_level(win, qtbot, dispatches):
    layer = _wtmm_layer(win, qtbot)
    _show(win, qtbot, layer, "modulus")
    k0 = _key(win, layer, "modulus", 0)
    qtbot.waitUntil(lambda: k0 in win.cache, timeout=60000)
    _level(win, qtbot, 2)
    k2 = _key(win, layer, "modulus", 2)
    qtbot.waitUntil(lambda: k2 in win.cache and not win.is_computing, timeout=60000)
    assert dispatches == ["modulus", "modulus"]
    shown = win._derived_fields[layer.layer_id].values
    np.testing.assert_array_equal(shown, win.cache.get(k2)["raster"].astype(np.float64))
    want = resolve_output(layer, win.field, win.cache, "modulus", level=2)
    assert win.cache.get(k2) is want                       # the window's key is the engine's
    assert "px" in _row_text(win, layer, "modulus")
    _level(win, qtbot, 0)                                  # back to a computed level: a hit
    assert dispatches == ["modulus", "modulus"]


def test_the_output_worker_computes_the_level_it_is_handed(qtbot, builtins):
    from dynamix.engine.cache import Cache
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.layer import Layer
    from dynamix.shell.worker import OutputWorker

    v = np.random.default_rng(4).standard_normal((40, 50)).cumsum(0).cumsum(1)
    field = RasterField(name="plain", values=v, frame=LocalFrame(), x_axis=np.arange(50.0),
                        y_axis=np.arange(40.0))
    chain = Chain((DeviceRef("wtmm2d", dict(FAST)),)).materialized()
    layer = Layer(layer_id=1, name="w", source_id="mem:out", chain=chain)
    landed = []
    worker = OutputWorker(layer, field, Cache(), "mem:out", "modulus", level=2)
    worker.finished.connect(landed.append)
    worker._run()                              # on this thread: the signal is delivered directly

    (value,) = landed
    assert value["diag"]["level"] == 2
    assert "ui.scale_idx" not in layer.tags    # the worker never writes view state


def test_the_window_cache_takes_its_byte_budget_from_settings(qtbot, builtins):
    from dynamix.shell.main_window import MainWindow
    from dynamix.shell.settings import update_settings

    w = MainWindow(steps=())
    qtbot.addWidget(w)
    assert w.cache.max_bytes == 2_000_000_000
    update_settings(output_cache_bytes=123_456_789)
    w2 = MainWindow(steps=())
    qtbot.addWidget(w2)
    assert w2.cache.max_bytes == 123_456_789


def _mark(win, layer, name):
    """Trigger the row's "Mark level" action; returns the action's text."""
    menu = win.layer_list._build_output_context_menu(layer.layer_id, name)
    action = menu.actions()[0]
    text = action.text()
    action.trigger()
    return text


def test_mark_pins_the_level_and_unmark_releases_it(win, qtbot):
    layer = _wtmm_layer(win, qtbot)
    _show(win, qtbot, layer, "modulus")
    _level(win, qtbot, 1)
    k1 = _key(win, layer, "modulus", 1)
    qtbot.waitUntil(lambda: k1 in win.cache, timeout=60000)
    assert _mark(win, layer, "modulus") == "Mark level 2 for figures"
    assert layer.tags["ui.marked_levels.modulus"] == "1"
    assert win.cache.is_pinned(k1)
    assert "marked" in _row_text(win, layer, "modulus")
    assert _mark(win, layer, "modulus") == "Unmark level 2 for figures"
    assert "ui.marked_levels.modulus" not in layer.tags
    assert not win.cache.is_pinned(k1)


def test_turning_the_row_off_releases_the_pins_and_keeps_the_marks(win, qtbot):
    layer = _wtmm_layer(win, qtbot)
    _show(win, qtbot, layer, "modulus")
    k0 = _key(win, layer, "modulus", 0)
    qtbot.waitUntil(lambda: k0 in win.cache, timeout=60000)
    _mark(win, layer, "modulus")
    assert win.cache.is_pinned(k0)
    win.layer_list.outputHideToggled.emit(layer.layer_id, "modulus", True)
    _idle(win, qtbot)
    assert not win.cache.is_pinned(k0) and layer.tags["ui.marked_levels.modulus"] == "0"
    _show(win, qtbot, layer, "modulus")
    assert win.cache.is_pinned(k0)


def test_a_marked_level_survives_the_byte_budget(win, qtbot):
    layer = _wtmm_layer(win, qtbot)
    _show(win, qtbot, layer, "modulus")
    k0 = _key(win, layer, "modulus", 0)
    qtbot.waitUntil(lambda: k0 in win.cache, timeout=60000)
    _mark(win, layer, "modulus")
    win.cache.max_bytes = 1                                # every unpinned output is over
    _level(win, qtbot, 1)
    k1 = _key(win, layer, "modulus", 1)
    qtbot.waitUntil(lambda: k1 in win.cache and not win.is_computing, timeout=60000)
    _level(win, qtbot, 2)
    k2 = _key(win, layer, "modulus", 2)
    qtbot.waitUntil(lambda: k2 in win.cache and not win.is_computing, timeout=60000)
    assert k0 in win.cache                                 # marked: pinned above the budget
    assert k1 not in win.cache                             # evicted by the newer level


def test_the_export_names_the_level_and_the_marked_levels(win, qtbot):
    layer = _wtmm_layer(win, qtbot)
    _show(win, qtbot, layer, "modulus")
    _mark(win, layer, "modulus")
    _level(win, qtbot, 2)
    k2 = _key(win, layer, "modulus", 2)
    qtbot.waitUntil(lambda: k2 in win.cache and not win.is_computing, timeout=60000)
    _mark(win, layer, "modulus")
    prov = win._derived_export_field(layer).provenance
    assert prov["output"] == "wtmm2d/modulus"
    assert prov["level"] == 2 and prov["marked_levels"] == [0, 2]
