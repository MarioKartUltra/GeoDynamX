# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Removing a layer lets go of what it held: the window's own copy of its last result, and the
cache pins its freeze put on its keys, except a key another frozen layer still holds."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.model.device import defaults_for, get_device

FAST = {"n_oct": 1, "n_voice": 3}


@pytest.fixture
def win(qtbot, clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.shell.main_window import MainWindow

    register_builtin_devices()
    v = np.random.default_rng(4).standard_normal((40, 50)).cumsum(0).cumsum(1)
    f = RasterField(name="plain", values=v, frame=LocalFrame(), x_axis=np.arange(50.0), y_axis=np.arange(40.0))
    w = MainWindow(steps=())
    qtbot.addWidget(w)
    w.load_field(f, "/nonexistent/plain.npz")
    qtbot.waitUntil(lambda: not w.is_computing, timeout=30000)
    return w


def _wtmm_layer(win, qtbot):
    desc = [{"device": "wtmm2d", "params": {**defaults_for(get_device("wtmm2d")), **FAST}}]
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.strips.set_steps(desc, field=win.field)
        win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    return win.layer


def test_a_removed_layer_s_last_result_is_let_go(win, qtbot):
    layer = _wtmm_layer(win, qtbot)
    assert layer.layer_id in win._landed_results
    win._on_remove_requested(layer.layer_id, confirm=False)
    assert layer.layer_id not in win._landed_results


def test_a_removed_frozen_layer_s_pins_are_released_unless_another_frozen_layer_holds_them(win, qtbot):
    layer = _wtmm_layer(win, qtbot)
    win._on_freeze_toggled(layer.layer_id, True)
    keys = win._cache_keys_for(layer)
    assert keys and all(win.cache.is_pinned(k) for k in keys)
    twin = win.project.add_layer("twin", layer.source_id, layer.chain)        # the same chain: the same keys
    twin.tags["ui.freeze"] = "1"
    win._layer_by_id[twin.layer_id] = twin
    win._on_remove_requested(layer.layer_id, confirm=False)
    assert all(win.cache.is_pinned(k) for k in keys)                          # the twin still holds them
    win._on_remove_requested(twin.layer_id, confirm=False)
    assert not any(win.cache.is_pinned(k) for k in keys)
