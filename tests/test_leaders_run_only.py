# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The wavelet leaders wait for Run whatever the auto-run setting: dropping the tool or changing a
box marks the layer pending, Run computes the ticked quantities, and an unticked quantity's
per-level row is greyed with what to tick."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.model.device import defaults_for, get_device


@pytest.fixture
def win(qtbot, clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.shell.main_window import MainWindow

    register_builtin_devices()
    v = np.random.default_rng(5).standard_normal((256, 256)).cumsum(0).cumsum(1)
    f = RasterField(name="plain", values=v, frame=LocalFrame(), x_axis=np.arange(256.0), y_axis=np.arange(256.0))
    w = MainWindow(steps=())
    qtbot.addWidget(w)
    w.load_field(f, "/nonexistent/plain.npz")
    qtbot.waitUntil(lambda: not w.is_computing, timeout=30000)
    w._auto_run_action.setChecked(True)
    return w


def test_the_leaders_wait_for_run_and_grey_the_unticked_quantities_rows(win, qtbot):
    from dynamix.shell.layer_panel import layer_outputs

    desc = [{"device": "wavelet_leaders", "params": {**defaults_for(get_device("wavelet_leaders")), "scale_max": 32}}]
    win.strips.set_steps(desc, field=win.field)
    win._on_chain_edited(desc)
    qtbot.wait(200)
    assert not win.is_computing and win.layer.layer_id in win._pending_layers
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.run_button.click()
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    assert set(win._active_result["samples"]["sets"]) == {"leaders", "p-leaders", "coefficients"}
    layer = win.layer
    _notes, disabled = win._output_row_state(layer, *layer_outputs(layer, win._layer_input()))
    assert "WSE leaders" in disabled.get("wse", "") and "WSE leaders" in disabled.get("oscillation", "")
    assert "leader" not in disabled and "p_leader" not in disabled
