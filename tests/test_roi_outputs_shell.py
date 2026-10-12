# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Lazy outputs on an ROI result in the window: computed on Run, never on show, and drawn at
the region."""
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
    w = MainWindow(steps=())
    qtbot.addWidget(w)
    v = np.random.default_rng(7).standard_normal((128, 128)).cumsum(0).cumsum(1)
    w.load_field(RasterField(name="fbm", values=v, frame=LocalFrame(),
                             x_axis=np.arange(128.0), y_axis=np.arange(128.0)),
                 "/nonexistent/fbm.npz")
    qtbot.waitUntil(lambda: not w.is_computing, timeout=30000)
    return w


@pytest.fixture
def dispatches(monkeypatch):
    import dynamix.shell.main_window as mw

    seen: list = []
    real = mw.OutputWorker

    class _Counted(real):
        def __init__(self, layer, field, cache, source_id, name, **kw):
            seen.append(name)
            super().__init__(layer, field, cache, source_id, name, **kw)

    monkeypatch.setattr(mw, "OutputWorker", _Counted)
    return seen


def _roi_tool(win, qtbot, device, **params):
    win._on_roi_save({"roi_row": 40, "roi_col": 44, "roi_h": 32, "roi_w": 32,
                      "boundary": "auto"})
    desc = [{"device": device, "params": {**defaults_for(get_device(device)), **params}}]
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.strips.set_steps(desc, field=win.field)
        win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    return win.layer


def _idle(win):
    return not win.is_computing and win._out_request is None


def test_an_roi_output_waits_for_run_then_lands_at_the_region(win, qtbot, dispatches):
    layer = _roi_tool(win, qtbot, "mz_edges", n_levels=3, recon_live=True)
    assert layer.tags.get("roi.window") == "40,44,32,32"
    win.layer_list.outputHideToggled.emit(layer.layer_id, "recon", False)
    qtbot.wait(200)
    assert dispatches == [] and _idle(win)                 # showing computes nothing
    notes, disabled = win._output_row_state(layer, *_step_outputs(win, layer))
    assert notes.get("recon") == "press Run" and "recon" not in disabled
    assert win.run_button.isEnabled()
    win._run_transforms()                                  # Run asks for the shown output
    key = win._output_keys(layer)["recon"]
    qtbot.waitUntil(lambda: key in win.cache and _idle(win), timeout=60000)
    assert dispatches == ["recon"]
    value = win.cache.get(key)
    np.testing.assert_array_equal(win.canvas._field.values, value["raster"])
    rect = win.canvas.image_item.mapRectToView(win.canvas.image_item.boundingRect())
    assert (rect.left(), rect.top(), rect.width(), rect.height()) == pytest.approx(
        (43.5, 39.5, 32, 32))


def test_another_row_waits_for_its_own_run(win, qtbot, dispatches):
    layer = _roi_tool(win, qtbot, "mz_edges", n_levels=3)
    win.layer_list.outputHideToggled.emit(layer.layer_id, "recon", False)
    win._run_transforms()
    keys = win._output_keys(layer)
    qtbot.waitUntil(lambda: keys["recon"] in win.cache and _idle(win), timeout=60000)
    win.layer_list.outputHideToggled.emit(layer.layer_id, "coarse", False)
    qtbot.wait(200)
    assert dispatches == ["recon"] and keys["coarse"] not in win.cache
    notes, _disabled = win._output_row_state(layer, *_step_outputs(win, layer))
    assert notes.get("coarse") == "press Run"
    win._run_transforms()
    qtbot.waitUntil(lambda: keys["coarse"] in win.cache and _idle(win), timeout=60000)
    assert dispatches == ["recon", "coarse"]


def test_a_tool_with_its_own_region_path_greys_its_outputs_with_the_reason(win, qtbot):
    layer = _roi_tool(win, qtbot, "wtmm2d")
    _notes, disabled = win._output_row_state(layer, *_step_outputs(win, layer))
    assert disabled.get("modulus") == "not on ROI results yet"


def test_an_output_on_its_own_coarser_grid_is_greyed_on_a_region(win, qtbot):
    layer = _roi_tool(win, qtbot, "mz_edges", n_levels=3)
    _notes, disabled = win._output_row_state(layer, *_step_outputs(win, layer))
    assert disabled.get("thumbnail") == "not on ROI results"
    assert "recon" not in disabled and "coarse" not in disabled


def _step_outputs(win, layer):
    from dynamix.shell.layer_panel import layer_outputs

    return layer_outputs(layer, win._layer_input(layer))
