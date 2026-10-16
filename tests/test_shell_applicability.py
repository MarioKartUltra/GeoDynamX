# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The shell reads the contracts: a step that does not apply is drawn inactive with the engine's
reason, and the browser greys what cannot follow the active layer's analysis. Offscreen Qt."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from PySide6 import QtWidgets  # noqa: F401

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.model.chain import DeviceRef
from dynamix.model.contracts import FIELD, StepState, next_step_state, step_states
from dynamix.model.device import get_device
from dynamix.model.families import display_name, tooltip_of
from tests.test_shell_window import _FIELD, stub_devices, window  # noqa: F401

PM_CHAIN = (("pm_edges", {}), ("chain_length", {}))


@pytest.fixture
def pm_window(qtbot, stub_devices):
    """pm_edges reads a RasterField (a bare array is refused as "not the layer's own field")."""
    from dynamix.shell.main_window import MainWindow

    field = RasterField(name="pm", values=np.asarray(_FIELD), frame=LocalFrame(),
                        x_axis=np.arange(16.0), y_axis=np.arange(16.0))
    win = MainWindow(steps=PM_CHAIN)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=20000):
        win.load_field(field, "mem:pm")
    return win


def test_a_filter_that_does_not_apply_reads_the_engines_reason_and_stays_in_the_chain(
        pm_window):
    from dynamix.devices.chain_filters import _UNAVAILABLE

    expected = next_step_state([DeviceRef("pm_edges", {})], FIELD, "chain_length").reason
    box = pm_window.strips.strip(1)
    assert expected and box.inactive_reason == expected
    assert box.reading_label.text() == expected != _UNAVAILABLE
    assert all(c.isEnabled() for c in box.controls.values())     # no NaN-hint greying
    assert pm_window.strips.strip(0).inactive_reason == ""
    assert pm_window._bypassed == [False, False]                  # never bypassed
    assert [s.device for s in pm_window.layer.chain.steps] == ["pm_edges", "chain_length"]


def test_strip_states_map_over_the_enabled_strips_and_skip_bypassed_ones(window, qtbot):
    with qtbot.waitSignal(window.resolved, timeout=10000) as blocker:
        window.load_field(_FIELD, "mem:stub")
    landed = blocker.args[0]
    window._bypassed[1] = True
    states = (StepState(True), StepState(False, "needs a reason"), StepState(True))
    window._update_readings(dataclasses.replace(landed, step_states=states))
    assert [window.strips.strip(i).inactive_reason for i in range(4)] == \
        ["", "", "needs a reason", ""]
    assert window.strips.strip(2).reading_label.text() == "needs a reason"
    # A pending layer resolves the last run's transforms with the current filters, so its strips
    # are judged on their own recipe.
    window._pending_layers.add(window.layer.layer_id)
    window._update_readings(dataclasses.replace(landed, step_states=states))
    refs = [DeviceRef(n, dict(p))
            for n, p, b in zip(window._names, window._params, window._bypassed) if not b]
    expected = step_states(refs, window._layer_input())
    assert [window.strips.strip(i).inactive_reason for i in (0, 2, 3)] == \
        ["" if s.applies else s.reason for s in expected]


def test_the_browser_greys_what_cannot_follow_the_active_layers_analysis(pm_window):
    rows = pm_window.browser._rows
    pm = [DeviceRef("pm_edges", {})]
    chain_holder = rows["chain_holder"][0]
    assert chain_holder.toolTip(0) == next_step_state(pm, FIELD, "chain_holder").reason
    assert chain_holder.foreground(0).color() == pm_window.browser._muted.color()
    assert rows["orientation_wedge"][0].toolTip(0) == \
        tooltip_of(get_device("orientation_wedge"))
    # an analyzer dropped on the master forks a sibling over the field: offered
    assert rows["wtmm2d"][0].toolTip(0) == tooltip_of(get_device("wtmm2d"))
    assert rows["pca"][0].toolTip(0) == next_step_state((), FIELD, "pca").reason


def test_a_filter_edit_lands_without_recomputing_the_browser_states(pm_window, monkeypatch):
    import dynamix.shell.main_window as mw

    calls = []
    real = mw.drop_states
    monkeypatch.setattr(mw, "drop_states", lambda *a, **k: calls.append(a) or real(*a, **k))
    pm_window._params[1] = {**pm_window._params[1], "min_len": 5}
    pm_window._sync_browser_states()
    assert calls == []
    pm_window._params[0] = {**pm_window._params[0], "n_levels": 3}
    pm_window._sync_browser_states()
    assert len(calls) == 1


def test_a_landing_re_greys_the_browser_through_the_one_resolved_connection(pm_window, qtbot):
    from PySide6 import QtCore

    calls = []
    pm_window._sync_browser_states = lambda: calls.append(True)
    with qtbot.waitSignal(pm_window.resolved, timeout=10000):
        pm_window._reresolve()
    assert calls
    assert pm_window.receivers(QtCore.SIGNAL("resolved(PyObject)")) == 1
