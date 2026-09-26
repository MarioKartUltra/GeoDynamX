# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Live layer sends: select layers, "Build band stack", and the stack follows them -- what
each layer shows, referenced, never copied."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PySide6 import QtWidgets  # noqa: F401

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.model.chain import Chain, DeviceRef
from tests.test_shell_window import stub_devices, window  # noqa: F401


def _noise_child(win, master, field, seed):
    chain = Chain((DeviceRef("noise", {"amplitude": 3.0, "seed": seed}),)).materialized()
    lay = win.project.add_layer(f"noise s{seed}", master.source_id, chain,
                                parent_id=master.layer_id)
    win.add_layer_row(lay, field)
    return lay


def _noised(field, seed):
    from dynamix.model.device import defaults_for, get_device

    dev = get_device("noise")
    return dev.compute(field, {**defaults_for(dev), "amplitude": 3.0, "seed": seed}).values


@pytest.fixture
def dem(window, qtbot):
    win = window
    v = np.random.default_rng(1).normal(size=(24, 28)).cumsum(0)
    field = RasterField(name="dem", values=v, frame=LocalFrame(), x_axis=np.arange(28.0),
                        y_axis=np.arange(24.0))
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(field, "mem:dem")
    master = win.layer
    a = _noise_child(win, master, field, 1)
    b = _noise_child(win, master, field, 2)
    return win, master, field, a, b


def _build(win, qtbot, ids):
    with qtbot.waitSignal(win.resolved, timeout=30000):
        win._on_stack_requested(ids)
    return win.layer


def test_a_stack_references_what_each_layer_shows(dem, qtbot):
    win, master, field, a, b = dem
    stack = _build(win, qtbot, [a.layer_id, b.layer_id])
    assert stack.parent_id == master.layer_id and stack.chain.steps[0].device == "bus"
    sends = json.loads(stack.chain.steps[0].params["_sends"])
    assert [s["layer"] for s in sends] == [a.layer_id, b.layer_id]
    shown = np.asarray(win.canvas._field.values)
    assert shown.shape == (24, 28, 2)
    np.testing.assert_array_equal(shown[..., 0], _noised(field, 1))
    np.testing.assert_array_equal(shown[..., 1], _noised(field, 2))
    assert len(win.project.layers) == 4                  # the stack is the ONE new row


def test_the_stack_follows_a_referenced_layer_live(dem, qtbot):
    win, master, field, a, b = dem
    stack = _build(win, qtbot, [a.layer_id, b.layer_id])
    before = stack.chain.steps[0].params["_sends"]
    a.chain = Chain((DeviceRef("noise", {"amplitude": 3.0, "seed": 7}),)).materialized()
    win.layer_list.select_layer(master.layer_id)
    with qtbot.waitSignal(win.resolved, timeout=30000):
        win.layer_list.select_layer(stack.layer_id)
    assert stack.chain.steps[0].params["_sends"] != before     # the key moved with it
    np.testing.assert_array_equal(np.asarray(win.canvas._field.values)[..., 0],
                                  _noised(field, 7))


def test_one_layer_or_an_roi_result_is_refused(dem, qtbot):
    win, master, field, a, b = dem
    notes = []
    win._notify = lambda msg, *a_, **k: notes.append(msg)
    n = len(win.project.layers)
    win._on_stack_requested([a.layer_id])
    assert len(win.project.layers) == n and "two or more" in notes[-1]
    b.tags["roi.window"] = "2,2,8,8"
    win._on_stack_requested([a.layer_id, b.layer_id])
    assert len(win.project.layers) == n and "ROI result" in notes[-1]


def test_the_multi_selection_menu_builds_a_stack_and_deletes(dem, qtbot):
    win, master, field, a, b = dem
    lp = win.layer_list
    items = [lp._layer_items[a.layer_id], lp._layer_items[b.layer_id]]
    texts = [x.text() for x in lp._build_multi_context_menu(items).actions()]
    assert texts == ["Build band stack from 2 selected layers", "Delete 2 selected rows"]
    with qtbot.waitSignal(lp.stackRequested) as blocker:
        lp._build_multi_context_menu(items).actions()[0].trigger()
    assert blocker.args == [[a.layer_id, b.layer_id]]


def test_a_deleted_referenced_layer_is_named_not_silently_dropped(dem, qtbot):
    win, master, field, a, b = dem
    stack = _build(win, qtbot, [a.layer_id, b.layer_id])
    from dynamix.core import bus

    bus._PLANES.clear()                                    # force the recompute path
    win.project.remove_layer(a.layer_id)
    win._layer_by_id.pop(a.layer_id, None)
    errors = []
    win._report_error = errors.append
    win._start_worker_for(stack)
    assert errors and "noise s1" in errors[0] and "deleted" in errors[0]


def test_a_routing_loop_is_refused(dem):
    win, master, field, a, b = dem
    loop_a = Chain((DeviceRef("bus", {"_sends": json.dumps(
        [{"layer": b.layer_id, "label": "b", "stamp": "x"}])}),)).materialized()
    loop_b = Chain((DeviceRef("bus", {"_sends": json.dumps(
        [{"layer": a.layer_id, "label": "a", "stamp": "y"}])}),)).materialized()
    a.chain, b.chain = loop_a, loop_b
    with pytest.raises(ValueError, match="routing loop"):
        win._bus_plan(a)
