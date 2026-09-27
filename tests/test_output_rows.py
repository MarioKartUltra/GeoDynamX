# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Output rows under a cdf_edges / pm_edges result: a collapsed "Outputs" group whose raster
rows are a radio over the step's view-only ``show`` param and whose "edges" row hides the maxima
(the ``ui.edges_hidden`` tag) while the result keeps them."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.model.device import defaults_for, get_device
from dynamix.model.project import Project


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


@pytest.fixture
def win(qtbot, builtins):
    from dynamix.shell.main_window import MainWindow

    v = np.random.default_rng(4).standard_normal((40, 50)).cumsum(0)
    f = RasterField(name="plain", values=v, frame=LocalFrame(), x_axis=np.arange(50.0),
                    y_axis=np.arange(40.0))
    w = MainWindow(steps=())
    qtbot.addWidget(w)
    w.load_field(f, "/nonexistent/plain.npz")
    qtbot.waitUntil(lambda: not w.is_computing, timeout=30000)
    return w


def _tool_layer(win, qtbot, device, **params):
    """Drop ``device`` on the dataset (it spawns a child one tick later) and let it land."""
    desc = [{"device": device, "params": {**defaults_for(get_device(device)), **params}}]
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.strips.set_steps(desc, field=win.field)
        win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    return win.layer


def _row_names(panel, layer_id):
    group = panel._output_groups[layer_id]
    return [group.child(i).text(0) for i in range(group.childCount())]


def _hidden(panel, layer_id, name):
    return panel._output_rows[layer_id][name].hide_button.isChecked()


def _show(layer):
    return next(s.params["show"] for s in layer.chain.steps if s.device in ("cdf_edges",
                                                                            "pm_edges"))


def test_a_cdf_layer_gets_a_collapsed_outputs_group_and_a_bare_dataset_none(win, qtbot):
    from dynamix.shell.layer_panel import _LAYER_ID_ROLE

    master = win.layer
    child = _tool_layer(win, qtbot, "cdf_edges", n_levels=2)
    panel = win.layer_list
    item = panel._layer_items[child.layer_id]
    group = item.child(0)
    assert group is panel._output_groups[child.layer_id]
    assert group.text(0) == "Outputs" and not group.isExpanded()
    assert _row_names(panel, child.layer_id) == ["edges", "filtered", "edge channel"]
    for i in range(group.childCount()):
        assert group.child(i).data(0, _LAYER_ID_ROLE) is None     # never a layer row
    assert group.data(0, _LAYER_ID_ROLE) is None
    # Show = edges: no raster output shown, edges drawn
    assert not _hidden(panel, child.layer_id, "edges")
    assert _hidden(panel, child.layer_id, "filtered")
    assert _hidden(panel, child.layer_id, "edge_channel")
    # the dataset (empty chain) has no Outputs group
    assert master.layer_id not in panel._output_groups
    header = panel._layer_items[master.layer_id]
    assert all(header.child(i).text(0) != "Outputs" for i in range(header.childCount()))


def test_rows_radio_the_show_param_and_edges_hide_only_the_drawing(win, qtbot):
    from dynamix.shell.main_window import _display_raster_of

    child = _tool_layer(win, qtbot, "cdf_edges", n_levels=2)
    lid = child.layer_id
    panel = win.layer_list
    misses = win.cache.misses

    # un-hide "filtered": Show -> filtered, a cache hit, the filtered raster on the canvas
    with qtbot.waitSignal(win.resolved, timeout=5000):
        panel.outputHideToggled.emit(lid, "filtered", False)
    result = win._active_result
    assert _show(child) == "filtered"
    assert win.cache.misses == misses
    assert _display_raster_of(result) is result["filtered"]
    np.testing.assert_array_equal(win.canvas._field.values, result["filtered"])
    assert not _hidden(panel, lid, "filtered") and _hidden(panel, lid, "edge_channel")
    assert win.canvas.extrema_raster_item.isVisible() is True

    # hide "edges" with a real click: the tag, the canvas maxima gone, the raster stays
    panel._output_rows[lid]["edges"].hide_button.click()
    assert child.tags["ui.edges_hidden"] == "1"
    assert win.canvas.extrema_raster_item.isVisible() is False
    assert win.canvas.arrow_item.isVisible() is False
    assert _hidden(panel, lid, "edges")
    np.testing.assert_array_equal(win.canvas._field.values, result["filtered"])
    assert any(len(e["x"]) for e in win._active_result["extrema"])   # filters still see them
    # ... and a redraw keeps them hidden ("filtered only, no chains")
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win._reresolve()
    assert win.canvas.extrema_raster_item.isVisible() is False
    assert win.cache.misses == misses

    # hide "filtered": Show -> edges, the raw field is back
    with qtbot.waitSignal(win.resolved, timeout=5000):
        panel.outputHideToggled.emit(lid, "filtered", True)
    assert _show(child) == "edges"
    assert _display_raster_of(win._active_result) is None
    np.testing.assert_array_equal(win.canvas._field.values, win.field.values)
    assert _hidden(panel, lid, "filtered") and _hidden(panel, lid, "edge_channel")

    # un-hide edges: the maxima draw again
    panel._output_rows[lid]["edges"].hide_button.click()
    assert "ui.edges_hidden" not in child.tags
    assert win.canvas.extrema_raster_item.isVisible() is True
    assert win.cache.misses == misses


def test_the_edges_gate_follows_the_active_layer_across_switches(win, qtbot):
    master = win.layer
    first = _tool_layer(win, qtbot, "cdf_edges", n_levels=2)
    win.layer_list._output_rows[first.layer_id]["edges"].hide_button.click()
    assert win.canvas.extrema_raster_item.isVisible() is False
    with qtbot.waitSignal(win.resolved, timeout=30000):
        win.layer_list.select_layer(master.layer_id)
    second = _tool_layer(win, qtbot, "cdf_edges", n_levels=2)
    assert second is not first
    assert win.canvas.extrema_raster_item.isVisible() is True     # its own edges are shown
    with qtbot.waitSignal(win.resolved, timeout=30000):
        win.layer_list.select_layer(first.layer_id)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=30000)
    assert win.layer is first
    assert win.canvas.extrema_raster_item.isVisible() is False


def test_the_strip_show_knob_moves_the_rows(win, qtbot):
    child = _tool_layer(win, qtbot, "cdf_edges", n_levels=2)
    lid = child.layer_id
    i = win._names.index("cdf_edges")
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win._on_param_changed(i, "show", "edge_channel")
    panel = win.layer_list
    assert not _hidden(panel, lid, "edge_channel")
    assert _hidden(panel, lid, "filtered") and not _hidden(panel, lid, "edges")
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win._on_param_changed(i, "show", "edges")
    assert _hidden(panel, lid, "edge_channel") and _hidden(panel, lid, "filtered")


def test_a_hidden_rows_hide_toggle_leaves_the_shown_output_alone(win, qtbot):
    child = _tool_layer(win, qtbot, "cdf_edges", n_levels=2)
    lid = child.layer_id
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win.layer_list.outputHideToggled.emit(lid, "filtered", False)
    win.layer_list.outputHideToggled.emit(lid, "edge_channel", True)   # already hidden
    assert _show(child) == "filtered"
    assert not _hidden(win.layer_list, lid, "filtered")


def test_a_row_chosen_output_survives_a_zone_gesture(win, qtbot):
    """A row click lands in the Show knob's box and the zone's descriptor list too, so a later
    zone gesture (bypass, remove, drop, reorder), whose ``chainEdited`` payload is built from
    those descriptors, keeps the chosen output."""
    child = _tool_layer(win, qtbot, "cdf_edges", n_levels=2)
    lid = child.layer_id
    i = win._names.index("cdf_edges")
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win.layer_list.outputHideToggled.emit(lid, "filtered", False)
    assert win.strips._descriptors[i]["params"]["show"] == "filtered"
    assert win.strips.strip(i)._params["show"] == "filtered"
    win.strips._commit_or_revert([dict(d) for d in win.strips._descriptors])
    qtbot.waitUntil(lambda: not win.is_computing, timeout=30000)
    assert _show(child) == "filtered"
    assert not _hidden(win.layer_list, lid, "filtered")
    assert win.strips.strip(i)._params["show"] == "filtered"


def test_a_locked_layer_refuses_raster_rows_but_edges_still_toggle(win, qtbot):
    child = _tool_layer(win, qtbot, "cdf_edges", n_levels=2)
    lid = child.layer_id
    panel = win.layer_list
    warnings = []
    win.strips._show_warning = warnings.append
    win._on_lock_toggled(lid, True)
    panel._output_rows[lid]["filtered"].hide_button.click()
    assert _show(child) == "edges"
    assert _hidden(panel, lid, "filtered")                # the row snaps back
    i = win._names.index("cdf_edges")
    assert win.strips.strip(i)._params["show"] == "edges"          # the knob never moved
    assert win.strips._descriptors[i]["params"]["show"] == "edges"
    assert warnings and "locked" in warnings[-1]
    panel._output_rows[lid]["edges"].hide_button.click()
    assert child.tags["ui.edges_hidden"] == "1"
    assert win.canvas.extrema_raster_item.isVisible() is False


def test_selecting_an_output_row_selects_its_layer(win, qtbot):
    master = win.layer
    child = _tool_layer(win, qtbot, "cdf_edges", n_levels=2)
    panel = win.layer_list
    with qtbot.waitSignal(win.resolved, timeout=30000):
        panel.select_layer(master.layer_id)
    assert win.layer is master
    with qtbot.waitSignal(win.resolved, timeout=30000):
        panel.setCurrentItem(panel._output_groups[child.layer_id].child(1))
    assert win.layer is child


def test_a_hidden_row_on_a_background_layer_selects_it_and_shows_the_output(win, qtbot):
    master = win.layer
    child = _tool_layer(win, qtbot, "cdf_edges", n_levels=2)
    with qtbot.waitSignal(win.resolved, timeout=30000):
        win.layer_list.select_layer(master.layer_id)
    win.layer_list.outputHideToggled.emit(child.layer_id, "filtered", False)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=30000)
    assert win.layer is child and _show(child) == "filtered"
    qtbot.waitUntil(lambda: win._active_result.get("raster_out") is not None, timeout=30000)
    assert not _hidden(win.layer_list, child.layer_id, "filtered")


def test_edges_hidden_survives_a_project_round_trip_and_seeds_rebuilt_rows(win, qtbot):
    from dynamix.shell.layer_panel import LayerPanel

    child = _tool_layer(win, qtbot, "cdf_edges", n_levels=2)
    lid = child.layer_id
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win.layer_list.outputHideToggled.emit(lid, "filtered", False)
    win.layer_list.outputHideToggled.emit(lid, "edges", True)
    back = Project.from_payload(win.project.to_payload())
    layer = next(l for l in back.layers if l.layer_id == lid)
    assert layer.tags["ui.edges_hidden"] == "1"
    assert _show(layer) == "filtered"
    panel = LayerPanel(back)
    qtbot.addWidget(panel)
    for l in back.layers:
        panel.add_layer_row(l, None)
    assert _hidden(panel, lid, "edges") and not _hidden(panel, lid, "filtered")
    assert _hidden(panel, lid, "edge_channel")
    assert not panel._output_groups[lid].isExpanded()


def test_pm_edges_gets_edges_and_filtered_rows(win, qtbot):
    child = _tool_layer(win, qtbot, "pm_edges", n_levels=2)
    assert _row_names(win.layer_list, child.layer_id) == ["edges", "filtered"]


def test_the_vector_view_entry_carries_the_edges_gate(win, qtbot):
    """The arrangement scene reads ``edges_hidden`` off its entry to drop the H-lines/dots."""
    child = _tool_layer(win, qtbot, "cdf_edges", n_levels=2)
    seen = []

    class _Arrangement:
        def set_layers(self, entries):
            seen.append({e["layer"].layer_id: e.get("edges_hidden") for e in entries
                         if e["status"] == "ok"})

    win._arrangement = _Arrangement()
    win._sync_arrangement(frame_mode=True)        # the field has a local frame, no CRS
    assert seen[-1].get(child.layer_id) is False
    win.layer_list.outputHideToggled.emit(child.layer_id, "edges", True)
    win._sync_arrangement(frame_mode=True)
    assert seen[-1].get(child.layer_id) is True


def test_the_outputs_group_follows_chain_edits(win, qtbot):
    child = _tool_layer(win, qtbot, "cdf_edges", n_levels=2)
    lid = child.layer_id
    panel = win.layer_list
    with qtbot.waitSignal(win.resolved, timeout=5000):
        panel.outputHideToggled.emit(lid, "filtered", False)
    i = win._names.index("cdf_edges")
    desc = [{"device": "cdf_edges", "params": dict(win._params[i]), "bypassed": True}]
    win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=30000)
    assert lid not in panel._output_groups                       # nothing declares outputs
    item = panel._layer_items[lid]
    assert all(item.child(k).text(0) != "Outputs" for k in range(item.childCount()))
    desc[0]["bypassed"] = False
    win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=30000)
    assert _row_names(panel, lid) == ["edges", "filtered", "edge channel"]
    assert not _hidden(panel, lid, "filtered")                   # rebuilt from the chain


def test_removing_a_layer_forgets_its_output_rows(win, qtbot):
    child = _tool_layer(win, qtbot, "cdf_edges", n_levels=2)
    lid = child.layer_id
    win.layer_list.remove_rows([lid])
    assert lid not in win.layer_list._output_groups
    assert lid not in win.layer_list._output_rows
    win.layer_list.set_output_state(lid, "filtered", True)        # a stale push is a no-op


def test_the_devices_declare_their_outputs():
    from dynamix.devices.cdf_edges import CDFEdges
    from dynamix.devices.pm_edges import PMEdges
    from dynamix.model.device import declared_outputs

    cdf = declared_outputs(CDFEdges())
    assert [(o.name, o.kind, o.lazy, o.label) for o in cdf] == [
        ("edges", "vector", False, "edges"), ("filtered", "raster", False, "filtered"),
        ("edge_channel", "raster", False, "edge channel")]
    pm = declared_outputs(PMEdges())
    assert [(o.name, o.kind, o.lazy) for o in pm] == [
        ("edges", "vector", False), ("filtered", "raster", False)]
    # every eager raster output is a Show choice
    for dev in (CDFEdges(), PMEdges()):
        show = next(p for p in dev.params if p.name == "show")
        assert {o.name for o in declared_outputs(dev) if o.kind == "raster"} <= set(show.choices)
