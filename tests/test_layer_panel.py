# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.layer_panel -- groups, hide/lock/freeze, remove, rename.

Two halves, per the file's own split convention elsewhere in this suite (test_shell_canvas.py's
docstring): the WIDGET half drives a bare ``LayerPanel`` directly against a real ``Project`` (no
``MainWindow``, no worker, no resolve); the INTEGRATION half drives ``MainWindow``'s own
enforcement (lock/freeze read-only, freeze's cache pins, the remove confirm/refuse flow, hide's
canvas clear, rename) through a small stub chain, same pattern ``test_shell_window.py`` and
``test_shell_roi_flow.py`` each duplicate locally rather than importing from one another.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from dynamix.devices import register_builtin_devices
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.param import Param, ParamKind
from dynamix.model.project import Project
from dynamix.shell.layer_panel import LayerPanel, _LAYER_ID_ROLE

# --------------------------------------------------------------------------- widget fixtures


@pytest.fixture
def project():
    return Project(title="t")


@pytest.fixture
def panel(qtbot, project):
    p = LayerPanel(project)
    qtbot.addWidget(p)
    return p


def _layer(project, name, source_id, *, parent_id=None, visible=True, tags=None):
    return project.add_layer(name, source_id, Chain(), visible=visible, tags=tags,
                             parent_id=parent_id)


# --------------------------------------------------------------------------- grouped rendering


def test_two_layers_over_one_source_group_under_one_header(panel, project):
    source = project.add_source("/data/dataset.tif")
    a = _layer(project, "A", source.source_id)
    b = _layer(project, "B", source.source_id)
    panel.add_layer_row(a, None)
    panel.add_layer_row(b, None)

    assert panel.topLevelItemCount() == 1                  # one source header
    header = panel.topLevelItem(0)
    assert header.text(0) == "dataset"
    assert header.childCount() == 2
    assert panel.count() == 2                               # headers excluded


def test_add_layer_row_emits_no_rename_requests(panel, project):
    """IMPORTANT 3 (review): the row's ``QTreeWidgetItem`` used to be constructed already
    parented, so the ``setData(0, _LAYER_ID_ROLE, ...)`` call right after construction fired
    ``itemChanged`` for real -- ``_on_item_changed`` reads exactly that role, so every single
    ``add_layer_row`` call emitted a spurious ``renameRequested(layer_id, layer.name)`` for a name
    nobody edited."""
    source = project.add_source("/data/dataset.tif")
    a = _layer(project, "A", source.source_id)
    b = _layer(project, "B", source.source_id)

    received = []
    panel.renameRequested.connect(lambda lid, name: received.append((lid, name)))
    panel.add_layer_row(a, None)
    panel.add_layer_row(b, None)

    assert received == []


def test_roi_child_nests_under_its_parent_layer_not_the_header(panel, project):
    source = project.add_source("/data/dataset.tif")
    parent = _layer(project, "A", source.source_id)
    child = _layer(project, "A ROI 1,1", source.source_id, parent_id=parent.layer_id)
    panel.add_layer_row(parent, None)
    panel.add_layer_row(child, None)

    parent_item = panel._layer_items[parent.layer_id]
    child_item = panel._layer_items[child.layer_id]
    assert child_item.parent() is parent_item
    header = panel.topLevelItem(0)
    assert header.childCount() == 1                         # only the parent is a direct child
    assert panel.count() == 2


def test_header_falls_back_to_field_name_with_no_project(qtbot):
    """No ``Project`` set (a bare panel) -- the header still renders, from ``field.name``."""
    bare = LayerPanel()
    qtbot.addWidget(bare)
    project = Project(title="t")
    source = project.add_source("mem:stub")
    layer = _layer(project, "A", source.source_id)

    class _Field:
        name = "my_raster"

    bare.add_layer_row(layer, _Field())

    assert bare.topLevelItem(0).text(0) == "my_raster"


def test_header_item_carries_no_layer_id_role(panel, project):
    """The guard every header-excluding handler (context menu, selection) relies on."""
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)

    header = panel.topLevelItem(0)
    assert header.data(0, _LAYER_ID_ROLE) is None


# --------------------------------------------------------------------------- collapse sync


def test_header_starts_collapsed_when_the_source_ref_already_is(panel, project):
    source = project.add_source("/data/dataset.tif")
    source.collapsed = True
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)

    assert panel.topLevelItem(0).isExpanded() is False


def test_collapsing_a_header_writes_back_into_the_source_ref(panel, project):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    header = panel.topLevelItem(0)

    header.setExpanded(False)
    assert source.collapsed is True

    header.setExpanded(True)
    assert source.collapsed is False


def test_select_layer_expands_a_collapsed_ancestor_and_syncs_it_open(panel, project):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    header = panel.topLevelItem(0)
    header.setExpanded(False)
    assert source.collapsed is True

    panel.select_layer(layer.layer_id)

    assert header.isExpanded() is True
    assert source.collapsed is False


# --------------------------------------------------------------------------- H/L/F buttons


def test_row_buttons_seed_from_the_layer_flags(panel, project):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id, visible=False,
                   tags={"ui.lock": "1"})
    panel.add_layer_row(layer, None)

    row = panel._row_widgets[layer.layer_id]
    assert row.hide_button.isChecked() is True       # not visible -> hidden is checked
    assert row.lock_button.isChecked() is True
    assert row.freeze_button.isChecked() is False


def test_hide_button_emits_hideToggled_with_the_layer_id(panel, project, qtbot):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    row = panel._row_widgets[layer.layer_id]

    with qtbot.waitSignal(panel.hideToggled, timeout=1000) as sig:
        row.hide_button.click()
    assert sig.args == [layer.layer_id, True]


def test_lock_button_emits_lockToggled_with_the_layer_id(panel, project, qtbot):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    row = panel._row_widgets[layer.layer_id]

    with qtbot.waitSignal(panel.lockToggled, timeout=1000) as sig:
        row.lock_button.click()
    assert sig.args == [layer.layer_id, True]


def test_freeze_button_emits_freezeToggled_with_the_layer_id(panel, project, qtbot):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    row = panel._row_widgets[layer.layer_id]

    with qtbot.waitSignal(panel.freezeToggled, timeout=1000) as sig:
        row.freeze_button.click()
    assert sig.args == [layer.layer_id, True]


# --------------------------------------------------------------- inspector toggle (shell v2 T8)
# "the layer list becomes a tree rooted at sources; ``inspectorToggled`` carries a
# ``source_id``, not a ``layer_id``." The tree is ALREADY rooted at sources
# (``_ensure_source_header``), so what sec 5d needs here is a BUTTON on the header, not a new tree
# -- and the toggle is per SOURCE, never per layer.


def test_source_header_carries_an_inspector_toggle(panel, project):
    source = project.add_source("/data/dataset.tif")
    panel.add_layer_row(_layer(project, "A", source.source_id), None)

    header = panel.topLevelItem(0)
    row = panel.itemWidget(header, 1)
    assert row is not None
    assert row is panel._source_rows[source.source_id]
    assert row.inspector_button.isCheckable() is True
    assert row.inspector_button.isChecked() is False       # closed until something opens it


def test_toggling_the_header_button_emits_inspector_toggled_with_the_source_id(
        panel, project, qtbot):
    source = project.add_source("/data/dataset.tif")
    panel.add_layer_row(_layer(project, "A", source.source_id), None)
    row = panel._source_rows[source.source_id]

    with qtbot.waitSignal(panel.inspectorToggled, timeout=1000) as sig:
        row.inspector_button.click()
    assert sig.args == [source.source_id, True]            # a str id, unlike every sibling signal


def test_set_inspector_open_updates_the_button_without_re_emitting(panel, project):
    """The other half of ``InspectorWindow.closed``: the window closing itself and the button
    un-checking are ONE state, so the push-back must not echo a second toggle straight back out
    (the guard ``set_layer_name`` already uses)."""
    source = project.add_source("/data/dataset.tif")
    panel.add_layer_row(_layer(project, "A", source.source_id), None)
    row = panel._source_rows[source.source_id]

    received = []
    panel.inspectorToggled.connect(lambda sid, state: received.append((sid, state)))
    panel.set_inspector_open(source.source_id, True)
    assert row.inspector_button.isChecked() is True
    panel.set_inspector_open(source.source_id, False)
    assert row.inspector_button.isChecked() is False

    assert received == []


def test_layer_rows_do_not_carry_an_inspector_toggle(panel, project):
    """The inspector is per SOURCE dataset. A layer row keeps its three H/L/F buttons and
    grows no fourth one."""
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)

    row = panel._row_widgets[layer.layer_id]
    assert not hasattr(row, "inspector_button")
    assert row.layout().count() == 3
    assert layer.layer_id not in panel._source_rows


def test_removing_the_last_layer_under_a_source_prunes_its_header_and_button(panel, project):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    assert source.source_id in panel._source_rows

    panel.remove_rows([layer.layer_id])

    assert panel.topLevelItemCount() == 0
    assert source.source_id not in panel._source_rows      # no leaked _SourceRow


def test_header_text_still_reads_the_file_stem(panel, project):
    """Guards the one existing line this task changes (``setFirstColumnSpanned`` True -> False):
    un-spanning column 0 is what lets the column-1 item widget show at all, and it must not cost
    the header its label."""
    source = project.add_source("/data/dataset.tif")
    panel.add_layer_row(_layer(project, "A", source.source_id), None)

    header = panel.topLevelItem(0)
    assert header.text(0) == "dataset"
    assert header.isFirstColumnSpanned() is False


def test_a_long_source_stem_is_not_elided_in_the_header(panel, project):
    """The OTHER half of un-spanning the header, and the half ``header.text(0)`` provably cannot
    see: ``text(0)`` is item DATA, so it reads the whole stem whether or not the
    view has room to PAINT it. A spanned header always had the full row's width; an un-spanned one
    is confined to column 0, which sits at ``QHeaderView``'s fixed 100 px default unless a resize
    mode says otherwise -- and every real dataset in this project (``data/`` holds
    ``gebco_2023_*``, the BOEM tiles) has a stem far longer than 100 px, so it painted as
    ``gebco_20...``.

    Asserted as the two things a stretched label column does that a 100 px one does not: it is
    wide enough for the stem's own pixel width, and it GROWS when the panel does."""
    stem = "gebco_2023_n60_s40_w150_e180"
    source = project.add_source(f"/data/{stem}.tif")
    panel.add_layer_row(_layer(project, "wtmm run 1", source.source_id), None)

    panel.resize(800, 240)
    panel.show()
    QtWidgets.QApplication.processEvents()
    assert panel.columnWidth(0) > panel.fontMetrics().horizontalAdvance(stem)

    narrow = panel.columnWidth(0)
    panel.resize(1000, 240)
    QtWidgets.QApplication.processEvents()
    assert panel.columnWidth(0) > narrow


def test_the_inspector_toggle_column_only_takes_the_width_its_buttons_need(panel, project):
    """The counterpart of the stretch above: column 1 must stay hard against the right edge
    (``ResizeToContents``), not swallow the surplus a widening panel produces. Otherwise the label
    column's stretch buys nothing."""
    source = project.add_source("/data/gebco_2023_n60_s40_w150_e180.tif")
    panel.add_layer_row(_layer(project, "wtmm run 1", source.source_id), None)

    panel.resize(800, 240)
    panel.show()
    QtWidgets.QApplication.processEvents()
    button_column = panel.columnWidth(1)

    panel.resize(1000, 240)
    QtWidgets.QApplication.processEvents()
    assert panel.columnWidth(1) == button_column


# --------------------------------------------------------------------------- selection


def test_select_layer_sets_current_and_emits_layerSelected(panel, project, qtbot):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)

    with qtbot.waitSignal(panel.layerSelected, timeout=1000) as sig:
        panel.select_layer(layer.layer_id)
    assert sig.args == [layer.layer_id]
    assert panel.current_layer_id() == layer.layer_id


def test_selecting_a_header_row_promotes_to_the_master_layer(panel, project, qtbot):
    """2026-09-18: a header click is
    a real selection target -- it promotes onto the source's first layer row (the master),
    which emits normally, instead of the old no-op that stranded the previous rack."""
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    header = panel.topLevelItem(0)

    received = []
    panel.layerSelected.connect(received.append)
    panel.setCurrentItem(header)

    assert received == [layer.layer_id]
    assert panel.currentItem() is not header          # current landed on the master row


# --------------------------------------------------------------------------- remove


def test_remove_rows_drops_a_leaf_layer_and_keeps_its_sibling(panel, project):
    source = project.add_source("/data/dataset.tif")
    a = _layer(project, "A", source.source_id)
    b = _layer(project, "B", source.source_id)
    panel.add_layer_row(a, None)
    panel.add_layer_row(b, None)

    panel.remove_rows([a.layer_id])

    assert panel.count() == 1
    assert a.layer_id not in panel._layer_items
    assert panel.topLevelItem(0).childCount() == 1


def test_remove_rows_prunes_an_emptied_source_header(panel, project):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)

    panel.remove_rows([layer.layer_id])

    assert panel.topLevelItemCount() == 0
    assert panel.count() == 0


def test_remove_rows_children_first_then_parent_removes_the_whole_group(panel, project):
    source = project.add_source("/data/dataset.tif")
    parent = _layer(project, "A", source.source_id)
    child = _layer(project, "A ROI", source.source_id, parent_id=parent.layer_id)
    panel.add_layer_row(parent, None)
    panel.add_layer_row(child, None)

    panel.remove_rows([child.layer_id, parent.layer_id])   # Project.remove_layer's own order

    assert panel.topLevelItemCount() == 0
    assert panel.count() == 0


# --------------------------------------------------------------------------- rename


def test_editing_the_name_column_emits_renameRequested(panel, project, qtbot):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    item = panel._layer_items[layer.layer_id]

    with qtbot.waitSignal(panel.renameRequested, timeout=1000) as sig:
        item.setText(0, "renamed")
    assert sig.args == [layer.layer_id, "renamed"]


def test_set_layer_name_updates_text_without_reemitting_rename(panel, project, qtbot):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)

    received = []
    panel.renameRequested.connect(lambda lid, name: received.append((lid, name)))
    panel.set_layer_name(layer.layer_id, "pushed")

    assert panel.layer_text(layer.layer_id) == "pushed"
    assert received == []


# --------------------------------------------------------------------------- context menu


def test_context_menu_has_rename_remove_and_refined_run(panel, project):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    item = panel._layer_items[layer.layer_id]

    menu = panel._build_context_menu(item, layer.layer_id)
    titles = [a.text() for a in menu.actions()]
    assert titles == ["Rename", "Remove", "New refined run"]


def test_context_menu_remove_action_emits_removeRequested(panel, project, qtbot):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    item = panel._layer_items[layer.layer_id]
    menu = panel._build_context_menu(item, layer.layer_id)
    remove_action = next(a for a in menu.actions() if a.text() == "Remove")

    with qtbot.waitSignal(panel.removeRequested, timeout=1000) as sig:
        remove_action.trigger()
    assert sig.args == [layer.layer_id]


def test_context_menu_refined_run_action_emits_refinedRunRequested(panel, project, qtbot):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    item = panel._layer_items[layer.layer_id]
    menu = panel._build_context_menu(item, layer.layer_id)
    refined_action = next(a for a in menu.actions() if a.text() == "New refined run")

    with qtbot.waitSignal(panel.refinedRunRequested, timeout=1000) as sig:
        refined_action.trigger()
    assert sig.args == [layer.layer_id]


def test_context_menu_rename_action_opens_the_inline_editor(panel, project):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    item = panel._layer_items[layer.layer_id]
    menu = panel._build_context_menu(item, layer.layer_id)
    rename_action = next(a for a in menu.actions() if a.text() == "Rename")

    rename_action.trigger()

    assert panel.state() == QtWidgets.QAbstractItemView.State.EditingState


# --------------------------------------------------------------------------- keyboard delete


def test_delete_key_on_a_selected_layer_emits_removeRequested(panel, project, qtbot):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    panel.select_layer(layer.layer_id)

    with qtbot.waitSignal(panel.removeRequested, timeout=1000) as sig:
        qtbot.keyClick(panel, QtCore.Qt.Key_Delete)
    assert sig.args == [layer.layer_id]


def test_backspace_key_on_a_selected_layer_also_emits_removeRequested(panel, project, qtbot):
    """Backspace, not just Delete: this app is macOS-only, and Backspace is the key physically
    labeled "delete" on a Mac keyboard -- there is no separate forward-delete key to bind
    instead."""
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    panel.select_layer(layer.layer_id)

    with qtbot.waitSignal(panel.removeRequested, timeout=1000) as sig:
        qtbot.keyClick(panel, QtCore.Qt.Key_Backspace)
    assert sig.args == [layer.layer_id]


def test_delete_key_after_a_header_click_targets_the_promoted_master(panel, project, qtbot):
    """The header promotion (above) means a header can no longer BE current -- Delete now
    honestly targets the master row the click landed on, same as clicking it directly."""
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    header = panel.topLevelItem(0)
    panel.setCurrentItem(header)

    received = []
    panel.removeRequested.connect(received.append)
    qtbot.keyClick(panel, QtCore.Qt.Key_Delete)

    assert received == [layer.layer_id]


# --------------------------------------------------------------------------- flash_row


def test_flash_row_sets_then_clears_the_property(panel, project):
    source = project.add_source("/data/dataset.tif")
    layer = _layer(project, "A", source.source_id)
    panel.add_layer_row(layer, None)
    widget = panel._row_widgets[layer.layer_id]

    panel.flash_row(layer.layer_id)
    assert widget.property("flash") == "true"

    panel._clear_flash(layer.layer_id)               # the 1s timer's own callback, called directly
    assert widget.property("flash") == "false"


# =================================================================================================
# MainWindow integration: lock/freeze enforcement, remove confirm/refuse, hide, rename.
# =================================================================================================


class _StubStack:
    """Minimal Transform, same shape test_shell_window.py's own stub uses."""

    name = "stub_stack"
    params = (Param("n_scales", ParamKind.INT, default=3, min=1, max=8, label="Scales"),)

    def compute(self, field, params, *, progress=None):
        n = int(params["n_scales"])
        values = np.asarray(getattr(field, "values", field), dtype=np.float64)
        ny, nx = values.shape[:2]
        layers = [{"x": np.arange(4) % nx, "y": np.arange(4) % ny,
                  "mod": np.linspace(1.0, 0.1, 4), "arg": np.zeros(4),
                  "line_id": np.array([0, 0, 0, -1], dtype=np.int64)} for _ in range(n)]
        return {"scales": [2.0 * (k + 1) for k in range(n)], "extrema": layers,
                "_shape": (ny, nx), "params": {}}

    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key
        return cache_key(self.name, source_id, params)


_STUB_CHAIN = (("stub_stack", {}), ("scale_select", {"scale_idx": 0}))
_FIELD = np.linspace(0.0, 1.0, 16 * 16).reshape(16, 16)


@pytest.fixture
def stub_devices(clean_registry):
    from dynamix.model.device import register_device

    register_builtin_devices()
    register_device(_StubStack())
    return clean_registry


@pytest.fixture
def loaded(qtbot, stub_devices):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=_STUB_CHAIN)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD, "mem:stub")
    return win


# --------------------------------------------------------------------------- lock / freeze


def test_locked_layer_param_change_is_a_noop(loaded):
    layer = loaded.layer
    loaded._on_lock_toggled(layer.layer_id, True)
    before = [dict(p) for p in loaded._params]

    loaded._on_param_changed(1, "scale_idx", 0)       # scale_select's own param

    assert loaded._params == before
    assert loaded.strips.reading_label.text() != ""


def test_locked_layer_chain_edit_is_a_noop(loaded):
    layer = loaded.layer
    loaded._on_lock_toggled(layer.layer_id, True)
    before_names = list(loaded._names)

    descriptors = [{"device": "scale_select", "params": {"scale_idx": 0}, "bypassed": False,
                    "rack": None}]
    loaded._on_chain_edited(descriptors)

    assert loaded._names == before_names


def test_freeze_pins_at_least_one_cache_key_and_unfreeze_unpins(loaded):
    layer = loaded.layer
    keys = loaded._cache_keys_for(layer)
    assert keys                                       # stub_stack is a transform -> >= 1 key

    loaded._on_freeze_toggled(layer.layer_id, True)
    assert all(loaded.cache.is_pinned(k) for k in keys)

    loaded._on_freeze_toggled(layer.layer_id, False)
    assert not any(loaded.cache.is_pinned(k) for k in keys)


def test_selecting_a_locked_layer_disables_the_zone_and_the_transport(loaded):
    layer = loaded.layer
    loaded._on_lock_toggled(layer.layer_id, True)

    loaded._select_layer(layer)

    assert loaded.strips.isEnabled() is False
    assert loaded.transport.isEnabled() is False


def test_unlocking_the_active_layer_reenables_the_zone_and_the_transport(loaded):
    layer = loaded.layer
    loaded._on_lock_toggled(layer.layer_id, True)
    assert loaded.strips.isEnabled() is False
    assert loaded.transport.isEnabled() is False

    loaded._on_lock_toggled(layer.layer_id, False)

    assert loaded.strips.isEnabled() is True
    assert loaded.transport.isEnabled() is True


def test_locked_layer_scale_change_is_a_noop(loaded):
    """IMPORTANT 4 (review): the transport is a SECOND write path into ``_params``/
    ``layer.chain``, reachable by scrub or Space even while the zone is disabled -- it needs its
    own lock/freeze refusal, not just a disabled zone."""
    layer = loaded.layer
    loaded._on_lock_toggled(layer.layer_id, True)
    before = [dict(p) for p in loaded._params]
    before_chain = layer.chain

    loaded._on_scale_changed(0)

    assert loaded._params == before
    assert layer.chain is before_chain
    assert loaded.strips.reading_label.text() != ""


def test_on_finished_leaves_the_transport_disabled_for_a_locked_active_layer(loaded, qtbot):
    """IMPORTANT 4 (review): ``_on_finished`` used to unconditionally re-enable the transport,
    silently undoing a lock that took effect (or was already in effect) while the worker it just
    tore down was running. Locking a layer BEFORE it is ever selected, then selecting it (a real
    cache-miss dispatch, since its chain differs from the fixture's own), exercises exactly that
    landing."""
    first = loaded.layer
    second_chain = Chain((DeviceRef("stub_stack", {"n_scales": 5}),
                          DeviceRef("scale_select", {"scale_idx": 0})))
    second = loaded.project.add_layer("second", first.source_id, second_chain)
    loaded.add_layer_row(second, loaded.field)
    loaded._on_lock_toggled(second.layer_id, True)         # locked before it is ever the active one

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.layer_list.select_layer(second.layer_id)

    assert loaded.layer is second
    assert loaded.transport.isEnabled() is False
    assert loaded.strips.isEnabled() is False


# --------------------------------------------------------------------------- remove


def test_remove_requested_on_a_locked_layer_is_refused(loaded):
    layer = loaded.layer
    loaded._on_lock_toggled(layer.layer_id, True)

    loaded._on_remove_requested(layer.layer_id)

    assert layer in loaded.project.layers
    assert layer.layer_id in loaded.layer_list._layer_items


def test_remove_requested_on_a_leaf_layer_skips_the_confirm_dialog(loaded, monkeypatch):
    layer = loaded.layer

    def _boom(*a, **k):
        raise AssertionError("QMessageBox.question must not be called for a leaf layer")

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", _boom)

    loaded._on_remove_requested(layer.layer_id)

    assert layer not in loaded.project.layers
    assert layer.layer_id not in loaded.layer_list._layer_items


def test_remove_requested_on_a_layer_with_children_confirms_once_then_cascades(loaded, monkeypatch):
    parent = loaded.layer
    child = loaded.project.add_layer("child", parent.source_id, Chain(),
                                     parent_id=parent.layer_id)
    loaded.add_layer_row(child, loaded.field)

    calls = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question",
        lambda *a, **k: (calls.append(1), QtWidgets.QMessageBox.Yes)[1])

    loaded._on_remove_requested(parent.layer_id)

    assert len(calls) == 1
    assert parent not in loaded.project.layers
    assert child not in loaded.project.layers
    assert loaded.layer_list.count() == 0


def test_remove_requested_declined_confirmation_keeps_both_layers(loaded, monkeypatch):
    parent = loaded.layer
    child = loaded.project.add_layer("child", parent.source_id, Chain(),
                                     parent_id=parent.layer_id)
    loaded.add_layer_row(child, loaded.field)

    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a, **k: QtWidgets.QMessageBox.No)

    loaded._on_remove_requested(parent.layer_id)

    assert parent in loaded.project.layers
    assert child in loaded.project.layers


def test_remove_requested_on_the_only_layer_clears_the_zombie_active_layer(loaded):
    """``self.layer`` used to keep pointing at the just-removed ``Layer`` object
    whenever the LAST layer went (the reselect branch only fires ``if ... self.project.layers``,
    which is empty here) -- every knob kept editing an object nothing would ever read again, and a
    save would have silently discarded whatever the user thought was still open. The window must
    instead return to its pre-open state: no layer, no field, the plain title, and both remaining
    write paths into ``self.layer`` (a knob-shaped ``paramChanged``, the transport's ``scaleChanged``)
    must tolerate being called with nothing active rather than crash."""
    from dynamix.shell.main_window import TITLE

    layer = loaded.layer

    loaded._on_remove_requested(layer.layer_id)

    assert loaded.layer is None
    assert loaded.field is None
    assert loaded.windowTitle() == TITLE
    assert loaded.transport.isEnabled() is False
    assert loaded.strips.isEnabled() is False
    assert loaded.strips._boxes == []
    assert loaded.canvas.image_item.image is None

    # no crash on a subsequent paramChanged-shaped or scaleChanged-shaped call with no active layer
    loaded._on_param_changed(0, "scale_idx", 0)
    loaded._on_scale_changed(0)
    loaded._on_chain_edited([])


def test_remove_requested_reselects_a_remaining_layer_when_the_active_one_goes(loaded):
    first = loaded.layer
    second = loaded.project.add_layer("second", first.source_id,
                                      Chain(tuple(DeviceRef(n, dict(p)) for n, p in _STUB_CHAIN)))
    loaded.add_layer_row(second, loaded.field)

    loaded._on_remove_requested(first.layer_id)

    assert loaded.layer is second


def test_remove_requested_reviewer_probe_select_child_then_remove_its_group(loaded, monkeypatch):
    """IMPORTANT 2 (review): select an ROI child (the active layer), then remove its PARENT
    group. ``layer_list.remove_rows`` used to run BEFORE ``_layer_by_id``/``_fields``/``_recipes``
    were pruned, and Qt's own removal-time reselection (promoting some other row to current)
    fired a REAL ``layerSelected`` for a row about to be deleted two lines later -- a full
    ``_select_layer`` -> ``_start_worker`` dispatch for a layer that no longer exists in the
    project. Only the explicit reselect of the actual survivor, exactly once, is legitimate."""
    parent = loaded.layer
    child = loaded.project.add_layer(
        "child", parent.source_id, Chain(tuple(DeviceRef(n, dict(p)) for n, p in _STUB_CHAIN)),
        parent_id=parent.layer_id)
    loaded.add_layer_row(child, loaded.field)
    survivor = loaded.project.add_layer(
        "survivor", parent.source_id, Chain(tuple(DeviceRef(n, dict(p)) for n, p in _STUB_CHAIN)))
    loaded.add_layer_row(survivor, loaded.field)

    loaded.layer_list.select_layer(child.layer_id)
    assert loaded.layer is child

    selected_ids = []
    loaded.layer_list.layerSelected.connect(selected_ids.append)
    dispatched_for = []
    real_start_worker = loaded._start_worker

    def _spy_start_worker():
        dispatched_for.append(loaded.layer.layer_id)
        real_start_worker()

    monkeypatch.setattr(loaded, "_start_worker", _spy_start_worker)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a, **k: QtWidgets.QMessageBox.Yes)

    loaded._on_remove_requested(parent.layer_id)

    removed_ids = {parent.layer_id, child.layer_id}
    assert not (set(selected_ids) & removed_ids)       # no layerSelected for a removed id
    assert not (set(dispatched_for) & removed_ids)      # no worker dispatched for a removed layer
    assert selected_ids == [survivor.layer_id]          # the survivor, selected exactly once
    assert dispatched_for == [survivor.layer_id]
    assert loaded.layer is survivor


def test_remove_requested_reviewer_probe_round_2_removing_a_non_active_layer_keeps_the_highlight(
        loaded, monkeypatch):
    """IMPORTANT: round 1's fix made ``remove_rows`` clear the panel's current
    item UNCONDITIONALLY, which threw away the selection highlight even when the removed row
    wasn't the current one at all -- Qt's own default (leave an unrelated current item alone) was
    already correct there. Two layers, the FIRST active, remove the SECOND: the active layer and
    its highlight must both survive untouched, with no reselect round-trip at all (nothing ever
    changed, so nothing should fire)."""
    first = loaded.layer
    second = loaded.project.add_layer(
        "second", first.source_id, Chain(tuple(DeviceRef(n, dict(p)) for n, p in _STUB_CHAIN)))
    loaded.add_layer_row(second, loaded.field)
    assert loaded.layer_list.current_layer_id() == first.layer_id

    selected_ids = []
    loaded.layer_list.layerSelected.connect(selected_ids.append)
    dispatched_for = []
    real_start_worker = loaded._start_worker

    def _spy_start_worker():
        dispatched_for.append(loaded.layer.layer_id)
        real_start_worker()

    monkeypatch.setattr(loaded, "_start_worker", _spy_start_worker)

    def _boom(*a, **k):
        raise AssertionError("QMessageBox.question must not be called for a leaf layer")

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", _boom)

    loaded._on_remove_requested(second.layer_id)

    assert second not in loaded.project.layers
    assert loaded.layer is first
    assert loaded.layer_list.current_layer_id() == first.layer_id
    assert dispatched_for == []
    assert selected_ids == []


# --------------------------------------------------------------------------- rename


def test_rename_requested_updates_layer_name_and_row_text(loaded):
    layer = loaded.layer

    loaded._on_rename_requested(layer.layer_id, "new name")

    assert layer.name == "new name"
    assert loaded.layer_list.layer_text(layer.layer_id) == "new name"
    assert loaded.windowTitle().endswith("new name")


def test_rename_requested_with_a_blank_name_reverts_the_row_instead_of_renaming(loaded):
    layer = loaded.layer
    original = layer.name

    loaded._on_rename_requested(layer.layer_id, "")

    assert layer.name == original
    assert loaded.layer_list.layer_text(layer.layer_id) == original


# --------------------------------------------------------------------------- hide


def test_hiding_the_active_layer_clears_canvas_overlays(loaded, monkeypatch):
    layer = loaded.layer
    calls = []
    monkeypatch.setattr(loaded.canvas, "clear_overlays", lambda: calls.append(1))

    loaded._on_hide_toggled(layer.layer_id, True)

    assert layer.visible is False
    assert calls == [1]


def test_hiding_an_inactive_layer_does_not_touch_the_canvas(loaded, monkeypatch):
    active = loaded.layer
    other = loaded.project.add_layer("other", active.source_id, Chain())
    loaded.add_layer_row(other, loaded.field)
    calls = []
    monkeypatch.setattr(loaded.canvas, "clear_overlays", lambda: calls.append(1))

    loaded._on_hide_toggled(other.layer_id, True)

    assert other.visible is False
    assert calls == []
    assert active is loaded.layer


def test_unhiding_the_active_layer_reresolves_without_a_worker_dispatch(loaded, qtbot):
    layer = loaded.layer
    loaded._on_hide_toggled(layer.layer_id, True)
    assert loaded.canvas.extrema_item.data.size == 0

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded._on_hide_toggled(layer.layer_id, False)

    assert layer.visible is True
    assert loaded._thread is None                      # cache-hit filter-path redraw, no worker
    assert loaded.canvas.extrema_item.data.size > 0     # overlays are back


# IMPORTANT 1 (review): nothing on the render path used to consult ``layer.visible`` -- hiding
# only cleared the canvas ONCE, at the moment of the toggle. Any redraw after that (a filter edit,
# a layer switch away and back, an in-flight worker landing late) went through ``_apply``
# unconditionally and silently repainted the overlays the H button still claimed were hidden.


def test_hidden_layer_stays_hidden_after_a_filter_param_edit(loaded):
    layer = loaded.layer
    assert loaded.canvas.extrema_item.data.size > 0     # sanity: the stub stack does draw points

    loaded._on_hide_toggled(layer.layer_id, True)
    assert loaded.canvas.extrema_item.data.size == 0

    loaded._on_param_changed(1, "scale_idx", 0)          # scale_select: filter path, synchronous

    assert loaded.canvas.extrema_item.data.size == 0
    assert layer.visible is False


def test_hidden_layer_stays_hidden_after_switching_away_and_back(loaded, qtbot):
    first = loaded.layer
    loaded._on_hide_toggled(first.layer_id, True)
    assert loaded.canvas.extrema_item.data.size == 0

    second = loaded.project.add_layer(
        "second", first.source_id, Chain(tuple(DeviceRef(n, dict(p)) for n, p in _STUB_CHAIN)))
    loaded.add_layer_row(second, loaded.field)
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.layer_list.select_layer(second.layer_id)
    assert loaded.canvas.extrema_item.data.size > 0      # the visible layer draws normally

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.layer_list.select_layer(first.layer_id)

    assert loaded.canvas.extrema_item.data.size == 0
    assert first.visible is False


# =================================================================================================
# Refined-run action + source chip.
# =================================================================================================


def test_refined_run_creates_a_grouped_layer_with_transforms_only_and_copied_params(loaded, qtbot):
    """Filters (``scale_select``, the second _STUB_CHAIN step) are dropped; the transform
    (``stub_stack``) rides across with its OWN params, deep-copied -- a non-default value set on
    the parent BEFORE the refined run must show up unchanged on the child."""
    parent = loaded.layer
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded._on_param_changed(0, "n_scales", 5)          # stub_stack's own param, non-default

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded._on_refined_run(parent.layer_id)

    child = loaded.layer
    assert child is not parent
    assert child.name == f"{parent.name} · refined"
    assert child.parent_id == parent.layer_id
    assert child.tags.get("fed_by") == str(parent.layer_id)
    assert [ref.device for ref in child.chain.steps] == ["stub_stack"]     # filter dropped
    assert child.chain.steps[0].params["n_scales"] == 5                   # copied, not reset
    # deep-copied, not aliased: mutating the parent's own recipe params afterward must not reach
    # back into the child's already-materialised chain.
    assert parent.chain.steps[0].params is not child.chain.steps[0].params
    assert child.layer_id in loaded._fields                               # field registered
    assert child.layer_id in loaded.layer_list._layer_items
    assert loaded.layer_list._layer_items[child.layer_id].parent() is (
        loaded.layer_list._layer_items[parent.layer_id])                  # grouped under parent


def test_refined_run_selects_the_new_layer_and_dispatches_a_compute(loaded, qtbot):
    """The child's chain has a transform (``stub_stack``), so selecting it is a real cache-miss
    dispatch -- the worker-wait pattern the module docstring's own convention uses for every other
    real compute in this file."""
    parent = loaded.layer

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded._on_refined_run(parent.layer_id)

    assert loaded.layer is not parent
    assert loaded.layer.parent_id == parent.layer_id


def test_refined_run_on_a_locked_parent_is_allowed(loaded, qtbot):
    """Lock is an edit guard on the layer it is set on; spawning a NEW derived layer elsewhere in
    the project never writes to the parent, so it is not what lock exists to refuse (see
    ``MainWindow._on_refined_run``'s own docstring)."""
    parent = loaded.layer
    loaded._on_lock_toggled(parent.layer_id, True)

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded._on_refined_run(parent.layer_id)

    assert loaded.layer is not parent
    assert loaded.layer.parent_id == parent.layer_id
    assert parent.tags.get("ui.lock") == "1"            # the parent itself is untouched


def test_selecting_a_refined_layer_shows_the_fed_by_chip(loaded, qtbot):
    parent = loaded.layer
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded._on_refined_run(parent.layer_id)

    assert loaded.strips.source_box.chip_label.text() == f"◂ fed by {parent.name}"


def test_selecting_an_ordinary_layer_shows_no_chip(loaded, qtbot):
    parent = loaded.layer
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded._on_refined_run(parent.layer_id)             # child now active, chip showing

    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.layer_list.select_layer(parent.layer_id)      # back to the ordinary parent

    assert loaded.strips.source_box.chip_label.text() == ""
    assert loaded.strips.source_box.chip_label.isVisible() is False


def test_hovering_the_chip_flashes_the_parent_row(loaded, qtbot):
    """End-to-end wiring: ``SourceBox.chipHovered`` -> ``WorkflowZone.chipHovered`` ->
    ``MainWindow`` -> ``LayerPanel.flash_row``."""
    parent = loaded.layer
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded._on_refined_run(parent.layer_id)

    loaded.strips.chipHovered.emit(parent.layer_id)

    parent_row = loaded.layer_list._row_widgets[parent.layer_id]
    assert parent_row.property("flash") == "true"


# ------------------------------------------ the dataset's own hide, on the source row (2026-08-29)
# Hiding the dataset while still showing its extrema. The layer row's H hides that
# layer's products (unchanged); the SOURCE header's new H hides the raster itself ("the toggle sits on the source").

def test_source_header_hide_button_emits_sourceHideToggled(panel, project, qtbot):
    source = project.add_source("/data/dataset.tif")
    panel.add_layer_row(_layer(project, "A", source.source_id), None)
    row = panel._source_rows[source.source_id]
    assert row.hide_button.isChecked() is False
    with qtbot.waitSignal(panel.sourceHideToggled, timeout=1000) as sig:
        row.hide_button.click()
    assert sig.args == [source.source_id, True]


def test_source_header_hide_button_seeds_from_the_source_flag(panel, project):
    source = project.add_source("/data/dataset.tif")
    source.hidden = True
    panel.add_layer_row(_layer(project, "A", source.source_id), None)
    assert panel._source_rows[source.source_id].hide_button.isChecked() is True


def test_hiding_the_source_hides_the_raster_but_not_the_layers_products(loaded):
    layer = loaded.layer
    sid = layer.source_id
    assert loaded.canvas.image_item.isVisible()
    loaded._on_source_hide_toggled(sid, True)
    assert loaded.project.sources[sid].hidden is True
    assert not loaded.canvas.image_item.isVisible()
    assert layer.visible is True                                   # products untouched
    loaded._select_layer(layer)                                    # re-selecting keeps it hidden
    assert not loaded.canvas.image_item.isVisible()
    loaded._on_source_hide_toggled(sid, False)
    assert loaded.canvas.image_item.isVisible()


# ------------------------------------------------- master-row collapse (2026-09-18)

def test_lone_raw_master_row_is_hidden_until_a_second_row_or_a_chain(panel, project, qtbot):
    """A fresh open shows ONE row (the header) -- the identically-named master row stays
    hidden while it is the source's only layer with an empty chain (user: it read as "the
    dataset automatically forks a copy of itself"). It appears the moment a child forks, and
    re-collapses when the child is removed."""
    source = project.add_source("/data/dataset.tif")
    master = _layer(project, "dataset", source.source_id)
    panel.add_layer_row(master, None)
    assert panel._layer_items[master.layer_id].isHidden()

    # header click still reaches the hidden master (promotion)
    received = []
    panel.layerSelected.connect(received.append)
    panel.setCurrentItem(panel.topLevelItem(0))
    assert received == [master.layer_id]

    child = project.add_layer("dataset · wtmm2d", source.source_id,
                              parent_id=master.layer_id)
    panel.add_layer_row(child, None)
    assert not panel._layer_items[master.layer_id].isHidden()

    project.layers.remove(child)
    panel.remove_rows([child.layer_id])
    assert panel._layer_items[master.layer_id].isHidden()


def test_master_row_with_a_chain_is_never_hidden(panel, project, qtbot,
                                                 clean_registry, stub_transform):
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.device import register_device
    register_device(stub_transform)
    source = project.add_source("/data/other.tif")
    master = project.add_layer("other", source.source_id,
                               Chain((DeviceRef("t", {"scale": 4}),)))
    panel.add_layer_row(master, None)
    assert not panel._layer_items[master.layer_id].isHidden()
