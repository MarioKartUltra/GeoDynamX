# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The rack editor (dynamix.shell.rack_editor): add by drag, reorder by drag, the zone's knobs,
live validation from a chosen input, and Save / Save as / Delete / Duplicate to edit.

Offscreen Qt. A device drop calls ``dropEvent`` with a stub event; a drag reorder is the list
model's ``moveRow``, which is what the list's own internal drag calls."""
from __future__ import annotations

import pytest
from PySide6 import QtCore, QtWidgets

from dynamix.devices import register_builtin_devices


@pytest.fixture
def registered_builtins(clean_registry):
    register_builtin_devices()
    return clean_registry


class _StubDropEvent:
    def __init__(self, mime: QtCore.QMimeData):
        self._mime = mime
        self.accepted: bool | None = None

    def position(self):
        return QtCore.QPointF(0, 0)

    def mimeData(self):
        return self._mime

    def proposedAction(self):
        return QtCore.Qt.MoveAction

    def acceptProposedAction(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


def _device_drop(name: str) -> _StubDropEvent:
    from dynamix.shell.browser import DEVICE_MIME

    mime = QtCore.QMimeData()
    mime.setData(DEVICE_MIME, name.encode("utf-8"))
    return _StubDropEvent(mime)


@pytest.fixture
def store(tmp_path):
    return tmp_path / "racks"


def _editor(qtbot, store, name=None):
    from dynamix.model.racks import RackEntry, builtin_racks, load_racks
    from dynamix.shell.rack_editor import RackEditor

    entries = [RackEntry(r, None) for r in builtin_racks()] + load_racks(store)
    editor = RackEditor(store, entries, name=name)
    qtbot.addWidget(editor)
    return editor


def _devices(editor):
    return [d["device"] for d in editor.descriptors()]


def test_a_device_dragged_from_the_tree_becomes_a_step_with_its_knobs(
        qtbot, registered_builtins, store):
    from dynamix.model.device import defaults_for, get_device

    editor = _editor(qtbot, store)
    event = _device_drop("wtmm2d")
    editor.steps_list.dropEvent(event)
    assert event.accepted is True
    assert _devices(editor) == ["wtmm2d"]
    assert editor.descriptors()[0]["params"] == defaults_for(get_device("wtmm2d"))
    assert editor.knob_box is not None and editor.knob_box.device.name == "wtmm2d"
    assert "wtmm2d" in editor.steps_list.item(0).text()


def test_reordering_by_drag_reorders_the_rack_and_revalidates(qtbot, registered_builtins,
                                                               store):
    editor = _editor(qtbot, store)
    for name in ("wtmm2d", "chain_topology", "chain_holder"):
        editor.add_step(name)
    editor.name_edit.setText("R")
    model = editor.steps_list.model()
    model.moveRow(QtCore.QModelIndex(), 2, QtCore.QModelIndex(), 1)    # a filter before a transform
    assert _devices(editor) == ["wtmm2d", "chain_holder", "chain_topology"]
    assert "transforms-then-filters" in editor.validation_label.text()
    assert not editor.save_button.isEnabled()
    model.moveRow(QtCore.QModelIndex(), 2, QtCore.QModelIndex(), 1)
    assert _devices(editor) == ["wtmm2d", "chain_topology", "chain_holder"]
    assert editor.save_button.isEnabled()


def test_the_knobs_edit_the_selected_step(qtbot, registered_builtins, store):
    editor = _editor(qtbot, store)
    editor.add_step("wtmm2d")
    editor.add_step("chain_holder")
    editor.steps_list.setCurrentRow(0)
    editor.knob_box._on_control_changed("n_oct", 5)          # the path a knob drag takes
    assert editor.descriptors()[0]["params"]["n_oct"] == 5
    editor.steps_list.setCurrentRow(1)
    assert editor.knob_box.device.name == "chain_holder"


def test_validation_names_a_step_that_does_not_apply_from_the_chosen_input(
        qtbot, registered_builtins, store):
    from dynamix.model.contracts import CONTRACTS, STACK, VCHAINS

    editor = _editor(qtbot, store)
    editor.add_step("mz_edges")
    editor.add_step("chain_holder")
    assert CONTRACTS[VCHAINS] in editor.validation_label.text()
    assert "needs" in editor.steps_list.item(1).toolTip()
    editor.input_combo.setCurrentIndex(editor.input_combo.findData(STACK))
    assert "band stack" in editor.validation_label.text()


def test_the_bypass_box_marks_a_step_bypassed_and_save_keeps_it(qtbot, registered_builtins,
                                                                store):
    from dynamix.model.racks import load_racks

    editor = _editor(qtbot, store)
    editor.add_step("wtmm2d")
    editor.add_step("chain_holder")
    editor.steps_list.item(1).setCheckState(QtCore.Qt.Checked)
    assert editor.descriptors()[1]["bypassed"] is True
    editor.name_edit.setText("My WTMM")
    with qtbot.waitSignal(editor.racksChanged):
        assert editor.save() is True
    (entry,) = load_racks(store)
    assert entry.rack.name == "My WTMM" and entry.problem == ""
    assert [(s.device, s.bypassed) for s in entry.rack.steps] == [("wtmm2d", False),
                                                                 ("chain_holder", True)]
    assert entry.rack.steps[0].params["n_oct"] == 3              # every param, filled


def test_a_builtin_rack_is_read_only_until_duplicated(qtbot, registered_builtins, store):
    from dynamix.model.presets import PRESETS
    from dynamix.model.racks import load_racks

    editor = _editor(qtbot, store, name="WTMM + Hölder")
    assert _devices(editor) == [d for d, _ in PRESETS["WTMM + Hölder"]]
    assert not editor.save_button.isEnabled() and not editor.delete_button.isEnabled()
    assert editor.duplicate_button.isEnabled()
    editor.add_step("noise")
    assert len(editor.descriptors()) == len(PRESETS["WTMM + Hölder"])
    editor.duplicate()
    assert editor.name_edit.text() == "WTMM + Hölder copy"
    assert editor.save_button.isEnabled() and not editor.duplicate_button.isEnabled()
    assert editor.save() is True
    assert [e.rack.name for e in load_racks(store)] == ["WTMM + Hölder copy"]


def test_save_refuses_a_builtin_name(qtbot, registered_builtins, store):
    from dynamix.model.racks import load_racks

    editor = _editor(qtbot, store)
    editor.add_step("wtmm2d")
    editor.name_edit.setText("M–Z edges")
    assert editor.save() is False
    assert "built-in" in editor.status_label.text()
    assert load_racks(store) == []


def test_save_as_copies_and_delete_removes(qtbot, registered_builtins, store, monkeypatch):
    from dynamix.model.racks import load_racks

    editor = _editor(qtbot, store)
    editor.add_step("wtmm2d")
    editor.name_edit.setText("First")
    assert editor.save() is True
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("Second", True)))
    assert editor.save_as() is True
    assert sorted(e.rack.name for e in load_racks(store)) == ["First", "Second"]
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
    with qtbot.waitSignal(editor.racksChanged):
        editor.delete()
    assert [e.rack.name for e in load_racks(store)] == ["First"]
    assert editor.descriptors() == [] and editor.name_edit.text() == ""


def test_the_rack_box_opens_a_saved_rack(qtbot, registered_builtins, store):
    from dynamix.model.racks import Rack, RackStep, save_rack

    save_rack(store, Rack("Saved", (RackStep("mz_edges", {"n_levels": 3}),
                                    RackStep("hline_length", {}))))
    editor = _editor(qtbot, store)
    editor.rack_combo.setCurrentIndex(editor.rack_combo.findData("Saved"))
    assert _devices(editor) == ["mz_edges", "hline_length"]
    assert editor.name_edit.text() == "Saved" and editor.delete_button.isEnabled()


def test_the_window_opens_the_editor_and_lists_what_it_saves(qtbot, registered_builtins):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    action = next(a for m in win.menuBar().actions() if m.text() == "Racks"
                  for a in m.menu().actions() if a.text() == "Rack Editor…")
    action.trigger()
    editor = win._rack_editor
    assert editor is not None and editor.isVisible()
    editor.add_step("wtmm2d")
    editor.name_edit.setText("From the editor")
    assert editor.save() is True
    racks = win.browser._categories["Racks"]
    assert "From the editor" in [racks.child(i).text(0) for i in range(racks.childCount())]
    editor.close()
    win.browser.rackEditRequested.emit("WTMM + Hölder")
    assert win._rack_editor.name_edit.text() == "WTMM + Hölder"
    win._rack_editor.close()
