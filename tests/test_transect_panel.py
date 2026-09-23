# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.transect_panel.TransectPanel -- pure Qt, offscreen (the runner sets
QT_QPA_PLATFORM=offscreen). No MainWindow here: this widget is exercised standalone, the same
precedent tests/test_topology_panel.py already sets."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from dynamix.model.project import TransectRecord
from dynamix.shell.transect_panel import TransectPanel


def _panel(qtbot):
    w = TransectPanel()
    qtbot.addWidget(w)
    return w


# --------------------------------------------------------------------------- add / ids


def test_panel_starts_empty(qtbot):
    w = _panel(qtbot)
    assert w._list.count() == 0
    assert w.records() == []
    assert w.selected_id() is None


def test_add_record_appends_names_and_selects_the_new_row(qtbot):
    w = _panel(qtbot)
    received = []
    w.selectionChanged.connect(received.append)

    record = w.add_record((1.0, 2.0), (3.0, 4.0))

    assert record.transect_id == 1
    assert record.a == (1.0, 2.0) and record.b == (3.0, 4.0)
    assert record.visible is True
    assert w._list.count() == 1
    assert w._list.item(0).text() == "T1"
    assert w.selected_id() == 1
    assert received[-1] is record                                # newly-added row auto-selected


def test_ids_are_monotonic_and_never_reused_after_a_middle_delete(qtbot):
    """The EQSelect collision lesson: delete a MIDDLE row, add a new one, and confirm the
    new row's id is strictly past every id ever handed out -- never a value freed up by the
    delete."""
    w = _panel(qtbot)
    r1 = w.add_record((0.0, 0.0), (1.0, 1.0))
    r2 = w.add_record((0.0, 0.0), (1.0, 1.0))
    r3 = w.add_record((0.0, 0.0), (1.0, 1.0))
    assert (r1.transect_id, r2.transect_id, r3.transect_id) == (1, 2, 3)

    w._list.setCurrentRow(1)                       # r2, the middle row
    w.delete_selected()

    r4 = w.add_record((0.0, 0.0), (1.0, 1.0))
    assert r4.transect_id == 4                      # not 2 -- the freed id is never reused
    assert [r.transect_id for r in w.records()] == [1, 3, 4]


# --------------------------------------------------------------------------- selection / highlight


def test_selecting_a_row_pushes_its_buffer_into_the_spin_and_emits_the_record(qtbot):
    w = _panel(qtbot)
    a = w.add_record((0.0, 0.0), (1.0, 0.0))
    a.buffer_px = 77.0
    b = w.add_record((0.0, 0.0), (2.0, 0.0))          # now selected
    received = []
    w.selectionChanged.connect(received.append)

    w._list.setCurrentRow(0)                          # select the FIRST row (record a)

    assert received[-1] is a
    assert w._buffer_spin.value() == 77


def test_deleting_the_last_row_selects_nothing(qtbot):
    w = _panel(qtbot)
    w.add_record((0.0, 0.0), (1.0, 0.0))
    received = []
    w.selectionChanged.connect(received.append)

    w.delete_selected()

    assert received[-1] is None
    assert w.selected_id() is None


# --------------------------------------------------------------------------- visibility / buffer


def test_visibility_checkbox_toggle_updates_the_record_and_emits(qtbot):
    w = _panel(qtbot)
    record = w.add_record((0.0, 0.0), (1.0, 0.0))
    received = []
    w.recordsChanged.connect(received.append)

    item = w._list.item(0)
    item.setCheckState(QtCore.Qt.Unchecked)

    assert record.visible is False
    assert received and received[-1][0].visible is False


def test_buffer_spin_edit_writes_the_selected_records_buffer_and_reemits_selection(qtbot):
    w = _panel(qtbot)
    record = w.add_record((0.0, 0.0), (1.0, 0.0))
    selections = []
    w.selectionChanged.connect(selections.append)

    w._buffer_spin.setValue(123)

    assert record.buffer_px == 123.0
    assert selections[-1] is record                  # re-selected at the new buffer


def test_buffer_spin_is_a_noop_with_nothing_selected(qtbot):
    w = _panel(qtbot)
    received = []
    w.recordsChanged.connect(received.append)

    w._buffer_spin.setValue(50)

    assert received == []


# --------------------------------------------------------------------------- plot


def test_plot_button_emits_the_selected_record(qtbot):
    w = _panel(qtbot)
    record = w.add_record((0.0, 0.0), (1.0, 0.0))
    received = []
    w.plotRequested.connect(received.append)

    w._plot_button.click()

    assert received == [record]


def test_plot_button_is_a_noop_with_nothing_selected(qtbot):
    w = _panel(qtbot)
    received = []
    w.plotRequested.connect(received.append)

    w._plot_button.click()

    assert received == []


def test_double_click_emits_plot_requested(qtbot):
    w = _panel(qtbot)
    record = w.add_record((0.0, 0.0), (1.0, 0.0))
    received = []
    w.plotRequested.connect(received.append)

    w._on_item_double_clicked(w._list.item(0))        # QTest.mouseDClick is flaky offscreen;
                                                        # this is the exact slot Qt would invoke.

    assert received == [record]


# --------------------------------------------------------------------------- delete + undo


def test_delete_selected_is_a_noop_with_nothing_selected(qtbot):
    w = _panel(qtbot)
    received = []
    w.recordsChanged.connect(received.append)

    w.delete_selected()

    assert received == []


def test_delete_removes_the_row_and_undo_restores_it_with_the_same_id(qtbot):
    w = _panel(qtbot)
    record = w.add_record((5.0, 6.0), (7.0, 8.0))
    record.buffer_px = 42.0

    w.delete_selected()
    assert w.records() == []

    w.undo()

    (restored,) = w.records()
    assert restored.transect_id == record.transect_id
    assert restored.a == (5.0, 6.0) and restored.b == (7.0, 8.0)
    assert restored.buffer_px == 42.0
    assert w._list.item(0).text() == f"T{record.transect_id}"


def test_undo_is_a_noop_with_an_empty_stack(qtbot):
    w = _panel(qtbot)
    received = []
    w.recordsChanged.connect(received.append)

    w.undo()

    assert received == []


def test_undo_stack_of_two_restores_in_reverse_order(qtbot):
    w = _panel(qtbot)
    w.add_record((0.0, 0.0), (1.0, 0.0))               # T1
    w.add_record((0.0, 0.0), (2.0, 0.0))               # T2

    w._list.setCurrentRow(0)
    w.delete_selected()                                 # removes T1
    w._list.setCurrentRow(0)
    w.delete_selected()                                 # removes T2 (only row left)
    assert w.records() == []

    w.undo()                                             # restores T2 first (LIFO)
    assert [r.transect_id for r in w.records()] == [2]

    w.undo()                                             # then T1
    assert sorted(r.transect_id for r in w.records()) == [1, 2]

    w.undo()                                             # stack now empty -- a no-op
    assert sorted(r.transect_id for r in w.records()) == [1, 2]


# --------------------------------------------------------------------------- records / set_records


def test_records_reflects_on_screen_row_order(qtbot):
    w = _panel(qtbot)
    w.add_record((0.0, 0.0), (1.0, 0.0))
    w.add_record((0.0, 0.0), (2.0, 0.0))
    assert [r.transect_id for r in w.records()] == [1, 2]


def test_set_records_is_silent_and_rebuilds_the_list(qtbot):
    w = _panel(qtbot)
    received = []
    w.recordsChanged.connect(received.append)
    w.selectionChanged.connect(received.append)

    records = [
        TransectRecord(transect_id=5, a=(1.0, 1.0), b=(2.0, 2.0), visible=False, buffer_px=9.0),
        TransectRecord(transect_id=8, a=(3.0, 3.0), b=(4.0, 4.0)),
    ]
    w.set_records(records)

    assert received == []                              # silent -- no signal fired
    assert [r.transect_id for r in w.records()] == [5, 8]
    assert w._list.item(0).checkState() == QtCore.Qt.Unchecked
    assert w._list.item(1).checkState() == QtCore.Qt.Checked


def test_set_records_resumes_the_id_counter_past_the_highest_restored_id(qtbot):
    w = _panel(qtbot)
    w.set_records([TransectRecord(transect_id=7, a=(0.0, 0.0), b=(1.0, 1.0))])

    record = w.add_record((0.0, 0.0), (1.0, 0.0))

    assert record.transect_id == 8


def test_set_records_with_an_empty_list_clears_the_panel(qtbot):
    w = _panel(qtbot)
    w.add_record((0.0, 0.0), (1.0, 0.0))

    w.set_records([])

    assert w.records() == []
    assert w._list.count() == 0
