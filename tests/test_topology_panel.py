# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.topology_panel.TopologyPanel -- pure Qt, offscreen (the runner sets
QT_QPA_PLATFORM=offscreen). No MainWindow here: this widget is exercised standalone, the same
precedent tests/test_arrangement_mask.py sets for MaskRow."""
from __future__ import annotations

from PySide6 import QtWidgets

from dynamix.shell.topology_panel import TopologyPanel
from dynamix.topology.codes import LINE, name_for, permitted


def test_panel_has_a_list_combo_and_link_unlink_buttons(qtbot):
    w = TopologyPanel()
    qtbot.addWidget(w)
    assert isinstance(w._list, QtWidgets.QListWidget)
    assert isinstance(w._code_combo, QtWidgets.QComboBox)
    assert w._link_button.text() == "Link"
    assert w._unlink_button.text() == "Unlink"


def test_code_combo_starts_with_only_the_auto_entry(qtbot):
    w = TopologyPanel()
    qtbot.addWidget(w)
    assert w._code_combo.count() == 1
    assert w._code_combo.itemData(0) is None


def test_set_code_choices_populates_from_permitted_via_name_for(qtbot):
    w = TopologyPanel()
    qtbot.addWidget(w)
    w.set_code_choices(LINE, LINE, 1)      # line-line-in-R^1 has every code named
    allowed = sorted(permitted(LINE, LINE, 1))
    assert w._code_combo.count() == len(allowed) + 1     # +1 for "(auto)"
    for i, code in enumerate(allowed, start=1):
        assert w._code_combo.itemData(i) == code
        assert name_for(code, LINE, LINE, 1) in w._code_combo.itemText(i)


def test_set_code_choices_falls_back_to_the_bare_code_when_unnamed(qtbot):
    """line-line-in-R^2 has no entry in codes._NAMED at all (only R^1 line-line is named) -- every
    combo entry must fall back to the bare integer, per name_for's own documented contract."""
    w = TopologyPanel()
    qtbot.addWidget(w)
    w.set_code_choices(LINE, LINE, 2)
    for i, code in enumerate(sorted(permitted(LINE, LINE, 2)), start=1):
        assert w._code_combo.itemText(i) == f"{code} ({code})"


def test_link_button_emits_none_by_default_for_auto_suggest(qtbot):
    w = TopologyPanel()
    qtbot.addWidget(w)
    received = []
    w.linkRequested.connect(received.append)
    w._link_button.click()
    assert received == [None]


def test_link_button_emits_the_selected_override_code(qtbot):
    w = TopologyPanel()
    qtbot.addWidget(w)
    w.set_code_choices(LINE, LINE, 1)
    w._code_combo.setCurrentIndex(1)       # first real code, right after "(auto)"
    received = []
    w.linkRequested.connect(received.append)
    w._link_button.click()
    assert received == [sorted(permitted(LINE, LINE, 1))[0]]


def test_unlink_button_is_a_noop_with_no_row_selected(qtbot):
    w = TopologyPanel()
    qtbot.addWidget(w)
    received = []
    w.unlinkRequested.connect(received.append)
    w._unlink_button.click()
    assert received == []


def test_unlink_button_emits_the_selected_rows_index(qtbot):
    w = TopologyPanel()
    qtbot.addWidget(w)
    w.set_rows([
        {"code": 31, "kind_a": LINE, "kind_b": LINE, "space_dim": 2,
         "a_id": "L1:t:line:0", "b_id": "L2:t:line:1", "scale_first_contact": 2.0},
        {"code": 287, "kind_a": LINE, "kind_b": LINE, "space_dim": 2,
         "a_id": "L1:t:line:0", "b_id": "L2:t:line:2", "scale_first_contact": None},
    ])
    w._list.setCurrentRow(1)
    received = []
    w.unlinkRequested.connect(received.append)
    w._unlink_button.click()
    assert received == [1]


def test_set_rows_shows_node_ids_and_scale_reading(qtbot):
    w = TopologyPanel()
    qtbot.addWidget(w)
    w.set_rows([{"code": 31, "kind_a": LINE, "kind_b": LINE, "space_dim": 2,
                "a_id": "L1:t:line:0", "b_id": "L2:t:line:1", "scale_first_contact": 2.5}])
    assert w._list.count() == 1
    text = w._list.item(0).text()
    assert "L1:t:line:0" in text and "L2:t:line:1" in text and "2.5" in text


def test_set_rows_with_named_code_shows_the_name(qtbot):
    w = TopologyPanel()
    qtbot.addWidget(w)
    w.set_rows([{"code": 287, "kind_a": LINE, "kind_b": LINE, "space_dim": 1,
                "a_id": "a", "b_id": "b", "scale_first_contact": None}])
    assert "meet" in w._list.item(0).text()


def test_set_rows_replaces_wholesale(qtbot):
    w = TopologyPanel()
    qtbot.addWidget(w)
    w.set_rows([{"code": 31, "kind_a": LINE, "kind_b": LINE, "space_dim": 2,
                "a_id": "a", "b_id": "b", "scale_first_contact": None}])
    w.set_rows([])
    assert w._list.count() == 0


def test_chain_graph_count_shows_dash_when_none(qtbot):
    w = TopologyPanel()
    qtbot.addWidget(w)
    assert "—" in w._chain_graph_label.text()
    w.set_chain_graph_count(5)
    assert w._chain_graph_label.text() == "chain graph: 5 edges"
    w.set_chain_graph_count(None)
    assert "—" in w._chain_graph_label.text()
