# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Layers panel: the stack as a tree; eye, opacity, groups, reasons, selection."""
from __future__ import annotations

import pytest
from PySide6 import QtCore

from dynamix.model.layer_stack import Group, LayerStack, Row
from dynamix.shell.layers_panel import LayersPanel


def _panel(qtbot):
    s = LayerStack()
    s.insert(Group("ds:src0", "dem", auto=True))
    s.insert(Row("src:src0", "raster"), "ds:src0", 0)
    s.insert(Row("prod:1", "product"), "ds:src0", 0)
    p = LayersPanel(s, label_of=lambda rid: {"src:src0": "dem", "prod:1": "h(x)"}.get(rid, rid))
    qtbot.addWidget(p)
    return s, p


def test_the_tree_shows_groups_and_rows_top_first(qtbot):
    _s, p = _panel(qtbot)
    top = p.tree.topLevelItem(0)
    assert top.text(0) == "dem" and [top.child(i).text(0) for i in range(2)] == ["h(x)", "dem"]


@pytest.mark.skip(reason="one opacity slider for the selected row "
                         "(test_the_eye_and_the_opacity_slider_write_the_stack)")
def test_the_eye_and_opacity_write_the_stack(qtbot):
    s, p = _panel(qtbot)
    with qtbot.waitSignal(p.stackChanged):
        p.item_for("prod:1").setCheckState(0, QtCore.Qt.Unchecked)
    assert s.node("prod:1").visible is False
    with qtbot.waitSignal(p.stackChanged):
        p.opacity_spin("prod:1").setValue(0.4)
    assert s.node("prod:1").opacity == 0.4


def test_new_group_ungroup_and_move(qtbot):
    s, p = _panel(qtbot)
    g = p.new_group_above("prod:1", "mine")
    assert s.parent_of("prod:1").id == g.id
    p.ungroup(g.id)
    assert s.parent_of("prod:1").id == "ds:src0"
    p.move("prod:1", "root", 0)
    assert s.root.children[0].id == "prod:1" and s.node("prod:1").moved


def test_reasons_show_greyed(qtbot):
    _s, p = _panel(qtbot)
    p.set_reasons({"prod:1": "no CRS"})
    item = p.item_for("prod:1")
    assert "no CRS" in item.toolTip(0) and item.foreground(0).color().alpha() < 255


@pytest.mark.skip(reason="a click on a row's name selects it; moving the current row (an eye "
                         "click does) selects nothing (test_a_click_on_a_rows_name_selects_it)")
def test_selecting_a_row_emits_its_id(qtbot):
    _s, p = _panel(qtbot)
    with qtbot.waitSignal(p.rowSelected) as sig:
        p.tree.setCurrentItem(p.item_for("prod:1"))
    assert sig.args == ["prod:1"]


def test_a_click_on_a_rows_name_selects_it(qtbot):
    _s, p = _panel(qtbot)
    item = p.item_for("prod:1")
    with qtbot.waitSignal(p.rowSelected) as sig:
        p.tree.itemClicked.emit(item, 0)
    assert sig.args == ["prod:1"]


def test_an_eye_click_selects_nothing(qtbot):
    """Qt makes the clicked row current and toggles its check, then reports the click: the eye
    changes the stack and no layer is selected."""
    _s, p = _panel(qtbot)
    seen = []
    p.rowSelected.connect(seen.append)
    item = p.item_for("prod:1")
    p.tree.setCurrentItem(item)                       # the press
    item.setCheckState(0, QtCore.Qt.Unchecked)        # the release on the check box
    p.tree.itemClicked.emit(item, 0)
    assert seen == []


def test_rows_carry_an_eye_and_a_name_only(qtbot):
    _s, p = _panel(qtbot)
    assert p.tree.columnCount() == 1 and p.tree.isHeaderHidden()
    assert p.tree.itemWidget(p.item_for("prod:1"), 0) is None


def test_the_eye_and_the_opacity_slider_write_the_stack(qtbot):
    s, p = _panel(qtbot)
    with qtbot.waitSignal(p.stackChanged):
        p.item_for("prod:1").setCheckState(0, QtCore.Qt.Unchecked)
    assert s.node("prod:1").visible is False
    p.select_row("prod:1")
    assert p.opacity_slider.isEnabled() and p.opacity_slider.value() == 100
    with qtbot.waitSignal(p.opacityChanged):
        p.opacity_slider.setValue(40)
    assert s.node("prod:1").opacity == 0.4
    assert p.item_for("prod:1").text(0) == "h(x)  40%" and p.opacity_label.text() == "40%"
    p.select_row("src:src0")
    assert p.opacity_slider.value() == 100                      # it follows the selection


def test_up_and_down_move_the_selected_row_in_its_group(qtbot):
    s, p = _panel(qtbot)
    p.select_row("src:src0")
    with qtbot.waitSignal(p.stackChanged):
        p.up_button.click()
    assert [c.id for c in s.node("ds:src0").children] == ["src:src0", "prod:1"]
    assert p.current_id() == "src:src0"
    p.down_button.click()
    assert [c.id for c in s.node("ds:src0").children] == ["prod:1", "src:src0"]
    p.down_button.click()                                       # already last: stays
    assert [c.id for c in s.node("ds:src0").children] == ["prod:1", "src:src0"]


def test_the_grip_resizes_the_list(qtbot):
    _s, p = _panel(qtbot)
    p.set_tree_height(360)
    assert p.tree.height() == 360
    p.set_tree_height(10)
    assert p.tree.height() >= 100


def test_show_only_this_on_a_group_keeps_its_rows(qtbot):
    s, p = _panel(qtbot)
    s.insert(Group("grp:9", "mine"), "root", 0)
    s.insert(Row("ext:2", "extrema"), "grp:9", 0)
    p.refresh()
    p._solo("ds:src0")
    assert {r.id: v for r, v, _o in s.drawn(include_hidden=True)} == {
        "ext:2": False, "src:src0": True, "prod:1": True}


def test_a_name_click_after_a_keyboard_eye_toggle_selects(qtbot):
    s, p = _panel(qtbot)
    p.show()
    seen = []
    p.rowSelected.connect(seen.append)
    item = p.item_for("prod:1")
    p.tree.setCurrentItem(item)
    p.tree.setFocus()
    qtbot.keyClick(p.tree, QtCore.Qt.Key_Space)          # Qt toggles the eye; no click follows
    assert s.node("prod:1").visible is False
    qtbot.mouseClick(p.tree.viewport(), QtCore.Qt.LeftButton,
                     pos=p.tree.visualItemRect(item).center())
    assert seen == ["prod:1"]


def test_a_collapsed_group_stays_collapsed_across_a_refresh(qtbot):
    s, p = _panel(qtbot)
    p.item_for("ds:src0").setExpanded(False)
    s.insert(Group("grp:3", "new"), "root", 0)
    s.insert(Row("ext:5", "extrema"), "grp:3", 0)
    p.refresh()
    assert not p.item_for("ds:src0").isExpanded() and p.item_for("grp:3").isExpanded()
