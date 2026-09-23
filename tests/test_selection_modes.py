# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for the selection-mode row, box mode, ⌥-subtract ops, and
visible canvas selection styling.

Offscreen Qt (``QT_QPA_PLATFORM=offscreen``, mandated repo-wide). Canvas-gesture tests reuse
``tests/test_canvas_pick.py``'s own ``canvas`` fixture and ``_screen_xy`` helper, and
``tests/test_shell_roi_flow.py``'s ``_press``/``_move``/``_release`` synthetic-event builders --
the established cross-file fixture-reuse pattern those files already use throughout.
MainWindow-level tests reuse ``tests/test_shell_window.py``'s ``loaded``/``window``/
``stub_devices`` fixtures. GroupPalette-level tests build a bare palette directly, matching
``tests/test_arrangement_picking.py``'s own ``add_pick`` tests -- kept HERE rather than there
because that module is pyvista-gated (``pytest.importorskip("pyvista")``) even though
``GroupPalette`` itself has no pyvista dependency of its own (its own module docstring).

The end-to-end MainWindow test emits ``Canvas.chainsBoxed`` directly rather than driving a real
pixel-accurate mouse gesture through a full ``MainWindow``'s layout -- the same choice
``tests/test_shell_roi_flow.py``'s own MainWindow-level ROI tests already make (``roi_window.
canvas.roiDrawn.emit(...)``, never a driven drag): the geometry math is independently proven by
the standalone-``canvas``-fixture gesture tests below, so the MainWindow-level test only needs to
prove the WIRING from a landed signal to the palette/canvas/panel.
"""
from __future__ import annotations

from unittest.mock import Mock

import pytest
from PySide6 import QtCore, QtGui

from dynamix.shell.arrangement.group_palette import GroupPalette
from dynamix.shell.canvas import Canvas
from dynamix.shell.theme import RESTRAINED_DARK
from tests.test_canvas_pick import _screen_xy, canvas  # noqa: F401 -- fixture reuse
from tests.test_chain_pick import CHAINS
from tests.test_shell_roi_flow import _move, _press, _release
from tests.test_shell_window import loaded, stub_devices, window  # noqa: F401 -- fixture reuse

_NO_MOD = QtCore.Qt.KeyboardModifier.NoModifier
_ALT = QtCore.Qt.KeyboardModifier.AltModifier

# --------------------------------------------------------------------------- mode state machine


def test_selection_mode_defaults_to_click(qtbot, loaded):
    assert loaded._selection_mode == "click"
    assert loaded.canvas._selection_mode == "click"


def test_set_selection_mode_raises_on_unknown(qtbot, loaded):
    with pytest.raises(ValueError):
        loaded.set_selection_mode("freehand")


def test_set_selection_mode_syncs_the_row_and_the_canvas(qtbot, loaded):
    loaded.set_selection_mode("box")

    assert loaded._selection_mode_buttons["box"].isChecked() is True
    assert loaded._selection_mode_buttons["click"].isChecked() is False
    assert loaded.canvas._selection_mode == "box"


def test_clicking_a_mode_row_button_switches_mode(qtbot, loaded):
    loaded._selection_mode_buttons["lasso"].click()

    assert loaded._selection_mode == "lasso"
    assert loaded.canvas._selection_mode == "lasso"


def test_canvas_set_selection_mode_stores_whatever_it_is_given(qtbot):
    c = Canvas()
    qtbot.addWidget(c)

    c.set_selection_mode("box")

    assert c._selection_mode == "box"


def test_c_hotkey_switches_to_click_mode(qtbot, loaded):
    loaded.set_selection_mode("lasso")
    shortcuts = [s for s in loaded.findChildren(QtGui.QShortcut)
                 if s.key() == QtGui.QKeySequence(QtCore.Qt.Key_C)]
    assert len(shortcuts) == 1
    assert shortcuts[0].context() == QtCore.Qt.WindowShortcut

    shortcuts[0].activated.emit()

    assert loaded._selection_mode == "click"


def test_v_hotkey_cycles_box_lasso_transect_box(qtbot, loaded):
    shortcuts = [s for s in loaded.findChildren(QtGui.QShortcut)
                 if s.key() == QtGui.QKeySequence(QtCore.Qt.Key_V)]
    assert len(shortcuts) == 1
    assert shortcuts[0].context() == QtCore.Qt.WindowShortcut

    order = []
    for _ in range(4):
        shortcuts[0].activated.emit()
        order.append(loaded._selection_mode)

    assert order == ["box", "lasso", "transect", "box"]


def test_v_hotkey_from_click_mode_starts_the_cycle_at_box(qtbot, loaded):
    assert loaded._selection_mode == "click"
    shortcuts = [s for s in loaded.findChildren(QtGui.QShortcut)
                 if s.key() == QtGui.QKeySequence(QtCore.Qt.Key_V)]

    shortcuts[0].activated.emit()

    assert loaded._selection_mode == "box"


# --------------------------------------------------------------------------- box mode gesture


def test_box_drag_in_box_mode_emits_chains_boxed(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)
    canvas.set_selection_mode("box")
    p0 = _screen_xy(canvas, 5.0, 3.0)
    p1 = _screen_xy(canvas, 15.0, 9.0)          # encloses chains 0 and 2, same box as CHAINS'
                                                  # own lasso fixture in test_chain_pick.py

    with qtbot.waitSignal(canvas.chainsBoxed, timeout=1000) as sig:
        canvas.mousePressEvent(_press(p0, _NO_MOD))
        canvas.mouseMoveEvent(_move(p1, _NO_MOD))
        canvas.mouseReleaseEvent(_release(p1, _NO_MOD))

    assert sig.args == [[0, 2], False]
    bx, _ = canvas.box_item.getData()
    assert bx is None or bx.size == 0           # the rubber band is removed on release


def test_box_drag_with_alt_subtracts(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)
    canvas.set_selection_mode("box")
    p0 = _screen_xy(canvas, 5.0, 3.0)
    p1 = _screen_xy(canvas, 15.0, 9.0)

    with qtbot.waitSignal(canvas.chainsBoxed, timeout=1000) as sig:
        canvas.mousePressEvent(_press(p0, _ALT))
        canvas.mouseMoveEvent(_move(p1, _ALT))
        canvas.mouseReleaseEvent(_release(p1, _ALT))

    assert sig.args == [[0, 2], True]


def test_box_mode_alt_drag_never_starts_a_lasso(qtbot, canvas):
    """⌥ held during a box-mode drag still resolves via ``chains_in_box``, not
    ``chains_in_polygon`` -- the click-mode-only ⌥-lasso shortcut is gated off in every OTHER
    mode (``Canvas.mousePressEvent``'s own comment), so it must never compete with box mode."""
    canvas.set_pick_chains(CHAINS)
    canvas.set_selection_mode("box")
    emitted_lasso = []
    canvas.chainsLassoed.connect(lambda *a: emitted_lasso.append(a))
    p0 = _screen_xy(canvas, 5.0, 3.0)
    p1 = _screen_xy(canvas, 15.0, 9.0)

    with qtbot.waitSignal(canvas.chainsBoxed, timeout=1000):
        canvas.mousePressEvent(_press(p0, _ALT))
        canvas.mouseMoveEvent(_move(p1, _ALT))
        canvas.mouseReleaseEvent(_release(p1, _ALT))

    assert emitted_lasso == []


def test_box_mode_not_active_in_click_mode(qtbot, canvas):
    """A plain drag in click mode (default) never emits ``chainsBoxed`` -- box is only live
    while the mode is actually "box"."""
    canvas.set_pick_chains(CHAINS)
    assert canvas._selection_mode == "click"
    emitted = []
    canvas.chainsBoxed.connect(lambda *a: emitted.append(a))
    p0 = _screen_xy(canvas, 5.0, 3.0)
    p1 = _screen_xy(canvas, 15.0, 9.0)

    canvas.mousePressEvent(_press(p0, _NO_MOD))
    canvas.mouseMoveEvent(_move(p1, _NO_MOD))
    canvas.mouseReleaseEvent(_release(p1, _NO_MOD))

    assert emitted == []


# --------------------------------------------------------------------------- lasso mode gesture


def test_lasso_mode_drag_without_alt_lassos(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)
    canvas.set_selection_mode("lasso")
    verts = [(5.0, 3.0), (15.0, 3.0), (15.0, 9.0), (5.0, 9.0)]        # encloses chains 0 and 2
    screen = [_screen_xy(canvas, dx, dy) for dx, dy in verts]

    with qtbot.waitSignal(canvas.chainsLassoed, timeout=1000) as sig:
        canvas.mousePressEvent(_press(screen[0], _NO_MOD))
        for pt in screen[1:]:
            canvas.mouseMoveEvent(_move(pt, _NO_MOD))
        canvas.mouseReleaseEvent(_release(screen[-1], _NO_MOD))

    assert sig.args == [[0, 2], False]


def test_lasso_mode_drag_with_alt_subtracts(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)
    canvas.set_selection_mode("lasso")
    verts = [(5.0, 3.0), (15.0, 3.0), (15.0, 9.0), (5.0, 9.0)]
    screen = [_screen_xy(canvas, dx, dy) for dx, dy in verts]

    with qtbot.waitSignal(canvas.chainsLassoed, timeout=1000) as sig:
        canvas.mousePressEvent(_press(screen[0], _ALT))
        for pt in screen[1:]:
            canvas.mouseMoveEvent(_move(pt, _ALT))
        canvas.mouseReleaseEvent(_release(screen[-1], _ALT))

    assert sig.args == [[0, 2], True]


def test_click_mode_alt_shortcut_lasso_still_works_and_always_adds(qtbot, canvas):
    """The pre-existing ⌥-drag shortcut stays working in click
    mode -- and always reports ``subtract=False``, since Alt is what triggers it there at all."""
    canvas.set_pick_chains(CHAINS)
    assert canvas._selection_mode == "click"
    verts = [(5.0, 3.0), (15.0, 3.0), (15.0, 9.0), (5.0, 9.0)]
    screen = [_screen_xy(canvas, dx, dy) for dx, dy in verts]

    with qtbot.waitSignal(canvas.chainsLassoed, timeout=1000) as sig:
        canvas.mousePressEvent(_press(screen[0], _ALT))
        for pt in screen[1:]:
            canvas.mouseMoveEvent(_move(pt, _ALT))
        canvas.mouseReleaseEvent(_release(screen[-1], _ALT))

    assert sig.args == [[0, 2], False]


# --------------------------------------------------------------------------- plain drag still pans


def test_click_mode_plain_drag_still_pans_and_fires_no_box_or_lasso(qtbot, canvas):
    """Sibling of ``test_canvas_pick.py``'s own ``test_a_plain_drag_still_pans_and_emits_no_
    pick``, extended to assert the new box/lasso signals never fire in click mode either."""
    canvas.set_pick_chains(CHAINS)
    assert canvas._selection_mode == "click"
    emitted_boxed, emitted_lassoed = [], []
    canvas.chainsBoxed.connect(lambda *a: emitted_boxed.append(a))
    canvas.chainsLassoed.connect(lambda *a: emitted_lassoed.append(a))
    before = [list(axis) for axis in canvas.view.viewRange()]

    canvas.mousePressEvent(_press((140.0, 160.0), _NO_MOD))
    canvas.mouseMoveEvent(_move((300.0, 280.0), _NO_MOD))
    canvas.mouseReleaseEvent(_release((300.0, 280.0), _NO_MOD))

    assert emitted_boxed == []
    assert emitted_lassoed == []
    assert [list(axis) for axis in canvas.view.viewRange()] != before


# --------------------------------------------------------------------------- GroupPalette.apply_picks


def test_apply_picks_add_unions_into_selection_and_active_group(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)
    g = palette.new_group("g")

    palette.apply_picks([(1, 0), (1, 2)], "add")

    assert palette.selection() == {(1, 0), (1, 2)}
    assert palette.groups()[g]["chains"] == [(1, 0), (1, 2)]


def test_apply_picks_subtract_shrinks_selection_but_not_membership(qtbot):
    """The membership doctrine (group_palette.py's own docstring): subtract operates on the
    SELECTION buffer only -- a group's membership never shrinks through this op."""
    palette = GroupPalette()
    qtbot.addWidget(palette)
    g = palette.new_group("g")
    palette.apply_picks([(1, 0), (1, 2)], "add")

    palette.apply_picks([(1, 0)], "subtract")

    assert palette.selection() == {(1, 2)}
    assert palette.groups()[g]["chains"] == [(1, 0), (1, 2)]      # unchanged


def test_apply_picks_replace_replaces_the_whole_selection(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)
    palette.apply_picks([(1, 0)], "add")

    palette.apply_picks([(1, 9)], "replace")

    assert palette.selection() == {(1, 9)}


def test_apply_picks_unknown_op_raises(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)

    with pytest.raises(ValueError):
        palette.apply_picks([(1, 0)], "frobnicate")


def test_apply_picks_always_emits_membership_changed(qtbot):
    palette = GroupPalette()
    qtbot.addWidget(palette)

    with qtbot.waitSignal(palette.membershipChanged, timeout=1000):
        palette.apply_picks([], "add")             # a degenerate empty box/lasso -- still emits


def test_add_pick_still_delegates_to_apply_picks_for_a_hit(qtbot):
    """``add_pick`` is now a thin wrapper (the design's "existing add_pick delegates")."""
    palette = GroupPalette()
    qtbot.addWidget(palette)
    palette.apply_picks = Mock(wraps=palette.apply_picks)

    palette.add_pick((1, 5), shift=False)

    palette.apply_picks.assert_called_once_with([(1, 5)], "add")


# --------------------------------------------------------------------------- visible selection styling


def test_set_selection_chains_draws_the_selected_trails_in_the_accent_at_2x_width(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)

    canvas.set_selection_chains([0, 2])

    pen = canvas.selection_item.opts["pen"]
    assert pen.color().name() == RESTRAINED_DARK.selection_accent
    assert pen.widthF() == pytest.approx(2.0)          # default overlay line width is 1.0
    sx, _ = canvas.selection_item.getData()
    assert sx.size > 0


def test_set_selection_chains_none_clears_the_overlay(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)
    canvas.set_selection_chains([0])

    canvas.set_selection_chains(None)

    sx, _ = canvas.selection_item.getData()
    assert sx is None or sx.size == 0


def test_set_selection_chains_empty_list_clears_the_overlay(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)
    canvas.set_selection_chains([0])

    canvas.set_selection_chains([])

    sx, _ = canvas.selection_item.getData()
    assert sx is None or sx.size == 0


def test_set_display_style_doubles_the_selection_pen_width(qtbot, canvas):
    canvas.set_display_style(line_width=3.0)

    pen = canvas.selection_item.opts["pen"]
    assert pen.widthF() == pytest.approx(6.0)


def test_clear_overlays_clears_the_selection_item_too(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)
    canvas.set_selection_chains([0])

    canvas.clear_overlays()

    sx, _ = canvas.selection_item.getData()
    assert sx is None or sx.size == 0


# --------------------------------------------------------------------------- MainWindow end-to-end


def test_box_select_end_to_end_updates_palette_and_canvas(qtbot, loaded):
    loaded.canvas.set_pick_chains(CHAINS)

    loaded.canvas.chainsBoxed.emit([0, 2], False)

    lid = loaded.layer.layer_id
    assert loaded._group_palette.selection() == {(lid, 0), (lid, 2)}
    sx, _ = loaded.canvas.selection_item.getData()
    assert sx.size > 0


def test_box_select_alt_subtracts_end_to_end(qtbot, loaded):
    loaded.canvas.set_pick_chains(CHAINS)
    loaded.canvas.chainsBoxed.emit([0, 2], False)

    loaded.canvas.chainsBoxed.emit([0], True)

    lid = loaded.layer.layer_id
    assert loaded._group_palette.selection() == {(lid, 2)}


def test_groups_section_is_not_view_scoped_and_visible_at_raster_stack_index(qtbot, loaded):
    loaded.show()

    # Walk up from the palette to its section frame and check the `view_scoped` property the
    # same way `RightPanel.add_section` itself stamps it (mirrors `tests/test_right_panel.py`'s
    # own Topology check, which pins the CONTRASTING case -- Groups used to be True there).
    frame = loaded._group_palette.parentWidget()
    while frame is not None and frame.property("view_scoped") is None:
        frame = frame.parentWidget()
    assert frame is not None
    assert frame.property("view_scoped") is False

    assert loaded._center_stack.currentIndex() == 0            # raster view is showing
    assert loaded._group_palette.isVisible()
    assert loaded._commit_button.isVisible()
