# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for raster-canvas picking (click / shift-click /
⌥-lasso) into the SAME group palette the arrangement views use.

Offscreen Qt (``QT_QPA_PLATFORM=offscreen``, mandated repo-wide). Gesture-level: handlers are
driven with synthesized ``QMouseEvent``s exactly the way ``tests/test_shell_roi_flow.py`` already
drives the ⌘-drag ROI gesture on this same widget (pyqtgraph's ``ViewBox`` sees mouse input through
the graphics SCENE, and a real ``QTest.mouse*`` drag through that chain is flaky offscreen in a way
that would make a red here mean nothing) -- the ``_press``/``_move``/``_release`` helpers are
imported straight from that file, the established cross-file fixture-reuse pattern
``tests/test_arrangement_picking.py`` already uses throughout. ``CHAINS`` is the fixture
(``tests/test_chain_pick.py``), reused rather than redefined for the same reason.

The MainWindow-wiring half (bottom of this file) follows ``tests/test_arrangement_picking.py``'s
own ArrangementView-wiring section: mock ``GroupPalette.add_pick`` and assert what it was called
with, rather than asserting on ``GroupPalette``'s own (separately tested) internal state.
"""
from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from PySide6 import QtCore

from dynamix.engine import Renderable
from tests.test_chain_pick import CHAINS
from tests.test_shell_roi_flow import _move, _press, _release
from tests.test_shell_window import loaded, stub_devices, window  # noqa: F401

# --------------------------------------------------------------------------- canvas fixture

_FIELD_SHAPE = (60, 60)
_FIELD = np.zeros(_FIELD_SHAPE)


@pytest.fixture
def canvas(qtbot):
    from dynamix.shell.canvas import Canvas

    c = Canvas()
    qtbot.addWidget(c)
    c.resize(600, 600)
    c.set_field(_FIELD)
    c.view.setRange(xRange=(0, _FIELD_SHAPE[1]), yRange=(0, _FIELD_SHAPE[0]), padding=0)
    return c


def _screen_xy(canvas, dx, dy):
    """DATA coords -> widget/scene px -- the forward map, the algebraic inverse of
    ``test_shell_roi_flow.py``'s own ``_data_from_geometry`` (proven correct there against real
    ``roiDrawn`` output on this same ``invertY(True)`` canvas)."""
    box = canvas.view.sceneBoundingRect()
    (x0, x1), (y0, y1) = canvas.view.viewRange()
    fx = (dx - x0) / (x1 - x0)
    fy = (dy - y0) / (y1 - y0)
    return box.left() + fx * box.width(), box.top() + fy * box.height()


# --------------------------------------------------------------------------- click / shift-click


def test_click_at_a_chain_point_emits_chain_picked(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)
    x, y = _screen_xy(canvas, 11.0, 5.0)          # chain 0's middle point

    with qtbot.waitSignal(canvas.chainPicked, timeout=1000) as sig:
        canvas.mousePressEvent(_press((x, y), QtCore.Qt.KeyboardModifier.NoModifier))
        canvas.mouseReleaseEvent(_release((x, y), QtCore.Qt.KeyboardModifier.NoModifier))

    assert sig.args == [0, False]


def test_shift_click_emits_shift_true(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)
    x, y = _screen_xy(canvas, 11.0, 5.0)
    shift = QtCore.Qt.KeyboardModifier.ShiftModifier

    with qtbot.waitSignal(canvas.chainPicked, timeout=1000) as sig:
        canvas.mousePressEvent(_press((x, y), shift))
        canvas.mouseReleaseEvent(_release((x, y), shift))

    assert sig.args == [0, True]


def test_a_far_click_emits_none(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)
    x, y = _screen_xy(canvas, 25.0, 25.0)          # far from every chain

    with qtbot.waitSignal(canvas.chainPicked, timeout=1000) as sig:
        canvas.mousePressEvent(_press((x, y), QtCore.Qt.KeyboardModifier.NoModifier))
        canvas.mouseReleaseEvent(_release((x, y), QtCore.Qt.KeyboardModifier.NoModifier))

    assert sig.args == [None, False]


def test_a_click_with_no_chains_ever_loaded_still_emits_a_miss(qtbot, canvas):
    """``set_pick_chains`` never called -- a plain click over empty data still fires the miss, the
    same "click empty space to deselect" contract ``GroupPalette.add_pick`` documents for a
    plain-click miss."""
    x, y = _screen_xy(canvas, 25.0, 25.0)

    with qtbot.waitSignal(canvas.chainPicked, timeout=1000) as sig:
        canvas.mousePressEvent(_press((x, y), QtCore.Qt.KeyboardModifier.NoModifier))
        canvas.mouseReleaseEvent(_release((x, y), QtCore.Qt.KeyboardModifier.NoModifier))

    assert sig.args == [None, False]


def test_small_travel_still_counts_as_a_click(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)
    x, y = _screen_xy(canvas, 11.0, 5.0)
    no_mod = QtCore.Qt.KeyboardModifier.NoModifier

    with qtbot.waitSignal(canvas.chainPicked, timeout=1000) as sig:
        canvas.mousePressEvent(_press((x, y), no_mod))
        canvas.mouseReleaseEvent(_release((x + 2.0, y), no_mod))     # 2px travel: still a click

    assert sig.args[0] == 0


def test_travel_past_the_threshold_suppresses_the_pick(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)
    x, y = _screen_xy(canvas, 11.0, 5.0)
    no_mod = QtCore.Qt.KeyboardModifier.NoModifier
    emitted = []
    canvas.chainPicked.connect(lambda *a: emitted.append(a))

    canvas.mousePressEvent(_press((x, y), no_mod))
    canvas.mouseReleaseEvent(_release((x + 5.0, y), no_mod))          # 5px travel: a drag, not a click

    assert emitted == []


# --------------------------------------------------------------------------- ⌥-lasso


def test_alt_drag_lassos_the_enclosed_chains(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)
    alt = QtCore.Qt.KeyboardModifier.AltModifier
    verts = [(5.0, 3.0), (15.0, 3.0), (15.0, 9.0), (5.0, 9.0)]         # encloses chains 0 and 2
    screen = [_screen_xy(canvas, dx, dy) for dx, dy in verts]

    with qtbot.waitSignal(canvas.chainsLassoed, timeout=1000) as sig:
        canvas.mousePressEvent(_press(screen[0], alt))
        for pt in screen[1:]:
            canvas.mouseMoveEvent(_move(pt, alt))
        canvas.mouseReleaseEvent(_release(screen[-1], alt))

    # subtract=False: this is the click-mode ⌥-SHORTCUT (default mode is "click"), where Alt
    # only ever TRIGGERS the gesture and can never also mean "subtract" -- see
    # Canvas.mousePressEvent's own comment.
    assert sig.args == [[0, 2], False]
    lx, _ = canvas.lasso_item.getData()
    assert lx is None or lx.size == 0                # the temp polyline is removed on release


def test_a_short_alt_drag_lassos_nothing(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)
    alt = QtCore.Qt.KeyboardModifier.AltModifier
    x, y = _screen_xy(canvas, 25.0, 25.0)

    with qtbot.waitSignal(canvas.chainsLassoed, timeout=1000) as sig:
        canvas.mousePressEvent(_press((x, y), alt))
        canvas.mouseReleaseEvent(_release((x, y), alt))              # a degenerate 1-point polygon

    assert sig.args == [[], False]


# --------------------------------------------------------------------------- plain drag still pans


def test_a_plain_drag_still_pans_and_emits_no_pick(qtbot, canvas):
    canvas.set_pick_chains(CHAINS)
    no_mod = QtCore.Qt.KeyboardModifier.NoModifier
    emitted = []
    canvas.chainPicked.connect(lambda *a: emitted.append(a))
    before = [list(axis) for axis in canvas.view.viewRange()]

    canvas.mousePressEvent(_press((140.0, 160.0), no_mod))
    canvas.mouseMoveEvent(_move((300.0, 280.0), no_mod))
    canvas.mouseReleaseEvent(_release((300.0, 280.0), no_mod))

    assert emitted == []
    assert [list(axis) for axis in canvas.view.viewRange()] != before


def test_the_roi_modifier_drag_is_unaffected_by_picking(qtbot, canvas):
    """(d) from the design: ⌘-drag ROI behavior is untouched -- no chainPicked leak alongside it."""
    from dynamix.shell.canvas import ROI_MODIFIER

    canvas.set_pick_chains(CHAINS)
    emitted = []
    canvas.chainPicked.connect(lambda *a: emitted.append(a))

    with qtbot.waitSignal(canvas.roiDrawn, timeout=1000):
        canvas.mousePressEvent(_press((140.0, 160.0), ROI_MODIFIER))
        canvas.mouseMoveEvent(_move((300.0, 280.0), ROI_MODIFIER))
        canvas.mouseReleaseEvent(_release((300.0, 280.0), ROI_MODIFIER))

    assert emitted == []


# --------------------------------------------------------------------------- transect


def _approx_point(pt, expected, tol=0.25):
    """Coordinate round-tripped through ``_screen_xy`` -> Qt's scene transform -> ``_data_xy`` is
    not bit-exact (unlike the ROI gesture's own tests, which only ever assert an INTEGER row/col/
    h/w derived by floor/ceil -- see ``roi_from_corners``): ``Canvas._data_at`` converts its
    widget-pixel argument through a ``QPoint`` (int), so up to ~1 SCREEN pixel of sub-pixel
    position is lost on the way in -- on this fixture's ~0.13 data-units-per-screen-pixel scale
    (measured directly: a 600x600 widget over a ~60-unit view range), that is a real, bounded
    ~0.13-unit round-trip error, not test flakiness. ``tol`` is set comfortably above that bound.
    """
    assert pt[0] == pytest.approx(expected[0], abs=tol)
    assert pt[1] == pytest.approx(expected[1], abs=tol)


def test_two_clicks_in_transect_mode_emit_left_to_right_oriented_endpoints(qtbot, canvas):
    """Click order A=(10, 5), B=(2, 5) -- a horizontal segment (|dx| >= |dy|) clicked RIGHT then
    LEFT -- must come out oriented LEFT->RIGHT regardless of click order."""
    canvas.set_selection_mode("transect")
    no_mod = QtCore.Qt.KeyboardModifier.NoModifier
    ax, ay = _screen_xy(canvas, 10.0, 5.0)
    bx, by = _screen_xy(canvas, 2.0, 5.0)

    canvas.mousePressEvent(_press((ax, ay), no_mod))
    canvas.mouseReleaseEvent(_release((ax, ay), no_mod))
    _approx_point(canvas._transect_a, (10.0, 5.0))                # A recorded, marker drawn
    mx, my = canvas.transect_marker_item.getData()
    _approx_point((mx[0], my[0]), (10.0, 5.0))

    with qtbot.waitSignal(canvas.transectDrawn, timeout=1000) as sig:
        canvas.mousePressEvent(_press((bx, by), no_mod))
        canvas.mouseReleaseEvent(_release((bx, by), no_mod))

    emitted_a, emitted_b = sig.args
    _approx_point(emitted_a, (2.0, 5.0))                          # swapped: left endpoint first
    _approx_point(emitted_b, (10.0, 5.0))
    assert canvas._transect_a is None
    mx, my = canvas.transect_marker_item.getData()
    assert (mx is None or mx.size == 0) and (my is None or my.size == 0)


def test_two_clicks_in_transect_mode_emit_bottom_to_top_oriented_endpoints(qtbot, canvas):
    """The Y-DIRECTION case (the design's own emphasis): a vertical segment (|dy| > |dx|) clicked
    A at the SMALLER y (the top of the raster, canvas y grows downward) then B at the LARGER y
    (the bottom) must come out with the LARGER-y point FIRST ("bottom -> top" -- canvas y grows
    downward, so the start is the point closer to the bottom of the screen/raster)."""
    canvas.set_selection_mode("transect")
    no_mod = QtCore.Qt.KeyboardModifier.NoModifier
    ax, ay = _screen_xy(canvas, 3.0, 2.0)          # top (smaller y)
    bx, by = _screen_xy(canvas, 3.0, 9.0)          # bottom (larger y)

    canvas.mousePressEvent(_press((ax, ay), no_mod))
    canvas.mouseReleaseEvent(_release((ax, ay), no_mod))

    with qtbot.waitSignal(canvas.transectDrawn, timeout=1000) as sig:
        canvas.mousePressEvent(_press((bx, by), no_mod))
        canvas.mouseReleaseEvent(_release((bx, by), no_mod))

    emitted_a, emitted_b = sig.args
    _approx_point(emitted_a, (3.0, 9.0))                          # larger-y (bottom) point first
    _approx_point(emitted_b, (3.0, 2.0))


def test_esc_cancels_an_in_progress_first_click(qtbot, canvas):
    canvas.set_selection_mode("transect")
    no_mod = QtCore.Qt.KeyboardModifier.NoModifier
    x, y = _screen_xy(canvas, 10.0, 5.0)
    canvas.mousePressEvent(_press((x, y), no_mod))
    canvas.mouseReleaseEvent(_release((x, y), no_mod))
    assert canvas._transect_a is not None

    canvas.cancel_transect()

    assert canvas._transect_a is None
    mx, my = canvas.transect_marker_item.getData()
    assert (mx is None or mx.size == 0) and (my is None or my.size == 0)

    # a harmless no-op with nothing in progress
    canvas.cancel_transect()
    assert canvas._transect_a is None


def test_a_click_after_esc_starts_a_fresh_gesture_not_paired_with_the_cancelled_one(qtbot, canvas):
    canvas.set_selection_mode("transect")
    no_mod = QtCore.Qt.KeyboardModifier.NoModifier
    ax, ay = _screen_xy(canvas, 10.0, 5.0)
    canvas.mousePressEvent(_press((ax, ay), no_mod))
    canvas.mouseReleaseEvent(_release((ax, ay), no_mod))
    canvas.cancel_transect()

    bx, by = _screen_xy(canvas, 1.0, 1.0)
    canvas.mousePressEvent(_press((bx, by), no_mod))
    canvas.mouseReleaseEvent(_release((bx, by), no_mod))

    _approx_point(canvas._transect_a, (1.0, 1.0))                 # this is a FIRST click, not a B


def test_switching_selection_mode_away_from_transect_cancels_an_in_progress_click(qtbot, canvas):
    canvas.set_selection_mode("transect")
    no_mod = QtCore.Qt.KeyboardModifier.NoModifier
    x, y = _screen_xy(canvas, 10.0, 5.0)
    canvas.mousePressEvent(_press((x, y), no_mod))
    canvas.mouseReleaseEvent(_release((x, y), no_mod))
    assert canvas._transect_a is not None

    canvas.set_selection_mode("click")

    assert canvas._transect_a is None
    mx, my = canvas.transect_marker_item.getData()
    assert (mx is None or mx.size == 0) and (my is None or my.size == 0)


def test_pick_chains_accessor_mirrors_set_pick_chains(canvas):
    assert canvas.pick_chains() is None
    canvas.set_pick_chains(CHAINS)
    assert canvas.pick_chains() is CHAINS


def test_set_transects_draws_visible_lines_and_highlights_the_selected_one(canvas):
    from dynamix.model.project import TransectRecord

    records = [
        TransectRecord(transect_id=1, a=(0.0, 0.0), b=(10.0, 0.0), visible=True),
        TransectRecord(transect_id=2, a=(0.0, 5.0), b=(10.0, 5.0), visible=True),
        TransectRecord(transect_id=3, a=(0.0, 9.0), b=(10.0, 9.0), visible=False),
    ]

    canvas.set_transects(records, selected_id=2)

    xs, ys = canvas.transect_item.getData()
    # both visible segments drawn (NaN-separated), the hidden one (id 3) is not
    assert 0.0 in xs and 5.0 in ys and 9.0 not in ys
    assert np.isnan(xs).any()                                     # the NaN separator is present

    hx, hy = canvas.transect_highlight_item.getData()
    assert list(hx) == [0.0, 10.0] and list(hy) == [5.0, 5.0]     # record id 2's own segment


def test_set_transects_clears_highlight_when_selected_id_matches_nothing_visible(canvas):
    from dynamix.model.project import TransectRecord

    records = [TransectRecord(transect_id=1, a=(0.0, 0.0), b=(10.0, 0.0), visible=True)]

    canvas.set_transects(records, selected_id=99)

    hx, hy = canvas.transect_highlight_item.getData()
    assert (hx is None or hx.size == 0) and (hy is None or hy.size == 0)


def test_set_transects_with_no_records_clears_both_items(canvas):
    from dynamix.model.project import TransectRecord

    canvas.set_transects([TransectRecord(transect_id=1, a=(0.0, 0.0), b=(1.0, 1.0))], selected_id=1)
    canvas.set_transects([], selected_id=None)

    xs, ys = canvas.transect_item.getData()
    hx, hy = canvas.transect_highlight_item.getData()
    assert (xs is None or xs.size == 0) and (ys is None or ys.size == 0)
    assert (hx is None or hx.size == 0) and (hy is None or hy.size == 0)


# --------------------------------------------------------------------------- set_pick_chains / radius


def test_set_pick_chains_defaults_to_none():
    from dynamix.shell.canvas import Canvas

    c = Canvas()
    assert c._pick_chains is None


def test_pick_radius_converts_screen_px_through_the_viewboxs_own_scale(canvas):
    """The actual conversion this task wires up: ``ViewBox.viewPixelSize()`` -- ``self.view`` is
    already a plain ``pg.ViewBox`` (``self.addViewBox()``, not wrapped in a ``PlotItem``), so this
    is called directly, no ``.vb`` indirection (the design's ``self.plot.vb...`` was a guess)."""
    dx, dy = canvas.view.viewPixelSize()
    expected = 8.0 * (abs(dx) + abs(dy)) / 2.0

    assert canvas._pick_radius_data(8.0) == pytest.approx(expected)


# --------------------------------------------------------------------------- MainWindow wiring


def _wtmm_result(chains):
    return {
        "extrema": [{"x": np.array([0]), "y": np.array([0]), "mod": np.array([1.0]),
                     "arg": np.array([0.0]), "line_id": np.array([-1])}],
        "_shape": (16, 16),
        "chains": chains,
        "params": {},
    }


def test_chain_picked_hit_reaches_the_palette_naming_the_active_layer(qtbot, loaded):
    loaded._group_palette.add_pick = Mock()

    loaded.canvas.chainPicked.emit(3, False)

    loaded._group_palette.add_pick.assert_called_once_with((loaded.layer.layer_id, 3), False)


def test_chain_picked_miss_passes_none_through(qtbot, loaded):
    loaded._group_palette.add_pick = Mock()

    loaded.canvas.chainPicked.emit(None, True)

    loaded._group_palette.add_pick.assert_called_once_with(None, True)


def test_chains_lassoed_adds_all_indices_in_one_apply_picks_call(qtbot, loaded):
    loaded._group_palette.apply_picks = Mock()

    loaded.canvas.chainsLassoed.emit([1, 2], False)

    lid = loaded.layer.layer_id
    loaded._group_palette.apply_picks.assert_called_once_with([(lid, 1), (lid, 2)], "add")


def test_chains_lassoed_with_alt_subtracts(qtbot, loaded):
    loaded._group_palette.apply_picks = Mock()

    loaded.canvas.chainsLassoed.emit([1, 2], True)

    lid = loaded.layer.layer_id
    loaded._group_palette.apply_picks.assert_called_once_with([(lid, 1), (lid, 2)], "subtract")


def test_chains_boxed_adds_all_indices_in_one_apply_picks_call(qtbot, loaded):
    loaded._group_palette.apply_picks = Mock()

    loaded.canvas.chainsBoxed.emit([0, 3], False)

    lid = loaded.layer.layer_id
    loaded._group_palette.apply_picks.assert_called_once_with([(lid, 0), (lid, 3)], "add")


def test_chains_boxed_with_alt_subtracts(qtbot, loaded):
    loaded._group_palette.apply_picks = Mock()

    loaded.canvas.chainsBoxed.emit([0, 3], True)

    lid = loaded.layer.layer_id
    loaded._group_palette.apply_picks.assert_called_once_with([(lid, 0), (lid, 3)], "subtract")


def test_apply_pushes_the_landing_results_chains_to_the_canvas(qtbot, loaded):
    chains = [{"x": np.array([1.0, 2.0]), "y": np.array([1.0, 2.0])}]
    renderable = Renderable(layer_id=loaded.layer.layer_id, result=_wtmm_result(chains))

    loaded._apply(renderable)

    assert loaded.canvas._pick_chains is chains


def test_hiding_the_active_layer_clears_pick_chains(qtbot, loaded):
    loaded.canvas.set_pick_chains(CHAINS)

    loaded._on_hide_toggled(loaded.layer.layer_id, True)

    assert loaded.canvas._pick_chains is None


def test_apply_clears_pick_chains_for_an_invisible_active_layer(qtbot, loaded):
    loaded.canvas.set_pick_chains(CHAINS)
    loaded.layer.visible = False

    loaded._apply(Renderable(layer_id=loaded.layer.layer_id, result=_wtmm_result(CHAINS)))

    assert loaded.canvas._pick_chains is None


# --------------------------------------------------------------------------- Fix round 1


def test_apply_shifts_roi_chains_by_the_display_offset(qtbot, loaded):
    """`set_result` draws an ROI result's trail shifted by
    `display_offset` (the ROI's own on-screen origin) -- the pushed pick chains must be shifted
    by that SAME offset, not left in the result's own un-shifted frame."""
    chains = [{"x": np.array([1.0, 2.0]), "y": np.array([1.0, 2.0])}]
    result = _wtmm_result(chains)
    result["_roi"] = {"roi": (30, 40, 16, 16)}          # row_off=30, col_off=40 (bare-array field)
    renderable = Renderable(layer_id=loaded.layer.layer_id, result=result)

    loaded._apply(renderable)

    pushed = loaded.canvas._pick_chains
    assert pushed is not chains                          # a fresh, shifted copy -- not the original
    assert np.array_equal(pushed[0]["x"], np.array([41.0, 42.0]))     # +col_off
    assert np.array_equal(pushed[0]["y"], np.array([31.0, 32.0]))     # +row_off
    assert np.array_equal(chains[0]["x"], np.array([1.0, 2.0]))       # the original is untouched


def test_a_click_at_the_roi_shifted_position_still_picks_the_right_chain(qtbot, canvas):
    """End-to-end companion to the test above: once a result's chains have been shifted by an
    ROI's own `display_offset` (`main_window._shift_chains`, the same helper `_apply` calls), a
    click at the SHIFTED on-screen position -- where the trail is actually drawn -- must still
    resolve to the right chain."""
    from dynamix.shell.main_window import _shift_chains

    row_off, col_off = 100, 200
    shifted = _shift_chains(CHAINS, row_off, col_off)
    canvas.set_pick_chains(shifted)
    canvas.view.setRange(xRange=(0, 60 + col_off), yRange=(0, 60 + row_off), padding=0)
    x, y = _screen_xy(canvas, 11.0 + col_off, 5.0 + row_off)     # chain 0's shifted middle point

    with qtbot.waitSignal(canvas.chainPicked, timeout=1000) as sig:
        canvas.mousePressEvent(_press((x, y), QtCore.Qt.KeyboardModifier.NoModifier))
        canvas.mouseReleaseEvent(_release((x, y), QtCore.Qt.KeyboardModifier.NoModifier))

    assert sig.args == [0, False]


def test_switching_to_a_points_px_result_clears_stale_pick_chains(qtbot, loaded):
    """Stale-pick corruption vector: the `points_px` branch never calls
    `set_result`, so a WTMM layer's leftover `_pick_chains` must be cleared here too -- otherwise
    a click on the (still visually stale) overlay would silently write `(new_layer_id,
    old_chain_index)` into the palette instead of an honest miss."""
    loaded.canvas.set_pick_chains(CHAINS)

    loaded._apply(Renderable(layer_id=loaded.layer.layer_id, result={"points_px": None}))

    assert loaded.canvas._pick_chains is None
