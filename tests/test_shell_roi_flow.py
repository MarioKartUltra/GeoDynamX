# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The ROI gesture, the precision panel and Create -- the flow that makes an ROI real.

Offscreen Qt (the runner sets ``QT_QPA_PLATFORM=offscreen``). Three layers of test here, and they
are deliberately separate:

- the pure seams (``roi_from_corners``, ``roi_chain``, ``_roi_margin_reading``), which need no
  window at all and pin the arithmetic;
- the widgets (``Canvas``'s drag, ``RoiPanel``'s fields), driven by handing the handlers
  synthesized ``QMouseEvent``s rather than ``QTest.mouse*`` -- pyqtgraph's ``ViewBox`` sees mouse
  input through the graphics SCENE, and synthesizing a drag through that chain offscreen is
  flaky in a way that would make a red here mean nothing;
- the whole flow through ``MainWindow``, over a small synthetic parent GeoTIFF (the duplication
  pattern ``tests/test_wtmm_roi_device.py`` established: a local writer, its own fixture).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.devices import register_builtin_devices
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.param import Param, ParamKind
from dynamix.shell.units import px_to_metres

# --------------------------------------------------------------------------- fixtures


_FIELD_SHAPE = (64, 64)
_FIELD = np.linspace(0.0, 1.0, _FIELD_SHAPE[0] * _FIELD_SHAPE[1]).reshape(_FIELD_SHAPE)

#: n_oct=1, n_voice=2, a_min=1.0 -> 2 scales; the same small stack tests/test_wtmm_roi_device.py
#: uses, for the same reason (margins 18 and 25 px, so a real compute stays under a second).
_SMALL_WTMM = {"n_oct": 1, "n_voice": 2, "a_min": 1.0}

# Auto-run-on-by-default for this whole file (every ``load_field``/``open_path`` call below
# expects the worker to dispatch immediately, the same as it always did -- that dispatch is what
# the ROI flow this file tests runs on) now lives in ONE suite-wide fixture,
# ``tests/conftest.py::_dynamix_settings_isolated``.


@pytest.fixture
def canvas(qtbot):
    from dynamix.shell.canvas import Canvas

    c = Canvas()
    qtbot.addWidget(c)
    c.resize(400, 400)
    c.set_field(_FIELD)
    c.view.setRange(xRange=(0, _FIELD_SHAPE[1]), yRange=(0, _FIELD_SHAPE[0]), padding=0)
    return c


def _mouse(kind, pos, modifiers=QtCore.Qt.KeyboardModifier.NoModifier,
           button=QtCore.Qt.MouseButton.LeftButton):
    buttons = QtCore.Qt.MouseButton.NoButton if kind == QtCore.QEvent.Type.MouseButtonRelease \
        else button
    # The (type, localPos, globalPos, button, buttons, modifiers) overload: the shorter one
    # without a global position is deprecated in Qt 6 and warns on every call.
    return QtGui.QMouseEvent(kind, QtCore.QPointF(*pos), QtCore.QPointF(*pos),
                             button, buttons, modifiers)


def _press(pos, modifiers):
    return _mouse(QtCore.QEvent.Type.MouseButtonPress, pos, modifiers)


def _move(pos, modifiers):
    return _mouse(QtCore.QEvent.Type.MouseMove, pos, modifiers)


def _release(pos, modifiers):
    return _mouse(QtCore.QEvent.Type.MouseButtonRelease, pos, modifiers)


def _drag(canvas, start, end, modifiers):
    canvas.mousePressEvent(_press(start, modifiers))
    canvas.mouseMoveEvent(_move(end, modifiers))
    canvas.mouseReleaseEvent(_release(end, modifiers))


def _data_from_geometry(canvas, px, py):
    """Widget/scene px -> data coords, re-derived from PUBLISHED geometry.

    Deliberately NOT ``canvas._data_at`` (which is what the gesture itself calls): re-deriving the
    affine from the ViewBox's own scene rectangle and its view range is an independent check of
    the mapping, where calling the same helper would only prove it equals itself. ``invertY`` puts
    the data-y MINIMUM at the top of the screen, which is why ``fy`` is measured downwards.
    """
    box = canvas.view.sceneBoundingRect()
    (x0, x1), (y0, y1) = canvas.view.viewRange()
    fx = (px - box.left()) / box.width()
    fy = (py - box.top()) / box.height()
    return x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)


# --------------------------------------------------------------------------- the modifier


@pytest.mark.skipif(sys.platform != "darwin", reason="the ⌘ mapping is a macOS question")
def test_the_roi_modifier_is_the_command_key(qtbot):
    """WHICH modifier arrives when the user holds ⌘.

    Qt swaps Control and Meta on macOS unless ``AA_MacDontSwapCtrlAndMeta`` is set, and this app
    never sets it -- so ⌘ arrives as ``ControlModifier`` and the physical Control key arrives as
    ``MetaModifier``, the opposite of what the names suggest. ``QKeySequence``'s NATIVE rendering
    is the platform's own answer to the question, so it is what the assertion asks: the modifier
    the canvas binds must render as ⌘, and ``MetaModifier`` must render as ⌃ (i.e. binding Meta
    would have given a Control-drag gesture, not a Command-drag one).
    """
    from dynamix.shell.canvas import ROI_MODIFIER

    app = QtWidgets.QApplication.instance()
    assert app.testAttribute(QtCore.Qt.AA_MacDontSwapCtrlAndMeta) is False
    native = QtGui.QKeySequence.SequenceFormat.NativeText
    assert QtGui.QKeySequence(ROI_MODIFIER | QtCore.Qt.Key_A).toString(native) == "⌘A"
    assert QtGui.QKeySequence(
        QtCore.Qt.KeyboardModifier.MetaModifier | QtCore.Qt.Key_A).toString(native) == "⌃A"


# --------------------------------------------------------------------------- the pure rect


def test_roi_from_corners_is_row_col_h_w_in_image_pixels():
    from dynamix.shell.canvas import roi_from_corners

    # Center convention: a drag whose endpoints ARE pixel centers touches both endpoint
    # pixels, so 20.0..44.0 covers rows 20..44 inclusive (h=25), 10.0..30.0 cols 10..30 (w=21).
    assert roi_from_corners((10.0, 20.0), (30.0, 44.0), (64, 64)) == (20, 10, 25, 21)


def test_roi_from_corners_normalizes_a_backwards_drag():
    from dynamix.shell.canvas import roi_from_corners

    forward = roi_from_corners((10.0, 20.0), (30.0, 44.0), (64, 64))
    assert roi_from_corners((30.0, 44.0), (10.0, 20.0), (64, 64)) == forward


def test_roi_from_corners_clamps_to_the_field():
    """A drag that starts or ends off the raster must not name pixels that do not exist."""
    from dynamix.shell.canvas import roi_from_corners

    assert roi_from_corners((-50.0, -50.0), (500.0, 500.0), (64, 48)) == (0, 0, 64, 48)
    assert roi_from_corners((40.0, -10.0), (500.0, 30.0), (64, 48)) == (0, 40, 31, 8)


def test_roi_from_corners_covers_every_pixel_the_box_touched():
    """floor on the near corner, ceil on the far one: a box drawn from 10.3 to 12.1 covers the
    three pixels it visibly crosses, rather than rounding one of them away."""
    from dynamix.shell.canvas import roi_from_corners

    assert roi_from_corners((10.3, 10.3), (12.1, 12.1), (64, 64)) == (10, 10, 3, 3)


# --------------------------------------------------------------------------- the gesture


def test_modifier_drag_emits_the_rect_in_image_coordinates(qtbot, canvas):
    from dynamix.shell.canvas import ROI_MODIFIER

    start, end = (140.0, 160.0), (300.0, 280.0)
    with qtbot.waitSignal(canvas.roiDrawn, timeout=1000) as sig:
        _drag(canvas, start, end, ROI_MODIFIER)

    row, col, h, w = sig.args
    x_start, y_start = _data_from_geometry(canvas, *start)
    x_end, y_end = _data_from_geometry(canvas, *end)
    assert abs(row - int(np.floor(y_start))) <= 1
    assert abs(col - int(np.floor(x_start))) <= 1
    assert abs((row + h) - int(np.ceil(y_end))) <= 1
    assert abs((col + w) - int(np.ceil(x_end))) <= 1
    assert all(isinstance(v, int) for v in sig.args)


def test_a_drag_off_the_edge_is_clamped_to_the_field(qtbot, canvas):
    from dynamix.shell.canvas import ROI_MODIFIER

    with qtbot.waitSignal(canvas.roiDrawn, timeout=1000) as sig:
        _drag(canvas, (-4000.0, -4000.0), (4000.0, 4000.0), ROI_MODIFIER)

    assert tuple(sig.args) == (0, 0, _FIELD_SHAPE[0], _FIELD_SHAPE[1])


def test_the_gesture_does_not_pan_but_a_plain_drag_still_does(qtbot, canvas):
    """Both halves of "⌘-drag selects": the selection must not ALSO move the camera it is drawn
    on, and suppressing the pan must not have suppressed panning generally. The pan lives behind
    the graphics scene AND behind pyqtgraph's own GraphicsView.mouseMoveEvent, so this is the
    assertion that the canvas stopped the right one -- both of them -- without breaking the other.
    """
    from dynamix.shell.canvas import ROI_MODIFIER

    before = [list(axis) for axis in canvas.view.viewRange()]
    _drag(canvas, (140.0, 160.0), (300.0, 280.0), ROI_MODIFIER)
    assert [list(axis) for axis in canvas.view.viewRange()] == before

    _drag(canvas, (140.0, 160.0), (300.0, 280.0), QtCore.Qt.KeyboardModifier.NoModifier)
    assert [list(axis) for axis in canvas.view.viewRange()] != before


def test_a_plain_drag_draws_no_band_and_emits_nothing(qtbot, canvas):
    emitted = []
    canvas.roiDrawn.connect(lambda *a: emitted.append(a))

    _drag(canvas, (140.0, 160.0), (300.0, 280.0), QtCore.Qt.KeyboardModifier.NoModifier)

    assert emitted == []
    assert canvas.roi_band_item.getData()[0] is None or \
        canvas.roi_band_item.getData()[0].size == 0


def test_the_band_is_live_during_the_drag_and_snaps_to_the_emitted_rect(qtbot, canvas):
    from dynamix.shell.canvas import ROI_MODIFIER

    canvas.mousePressEvent(_press((140.0, 160.0), ROI_MODIFIER))
    canvas.mouseMoveEvent(_move((300.0, 280.0), ROI_MODIFIER))
    live_x, live_y = canvas.roi_band_item.getData()
    assert live_x is not None and live_x.size == 5          # closed rectangle
    assert live_x.max() > live_x.min() and live_y.max() > live_y.min()

    with qtbot.waitSignal(canvas.roiDrawn, timeout=1000) as sig:
        canvas.mouseReleaseEvent(_release((300.0, 280.0), ROI_MODIFIER))

    row, col, h, w = sig.args
    band_x, band_y = canvas.roi_band_item.getData()
    # Center registration: the snapped band ENCLOSES the named pixels' cells.
    assert (band_x.min(), band_x.max()) == (col - 0.5, col + w - 0.5)
    assert (band_y.min(), band_y.max()) == (row - 0.5, row + h - 0.5)


@pytest.mark.skipif(sys.platform != "darwin", reason="the ⌘ mapping is a macOS question")
def test_the_live_band_already_sits_on_the_whole_pixels_the_release_will_name(qtbot, canvas):
    """What the user sees mid-drag is what they get: the live band covers whole pixel cells,
    the same ones the release emits -- never the raw sub-pixel cursor position."""
    from dynamix.shell.canvas import ROI_MODIFIER

    canvas.mousePressEvent(_press((143.0, 161.0), ROI_MODIFIER))
    canvas.mouseMoveEvent(_move((301.0, 277.0), ROI_MODIFIER))
    live_x, live_y = canvas.roi_band_item.getData()
    with qtbot.waitSignal(canvas.roiDrawn, timeout=1000) as sig:
        canvas.mouseReleaseEvent(_release((301.0, 277.0), ROI_MODIFIER))
    row, col, h, w = sig.args
    assert (live_x.min(), live_x.max()) == (col - 0.5, col + w - 0.5)
    assert (live_y.min(), live_y.max()) == (row - 0.5, row + h - 0.5)


def test_a_drag_with_no_field_loaded_emits_nothing(qtbot):
    from dynamix.shell.canvas import ROI_MODIFIER, Canvas

    bare = Canvas()
    qtbot.addWidget(bare)
    bare.resize(400, 400)
    emitted = []
    bare.roiDrawn.connect(lambda *a: emitted.append(a))

    _drag(bare, (10.0, 10.0), (100.0, 100.0), ROI_MODIFIER)

    assert emitted == []


# --------------------------------------------------------------------------- in-context render


def _ext(xs, ys, line_ids):
    return {"x": np.asarray(xs, dtype=np.int64), "y": np.asarray(ys, dtype=np.int64),
            "mod": np.linspace(1.0, 0.5, len(xs)), "arg": np.zeros(len(xs)),
            "line_id": np.asarray(line_ids, dtype=np.int64)}


_SHARED_MASK = np.zeros((16, 16), dtype=bool)
_SHARED_MASK[8, 8] = True


def _overlay_result(roi=None):
    """One WTMM-shaped result over a 16x16 grid, optionally stamped as an ROI of a parent.

    The two variants share ONE ``_missing_mask`` ARRAY on purpose: the canvas caches the COI
    dilation and isocurve against that array's identity, so drawing both through the same canvas
    proves the parent-coordinate translation is applied at draw time rather than baked into the
    cached (and still perfectly valid) ROI-local outline.
    """
    result = {
        "extrema": [_ext([0, 1, 2, 5], [0, 0, 0, 3], [0, 0, 0, -1])],
        "chains": [{"x": np.array([0, 1]), "y": np.array([0, 1]), "mod": np.array([1.0, 0.5])}],
        "scales": np.array([1.0]), "_shape": (16, 16), "params": {},
        "_missing_mask": _SHARED_MASK, "_coi_radii": [1],
    }
    if roi is not None:
        result["_roi"] = {"source": "mem:parent", "roi": roi, "boundary": "auto"}
    return result


def test_roi_offset_reads_the_results_own_window():
    from dynamix.shell.canvas import roi_offset

    assert roi_offset(_overlay_result(roi=(30, 40, 16, 16))) == (30, 40)


def test_roi_offset_is_zero_for_an_ordinary_result():
    from dynamix.shell.canvas import roi_offset

    assert roi_offset(_overlay_result()) == (0, 0)
    assert roi_offset({}) == (0, 0)


def test_an_roi_result_draws_every_overlay_in_parent_coordinates(qtbot):
    """The in-context read: an ROI's extrema are ROI-LOCAL and the raster on screen is the
    PARENT, so every result-space overlay is translated by the ROI's origin. Asserted as a
    DIFFERENCE against the identical result without the ``_roi`` stamp, so the claim is exactly
    "the same drawing, moved by (row, col)" and nothing else changed on the way."""
    from dynamix.shell.canvas import Canvas

    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_show_trails(True)
    row_off, col_off = 30, 40

    canvas.set_result(_overlay_result(), 0)
    plain = {
        "extrema": canvas.extrema_item.getData(),
        "hchain": canvas.hchain_item.getData(),
        "vtrail": canvas.vtrail_item.getData(),
        "coi": canvas.coi_item.getData(),
    }
    canvas.set_result(_overlay_result(roi=(row_off, col_off, 16, 16)), 0)

    for name, (px, py) in plain.items():
        qx, qy = {"extrema": canvas.extrema_item, "hchain": canvas.hchain_item,
                  "vtrail": canvas.vtrail_item, "coi": canvas.coi_item}[name].getData()
        assert px is not None and px.size > 0, name
        assert np.allclose(qx, np.asarray(px) + col_off, equal_nan=True), name
        assert np.allclose(qy, np.asarray(py) + row_off, equal_nan=True), name


class _WindowedField:
    """The shape of a field ``open_field`` returns for a raster too big to load whole: values for
    the WINDOW it read, provenance recording where in the file that window starts."""

    def __init__(self, row_off, col_off, shape=(60, 60)):
        self.values = np.zeros(shape)
        self.provenance = {"source": "mem:parent",
                           "window": {"row_off": row_off, "col_off": col_off,
                                      "height": shape[0], "width": shape[1]}}


def test_display_offset_subtracts_the_displayed_windows_own_origin():
    """An ROI result's ``_roi`` is FILE-absolute (the device reads its halos off disk), but the
    image on screen may itself be a window of that file. The overlay offset is the difference --
    not the raw ROI origin, which would land everything one window-offset too far out."""
    from dynamix.shell.canvas import display_offset

    result = _overlay_result(roi=(30, 22, 16, 16))
    assert display_offset(result, _WindowedField(20, 12)) == (10, 10)
    assert display_offset(result, None) == (30, 22)          # whole-file field: no window to undo


def test_display_offset_is_zero_for_a_non_roi_result_on_a_windowed_field():
    """A plain wtmm2d result was computed ON the window, so it is ALREADY in the displayed image's
    coordinates -- subtracting the window origin there would push every overlay off by the offset
    in the opposite direction."""
    from dynamix.shell.canvas import display_offset

    assert display_offset(_overlay_result(), _WindowedField(20, 12)) == (0, 0)


def test_a_windowed_parents_roi_draws_at_image_coordinates(qtbot):
    """End to end at the canvas: window at (20, 12), ROI at file (30, 22) -- so the box and
    everything in it belong at image (10, 10), which is where the user drew it."""
    from dynamix.shell.canvas import Canvas

    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(_WindowedField(20, 12))

    canvas.set_result(_overlay_result(roi=(30, 22, 16, 16)), 0)

    # pin-in-place: the data space is FILE-ABSOLUTE -- the window's image sits
    # at its offset, so the box draws at file coordinates: still exactly over the same pixels.
    bx, by = canvas.roi_bounds_item.getData()
    assert (bx.min(), bx.max()) == (22, 38)
    assert (by.min(), by.max()) == (30, 46)
    hx, hy = canvas.hchain_item.getData()
    assert (hx[0], hy[0]) == (22, 30)              # the H-line's ROI-local (0, 0) point
    ex, ey = canvas.extrema_item.getData()
    assert (ex[0], ey[0]) == (27, 33)              # the isolated extremum, ROI-local (5, 3)


def test_an_roi_result_outlines_the_region_it_was_measured_over(qtbot):
    from dynamix.shell.canvas import Canvas

    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas.set_result(_overlay_result(roi=(30, 40, 16, 24)), 0)

    x, y = canvas.roi_bounds_item.getData()
    assert (x.min(), x.max()) == (40, 64)          # col .. col + w
    assert (y.min(), y.max()) == (30, 46)          # row .. row + h
    assert x.size == 5 and x[0] == x[-1] and y[0] == y[-1]      # closed


def test_an_ordinary_result_is_drawn_exactly_as_before(qtbot):
    """No offset and no bounds rectangle for a whole-raster result -- the pre-ROI drawing path,
    unchanged. (The rest of that path is pinned by tests/test_shell_canvas.py, whose fixtures
    carry no ``_roi`` at all and must keep passing untouched.)"""
    from dynamix.shell.canvas import Canvas

    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas.set_result(_overlay_result(), 0)

    ex, ey = canvas.extrema_item.getData()
    assert (ex[0], ey[0]) == (5, 3)                # the isolated extremum, where the result put it
    bx, _ = canvas.roi_bounds_item.getData()
    assert bx is None or bx.size == 0


def test_the_bounds_outline_clears_when_a_non_roi_result_follows_an_roi_one(qtbot):
    from dynamix.shell.canvas import Canvas

    canvas = Canvas()
    qtbot.addWidget(canvas)

    canvas.set_result(_overlay_result(roi=(30, 40, 16, 16)), 0)
    assert canvas.roi_bounds_item.getData()[0].size == 5
    canvas.set_result(_overlay_result(), 0)
    bx, _ = canvas.roi_bounds_item.getData()
    assert bx is None or bx.size == 0


# --------------------------------------------------------------------------- the panel


@pytest.fixture
def panel(qtbot):
    from dynamix.shell.roi_panel import RoiPanel

    p = RoiPanel()
    qtbot.addWidget(p)
    return p


def test_the_panel_is_hidden_until_a_roi_is_drawn(panel):
    """``isHidden`` rather than ``isVisible``: nothing in this test ever shows the panel's
    ancestors, and the question is whether the panel HIDES ITSELF until there is an ROI to talk
    about. The in-window half of the same question is
    ``test_a_drag_shows_the_panel_with_the_drawn_numbers``."""
    assert panel.isHidden() is True
    panel.show_roi(16, 24, 32, 40, (None, "px"))
    assert panel.isHidden() is False


def test_show_roi_fills_the_pixel_fields(panel):
    panel.show_roi(16, 24, 32, 40, (None, "px"))
    assert panel.edit("roi_row").text() == "16"
    assert panel.edit("roi_col").text() == "24"
    assert panel.edit("roi_h").text() == "32"
    assert panel.edit("roi_w").text() == "40"


def test_physical_readouts_agree_with_the_conversion(panel):
    """The panel's physical half goes through the SAME (factor, unit) pair every other reading in
    the shell uses -- ``dynamix.shell.units.px_to_metres``'s return value, handed in whole."""
    panel.show_roi(16, 24, 32, 40, (12.192024384048768, "m"))
    assert panel.physical("roi_h") == "390.1 m"
    assert panel.physical("roi_w") == "487.7 m"
    assert panel.physical("roi_row") == "195.1 m"


def test_physical_readouts_follow_an_edit_live(panel):
    panel.show_roi(16, 24, 32, 40, (2.0, "m"))
    assert panel.physical("roi_w") == "80 m"
    panel.edit("roi_w").setText("50")
    assert panel.physical("roi_w") == "100 m"


def test_a_legal_edit_emits_values_edited_and_an_illegal_one_does_not(panel):
    """Typed coordinates get a visual echo: the panel announces every edit that describes a
    legal box so the canvas can move the drawn band."""
    panel.show_roi(16, 24, 32, 40, (None, "px"))
    seen = []
    panel.valuesEdited.connect(seen.append)
    panel.edit("roi_row").setText("20")
    assert seen and seen[-1]["roi_row"] == 20 and seen[-1]["roi_w"] == 40
    n = len(seen)
    panel.edit("roi_h").setText("x")           # unparsable -> values() is None -> no emit
    assert len(seen) == n


def test_editing_the_panel_moves_the_canvas_band(roi_window):
    """End to end: draw, then type -- the amber band must track the NUMBERS."""
    roi_window._on_roi_drawn(10, 10, 24, 24)
    roi_window.roi_panel.edit("roi_col").setText("30")
    xs, ys = roi_window.canvas.roi_band_item.getData()
    assert xs is not None and xs.min() == 29.5 and xs.max() == 30 + 24 - 0.5
    assert ys.min() == 9.5 and ys.max() == 10 + 24 - 0.5


def test_a_field_with_no_physical_unit_shows_no_physical_readout(panel):
    """``px_to_metres`` returns ``(None, ...)`` for a field with no usable pixel size, and "px"
    for a non-georeferenced one. Neither has a physical number to state, and a fabricated 1.0
    would be exactly the lie units.py exists to prevent."""
    panel.show_roi(16, 24, 32, 40, (None, "px"))
    assert panel.physical("roi_h") == ""
    panel.show_roi(16, 24, 32, 40, (1.0, "px"))
    assert panel.physical("roi_h") == ""


def test_a_side_below_the_minimum_disables_create_with_a_message(panel):
    from dynamix.roi.halo import MIN_ROI_SIDE

    panel.show_roi(16, 24, 32, 40, (None, "px"))
    assert panel.create_button.isEnabled() is True
    assert panel.message_label.text() == ""

    panel.edit("roi_h").setText(str(MIN_ROI_SIDE - 1))

    assert panel.create_button.isEnabled() is False
    assert str(MIN_ROI_SIDE) in panel.message_label.text()
    assert panel.message_label.property("muted") == "true"

    panel.edit("roi_h").setText(str(MIN_ROI_SIDE))
    assert panel.create_button.isEnabled() is True
    assert panel.message_label.text() == ""


def test_a_roi_reaching_past_known_dims_disables_create_with_a_message(panel):
    """``dims`` is the drawn-on image's own shape, when the caller has one -- an edit that walks
    the box off that image is refused the same muted way a too-small side is, no dialog, rather
    than reaching Create and failing on the strip."""
    panel.show_roi(16, 24, 32, 40, (None, "px"), dims=(64, 64))
    assert panel.create_button.isEnabled() is True
    assert panel.message_label.text() == ""

    panel.edit("roi_h").setText("60")             # 16 + 60 = 76 > 64

    assert panel.create_button.isEnabled() is False
    assert panel.values() is None
    assert "64" in panel.message_label.text()
    assert panel.message_label.property("muted") == "true"

    panel.edit("roi_h").setText("32")
    assert panel.create_button.isEnabled() is True
    assert panel.message_label.text() == ""


def test_a_roi_with_no_known_dims_is_never_bound_checked(panel):
    """``dims=None`` (the default) is "no honest answer", not "the image is 0x0" -- a box that
    would be illegal against SOME extent must stay legal when none was ever supplied."""
    panel.show_roi(16, 24, 32, 40, (None, "px"))
    panel.edit("roi_h").setText("100000")
    assert panel.create_button.isEnabled() is True


def test_an_unparsable_field_disables_create(panel):
    panel.show_roi(16, 24, 32, 40, (None, "px"))
    panel.edit("roi_col").setText("")
    assert panel.create_button.isEnabled() is False


def test_a_blocked_panel_disables_create_and_says_why(panel):
    """The numbers can be perfectly legal and the ANSWER still be no -- an ROI of an ROI is not
    supported in this slice. The panel states the reason where the button is, muted, no dialog."""
    panel.show_roi(16, 24, 32, 40, (None, "px"), blocked="already an ROI")

    assert panel.values() is not None              # the geometry itself is fine
    assert panel.create_button.isEnabled() is False
    assert panel.message_label.text() == "already an ROI"
    assert panel.message_label.property("muted") == "true"


def test_a_blocked_panel_refuses_to_emit_even_if_the_button_is_reached(qtbot, panel):
    emitted = []
    panel.createRequested.connect(emitted.append)
    panel.show_roi(16, 24, 32, 40, (None, "px"), blocked="already an ROI")

    panel.create_button.click()
    panel._on_create_clicked()                     # past the disabled button, belt and braces

    assert emitted == []


def test_showing_an_unblocked_roi_clears_a_previous_block(panel):
    panel.show_roi(16, 24, 32, 40, (None, "px"), blocked="already an ROI")
    panel.show_roi(16, 24, 32, 40, (None, "px"))
    assert panel.create_button.isEnabled() is True
    assert panel.message_label.text() == ""


def test_create_emits_the_ints_and_the_boundary(qtbot, panel):
    panel.show_roi(16, 24, 32, 40, (None, "px"))
    with qtbot.waitSignal(panel.createRequested, timeout=1000) as sig:
        panel.create_button.click()
    assert sig.args[0] == {"roi_row": 16, "roi_col": 24, "roi_h": 32, "roi_w": 40,
                           "boundary": "auto"}


def test_the_boundary_button_has_a_mini_label_like_the_int_fields(panel):
    """Every ROI field gets a muted label naming it (the grid built in ``__init__``); the boundary
    CycleButton sat unlabelled next to them. It gets the same treatment: a muted label reading the
    device's own ``"Boundary"`` text, the same source ``_PARAMS`` the rest of the panel reads."""
    from dynamix.devices.wtmm_roi import WTMM2DROI

    label_texts = [w.text() for w in panel.findChildren(QtWidgets.QLabel)
                  if w.property("muted") == "true"]
    boundary_label = next(p for p in WTMM2DROI.params if p.name == "boundary").label
    assert boundary_label in label_texts


def test_the_boundary_button_cycles_the_devices_own_choices(qtbot, panel):
    from dynamix.devices.wtmm_roi import WTMM2DROI

    choices = next(p for p in WTMM2DROI.params if p.name == "boundary").choices
    assert choices == ("auto", "reflective")

    panel.show_roi(16, 24, 32, 40, (None, "px"))
    panel.boundary_button.click()
    with qtbot.waitSignal(panel.createRequested, timeout=1000) as sig:
        panel.create_button.click()
    assert sig.args[0]["boundary"] == "reflective"


# --------------------------------------------------------------------------- the chain clone


def test_roi_chain_replaces_wtmm2d_and_keeps_its_params(clean_registry):
    from dynamix.shell.main_window import roi_chain

    register_builtin_devices()
    parent = Chain((DeviceRef("wtmm2d", dict(_SMALL_WTMM, wavelet="gaussian", min_chain_len=3)),
                    DeviceRef("scale_select", {"scale_idx": 1}))).materialized()
    roi = {"roi_row": 16, "roi_col": 24, "roi_h": 32, "roi_w": 40, "boundary": "reflective"}

    chain = roi_chain(parent, roi)

    assert [s.device for s in chain.steps] == ["wtmm2d_roi", "scale_select"]
    step = chain.steps[0].params
    assert step["n_oct"] == 1 and step["n_voice"] == 2 and step["wavelet"] == "gaussian"
    assert step["min_chain_len"] == 3 and step["a_min"] == 1.0
    assert {k: step[k] for k in roi} == roi
    assert chain.steps[1].params["scale_idx"] == 1          # the filters ride along untouched


def test_roi_chain_carries_the_task_6_params_from_a_tuned_parent(clean_registry):
    """``_SHARED_WTMM_PARAMS`` used to stop at the original five, so a
    parent tuned away from the new defaults silently lost that tuning on every ROI child --
    a DIFFERENT analysis presented as the same one over a smaller window, which is exactly what
    ``roi_chain``'s own docstring says an ROI must never do."""
    from dynamix.shell.main_window import roi_chain

    register_builtin_devices()
    parent = Chain((DeviceRef("wtmm2d", dict(_SMALL_WTMM, smooth=False, thresh=0.05,
                                             dist2_max=80.0, box_ratio=2.0, similitude=0.6)),
                    DeviceRef("scale_select", {"scale_idx": 0}))).materialized()
    roi = {"roi_row": 0, "roi_col": 0, "roi_h": 16, "roi_w": 16, "boundary": "auto"}

    chain = roi_chain(parent, roi)

    step = chain.steps[0].params
    assert step["smooth"] is False
    assert step["thresh"] == 0.05
    assert step["dist2_max"] == 80.0
    assert step["box_ratio"] == 2.0
    assert step["similitude"] == 0.6


def test_roi_chain_carries_a_tuned_fracint_alpha(clean_registry):
    """``run_wtmm2d_roi`` always APPLIES the ``a**fracint_alpha`` lift (it resolves the backend
    default), so the knob must be declared on ``wtmm2d_roi`` and in ``_SHARED_WTMM_PARAMS`` --
    otherwise a parent tuned to η ≠ 1 gets a child computed at η = 1 while presenting as the
    same analysis over a window."""
    from dynamix.shell.main_window import roi_chain

    register_builtin_devices()
    parent = Chain((DeviceRef("wtmm2d", dict(_SMALL_WTMM, fracint_alpha=0.5)),)).materialized()

    chain = roi_chain(parent, {"roi_row": 0, "roi_col": 0, "roi_h": 16, "roi_w": 16,
                               "boundary": "auto"})

    assert chain.steps[0].params["fracint_alpha"] == 0.5


def test_roi_chain_floors_a_min_at_the_engines_own_minimum(clean_registry):
    """wtmm2d's a_min goes down to 0.25; the ROI path refuses anything under 1.0 (the halo would
    be fabricated), and its Param's hard min says so -- so a parent tuned below the floor must be
    raised on the way in, not carried into a chain that cannot be constructed at all."""
    from dynamix.roi.halo import MIN_A_MIN
    from dynamix.shell.main_window import roi_chain

    register_builtin_devices()
    parent = Chain((DeviceRef("wtmm2d", dict(_SMALL_WTMM, a_min=0.5)),)).materialized()

    chain = roi_chain(parent, {"roi_row": 0, "roi_col": 0, "roi_h": 16, "roi_w": 16,
                               "boundary": "auto"})

    assert chain.steps[0].params["a_min"] == MIN_A_MIN


def test_roi_chain_prepends_when_the_parent_has_no_wtmm2d(clean_registry):
    from dynamix.model.device import defaults_for, get_device
    from dynamix.shell.main_window import roi_chain

    register_builtin_devices()
    parent = Chain((DeviceRef("scale_select", {"scale_idx": 0}),)).materialized()
    roi = {"roi_row": 8, "roi_col": 8, "roi_h": 16, "roi_w": 16, "boundary": "auto"}

    chain = roi_chain(parent, roi)

    assert [s.device for s in chain.steps] == ["wtmm2d_roi", "scale_select"]
    defaults = defaults_for(get_device("wtmm2d_roi"))
    assert chain.steps[0].params["n_oct"] == defaults["n_oct"]
    assert {k: chain.steps[0].params[k] for k in roi} == roi


# --------------------------------------------------------------------------- the margin reading


def _margins(fracs, edges=()):
    return [{"a": 1.0 + i, "margin": 10, "real_frac": f,
             "reflected_edges": tuple(e)} for i, (f, e) in enumerate(zip(fracs, edges or
                                                                        [()] * len(fracs)))]


def test_margin_reading_states_every_scale_is_real():
    from dynamix.shell.main_window import _roi_margin_reading

    assert _roi_margin_reading(_margins([1.0] * 12)) == "margins real 12/12"


def test_margin_reading_names_the_union_of_reflected_edges():
    from dynamix.shell.main_window import _roi_margin_reading

    fracs = [1.0] * 9 + [0.8, 0.7, 0.6]
    edges = [()] * 9 + [("N",), ("N", "E"), ("E",)]
    assert _roi_margin_reading(_margins(fracs, edges)) == "real 9/12 (NE reflected)"


def test_margin_reading_orders_the_edges_the_way_the_engine_names_them():
    from dynamix.roi.halo import _EDGES
    from dynamix.shell.main_window import _roi_margin_reading

    assert _EDGES == ("N", "S", "E", "W")
    reading = _roi_margin_reading(_margins([0.5, 0.5], [("E", "W"), ("N", "S")]))
    assert reading == "real 0/2 (NSEW reflected)"


def test_margin_reading_names_missing_data_when_nothing_was_reflected():
    """Final-review F2: `real_frac` is deliberately reduced by zero-filled nodata as well as by
    reflection (dynamix/roi/halo.py's module docstring, "Missing data"), so a shortfall with NO
    reflected edges at all is NOT a reflection question -- the old code rendered it as
    ``real 0/12 ( reflected)`` (malformed AND untrue: nothing was reflected). The exact repro."""
    from dynamix.shell.main_window import _roi_margin_reading

    assert _roi_margin_reading(_margins([0.98] * 12)) == "real 0/12 (missing data)"


def test_margin_reading_combines_reflected_edges_and_missing_data():
    """A shortfall can have BOTH causes across the twelve scales: some margins short because their
    halo ran off the parent (reflected), others short with no reflection at all (nodata only) --
    the reading must name both, not silently drop one."""
    from dynamix.shell.main_window import _roi_margin_reading

    fracs = [1.0] * 8 + [0.9, 0.9, 0.85, 0.7]
    edges = [()] * 8 + [("N",), ("W",), (), ()]
    reading = _roi_margin_reading(_margins(fracs, edges))
    assert reading == "real 8/12 (NW reflected, missing data)"


def test_margin_reading_never_produces_a_malformed_parenthetical():
    """The malformed string (a stray leading space from an empty edge-name union, or an untrue
    claim of "reflected" for a purely-nodata shortfall) can never recur, across every shape of
    shortfall the two causes can combine into."""
    from dynamix.shell.main_window import _roi_margin_reading

    cases = [
        _margins([0.98] * 12),                                     # the exact repro
        _margins([1.0] * 9 + [0.8, 0.7, 0.6],
                [()] * 9 + [("N",), ("N", "E"), ("E",)]),          # reflection-only shortfall
        _margins([1.0] * 8 + [0.9, 0.9, 0.85, 0.7],
                [()] * 8 + [("N",), ("W",), (), ()]),              # combined
        _margins([1.0] * 12),                                      # no shortfall at all
    ]
    for margins in cases:
        reading = _roi_margin_reading(margins)
        assert "( " not in reading
        assert "()" not in reading
        assert not reading.endswith("(")


# --------------------------------------------------------------------------- the whole flow

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
from rasterio.transform import from_origin                          # noqa: E402

_CRS = "EPSG:32615"


def _synthetic_parent(n=100, seed=5):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    f = 3.0 * np.sin(2 * np.pi * xx / 23.0) * np.cos(2 * np.pi * yy / 19.0)
    for _ in range(8):
        cy, cx = rng.uniform(4, n - 4, 2)
        f += rng.uniform(-6.0, 6.0) * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rng.uniform(2.0, 6.0) ** 2))
    return (f - f.mean()).astype(np.float32)


@pytest.fixture
def parent_tif(tmp_path):
    values = _synthetic_parent()
    path = tmp_path / "parent.tif"
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                       count=1, dtype="float32", crs=_CRS,
                       transform=from_origin(500000.0, 3200000.0, 2.0, 2.0)) as dst:
        dst.write(values, 1)
    return str(path)


def _parent_field(path, row_off=0, col_off=0, size=100):
    from dynamix.core.rasterfield import RasterField

    return RasterField.from_geotiff_window(path, row_off=row_off, col_off=col_off,
                                           height=size, width=size)


_PARENT_STEPS = (("wtmm2d", dict(_SMALL_WTMM)), ("scale_select", {"scale_idx": 0}))


@pytest.fixture
def roi_window(qtbot, clean_registry, parent_tif):
    """A window on the synthetic parent, its first (real) resolve landed."""
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=_PARENT_STEPS)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.load_field(_parent_field(parent_tif), parent_tif)
    return win


def test_a_drag_shows_the_panel_with_the_drawn_numbers(roi_window):
    assert roi_window.roi_panel.isVisibleTo(roi_window) is False

    roi_window.canvas.roiDrawn.emit(32, 34, 32, 30)

    assert roi_window.roi_panel.isVisibleTo(roi_window) is True
    assert roi_window.roi_panel.edit("roi_row").text() == "32"
    assert roi_window.roi_panel.edit("roi_w").text() == "30"
    # The claim is AGREEMENT, not a hardcoded string: the panel's physical half must be the same
    # conversion the transport, the scale bar and the wavelet bar are using for this field. (This
    # fixture's CRS is written as a bare "EPSG:32615" with no WKT unit text to parse, so
    # px_to_metres stays in the CRS's own native unit and labels it as such -- units.py's
    # documented "stay native rather than lie" case, shared by every reading in the window.)
    factor, unit = px_to_metres(roi_window.field)
    assert factor is not None
    assert roi_window.roi_panel.physical("roi_w") == f"{30 * factor:.4g} {unit}"


def test_create_spawns_a_grouped_roi_layer_and_resolves_it(qtbot, roi_window):
    """The whole flow: drag -> panel -> Create -> a grouped layer whose chain carries the drawn
    ROI, selected in the list, resolved through the normal worker path, on screen."""
    parent = roi_window.layer
    roi_window.canvas.roiDrawn.emit(32, 32, 32, 32)

    with qtbot.waitSignal(roi_window.resolved, timeout=60000) as sig:
        roi_window.roi_panel.create_button.click()

    assert len(roi_window.project.layers) == 2
    child = roi_window.project.layers[1]
    assert child.parent_id == parent.layer_id
    assert child.name == f"{parent.name} ROI 32,32"
    assert child.source_id == parent.source_id

    step = child.chain.steps[0]
    assert step.device == "wtmm2d_roi"
    assert (step.params["roi_row"], step.params["roi_col"]) == (32, 32)
    assert (step.params["roi_h"], step.params["roi_w"]) == (32, 32)
    assert step.params["n_oct"] == 1 and step.params["n_voice"] == 2   # the parent's own wtmm
    assert step.params["boundary"] == "auto"

    # the list groups it under its parent, and the selection moved to it
    assert roi_window.layer_list.layer_text(child.layer_id) == child.name
    assert roi_window.layer_list.current_layer_id() == child.layer_id
    assert roi_window.layer is child
    assert [roi_window.strips.strip(i).device.name
            for i in range(2)] == ["wtmm2d_roi", "scale_select"]

    # ... and what reached the canvas is the ROI-sized result, not the parent's
    result = sig.args[0].result
    assert result["_shape"] == (32, 32)
    assert result["_roi"]["roi"] == (32, 32, 32, 32)
    assert roi_window.canvas.image_item.image is not None


def test_the_roi_strip_reads_its_margin_honesty(qtbot, roi_window):
    """ROI (32,32,32,32) in a 100x100 parent: the widest margin is 25 px, so 7..89 is inside the
    parent on every side and every scale's halo is real data."""
    roi_window.canvas.roiDrawn.emit(32, 32, 32, 32)
    with qtbot.waitSignal(roi_window.resolved, timeout=60000):
        roi_window.roi_panel.create_button.click()

    assert roi_window.strips.strip(0).reading_label.text() == "margins real 2/2"


def test_a_layer_without_roi_margins_keeps_the_elapsed_reading(roi_window):
    """The parent layer's own wtmm2d carries no ``_roi_margins``; its strip must still report the
    compute cost, exactly as before this slice."""
    assert roi_window.strips.strip(0).reading_label.text().endswith(" ms")


def test_a_windowed_parent_offsets_the_roi_into_source_coordinates(qtbot, clean_registry,
                                                                   parent_tif):
    """The BOEM case: a raster too big to load whole is opened as a WINDOW, so the pixel the user
    drew on is not the pixel the file calls that. The ROI params address the FILE (the device
    re-reads its own halos from disk), so the window's own offset has to be added -- without it
    the analysed region is silently somewhere else entirely."""
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=_PARENT_STEPS)
    qtbot.addWidget(win)
    field = _parent_field(parent_tif, row_off=20, col_off=12, size=60)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.load_field(field, parent_tif)

    win.canvas.roiDrawn.emit(10, 10, 16, 16)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.roi_panel.create_button.click()

    step = win.project.layers[1].chain.steps[0]
    assert (step.params["roi_row"], step.params["roi_col"]) == (30, 22)
    assert win.project.layers[1].name.endswith("ROI 30,22")
    assert win.project.layers[1].chain.steps[0].params["roi_h"] == 16

    # ... and on screen (pin-in-place): the data space is file-absolute, so the
    # box draws at FILE coordinates -- still over the same pixels, the image moved with it.
    bx, by = win.canvas.roi_bounds_item.getData()
    assert (bx.min(), bx.max()) == (22, 38)
    assert (by.min(), by.max()) == (30, 46)


def test_a_second_create_on_the_roi_layer_is_refused(qtbot, roi_window):
    """Create fires against whatever layer is selected NOW, and
    after a Create that is the ROI layer itself -- so a second press minted "parent ROI ROI" with
    a chain of two wtmm2d_roi steps that could only ever error. Drawing on an ROI layer is
    allowed (the box is a fine thing to look at); creating from it is not, yet."""
    roi_window.canvas.roiDrawn.emit(32, 32, 32, 32)
    with qtbot.waitSignal(roi_window.resolved, timeout=60000):
        roi_window.roi_panel.create_button.click()
    assert len(roi_window.project.layers) == 2

    roi_window.canvas.roiDrawn.emit(40, 40, 16, 16)      # drawn on the ROI layer this time

    assert roi_window.roi_panel.create_button.isEnabled() is False
    assert "ROI" in roi_window.roi_panel.message_label.text()
    roi_window.roi_panel.create_button.click()
    assert len(roi_window.project.layers) == 2           # nothing minted


def test_selecting_another_layer_hides_the_panel_and_clears_the_band(qtbot, roi_window):
    """A drawn box belongs to the layer it was drawn on. Leaving the panel up across a layer
    switch is what let Create fire against the wrong target in the first place, and leaving the
    amber band on the canvas claims a selection that no longer means anything."""
    from dynamix.shell.canvas import ROI_MODIFIER

    roi_window.resize(500, 500)
    _drag(roi_window.canvas, (150.0, 150.0), (300.0, 300.0), ROI_MODIFIER)
    assert roi_window.roi_panel.isVisibleTo(roi_window) is True
    assert roi_window.canvas.roi_band_item.getData()[0].size == 5

    with qtbot.waitSignal(roi_window.resolved, timeout=60000):
        roi_window.roi_panel.create_button.click()       # -> selects the new ROI layer

    assert roi_window.roi_panel.isVisibleTo(roi_window) is False
    bx, _ = roi_window.canvas.roi_band_item.getData()
    assert bx is None or bx.size == 0


def test_opening_another_file_resets_the_roi_selection(qtbot, roi_window, parent_tif):
    roi_window.canvas.roiDrawn.emit(32, 32, 32, 32)
    assert roi_window.roi_panel.isVisibleTo(roi_window) is True

    with qtbot.waitSignal(roi_window.resolved, timeout=60000):
        roi_window.load_field(_parent_field(parent_tif), parent_tif)

    assert roi_window.roi_panel.isVisibleTo(roi_window) is False
    bx, _ = roi_window.canvas.roi_band_item.getData()
    assert bx is None or bx.size == 0


def test_the_roi_layer_reuses_the_parents_field_object(qtbot, roi_window):
    """The ROI device reads its windows off the SOURCE itself; the field only supplies provenance
    and the frame. So the parent's field object is handed straight to the child's resolve -- no
    second read of the parent raster to build a field nothing will look at."""
    parent_field = roi_window.field
    roi_window.canvas.roiDrawn.emit(32, 32, 32, 32)
    with qtbot.waitSignal(roi_window.resolved, timeout=60000):
        roi_window.roi_panel.create_button.click()

    assert roi_window.field is parent_field


# --------------------------------------------------------------------------- the provenance stamp


def test_open_field_stamps_the_source_on_a_whole_file_geotiff(parent_tif):
    """``RasterField._from_geotiff`` (the whole-file loader) records no provenance at all, and
    ``WTMM2DROI`` REFUSES a field with no ``source`` to halo against -- so without this stamp an
    ROI could be drawn on any raster small enough to load whole and never be computable. Core is
    a verbatim copy and cannot grow the key; the shell's own opener fills it."""
    from dynamix.shell.opening import open_field

    field = open_field(parent_tif)

    assert field.provenance["source"] == parent_tif
    assert field.provenance["full_dims"] == (100, 100)
    assert field.values.shape == (100, 100)          # whole file, not a window


def test_open_field_stamps_full_dims_on_a_windowed_field_too(parent_tif):
    """The whole point of recording the FILE's dimensions is the case where they are NOT the
    field's own shape -- so the windowed branch stamps them too, or the key would be worthless to
    anyone who could not already read it off ``values.shape``."""
    from dynamix.shell.opening import open_field

    field = open_field(parent_tif, max_pixels=1000, window_size=32, mode="window")

    assert field.values.shape == (32, 32)
    assert field.provenance["full_dims"] == (100, 100)
    assert field.provenance["source"] == parent_tif
    assert field.provenance["window"]["row_off"] == 34


def test_create_works_end_to_end_on_a_whole_file_layer(qtbot, clean_registry, parent_tif):
    """The gap the stamp closes, exercised the way a user meets it: open a small raster through
    the app's own opener, drag, Create -- and get a real ROI result rather than the device's
    "provenance has no 'source'" refusal on the strip."""
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=_PARENT_STEPS)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.open_path(parent_tif)                    # the whole-file route, provenance and all

    win.canvas.roiDrawn.emit(32, 32, 32, 32)
    with qtbot.waitSignal(win.resolved, timeout=60000) as sig:
        win.roi_panel.create_button.click()

    assert sig.args[0].result["_roi"]["roi"] == (32, 32, 32, 32)
    assert sig.args[0].result["_shape"] == (32, 32)
    assert win.strips.strip(0).state_dot.property("state") != "error"
    # ... and it lands over the region it was measured on, not in the corner
    bx, by = win.canvas.roi_bounds_item.getData()
    assert (bx.min(), by.min()) == (32, 32)


# --------------------------------------------------------------------------- layer switching


def _stub_stack(field, n_scales: int) -> dict:
    values = np.asarray(getattr(field, "values", field), dtype=np.float64)
    ny, nx = values.shape[:2]
    layers = []
    for k in range(n_scales):
        m = max(4, 12 - 2 * k)
        idx = np.arange(m, dtype=np.int64)
        layers.append({"x": idx % nx, "y": (idx * 2) % ny, "mod": np.linspace(1.0, 0.1, m),
                       "arg": np.linspace(0.0, np.pi, m),
                       "line_id": np.where(idx < 4, 0, -1).astype(np.int64)})
    return {"scales": [2.0 * (k + 1) for k in range(n_scales)], "extrema": layers,
            "_shape": (ny, nx), "params": {}}


class _StubStack:
    name = "stub_stack"
    params = (Param("n_scales", ParamKind.INT, default=3, min=1, max=8, label="Scales"),)

    def compute(self, field, params, *, progress=None):
        return _stub_stack(field, int(params["n_scales"]))

    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key
        return cache_key(self.name, source_id, params)


@pytest.fixture
def two_layer_window(qtbot, clean_registry):
    from dynamix.model.device import register_device
    from dynamix.shell.main_window import MainWindow

    register_builtin_devices()
    register_device(_StubStack())
    win = MainWindow(steps=(("stub_stack", {}), ("scale_select", {"scale_idx": 0})))
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD, "mem:layer_a")

    second = win.project.add_layer(
        "B", win.layer.source_id,
        Chain((DeviceRef("stub_stack", {"n_scales": 5}),
               DeviceRef("scale_select", {"scale_idx": 0}),
               DeviceRef("modulus_threshold", {"frac": 0.0}))))
    win.add_layer_row(second, _FIELD)
    return win


def test_selecting_a_layer_switches_the_chain_the_strips_and_the_resolve(qtbot, two_layer_window):
    win = two_layer_window
    first = win.layer

    with qtbot.waitSignal(win.resolved, timeout=10000) as sig:
        win.layer_list.select_layer(win.project.layers[1].layer_id)

    assert win.layer is win.project.layers[1]
    assert [win.strips.strip(i).device.name for i in range(3)] == \
        ["stub_stack", "scale_select", "modulus_threshold"]
    assert win._names == ["stub_stack", "scale_select", "modulus_threshold"]
    assert len(sig.args[0].result["scales"]) == 5          # layer B's own n_scales

    with qtbot.waitSignal(win.resolved, timeout=10000) as back:
        win.layer_list.select_layer(first.layer_id)

    assert win.layer is first
    assert [win.strips.strip(i).device.name for i in range(2)] == ["stub_stack", "scale_select"]
    assert len(back.args[0].result["scales"]) == 3


def test_switching_back_to_a_computed_layer_is_a_cache_hit(qtbot, two_layer_window):
    """Switching is the normal worker path, and the worker path is cache-keyed on the transform.
    Coming back to a layer already computed must therefore hit, not recompute."""
    win = two_layer_window
    first = win.layer
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.layer_list.select_layer(win.project.layers[1].layer_id)
    with qtbot.waitSignal(win.resolved, timeout=10000) as sig:
        win.layer_list.select_layer(first.layer_id)

    assert sig.args[0].cache_misses == 0


# --------------------------------------------------------------------------- real-data smoke

_WEST_ZIP = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "BOEM_Bathymetry_West_meters_tiff(1).zip")
)


@pytest.mark.skipif(not os.path.exists(_WEST_ZIP), reason="real BOEM fixture not present on this machine")
def test_real_boem_west_roi_resolves_with_honest_margins(qtbot, clean_registry):
    """Real-data smoke, end to end: open the West zip through the shell's own opening path (a
    38470x20782-ish raster reads back as a centered 4096x4096 window --
    ``tests/test_shell_opening.py``'s own real-data smoke pins that shape), drive Create's handler
    directly with a 512x512 ROI well inside that window, and confirm the grouped layer resolves
    with every margin real.

    The parent layer carries NO transform of its own (``steps=()``) -- the point of this test is
    the ROI's compute, and a full ``wtmm2d`` over the whole 4096x4096 window first would dwarf it
    for no assertion this test makes. ``roi_chain``'s no-``wtmm2d``-in-parent branch prepends
    ``wtmm2d_roi`` at its own defaults, and ``n_oct=2`` in ``spec`` overrides that default down to
    the oracle's own small stack (8 scales, margins 18..59 px) to keep the compute quick.

    The ROI sits deep enough inside the WINDOW that it is also deep inside the FILE --
    ``boundary="auto"`` clamps against the file's full extent (``WTMM2DROI.compute``'s stamped
    ``full_dims``), so ``real_frac == 1.0`` at every scale is the honest answer here, not a
    fixture artifact.

    Deliberately NOT the oracle: this proves the real pipeline resolves on real data with honest
    margins, nothing about extrema-position exactness -- ``dynamix.roi.halo.
    MAX_NMS_TIES_PER_SCALE`` is not imported anywhere in this test.
    """
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=())
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=120_000):
        win.open_path(_WEST_ZIP)

    parent = win.layer
    # The default open is the display PICTURE -- whole extent, and the canvas speaks FILE
    # pixels, so the ROI spec is in file pixels with no stride mapping at all: a 512x512
    # native window.
    from dynamix.roi.picture import native_shape

    assert "overview" not in win.field.provenance
    assert int(win.field.provenance["display_stride"]) > 1
    assert native_shape(win.field) == (20782, 38470)

    spec = {"roi_row": 10135, "roi_col": 18979,
            "roi_h": 512, "roi_w": 512, "boundary": "auto",
            "n_oct": 2}
    with qtbot.waitSignal(win.resolved, timeout=300_000) as sig:
        win._on_roi_create(spec)

    assert len(win.project.layers) == 2
    child = win.project.layers[1]
    assert child.parent_id == parent.layer_id

    result = sig.args[0].result
    assert result["_shape"] == (512, 512)
    margins = result["_roi_margins"]
    assert len(margins) == 8                        # n_oct=2, n_voice=4 (the device's own default)
    assert all(m["real_frac"] == 1.0 for m in margins)
    assert all(m["reflected_edges"] == () for m in margins)


# --------------------------------------------------------------------------- discipline


def test_roi_panel_is_in_the_devloop_reload_set():
    import dynamix.devloop as devloop

    assert "roi_panel" in devloop._MODULE_ORDER
    assert "roi_panel" in devloop._DEPENDS_ON["main_window"]
    assert devloop.modules_to_reload({"roi_panel"})[-1] == "dynamix.shell.main_window"


# ------------------------------------------- armed ROI placement
# Dropping the ROI tool arms a placement instead of running at once: choose the window dimensions in units or pixels, hover to see the footprint, click to set the ROI and run WTMM within it.

_NO = QtCore.Qt.KeyboardModifier.NoModifier


def _scene_pt(canvas, x, y):
    """Data coords -> widget px, inverting the published geometry (see _data_from_geometry)."""
    box = canvas.view.sceneBoundingRect()
    (x0, x1), (y0, y1) = canvas.view.viewRange()
    return (box.left() + (x - x0) / (x1 - x0) * box.width(),
            box.top() + (y - y0) / (y1 - y0) * box.height())


def test_armed_canvas_ghosts_the_footprint_under_the_cursor_and_places_on_click(qtbot, canvas):
    canvas.arm_roi_placement(24, 30)
    canvas.mouseMoveEvent(_move(_scene_pt(canvas, 40.0, 40.0), _NO))
    xs, ys = canvas.roi_band_item.getData()
    assert xs is not None and (max(xs) - min(xs)) == 30 and (max(ys) - min(ys)) == 24
    assert abs((max(xs) + min(xs)) / 2 - 40.0) <= 1.5             # centred on the cursor
    pos = _scene_pt(canvas, 40.0, 40.0)
    with qtbot.waitSignal(canvas.roiPlaced, timeout=1000) as sig:
        canvas.mousePressEvent(_press(pos, _NO))
        canvas.mouseReleaseEvent(_release(pos, _NO))
    row, col, h, w = sig.args
    assert (h, w) == (24, 30)
    assert abs(row - 28) <= 1 and abs(col - 25) <= 1
    assert canvas._roi_place is None                              # placing disarms


def test_armed_placement_clamps_the_footprint_to_the_field(qtbot, canvas):
    canvas.arm_roi_placement(24, 30)
    pos = _scene_pt(canvas, 2.0, 2.0)                             # near the corner
    with qtbot.waitSignal(canvas.roiPlaced, timeout=1000) as sig:
        canvas.mousePressEvent(_press(pos, _NO))
        canvas.mouseReleaseEvent(_release(pos, _NO))
    row, col, h, w = sig.args
    assert (row, col) == (0, 0) and (h, w) == (24, 30)


def test_dropping_wtmm2d_roi_arms_placement_instead_of_running(roi_window):
    win = roi_window
    chain_before = win.layer.chain
    n_layers = len(win.project.layers)
    descriptors = ([{"device": ref.device, "params": dict(ref.params)} for ref in chain_before.steps]
                   + [{"device": "wtmm2d_roi", "params": {}}])
    win._on_chain_edited(descriptors)
    assert win.layer.chain is chain_before                        # nothing committed, nothing ran
    assert len(win.project.layers) == n_layers
    assert win.canvas._roi_place is not None                      # the viewer is armed
    assert win.roi_panel.isVisibleTo(win) is True                 # size is editable before placing


def test_placing_the_armed_footprint_mints_the_roi_layer_and_runs(roi_window, qtbot):
    win = roi_window
    descriptors = ([{"device": ref.device, "params": dict(ref.params)} for ref in win.layer.chain.steps]
                   + [{"device": "wtmm2d_roi", "params": {}}])
    win._on_chain_edited(descriptors)
    n_layers = len(win.project.layers)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.canvas.roiPlaced.emit(16, 18, 40, 40)
    assert len(win.project.layers) == n_layers + 1                # the ROI child layer exists
    assert any(ref.device == "wtmm2d_roi" for ref in win.layer.chain.steps)
    assert win.canvas._roi_place is None


def test_roi_chain_carries_interpolate_and_detector(clean_registry):
    """The interpolate and detector knobs join the shared set: an ROI child runs the parent's
    detector and refinement or it is a different analysis presented as the same one."""
    from dynamix.shell.main_window import roi_chain

    register_builtin_devices()
    parent = Chain((DeviceRef("wtmm2d", dict(_SMALL_WTMM, interpolate=True,
                                             detector="follow")),)).materialized()
    chain = roi_chain(parent, {"roi_row": 0, "roi_col": 0, "roi_h": 16, "roi_w": 16,
                               "boundary": "auto"})
    assert chain.steps[0].params["interpolate"] is True
    assert chain.steps[0].params["detector"] == "follow"


# ------------------------------------------------------------------------ the child dataset


def test_child_button_is_enabled_even_when_roi_create_is_blocked(panel):
    """The wtmm2d_roi nesting block gates Create only -- a child CROP is legal wherever the
    box itself is."""
    panel.show_roi(4, 6, 24, 20, (None, "px"), blocked="already an ROI layer")
    assert panel.create_button.isEnabled() is False
    assert panel.child_button.isEnabled() is True


def test_child_click_emits_the_same_int_dict(qtbot, panel):
    panel.show_roi(4, 6, 24, 20, (None, "px"))
    with qtbot.waitSignal(panel.childRequested, timeout=1000) as sig:
        panel.child_button.click()
    spec = sig.args[0]
    assert (spec["roi_row"], spec["roi_col"], spec["roi_h"], spec["roi_w"]) == (4, 6, 24, 20)


def test_child_create_spawns_a_cropped_layer_with_identity_tag(roi_window):
    """The full flow: Child dataset -> a grouped child layer with an EMPTY chain, its field the
    native-pixel crop (values AND axes sliced), tags["roi.window"] in file-absolute coords,
    and a resolve identity distinct from the parent's (the cache-collision guard)."""
    from dynamix.engine.resolve import source_identity

    win = roi_window
    parent = win.layer
    n_before = len(win.project.layers)
    win._on_roi_child_create({"roi_row": 8, "roi_col": 6, "roi_h": 24, "roi_w": 20,
                              "boundary": "auto"})
    assert len(win.project.layers) == n_before + 1
    child = win.layer
    assert child.layer_id != parent.layer_id
    assert child.chain.steps == ()
    assert child.tags["roi.window"] == "8,6,24,20"
    field = win._fields[child.layer_id]
    assert field.values.shape == (24, 20)
    parent_field = win._fields[parent.layer_id]
    np.testing.assert_array_equal(field.values, parent_field.values[8:32, 6:26])
    np.testing.assert_array_equal(field.x_axis, parent_field.x_axis[6:26])
    np.testing.assert_array_equal(field.y_axis, parent_field.y_axis[8:32])
    assert field.provenance["window"] == {"row_off": 8, "col_off": 6}
    assert source_identity(child) != source_identity(parent)
    # two DIFFERENT windows must not share an identity either
    child.tags["roi.window"] = "0,0,24,20"
    other = source_identity(child)
    child.tags["roi.window"] = "8,6,24,20"
    assert other != source_identity(child)


# ------------------------------------------------------------- the dedicated ROI tool button


def test_roi_tool_button_opens_the_panel_with_a_centered_seed(roi_window):
    """No drag: the "ROI…" button shows the panel seeded with a centered box, and the amber
    band is painted from the seed (the drag paints its own; the tool must echo)."""
    win = roi_window
    assert win.roi_panel.isVisibleTo(win) is False
    win._on_roi_tool_clicked()
    assert win.roi_panel.isVisibleTo(win) is True
    spec = win.roi_panel.values()
    assert spec is not None
    ny, nx = np.asarray(win.field.values).shape[:2]
    assert spec["roi_h"] == min(512, ny) and spec["roi_w"] == min(512, nx)
    assert spec["roi_row"] == max(0, (ny - spec["roi_h"]) // 2)
    assert win.canvas.roi_band_item.isVisible()


def test_roi_tool_reopens_over_previous_numbers(roi_window):
    win = roi_window
    win.canvas.roiDrawn.emit(8, 6, 24, 20)          # a drag set specific numbers
    win.roi_panel.setVisible(False)                 # panel dismissed (layer switch etc.)
    win._on_roi_tool_clicked()
    spec = win.roi_panel.values()
    assert (spec["roi_row"], spec["roi_col"], spec["roi_h"], spec["roi_w"]) == (8, 6, 24, 20)


def test_roi_tool_with_no_field_notifies_and_stays_hidden(qtbot, clean_registry):
    from dynamix.shell.main_window import MainWindow

    register_builtin_devices()
    win = MainWindow(steps=())
    qtbot.addWidget(win)
    win._on_roi_tool_clicked()
    assert win.roi_panel.isVisibleTo(win) is False


def test_child_is_a_core_only_crop_with_parent_linkage(roi_window):
    """The ROI defines the CORE; no crop-time margin exists. The child's provenance keeps
    source + absolute window + full dims -- the linkage a future per-TOOL apron sampler
    (COI/kernel-support margins at compute time) will consume."""
    win = roi_window
    win._on_roi_child_create({"roi_row": 8, "roi_col": 6, "roi_h": 24, "roi_w": 20,
                              "boundary": "auto"})
    field = win._fields[win.layer.layer_id]
    assert field.values.shape == (24, 20)
    assert "roi_core" not in field.provenance
    # source + absolute window ALWAYS ride; full_dims joins when the parent came through
    # opening.open_field (_stamp_source) -- this fixture's parent is a direct windowed read.
    assert field.provenance.get("source")
    assert field.provenance["window"] == {"row_off": 8, "col_off": 6}


def test_place_box_repositions_the_panel_and_never_creates(roi_window):
    """"Place box" arms the hover ghost in REPOSITION mode: the click updates row/col and the
    band; no layer is created, nothing runs -- unlike the armed-DROP path, which creates."""
    win = roi_window
    win.canvas.roiDrawn.emit(4, 4, 24, 20)
    n_before = len(win.project.layers)
    win.roi_panel.place_button.click()
    assert win.canvas._roi_place is not None            # ghost armed
    win.canvas.roiPlaced.emit(30, 28, 24, 20)           # the click lands elsewhere
    assert len(win.project.layers) == n_before          # NOTHING created
    spec = win.roi_panel.values()
    assert (spec["roi_row"], spec["roi_col"]) == (30, 28)
    # and the armed-drop path still creates (the flag is one-shot)
    assert getattr(win, "_roi_place_reposition", False) is False


def test_show_roi_band_encloses_its_pixels_under_center_registration(qtbot):
    """A box over pixels [row, row+h) x [col, col+w) must DRAW from
    (col-0.5, row-0.5) to (col+w-0.5, row+h-0.5) -- enclosing the cells whose centers are the
    integer coordinates every overlay draws at."""
    from dynamix.shell.canvas import Canvas

    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.show_roi_band(2, 3, 4, 5)
    xs, ys = canvas.roi_band_item.getData()
    assert min(xs) == 2.5 and max(xs) == 7.5
    assert min(ys) == 1.5 and max(ys) == 5.5


def test_roi_place_origin_centers_the_box_on_the_cursor(qtbot):
    """An odd-sized box placed at a pixel center must center exactly on that pixel."""
    from dynamix.shell.canvas import Canvas

    canvas = Canvas()
    qtbot.addWidget(canvas)
    import numpy as np
    canvas.set_field(np.zeros((32, 32), dtype=np.float32))
    row, col = canvas._roi_place_origin((10.0, 10.0), 3, 3)
    assert (row, col) == (9, 9)                    # pixels 9,10,11 centred on 10
