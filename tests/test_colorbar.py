# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The colour bar on the canvas and in its export, and the centre of a diverging ramp on screen.

The bar is drawn from the state that painted the raster (ramp, stretch limits, slice, Bands,
centre), titled with the quantity and its units, and rendered by the scene, so ``grab()`` and
pyqtgraph's exporter carry it exactly as the screen shows it and nothing of it while it is off.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from tests.test_shell_window import stub_devices, window  # noqa: F401

#: A signed field: h from −0.4 to 1.19 along x, zero at column 40.
_H = np.tile((np.arange(160.0) - 40.0) / 100.0, (120, 1))


def _canvas(qtbot, values=_H, colormap="RdBu"):
    from dynamix.shell.canvas import Canvas

    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.resize(700, 450)
    canvas.set_field(values)
    canvas.set_colormap(colormap)
    canvas.grab()                                   # lays the view out at the widget's size
    canvas.view.autoRange()
    QtWidgets.QApplication.processEvents()
    return canvas


def _rgb(image: QtGui.QImage, point: QtCore.QPointF) -> np.ndarray:
    c = image.pixelColor(int(round(point.x())), int(round(point.y())))
    return np.array([c.red(), c.green(), c.blue()])


def _pixels(canvas) -> bytes:
    """The bytes of the canvas's frame as ``grab()`` renders it."""
    QtWidgets.QApplication.processEvents()
    image = canvas.grab().toImage()
    return bytes(image.constBits())


def _middle(name="RdBu") -> np.ndarray:
    from dynamix.shell.canvas import _resolve_cmap

    return _resolve_cmap(name).map(np.array([0.5]), mode="byte")[0][:3].astype(int)


def test_a_diverging_ramp_puts_its_middle_colour_on_zero_in_the_picture_and_the_bar(qtbot):
    canvas = _canvas(qtbot)
    canvas.set_colorbar(True, "h(x)")
    QtWidgets.QApplication.processEvents()
    bar = canvas.color_bar
    assert bar.isVisible() and bar.label == "h(x)"
    assert (0.5, "0") in bar.ticks() and (0.5, "0") in bar.labelled()     # the centre is labelled
    image = canvas.grab().toImage()
    ratio = image.devicePixelRatio()
    at = canvas.mapFromScene(bar.strip_scene_point(0.5))
    assert np.abs(_rgb(image, QtCore.QPointF(at) * ratio) - _middle()).max() <= 6
    # The raster's own column of zeros wears the same colour.
    zero = canvas.mapFromScene(canvas.view.mapViewToScene(QtCore.QPointF(40.0, 60.0)))
    assert np.abs(_rgb(image, QtCore.QPointF(zero) * ratio) - _middle()).max() <= 6
    # Ticks below and above the centre sit where the two slopes put them.
    from dynamix.core.stretch import centered

    for position, text in bar.ticks():
        value = float(text.replace("−", "-"))
        assert position == pytest.approx(float(centered(value, -0.4, 1.19)), abs=1e-9)


def test_a_sequential_ramp_keeps_the_raw_picture_and_its_bar_reads_the_image_levels(qtbot):
    canvas = _canvas(qtbot, colormap="viridis")
    assert np.array_equal(canvas.image_item.image, _H.T)                 # unchanged path
    canvas.set_colorbar(True, "h(x)")
    lo, hi = canvas.image_item.getLevels()
    ticks = canvas.color_bar.ticks()
    assert [t for _p, t in ticks] == ["−0.4", "−0.2", "0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    for position, text in ticks:
        assert position == pytest.approx((float(text.replace("−", "-")) - lo) / (hi - lo))


def test_the_centre_and_symmetric_move_the_middle_colour(qtbot):
    from dynamix.shell.canvas import _resolve_cmap

    canvas = _canvas(qtbot)
    canvas.set_center(0.5)
    shown = np.asarray(canvas.image_item.image)                   # (x, y) display scalars
    assert shown[90, 0] == pytest.approx(0.5)                     # h = 0.5 at column 90
    canvas.set_colorbar(True, "")
    assert (0.5, "0.5") in canvas.color_bar.ticks()
    canvas.set_center(0.0, symmetric=True)                        # limits ±1.19
    shown = np.asarray(canvas.image_item.image)
    assert shown[0, 0] == pytest.approx(0.5 - 0.5 * 0.4 / 1.19)
    rgba = canvas.color_bar.colours()
    assert np.array_equal(rgba, _resolve_cmap("RdBu").map(np.linspace(0, 1, 256), mode="byte"))


def test_the_bar_of_a_slice_shows_its_classes_as_blocks_with_the_breaks(qtbot):
    canvas = _canvas(qtbot)
    canvas.set_colorbar(True, "h(x)")
    canvas.set_levels("-0.2, 0, 0.3", "#ff0000,#00ff00,#0000ff,none")
    bar = canvas.color_bar
    assert bar.ticks() == [(0.25, "−0.2"), (0.5, "0"), (0.75, "0.3")]
    assert bar.colours().tolist() == [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255],
                                      [0, 0, 0, 0]]
    image = canvas.grab().toImage()
    ratio = image.devicePixelRatio()
    for position, rgb in ((0.125, (255, 0, 0)), (0.375, (0, 255, 0)), (0.625, (0, 0, 255))):
        at = QtCore.QPointF(canvas.mapFromScene(bar.strip_scene_point(position))) * ratio
        assert _rgb(image, at).tolist() == list(rgb)
    canvas.set_levels("3")                                          # quantile classes, the ramp
    breaks = np.quantile(_H, [1 / 3, 2 / 3])
    assert [p for p, _t in bar.ticks()] == pytest.approx([1 / 3, 2 / 3])
    assert [float(t.replace("−", "-")) for _p, t in bar.ticks()] == pytest.approx(breaks, abs=5e-4)


def test_the_bar_of_bands_shows_the_periodic_colouring_with_the_band_edges(qtbot):
    from dynamix.core.stretch import bands
    from dynamix.shell.canvas import _resolve_cmap

    canvas = _canvas(qtbot, colormap="binary_r")
    canvas.set_colorbar(True, "z (m)")
    canvas.set_bands("alternate", 0.25, 0.0)
    bar = canvas.color_bar
    edges = [-0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    assert [float(t.replace("−", "-")) for _p, t in bar.ticks()] == pytest.approx(edges)
    assert [p for p, _t in bar.ticks()] == pytest.approx([(e + 0.4) / 1.59 for e in edges])
    v = np.linspace(-0.4, 1.19, len(bar.colours()))
    expected = _resolve_cmap("binary_r").map(bands(v, "alternate", 0.25), mode="byte")
    assert np.array_equal(bar.colours(), expected)
    assert bar.label == "z (m)"


def test_the_export_carries_the_bar_only_while_it_is_on(qtbot):
    import pyqtgraph.exporters as exporters

    canvas = _canvas(qtbot, colormap="viridis")
    before = _pixels(canvas)
    exported_off = exporters.ImageExporter(canvas.view).export(toBytes=True)
    canvas.set_colorbar(True, "h(x)")
    on = _pixels(canvas)
    exported_on = exporters.ImageExporter(canvas.view).export(toBytes=True)
    assert on != before and exported_on != exported_off
    canvas.set_colorbar(False, "h(x)")
    assert _pixels(canvas) == before
    assert exporters.ImageExporter(canvas.view).export(toBytes=True) == exported_off
    # Off, the canvas renders as it does with no bar item in the scene at all.
    canvas.scene().removeItem(canvas.color_bar)
    assert _pixels(canvas) == before


def test_the_bar_menu_sets_its_side_and_size(qtbot):
    canvas = _canvas(qtbot)
    canvas.set_colorbar(True, "h(x)")
    bar = canvas.color_bar
    styles = []
    bar.styleChanged.connect(styles.append)
    menu = bar.context_menu()
    side = next(a.menu() for a in menu.actions() if a.text() == "Side")
    next(a for a in side.actions() if a.text() == "bottom").trigger()
    assert styles[-1]["side"] == "bottom" and not bar.vertical()
    view = canvas.view.boundingRect()
    placed = bar.mapRectToParent(bar.boundingRect())
    assert placed.bottom() <= view.bottom() and placed.center().x() == pytest.approx(
        view.center().x(), abs=60)
    small = bar._strip.width()
    bar.set_style({"size": "large"})
    assert bar._strip.width() > small


def _load(qtbot, window, values, name, units=""):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    field = RasterField(name=name, values=values, frame=LocalFrame(units="px"),
                        x_axis=np.arange(float(values.shape[1])),
                        y_axis=np.arange(float(values.shape[0])), units=units)
    with qtbot.waitSignal(window.resolved, timeout=10000):
        window.load_field(field, f"mem:{name}")
    return window


def test_the_toggle_is_a_view_tag_and_the_bar_is_titled_with_the_quantity_and_units(qtbot, window):
    from dynamix.engine.execution import recipe_identity
    from dynamix.shell.main_window import _display_style_of

    _load(qtbot, window, np.abs(_H[:16, :16]) * 3.0, "KAM", "deg")
    layer = window.layer
    key = recipe_identity(layer, window._layer_input(layer))
    assert not window.canvas.color_bar.isVisible()
    window._on_display_style_changed("colorbar", True)
    assert layer.tags["ui.colorbar"] == "True"
    assert window.canvas.color_bar.isVisible() and window.canvas.color_bar.label == "KAM (deg)"
    assert window.right_panel.colorbar_check.isChecked()
    window.canvas.color_bar.labelEdited.emit("KAM / °")             # the bar's Title…
    assert layer.tags["ui.colorbar_label"] == "KAM / °"
    assert window.canvas.color_bar.label == "KAM / °"
    window.canvas.color_bar.labelEdited.emit("")
    assert window.canvas.color_bar.label == "KAM (deg)"
    # A derived raster on show is named as its row names it, without the dataset's units.
    window._sync_holder_raster({"h_map": _H[:16, :16]})
    assert window.canvas.color_bar.label == "h(x)"
    window._sync_holder_raster({})
    assert window.canvas.color_bar.label == "KAM (deg)"
    window._on_display_style_changed("colorbar", False)
    assert not window.canvas.color_bar.isVisible()
    assert _display_style_of(layer)["colorbar"] is False
    assert recipe_identity(layer, window._layer_input(layer)) == key


def test_the_centre_controls_show_for_a_diverging_ramp_and_reach_the_canvas(qtbot, window):
    loaded = _load(qtbot, window, _H[:16, :24], "h")
    panel = loaded.right_panel
    centre = panel._relevant_items["knob:center"]
    symmetric = panel._relevant_items["symmetric"]
    assert centre.isHidden() and symmetric.isHidden()             # binary_r: nothing to centre
    loaded._on_display_style_changed("colormap", "coolwarm")
    assert not centre.isHidden() and not symmetric.isHidden()
    loaded._on_display_style_changed("center", 0.25)
    loaded._on_display_style_changed("symmetric", True)
    assert loaded.layer.tags["ui.center"] == repr(0.25) and loaded.layer.tags["ui.symmetric"] == "True"
    assert loaded.canvas._center == (0.25, True) and loaded.canvas._diverging
    shown = np.asarray(loaded.canvas.image_item.image)
    values = _H[:16, :24]
    from dynamix.core.stretch import centered

    assert np.allclose(shown.T, centered(values, values.min(), values.max(), 0.25, True))
    loaded._sync_display_controls(loaded.layer)
    assert panel.symmetric_check.isChecked()
    assert panel._display_controls["center"]._value == pytest.approx(0.25)


def test_the_ramp_editor_flags_a_diverging_ramp(qtbot):
    from dynamix.shell import ramps
    from dynamix.shell.ramp_editor import RampEditorDialog

    seeded = RampEditorDialog("RdBu")
    qtbot.addWidget(seeded)
    assert seeded.diverging_check.isChecked()
    plain = RampEditorDialog("viridis")
    qtbot.addWidget(plain)
    assert not plain.diverging_check.isChecked()
    seeded.name_edit.setText("h signed")
    assert seeded.save() and ramps.is_diverging("h signed")
    again = RampEditorDialog("h signed")
    qtbot.addWidget(again)
    assert again.diverging_check.isChecked()
    again.diverging_check.setChecked(False)
    assert again.save() and not ramps.is_diverging("h signed")


def test_the_map_view_paints_a_diverging_row_about_its_centre(qtbot):
    from dynamix.core.stretch import centered
    from dynamix.shell.main_window import _display_style_of
    from dynamix.shell.map_view import _RowPainter
    from dynamix.model.layer import Layer

    layer = Layer(0, "h", "s0", tags={"ui.colormap": "RdBu", "ui.center": repr(0.2)})
    painter = _RowPainter()
    painter.apply_style(_display_style_of(layer), None)
    shown = painter._display_scalars(_H)
    assert np.allclose(shown, centered(_H, _H.min(), _H.max(), 0.2))
