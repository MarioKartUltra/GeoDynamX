# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""File > Export Figure: the Raster canvas or the Map view rendered as the screen shows it, at a
scale factor, with the distance bar's text, the wavelet bar and the colour bar (over the image, or
beside it in a strip of its own), and the figure's .gprv record beside the PNG."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from tests.test_colorbar import _H, _load, _middle, _rgb
from tests.test_shell_window import stub_devices, window  # noqa: F401

DEM = pathlib.Path(__file__).resolve().parents[1] / "docs" / "demo" / "dem_crop.npz"


def _shown(qtbot, window):
    window.resize(1200, 800)
    window.show()
    qtbot.waitExposed(window)
    QtWidgets.QApplication.processEvents()
    return window


def _signed(qtbot, window):
    """A window showing a signed field on RdBu with its colour bar on."""
    _load(qtbot, window, _H, "h")
    window._on_display_style_changed("colormap", "RdBu")
    window._on_display_style_changed("colorbar", True)
    _shown(qtbot, window)
    window.canvas.view.autoRange()
    QtWidgets.QApplication.processEvents()
    return window


def _export(window, monkeypatch, chosen, placement="over"):
    """Run the command with the save dialog answering ``chosen`` and the options their defaults
    (2x) with the colour bar ``placement``; returns the PNG's path in its bundle."""
    from dynamix.shell.figure_export import FigureExportDialog

    def accept(dialog):
        dialog.placement_combo.setCurrentIndex(dialog.placement_combo.findData(placement))
        return 1

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(chosen), ""))
    monkeypatch.setattr(FigureExportDialog, "exec", accept)
    window.commands["file.export_figure"].trigger()
    return chosen.parent / chosen.stem / chosen.name


def _record(png) -> dict:
    """The record beside ``png``: checked by the gprv library, its print step returned."""
    gprv = pytest.importorskip("gprv")
    record = png.with_name(png.name + ".gprv")
    assert record.is_file()
    assert gprv.check(record.read_bytes()) == []
    step = json.loads(record.read_text(encoding="utf-8"))["processStep"][-1]
    assert step["processingInformation"]["identifier"]["code"] == "export/figure_png"
    return step["processingInformation"]["runTimeParameters"]["figure"]


def _array(image: QtGui.QImage) -> np.ndarray:
    """``image``'s RGB pixels as an ``(h, w, 3)`` int array."""
    rgba = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    arr = np.frombuffer(rgba.constBits(), np.uint8).reshape(rgba.height(), rgba.bytesPerLine())
    return arr[:, :4 * rgba.width()].reshape(rgba.height(), rgba.width(), 4)[..., :3].astype(int)


def _region(image: QtGui.QImage, rect: QtCore.QRect, scale: int) -> np.ndarray:
    """The RGB pixels of ``image`` under the widget rect ``rect`` at ``scale``."""
    r = QtCore.QRect(rect.topLeft() * scale, rect.size() * scale)
    return _array(image)[r.top():r.bottom() + 1, r.left():r.right() + 1]


def test_the_options_default_to_twice_the_size_and_offer_the_placement_with_the_colour_bar(qtbot):
    from dynamix.shell.figure_export import FigureExportDialog

    plain = FigureExportDialog()
    qtbot.addWidget(plain)
    assert plain.scale() == 2 and plain.placement() == "over"
    assert [plain.scale_combo.itemData(i) for i in range(plain.scale_combo.count())] == [1, 2, 3, 4]
    with_bar = FigureExportDialog(colorbar=True)
    qtbot.addWidget(with_bar)
    with_bar.placement_combo.setCurrentIndex(1)
    assert with_bar.placement() == "beside"


def test_the_command_renders_the_canvas_at_twice_its_size_with_its_bars_and_a_checked_record(
        qtbot, window, tmp_path, monkeypatch):
    w = _signed(qtbot, window)
    canvas = w.canvas
    label = canvas.scale_bar_label
    assert label.isVisible() and label.text()
    png = _export(w, monkeypatch, tmp_path / "h_figure.png")
    assert png.is_file()
    image = QtGui.QImage(str(png))
    assert (image.width(), image.height()) == (2 * canvas.width(), 2 * canvas.height())
    # The colour bar as on screen: the middle of its strip wears the ramp's middle colour.
    at = QtCore.QPointF(canvas.mapFromScene(canvas.color_bar.strip_scene_point(0.5))) * 2
    assert np.abs(_rgb(image, at) - _middle()).max() <= 6
    # The distance bar's text: its colour is drawn inside its label's rect, and nothing of that
    # colour is there once the label is hidden.
    colour = QtGui.QColor(canvas._scale_bar_colour())
    target = np.array([colour.red(), colour.green(), colour.blue()])
    text = _region(image, label.geometry(), 2)
    assert np.abs(text - target).max(axis=-1).min() <= 12
    from dynamix.shell.figure_export import render_widget

    label.hide()
    bare = _region(render_widget(canvas, 2), label.geometry(), 2)
    label.show()
    assert np.abs(bare - target).max(axis=-1).min() > 12
    figure = _record(png)
    assert figure["view"] == "raster view" and figure["scale"] == 2
    assert figure["size_px"] == [image.width(), image.height()]
    assert figure["scale_bar"] == label.text()
    assert figure["colorbar"]["placement"] == "over" and figure["colorbar"]["title"] == "h"
    assert "gprv.check: valid" in w._status_message


@pytest.mark.parametrize("side", ["right", "left", "top", "bottom"])
def test_the_colour_bar_beside_the_image_leaves_the_canvas_as_a_bar_off_render(
        qtbot, window, tmp_path, monkeypatch, side):
    from dynamix.shell.figure_export import render_widget

    w = _signed(qtbot, window)
    canvas, bar = w.canvas, w.canvas.color_bar
    bar.set_style({"side": side})
    png = _export(w, monkeypatch, tmp_path / "h_beside.png", placement="beside")
    image = QtGui.QImage(str(png))
    width, height = 2 * canvas.width(), 2 * canvas.height()
    # Widened across the bar's side only, the canvas shifted past a strip on the left or top.
    assert (image.width() > width, image.height() > height) == (bar.vertical(), not bar.vertical())
    dx = image.width() - width if side == "left" else 0
    dy = image.height() - height if side == "top" else 0
    shown_at = QtCore.QPointF(canvas.mapFromScene(bar.strip_scene_point(0.5))) * 2
    canvas.set_colorbar(False, bar.label)
    off = _array(render_widget(canvas, 2))
    canvas.set_colorbar(True, bar.label)
    pixels = _array(image)
    assert np.array_equal(pixels[dy:dy + height, dx:dx + width], off)       # no data covered
    # The bar sits in the strip where the screen has it along that side: the ramp's middle
    # colour on the strip's line through its centre.
    x, y = int(shown_at.x()), int(shown_at.y())
    if bar.vertical():
        line = pixels[y, :dx] if side == "left" else pixels[y, width:]
    else:
        line = pixels[:dy, x] if side == "top" else pixels[height:, x]
    assert np.abs(line - _middle()).max(axis=-1).min() <= 6
    assert _record(png)["colorbar"]["placement"] == "beside"


def test_the_command_renders_the_map_view_with_its_bars(qtbot, window, tmp_path, monkeypatch):
    if not DEM.is_file():
        pytest.skip("the demo raster is not in this checkout")
    from dynamix.core.rasterfield import RasterField

    field = RasterField.load_npz(str(DEM))
    with qtbot.waitSignal(window.resolved, timeout=10000):
        window.load_field(field, str(DEM))
    window._on_display_style_changed("colormap", "RdBu")
    w = _shown(qtbot, window)
    w._set_center_view("map")
    QtWidgets.QApplication.processEvents()
    plot = w.map_view.plot
    scale_text = w.map_view.scale_bar.text.toPlainText()
    assert scale_text
    png = _export(w, monkeypatch, tmp_path / "dem_map.png")
    image = QtGui.QImage(str(png))
    assert (image.width(), image.height()) == (2 * plot.width(), 2 * plot.height())
    # The distance bar is in the figure: its label's text colour under its scene rect.
    text = w.map_view.scale_bar.text
    rect = plot.mapFromScene(text.mapToScene(text.boundingRect())).boundingRect()
    colour = text.textItem.defaultTextColor()
    target = np.array([colour.red(), colour.green(), colour.blue()])
    assert np.abs(_region(image, rect, 2) - target).max(axis=-1).min() <= 12
    figure = _record(png)
    assert figure["view"] == "map view" and figure["scale_bar"] == scale_text
    assert figure["view_range"]["units"] == "map units" and figure["colorbar"] is None
    assert figure["display"] is None
    [row] = figure["rows"]                       # one raster row on the map, with its mapping
    assert row["colormap"] == "RdBu" and row["limits"][0] < row["limits"][1]
    assert "gprv.check: valid" in w._status_message


def test_the_command_says_which_views_it_draws(qtbot, window, monkeypatch):
    _load(qtbot, window, _H[:16, :16], "h")
    asked = []
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *a, **k: asked.append(a) or ("", ""))
    window._center_view = "vector"
    window.commands["file.export_figure"].trigger()
    assert not asked and "Raster or the Map view" in window._status_message


def test_the_record_says_what_the_colours_stand_for(qtbot, window, tmp_path, monkeypatch):
    """The figure's mapping from values to colours: the colormap, the stretch, the raw values at
    the ramp's two ends and the centre a diverging ramp is painted about with its resolved mode
    (what the colour bar's ticks read), the overlays and the theme: a replay of the record's chain
    can paint the same picture."""
    from dynamix.core import stretch as st
    from dynamix.shell.settings import load_settings

    w = _signed(qtbot, window)
    canvas = w.canvas
    s = canvas._image_stride
    small = np.asarray(canvas._field.values, dtype=np.float64)[::s, ::s]
    lo, hi = st.stretch_limits(small, "linear", percent=2.0)
    figure = _record(_export(w, monkeypatch, tmp_path / "h_figure.png"))
    display = figure["display"]
    assert display["colormap"] == "RdBu"
    assert display["stretch"] == {"mode": "linear", "percent": 2.0}
    assert display["center"] == {"value": 0.0, "mode": "value"}      # auto: the field crosses 0
    assert display["limits"] == list(st.centered_limits(lo, hi, 0.0, False))
    assert display["slice"] is None and display["bands"] is None and display["hillshade"] is None
    assert figure["rows"] is None
    assert figure["overlays"]["arrows"] in ("off", "wtmmm", "all")
    assert figure["overlays"]["colors"]["color_hchain"].startswith("#")
    assert figure["theme"] == load_settings().theme

    w._on_display_style_changed("center_mode", "symmetric")
    QtWidgets.QApplication.processEvents()
    display = _record(_export(w, monkeypatch, tmp_path / "h_symmetric.png"))["display"]
    assert display["center"] == {"value": 0.0, "mode": "symmetric"}
    m = max(-lo, hi)
    assert display["limits"] == [-m, m]

    w._on_display_style_changed("center_mode", "none")
    QtWidgets.QApplication.processEvents()
    display = _record(_export(w, monkeypatch, tmp_path / "h_plain.png"))["display"]
    assert display["center"] is None                                 # the plain min..max mapping
    assert display["limits"] == [float(v) for v in canvas.image_item.getLevels()]


def _export_window(window, monkeypatch, chosen) -> tuple:
    """A spectrum window's Export figure… with the save dialog answering ``chosen`` and the
    options their defaults (2x): the PNG's path in its bundle and the name the dialog offered."""
    from dynamix.shell.figure_export import FigureExportDialog

    offered = []

    def save(*args, **_kw):
        offered.append(args[2])
        return str(chosen), ""

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", save)
    monkeypatch.setattr(FigureExportDialog, "exec", lambda dialog: 1)
    QtWidgets.QApplication.processEvents()                     # the inspector's rows settle
    window.export_button.click()
    return chosen.parent / chosen.stem / chosen.name, pathlib.Path(offered[0]).name


def test_a_figure_from_the_partition_fitter_records_its_own_fit_only(qtbot, window, tmp_path, monkeypatch):
    """The fitter's Export figure… renders its plots and the record names its own fit: the name,
    kind, scale window, fit mode, table, settings and the tangent drawn; it draws no D(h)
    comparison, so its record lists no spectra even while the board holds some."""
    from dynamix.shell.multifractal_window import MultifractalWindow
    from tests.test_multifractal_window import _broken_pair

    w = _signed(qtbot, window)
    fitter = MultifractalWindow(*_broken_pair(), parent=w)
    qtbot.addWidget(fitter)
    fitter.show()
    qtbot.waitExposed(fitter)
    fitter._table_combo.setCurrentText("sup (cmax)")
    fitter._region.setRegion((1.0, 6.0))
    fitter.send_primary()
    assert len(w.spectra_board) == 1
    png, offered = _export_window(fitter, monkeypatch, tmp_path / "fit.png")
    assert offered == f"{w.layer.name}_partition_functions.png"
    image = QtGui.QImage(str(png))
    assert (image.width(), image.height()) == (2 * fitter._glw.width(), 2 * fitter._glw.height())
    figure = _record(png)
    assert figure["view"] == "partition functions" and figure["view_range"]["units"] == "h, D(h)"
    assert figure["fit"]["name"] == "canonical (Arneodo WTMM) · sup (cmax)"
    assert figure["fit"]["scale_window"] == [1.0, 6.0] and figure["fit"]["table"] == "cmax"
    assert figure["fit"]["tangent"]["q_c"] == pytest.approx(fitter._tangent_now()["q_c"])
    assert "spectra" not in figure
    assert "gprv.check: valid" in w._status_message


def test_a_figure_from_the_construction_window_lists_the_spectra_drawn_with_their_identities(
        qtbot, window, tmp_path, monkeypatch):
    """The construction window's Export figure… renders its plots and the record names its own
    canonical fit with its bars and every spectrum it draws from the D(h) comparison: the display
    and full names, the kind, the identity, the scale window, the fit mode, the table, whether
    its bars were drawn and the tangent drawn with it; a hidden entry is not drawn and not
    listed."""
    from dynamix.model.spectra_board import SentSpectrum
    from dynamix.shell.multifractal_window import MultifractalWindow
    from dynamix.shell.spectrum_window import SpectrumWindow
    from tests.test_multifractal_window import _broken_pair

    w = _signed(qtbot, window)
    fitter = MultifractalWindow(*_broken_pair(), parent=w)
    qtbot.addWidget(fitter)
    fitter._table_combo.setCurrentText("sup (cmax)")
    fitter.with_tangent_check.setChecked(True)
    fitter.send_primary()
    fitter._region.setRegion((1.0, 6.0))
    fitter.with_tangent_check.setChecked(False)
    fitter.send_primary()
    board = w.spectra_board
    first, second = board.entries()
    board.rename(first.key, "cmax, full window")
    board.set_show_errors(second.key, False)
    hidden = board.add(SentSpectrum(name="hidden", full_name="hidden", kind="histogram", h=first.h, D=first.D))
    board.set_visible(hidden, False)
    construction = SpectrumWindow(*_broken_pair(), parent=w)
    qtbot.addWidget(construction)
    construction.show()
    qtbot.waitExposed(construction)
    png, offered = _export_window(construction, monkeypatch, tmp_path / "dh.png")
    assert offered == f"{w.layer.name}_dh_comparison.png"
    image = QtGui.QImage(str(png))
    assert (image.width(), image.height()) == (2 * construction._glw.width(), 2 * construction._glw.height())
    figure = _record(png)
    assert figure["view"] == "D(h) construction" and figure["view_range"]["units"] == "h, D(h)"
    fit = figure["fit"]
    assert fit["name"] == "canonical (Arneodo direct)" and fit["kind"] == "canonical" and fit["errors"] is True
    assert fit["table"] == "cmax" and fit["fit_mode"] == "naive"     # the sup table, the default
    a, b = figure["spectra"]                                   # the hidden entry is not drawn
    assert (a["name"], a["full_name"]) == ("cmax, full window", first.full_name)
    assert a["kind"] == "canonical" and a["table"] == "cmax" and a["fit_mode"] == "naive"
    assert a["identity"]["tool"] == first.identity["tool"] and a["errors"] is True
    assert a["tangent"]["q_c"] == pytest.approx(first.tangent["q_c"])
    assert b["scale_window"] == [1.0, 6.0] and b["errors"] is False and b["tangent"] is None
    assert "gprv.check: valid" in w._status_message


def test_a_figure_with_log_cumulants_records_their_sign(qtbot, window, tmp_path, monkeypatch):
    """The cumulants the fitter draws ride in its figure's record with their scale window, the
    sign they follow (the cumulant expansion, c₂ < 0 intermittent) and Arneodo's values; a
    cumulant spectrum drawn from the D(h) comparison names the same sign."""
    from dynamix.core import spectra
    from dynamix.model.spectra_board import SentSpectrum
    from dynamix.shell.multifractal_window import MultifractalWindow
    from dynamix.shell.spectrum_window import SpectrumWindow
    from tests.test_multifractal_window import _broken_pair, _hd_q

    w = _signed(qtbot, window)
    q = np.linspace(-3.0, 3.0, 25)
    hd = _hd_q(q, 0.7 * q - 0.3 * q ** 2 / 2)
    rng = np.random.default_rng(3)
    mods = [np.exp(rng.normal(0.7 * np.log(a), np.sqrt(3.0 - 0.3 * np.log(a)), 20000)) for a in hd["scales"]]
    fitter = MultifractalWindow(hd, hd, extrema_mods=mods, scales=hd["scales"], parent=w)
    qtbot.addWidget(fitter)
    fitter.show()
    qtbot.waitExposed(fitter)
    assert "cumulants" not in fitter.figure_record()               # opt-in: none fitted yet
    fitter.cumulant_check.setChecked(True)
    c = fitter.current_cumulants()
    cum = _record(_export_window(fitter, monkeypatch, tmp_path / "cumulants.png")[0])["cumulants"]
    assert cum["cumulant_sign"] == "expansion (c2 < 0 intermittent)"
    assert cum["c2"] == pytest.approx(c["c2"]) and cum["c2"] < 0
    assert cum["arneodo"]["c2"] == pytest.approx(-c["c2"]) and cum["arneodo"]["c1"] == pytest.approx(c["c1"])
    assert cum["scale_window"] == pytest.approx(list(fitter._region.getRegion()))

    h = np.linspace(0.2, 0.8, 7)
    w.spectra_board.add(SentSpectrum(name="LC · leaders", full_name="leaders, log-cumulants", kind="cumulant",
                                     h=h, D=2.0 - 4.0 * (h - 0.5) ** 2))
    construction = SpectrumWindow(*_broken_pair(), parent=w)
    qtbot.addWidget(construction)
    construction.show()
    qtbot.waitExposed(construction)
    (entry,) = _record(_export_window(construction, monkeypatch, tmp_path / "dh.png")[0])["spectra"]
    assert entry["kind"] == "cumulant" and entry["cumulant_sign"] == spectra.CUMULANT_SIGN
