# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Gradient arrows on the canvas and the Anisotropy window (WTMMM angle statistics)."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets  # noqa: F401

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.shell.canvas import Canvas, _arrow_segments


def test_arrow_segments_point_uphill_with_two_head_strokes():
    xs, ys = _arrow_segments([0.0], [0.0], [0.0], 10.0)
    assert xs.size == 6                                   # shaft + two head strokes, as pairs
    assert (xs[0], ys[0], xs[1], ys[1]) == pytest.approx((0.0, 0.0, 10.0, 0.0))
    assert xs[3] < 10.0 and xs[5] < 10.0 and ys[3] == pytest.approx(-ys[5])   # heads fold back
    xs, ys = _arrow_segments([0.0], [0.0], [np.pi / 2], 4.0)
    assert (xs[1], ys[1]) == pytest.approx((0.0, 4.0))    # +90 = +row (down on screen)


def _wtmm_result(field, clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device

    register_builtin_devices()
    dev = get_device("wtmm2d")
    res = dev.compute(field, {**defaults_for(dev), "n_oct": 3})
    res.setdefault("_shape", field.values.shape)
    return res


def _field(crs=None):
    rng = np.random.default_rng(2)
    v = rng.normal(size=(64, 64)).cumsum(0).cumsum(1)
    x0, y0 = 500_000.0, 4_000_000.0
    return RasterField(name="f", values=v,
                       frame=LocalFrame(x0=x0, y0=y0, dx=30.0, dy=30.0, units="metre"),
                       x_axis=x0 + 30.0 * np.arange(64), y_axis=y0 - 30.0 * np.arange(64),
                       provenance={"crs": crs} if crs else {})


def test_the_canvas_draws_arrows_only_when_asked_fewer_on_wtmmm(qtbot, clean_registry):
    f = _field()
    res = _wtmm_result(f, clean_registry)
    c = Canvas()
    qtbot.addWidget(c)
    c.set_field(f)
    c.set_result(res, 1)
    assert c.arrow_item.getData()[0] is None or len(c.arrow_item.getData()[0]) == 0
    c.set_arrow_mode("all")
    n_all = len(c.arrow_item.getData()[0])
    assert n_all == 6 * len(res["extrema"][1]["x"])
    c.set_arrow_mode("wtmmm")
    n_w = len(c.arrow_item.getData()[0])
    assert 0 < n_w < n_all
    c.clear_overlays()
    assert len(c.arrow_item.getData()[0] if c.arrow_item.getData()[0] is not None else []) == 0


from tests.test_shell_window import stub_devices, window  # noqa: E402,F401


def test_the_arrow_setting_is_saved_with_the_layer(window, qtbot):
    win = window
    win._start_worker = lambda: None
    win.load_field(_field(), "mem:arrows")
    combo = win.right_panel.arrows_combo
    combo.setCurrentIndex(combo.findData("wtmmm"))
    assert win.layer.tags["ui.arrows"] == "wtmmm" and win.canvas._arrow_mode == "wtmmm"


@pytest.mark.parametrize("crs, frames", [(None, ["pixel"]),
                                         ("EPSG:32611", ["pixel", "grid", "true"])])
def test_the_anisotropy_window_plots_arneodos_three_views(window, qtbot, crs, frames):
    pytest.importorskip("rasterio")
    win = window
    win._start_worker = lambda: None
    f = _field(crs)
    win.load_field(f, f"mem:aniso{crs}")
    win._active_result = _wtmm_result(f, None)
    win._refresh_anisotropy()
    assert win._anisotropy_button.isEnabled()
    win._on_anisotropy_button_clicked()
    aw = win._anisotropy_window
    assert aw.isVisible()
    got = [aw.frame_combo.itemData(i) for i in range(aw.frame_combo.count())]
    assert got == frames
    for i in range(aw.frame_combo.count()):                  # every frame draws all three
        aw.frame_combo.setCurrentIndex(i)
        assert len(aw.pdf_plot.listDataItems()) >= 2
        assert len(aw.plane_plot.listDataItems()) == 1
        assert len(aw.sector_plot.listDataItems()) == 4


def test_the_directional_roughness_rows_read_the_components_and_follow_their_controls(qtbot, clean_registry):
    """Khalil et al. 2006 in the window: the component pdfs raw and collapsed, the exponents in
    the readout, the rescaling spins seeded from the estimate until touched, the σ sweep, the
    q-collapse, the anisotropy against scale, the cut spectra, and the leaders' readings."""
    from dynamix.core import anisotropy as an
    from dynamix.shell.anisotropy_window import ESTIMATORS, AnisotropyWindow

    f = _field()
    res = _wtmm_result(f, clean_registry)
    aw = AnisotropyWindow()
    qtbot.addWidget(aw)
    aw.set_result(res, f, scale_idx=1, directional=[("L2", 0.41, 0.52)])
    scales = np.asarray(res["scales"], float)
    assert (aw.lo_spin.value(), aw.hi_spin.value()) == pytest.approx((np.log2(scales.min()), np.log2(scales.max())), abs=0.01)
    for plot in aw.comp_plots + aw.collapse_plots + [aw.mod_plot, aw.qcollapse_plot, aw.height_plot,
                                                      aw.angle_plot, aw.spectra_plot]:
        assert len(plot.listDataItems()) >= 1, plot.plotItem.titleLabel.text
    assert aw._power_item.image is not None and aw._power_item.image.shape == (64, 64)
    text = aw.readout.text()
    assert text.startswith("H_x ") and "H_y " in text and "median slope" in text and "M follows T_ψ" in text
    assert "β_x" in text and "q within 10 %" in text or "no q keeps" in text
    assert "DWT (L2): η(2, e₁)/2 0.41, η(2, e₂)/2 0.52" in text
    # the spins carry the estimate of the chosen estimator
    w = an.wtmmm_by_scale(res["extrema"])
    comps = [an.gradient_plane(s["mod"], s["arg"]) for s in w]
    h = an.directional_hurst(comps, scales, *aw._window())
    assert aw.hx_spin.value() == pytest.approx(h["median"][0][0], abs=1e-3)
    assert aw.hy_spin.value() == pytest.approx(h["median"][1][0], abs=1e-3)
    aw.estimator_combo.setCurrentText("std slope")
    assert aw.hx_spin.value() == pytest.approx(h["std"][0][0], abs=1e-3)
    assert ESTIMATORS["collapse"] == "collapse"
    # a touched spin holds; the seed button hands it back to the estimate
    n_before = len(aw.collapse_plots[0].listDataItems())
    aw.hx_spin.setValue(0.9)
    assert aw._seeded and "a^H_x = 0.900" in aw.collapse_plots[0].plotItem.titleLabel.text
    aw.estimator_combo.setCurrentText("median slope")                 # a new fit re-seeds
    assert not aw._seeded and aw.hx_spin.value() == pytest.approx(h["median"][0][0], abs=1e-3)
    aw.hx_spin.setValue(0.9)
    aw.seed_button.click()
    assert aw.hx_spin.value() == pytest.approx(h["median"][0][0], abs=1e-3)
    aw.sweep_check.setChecked(True)
    assert len(aw.collapse_plots[0].listDataItems()) > n_before      # the ± σ companions
    aw.q_spin.setValue(-1.0)
    assert "q = -1" in aw.qcollapse_plot.plotItem.titleLabel.text
    # the fit window from the fitter narrows the scales drawn
    aw.set_scale_window(float(np.log2(scales[1])), float(np.log2(scales[-2])))
    assert aw.lo_spin.value() == pytest.approx(np.log2(scales[1]), abs=0.01)
    assert len(aw.comp_plots[0].listDataItems()) == scales.size - 2
    # no result: the rows empty and the readout says what is needed
    aw.set_result({}, f)
    assert "maxima are needed" in aw.readout.text() and not aw.comp_plots[0].listDataItems()


def test_bearings_relabel_the_exponents_and_rotate_the_components(qtbot, clean_registry):
    pytest.importorskip("rasterio")
    from dynamix.shell.anisotropy_window import AnisotropyWindow

    f = _field("EPSG:32611")
    res = _wtmm_result(f, clean_registry)
    aw = AnisotropyWindow()
    qtbot.addWidget(aw)
    aw.set_result(res, f)
    pixel = aw.readout.text()
    aw.frame_combo.setCurrentIndex(aw.frame_combo.findData("grid"))
    bearing = aw.readout.text()
    assert pixel.startswith("H_x") and bearing.startswith("H_E–W") and "H_N–S" in bearing
    assert aw.hx_label.text() == "rescale with H_E–W" and "T E–W" in aw.comp_plots[0].plotItem.titleLabel.text
    assert "E–W cuts" in aw.spectra_plot.plotItem.legend.items[0][1].text


def test_the_main_window_collects_the_directional_scaling_readings(window):
    from dynamix.shell.main_window import MainWindow

    win = window
    assert isinstance(win, MainWindow)
    layer = type("L", (), {"name": "directional"})()
    win._layer_by_id = {7: layer, 8: layer}
    win._landed_results = {
        7: {"_directional": {"p": np.array([1.0, 2.0, 3.0]), "eta_e1": np.array([0.3, 0.84, 1.2]),
                             "eta_e2": np.array([0.7, 1.0, 1.5])}},
        8: {"_directional": {"p": np.array([1.0, 3.0]), "eta_e1": np.array([0.3, 1.2]), "eta_e2": np.array([0.7, 1.5])}},
        9: {"h_map": None}}
    assert win._directional_readings() == [("directional", pytest.approx(0.42), pytest.approx(0.5))]
