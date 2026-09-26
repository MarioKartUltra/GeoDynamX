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
