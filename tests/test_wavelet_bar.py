# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The wavelet bar item: a λ* bracket in data units, a screen-constant curve, draggable, anchored
inside the view; the draggable distance bar."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pytest

from dynamix.core.wavelet_profiles import Profile
from dynamix.shell.wavelet_bar import DraggableScaleBar, WaveletBar


def _profile(lam=20.0):
    x = np.linspace(-40.0, 40.0, 401)
    return Profile(x_px=x, psi=-x * np.exp(-x * x / 50.0) / 5.0, scale_px=10.0, lambda_px=lam,
                   family="test")


def _view(qtbot, w=600, h=400):
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.resize(w, h)
    widget.show()
    widget.getViewBox().setRange(xRange=(0, w), yRange=(0, h), padding=0)
    qtbot.wait(20)
    return widget


def test_the_bracket_is_lambda_wide_in_data_units(qtbot):
    w = _view(qtbot)
    bar = WaveletBar()
    bar.setParentItem(w.getViewBox())
    bar.set_profile(_profile(20.0), "20 m", 1.0)
    px_per_unit = w.getViewBox().width() / 600.0                       # the axes take some width
    assert bar.bracket_px() == pytest.approx(20.0 * px_per_unit, rel=0.02)
    w.getViewBox().setRange(xRange=(0, 300), yRange=(0, 200), padding=0)
    assert bar.bracket_px() == pytest.approx(40.0 * px_per_unit, rel=0.02)   # zoomed 2x
    bar.set_profile(_profile(20.0), "200 m", 10.0)                     # dx_data: 10 map units / px
    assert bar.bracket_px() == pytest.approx(400.0 * px_per_unit, rel=0.02)


def test_the_curve_stops_at_one_percent_and_two_lambda(qtbot):
    w = _view(qtbot)
    bar = WaveletBar()
    bar.setParentItem(w.getViewBox())
    bar.set_profile(_profile(20.0), "20 m", 1.0)
    xs = bar.curve_x_px()
    assert xs.min() >= -40.0 - 1e-9 and xs.max() <= 40.0 + 1e-9        # the ±2λ cap
    bar.set_profile(_profile(200.0), "200 m", 1.0)                      # a wide λ: the 1 % rule binds
    xs = bar.curve_x_px()
    assert xs.max() < 40.0


def test_the_bar_stays_inside_the_view_and_a_drag_moves_its_offset(qtbot):
    w = _view(qtbot)
    bar = WaveletBar()
    bar.setParentItem(w.getViewBox())
    bar.set_profile(_profile(), "20 m", 1.0)
    moved = []
    bar.moved.connect(moved.append)
    assert w.getViewBox().sceneBoundingRect().contains(bar.sceneBoundingRect())
    bar.drag_by(-100.0, -50.0)
    assert moved and tuple(moved[-1]) == (-120.0, -70.0)
    assert w.getViewBox().sceneBoundingRect().contains(bar.sceneBoundingRect())
    bar.drag_by(-1000.0, -1000.0)                                       # clamped inside
    assert w.getViewBox().sceneBoundingRect().contains(bar.sceneBoundingRect())


def test_the_label_text_and_the_hidden_state(qtbot):
    w = _view(qtbot)
    bar = WaveletBar()
    bar.setParentItem(w.getViewBox())
    bar.set_profile(_profile(), "52.0 μm", 1.0)
    assert bar.label == "52.0 μm" and bar.isVisible()
    bar.set_profile(None, "", 1.0)
    assert not bar.isVisible()


def test_the_distance_bar_drags(qtbot):
    w = _view(qtbot)
    sb = DraggableScaleBar(size=100, suffix="m")
    sb.setParentItem(w.getViewBox())
    sb.anchor((1, 1), (1, 1), offset=(-20, -20))
    moved = []
    sb.moved.connect(moved.append)
    sb.drag_by(-30.0, -10.0)
    assert tuple(moved[-1]) == (-50.0, -30.0)


def test_the_canvas_bar_follows_set_wavelet_profile_and_the_duo_stays_empty(qtbot):
    from dynamix.shell.canvas import Canvas

    c = Canvas()
    qtbot.addWidget(c)
    c.resize(500, 400)
    c.show()
    c.set_field(np.random.default_rng(0).standard_normal((64, 64)))
    c.set_wavelet_profile(_profile(20.0), "20 px")
    assert c.wavelet_bar.isVisible() and c.wavelet_bar.label == "20 px"
    assert c.wavelet_item.xData is None or len(c.wavelet_item.xData) == 0   # the duo draws nothing
    c.set_wavelet_profile(None, "")
    assert not c.wavelet_bar.isVisible()


def test_the_cascade_nests_its_curves_in_the_ladders_ratios(qtbot):
    from dynamix.core.wavelet_profiles import measure_profile
    from dynamix.shell.wavelet_bar import WaveletCascade

    w = _view(qtbot)
    c = WaveletCascade()
    c.setParentItem(w.getViewBox())
    profiles = [measure_profile("gaussian", r) for r in (2.0, 4.0, 8.0)]
    c.set_profiles(profiles, ("2 px", "8 px"), shown=1)
    widths = c.widths_px()
    assert len(widths) == 3
    assert widths[1] / widths[0] == pytest.approx(2.0, rel=0.05)
    assert widths[2] / widths[1] == pytest.approx(2.0, rel=0.05)
    assert c.isVisible() and c.labels == ("2 px", "8 px") and c.shown == 1
    assert w.getViewBox().sceneBoundingRect().contains(c.sceneBoundingRect())
    c.set_profiles([], ("", ""), 0)
    assert not c.isVisible()
