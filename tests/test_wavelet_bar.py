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
    ox, oy = bar.offset()
    bar.drag_by(-100.0, -50.0)
    assert moved and tuple(moved[-1]) == (ox - 100.0, oy - 50.0)
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


def test_the_bar_is_anchored_at_its_centre(qtbot):
    """A wider profile (or a zoom) grows the bar about its centre, never from one edge."""
    w = _view(qtbot)
    bar = WaveletBar()
    bar.setParentItem(w.getViewBox())
    bar.set_profile(_profile(20.0), "20 m", 1.0)
    x0 = bar.scenePos().x()
    bar.set_profile(_profile(80.0), "80 m", 1.0)
    assert bar.bracket_px() > 2 * 20.0 and bar.scenePos().x() == pytest.approx(x0)


def test_a_deep_zoom_in_and_back_out_leaves_the_bar_where_it_was(qtbot):
    w = _view(qtbot)
    bar = WaveletBar()
    bar.setParentItem(w.getViewBox())
    bar.set_profile(_profile(20.0), "20 m", 1.0)
    offset0, x0 = bar.offset(), bar.scenePos().x()
    w.getViewBox().setRange(xRange=(295.0, 305.0), yRange=(195.0, 205.0), padding=0)   # 60x: wider than the view
    assert w.getViewBox().sceneBoundingRect().contains(bar.scenePos())    # its centre stays on screen
    w.getViewBox().setRange(xRange=(0.0, 600.0), yRange=(0.0, 400.0), padding=0)
    assert bar.offset() == offset0 and bar.scenePos().x() == pytest.approx(x0)


def test_the_cascade_keeps_its_centre_when_the_ladder_widens(qtbot):
    from dynamix.core.wavelet_profiles import measure_profile
    from dynamix.shell.wavelet_bar import WaveletCascade

    w = _view(qtbot)
    c = WaveletCascade()
    c.setParentItem(w.getViewBox())
    c.set_profiles([measure_profile("gaussian", r) for r in (2.0, 4.0)], ("2", "4"), 0)
    x0 = c.scenePos().x()
    c.set_profiles([measure_profile("gaussian", r) for r in (2.0, 4.0, 8.0, 16.0)], ("2", "16"), 0)
    assert c.scenePos().x() == pytest.approx(x0)


def test_the_curves_amplitude_grows_with_their_width_on_screen(qtbot):
    """Zoomed out the curve keeps its floor height; zoomed in it stays proportional to its own
    width on screen (never a long flat line at pixel level), each cascade level at its own."""
    from dynamix.core.wavelet_profiles import measure_profile
    from dynamix.shell.wavelet_bar import CURVE_ASPECT, CURVE_HEIGHT_PX, WaveletCascade

    w = _view(qtbot)
    bar = WaveletBar()
    bar.setParentItem(w.getViewBox())
    bar.set_profile(_profile(20.0), "20 m", 1.0)
    assert bar.amplitude_px() == CURVE_HEIGHT_PX                        # the floor
    w.getViewBox().setRange(xRange=(0, 30), yRange=(0, 20), padding=0)  # 20x in
    assert bar.amplitude_px() == pytest.approx(CURVE_ASPECT * bar.curve_width_px())
    assert bar.amplitude_px() > CURVE_HEIGHT_PX
    c = WaveletCascade()
    c.setParentItem(w.getViewBox())
    c.set_profiles([measure_profile("gaussian", r) for r in (2.0, 4.0, 8.0)], ("2 px", "8 px"), 1)
    amps, widths = c.amplitudes_px(), c.widths_px()
    assert amps[2] == pytest.approx(CURVE_ASPECT * widths[2])
    assert amps[2] > amps[0] >= CURVE_HEIGHT_PX


def test_the_wtmm_knob_names_the_smoother_and_the_bar_names_the_analyser(clean_registry):
    """The wtmm2d choice "mexican"/"gaussian" is the smoothing function θ; the analysing
    wavelet is its gradient, g3 or g1 -- the knob says so and the bar's family does too."""
    from dynamix.devices import register_builtin_devices
    from dynamix.core.wavelet_profiles import wtmm_profile
    from dynamix.model.device import get_device

    register_builtin_devices()
    param = next(p for p in get_device("wtmm2d").params if p.name == "wavelet")
    assert "θ" in param.label
    labels = dict(zip(param.choices, param.choice_labels))
    assert "g3" in labels["mexican"] and "g1" in labels["gaussian"]
    assert "g3" in wtmm_profile("mexican", 8.0).family and "θ" in wtmm_profile("mexican", 8.0).family
    assert "g1" in wtmm_profile("gaussian", 8.0).family


def _action(menu, *path):
    """The QAction reached by menu titles then an action text."""
    node = menu
    for title in path[:-1]:
        node = next(a.menu() for a in node.actions() if a.menu() is not None and a.text() == title)
    return next(a for a in node.actions() if a.text() == path[-1])


def test_the_bars_carry_a_style_with_a_right_click_menu(qtbot):
    from dynamix.shell.wavelet_bar import DEFAULT_WAVELET_STYLE, WaveletCascade

    w = _view(qtbot)
    bar = WaveletBar()
    bar.setParentItem(w.getViewBox())
    bar.set_profile(_profile(20.0), "20 m", 1.0)
    assert bar.style() == DEFAULT_WAVELET_STYLE and bar.style()["units"] == "auto"
    seen = []
    bar.styleChanged.connect(seen.append)
    menu = bar.context_menu()
    _action(menu, "Units", "km").trigger()
    assert bar.style()["units"] == "km" and seen[-1]["units"] == "km"
    _action(menu, "Outline").trigger()
    assert bar.style()["outline"] is False
    _action(menu, "Line width", "bold").trigger()
    assert bar.style()["width"] == "bold"
    bar.set_style({"color": "#ff00ff"})
    assert bar.style()["color"] == "#ff00ff" and bar.colour().name() == "#ff00ff"
    c = WaveletCascade()
    c.set_style({"color": "#00ffff", "outline": False})
    assert c.style()["color"] == "#00ffff" and c.colour().name() == "#00ffff"
    scale = DraggableScaleBar(size=100, suffix="m")
    scale.set_style({"color": "#ff0000"})
    assert scale.bar.pen().color().name() == "#ff0000" and scale.style()["color"] == "#ff0000"
    assert _action(scale.context_menu(), "Units", "m") is not None
