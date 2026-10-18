# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The edge footprint in the window (spec 25 §3.4): moving one of WTMM's Edge, padding or Border
cut knobs draws it on the Raster canvas for a few seconds; View ▸ Edge Footprint keeps it on and
lets it follow the shown scale. Driven through the real window offscreen."""
from __future__ import annotations

import pytest
from PySide6 import QtGui

from dynamix.core.rasterfield import RasterField
from test_roi_shell_flow import registered_builtins  # noqa: F401

FIXTURE = "tests/fixtures/kam_64.npz"


def _drawn(win) -> bool:
    return not win.canvas.footprint_band_item.path().isEmpty() or bool(
        len(win.canvas.footprint_outline_item.getData()[0] or ()))


@pytest.fixture
def window(qtbot, clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.shell.main_window import MainWindow

    register_builtin_devices()
    win = MainWindow(steps=(("wtmm2d", {"n_oct": 2, "n_voice": 2}),))
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.load_field(RasterField.load_npz(FIXTURE), FIXTURE)
    win._auto_run_action.setChecked(False)           # an edge edit goes pending, nothing runs
    return win


def test_an_edge_knob_shows_the_footprint_for_a_while(window):
    win = window
    i = win._index_of("wtmm2d")
    assert not _drawn(win)
    win._on_param_changed(i, "thresh", 0.002)        # not an edge knob
    assert not _drawn(win)
    win._on_param_changed(i, "border_cut", "border_pct")
    assert _drawn(win) and win._edge_footprint_timer.isActive()
    win._edge_footprint_timer.timeout.emit()
    assert not _drawn(win)


def test_the_view_toggle_keeps_it_and_it_follows_the_scale(window):
    win = window
    i = win._index_of("wtmm2d")
    win._on_param_changed(i, "border_scope", "per_scale")
    win._on_param_changed(i, "border_cut", "border_pct")
    win.commands["view.edge_footprint"].setChecked(True)
    win._edge_footprint_timer.timeout.emit()
    assert _drawn(win)
    first = QtGui.QPainterPath(win.canvas.footprint_band_item.path())
    win._on_scale_changed(3)                         # per scale: the coarser scale's own border
    later = win.canvas.footprint_band_item.path()
    assert later.boundingRect() == first.boundingRect() and later != first
    win.commands["view.edge_footprint"].setChecked(False)
    assert not _drawn(win)


def test_on_an_roi_layer_the_footprint_sits_on_the_roi(qtbot, registered_builtins):
    """The ROI's run reads its halo around the ROI: the footprint is drawn there, as wide as the
    shown scale's halo (it was drawn around the whole raster)."""
    import numpy as np

    from dynamix.core.frames import LocalFrame
    from dynamix.devices.wtmm import WTMM2D
    from dynamix.model.device import validate_params
    from dynamix.shell.main_window import MainWindow
    from test_roi_shell_flow import _roi_child, _save

    v = np.random.default_rng(4).standard_normal((60, 70)).cumsum(0)
    win = MainWindow(steps=())
    qtbot.addWidget(win)
    win.load_field(RasterField(name="plain", values=v, frame=LocalFrame(), x_axis=np.arange(70.0),
                               y_axis=np.arange(60.0)), "/nonexistent/plain.npz")
    _save(win, row=10, col=12, h=20, w=24)
    _roi_child(win, qtbot, "wtmm2d")
    win._auto_run_action.setChecked(False)
    i = win._index_of("wtmm2d")
    win._on_param_changed(i, "pad_width", "fixed_px")
    params = validate_params(WTMM2D(), win._params[i])
    m = WTMM2D()._halo_plan(params, parent_shape=(60, 70))[0].margin
    x, y = win.canvas.footprint_outline_item.getData()
    assert (min(x), min(y)) == (12 - 0.5 - m, 10 - 0.5 - m)
    assert (max(x), max(y)) == (12 + 24 - 0.5 + m, 10 + 20 - 0.5 + m)
