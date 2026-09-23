# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Double-update guard (2026-09-14): when the 3-D view is on screen, a filter tweak must NOT
redraw the hidden raster canvas -- it defers the overlay draw and marks it dirty, then redraws
once on flip back to raster. 
Driven offscreen with a plain dummy widget standing in at stack index 1, so the real (pyvista)
arrangement never has to be built -- this exercises the guard logic, not the scene.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.rasterfield import RasterField

FIXTURE = "tests/fixtures/kam_64.npz"


@pytest.fixture
def computed_window(qtbot, clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.shell.main_window import MainWindow, DEMO_CHAIN

    register_builtin_devices()
    win = MainWindow(steps=DEMO_CHAIN)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.load_field(RasterField.load_npz(FIXTURE), FIXTURE)
    return win, qtbot


def test_tweak_while_canvas_hidden_defers_the_overlay_draw(computed_window, monkeypatch):
    from PySide6 import QtWidgets

    win, qtbot = computed_window
    assert win._center_stack.currentIndex() == 0
    assert win._canvas_overlay_dirty is False

    # Stand a dummy widget in at index 1 and show it -- the 3-D view is "on screen" without
    # building the real arrangement.
    win._center_stack.addWidget(QtWidgets.QWidget())
    win._center_stack.setCurrentIndex(1)

    calls = []
    orig = win.canvas.set_result
    monkeypatch.setattr(win.canvas, "set_result", lambda *a, **k: calls.append(a))

    mi = win._index_of("modulus_threshold")
    win._on_param_changed(mi, "frac", 0.3)          # a live filter tweak while canvas hidden

    assert calls == []                               # the hidden canvas was NOT redrawn
    assert win._canvas_overlay_dirty is True         # ... but the deferral was recorded


def test_flip_back_to_raster_redraws_the_deferred_overlay(computed_window, monkeypatch):
    from PySide6 import QtWidgets

    win, qtbot = computed_window
    win._center_stack.addWidget(QtWidgets.QWidget())
    win._center_stack.setCurrentIndex(1)
    win._center_view = "geo"                          # simulate having flipped to the 3-D view
    win._on_param_changed(win._index_of("modulus_threshold"), "frac", 0.3)
    assert win._canvas_overlay_dirty is True

    calls = []
    real = win.canvas.set_result
    monkeypatch.setattr(win.canvas, "set_result",
                        lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    win._set_center_view("raster")                    # flip back to the canvas

    assert len(calls) == 1                            # redrawn exactly once, on flip-back
    assert win._canvas_overlay_dirty is False


def test_canvas_view_still_draws_every_tweak(computed_window, monkeypatch):
    """The guard must not touch the ordinary case: on the raster view, every tweak still draws."""
    win, qtbot = computed_window
    assert win._center_stack.currentIndex() == 0
    calls = []
    real = win.canvas.set_result
    monkeypatch.setattr(win.canvas, "set_result",
                        lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    win._on_param_changed(win._index_of("modulus_threshold"), "frac", 0.3)
    assert len(calls) == 1
    assert win._canvas_overlay_dirty is False
