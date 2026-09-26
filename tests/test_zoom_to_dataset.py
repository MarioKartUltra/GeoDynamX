# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""View > Zoom to Dataset (⌘0 / F): frame the selected row's raster, or its ROI window."""
from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets  # noqa: F401

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from tests.test_shell_window import stub_devices, window  # noqa: F401


def test_zoom_to_dataset_frames_the_raster_then_an_roi_window(window, qtbot):
    win = window
    win._start_worker = lambda: None
    f = RasterField(name="d", values=np.zeros((40, 60)), frame=LocalFrame(),
                    x_axis=np.arange(60.0), y_axis=np.arange(40.0))
    win.load_field(f, "mem:d")
    win.canvas.view.setRange(xRange=(5000, 6000), yRange=(5000, 6000), padding=0)
    assert [s.toString() for s in win._zoom_action.shortcuts()] == ["Ctrl+0", "F"]
    win._zoom_action.trigger()
    (x0, x1), (y0, y1) = win.canvas.view.viewRange()
    assert x0 <= 0 and x1 >= 59 and x1 - x0 < 90 and y0 <= 0 and y1 >= 39
    win.layer.tags["roi.window"] = "10,20,8,12"              # rows 10..17, cols 20..31
    win._zoom_action.trigger()
    (x0, x1), (y0, y1) = win.canvas.view.viewRange()
    assert 18 <= x0 <= 20 and 31 <= x1 <= 33 and y1 - y0 < 12
