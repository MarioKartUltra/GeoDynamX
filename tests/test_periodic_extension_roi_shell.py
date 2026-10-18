# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Periodic extension dropped on an ROI, through the real window offscreen: the ROI's own pixels
are extended, and the extended field lands on the canvas as the row's result (it failed with
"argument of type 'RasterField' is not iterable" in the region runner)."""
from __future__ import annotations

import numpy as np

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from test_roi_shell_flow import _drop, _save, registered_builtins  # noqa: F401
from tests.test_periodic_extension import _smooth


def test_an_roi_gets_its_periodic_extension(qtbot, registered_builtins):
    from dynamix.shell.main_window import MainWindow

    v = _smooth(160, seed=2)
    field = RasterField(name="smooth", values=v, frame=LocalFrame(), x_axis=np.arange(160.0),
                        y_axis=np.arange(160.0))
    win = MainWindow(steps=())
    qtbot.addWidget(win)
    win.load_field(field, "/nonexistent/smooth.npz")
    _save(win, row=20, col=30, h=96, w=96)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "periodic_extension")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    layer_id, produced = win._active_field
    assert layer_id == win.layer.layer_id
    assert produced.provenance["edges.filled"] == {"rows": 96, "cols": 96}
    np.testing.assert_array_equal(np.asarray(produced.values)[:96, :96], v[20:116, 30:126])
