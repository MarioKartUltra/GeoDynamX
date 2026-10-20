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
    stamp = produced.provenance["edges.filled"]
    assert (stamp["rows"], stamp["cols"]) == (96, 96)
    r0, c0 = stamp.get("row0", 0), stamp.get("col0", 0)              # centred: the block sits inside
    np.testing.assert_array_equal(np.asarray(produced.values)[r0:r0 + 96, c0:c0 + 96], v[20:116, 30:126])


def test_the_canvas_draws_a_result_on_the_extended_raster_back_on_the_data(qtbot, registered_builtins):
    """The chain extension (centred) → wtmm2d drawn over the layer's own field: the maxima land
    on the data (the extended frame's origin subtracted), as the author saw them displaced."""
    from dynamix.engine.cache import Cache
    from dynamix.engine.resolve import resolve
    from dynamix.shell.canvas import Canvas
    from tests.conftest import fbm2d, make_field, make_layer

    field = make_field(fbm2d(256, 0.5, seed=1))
    res = resolve(make_layer(("periodic_extension", {}), ("wtmm2d", {"n_oct": 2})), field, Cache()).result
    r0, c0 = res["_stage"]["origin"]
    assert r0 > 0 and c0 > 0
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(field)
    canvas.set_result(res, 0)
    ext = res["extrema"][0]
    iso = np.asarray(ext["line_id"]) == -1
    xs, ys = canvas.extrema_item.getData()
    if iso.any():
        assert np.allclose(xs, np.asarray(ext["x"])[iso] - c0) and np.allclose(ys, np.asarray(ext["y"])[iso] - r0)
    assert canvas._draw_off == (-float(r0), -float(c0))
    # drawn on the extended raster itself (a fork of the extension), nothing moves
    extended = RasterField(name="ext", values=np.zeros(res["_shape"]), frame=field.frame,
                           x_axis=np.arange(float(res["_shape"][1])), y_axis=np.arange(float(res["_shape"][0])))
    canvas.set_field(extended)
    canvas.set_result(res, 0)
    assert canvas._draw_off == (0.0, 0.0)


def test_a_centred_extension_lands_with_its_data_block_on_the_roi(qtbot, registered_builtins):
    """The produced field's place on the canvas comes from its own axes against the dataset's
    grid, as a fork's does: a centred extension starts half a gap before the ROI, so its raster
    draws from there and the data block sits exactly on the ROI box (it drew from the ROI's
    corner, the block half a gap down and right of the box, as the author saw)."""
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
    _layer_id, produced = win._active_field
    stamp = produced.provenance["edges.filled"]
    r0, c0 = stamp.get("row0", 0), stamp.get("col0", 0)
    assert r0 > 0 and c0 > 0                                            # the centred layout
    rect = win.canvas.image_item.mapRectToView(win.canvas.image_item.boundingRect())
    assert (rect.x(), rect.y()) == (30 - c0 - 0.5, 20 - r0 - 0.5)
    assert (rect.width(), rect.height()) == tuple(float(n) for n in produced.values.shape[::-1])
