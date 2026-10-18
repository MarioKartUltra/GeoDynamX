# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The edge footprint on the Raster canvas (spec 25 §3.4): the padded extent as a dashed outline,
the padding's content faintly outside the raster, and the band the Border cut removes hatched
inside it, registered like every other overlay (pixel centres on integers, the window offset)."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.shell.canvas import Canvas


def _field(n=16, window=None):
    f = RasterField(name="f", values=np.arange(n * n, dtype=np.float64).reshape(n, n),
                    frame=LocalFrame(), x_axis=np.arange(float(n)), y_axis=np.arange(float(n)))
    if window:
        f.provenance["window"] = {"row_off": window[0], "col_off": window[1]}
    return f


def _footprint(n=16, pad=4, border=3, stride=1):
    v = np.arange(n * n, dtype=np.float64).reshape(n, n)[::stride, ::stride]
    q = -(-pad // stride)
    return {"edge": "reflect", "pad": pad, "shown": pad, "padded": np.pad(v, q, mode="reflect"),
            "stride": stride, "offset": -q * stride, "border": border, "band": "border_cut",
            "shape": (n, n)}


def _extent(item):
    x, y = item.getData()
    return float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y))


def test_the_footprint_draws_the_outline_the_padding_and_the_band(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(_field())
    canvas.set_footprint(_footprint())
    assert _extent(canvas.footprint_outline_item) == (-4.5, 19.5, -4.5, 19.5)
    image = canvas.footprint_image_item
    assert image.isVisible() and image.zValue() < canvas.image_item.zValue()
    rect = image.mapRectToView(image.boundingRect())
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == (-4.5, -4.5, 24.0, 24.0)
    shown = np.asarray(image.image)
    assert np.isnan(shown[4:-4, 4:-4]).all() and np.isfinite(shown[:4]).all()   # outside only
    band = canvas.footprint_band_item.path()
    assert band.contains(QtCore.QPointF(0.0, 8.0)) and not band.contains(QtCore.QPointF(8.0, 8.0))
    assert band.boundingRect() == QtCore.QRectF(-0.5, -0.5, 16.0, 16.0)
    canvas.clear_footprint()
    assert canvas.footprint_outline_item.getData()[0] is None or not len(
        canvas.footprint_outline_item.getData()[0])
    assert not image.isVisible() and canvas.footprint_band_item.path().isEmpty()


def test_a_windowed_field_draws_its_footprint_at_its_file_position(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(_field(window=(5, 7)))
    canvas.set_footprint(_footprint())
    assert _extent(canvas.footprint_outline_item) == (7 - 4.5, 7 + 19.5, 5 - 4.5, 5 + 19.5)
    assert canvas.footprint_band_item.path().boundingRect() == QtCore.QRectF(6.5, 4.5, 16.0, 16.0)


def test_a_decimated_footprint_spans_the_padded_extent(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(_field())
    canvas.set_footprint(_footprint(stride=2))
    image = canvas.footprint_image_item
    rect = image.mapRectToView(image.boundingRect())
    assert (rect.x(), rect.width()) == (-4.5, 24.0)


def test_no_band_and_no_padding_draw_nothing_there(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(_field())
    canvas.set_footprint({**_footprint(), "border": 0})
    assert canvas.footprint_band_item.path().isEmpty()
    canvas.set_footprint(None)
    assert not canvas.footprint_image_item.isVisible()


def test_an_extensions_gap_beyond_the_shown_raster_is_drawn_faintly(qtbot):
    """After an in-chain Periodic extension the footprint is the extended field's, larger than
    the dataset on the canvas: only the dataset's own pixels are left to the raster."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(_field(16))
    canvas.set_footprint(_footprint(n=20))
    shown = np.asarray(canvas.footprint_image_item.image)           # (x, y)
    assert np.isnan(shown[4:20, 4:20]).all()
    assert np.isfinite(shown[20:24, 4:24]).all() and np.isfinite(shown[4:24, 20:24]).all()
    assert _extent(canvas.footprint_outline_item) == (-4.5, 23.5, -4.5, 23.5)


def test_an_roi_footprint_sits_on_the_roi_and_only_the_mirror_is_drawn(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(_field(16))
    fp = {**_footprint(n=6, pad=4, border=0), "origin": (2, 3)}          # a 6 px ROI at (2, 3)
    canvas.set_footprint(fp)
    assert _extent(canvas.footprint_outline_item) == (3 - 4.5, 3 + 5.5 + 4, 2 - 4.5, 2 + 5.5 + 4)
    shown = np.asarray(canvas.footprint_image_item.image)          # (x, y), first sample at (-1, -2)
    assert np.isfinite(shown[0]).all() and np.isfinite(shown[:, 0]).all()   # past the dataset
    assert np.isnan(shown[1:, 2:]).all()                                       # on the dataset
    rect = canvas.footprint_image_item.mapRectToView(canvas.footprint_image_item.boundingRect())
    assert (rect.x(), rect.y()) == (-1.5, -2.5)
