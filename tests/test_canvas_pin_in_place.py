# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Child datasets pin IN PLACE.

The session canvas's data space becomes FILE-ABSOLUTE: a field carrying a window offset
(``provenance["window"]``) draws at that offset — image, extrema overlay, bands, ghosts —
and every input gesture (ROI drag, place-box, picks, transects) converts back through the
same base. A whole-file field has offset (0, 0) and nothing changes; a wtmm2d_roi overlay
on a windowed parent lands exactly where it always did (display_offset + base compose to
the file-absolute position).
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.shell.canvas import Canvas


def _child_field(n=16, row_off=5, col_off=7):
    f = RasterField(name="child", values=np.zeros((n, n)), frame=LocalFrame(),
                    x_axis=np.arange(n, dtype=np.float64),
                    y_axis=np.arange(n, dtype=np.float64))
    f.provenance["window"] = {"row_off": row_off, "col_off": col_off}
    return f


def _result(n=16):
    ext = {"x": np.array([3], dtype=np.int64), "y": np.array([2], dtype=np.int64),
           "mod": np.ones(1), "arg": np.zeros(1),
           "line_id": np.array([-1], dtype=np.int64)}
    return {"extrema": [ext], "chains": [], "_shape": (n, n), "scales": np.array([1.0])}


def test_windowed_field_draws_at_its_file_position(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(_child_field())
    rect = canvas.image_item.mapRectToView(canvas.image_item.boundingRect())
    assert (rect.x(), rect.y()) == (6.5, 4.5)          # (col_off-0.5, row_off-0.5)
    assert (rect.width(), rect.height()) == (16.0, 16.0)


def test_result_overlays_ride_at_the_same_offset(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(_child_field())
    canvas.set_result(_result(), 0)
    rect = canvas.extrema_raster_item.mapRectToView(
        canvas.extrema_raster_item.boundingRect())
    assert (rect.x(), rect.y()) == (6.5, 4.5)
    # the orphan dot at field-local (3, 2) lights file-absolute pixel (7+3, 5+2)
    rgba = np.asarray(canvas.extrema_raster_item.image).transpose(1, 0, 2)
    assert set(zip(*np.nonzero(rgba[..., 3]))) == {(2, 3)}   # local inside the overlay image


def test_whole_file_fields_are_unchanged(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(np.zeros((16, 16), dtype=np.float32))
    rect = canvas.image_item.mapRectToView(canvas.image_item.boundingRect())
    assert (rect.x(), rect.y()) == (-0.5, -0.5)


def test_roi_band_and_place_origin_convert_through_the_base(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(_child_field())
    # numeric-edit band: local pixels [row=2, col=3, 4x5] must DRAW at file-absolute cells
    canvas.show_roi_band(2, 3, 4, 5)
    xs, ys = canvas.roi_band_item.getData()
    assert (xs.min(), xs.max()) == (3 + 7 - 0.5, 3 + 5 + 7 - 0.5)
    assert (ys.min(), ys.max()) == (2 + 5 - 0.5, 2 + 4 + 5 - 0.5)
    # place-box origin: a click at file-absolute (12, 8) is field-local (5, 3);
    # a 3x3 box centred there starts at local (row=2, col=4)
    assert canvas._roi_place_origin((12.0, 8.0), 3, 3) == (2, 4)


def test_set_colormap_resolves_matplotlib_ramps(qtbot):
    """2026-09-22: the ramp
    combo lists EVERY matplotlib colormap, but pg.ImageItem.setColorMap(str) resolves only
    pyqtgraph's few local maps -- everything else raised and was swallowed. set_colormap now
    falls back to source='matplotlib', so 'terrain'/'RdBu'/'gray' actually apply."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(np.zeros((8, 8), dtype=np.float32))
    canvas.set_colormap("terrain")
    assert canvas._colormap == "terrain"
    import pyqtgraph as pg
    expected = pg.colormap.get("terrain", source="matplotlib").getLookupTable(nPts=32)
    got = canvas.image_item.getColorMap().getLookupTable(nPts=32)
    assert np.array_equal(got, expected)
    canvas.set_colormap("no-such-map-xyz")          # unknown stays tolerant, keeps terrain
    assert canvas._colormap == "terrain"


# ------------------------------------------------ The display picture

def _picture(ny=10, nx=14, s=5, full=(50, 70)):
    f = RasterField(name="pic", values=np.arange(ny * nx, dtype=float).reshape(ny, nx),
                    frame=LocalFrame(), x_axis=np.arange(nx, dtype=np.float64),
                    y_axis=np.arange(ny, dtype=np.float64))
    f.provenance.update({"display_stride": s, "full_dims": full,
                         "window": {"row_off": 0, "col_off": 0}})
    return f


def _rect(canvas):
    r = canvas.image_item.mapRectToView(canvas.image_item.boundingRect())
    return r.x(), r.y(), r.width(), r.height()


def test_picture_samples_are_drawn_over_their_native_blocks(qtbot):
    """Sample k covers FILE pixels [k*s, (k+1)*s): the image spans the native extent, so a
    box, an ROI child and a result all sit on the file pixels the user drew over."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(_picture())
    assert _rect(canvas) == (-0.5, -0.5, 70.0, 50.0)
    assert canvas._field_shape() == (50, 70)


def test_a_partial_last_block_keeps_exact_blocks_and_gestures_clamp_to_the_file(qtbot):
    """A file whose dims are not a multiple of the stride: every sample still covers exactly
    s pixels (the rect overhangs the file by < s), while gestures clamp to the FILE."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(_picture(full=(48, 69)))
    assert _rect(canvas) == (-0.5, -0.5, 70.0, 50.0)
    assert canvas._field_shape() == (48, 69)
    assert canvas._roi_place_origin((68.0, 47.0), 10, 10) == (38, 59)


def test_canvas_lod_decimation_has_no_far_edge_drift(qtbot):
    """The canvas's own display decimation (lod_stride): ceil(n/s) samples each cover exactly
    s pixels -- the rect is ceil(n/s)*s, never n (which squeezed every sample by up to one
    displayed sample at the far edge)."""
    f = RasterField(name="big", values=np.zeros((100, 100)), frame=LocalFrame(),
                    x_axis=np.arange(100.0), y_axis=np.arange(100.0))
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(f, max_dim=40)                      # lod stride 3 -> 34 samples
    assert _rect(canvas) == (-0.5, -0.5, 102.0, 102.0)
    assert canvas._field_shape() == (100, 100)


def test_saved_rois_are_outlined_at_their_file_pixels(qtbot):
    """Saved ROIs draw as outlines around the CELLS of their file pixels
    (centre registration: [c-0.5, c+w-0.5] x [r-0.5, r+h-0.5]), active one separately."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(_picture())
    canvas.set_saved_rois([{"label": "A", "row": 10, "col": 20, "h": 8, "w": 6, "active": False},
                           {"label": "B", "row": 30, "col": 40, "h": 5, "w": 5, "active": True}])
    x, y = canvas.saved_roi_item.getData()
    fx = x[np.isfinite(x)]
    assert fx.min() == 19.5 and fx.max() == 25.5
    ax, ay = canvas.active_roi_item.getData()
    assert np.nanmin(ax) == 39.5 and np.nanmax(ay) == 34.5
    assert [t.toPlainText() for t in canvas._saved_roi_labels] == ["A", "B"]
    canvas.set_saved_rois([])
    x, _ = canvas.saved_roi_item.getData()
    assert x is None or len(x) == 0
