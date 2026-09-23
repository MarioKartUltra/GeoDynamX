# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Raster-view extrema as PIXELS.

The session canvas renders the displayed layer's extrema as an RGBA overlay on the ORIGINAL
grid -- each surviving extremum lights its own pixel cell, xsmurf ext-image style: H-line
members in the H-line color, orphan dots (line_id == -1) in the extrema color. The polyline
items (``hchain_item``/``extrema_item``) keep their memoized geometry (the vector view and
the selection machinery still consume it) but are INVISIBLE here -- interpolated lines and
the line-width style belong to the vector view alone. V-trails (cross-scale chains) remain
polylines: a chain spans scales and has no single-pixel representation.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.shell.canvas import Canvas


def _result(n=16):
    ext = {
        "x": np.array([3, 4, 5, 9], dtype=np.int64),
        "y": np.array([7, 7, 7, 2], dtype=np.int64),
        "mod": np.ones(4), "arg": np.zeros(4),
        "line_id": np.array([0, 0, 0, -1], dtype=np.int64),
    }
    return {"extrema": [ext], "chains": [], "_shape": (n, n),
            "scales": np.array([1.0])}


def test_overlay_lights_exactly_the_extrema_pixels(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(np.zeros((16, 16), dtype=np.float32))
    canvas.set_result(_result(), 0)
    img = canvas.extrema_raster_item.image
    assert img is not None and img.shape[:2] == (16, 16)
    # the item stores the canvas's col-major transpose (the image_item convention) --
    # transpose back to (row, col) to compare against the extrema grid
    rgba = np.asarray(img).transpose(1, 0, 2)
    lit = set(zip(*np.nonzero(rgba[..., 3])))
    assert lit == {(7, 3), (7, 4), (7, 5), (2, 9)}
    # line members and orphans carry DIFFERENT colors
    assert not np.array_equal(rgba[7, 3, :3], rgba[2, 9, :3])


def test_overlay_rect_is_center_registered_like_the_image(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(np.zeros((16, 16), dtype=np.float32))
    canvas.set_result(_result(), 0)
    rect = canvas.extrema_raster_item.mapRectToView(
        canvas.extrema_raster_item.boundingRect())
    assert (rect.x(), rect.y()) == (-0.5, -0.5)
    assert (rect.width(), rect.height()) == (16.0, 16.0)


def test_polyline_items_are_invisible_in_the_raster_view(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(np.zeros((16, 16), dtype=np.float32))
    canvas.set_result(_result(), 0)
    assert not canvas.hchain_item.isVisible()
    assert not canvas.extrema_item.isVisible()
    # the geometry itself still flows (the memoized pipeline stays live for the
    # vector view / selection machinery)
    hx, _hy = canvas.hchain_item.getData()
    assert hx is not None and len(hx) > 0


def test_scale_select_narrows_the_closed_flags():
    from dynamix.devices.filters import ScaleSelect
    from dynamix.model.device import defaults_for

    res = _result()
    res["extrema"] = [res["extrema"][0], dict(res["extrema"][0])]
    res["scales"] = np.array([1.0, 2.0])
    res["_hline_runs"] = [[np.array([0, 1, 2])], [np.array([2, 1, 0])]]
    res["_hline_closed"] = [[False], [True]]
    dev = ScaleSelect()
    out = dev.apply(res, dict(defaults_for(dev), scale_idx=1))
    assert out["_ext_base_closed"] == [True]
    assert out["_ext_base_runs"][0].tolist() == [2, 1, 0]


def test_clear_overlays_takes_the_pixel_overlay_off_too(qtbot):
    """2026-09-22 (user: hiding the child layer "does not toggle" the extrema, and removing
    the parent leaves them standing): clear_overlays is the hide/removal path's sweep -- the
    raster-pixel overlay must leave with everything else set_result draws."""
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(np.zeros((16, 16), dtype=np.float32))
    canvas.set_result(_result(), 0)
    assert np.asarray(canvas.extrema_raster_item.image)[..., 3].any()
    canvas.clear_overlays()
    img = canvas.extrema_raster_item.image
    assert img is None or not np.asarray(img)[..., 3].any()
