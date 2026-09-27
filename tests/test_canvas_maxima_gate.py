# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The raster view's maxima overlays: the Overlay opacity reaches them, and one switch hides them.

In the raster view the maxima draw as the pixel overlay (``extrema_raster_item``) and, when an
arrow mode is on, as gradient arrows (``arrow_item``). ``set_display_style``'s opacity fades
both, and ``set_maxima_visible`` hides both immediately and on every later ``set_result``,
leaving the result's extrema untouched for filters, the spectrum and anisotropy.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.shell.canvas import Canvas

from tests.test_canvas_raster_extrema import _result


def _canvas_with_extrema(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.set_field(np.zeros((16, 16), dtype=np.float32))
    result = _result()
    canvas.set_arrow_mode("all")
    canvas.set_result(result, 0)
    return canvas, result


def _arrow_count(canvas) -> int:
    xs = canvas.arrow_item.getData()[0]
    return 0 if xs is None else len(xs)


def test_overlay_opacity_reaches_the_raster_view_maxima_and_arrows(qtbot):
    canvas, _ = _canvas_with_extrema(qtbot)
    canvas.set_display_style(opacity=0.3)
    assert canvas.extrema_raster_item.opacity() == pytest.approx(0.3)
    assert canvas.arrow_item.opacity() == pytest.approx(0.3)
    assert canvas.extrema_item.opacity() == pytest.approx(0.3)


def test_line_width_leaves_the_arrow_pen_alone(qtbot):
    canvas, _ = _canvas_with_extrema(qtbot)
    before = canvas.arrow_item.opts["pen"].widthF()
    canvas.set_display_style(line_width=4.0)
    assert canvas.arrow_item.opts["pen"].widthF() == pytest.approx(before)


def test_set_maxima_visible_hides_dots_and_arrows_and_survives_a_redraw(qtbot):
    canvas, result = _canvas_with_extrema(qtbot)
    assert canvas.extrema_raster_item.isVisible() and canvas.arrow_item.isVisible()
    assert _arrow_count(canvas) > 0
    canvas.set_maxima_visible(False)
    assert not canvas.extrema_raster_item.isVisible()
    assert not canvas.arrow_item.isVisible()
    canvas.set_result(result, 0)                      # a later redraw keeps them hidden
    assert not canvas.extrema_raster_item.isVisible()
    assert not canvas.arrow_item.isVisible()
    assert len(result["extrema"][0]["x"]) == 4        # the measurement itself is untouched
    canvas.set_maxima_visible(True)
    canvas.set_result(result, 0)
    assert canvas.extrema_raster_item.isVisible()
    assert canvas.arrow_item.isVisible()
    assert np.asarray(canvas.extrema_raster_item.image)[..., 3].any()


def test_showing_the_maxima_keeps_arrow_mode_off_drawing_nothing(qtbot):
    canvas, result = _canvas_with_extrema(qtbot)
    canvas.set_arrow_mode("off")
    canvas.set_maxima_visible(False)
    canvas.set_maxima_visible(True)
    canvas.set_result(result, 0)
    assert _arrow_count(canvas) == 0


def test_the_maxima_default_to_visible(qtbot):
    canvas, _ = _canvas_with_extrema(qtbot)
    assert canvas._maxima_visible is True
    assert canvas.extrema_raster_item.isVisible()
