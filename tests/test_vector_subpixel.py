# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Vector view draws single-scale maxima at their SUBPIXEL positions (pyvista suite)."""
from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from dynamix.geo.mapping import axis_at  # noqa: E402
from dynamix.model.layer import Layer  # noqa: E402
from dynamix.shell.arrangement.scene import Scene, _subpixel_of  # noqa: E402
from tests.test_arrangement_scene import _frame_field, _hline_result, _plotter  # noqa: E402


def test_subpixel_of_falls_back_to_the_pixel_and_is_none_without_refinement():
    layer = {"x": np.array([3, 4]), "y": np.array([2, 2]),
             "x_sub": np.array([3.3, np.nan]), "y_sub": np.array([1.8, 2.4])}
    xs, ys = _subpixel_of(layer, np.array([0, 1]))
    assert xs.tolist() == [3.3, 4.0] and ys.tolist() == [1.8, 2.4]
    assert _subpixel_of({"x": np.array([1]), "y": np.array([1])}, np.array([0])) is None


def test_h_lines_and_dots_sit_at_their_subpixel_positions():
    p = _plotter()
    scene = Scene(p)
    field = _frame_field()
    res = _hline_result()
    ext0 = res["extrema"][0]
    ext0["x_sub"] = ext0["x"] + np.array([0.3, -0.2, 0.1, 0.4, -0.4, 0.25])
    ext0["y_sub"] = ext0["y"] + np.array([0.1, 0.2, -0.3, 0.0, 0.2, -0.35])
    scene.set_frame_mode(True)
    scene.set_layers([{"layer": Layer(layer_id=0, name="l", source_id="s0"), "field": field,
                       "result": res, "status": "ok"}])
    hl = np.array(p.renderer.actors["layer-0-hlines"].mapper.dataset.points)
    on_line = np.flatnonzero(ext0["line_id"] >= 0)
    want_x = axis_at(field.x_axis, ext0["x_sub"][on_line])
    assert np.allclose(np.sort(hl[:, 0]), np.sort(want_x))          # not the pixel centers
    assert not np.allclose(np.sort(hl[:, 0]),
                           np.sort(np.asarray(field.x_axis)[ext0["x"][on_line]]))
    dot = np.array(p.renderer.actors["layer-0-extrema"].mapper.dataset.points)
    assert dot[0, 0] == pytest.approx(axis_at(field.x_axis, [ext0["x_sub"][5]])[0])
    assert dot[0, 1] == pytest.approx(axis_at(field.y_axis, [ext0["y_sub"][5]])[0])


def test_without_refinement_the_view_is_unchanged():
    p = _plotter()
    scene = Scene(p)
    field = _frame_field()
    res = _hline_result()
    scene.set_frame_mode(True)
    scene.set_layers([{"layer": Layer(layer_id=0, name="l", source_id="s0"), "field": field,
                       "result": res, "status": "ok"}])
    hl = np.array(p.renderer.actors["layer-0-hlines"].mapper.dataset.points)
    ext0 = res["extrema"][0]
    on_line = np.flatnonzero(ext0["line_id"] >= 0)
    assert np.allclose(np.sort(hl[:, 0]), np.sort(np.asarray(field.x_axis)[ext0["x"][on_line]]))
