# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""xsmurf-style extrema geometry: unconnected blocks, argument ticks through the grid's axes."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.geo.extrema_draw import (block_positions, group_rgb, modulus_scalars, tick_ends,
                                      to_axes)


def test_blocks_sit_on_their_pixel_or_their_subpixel_crossing():
    ext = {"x": np.array([3, 4]), "y": np.array([5, 5]),
           "x_sub": np.array([3.25, 4.5]), "y_sub": np.array([5.0, 4.75])}
    assert np.allclose(block_positions(ext, subpixel=False)[0], [3, 4])
    cols, rows = block_positions(ext, subpixel=True)
    assert np.allclose(cols, [3.25, 4.5]) and np.allclose(rows, [5.0, 4.75])
    assert np.allclose(block_positions({"x": [1], "y": [2]}, subpixel=True)[0], [1])


def test_ticks_are_xsmurf_length_along_the_argument():
    c1, r1 = tick_ends(np.array([0.0, 0.0]), np.array([0.0, 0.0]),
                       np.array([0.0, np.pi / 2]), np.array([1.0, 0.5]), 1.0)
    assert np.allclose(c1, [33.0, 0.0]) and np.allclose(r1, [0.0, 18.0])


def test_a_tick_through_north_up_axes_points_where_the_gradient_does():
    # pixel frame: +row is DOWN the image; on a north-up grid that is SOUTH (y decreases)
    x_axis = 500000.0 + 10.0 * np.arange(50)
    y_axis = 4400000.0 - 10.0 * np.arange(40)
    c1, r1 = tick_ends(np.array([10.0]), np.array([10.0]), np.array([np.pi / 2]),
                       np.array([0.0]), 1.0)
    x0, y0 = to_axes(np.array([10.0]), np.array([10.0]), x_axis, y_axis)
    x1, y1 = to_axes(c1, r1, x_axis, y_axis)
    assert x1[0] == pytest.approx(x0[0]) and y1[0] == pytest.approx(y0[0] - 30.0)


def test_modulus_scalars_modes():
    m = np.array([1.0, 10.0, 100.0])
    assert np.allclose(modulus_scalars(m, "linear"), [0.01, 0.1, 1.0])
    assert np.allclose(modulus_scalars(m, "log"), [0.0, 0.5, 1.0])
    assert np.allclose(modulus_scalars(m, "q", q=2.0), [1e-4, 1e-2, 1.0])


def test_group_colour_reads_the_chain_point_at_the_scale():
    chains = [{"x": np.array([3, 4, 9]), "y": np.array([5, 5, 9]), "tags": ["group:g"],
               "group_color": (255, 0, 0)},
              {"x": np.array([7]), "y": np.array([7]), "tags": []}]
    rgb = group_rgb(chains, 1, np.array([4.0, 7.0]), np.array([5.0, 7.0]))
    assert rgb[0].tolist() == [255, 0, 0] and rgb[1].tolist() == [-1, -1, -1]
