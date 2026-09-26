# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Fractional pixel positions (a maximum's subpixel refinement) placed along the axes."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import GeographicFrame
from dynamix.core.rasterfield import RasterField
from dynamix.geo.mapping import axis_at, points_lonlat


def test_axis_at_is_linear_between_pixel_centers_and_exact_on_them():
    axis = 500_000.0 + 30.0 * np.arange(10) + 15.0
    assert axis_at(axis, [0, 3, 9]).tolist() == axis[[0, 3, 9]].tolist()
    assert axis_at(axis, [2.5])[0] == pytest.approx(axis[2] + 15.0)
    assert axis_at(axis, [-0.5])[0] == pytest.approx(axis[0] - 15.0)     # half a pixel out


def test_points_lonlat_places_subpixel_positions_between_centers():
    pytest.importorskip("rasterio")
    f = RasterField(name="g", values=np.zeros((5, 6)), frame=GeographicFrame(),
                    x_axis=-120.0 + 0.01 * np.arange(6), y_axis=45.0 - 0.01 * np.arange(5),
                    provenance={"crs": "EPSG:4326"})
    lon_i, lat_i = points_lonlat(f, [2], [1])
    lon_s, lat_s = points_lonlat(f, [2], [1], sub=([2.5], [1.25]))
    assert lon_s[0] == pytest.approx(lon_i[0] + 0.005)
    assert lat_s[0] == pytest.approx(lat_i[0] - 0.0025)
