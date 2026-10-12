# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Where a field's image sits on the map; display reprojection; the frame-family rule."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import GeographicFrame, LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.geo.placement import (field_crs, image_transform, reproject_for_display,
                                   same_frame_family)


def _utm(ny=40, nx=50, x0=500005.0, y0=4399995.0, d=10.0):
    f = RasterField(name="f", values=np.arange(ny * nx, dtype=float).reshape(ny, nx),
                    frame=LocalFrame(x0=x0, y0=y0, dx=d, dy=d, units="metre"),
                    x_axis=x0 + d * np.arange(nx), y_axis=y0 - d * np.arange(ny))
    f.provenance["crs"] = "EPSG:32613"
    return f


def test_the_image_corner_is_half_a_pixel_out_from_the_first_centre():
    assert image_transform(_utm()) == pytest.approx((500000.0, 4400000.0, 10.0, -10.0))
    assert image_transform(_utm(), s=4)[2:] == pytest.approx((40.0, -40.0))
    assert field_crs(_utm()) == "EPSG:32613"


def test_reprojection_lands_inside_the_source_footprint():
    img, (x0, y0, sx, sy) = reproject_for_display(_utm().values, _utm(), "EPSG:4326",
                                                  max_dim=64)
    assert max(img.shape) <= 64 and np.isfinite(img).any()
    assert -105.1 < x0 < -104.9 and 39.7 < y0 < 39.8 and sx > 0 > sy


def test_frame_family_ignores_the_anchor_but_not_the_pixel_size():
    a = LocalFrame(x0=0.0, y0=0.0, dx=10.0, dy=10.0, units="metre")
    assert same_frame_family(a, LocalFrame(x0=500.0, y0=-90.0, dx=10.0, dy=10.0, units="metre"))
    assert not same_frame_family(a, LocalFrame(x0=0.0, y0=0.0, dx=5.0, dy=10.0, units="metre"))
    assert not same_frame_family(a, LocalFrame(x0=0.0, y0=0.0, dx=10.0, dy=10.0, units="px"))
    assert not same_frame_family(a, GeographicFrame())
