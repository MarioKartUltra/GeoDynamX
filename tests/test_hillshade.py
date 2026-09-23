# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""dynamix.core.hillshade — shaded relief of a field.

Horn's gradient (the standard 3x3 kernel) with the field's own pixel spacing, illuminated from
(azimuth, altitude); output in [0, 1], NaN where the input is NaN. Headless, numpy only."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.hillshade import hillshade


def _plane(nx=32, ny=32, gx=0.0, gy=0.0, dx=1.0, dy=1.0):
    y, x = np.mgrid[0:ny, 0:nx]
    return gx * x * dx + gy * y * dy


def test_flat_ground_is_lit_by_the_sun_altitude_alone():
    z = _plane()
    s = hillshade(z, dx=1.0, dy=1.0, azimuth=315.0, altitude=45.0)
    assert s.shape == z.shape
    assert np.allclose(s[2:-2, 2:-2], np.sin(np.radians(45.0)), atol=1e-6)   # cos(zenith)
    assert np.allclose(hillshade(z, 1.0, 1.0, azimuth=315.0, altitude=90.0)[2:-2, 2:-2], 1.0)


def test_a_slope_facing_the_sun_is_brighter_than_one_facing_away():
    # sun from the north-west (315°). Rows increase SOUTHWARD (north-up array), so a slope that
    # FACES north-west descends toward west and north: z grows eastward and southward.
    z_toward = _plane(gx=+0.5, gy=+0.5)
    z_away = _plane(gx=-0.5, gy=-0.5)
    s_t = hillshade(z_toward, 1.0, 1.0, azimuth=315.0, altitude=45.0)[4:-4, 4:-4]
    s_a = hillshade(z_away, 1.0, 1.0, azimuth=315.0, altitude=45.0)[4:-4, 4:-4]
    assert s_t.mean() > 0.9 and s_a.mean() < 0.2
    assert 0.0 <= s_t.min() and s_t.max() <= 1.0


def test_pixel_spacing_and_z_factor_scale_the_slope():
    z = _plane(gx=1.0)
    steep = hillshade(z, dx=1.0, dy=1.0, azimuth=270.0, altitude=45.0)[4:-4, 4:-4].mean()
    gentle = hillshade(z, dx=10.0, dy=10.0, azimuth=270.0, altitude=45.0)[4:-4, 4:-4].mean()
    exaggerated = hillshade(z, dx=10.0, dy=10.0, azimuth=270.0, altitude=45.0, z_factor=10.0)[4:-4, 4:-4].mean()
    assert gentle != steep and np.isclose(exaggerated, steep, atol=1e-6)


def test_nan_cells_stay_nan_and_do_not_poison_neighbours():
    z = _plane(gx=0.2)
    z[10, 10] = np.nan
    s = hillshade(z, 1.0, 1.0)
    assert np.isnan(s[10, 10])
    assert np.isfinite(s[8, 8]) and np.isfinite(s[12, 12])


def test_multi_component_fields_are_refused_with_a_message():
    with pytest.raises(ValueError, match="2-D"):
        hillshade(np.zeros((4, 4, 3)), 1.0, 1.0)
