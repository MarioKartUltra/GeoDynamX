# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Orientations as (angle, frame), and the WTMMM anisotropy statistics built on them."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.anisotropy import angle_pdf, gradient_plane, sector_modulus_pdfs
from dynamix.core.orientation import (Frame, Orientation, axial_sector, from_arg,
                                      grid_to_true, to_azimuth)


def test_a_pixel_angle_becomes_an_azimuth_from_the_axis_directions():
    o = from_arg(np.radians([0.0, 90.0, -90.0, 180.0]))            # right, down, up, left
    north_up = to_azimuth(o, col_east=True, row_north=False, north=Frame.GRID)
    assert north_up.frame is Frame.GRID
    np.testing.assert_allclose(north_up.degrees, [90.0, 180.0, 0.0, 270.0], atol=1e-9)
    south_first = to_azimuth(o, col_east=True, row_north=True, north=Frame.TRUE)
    np.testing.assert_allclose(south_first.degrees, [90.0, 0.0, 180.0, 270.0], atol=1e-9)


def test_frames_are_never_mixed_silently():
    with pytest.raises(ValueError):
        to_azimuth(Orientation(10.0, Frame.GRID), col_east=True, row_north=False,
                   north=Frame.GRID)
    with pytest.raises(ValueError):
        grid_to_true(Orientation(10.0, Frame.PIXEL), 1.0)
    t = grid_to_true(Orientation(np.array([359.5, 10.0]), Frame.GRID), 1.0)
    assert t.frame is Frame.TRUE and np.allclose(t.degrees, [0.5, 11.0])


def test_axial_sectors_are_arneodos_four_orientation_bins():
    o = Orientation(np.array([0.0, 180.0, 44.0, 90.0, 135.0, 170.0, -45.0]), Frame.PIXEL)
    assert axial_sector(o).tolist() == [0, 0, 1, 2, 3, 0, 3]


def test_angle_pdf_is_flat_for_isotropic_angles_and_integrates_to_one():
    o = Orientation(np.linspace(-180.0, 180.0, 36000, endpoint=False), Frame.PIXEL)
    centres, pdf = angle_pdf(o, bins=36)
    assert np.allclose(pdf, 1.0 / 360.0, rtol=0.02)
    assert np.sum(pdf) * 10.0 == pytest.approx(1.0)
    assert centres[0] == pytest.approx(-175.0)


def test_gradient_plane_and_sector_pdfs():
    t1, t2 = gradient_plane([2.0, 3.0], np.radians([0.0, 90.0]))
    assert np.allclose(t1, [2.0, 0.0]) and np.allclose(t2, [0.0, 3.0])
    rng = np.random.default_rng(0)
    a = rng.uniform(-180, 180, 20000)
    m = np.where(np.abs(np.mod(a + 22.5, 180.0) - 22.5) < 22.5, 8.0, 1.0)   # sector 0 strong
    pdfs = sector_modulus_pdfs(m, Orientation(a, Frame.PIXEL))
    assert [p[0] for p in pdfs] == [0, 1, 2, 3] and sum(p[3] for p in pdfs) == 20000
    peak = [p[1][np.argmax(p[2])] for p in pdfs]
    assert peak[0] > 2.5 and all(pk < 0.5 for pk in peak[1:])   # log2 8 = 3 vs log2 1 = 0


def test_grid_north_bearing_is_the_meridian_convergence():
    rasterio = pytest.importorskip("rasterio")
    from rasterio.warp import transform

    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.geo.mapping import field_north, grid_north_bearing

    (x0,), (y0,) = transform("EPSG:4326", "EPSG:32611", [-115.0], [40.0])
    f = RasterField(name="utm", values=np.zeros((8, 8)),
                    frame=LocalFrame(x0=x0, y0=y0, dx=30.0, dy=30.0, units="metre"),
                    x_axis=x0 + 30.0 * np.arange(8), y_axis=y0 - 30.0 * np.arange(8),
                    provenance={"crs": "EPSG:32611"})
    assert field_north(f) == (True, False, Frame.GRID)
    gamma = grid_north_bearing(f, [0.0], [0.0])[0]
    expected = np.degrees(np.arctan(np.tan(np.radians(2.0)) * np.sin(np.radians(40.0))))
    assert gamma == pytest.approx(expected, abs=0.02)             # ~ +1.29 deg: east of true


def test_a_striped_field_is_anisotropic_and_points_across_the_stripes(clean_registry):
    from dynamix.core.anisotropy import wtmmm_by_scale
    from dynamix.core.rasterfield import RasterField
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device

    register_builtin_devices()
    rng = np.random.default_rng(1)
    stripes = np.repeat(rng.normal(size=(1, 96)).cumsum(1), 96, axis=0)   # varies along x only
    from dynamix.core.frames import LocalFrame

    f = RasterField(name="s", values=stripes + 1e-3 * rng.normal(size=(96, 96)),
                    frame=LocalFrame(), x_axis=np.arange(96.0), y_axis=np.arange(96.0))
    dev = get_device("wtmm2d")
    res = dev.compute(f, {**defaults_for(dev), "n_oct": 3})
    w = wtmmm_by_scale(res["extrema"])[1]
    o = from_arg(w["arg"])
    _c, pdf = angle_pdf(o, bins=8)
    across = np.isin(axial_sector(o), [0]).mean()                # arg ~ 0 / 180: along x
    # bins centred -157.5 ... 157.5: nothing near ±90 (along the stripes), all near 0 / 180
    assert pdf[[1, 2, 5, 6]].sum() == 0.0 and pdf.max() > 1.5 * (1.0 / 360.0)
    assert across > 0.8
