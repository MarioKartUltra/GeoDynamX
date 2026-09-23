# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The display-only whole-extent picture.

A raster too big to load opens as a PICTURE: exact block-centre samples of the file, drawn
over exactly their native blocks, never analysed. These pin the sampling rule and the
georeference -- the registration the user lost a day to when the old overview was a second
coordinate system.
"""
from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from dynamix.core.frames import LocalFrame  # noqa: E402
from dynamix.core.rasterfield import RasterField  # noqa: E402
from dynamix.roi.picture import (block_centres, display_stride, native_shape,  # noqa: E402
                                 picture_stride, read_picture)


def _tif(path, h=50, w=70, nodata=None):
    rows, cols = np.mgrid[0:h, 0:w]
    vals = (rows * 1000 + cols).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs="EPSG:32615", nodata=nodata,
                       transform=from_origin(500000.0, 3200000.0, 2.0, 2.0)) as dst:
        dst.write(vals, 1)
    return vals


def test_picture_stride_is_ceil_of_long_side_over_budget():
    assert picture_stride(50, 70, max_dim=16) == 5
    assert picture_stride(10, 10, max_dim=16) == 1


def test_block_centres_clamp_the_partial_last_block():
    assert block_centres(70, 5).tolist() == [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67]
    assert block_centres(69, 5)[-1] == 67          # last block [65, 69) -> centre 67, in range
    assert block_centres(66, 5)[-1] == 65          # last block [65, 66) -> centre 67 clamps to 65


def test_picture_values_are_the_exact_block_centre_pixels(tmp_path):
    vals = _tif(tmp_path / "g.tif")
    pic = read_picture(tmp_path / "g.tif", max_dim=16)
    assert pic.values.shape == (10, 14)
    rr, cc = block_centres(50, 5), block_centres(70, 5)
    assert np.array_equal(pic.values, vals[np.ix_(rr, cc)].astype(np.float64))


def test_picture_axes_are_the_file_transform_at_those_pixel_centres(tmp_path):
    _tif(tmp_path / "g.tif")
    pic = read_picture(tmp_path / "g.tif", max_dim=16)
    cc, rr = block_centres(70, 5), block_centres(50, 5)
    assert np.allclose(pic.x_axis, 500000.0 + 2.0 * (cc + 0.5))
    assert np.allclose(pic.y_axis, 3200000.0 - 2.0 * (rr + 0.5))
    # the frame carries the NATIVE pixel size: px_to_metres reads file pixels
    assert isinstance(pic.frame, LocalFrame) and pic.frame.dx == 2.0 and pic.frame.dy == 2.0


def test_picture_nodata_and_float32_sentinel_become_nan(tmp_path):
    path = tmp_path / "n.tif"
    vals = np.full((20, 20), -25.0, dtype=np.float32)
    vals[:5, :] = np.float32(-3.4028235e38)          # BOEM's undeclared fill
    vals[10:, :] = 0.0                                # the DECLARED nodata
    with rasterio.open(path, "w", driver="GTiff", height=20, width=20, count=1, dtype="float32",
                       crs="EPSG:32615", nodata=0.0,
                       transform=from_origin(500000.0, 3200000.0, 2.0, 2.0)) as dst:
        dst.write(vals, 1)
    pic = read_picture(path, max_dim=10)
    assert np.isnan(pic.values[:2]).all()             # sentinel rows
    assert np.isnan(pic.values[5:]).all()             # declared nodata rows
    assert (pic.values[2:5] == -25.0).all()


def test_picture_provenance_names_the_file_and_never_the_overview_key(tmp_path):
    _tif(tmp_path / "g.tif")
    pic = read_picture(tmp_path / "g.tif", max_dim=16)
    p = pic.provenance
    assert p["source"] == str(tmp_path / "g.tif")
    assert tuple(p["full_dims"]) == (50, 70)
    assert p["window"] == {"row_off": 0, "col_off": 0}
    assert p["display_stride"] == 5
    assert "overview" not in p                        # the old x-ov paths stay inert
    assert display_stride(pic) == 5


def test_native_shape_is_the_file_for_a_picture_and_the_array_otherwise(tmp_path):
    _tif(tmp_path / "g.tif")
    assert native_shape(read_picture(tmp_path / "g.tif", max_dim=16)) == (50, 70)
    plain = RasterField(name="p", values=np.zeros((7, 9)), frame=LocalFrame(),
                        x_axis=np.arange(9.0), y_axis=np.arange(7.0))
    assert native_shape(plain) == (7, 9) and display_stride(plain) == 1
    assert display_stride(None) == 1


def test_sample_spacing_is_native_pixel_size_times_every_stride(tmp_path):
    """Hillshade needs the ground distance between the samples it sees: native pixel size x
    the picture stride x the view's own decimation -- one helper for the canvas and the scene."""
    from dynamix.roi.picture import sample_spacing

    _tif(tmp_path / "g.tif")
    pic = read_picture(tmp_path / "g.tif", max_dim=16)            # 2 m pixels, stride 5
    assert sample_spacing(pic) == (10.0, 10.0)
    assert sample_spacing(pic, 3) == (30.0, 30.0)
    plain = RasterField(name="p", values=np.zeros((4, 4)), frame=LocalFrame(dx=2.0, dy=3.0),
                        x_axis=np.arange(4.0), y_axis=np.arange(4.0))
    assert sample_spacing(plain, 2) == (4.0, 6.0)


# ------------------------------------------- Vectors on a picture (file pixels)

def _stride5_picture(tmp_path, h=48, w=69):
    """A stride-5 picture whose file dims are NOT a multiple of the stride (partial blocks)."""
    _tif(tmp_path / "v.tif", h=h, w=w)
    return read_picture(tmp_path / "v.tif", max_dim=16)


def _world_of(col, row):
    return 500000.0 + 2.0 * (col + 0.5), 3200000.0 - 2.0 * (row + 0.5)


def test_reference_vectors_land_on_file_pixels_over_a_picture(tmp_path):
    """Final review: to_field_pixels inverted the PICTURE's sample axes, so every
    reference polygon on a picture was drawn ~s x too close to the origin on a file-pixel
    canvas -- the very second-coordinate-system class this run removes."""
    from dynamix.geo.vectors import Feature, VectorLayer, to_field_pixels

    pic = _stride5_picture(tmp_path)
    x, y = _world_of(60, 40)
    layer = VectorLayer(name="seeps", kind="point", crs="EPSG:32615",
                        features=[Feature(parts=[np.array([[x, y]])])], bounds=(x, y, x, y))
    col, row = to_field_pixels(layer, pic).features[0].parts[0][0]
    assert (col, row) == pytest.approx((60.0, 40.0))


def test_lonlat_to_pixels_answers_in_file_pixels_over_a_picture(tmp_path):
    from rasterio.warp import transform as warp

    from dynamix.geo.mapping import lonlat_to_pixels

    pic = _stride5_picture(tmp_path)
    x, y = _world_of(60, 40)
    lon, lat = warp("EPSG:32615", "EPSG:4326", [x], [y])
    cols, rows = lonlat_to_pixels(pic, np.array(lon), np.array(lat))
    assert (cols[0], rows[0]) == pytest.approx((60.0, 40.0), abs=1e-6)


def test_file_pixel_grid_of_a_picture_and_none_for_an_ordinary_field(tmp_path):
    from dynamix.roi.picture import file_pixel_grid

    x0, dx, y0, dy, nx, ny = file_pixel_grid(_stride5_picture(tmp_path))
    assert (x0, dx, y0, dy) == pytest.approx((500001.0, 2.0, 3199999.0, -2.0))
    assert (nx, ny) == (69, 48)
    plain = RasterField(name="p", values=np.zeros((4, 4)), frame=LocalFrame(),
                        x_axis=np.arange(4.0), y_axis=np.arange(4.0))
    assert file_pixel_grid(plain) is None
