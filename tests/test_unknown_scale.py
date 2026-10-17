# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A raster whose scale is unknown reads in pixels.

A GeoTIFF with no CRS has no known ground scale, whatever its transform says, so every reader
frames it in pixels (pixel-centre axes, ``LocalFrame`` in ``px``); a geographic CRS still reads in
degrees and a projected one in its linear unit.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from dynamix.core.frames import GeographicFrame, LocalFrame  # noqa: E402


def _tif(path, *, crs=None, transform=True, n=40):
    from rasterio.transform import from_origin

    v = np.random.default_rng(0).standard_normal((n, n)).astype("float32")
    kw = {"transform": from_origin(1000.0, 2000.0, 10.0, 10.0)} if transform else {}
    if crs:
        kw["crs"] = crs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # rasterio's NotGeoreferencedWarning
        with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1,
                           dtype="float32", **kw) as ds:
            ds.write(v, 1)
    return path


def _readers():
    from dynamix.core.ingest import load_grid
    from dynamix.core.rasterfield import RasterField
    from dynamix.shell.opening import open_field

    return {"open_field": lambda p: open_field(p),
            "from_file": lambda p: RasterField.from_file(p),
            "window": lambda p: RasterField.from_geotiff_window(p, row_off=4, col_off=6,
                                                                height=20, width=24),
            "load_grid": lambda p: load_grid(p),
            "overview": lambda p: open_field(p, max_pixels=400, mode="overview"),
            "picture": lambda p: open_field(p, max_pixels=400)}


@pytest.mark.parametrize("transform", [True, False], ids=["transform", "no-transform"])
@pytest.mark.parametrize("reader", list(_readers()))
def test_a_geotiff_without_a_crs_reads_in_pixels(tmp_path, reader, transform):
    field = _readers()[reader](_tif(tmp_path / "f.tif", transform=transform))
    assert isinstance(field.frame, LocalFrame) and field.frame.units == "px", reader
    step = field.x_axis[1] - field.x_axis[0]
    assert step == pytest.approx(field.frame.dx), reader
    assert (field.frame.x0, field.frame.y0) == pytest.approx(
        (field.x_axis[0], field.y_axis[0])), reader
    assert field.x_axis[-1] < 40, "native pixel indices, not the transform's coordinates"
    first = {"window": (6.5, 4.5), "overview": (0.5 * step, 0.5 * step)}.get(reader)
    if reader != "picture":
        assert (field.x_axis[0], field.y_axis[0]) == pytest.approx(first or (0.5, 0.5)), reader


@pytest.mark.parametrize("reader", list(_readers()))
def test_a_geographic_crs_still_reads_in_degrees_and_a_projected_one_in_metres(tmp_path, reader):
    geo = _readers()[reader](_tif(tmp_path / "g.tif", crs="EPSG:4326"))
    assert isinstance(geo.frame, GeographicFrame), reader
    utm = _readers()[reader](_tif(tmp_path / "u.tif", crs="EPSG:32611"))
    assert isinstance(utm.frame, LocalFrame) and utm.frame.units == "metre", reader
