# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.ingest + the opening route -- the 2026-09-21 sensor-format support.

Fixtures are GENERATED (tiny netCDF via netCDF4, HDF5 via h5py, multiband GeoTIFF via
rasterio); HDF4 writing needs pyhdf's SD create path and is exercised only when it works on
this machine (skip otherwise -- reading real ASTER granules is the target, and the loader is
pinned through the GDAL-shaped paths regardless)."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import ingest


@pytest.fixture
def nc_file(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    p = tmp_path / "toy.nc"
    ds = netCDF4.Dataset(p, "w")
    ds.createDimension("y", 6)
    ds.createDimension("x", 5)
    for name, scale in (("temp", 1.0), ("salt", 10.0)):
        v = ds.createVariable(name, "f8", ("y", "x"), fill_value=-999.0)
        v[:] = np.arange(30.0).reshape(6, 5) * scale
    ds.variables["temp"][0, 0] = -999.0
    ds.close()
    return p


@pytest.fixture
def h5_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "toy.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("grid", data=np.arange(20.0).reshape(4, 5))
    return p


@pytest.fixture
def multiband_tif(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    p = tmp_path / "stack.tif"
    data = np.stack([np.full((8, 6), b, dtype=np.float32) for b in range(1, 5)])
    with rasterio.open(p, "w", driver="GTiff", height=8, width=6, count=4,
                       dtype="float32", crs="EPSG:32615",
                       transform=from_origin(500000, 4000000, 30, 30)) as dst:
        dst.write(data)
    return p


def test_probe_lists_netcdf_variables(nc_file):
    info = ingest.probe(nc_file)
    assert info["kind"] == "gdal"
    descs = sorted(d for _s, d in info["subdatasets"])
    assert descs == ["salt", "temp"]


def test_load_netcdf_variable_masks_fill_and_uses_index_axes(nc_file):
    """GDAL reads netCDF BOTTOM-UP (its north-up convention) -- the raw array's row 0 lands
    at the bottom; pinned here so a future orientation change is a decision, not drift."""
    info = ingest.probe(nc_file)
    sid = next(s for s, d in info["subdatasets"] if d == "temp")
    field = ingest.load_grid(nc_file, subdataset=sid)
    assert field.values.shape == (6, 5)
    assert np.isnan(field.values[5, 0])                 # raw [0,0] fill, flipped -> NaN
    np.testing.assert_allclose(field.values[4, 0], 5.0)
    assert field.provenance["subdataset"] == sid
    assert field.name.endswith("temp")


def test_load_hdf5_dataset(h5_file):
    """A single-dataset HDF5 opens DIRECTLY (GDAL exposes no subdatasets for it)."""
    info = ingest.probe(h5_file)
    assert info["subdatasets"] == [] and info["count"] >= 1
    field = ingest.load_grid(h5_file)
    np.testing.assert_allclose(field.values, np.arange(20.0).reshape(4, 5))


def test_multiband_geotiff_lands_as_a_stack_with_real_georeference(multiband_tif):
    assert ingest.multiband_count(multiband_tif) == 4
    field = ingest.load_grid(multiband_tif)
    assert field.values.shape == (8, 6, 4)
    np.testing.assert_allclose(field.values[..., 2], 3.0)
    # rasterfield's own conventions: pixel-center axes through the transform, projected units
    np.testing.assert_allclose(field.x_axis[0], 500000 + 15.0)
    np.testing.assert_allclose(field.y_axis[0], 4000000 - 15.0)
    assert field.units in ("metre", "meter", "m")


def test_open_field_routes_containers_and_multiband(nc_file, multiband_tif):
    from dynamix.shell.opening import open_field

    with pytest.raises(ValueError, match="pick one"):
        open_field(str(nc_file))                        # two variables, no choice given
    info = ingest.probe(nc_file)
    sid = next(s for s, d in info["subdatasets"] if d == "salt")
    field = open_field(str(nc_file), subdataset=sid)
    np.testing.assert_allclose(field.values[4, 0], 50.0)   # GDAL bottom-up (pinned above)
    assert field.provenance["full_dims"] == (6, 5)      # the _stamp_source contract

    stack = open_field(str(multiband_tif))
    assert stack.values.shape == (8, 6, 4)
    assert stack.provenance["source"] == str(multiband_tif)


def test_single_variable_container_needs_no_ceremony(h5_file):
    from dynamix.shell.opening import open_field

    field = open_field(str(h5_file))
    assert field.values.shape == (4, 5)
