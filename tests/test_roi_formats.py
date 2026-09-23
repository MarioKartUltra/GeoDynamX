# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Every format honors the open threshold: a netCDF/HDF5 grid or an HDF4 SDS over ``open_max_pixels`` opens as the
display picture, and the ROI runner reads native pixels from the RIGHT grid -- never the
container's band 1."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def big_nc(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    p = tmp_path / "big.nc"
    ds = netCDF4.Dataset(p, "w")
    ds.createDimension("y", 40)
    ds.createDimension("x", 50)
    for name, scale in (("temp", 1.0), ("salt", 10.0)):
        v = ds.createVariable(name, "f8", ("y", "x"), fill_value=-999.0)
        v[:] = np.arange(2000.0).reshape(40, 50) * scale
    ds.close()
    return p


@pytest.fixture
def big_hdf4(tmp_path):
    pytest.importorskip("pyhdf")
    from pyhdf.SD import SD, SDC

    p = tmp_path / "big.hdf"
    try:
        sd = SD(str(p), SDC.WRITE | SDC.CREATE)
        sds = sd.create("band", SDC.FLOAT64, (40, 50))
        sds[:] = np.arange(2000.0).reshape(40, 50)
        sds.setfillvalue(-999.0)
        sds.endaccess()
        sd.end()
    except Exception as exc:                              # pragma: no cover - env guard
        pytest.skip(f"pyhdf cannot create HDF4 here: {exc}")
    return p


def _salt_target(path):
    from dynamix.core import ingest

    return next(s for s, _d in ingest.probe(path)["subdatasets"] if s.endswith("salt"))


def test_a_picture_of_a_non_georeferenced_grid_gets_pixel_axes_and_a_local_frame(big_nc):
    from dynamix.core.frames import LocalFrame
    from dynamix.roi.picture import read_picture

    pic = read_picture(_salt_target(big_nc), max_dim=10)
    assert isinstance(pic.frame, LocalFrame)


def test_a_big_netcdf_grid_opens_as_the_picture_of_the_chosen_variable(big_nc):
    from dynamix.shell.opening import open_field

    target = _salt_target(big_nc)
    f = open_field(str(big_nc), max_pixels=100, subdataset=target)
    assert "display_stride" in f.provenance and "overview" not in f.provenance
    assert f.provenance["source"] == target               # the grid, not the container
    assert tuple(f.provenance["full_dims"]) == (40, 50)


def test_the_runner_reads_native_pixels_of_the_chosen_variable(big_nc):
    """salt = 10 x temp: reading the container's band 1 would silently give temp."""
    from dynamix.roi.picture import read_picture
    from dynamix.roi.runner import read_processing_window

    import rasterio

    target = _salt_target(big_nc)
    with rasterio.open(target) as src:
        native = src.read(1).astype(float)
    win, _info = read_processing_window(read_picture(target, max_dim=10), (10, 12, 8, 9), 2)
    np.testing.assert_array_equal(win.values, native[8:20, 10:23])


def test_a_big_hdf4_sds_opens_as_a_picture_and_the_runner_reads_it(big_hdf4):
    from dynamix.roi.runner import read_processing_window
    from dynamix.shell.opening import open_field

    f = open_field(str(big_hdf4), max_pixels=100, subdataset="band")
    assert "display_stride" in f.provenance
    assert tuple(f.provenance["full_dims"]) == (40, 50)
    native = np.arange(2000.0).reshape(40, 50)
    win, _info = read_processing_window(f, (10, 12, 8, 9), 2)
    np.testing.assert_array_equal(win.values, native[8:20, 10:23])


def test_the_runner_reads_an_hdf4_picture_window_through_pyhdf(big_hdf4):
    """A stride > 1 picture forces the FILE path: native pixels come back through pyhdf."""
    from dynamix.roi.picture import read_picture_hdf4
    from dynamix.roi.runner import read_processing_window

    pic = read_picture_hdf4(big_hdf4, "band", max_dim=10)
    assert pic.provenance["display_stride"] == 5 and pic.provenance["reader"] == "hdf4"
    native = np.arange(2000.0).reshape(40, 50)
    np.testing.assert_array_equal(pic.values, native[np.ix_([2, 7, 12, 17, 22, 27, 32, 37],
                                                            [2, 7, 12, 17, 22, 27, 32, 37,
                                                             42, 47])])
    win, info = read_processing_window(pic, (0, 12, 8, 9), 2)
    np.testing.assert_array_equal(win.values[2:, :], native[0:10, 10:23])
    assert info["reflected_edges"] == ("N",)
    assert win.x_axis[2] == pytest.approx(12.5)        # pixel-index axes, like the loader
