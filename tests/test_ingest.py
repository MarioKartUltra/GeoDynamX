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


# ------------------------------------------- HDF4 name collisions (the ASTER layout)

def _aster_like(tmp_path):
    """Two HDF-EOS-style swaths whose image SDS share one NAME -- the ASTER layout: every
    band's array is called ImageData, distinguished only by its swath Vgroup."""
    from pyhdf import V  # noqa: F401 -- vgstart() needs the submodule linked
    from pyhdf.HDF import HC, HDF
    from pyhdf.SD import SD, SDC

    p = tmp_path / "aster_like.hdf"
    sd = SD(str(p), SDC.WRITE | SDC.CREATE)
    refs = {}
    for swath, shape, base in (("VNIR_Band1", (12, 10), 100.0),
                               ("VNIR_Band2", (12, 10), 500.0),
                               ("SWIR_Band4", (6, 8), 900.0)):
        sds = sd.create("ImageData", SDC.FLOAT64, shape)
        sds[:] = base + np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        refs[swath] = sds.ref()
        sds.endaccess()
    lone = sd.create("Cloud_Table", SDC.FLOAT64, (4, 4))
    lone[:] = np.zeros((4, 4))
    lone.endaccess()
    sd.end()
    h = HDF(str(p), HC.WRITE)
    v = h.vgstart()
    for swath, ref in refs.items():
        outer = v.attach(-1, 1)
        outer._name, outer._class = swath, "SWATH"
        inner = v.attach(-1, 1)
        inner._name, inner._class = "Data Fields", "SWATH Vgroup"
        inner.add(getattr(HC, "DFTAG_NDG", 720), ref)
        outer.insert(inner)
        inner.detach()
        outer.detach()
    v.end()
    h.close()
    return p


@pytest.fixture
def aster_like_hdf4(tmp_path):
    pytest.importorskip("pyhdf")
    try:
        return _aster_like(tmp_path)
    except Exception as exc:                              # pragma: no cover - env guard
        pytest.skip(f"pyhdf cannot create HDF4 groups here: {exc}")


def test_hdf4_same_named_bands_probe_as_swath_qualified_grids(aster_like_hdf4):
    from dynamix.core import ingest

    subs = ingest.probe(aster_like_hdf4)["subdatasets"]
    ids = [s for s, _d in subs]
    assert "VNIR_Band1/ImageData" in ids and "SWIR_Band4/ImageData" in ids
    assert ids.index("VNIR_Band1/ImageData") < ids.index("Cloud_Table")  # images list first
    desc = dict(subs)
    assert "(12, 10)" in desc["VNIR_Band1/ImageData"]


def test_hdf4_qualified_id_loads_that_swaths_band(aster_like_hdf4):
    from dynamix.core import ingest

    f = ingest.load_grid(aster_like_hdf4, subdataset="SWIR_Band4/ImageData")
    assert f.values.shape == (6, 8) and f.values[0, 0] == 900.0
    assert f.provenance["subdataset"] == "SWIR_Band4/ImageData"


def test_hdf4_bare_name_still_selects_the_first_match(aster_like_hdf4):
    from dynamix.core import ingest

    f = ingest.load_grid(aster_like_hdf4, subdataset="ImageData")
    assert f.values.shape == (12, 10)                     # how it always resolved


def test_hdf4_picture_and_window_read_follow_the_qualified_id(aster_like_hdf4):
    from dynamix.roi.picture import read_picture_hdf4
    from dynamix.roi.runner import _read_hdf4

    pic = read_picture_hdf4(aster_like_hdf4, "SWIR_Band4/ImageData", max_dim=4)
    assert pic.provenance["full_dims"] == (6, 8)
    vals, _t = _read_hdf4(aster_like_hdf4, "SWIR_Band4/ImageData", 0, 0, 2, 3)
    assert vals[0, 0] == 900.0 and vals.shape == (2, 3)


def test_probe_reports_hdf4_dims_per_id(aster_like_hdf4):
    from dynamix.core import ingest

    info = ingest.probe(aster_like_hdf4)
    assert info["dims"]["VNIR_Band1/ImageData"] == (12, 10)
    assert info["dims"]["SWIR_Band4/ImageData"] == (6, 8)


def test_same_grid_bands_stack_in_the_given_order(aster_like_hdf4):
    from dynamix.core import ingest

    f = ingest.load_grid_stack(aster_like_hdf4,
                               ["VNIR_Band2/ImageData", "VNIR_Band1/ImageData"],
                               name="vnir")
    assert f.values.shape == (12, 10, 2)
    assert f.values[0, 0, 0] == 500.0 and f.values[0, 0, 1] == 100.0
    assert f.provenance["bands"] == ["VNIR_Band2/ImageData", "VNIR_Band1/ImageData"]
    assert f.provenance["subdataset"] is None


def test_bands_on_different_grids_refuse_to_stack(aster_like_hdf4):
    from dynamix.core import ingest

    with pytest.raises(ValueError, match="different grids"):
        ingest.load_grid_stack(aster_like_hdf4,
                               ["VNIR_Band1/ImageData", "SWIR_Band4/ImageData"])


def test_a_single_id_stack_is_a_plain_load(aster_like_hdf4):
    from dynamix.core import ingest

    f = ingest.load_grid_stack(aster_like_hdf4, ["SWIR_Band4/ImageData"], name="one")
    assert f.values.shape == (6, 8) and f.name == "one"


def test_band_index_tokens_reopen_a_subset_of_a_multiband_file(tmp_path):
    from dynamix.core import ingest
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    v = np.stack([np.full((4, 5), float(k)) for k in range(3)], axis=-1)
    p = tmp_path / "stack.npz"
    RasterField(name="stack", values=v, frame=LocalFrame(), x_axis=np.arange(5.0),
                y_axis=np.arange(4.0)).save_npz(p)
    two = ingest.load_grid_stack(p, ["#0", "#2"])
    assert two.values.shape == (4, 5, 2)
    assert two.values[0, 0, 0] == 0.0 and two.values[0, 0, 1] == 2.0
    assert two.provenance["bands"] == ["#0", "#2"]
    one = ingest.load_grid_stack(p, ["#1"])
    assert one.values.shape == (4, 5) and one.values[0, 0] == 1.0
    with pytest.raises(ValueError, match="do not exist"):
        ingest.load_grid_stack(p, ["#7"])
