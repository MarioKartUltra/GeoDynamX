# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""dynamix.geo.footprints -- metadata-only raster footprints for the data browser (2026-08-28).

The user's ASTER GDEM folder: 1x1-degree EPSG:4326 tiles, each with a ``_dem`` and a ``_num``
raster sharing one footprint. Scanning must read headers only (bounds/CRS/shape), never pixels;
the footprint's corners are carried in WGS84 lon/lat so any display mode can place them.
"""
from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from dynamix.geo.footprints import Footprint, footprints_at, scan_footprints  # noqa: E402


def _tile(path, west, north, px, n, crs="EPSG:4326", dtype="int16"):
    transform = from_origin(west, north, px, px)
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype=dtype,
                       crs=CRS.from_string(crs), transform=transform) as dst:
        dst.write(np.zeros((n, n), dtype=dtype), 1)


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "Aster_DEM/batch1").mkdir(parents=True)
    d = tmp_path / "Aster_DEM/batch1"
    _tile(d / "ASTGTMV003_S22E119_dem.tif", 119.0, -21.0, 0.1, 10)          # covers 119..120 E, -22..-21
    _tile(d / "ASTGTMV003_S22E119_num.tif", 119.0, -21.0, 0.1, 10, dtype="uint8")
    _tile(d / "ASTGTMV003_S22E120_dem.tif", 120.0, -21.0, 0.1, 10)          # the eastern neighbour
    (d / "ASTGTMV003_S22E119_dem.tif.aux.xml").write_text("<x/>")
    (d / "README.txt").write_text("not a raster")
    # a projected tile: UTM 50S, 10 km square starting at (500000, 7600000)
    _tile(tmp_path / "utm_tile.tif", 500000.0, 7600000.0, 1000.0, 10, crs="EPSG:32750", dtype="float32")
    return tmp_path


def test_scan_finds_every_raster_recursively_and_ignores_sidecars(folder):
    fps = scan_footprints(folder)
    assert [f.name for f in fps] == ["ASTGTMV003_S22E119_dem", "ASTGTMV003_S22E119_num",
                                     "ASTGTMV003_S22E120_dem", "utm_tile"]
    assert all(isinstance(f, Footprint) for f in fps)
    f = fps[0]
    assert (f.width, f.height, f.count, f.dtype) == (10, 10, 1, "int16")
    assert f.crs == "EPSG:4326"
    assert f.path.endswith("Aster_DEM/batch1/ASTGTMV003_S22E119_dem.tif")


def test_corners_are_wgs84_lon_lat_counter_clockwise_from_south_west(folder):
    fps = {f.name: f for f in scan_footprints(folder)}
    sw, se, ne, nw = fps["ASTGTMV003_S22E119_dem"].corners
    assert sw == pytest.approx((119.0, -22.0)) and se == pytest.approx((120.0, -22.0))
    assert ne == pytest.approx((120.0, -21.0)) and nw == pytest.approx((119.0, -21.0))
    utm = fps["utm_tile"].corners
    lons, lats = zip(*utm)
    assert 116.9 < min(lons) < max(lons) < 117.2 and -21.8 < min(lats) < max(lats) < -21.6


def test_footprints_at_returns_every_raster_under_a_point_stacked_first_by_name(folder):
    fps = scan_footprints(folder)
    hits = footprints_at(fps, 119.5, -21.5)
    assert [h.name for h in hits] == ["ASTGTMV003_S22E119_dem", "ASTGTMV003_S22E119_num"]
    assert footprints_at(fps, 120.5, -21.5)[0].name == "ASTGTMV003_S22E120_dem"
    assert footprints_at(fps, 125.0, -21.5) == []


def test_stacked_variants_share_a_stack_key_and_neighbours_do_not(folder):
    fps = {f.name: f for f in scan_footprints(folder)}
    assert fps["ASTGTMV003_S22E119_dem"].stack_key == fps["ASTGTMV003_S22E119_num"].stack_key
    assert fps["ASTGTMV003_S22E119_dem"].stack_key != fps["ASTGTMV003_S22E120_dem"].stack_key


def test_scan_reads_headers_only(folder, monkeypatch):
    """A scan may open every file but must never call ``read`` -- a spy around ``rasterio.open``
    (its Cython dataset class cannot be patched directly) raises on any pixel read."""
    real_open = rasterio.open

    class _Spy:
        def __init__(self, ds):
            self._ds = ds

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return self._ds.__exit__(*a)

        def read(self, *a, **k):
            raise AssertionError("scan_footprints read pixels")

        def __getattr__(self, name):
            return getattr(self._ds, name)

    monkeypatch.setattr(rasterio, "open", lambda *a, **k: _Spy(real_open(*a, **k)))
    assert len(scan_footprints(folder)) == 4


def test_missing_folder_is_an_honest_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_footprints(tmp_path / "nowhere")


def test_a_raster_without_a_crs_is_skipped_not_fatal(folder):
    with rasterio.open(folder / "nocrs.tif", "w", driver="GTiff", height=4, width=4, count=1,
                       dtype="uint8") as dst:
        dst.write(np.zeros((4, 4), dtype=np.uint8), 1)
    names = [f.name for f in scan_footprints(folder)]
    assert "nocrs" not in names and len(names) == 4


# ------------------------------------------------------------ scene labels (2026-08-28, AST_07XT)

from dynamix.geo.footprints import band_label, scene_label  # noqa: E402


def test_aster_granule_names_parse_to_product_and_acquisition_time():
    names = ["AST_07XT_00411282015022621_20250809164339_SRF_VNIR_B01",
             "AST_07XT_00411282015022621_20250809164339_SRF_VNIR_B03N",
             "AST_07XT_00411282015022621_20250809164339_SRF_VNIR_QA_DataPlane"]
    label, when = scene_label(names)
    assert label == "AST_07XT · 2015-11-28 02:26"
    assert when == "2015-11-28T02:26:21"
    assert scene_label(["AST_L1T_00310162007082843_20150521160929_28422"])[0] == "AST_L1T · 2007-10-16 08:28"


def test_non_aster_names_fall_back_to_the_common_prefix_and_no_time():
    assert scene_label(["ASTGTMV003_S22E119_dem", "ASTGTMV003_S22E119_num"]) == ("ASTGTMV003_S22E119", None)
    assert scene_label(["lonely"]) == ("lonely", None)


def test_band_label_is_the_part_after_the_scenes_common_prefix():
    names = ["AST_07XT_00411282015022621_20250809164339_SRF_VNIR_B01",
             "AST_07XT_00411282015022621_20250809164339_SRF_VNIR_QA_DataPlane2"]
    assert band_label(names[0], names) == "B01"
    assert band_label(names[1], names) == "QA_DataPlane2"
    assert band_label("ASTGTMV003_S22E119_dem", ["ASTGTMV003_S22E119_dem", "ASTGTMV003_S22E119_num"]) == "dem"
    assert band_label("solo", ["solo"]) == "solo"


def test_group_key_joins_a_granules_vnir_and_swir_files_despite_their_different_bounds():
    """One AST_07XT granule: VNIR at 15 m and SWIR at 30 m have bounds a few metres apart
    (582237 vs 582243 E in the user's Pilbara download), so stack_key differs; the granule id
    must still put them in one group. GDEM tiles keep the geometric key."""
    from dynamix.geo.footprints import group_key
    def fp(name, west):
        return Footprint(path=f"/d/{name}.tif", name=name, width=10, height=10, count=1, dtype="int16",
                         crs="EPSG:32750", bounds=(west, 0, west + 1, 1),
                         corners=((west, 0), (west + 1, 0), (west + 1, 1), (west, 1)))
    vnir = fp("AST_07XT_00410302004021233_20250402184734_SRF_VNIR_B01", 119.0000)
    swir = fp("AST_07XT_00410302004021233_20250402184734_SRF_SWIR_B04", 119.0001)
    other = fp("AST_07XT_00411282015022621_20250809164339_SRF_VNIR_B01", 119.0000)
    assert group_key(vnir) == group_key(swir)
    assert group_key(vnir) != group_key(other)
    dem = fp("ASTGTMV003_S22E119_dem", 119.0); num = fp("ASTGTMV003_S22E119_num", 119.0)
    assert group_key(dem) == group_key(num) == dem.stack_key


# ------------------------------------------------------------- true swaths (2026-08-28, AST_07XT)

def _swath_tile(path, n=64, overviews=True, dtype="int16"):
    """A north-up 1x1 degree grid holding a rotated parallelogram of valid data (the ASTER
    shape): row r is valid for columns [n//4 - r//4, 3n//4 - r//4)."""
    vals = np.zeros((n, n), dtype=dtype)
    for r in range(n):
        lo, hi = n // 4 - r // 4, 3 * n // 4 - r // 4
        vals[r, lo:hi] = 100 + r
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype=dtype,
                       crs=CRS.from_epsg(4326), transform=from_origin(119.0, -21.0, 1.0 / n, 1.0 / n),
                       tiled=True, blockxsize=16, blockysize=16) as dst:
        dst.write(vals, 1)
        if overviews:
            dst.build_overviews([2, 4], rasterio.enums.Resampling.nearest)
    return vals


def test_scan_uses_the_valid_swath_as_the_footprint_when_overviews_exist(tmp_path):
    _swath_tile(tmp_path / "AST_07XT_00401022008022006_20250619233734_SRF_VNIR_B01.tif")
    (fp,) = scan_footprints(tmp_path)
    assert fp.bounds == (119.0, -22.0, 120.0, -21.0)                # the box is still the box
    assert fp.swath is True
    sw, se, ne, nw = fp.corners
    px = 1.0 / 64
    # top row (r=0): cols 16..48 -> lon 119.25..119.75 ; bottom row (r=63): cols 1..33
    assert nw[0] == pytest.approx(119.0 + 16 * px, abs=2 * px) and ne[0] == pytest.approx(119.0 + 48 * px, abs=2 * px)
    assert sw[0] == pytest.approx(119.0 + 1 * px, abs=2 * px) and se[0] == pytest.approx(119.0 + 33 * px, abs=2 * px)
    assert nw[1] == pytest.approx(-21.0, abs=2 * px) and sw[1] == pytest.approx(-22.0, abs=2 * px)


def test_scan_keeps_the_bounding_box_when_a_raster_has_no_overviews(tmp_path):
    _swath_tile(tmp_path / "AST_07XT_00401022008022006_20250619233734_SRF_VNIR_B02.tif", overviews=False)
    (fp,) = scan_footprints(tmp_path)
    assert fp.swath is False
    assert fp.corners == ((119.0, -22.0), (120.0, -22.0), (120.0, -21.0), (119.0, -21.0))


def test_a_full_raster_keeps_the_box_even_with_overviews(folder):
    """GDEM-style tiles are valid to the edge: no swath detection, corners stay the box."""
    fps = {f.name: f for f in scan_footprints(folder)}
    f = fps["ASTGTMV003_S22E119_dem"]
    assert f.swath is False and f.corners[0] == (119.0, -22.0)


def test_swath_is_read_once_per_aster_granule(tmp_path, monkeypatch):
    import dynamix.geo.footprints as fpmod
    for band in ("B01", "B02", "B03N"):
        _swath_tile(tmp_path / f"AST_07XT_00401022008022006_20250619233734_SRF_VNIR_{band}.tif")
    calls = []
    real = fpmod._swath_corners
    monkeypatch.setattr(fpmod, "_swath_corners", lambda ds: calls.append(1) or real(ds))
    fps = scan_footprints(tmp_path)
    assert len(fps) == 3 and len(calls) == 1
    assert len({f.corners for f in fps}) == 1


def test_band_sort_key_orders_bands_numerically_then_qa_planes():
    from dynamix.geo.footprints import band_sort_key
    names = ["AST_x_SRF_VNIR_QA_DataPlane", "AST_x_SRF_SWIR_B04", "AST_x_SRF_VNIR_B03N",
             "AST_x_SRF_VNIR_B01", "AST_x_SRF_SWIR_QA_DataPlane2", "AST_x_SRF_SWIR_B09", "ASTGTM_num", "ASTGTM_dem"]
    assert sorted(names, key=band_sort_key) == ["AST_x_SRF_VNIR_B01", "AST_x_SRF_VNIR_B03N", "AST_x_SRF_SWIR_B04",
                                                "AST_x_SRF_SWIR_B09", "AST_x_SRF_SWIR_QA_DataPlane2",
                                                "AST_x_SRF_VNIR_QA_DataPlane", "ASTGTM_dem", "ASTGTM_num"]


def test_scan_ignores_appledouble_sidecars_and_other_dotfiles(folder):
    """An exFAT drive gets a hidden ``._name.tif`` (4 KB of extended attributes) beside every
    file the Finder copies; they are not rasters and doubled the scan's work on the user's
    drive (3 285 real + 3 285 sidecars, 2026-08-28)."""
    d = folder / "Aster_DEM/batch1"
    (d / "._ASTGTMV003_S22E119_dem.tif").write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 4092)
    (d / ".hidden.tif").write_bytes(b"not a raster")
    names = [f.name for f in scan_footprints(folder)]
    assert len(names) == 4 and not any(n.startswith(".") for n in names)


# ------------------------------------------------------------- previews (2026-08-28)

from dynamix.core.frames import GeographicFrame, LocalFrame  # noqa: E402
from dynamix.core.rasterfield import RasterField  # noqa: E402
from dynamix.geo.footprints import overview_field  # noqa: E402


def test_overview_field_is_a_small_raster_field_with_the_fill_border_as_nan(tmp_path):
    vals = _swath_tile(tmp_path / "AST_07XT_00401022008022006_20250619233734_SRF_VNIR_B03N.tif", n=64)
    f = overview_field(tmp_path / "AST_07XT_00401022008022006_20250619233734_SRF_VNIR_B03N.tif", max_px=32)
    assert isinstance(f, RasterField)
    assert max(f.values.shape) <= 32 and f.values.shape == (32, 32)
    assert np.isnan(f.values[0, 0]) and np.isnan(f.values[-1, -1])         # outside the swath
    assert not np.isnan(f.values[16, 16])                                    # inside it
    assert isinstance(f.frame, GeographicFrame)
    assert f.x_axis[0] == pytest.approx(119.0 + 0.5 / 32) and f.x_axis[-1] == pytest.approx(120.0 - 0.5 / 32)
    assert f.y_axis[0] == pytest.approx(-21.0 - 0.5 / 32) and f.y_axis[-1] == pytest.approx(-22.0 + 0.5 / 32)
    assert f.provenance["crs"] == "EPSG:4326" and f.provenance["overview"] == 2
    assert f.name == "AST_07XT_00401022008022006_20250619233734_SRF_VNIR_B03N@preview"


def test_overview_field_of_a_projected_raster_gets_a_local_frame_in_its_linear_units(tmp_path):
    _tile(tmp_path / "utm.tif", 500000.0, 7600000.0, 30.0, 40, crs="EPSG:32750", dtype="float32")
    f = overview_field(tmp_path / "utm.tif", max_px=20)
    assert isinstance(f.frame, LocalFrame) and f.frame.units == "metre"
    assert f.values.shape == (20, 20) and f.frame.dx == pytest.approx(60.0)
    assert f.provenance["crs"] == "EPSG:32750"


def test_overview_field_without_overviews_still_works_by_decimated_read(tmp_path):
    _swath_tile(tmp_path / "plain.tif", n=64, overviews=False)
    f = overview_field(tmp_path / "plain.tif", max_px=16)
    assert f.values.shape == (16, 16) and f.provenance["overview"] == 4


def test_overview_field_masks_float32_extreme_fill_even_when_nodata_lies(tmp_path):
    # BOEM west (2026-08-30): declared nodata 0.0, actual border fill float32 lowest -> the
    # levels crushed to a flat colour. Extreme sentinels are NaN regardless of the declaration.
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from dynamix.geo.footprints import overview_field
    path = tmp_path / "liar.tif"
    a = np.full((16, 16), -50.0, dtype=np.float32)
    a[:2, :] = np.finfo(np.float32).min                    # the real fill
    with rasterio.open(path, "w", driver="GTiff", height=16, width=16, count=1, dtype="float32",
                       nodata=0.0, crs="EPSG:32750", transform=from_origin(0, 480, 30, 30)) as dst:
        dst.write(a, 1)
    f = overview_field(path, max_px=16)
    assert np.isnan(f.values[:2, :]).all()
    assert np.nanmin(f.values) == -50.0
