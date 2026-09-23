# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.geo.mapping -- the Qt-free pixel -> lon/lat mapping layer.

No Qt here: dynamix.geo is Qt-free by law ("Nothing outside dynamix/shell/ imports
PySide6, pyvista or pyqtgraph") and headless-testable without a display. rasterio is required for
the CRS transform itself, same as every other GeoTIFF-touching test in this suite, so the whole
module is skipped without it.

Fixture: a synthetic 64x64 GeoTIFF written by THIS test with rasterio, on a custom NAD27
Transverse-Mercator, US-survey-foot CRS -- the shape of BOEM's real Gulf bathymetry grids
("Both files confirmed square 40x40 US-survey-ft
pixels, same NAD27 TM CRS"). The exact projection parameters are this test's own invention (no
literal WKT ships in this repo for the real BOEM grid); what matters is the SHAPE -- NAD27 datum,
Transverse Mercator, US survey foot linear unit -- matching test_shell_units.py's
_SURVEY_FOOT_WKT convention of a hand-built-but-real WKT1 string rasterio can parse and round-trip
through rasterio.warp.transform.

The fixture also reproduces the BOEM East nodata-sentinel trap: a
DECLARED nodata value that rasterio's own loader (RasterField.from_geotiff_window) already masks
to NaN, plus separate raw float-max sentinel cells (|v| >= 3e38) that are NOT the declared nodata
value and therefore survive that load untouched -- exactly the gap dynamix.geo.mapping's own
sentinel check exists to close.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
from rasterio.crs import CRS
from rasterio.transform import from_origin, xy as transform_xy
from rasterio.warp import transform as warp_transform

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.geo.mapping import NoGeoreference, _stride_for, field_lonlat_grid, has_georeference, points_lonlat, lonlat_to_pixels

# A hand-built NAD27 Transverse Mercator / US survey foot WKT1 string, same shape as
# tests/test_shell_units.py's _SURVEY_FOOT_WKT (NAD27 datum, Clarke 1866 spheroid, UNIT["US
# survey foot", 0.304800609601219] -- the exact GDAL/PROJ EPSG:9003 factor) but with
# PROJECTION["Transverse_Mercator"] instead of Lambert Conformal Conic, matching the "NAD27 TM" description of the real BOEM grids. Verified (while writing this test) to parse via
# rasterio.crs.CRS.from_wkt and round-trip through rasterio.warp.transform.
_BOEM_LIKE_WKT = (
    'PROJCS["NAD27 / BOEM Gulf TM",GEOGCS["NAD27",DATUM["North_American_Datum_1927",'
    'SPHEROID["Clarke 1866",6378206.4,294.978698213898,AUTHORITY["EPSG","7008"]],'
    'AUTHORITY["EPSG","6267"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4267"]],'
    'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],'
    'PARAMETER["central_meridian",-90],PARAMETER["scale_factor",0.9996],'
    'PARAMETER["false_easting",1640419.947506562],PARAMETER["false_northing",0],'
    'UNIT["US survey foot",0.304800609601219,AUTHORITY["EPSG","9003"]],'
    'AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
)

_NY = _NX = 64
_PX_FT = 40.0                    # square 40-ft pixels, matching the real BOEM grids
_WEST_FT = 1_800_000.0           # west edge, in the CRS's own US-survey-foot easting
_NORTH_FT = 9_910_000.0          # north edge, in the CRS's own US-survey-foot northing
_DECLARED_NODATA = -9999.0
_SENTINEL = -3.4e38
# Interior cells (never the extreme corners, so the corner round-trip check stays on real data).
_NODATA_CELLS = [(5, 5), (5, 6)]
_SENTINEL_CELLS = [(10, 10), (11, 11)]


def _write_boem_like_tif(path):
    crs = CRS.from_wkt(_BOEM_LIKE_WKT)
    transform = from_origin(_WEST_FT, _NORTH_FT, _PX_FT, _PX_FT)
    rows, cols = np.meshgrid(np.arange(_NY), np.arange(_NX), indexing="ij")
    vals = (-(500.0 + 0.1 * rows + 0.05 * cols)).astype(np.float32)
    for r, c in _NODATA_CELLS:
        vals[r, c] = _DECLARED_NODATA
    for r, c in _SENTINEL_CELLS:
        vals[r, c] = _SENTINEL
    with rasterio.open(path, "w", driver="GTiff", height=_NY, width=_NX, count=1,
                        dtype="float32", crs=crs, transform=transform,
                        nodata=_DECLARED_NODATA) as dst:
        dst.write(vals, 1)
    return crs, transform, vals


def _load_field(path):
    return RasterField.from_geotiff_window(str(path), row_off=0, col_off=0,
                                            height=_NY, width=_NX)


def _expected_lonlat(crs, transform, row, col):
    """Independent ("hand") pixel-center -> lon/lat, via rasterio directly -- not through
    dynamix.geo.mapping -- for row/col at the pixel CENTER convention (+0.5)."""
    x, y = transform_xy(transform, row, col)   # rasterio's own pixel-center convention
    lon, lat = warp_transform(crs, "EPSG:4326", [x], [y])
    return lon[0], lat[0]


# --------------------------------------------------------------------------- field_lonlat_grid


def test_corner_pixel_centers_match_a_hand_computed_rasterio_round_trip(tmp_path):
    path = tmp_path / "boem_like.tif"
    crs, transform, vals = _write_boem_like_tif(path)
    field = _load_field(path)

    lon2d, lat2d, values2d, stride = field_lonlat_grid(field, max_points=2_000_000)

    assert stride == 1
    assert lon2d.shape == lat2d.shape == values2d.shape == (_NY, _NX)

    exp_lon00, exp_lat00 = _expected_lonlat(crs, transform, 0, 0)
    exp_lonNN, exp_latNN = _expected_lonlat(crs, transform, _NY - 1, _NX - 1)

    assert lon2d[0, 0] == pytest.approx(exp_lon00, rel=1e-6)
    assert lat2d[0, 0] == pytest.approx(exp_lat00, rel=1e-6)
    assert lon2d[-1, -1] == pytest.approx(exp_lonNN, rel=1e-6)
    assert lat2d[-1, -1] == pytest.approx(exp_latNN, rel=1e-6)


def test_declared_nodata_and_float_max_sentinel_both_become_nan(tmp_path):
    path = tmp_path / "boem_like.tif"
    _write_boem_like_tif(path)
    field = _load_field(path)

    _, _, values2d, _ = field_lonlat_grid(field, max_points=2_000_000)

    for r, c in _NODATA_CELLS:
        assert math.isnan(values2d[r, c])
    for r, c in _SENTINEL_CELLS:
        assert math.isnan(values2d[r, c])
    # A cell that is neither must keep its real value.
    assert values2d[0, 0] == pytest.approx(-500.0, rel=1e-4)


def test_values2d_is_a_float64_copy_never_the_fields_own_array(tmp_path):
    path = tmp_path / "boem_like.tif"
    _write_boem_like_tif(path)
    field = _load_field(path)
    original = field.values.copy()

    _, _, values2d, _ = field_lonlat_grid(field, max_points=2_000_000)
    values2d[0, 0] = 12345.0

    assert values2d.dtype == np.float64
    assert values2d is not field.values
    np.testing.assert_array_equal(field.values[np.isfinite(original)],
                                   original[np.isfinite(original)])


def test_stride_decimates_to_honor_max_points(tmp_path):
    path = tmp_path / "boem_like.tif"
    _write_boem_like_tif(path)
    field = _load_field(path)

    lon2d, lat2d, values2d, stride = field_lonlat_grid(field, max_points=100)

    assert stride == _stride_for((_NY, _NX), 100)
    assert stride > 1
    ny_dec, nx_dec = _NY // stride, _NX // stride
    assert lon2d.shape == lat2d.shape == values2d.shape == (ny_dec, nx_dec)
    assert ny_dec * nx_dec <= 100


def test_max_points_large_enough_gives_stride_one(tmp_path):
    path = tmp_path / "boem_like.tif"
    _write_boem_like_tif(path)
    field = _load_field(path)

    _, _, _, stride = field_lonlat_grid(field, max_points=2_000_000)

    assert stride == 1


def test_field_with_no_crs_in_provenance_raises_no_georeference():
    field = RasterField._from_bare_array(np.zeros((8, 8)), "bare", frame=LocalFrame(units="px"),
                                          name="bare")
    assert "crs" not in field.provenance

    with pytest.raises(NoGeoreference):
        field_lonlat_grid(field)
    with pytest.raises(NoGeoreference):
        points_lonlat(field, cols=[0], rows=[0])


def test_field_with_explicit_none_crs_raises_no_georeference():
    field = RasterField._from_bare_array(np.zeros((8, 8)), "bare", frame=LocalFrame(units="px"),
                                          name="bare")
    field.provenance["crs"] = None

    with pytest.raises(NoGeoreference):
        field_lonlat_grid(field)


# --------------------------------------------------------------------------- points_lonlat


def test_points_lonlat_agrees_with_field_lonlat_grid_at_matching_pixels(tmp_path):
    path = tmp_path / "boem_like.tif"
    _write_boem_like_tif(path)
    field = _load_field(path)

    lon2d, lat2d, _, stride = field_lonlat_grid(field, max_points=2_000_000)
    assert stride == 1  # so grid index == pixel index, matching points_lonlat's (col, row)

    rows = np.array([0, 5, 20, 63])
    cols = np.array([0, 6, 40, 63])

    lon, lat = points_lonlat(field, cols=cols, rows=rows)

    np.testing.assert_allclose(lon, lon2d[rows, cols], rtol=1e-6)
    np.testing.assert_allclose(lat, lat2d[rows, cols], rtol=1e-6)


def test_points_lonlat_matches_the_hand_computed_round_trip(tmp_path):
    path = tmp_path / "boem_like.tif"
    crs, transform, _ = _write_boem_like_tif(path)
    field = _load_field(path)

    row, col = 30, 17
    exp_lon, exp_lat = _expected_lonlat(crs, transform, row, col)

    lon, lat = points_lonlat(field, cols=[col], rows=[row])

    assert lon[0] == pytest.approx(exp_lon, rel=1e-6)
    assert lat[0] == pytest.approx(exp_lat, rel=1e-6)


# --------------------------------------------------------------------------- _stride_for


def test_stride_for_is_the_smallest_stride_satisfying_the_budget():
    assert _stride_for((64, 64), 2_000_000) == 1
    s = _stride_for((64, 64), 100)
    assert (64 // s) * (64 // s) <= 100
    assert (64 // (s - 1)) * (64 // (s - 1)) > 100 if s > 1 else True


def test_stride_for_exact_fit_is_one():
    assert _stride_for((10, 10), 100) == 1


# --------------------------------------------------------------------------- lonlat_to_pixels


def test_lonlat_to_pixels_round_trip_with_points_lonlat(tmp_path):
    """The exact inverse: points_lonlat -> lonlat_to_pixels -> points_lonlat should recover original."""
    path = tmp_path / "boem_like.tif"
    crs, transform, _ = _write_boem_like_tif(path)
    field = _load_field(path)

    # Take some pixel indices, convert to lon/lat, then back to pixels
    cols_orig = np.array([0, 5, 20, 63], dtype=np.intp)
    rows_orig = np.array([0, 10, 40, 63], dtype=np.intp)

    lon, lat = points_lonlat(field, cols=cols_orig, rows=rows_orig)
    cols_back, rows_back = lonlat_to_pixels(field, lon, lat)

    np.testing.assert_allclose(cols_back, cols_orig.astype(np.float64), atol=1e-4)
    np.testing.assert_allclose(rows_back, rows_orig.astype(np.float64), atol=1e-4)


def test_lonlat_to_pixels_with_no_crs_raises_no_georeference():
    """Field without CRS in provenance should raise NoGeoreference."""
    field = RasterField._from_bare_array(np.zeros((8, 8)), "bare", frame=LocalFrame(units="px"),
                                          name="bare")
    with pytest.raises(NoGeoreference):
        lonlat_to_pixels(field, np.array([0.0]), np.array([0.0]))


def test_lonlat_to_pixels_off_raster_returns_float_unclipped(tmp_path):
    """Points outside the raster bounds should return out-of-range floats, not clipped or raised."""
    path = tmp_path / "boem_like.tif"
    crs, transform, _ = _write_boem_like_tif(path)
    field = _load_field(path)

    # Take a point in lon/lat, convert to pixels; then take a point far outside the grid
    lon_on, lat_on = points_lonlat(field, cols=[10], rows=[10])
    cols_on, rows_on = lonlat_to_pixels(field, lon_on, lat_on)

    # Should recover the original pixel index
    np.testing.assert_allclose(cols_on, [10.0], atol=1e-4)
    np.testing.assert_allclose(rows_on, [10.0], atol=1e-4)

    # Now create a lon/lat point far to the west (way off the raster)
    # The field's west edge is around _WEST_FT in the projected coords
    # We can invert the transform to estimate: go west by ~100 pixels worth
    west_lon = lon_on[0] - 1.0  # 1 degree west -- definitely off the raster
    cols_off, rows_off = lonlat_to_pixels(field, np.array([west_lon]), lat_on)

    # Should be negative (off the west edge), not clipped or raised
    assert cols_off[0] < 0.0
    assert not np.isnan(cols_off[0]) and not np.isinf(cols_off[0])
    assert rows_off[0] >= 0 and rows_off[0] < _NY


def test_whole_file_geotiff_stamps_crs_in_provenance(tmp_path):
    """The _from_geotiff whole-file path should stamp provenance["crs"] and has_georeference."""
    path = tmp_path / "boem_like_small.tif"
    crs, transform, _ = _write_boem_like_tif(path)

    # Load via the whole-file path (_from_geotiff)
    field = RasterField.from_file(path)

    # Provenance should include the CRS
    assert "crs" in field.provenance
    assert field.provenance["crs"] is not None
    # The CRS should stringify to something non-empty
    assert str(field.provenance["crs"]).strip()

    # has_georeference should return True
    assert has_georeference(field)


# --------------------------------------------------------------------------- Qt-free law


def test_geo_package_has_no_qt_import():
    """Belt-and-suspenders alongside tests/test_shell_boundaries.py's
    test_devloop_is_the_only_shell_importer_outside_shell, which already AST-scans every module
    under src/dynamix (outside dynamix/shell/) for PySide6/pyqtgraph/pyvista -- dynamix/geo is
    covered by that existing whole-tree scan automatically, with nothing to extend (it does not
    walk a hard-coded package list). This test just says so locally, next to the module it is
    about."""
    import ast
    import pathlib

    geo_dir = pathlib.Path(__file__).resolve().parents[1] / "src" / "dynamix" / "geo"
    for path in sorted(geo_dir.rglob("*.py")):
        imported = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported.add((node.module or "").split(".")[0])
        assert not (imported & {"PySide6", "pyqtgraph", "pyvista"}), path
