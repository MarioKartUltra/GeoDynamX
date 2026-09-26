# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.opening -- GeoTIFF and zip-of-GeoTIFF opening, never extracted.

No Qt here: :mod:`dynamix.shell.opening` is pure resolver + ``RasterField`` loader, so these run
as plain pytest against tiny synthetic fixtures. ``rasterio`` is required (as it is for any of
the GeoTIFF machinery in ``dynamix.core.rasterfield``), so the whole module is skipped without it.
"""
from __future__ import annotations

import os
import zipfile

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
from rasterio.transform import from_origin

from dynamix.shell.opening import open_field, resolve_openable

#: A tiny PROJECTED GeoTIFF, in the shape of the BOEM layout (float32, single band).
_CRS = "EPSG:32615"


def _write_tif(path, ny=32, nx=32, seed=0):
    rng = np.random.default_rng(seed)
    vals = (rng.random((ny, nx)).astype(np.float32) * 100.0) - 50.0
    with rasterio.open(path, "w", driver="GTiff", height=ny, width=nx, count=1, dtype="float32",
                        crs=_CRS, transform=from_origin(500000.0, 3200000.0, 2.0, 2.0)) as dst:
        dst.write(vals, 1)
    return vals


def _write_zip(zip_path, member_paths):
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in member_paths:
            zf.write(p, arcname=os.path.basename(str(p)))


# --------------------------------------------------------------------------- resolve_openable


def test_resolve_openable_passes_through_non_zip_suffixes(tmp_path):
    npz = tmp_path / "a.npz"
    tif = tmp_path / "b.tif"
    npz.write_bytes(b"")
    tif.write_bytes(b"")
    assert resolve_openable(str(npz)) == str(npz)
    assert resolve_openable(str(tif)) == str(tif)


def test_resolve_openable_zip_with_one_tif_builds_the_gdal_url(tmp_path):
    tif = tmp_path / "field.tif"
    _write_tif(tif)
    zpath = tmp_path / "field.zip"
    _write_zip(zpath, [tif])

    url = resolve_openable(str(zpath))

    assert url == f"zip://{zpath.resolve()}!field.tif"


def test_resolve_openable_zip_with_no_tif_raises_naming_contents(tmp_path):
    other = tmp_path / "readme.txt"
    other.write_text("hi")
    zpath = tmp_path / "empty.zip"
    _write_zip(zpath, [other])

    with pytest.raises(ValueError, match="no .tif"):
        resolve_openable(str(zpath))


def test_resolve_openable_zip_with_two_tifs_raises_listing_members(tmp_path):
    tif1 = tmp_path / "a.tif"
    tif2 = tmp_path / "b.tif"
    _write_tif(tif1)
    _write_tif(tif2, seed=1)
    zpath = tmp_path / "two.zip"
    _write_zip(zpath, [tif1, tif2])

    with pytest.raises(ValueError, match="multiple"):
        resolve_openable(str(zpath))


def test_resolve_openable_ignores_sidecars(tmp_path):
    """.ovr/.tfw/.aux.xml sidecars sit right next to the real member and must not compete for it."""
    tif = tmp_path / "field.tif"
    _write_tif(tif)
    ovr = tmp_path / "field.tif.ovr"
    ovr.write_bytes(b"not a tif")
    tfw = tmp_path / "field.tfw"
    tfw.write_text("not a tif either")
    aux = tmp_path / "field.tif.aux.xml"
    aux.write_text("<PAMDataset/>")
    zpath = tmp_path / "sidecars.zip"
    _write_zip(zpath, [tif, ovr, tfw, aux])

    url = resolve_openable(str(zpath))

    assert url == f"zip://{zpath.resolve()}!field.tif"


# --------------------------------------------------------------------------- open_field


def test_open_field_zip_matches_the_bare_tif(tmp_path):
    """The whole point: opening the zip must read the identical pixels as opening the tif itself."""
    tif = tmp_path / "field.tif"
    vals = _write_tif(tif)
    zpath = tmp_path / "field.zip"
    _write_zip(zpath, [tif])

    rf_bare = open_field(str(tif))
    rf_zip = open_field(str(zpath))

    assert np.allclose(rf_bare.values, vals)
    assert np.allclose(rf_zip.values, vals)
    assert np.allclose(rf_zip.values, rf_bare.values)


def test_open_field_whole_zip_route_has_a_clean_name(tmp_path):
    """F1 regression: the whole-file route must not let the mangled zip:// URL leak into
    ``.name`` -- wtmm_backend uses it verbatim as a cache directory / output filename, and
    ``zip://ABS!field.tif`` is not a filesystem-safe name on any platform."""
    tif = tmp_path / "field.tif"
    _write_tif(tif)
    zpath = tmp_path / "field.zip"
    _write_zip(zpath, [tif])

    rf = open_field(str(zpath))

    assert rf.name == "field"
    assert "!" not in rf.name and "zip:" not in rf.name


def test_open_field_window_guard_clamps_an_oversized_window_to_the_full_raster(tmp_path):
    """F2: window_size bigger than the raster itself must clamp to (0, 0) and the full extent,
    rather than producing a negative offset or a window rasterio can't satisfy."""
    tif = tmp_path / "small.tif"
    vals = _write_tif(tif, ny=10, nx=10)

    rf = open_field(str(tif), max_pixels=1, window_size=100, mode="window")

    assert rf.values.shape == (10, 10)
    assert np.allclose(rf.values, vals)
    assert "@x0y0" in rf.name


def test_open_field_window_guard_centers_a_native_window(tmp_path):
    tif = tmp_path / "field.tif"
    vals = _write_tif(tif, ny=32, nx=32)

    rf = open_field(str(tif), max_pixels=100, window_size=16, mode="window")

    assert rf.values.shape == (16, 16)
    assert np.allclose(rf.values, vals[8:24, 8:24])
    assert "@x8y8" in rf.name         # the offset is honest and visible in the name


def test_open_field_window_guard_applies_through_a_zip_too(tmp_path):
    tif = tmp_path / "field.tif"
    vals = _write_tif(tif, ny=32, nx=32)
    zpath = tmp_path / "field.zip"
    _write_zip(zpath, [tif])

    rf = open_field(str(zpath), max_pixels=100, window_size=16, mode="window")

    assert rf.values.shape == (16, 16)
    assert np.allclose(rf.values, vals[8:24, 8:24])
    assert "@x8y8" in rf.name


def test_open_field_never_extracts_anything_next_to_the_zip(tmp_path):
    tif = tmp_path / "field.tif"
    _write_tif(tif, ny=32, nx=32)
    zpath = tmp_path / "field.zip"
    _write_zip(zpath, [tif])
    before = set(os.listdir(tmp_path))

    open_field(str(zpath))                              # whole-file route
    open_field(str(zpath), max_pixels=100, window_size=16)   # windowed route

    after = set(os.listdir(tmp_path))
    assert after == before


# --------------------------------------------------------------------------- real-data smoke

_WEST_ZIP = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "BOEM_Bathymetry_West_meters_tiff(1).zip")
)


@pytest.mark.skipif(not os.path.exists(_WEST_ZIP), reason="real BOEM fixture not present on this machine")
def test_real_boem_west_opens_as_a_whole_extent_overview():
    """Real-data smoke: the overview open of BOEM West is the WHOLE 38470x20782 extent as a
    decimated overview -- finite, negative (below-sea-level) depths, provenance carrying the
    stride every native-mapping consumer needs. The default open is the display picture; this
    pins the overview path."""
    rf = open_field(_WEST_ZIP, mode="overview")

    ov = int(rf.provenance["overview"])
    assert ov > 1 and "@ov" in rf.name
    assert rf.provenance["full_dims"] == (20782, 38470)
    ny, nx = rf.values.shape[:2]
    assert ny * nx <= 64_000_000
    assert ny == (20782 + ov - 1) // ov and nx == (38470 + ov - 1) // ov
    finite = rf.values[np.isfinite(rf.values)]
    assert finite.size > 0
    assert (finite < 0).any()


# ------------------------------------------------------------------ whole-extent overview

def test_open_field_too_big_defaults_to_a_whole_extent_overview(tmp_path):
    """In overview mode a raster over max_pixels opens as a DECIMATED WHOLE-EXTENT overview
    (display/navigation; native pixels come back through the ROI child tool), stride chosen
    to fit the budget, provenance carrying overview/source/full_dims so every consumer can
    map display -> native as r*ov + off (the _on_roi_create convention)."""
    tif = tmp_path / "big.tif"
    vals = _write_tif(tif, 40, 60)
    rf = open_field(str(tif), max_pixels=600, mode="overview")   # 2400 px -> stride 2 -> 20x30
    assert rf.values.shape[:2] == (20, 30)
    assert "@ov2" in rf.name
    assert int(rf.provenance["overview"]) == 2
    assert rf.provenance["full_dims"] == (40, 60)
    assert rf.provenance["source"].endswith("big.tif")
    # nearest decimation: every kept value comes from ITS OWN 2x2 block of the native grid
    v = np.asarray(rf.values)
    for i, j in ((0, 0), (7, 11), (19, 29)):
        block = vals[2 * i:2 * i + 2, 2 * j:2 * j + 2]
        assert v[i, j] in block
    # georeference survives (opening the anomaly vectors "hid" the bathy --
    # the overview had dropped its CRS, so the geo view refused to drape it): provenance crs,
    # PROJECTED axes at the sampled blocks' native centers, and a metric frame.
    assert rf.provenance["crs"]
    assert rf.x_axis[0] == pytest.approx(500000.0 + 1.0 * 2.0)   # (j+.5)*stride px-centers
    assert rf.x_axis[1] - rf.x_axis[0] == pytest.approx(2 * 2.0)  # stride * native px size
    assert rf.y_axis[0] == pytest.approx(3200000.0 - 1.0 * 2.0)
    assert rf.frame.units and rf.frame.dx == pytest.approx(2 * 2.0)


@pytest.mark.skipif(not os.path.exists(_WEST_ZIP), reason="real BOEM fixture not present on this machine")
def test_real_boem_west_opens_as_the_display_picture():
    """The default open of BOEM West is the display PICTURE --
    whole extent, <= 4096 samples on the long side, no overview key, full_dims = the file."""
    from dynamix.roi.picture import native_shape

    rf = open_field(_WEST_ZIP)
    assert "overview" not in rf.provenance
    s = int(rf.provenance["display_stride"])
    assert s > 1 and max(rf.values.shape) <= 4096
    assert native_shape(rf) == (20782, 38470)
    finite = rf.values[np.isfinite(rf.values)]
    assert finite.size > 0 and (finite < 0).any()


def test_open_field_too_big_defaults_to_the_display_picture(tmp_path):
    """Over max_pixels the DEFAULT is the display-only picture (exact
    block-centre samples, file-pixel coordinates) -- not the overview dataset."""
    from dynamix.roi.picture import block_centres

    tif = tmp_path / "big.tif"
    vals = _write_tif(tif, 40, 60)
    rf = open_field(str(tif), max_pixels=600)
    assert "overview" not in rf.provenance
    s = int(rf.provenance["display_stride"])
    assert s >= 1 and "@pic" in rf.name
    assert rf.provenance["full_dims"] == (40, 60)
    assert rf.provenance["source"].endswith("big.tif")
    assert np.array_equal(rf.values, vals[np.ix_(block_centres(40, s), block_centres(60, s))])


def test_open_field_window_mode_is_still_available(tmp_path):
    tif = tmp_path / "big2.tif"
    _write_tif(tif, 40, 60)
    rf = open_field(str(tif), max_pixels=600, window_size=16, mode="window")
    assert rf.values.shape == (16, 16)
    assert "@x" in rf.name


def test_boem_style_undeclared_sentinels_read_as_nan(tmp_path):
    """The real BOEM tifs DECLARE nodata = 0.0 but FILL empty areas with float32-lowest
    (-3.4028235e38) -- the declaration is wrong, and the sentinel would pin every color ramp
    and poison analysis. Both read paths convert |v| >= 3e38 to NaN (the geo _mask_sentinels
    convention), in addition to the declared nodata."""
    import rasterio
    from rasterio.transform import from_origin

    tif = tmp_path / "sentinel.tif"
    vals = np.full((40, 60), -25.0, dtype=np.float32)
    vals[:5, :] = np.float32(-3.4028235e38)          # the undeclared fill
    vals[10, 10] = 0.0                                # the DECLARED nodata
    with rasterio.open(tif, "w", driver="GTiff", height=40, width=60, count=1,
                       dtype="float32", crs="EPSG:32615", nodata=0.0,
                       transform=from_origin(500000.0, 3200000.0, 2.0, 2.0)) as dst:
        dst.write(vals, 1)

    ov = open_field(str(tif), max_pixels=600)         # the overview path
    v = np.asarray(ov.values)
    assert np.isnan(v[0, :]).all()                    # sentinel rows -> NaN
    assert np.nanmin(v) == -25.0                      # the ramp never sees 3.4e38

    from dynamix.core.rasterfield import RasterField
    win = RasterField.from_geotiff_window(str(tif), row_off=0, col_off=0,
                                          height=20, width=20)
    wv = np.asarray(win.values)
    assert np.isnan(wv[0, :]).all()
    assert np.isnan(wv[10, 10])                       # declared nodata still honored
    assert np.nanmin(wv) == -25.0
