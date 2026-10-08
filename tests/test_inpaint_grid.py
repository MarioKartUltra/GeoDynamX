# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Grid sensing: same, nested, foreign, and the mapping between two georeferences."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.inpaint import grid as gr
from dynamix.core.rasterfield import RasterField

UTM = "EPSG:32615"


def _g(x0, dx, y0, dy, shape, crs=UTM, **extra):
    return gr.georef_from(crs, x0, dx, y0, dy, shape, **extra)


def test_georef_of_a_field_reads_its_axes_and_provenance():
    f = RasterField(name="f", values=np.zeros((4, 6)), frame=LocalFrame(units="m"),
                    x_axis=500000.0 + 30.0 * (np.arange(6) + 0.5),
                    y_axis=4000000.0 - 30.0 * (np.arange(4) + 0.5),
                    provenance={"crs": UTM, "source": "/a.tif"})
    g = gr.georef(f)
    assert g["crs"] == UTM and g["shape"] == (4, 6) and g["georeferenced"]
    assert g["x0"] == pytest.approx(500015.0) and g["dx"] == 30.0 and g["dy"] == -30.0
    bare = RasterField(name="b", values=np.zeros((4, 6)), frame=LocalFrame(),
                       x_axis=np.arange(6.0) + 0.5, y_axis=np.arange(4.0) + 0.5,
                       provenance={"source": "/b.hdf", "georef": "none (pixel-index axes)"})
    assert not gr.georef(bare)["georeferenced"]


def test_same_grid_with_an_integer_and_a_sub_pixel_offset():
    t = _g(500015.0, 30.0, 4000000.0 - 15.0, -30.0, (100, 100))
    r = gr.relate(t, _g(500015.0 + 3 * 30.0, 30.0, 4000000.0 - 15.0 - 2 * 30.0, -30.0, (50, 50)))
    assert (r.kind, r.ratio, r.direction, r.offset) == ("same", 1, "equal", (2, 3))
    assert r.subpixel == (0.0, 0.0)
    r = gr.relate(t, _g(500015.0 + 30.0 * 3.2, 30.0, 4000000.0 - 15.0, -30.0, (50, 50)))
    assert r.kind == "same" and r.offset == (0, 3) and r.subpixel == pytest.approx((0.0, 0.2))
    r = gr.relate(t, _g(500015.0 + 30.0 * 3.4, 30.0, 4000000.0 - 15.0, -30.0, (50, 50)))
    assert r.kind == "foreign" and r.mapping_source == "crs"      # past the quarter-pixel tolerance


@pytest.mark.parametrize("f", [2, 3, 6])
def test_nested_coarser_and_finer_guides(f):
    t = _g(500015.0, 30.0, 4000000.0 - 15.0, -30.0, (120, 120))
    # a coarser guide whose first pixel centre is the centre of the target's block at (6, 12)
    cx = 500015.0 + 30.0 * (12 + (f - 1) / 2)
    cy = 4000000.0 - 15.0 - 30.0 * (6 + (f - 1) / 2)
    r = gr.relate(t, _g(cx, 30.0 * f, cy, -30.0 * f, (10, 10)))
    assert (r.kind, r.ratio, r.direction, r.offset) == ("nested", f, "coarser", (6, 12))
    # a finer guide: the target's first block top-left sits at guide pixel (2 f, 4 f)
    gx0 = 500015.0 - 15.0 + 15.0 / f - 30.0 * 4
    gy0 = 4000000.0 - 15.0 + 15.0 - 15.0 / f + 30.0 * 2
    r = gr.relate(t, _g(gx0, 30.0 / f, gy0, -30.0 / f, (600, 600)))
    assert (r.kind, r.ratio, r.direction) == ("nested", f, "finer")
    assert r.offset == (2 * f, 4 * f)


def test_a_crs_mismatch_and_a_misaligned_coarse_grid_are_foreign_through_the_crs():
    t = _g(500015.0, 30.0, 4000000.0 - 15.0, -30.0, (100, 100))
    r = gr.relate(t, _g(-91.0, 0.0003, 36.0, -0.0003, (100, 100), crs="EPSG:4326"))
    assert r.kind == "foreign" and r.mapping_source == "crs"
    r = gr.relate(t, _g(500015.0 + 7.0, 60.0, 4000000.0 - 15.0, -60.0, (50, 50)))
    assert r.kind == "foreign"
    r = gr.relate(t, _g(500015.0, 150.0, 4000000.0 - 15.0, -150.0, (20, 20)))   # ratio 5
    assert r.kind == "foreign"


def test_lattice_and_georeference_free_rasters():
    lat = {"file": "/scene.hdf", "group": "VNIR_Band1"}
    a = _g(0.5, 1.0, 0.5, 1.0, (4200, 4100), crs=None, lattice=lat, source="/scene.hdf")
    b = _g(0.5, 1.0, 0.5, 1.0, (4200, 4100), crs=None,
           lattice={"file": "/scene.hdf", "group": "VNIR_Band2"}, source="/scene.hdf")
    r = gr.relate(a, b)                                     # two BANDS of one L1A file: not one grid
    assert (r.kind, r.mapping_source) == ("foreign", "lattice")
    swir = _g(0.5, 1.0, 0.5, 1.0, (2100, 2048), crs=None,
              lattice={"file": "/scene.hdf", "group": "SWIR_Band4"}, source="/scene.hdf")
    assert gr.relate(a, swir).kind == "foreign"
    window = _g(200.5, 1.0, 100.5, 1.0, (512, 512), crs=None, lattice=lat, source="/scene.hdf",
                full_shape=(4200, 4100))                    # a window of band 1 itself
    r = gr.relate(a, window)
    assert (r.kind, r.offset, r.mapping_source) == ("same", (100, 200), "file")
    dem = _g(500015.0, 30.0, 4000000.0 - 15.0, -30.0, (100, 100))
    assert gr.relate(a, dem).kind == "foreign" and gr.relate(a, dem).mapping_source == "lattice"
    assert gr.relate(dem, a).mapping_source == "lattice"
    c = _g(0.5, 1.0, 0.5, 1.0, (10, 10), crs=None, source="/other.npz")
    d = _g(0.5, 1.0, 0.5, 1.0, (10, 10), crs=None, source="/third.npz")
    r = gr.relate(c, d)
    assert r.kind == "foreign" and r.mapping_source == "none"
    assert gr.relate(c, _g(2.5, 1.0, 3.5, 1.0, (6, 6), crs=None, source="/other.npz",
                           full_shape=(10, 10))).offset == (3, 2)
    assert gr.relate(c, _g(0.5, 1.0, 0.5, 1.0, (6, 6), crs=None, source="/other.npz")).kind == "foreign"


def test_the_stamp_is_json_safe_and_complete():
    import json

    t = _g(500015.0, 30.0, 4000000.0 - 15.0, -30.0, (100, 100))
    s = gr.relate(t, _g(500015.0 + 30.0 * 3.2, 30.0, 4000000.0 - 15.0, -30.0, (50, 50))).stamp()
    assert set(s) == {"kind", "ratio", "direction", "offset", "subpixel", "mapping_source"}
    json.dumps(s)


def test_warp_rows_cols_between_two_crs_grids_lands_on_the_right_pixels():
    pytest.importorskip("rasterio")
    t = _g(500015.0, 30.0, 4000000.0 - 15.0, -30.0, (20, 20))
    g = _g(500015.0 + 45.0, 15.0, 4000000.0 - 15.0 - 45.0, -15.0, (40, 40))
    rows, cols = gr.warp_rows_cols(g, t)                  # each guide pixel in target pixels
    assert rows.shape == (40, 40)
    assert rows[0, 0] == pytest.approx(1.5) and cols[0, 0] == pytest.approx(1.5)
    assert rows[2, 0] == pytest.approx(2.5)
    # through a reprojection: the same points expressed in lon/lat land back where they were
    lon, lat = gr.lonlat_of(t, np.array([0.0, 19.0]), np.array([0.0, 19.0]))
    r2, c2 = gr.pixels_of(t, lon, lat)
    np.testing.assert_allclose(r2, [0.0, 19.0], atol=1e-6)
    np.testing.assert_allclose(c2, [0.0, 19.0], atol=1e-6)


def test_a_synthetic_lattice_maps_pixels_to_lonlat_and_back(tmp_path):
    pytest.importorskip("pyhdf")
    pytest.importorskip("scipy")
    from pyhdf.SD import SD, SDC

    p = tmp_path / "lat.hdf"
    sd = SD(str(p), SDC.WRITE | SDC.CREATE)
    rr, cc = np.meshgrid(np.linspace(0, 4200, 13), np.linspace(0, 4100, 11), indexing="ij")
    lat = 30.0 - rr * 1e-4                                  # a plain affine lattice (geodetic)
    lon = -91.0 + cc * 1e-4 + rr * 2e-5
    geocentric = np.degrees(np.arctan(np.tan(np.radians(lat)) * (1.0 - gr._WGS84_E2)))
    for name, arr in (("Latitude", geocentric), ("Longitude", lon)):   # the SDS holds geocentric latitude
        s = sd.create(name, SDC.FLOAT64, arr.shape)
        s[:] = arr
        s.endaccess()
    lp = sd.create("LatticePoint", SDC.FLOAT64, (13, 11, 2))
    lp[:] = np.stack([cc, rr], axis=-1)                     # (x, y) per node
    lp.endaccess()
    sd.end()
    ref = {"file": str(p), "sds": {"lat": "Latitude", "lon": "Longitude", "points": "LatticePoint"}}
    g = _g(0.5, 1.0, 0.5, 1.0, (4200, 4100), crs=None, lattice=ref, source=str(p))
    lo, la = gr.lonlat_of(g, np.array([100.0, 2000.0]), np.array([50.0, 3000.0]))
    assert la[0] == pytest.approx(30.0 - 100.0 * 1e-4, abs=1e-9)
    assert lo[1] == pytest.approx(-91.0 + 3000.0 * 1e-4 + 2000.0 * 2e-5, abs=1e-9)
    r, c = gr.pixels_of(g, lo, la)
    np.testing.assert_allclose(r, [100.0, 2000.0], atol=0.05)
    np.testing.assert_allclose(c, [50.0, 3000.0], atol=0.05)
    # a window of the band (origin row 1000, col 500) maps its local (0, 0) to the file's (1000, 500)
    w = _g(500.5, 1.0, 1000.5, 1.0, (256, 256), crs=None, lattice=ref, source=str(p),
           full_shape=(4200, 4100))
    lo_w, la_w = gr.lonlat_of(w, np.array([0.0]), np.array([0.0]))
    lo_f, la_f = gr.lonlat_of(g, np.array([1000.0]), np.array([500.0]))
    np.testing.assert_allclose([lo_w, la_w], [lo_f, la_f], atol=1e-9)
    rows, cols = gr.warp_rows_cols(w, g)
    assert rows[0, 0] == pytest.approx(1000.0, abs=1e-6) and cols[0, 0] == pytest.approx(500.0, abs=1e-6)


def test_a_strided_raster_of_the_same_band_is_not_the_same_grid():
    win = _g(200.5, 1.0, 100.5, 1.0, (512, 512), crs=None, source="/scene.hdf",
             full_shape=(4200, 4100))
    pic = _g(0.5, 4.0, 0.5, 4.0, (1050, 1025), crs=None, source="/scene.hdf",
             full_shape=(4200, 4100))                       # every fourth pixel, axes on the sampled pixel
    r = gr.relate(win, pic)
    assert r.kind == "foreign" and r.direction == "coarser" and r.mapping_source == "none"
    assert gr.relate(pic, win).direction == "finer"
    blocks = _g(1.0, 2.0, 1.0, 2.0, (2100, 2050), crs=None, source="/scene.hdf",
                full_shape=(4200, 4100))                    # 2 x 2 block means, axes on block centres
    r = gr.relate(win, blocks)
    assert (r.kind, r.ratio, r.direction, r.mapping_source) == ("nested", 2, "coarser", "file")
    half = _g(200.75, 1.0, 100.5, 1.0, (512, 512), crs=None, source="/scene.hdf",
              full_shape=(4200, 4100))
    r = gr.relate(win, half)
    assert r.kind == "same" and r.offset == (0, 0) and r.subpixel == pytest.approx((0.0, 0.25))
    assert gr.relate(win, _g(201.0, 1.0, 100.5, 1.0, (512, 512), crs=None, source="/scene.hdf",
                             full_shape=(4200, 4100))).kind == "foreign"   # half a pixel off: beyond tol_px


def test_a_foreign_relation_reports_the_guide_pixel_direction_when_the_steps_compare():
    target = _g(500015.0, 30.0, 4000000.0 - 15.0, -30.0, (100, 100))
    coarse = _g(500075.0, 150.0, 4000000.0 - 75.0, -150.0, (20, 20))   # ratio 5: not nested
    r = gr.relate(target, coarse)
    assert (r.kind, r.ratio, r.direction) == ("foreign", 1, "coarser")
    assert gr.relate(coarse, target).direction == "finer"
    other = _g(500075.0, 150.0, 4000000.0 - 75.0, -150.0, (20, 20), crs="EPSG:32616")
    assert gr.relate(target, other).direction == "equal"      # different units: incomparable
