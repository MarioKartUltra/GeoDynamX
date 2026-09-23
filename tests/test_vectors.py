# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""dynamix.geo.vectors — open GIS vector files (shapefiles) as reference layers (2026-08-29).

BOEM's seafloor-anomaly package is 33 shapefiles (polygonZ / polylineZ / point) in the same
CRS as the West bathymetry. The reader is stdlib-only (no fiona/pyogrio/shapely in the env);
the test writes its own tiny shapefiles so it never depends on the real package."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from dynamix.geo.vectors import read_shapefile, to_field_pixels, to_lonlat

pytest.importorskip("rasterio")


def _write_shapefile(base: Path, shape_type: int, records, fields, rows, prj_wkt: str):
    """Minimal ESRI shapefile writer (main + index + dBASE III + .prj). ``records`` is a list of
    parts-lists: [[(x, y), ...], ...] per feature (points: one part, one vertex)."""
    def rec_bytes(n, parts):
        if shape_type in (1, 11):                              # point / pointZ
            (x, y), = parts[0]
            body = struct.pack("<i", shape_type) + struct.pack("<2d", x, y)
            if shape_type == 11:
                body += struct.pack("<2d", 0.0, 0.0)          # z, m
            return body
        pts = [p for part in parts for p in part]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        body = struct.pack("<i", shape_type) + struct.pack("<4d", min(xs), min(ys), max(xs), max(ys))
        body += struct.pack("<2i", len(parts), len(pts))
        off = 0
        for part in parts:
            body += struct.pack("<i", off); off += len(part)
        for x, y in pts:
            body += struct.pack("<2d", x, y)
        if shape_type in (13, 15):                             # Z variants: zrange + z + mrange + m
            body += struct.pack("<2d", 0.0, 0.0) + struct.pack(f"<{len(pts)}d", *([0.0] * len(pts)))
            body += struct.pack("<2d", 0.0, 0.0) + struct.pack(f"<{len(pts)}d", *([0.0] * len(pts)))
        return body
    contents, index = b"", b""
    offset = 50                                                # in 16-bit words
    for n, parts in enumerate(records, 1):
        body = rec_bytes(n, parts)
        contents += struct.pack(">2i", n, len(body) // 2) + body
        index += struct.pack(">2i", offset, len(body) // 2)
        offset += 4 + len(body) // 2
    allx = [p[0] for r in records for part in r for p in part]; ally = [p[1] for r in records for part in r for p in part]
    def header(total_words):
        h = struct.pack(">i", 9994) + b"\x00" * 20 + struct.pack(">i", total_words)
        h += struct.pack("<2i", 1000, shape_type) + struct.pack("<4d", min(allx), min(ally), max(allx), max(ally))
        return h + struct.pack("<4d", 0, 0, 0, 0)
    (base.with_suffix(".shp")).write_bytes(header(50 + len(contents) // 2) + contents)
    (base.with_suffix(".shx")).write_bytes(header(50 + len(index) // 2) + index)
    # dBASE III: fields = [(name, type, length, decimals)]
    reclen = 1 + sum(f[2] for f in fields)
    hdr = bytes([3, 26, 1, 1]) + struct.pack("<I", len(rows)) + struct.pack("<2H", 32 + 32 * len(fields) + 1, reclen) + b"\x00" * 20
    for name, ftype, length, dec in fields:
        hdr += name.encode("ascii").ljust(11, b"\x00") + ftype.encode("ascii") + b"\x00" * 4 + bytes([length, dec]) + b"\x00" * 14
    hdr += b"\x0d"
    body = b""
    for row in rows:
        body += b" "
        for (name, ftype, length, dec), val in zip(fields, row):
            txt = (f"{val:>{length}.{dec}f}" if ftype in "NF" else str(val).ljust(length))[:length]
            body += txt.encode("latin-1")
    (base.with_suffix(".dbf")).write_bytes(hdr + body + b"\x1a")
    (base.with_suffix(".prj")).write_text(prj_wkt)


@pytest.fixture
def utm_prj():
    from rasterio.crs import CRS
    return CRS.from_epsg(32750).to_wkt()


@pytest.fixture
def polys(tmp_path, utm_prj):
    base = tmp_path / "anomaly_slumps"
    square = [(500000.0, 7600000.0), (501000.0, 7600000.0), (501000.0, 7601000.0), (500000.0, 7601000.0), (500000.0, 7600000.0)]
    hole = [(500200.0, 7600200.0), (500200.0, 7600400.0), (500400.0, 7600400.0), (500200.0, 7600200.0)]
    far = [(510000.0, 7610000.0), (511000.0, 7610000.0), (511000.0, 7611000.0), (510000.0, 7610000.0)]
    _write_shapefile(base, 15, [[square, hole], [far]], [("NAME", "C", 12, 0), ("AREA", "N", 10, 2)],
                     [("slump A", 1.5), ("slump B", 2.25)], utm_prj)
    return base.with_suffix(".shp")


def test_polygonz_shapefile_reads_features_parts_attributes_and_crs(polys):
    layer = read_shapefile(polys)
    assert layer.name == "anomaly_slumps" and layer.kind == "polygon"
    assert len(layer.features) == 2
    assert [len(f.parts) for f in layer.features] == [2, 1]
    assert layer.features[0].parts[0].shape == (5, 2) and layer.features[0].parts[0].dtype == np.float64
    assert layer.features[0].attrs == {"NAME": "slump A", "AREA": 1.5}
    assert layer.features[1].attrs["AREA"] == 2.25
    assert "32750" in layer.crs or "UTM zone 50S" in layer.crs
    assert layer.bounds == (500000.0, 7600000.0, 511000.0, 7611000.0)


def test_points_and_polylines_read_with_their_kinds(tmp_path, utm_prj):
    _write_shapefile(tmp_path / "plumes", 1, [[[(500500.0, 7600500.0)]], [[(500600.0, 7600700.0)]]],
                     [("ID", "N", 4, 0)], [(7,), (8,)], utm_prj)
    _write_shapefile(tmp_path / "channels", 13, [[[(500000.0, 7600000.0), (500300.0, 7600300.0), (500900.0, 7600100.0)]]],
                     [("ID", "N", 4, 0)], [(1,)], utm_prj)
    pts = read_shapefile(tmp_path / "plumes.shp"); lines = read_shapefile(tmp_path / "channels.shp")
    assert pts.kind == "point" and [f.parts[0].shape for f in pts.features] == [(1, 2), (1, 2)]
    assert pts.features[1].attrs == {"ID": 8}
    assert lines.kind == "polyline" and lines.features[0].parts[0].shape == (3, 2)


def test_to_lonlat_transforms_every_vertex_to_wgs84(polys):
    layer = read_shapefile(polys)
    ll = to_lonlat(layer)
    sq = ll.features[0].parts[0]
    assert sq.shape == (5, 2)
    assert 116.9 < sq[:, 0].min() < sq[:, 0].max() < 117.2 and -21.8 < sq[:, 1].min() < -21.6
    assert ll.crs == "EPSG:4326"


def test_to_field_pixels_lands_vertices_on_the_rasters_pixel_grid(polys):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from rasterio.crs import CRS
    # a 30 m raster whose top-left pixel centre is (499985, 7601015): the square's SW corner
    # (500000, 7600000) is 15 m east (column 0.5) and 1015 m south (row 33.833) of it
    x_axis = 499985.0 + 30.0 * np.arange(100); y_axis = 7601015.0 - 30.0 * np.arange(100)
    field = RasterField(name="dem", values=np.zeros((100, 100)), frame=LocalFrame(units="metre", dx=30.0, dy=30.0),
                        x_axis=x_axis, y_axis=y_axis, provenance={"crs": CRS.from_epsg(32750).to_string()})
    layer = read_shapefile(polys)
    px = to_field_pixels(layer, field)
    sq = px.features[0].parts[0]
    assert sq[0] == pytest.approx((0.5, 1015 / 30))           # (col, row)
    assert sq[2] == pytest.approx((1015 / 30, 0.5))           # NE corner
    assert px.crs == "pixel"


def test_to_field_pixels_reprojects_when_the_crs_differ(tmp_path, utm_prj):
    from dynamix.core.frames import GeographicFrame
    from dynamix.core.rasterfield import RasterField
    _write_shapefile(tmp_path / "pt", 1, [[[(500000.0, 7600000.0)]]], [("ID", "N", 4, 0)], [(1,)], utm_prj)
    layer = read_shapefile(tmp_path / "pt.shp")
    lon, lat = to_lonlat(layer).features[0].parts[0][0]
    x_axis = np.linspace(lon - 0.5, lon + 0.5, 101); y_axis = np.linspace(lat + 0.5, lat - 0.5, 101)
    field = RasterField(name="g", values=np.zeros((101, 101)), frame=GeographicFrame(), x_axis=x_axis, y_axis=y_axis,
                        provenance={"crs": "EPSG:4326"})
    px = to_field_pixels(layer, field)
    assert px.features[0].parts[0][0] == pytest.approx((50.0, 50.0), abs=0.01)


def test_the_real_boem_package_opens_if_present():
    # the repo's own (git-ignored) data/ folder -- present on the author's machine only
    d = (Path(__file__).resolve().parents[1] / "data"
         / "BOEM_Seafloor_Anomalies_Layer_Package_Aug_2019_to_June_2021")
    shp = next(iter(d.glob("**/anomaly_slumps.shp")), None)
    if shp is None:
        pytest.skip("BOEM package not on this machine")
    layer = read_shapefile(shp)
    assert layer.kind == "polygon" and len(layer.features) == 313
    assert "NAD_1927_BLM_Zone_16N" in layer.crs
    ll = to_lonlat(layer)
    assert -98 < ll.bounds[0] < ll.bounds[2] < -80 and 24 < ll.bounds[1] < ll.bounds[3] < 31


def test_to_crs_reprojects_into_a_named_crs_or_passes_through(polys):
    from dynamix.geo.vectors import to_crs
    layer = read_shapefile(polys)
    same = to_crs(layer, "EPSG:32750")
    assert np.allclose(same.features[0].parts[0], layer.features[0].parts[0]) and same.crs == "EPSG:32750"
    geo = to_crs(layer, "EPSG:4326")
    assert 116.9 < geo.features[0].parts[0][:, 0].min() < 117.2


def test_to_field_pixels_accepts_a_bare_array_field_as_a_pixel_frame(polys):
    """The shell still hands bare ndarrays around as fields (tests, plain-image loaders): pixel
    frame, no CRS, vertex coordinates taken as pixels."""
    layer = read_shapefile(polys)
    px = to_field_pixels(layer, np.zeros((20, 20)))
    assert px.features[0].parts[0][0] == pytest.approx((500000.0, 7600000.0))
