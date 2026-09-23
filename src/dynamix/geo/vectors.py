# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""GIS vector files as reference layers — a stdlib shapefile reader and the two transforms the
views need.

BOEM's seafloor-anomaly package is 33 ESRI shapefiles (``polygonZ`` / ``polylineZ`` / ``point``)
in NAD27 BLM Zone 16N (US survey feet) -- the West bathymetry's own CRS. The environment has no
vector stack (no fiona / pyogrio / shapely), and the format is simple and frozen (ESRI, 1998),
so this reads it directly: ``.shp`` geometry (Z/M variants accepted, Z and M dropped), ``.dbf``
attributes (dBASE III), ``.prj`` CRS text, ``.cpg`` encoding when present. ``rasterio`` (already
a dependency for rasters) does the CRS transforms, imported lazily like everywhere in ``geo``.

A reference layer is read-only interpretation drawn OVER the data -- EQSelect's ``reflayers``
idea. Nothing here is analysis.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: shape type code -> (kind, has_z)
_SHAPE_KINDS = {
    0: ("null", False),
    1: ("point", False), 11: ("point", True), 21: ("point", False),
    3: ("polyline", False), 13: ("polyline", True), 23: ("polyline", False),
    5: ("polygon", False), 15: ("polygon", True), 25: ("polygon", False),
    8: ("multipoint", False), 18: ("multipoint", True), 28: ("multipoint", False),
}


@dataclass
class Feature:
    parts: list                      # list of (N, 2) float64 arrays (x, y) in the layer's CRS
    attrs: dict = field(default_factory=dict)


@dataclass
class VectorLayer:
    name: str
    kind: str                        # point | multipoint | polyline | polygon
    crs: str                         # CRS text: .prj WKT, "EPSG:4326" after to_lonlat, "pixel" after to_field_pixels
    features: list
    bounds: tuple                    # (xmin, ymin, xmax, ymax) in the layer's CRS
    path: str = ""
    field_names: list = field(default_factory=list)


# ------------------------------------------------------------------------------ .shp

def _read_shp(path: Path):
    data = path.read_bytes()
    if len(data) < 100 or struct.unpack(">i", data[:4])[0] != 9994:
        raise ValueError(f"{path}: not an ESRI shapefile")
    shape_type = struct.unpack("<i", data[32:36])[0]
    if shape_type not in _SHAPE_KINDS or shape_type == 0:
        raise ValueError(f"{path}: unsupported shape type {shape_type}")
    kind, _ = _SHAPE_KINDS[shape_type]
    xmin, ymin, xmax, ymax = struct.unpack("<4d", data[36:68])
    pos = 100
    features = []
    n = len(data)
    while pos + 8 <= n:
        _num, length_words = struct.unpack(">2i", data[pos:pos + 8])
        pos += 8
        rec = data[pos:pos + 2 * length_words]
        pos += 2 * length_words
        if len(rec) < 4:
            break
        rtype = struct.unpack("<i", rec[:4])[0]
        if rtype == 0:
            features.append([])                       # null shape: keeps record numbering
            continue
        rkind, _ = _SHAPE_KINDS.get(rtype, ("null", False))
        if rkind == "point":
            x, y = struct.unpack("<2d", rec[4:20])
            features.append([np.array([[x, y]], dtype=np.float64)])
        elif rkind == "multipoint":
            npts = struct.unpack("<i", rec[36:40])[0]
            pts = np.frombuffer(rec, dtype="<f8", count=2 * npts, offset=40).reshape(npts, 2)
            features.append([np.array(pts, dtype=np.float64)])
        else:                                            # polyline / polygon
            nparts, npts = struct.unpack("<2i", rec[36:44])
            parts_idx = np.frombuffer(rec, dtype="<i4", count=nparts, offset=44)
            off = 44 + 4 * nparts
            pts = np.frombuffer(rec, dtype="<f8", count=2 * npts, offset=off).reshape(npts, 2)
            bounds_idx = list(parts_idx) + [npts]
            features.append([np.array(pts[bounds_idx[i]:bounds_idx[i + 1]], dtype=np.float64)
                             for i in range(nparts)])
    return kind, (xmin, ymin, xmax, ymax), features


# ------------------------------------------------------------------------------ .dbf

def _read_dbf(path: Path, encoding: str):
    if not path.is_file():
        return [], []
    data = path.read_bytes()
    n_records = struct.unpack("<I", data[4:8])[0]
    header_len, record_len = struct.unpack("<2H", data[8:12])
    fields = []
    pos = 32
    while pos + 32 <= header_len and data[pos] != 0x0D:
        desc = data[pos:pos + 32]
        name = desc[:11].split(b"\x00", 1)[0].decode("ascii", "replace").strip()
        ftype = chr(desc[11])
        length, decimals = desc[16], desc[17]
        fields.append((name, ftype, length, decimals))
        pos += 32
    rows = []
    pos = header_len
    for _ in range(n_records):
        rec = data[pos:pos + record_len]
        pos += record_len
        if len(rec) < record_len:
            break
        row, off = {}, 1                                     # byte 0: deletion flag
        for name, ftype, length, decimals in fields:
            raw = rec[off:off + length]
            off += length
            txt = raw.decode(encoding, "replace").strip()
            if ftype in ("N", "F"):
                try:
                    val = int(txt) if (decimals == 0 and "." not in txt) else float(txt)
                except ValueError:
                    val = None
            elif ftype == "L":
                val = txt.upper() in ("T", "Y")
            else:
                val = txt
            row[name] = val
        if rec[:1] != b"*":                                  # deleted records are skipped
            rows.append(row)
    return [f[0] for f in fields], rows


# ------------------------------------------------------------------------------ public

def read_shapefile(path) -> VectorLayer:
    """One shapefile -> :class:`VectorLayer`. Geometry from ``.shp``, attributes from ``.dbf``
    (zipped by record order), CRS text from ``.prj`` (empty when absent), text encoding from
    ``.cpg`` (default latin-1, the dBASE convention)."""
    path = Path(path)
    kind, bounds, geoms = _read_shp(path)
    cpg = path.with_suffix(".cpg")
    if not cpg.is_file():
        cpg = path.with_suffix(".CPG")
    encoding = cpg.read_text(errors="replace").strip() if cpg.is_file() else "latin-1"
    try:
        "x".encode(encoding)
    except LookupError:
        encoding = "latin-1"
    names, rows = _read_dbf(path.with_suffix(".dbf"), encoding)
    prj = path.with_suffix(".prj")
    crs = prj.read_text(errors="replace").strip() if prj.is_file() else ""
    features = [Feature(parts=g, attrs=(rows[i] if i < len(rows) else {}))
                for i, g in enumerate(geoms) if g]
    return VectorLayer(name=path.stem, kind=kind, crs=crs, features=features,
                       bounds=tuple(float(b) for b in bounds), path=str(path.resolve()),
                       field_names=names)


def _transform_layer(layer: VectorLayer, fn, crs_label: str) -> VectorLayer:
    feats = []
    for f in layer.features:
        parts = []
        for part in f.parts:
            x, y = fn(part[:, 0], part[:, 1])
            parts.append(np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)]))
        feats.append(Feature(parts=parts, attrs=f.attrs))
    allx = np.concatenate([p[:, 0] for f in feats for p in f.parts]) if feats else np.array([0.0])
    ally = np.concatenate([p[:, 1] for f in feats for p in f.parts]) if feats else np.array([0.0])
    return VectorLayer(name=layer.name, kind=layer.kind, crs=crs_label, features=feats,
                       bounds=(float(allx.min()), float(ally.min()), float(allx.max()), float(ally.max())),
                       path=layer.path, field_names=layer.field_names)


def to_lonlat(layer: VectorLayer) -> VectorLayer:
    """The layer in WGS84 lon/lat (for the globe). A layer with no CRS is assumed lon/lat."""
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp_transform

    if not layer.crs:
        return _transform_layer(layer, lambda x, y: (x, y), "EPSG:4326")
    src = CRS.from_wkt(layer.crs) if layer.crs.strip().upper().startswith(("PROJCS", "GEOGCS", "COMPD_CS")) \
        else CRS.from_string(layer.crs)
    if src.to_epsg() == 4326:
        return _transform_layer(layer, lambda x, y: (x, y), "EPSG:4326")
    return _transform_layer(layer, lambda x, y: warp_transform(src, "EPSG:4326", list(x), list(y)), "EPSG:4326")


def to_field_pixels(layer: VectorLayer, field) -> VectorLayer:
    """The layer in ``field``'s PIXEL frame -- ``(col, row)`` in pixel-centre coordinates, the
    frame the 2-D canvas draws overlays in -- so an interpretation can sit on the raster it was
    made from. Reprojects to the field's CRS (``provenance["crs"]``) when the two differ; a field
    with no CRS is taken to share the layer's. Axes are assumed regular (the ``RasterField``
    contract), so a vertex maps by ``(x - x0) / dx`` against the pixel-centre axis."""
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp_transform

    if getattr(field, "x_axis", None) is None:            # a bare array: pixel frame, no CRS
        shape = np.asarray(getattr(field, "values", field)).shape
        x_axis = np.arange(shape[1], dtype=np.float64)
        y_axis = np.arange(shape[0], dtype=np.float64)
    else:
        x_axis = np.asarray(field.x_axis, dtype=np.float64)
        y_axis = np.asarray(field.y_axis, dtype=np.float64)
    dx = (x_axis[-1] - x_axis[0]) / max(len(x_axis) - 1, 1) if len(x_axis) > 1 else 1.0
    dy = (y_axis[-1] - y_axis[0]) / max(len(y_axis) - 1, 1) if len(y_axis) > 1 else 1.0
    x0, y0 = x_axis[0], y_axis[0]
    # A display PICTURE is drawn in FILE pixels: invert the file grid,
    # never the picture's own sample axes (s x coarser).
    from dynamix.roi.picture import file_pixel_grid

    grid = file_pixel_grid(field)
    if grid is not None:
        x0, dx, y0, dy = grid[:4]
    field_crs = (getattr(field, "provenance", {}) or {}).get("crs")
    fn_proj = None
    if layer.crs and field_crs:
        src = CRS.from_wkt(layer.crs) if layer.crs.strip().upper().startswith(("PROJCS", "GEOGCS", "COMPD_CS")) \
            else CRS.from_string(layer.crs)
        dst = CRS.from_string(str(field_crs))
        if src != dst:
            fn_proj = lambda x, y: warp_transform(src, dst, list(x), list(y))  # noqa: E731

    def fn(x, y):
        if fn_proj is not None:
            x, y = fn_proj(x, y)
            x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        col = (x - x0) / dx
        row = (y - y0) / dy
        return col, row

    return _transform_layer(layer, fn, "pixel")


def _crs_of(text: str):
    from rasterio.crs import CRS
    t = text.strip()
    return CRS.from_wkt(t) if t.upper().startswith(("PROJCS", "GEOGCS", "COMPD_CS", "PROJCRS", "GEOGCRS")) \
        else CRS.from_string(t)


def to_crs(layer: VectorLayer, crs: str) -> VectorLayer:
    """The layer in another CRS (given as EPSG/proj/WKT text) -- the Vector view's native frame
    is the active field's CRS, so a reference layer is placed there in the field's own units."""
    from rasterio.warp import transform as warp_transform

    dst = _crs_of(str(crs))
    if not layer.crs:
        return _transform_layer(layer, lambda x, y: (x, y), str(crs))
    src = _crs_of(layer.crs)
    if src == dst:
        return _transform_layer(layer, lambda x, y: (x, y), str(crs))
    return _transform_layer(layer, lambda x, y: warp_transform(src, dst, list(x), list(y)), str(crs))
