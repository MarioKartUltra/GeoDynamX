# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""How a guide's grid relates to the target's: *same* (a shift), *nested* (a block average
or an exact aggregation), *foreign* (a warp built from two georeferences or two lattices).

A georeference here is a plain dict: ``crs``, the pixel-centre origin ``x0, y0`` and steps
``dx, dy`` (``dy`` signed as the y axis runs), ``shape``, an optional ``lattice`` reference
(an ASTER L1A band: the file and the SDS names of its Latitude, Longitude and LatticePoint
arrays, read lazily) and ``source``. The target is never resampled: a warp samples the guide
inside its own measurement.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = ["GridRelation", "georef", "georef_from", "lonlat_of", "pixels_of", "relate",
           "warp_rows_cols"]

NESTED_RATIOS = (2, 3, 4, 6, 8)


def georef_from(crs, x0, dx, y0, dy, shape, *, lattice=None, source=None,
                georeferenced=None, full_shape=None) -> dict:
    """``full_shape`` is the native raster's shape when the georef describes a window of it
    (an ROI processing window, a send read over a window); it decides *same* between two
    unreferenced rasters of one file."""
    if georeferenced is None:
        georeferenced = crs is not None and lattice is None
    shape = (int(shape[0]), int(shape[1]))
    return {"crs": None if crs is None else str(crs), "x0": float(x0), "dx": float(dx),
            "y0": float(y0), "dy": float(dy), "shape": shape, "lattice": lattice,
            "source": source, "georeferenced": bool(georeferenced),
            "full_shape": shape if full_shape is None else (int(full_shape[0]), int(full_shape[1]))}


def georef(field) -> dict:
    from dynamix.geo.mapping import has_georeference

    prov = getattr(field, "provenance", None) or {}
    x, y = np.asarray(field.x_axis, dtype=np.float64), np.asarray(field.y_axis, dtype=np.float64)
    dx = float(x[1] - x[0]) if x.size > 1 else float(getattr(field.frame, "dx", 1.0) or 1.0)
    dy = float(y[1] - y[0]) if y.size > 1 else float(getattr(field.frame, "dy", 1.0) or 1.0)
    lattice = prov.get("lattice")
    return georef_from(prov.get("crs"), x[0], dx, y[0], dy, np.shape(field.values)[:2],
                       lattice=lattice, source=prov.get("source"),
                       georeferenced=lattice is None and has_georeference(field),
                       full_shape=prov.get("full_dims"))


@dataclasses.dataclass(frozen=True)
class GridRelation:
    kind: str
    ratio: int
    direction: str
    offset: tuple
    subpixel: tuple
    mapping_source: str

    def stamp(self) -> dict:
        return {"kind": self.kind, "ratio": int(self.ratio), "direction": self.direction,
                "offset": [int(v) for v in self.offset],
                "subpixel": [float(v) for v in self.subpixel],
                "mapping_source": self.mapping_source}


def _foreign(target, guide) -> GridRelation:
    if target.get("lattice") is not None or guide.get("lattice") is not None:
        source = "lattice"
    elif target["georeferenced"] and guide["georeferenced"]:
        source = "crs"
    else:
        source = "none"
    return GridRelation("foreign", 1, "equal", (0, 0), (0.0, 0.0), source)


def _split(v: float, tol: float):
    """``(integer, residual)`` when ``v`` is within ``tol`` of an integer, else ``None``."""
    k = int(round(v))
    return (k, v - k) if abs(v - k) <= tol else None


def _lattice_group(g: dict):
    lat = g.get("lattice") or {}
    return lat.get("group") or (lat.get("sds") or {}).get("lat")


def relate(target: dict, guide: dict, tol_px: float = 0.25) -> GridRelation:
    unreferenced = (not target["georeferenced"] or not guide["georeferenced"])
    if unreferenced:
        # Two windows of ONE band of one file share a grid; two bands of one ASTER file do
        # not (VNIR 4200 x 4100, SWIR 2100 x 2048, TIR 700 x 700, and even VNIR bands are
        # up to a pixel apart through their lattices), so they relate through the lattices.
        same_file = bool(target.get("source")) and target.get("source") == guide.get("source")
        same_band = (same_file and target["full_shape"] == guide["full_shape"]
                     and _lattice_group(target) == _lattice_group(guide))
        if same_band:
            oy = (guide["y0"] - target["y0"]) / target["dy"]
            ox = (guide["x0"] - target["x0"]) / target["dx"]
            return GridRelation("same", 1, "equal", (int(round(oy)), int(round(ox))),
                                (0.0, 0.0), "file")
        return _foreign(target, guide)
    if target["crs"] != guide["crs"]:
        return _foreign(target, guide)
    rx, ry = guide["dx"] / target["dx"], guide["dy"] / target["dy"]
    if abs(rx - ry) > 1e-6 * abs(rx) or rx <= 0:
        return _foreign(target, guide)
    oy = (guide["y0"] - target["y0"]) / target["dy"]
    ox = (guide["x0"] - target["x0"]) / target["dx"]
    if abs(rx - 1.0) <= 1e-6:
        sy, sx = _split(oy, tol_px), _split(ox, tol_px)
        if sy is None or sx is None:
            return _foreign(target, guide)
        return GridRelation("same", 1, "equal", (sy[0], sx[0]), (sy[1], sx[1]), "crs")
    coarser = rx > 1.0
    f = rx if coarser else 1.0 / rx
    fi = int(round(f))
    if fi not in NESTED_RATIOS or abs(f - fi) > 1e-6 * f:
        return _foreign(target, guide)
    if coarser:
        # the guide's first centre must be a block centre: block top-left = o - (f - 1) / 2
        sy, sx = _split(oy - (fi - 1) / 2.0, tol_px), _split(ox - (fi - 1) / 2.0, tol_px)
        if sy is None or sx is None:
            return _foreign(target, guide)
        return GridRelation("nested", fi, "coarser", (sy[0], sx[0]), (sy[1], sx[1]), "crs")
    # finer guide: the target's first centre sits over a block of the guide
    gy = (target["y0"] - guide["y0"]) / guide["dy"] - (fi - 1) / 2.0
    gx = (target["x0"] - guide["x0"]) / guide["dx"] - (fi - 1) / 2.0
    sy, sx = _split(gy, tol_px), _split(gx, tol_px)
    if sy is None or sx is None:
        return _foreign(target, guide)
    return GridRelation("nested", fi, "finer", (sy[0], sx[0]), (sy[1], sx[1]), "crs")


# --- mappings --------------------------------------------------------------------------

def _xy(g: dict, rows, cols):
    return g["x0"] + np.asarray(cols, dtype=np.float64) * g["dx"], \
        g["y0"] + np.asarray(rows, dtype=np.float64) * g["dy"]


def _rc(g: dict, x, y):
    return (np.asarray(y, dtype=np.float64) - g["y0"]) / g["dy"], \
        (np.asarray(x, dtype=np.float64) - g["x0"]) / g["dx"]


_WGS84_E2 = 0.00669437999014


def _read_lattice(ref: dict):
    """The lattice's geodetic latitude, longitude and node pixel rows and columns. The ASTER
    L1A ``Latitude`` SDS is geocentric: converted here with ``atan(tan(phi) / (1 - e^2))``
    (WGS84), which reproduces the scene's own ``UPPERLEFT`` corner to 1e-3 degree where the
    raw value is 0.13 degree (14 km) off."""
    from pyhdf.SD import SD, SDC

    from dynamix.core.ingest import hdf4_select

    sds = ref.get("sds") or {}
    group = ref.get("group")
    names = {k: sds.get(k) or (f"{group}/{n}" if group else n)
             for k, n in (("lat", "Latitude"), ("lon", "Longitude"), ("points", "LatticePoint"))}
    sd = SD(str(ref["file"]), SDC.READ)
    try:
        lat = np.asarray(hdf4_select(ref["file"], sd, names["lat"]).get(), dtype=np.float64)
        lon = np.asarray(hdf4_select(ref["file"], sd, names["lon"]).get(), dtype=np.float64)
        lp = np.asarray(hdf4_select(ref["file"], sd, names["points"]).get(), dtype=np.float64)
    finally:
        sd.end()
    lat = np.degrees(np.arctan(np.tan(np.radians(lat)) / (1.0 - _WGS84_E2)))
    return lat, lon, lp[..., 1], lp[..., 0]                   # rows and cols of the nodes


def _file_rc(g: dict, rows, cols):
    """Window-local pixel coordinates on a lattice georef to the band's file indices: the
    georef's pixel-index axes carry the window's origin (``x0 = col_off + 0.5``)."""
    return ((g["y0"] - 0.5) + np.asarray(rows, dtype=np.float64) * g["dy"],
            (g["x0"] - 0.5) + np.asarray(cols, dtype=np.float64) * g["dx"])


def lonlat_of(g: dict, rows, cols):
    """Longitude and latitude of pixel coordinates on ``g`` (a CRS or a lattice)."""
    if g.get("lattice") is not None:
        from scipy.interpolate import RegularGridInterpolator

        lat, lon, node_r, node_c = _read_lattice(g["lattice"])
        axes = (node_r[:, 0], node_c[0, :])
        rr, cc = _file_rc(g, rows, cols)
        pts = np.column_stack([np.ravel(rr), np.ravel(cc)])
        f_lat = RegularGridInterpolator(axes, lat, bounds_error=False, fill_value=None)
        f_lon = RegularGridInterpolator(axes, lon, bounds_error=False, fill_value=None)
        return f_lon(pts).reshape(np.shape(rows)), f_lat(pts).reshape(np.shape(rows))
    from rasterio.crs import CRS
    from rasterio.warp import transform

    x, y = _xy(g, rows, cols)
    lon, lat = transform(CRS.from_user_input(g["crs"]), "EPSG:4326", np.ravel(x).tolist(),
                         np.ravel(y).tolist())
    return np.asarray(lon).reshape(np.shape(x)), np.asarray(lat).reshape(np.shape(y))


def pixels_of(g: dict, lon, lat):
    """``(rows, cols)`` on ``g`` of longitude and latitude (a CRS or a lattice)."""
    if g.get("lattice") is not None:
        from scipy.interpolate import LinearNDInterpolator

        la, lo, node_r, node_c = _read_lattice(g["lattice"])
        nodes = np.column_stack([lo.ravel(), la.ravel()])
        q = np.column_stack([np.ravel(lon), np.ravel(lat)])
        rows_f = LinearNDInterpolator(nodes, node_r.ravel())(q)
        cols_f = LinearNDInterpolator(nodes, node_c.ravel())(q)
        rows = (rows_f - (g["y0"] - 0.5)) / g["dy"]            # file indices to window-local
        cols = (cols_f - (g["x0"] - 0.5)) / g["dx"]
        return rows.reshape(np.shape(lon)), cols.reshape(np.shape(lon))
    from rasterio.crs import CRS
    from rasterio.warp import transform

    x, y = transform("EPSG:4326", CRS.from_user_input(g["crs"]), np.ravel(lon).tolist(),
                     np.ravel(lat).tolist())
    r, c = _rc(g, np.asarray(x), np.asarray(y))
    return r.reshape(np.shape(lon)), c.reshape(np.shape(lon))


def warp_rows_cols(src: dict, dst: dict):
    """For every pixel centre of ``src``, its ``(row, col)`` on ``dst``."""
    i, j = np.indices(src["shape"], dtype=np.float64)
    if (src.get("lattice") is None and dst.get("lattice") is None
            and src["crs"] == dst["crs"]):
        x, y = _xy(src, i, j)
        return _rc(dst, x, y)
    lon, lat = lonlat_of(src, i, j)
    return pixels_of(dst, lon, lat)
