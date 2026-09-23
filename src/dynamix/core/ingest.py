# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Multi-sensor grid ingestion: multiband GeoTIFF, netCDF, HDF5, HDF4 (2026-09-21).

Driver reality on this machine (measured): rasterio's bundled GDAL reads
**netCDF** and **HDF5** (subdataset enumeration + georeferencing when present) but NOT HDF4 --
ASTER L1T (HDF-EOS2) goes through **pyhdf** instead. All three libraries are already in the
env; nothing new is pinned.

Axis/frame conventions MIRROR the frozen EQSelect ``rasterfield`` (never modified, never
imported for this): pixel-CENTER axes (``arange + 0.5`` through the geotransform), nodata ->
NaN, projected CRS -> ``LocalFrame`` with the true ground pixel size (WTMM scales stay metric
-- never reproject before analysis), geographic CRS -> ``GeographicFrame``, no/identity
transform -> pixel-index axes with a default ``LocalFrame`` and an honest provenance note.

GDAL's own orientation conventions are KEPT (netCDF commonly reads bottom-up relative to
the raw array -- GDAL's north-up reading; the pinned test records it). Multi-band sources
land as ``(ny, nx, nc)`` fields -- the shape the tensor path and the new
``pca``/``tucker_havok`` devices consume.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["probe", "load_grid", "multiband_count", "GRID_SUFFIXES"]


def multiband_count(path) -> int:
    """The band count of a directly-openable raster (0 when unreadable) -- the cheap probe
    ``opening.open_field`` uses to route multiband GeoTIFFs through :func:`load_grid`."""
    try:
        import rasterio

        with rasterio.open(str(path)) as src:
            return int(src.count)
    except Exception:
        return 0

#: Suffixes this module ingests (beyond the plain-GeoTIFF path ``opening.open_field`` owns).
GRID_SUFFIXES = (".nc", ".nc4", ".cdf", ".h5", ".hdf5", ".he5", ".hdf")


def probe(path) -> dict:
    """What is in this file: ``{"kind": "gdal"|"hdf4", "subdatasets": [(id, description)],
    "count": n_bands_when_directly_openable}``.

    A netCDF/HDF5 container with multiple variables lists them as GDAL subdataset ids (the
    strings :func:`load_grid` accepts); a file GDAL opens directly reports its band count and
    an empty subdataset list. HDF4 falls back to ``pyhdf`` and lists 2-D/3-D scientific
    datasets by name.
    """
    p = Path(path)
    try:
        import rasterio

        with rasterio.open(p) as src:
            subs = list(src.subdatasets)
            if subs:
                return {"kind": "gdal",
                        "subdatasets": [(s, s.split(":")[-1]) for s in subs],
                        "count": 0}
            return {"kind": "gdal", "subdatasets": [], "count": src.count}
    except Exception:
        pass
    if p.suffix.lower() == ".hdf":
        try:
            from pyhdf.SD import SD, SDC

            sd = SD(str(p), SDC.READ)
            subs = []
            for dsname, info in sorted(sd.datasets().items()):
                dims = info[1]
                if len(dims) in (2, 3):
                    subs.append((dsname, f"{dsname} {tuple(dims)}"))
            sd.end()
            return {"kind": "hdf4", "subdatasets": subs, "count": 0}
        except Exception as exc:
            raise ValueError(f"{p}: not readable by GDAL, and pyhdf failed too ({exc})") from exc
    raise ValueError(f"{p}: no reader could open this file (GDAL refused; not .hdf)")


def load_grid(path, *, subdataset: "str | None" = None, name: "str | None" = None):
    """One grid (2-D or ``(ny, nx, nc)``) from ``path`` as a :class:`RasterField`.

    ``subdataset``: a GDAL subdataset id from :func:`probe` (netCDF/HDF5 variables), or an
    HDF4 scientific-dataset NAME; ``None`` opens the file directly (plain rasters).
    """
    p = Path(path)
    if subdataset is not None and not str(subdataset).startswith(
            ("NETCDF:", "HDF5:", "HDF4")) and p.suffix.lower() == ".hdf":
        return _load_hdf4(p, str(subdataset), name)
    target = subdataset if subdataset is not None else str(p)
    return _load_gdal(p, str(target), name)


def _load_gdal(p: Path, target: str, name: "str | None"):
    import rasterio

    from dynamix.core.frames import GeographicFrame, LocalFrame
    from dynamix.core.rasterfield import RasterField

    with rasterio.open(target) as src:
        arr = np.asarray(src.read(), dtype=np.float64)         # (bands, ny, nx)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    values = arr[0] if arr.shape[0] == 1 else np.moveaxis(arr, 0, -1)
    ny, nx = values.shape[:2]
    georef = transform is not None and not transform.is_identity
    if georef:
        cols = np.arange(nx, dtype=np.float64) + 0.5           # pixel CENTRES (rasterfield)
        rows = np.arange(ny, dtype=np.float64) + 0.5
        x_axis, _ = transform * (cols, np.zeros_like(cols))
        _, y_axis = transform * (np.zeros_like(rows), rows)
        x_axis = np.asarray(x_axis, dtype=np.float64)
        y_axis = np.asarray(y_axis, dtype=np.float64)
        if crs is not None and crs.is_projected:
            frame = LocalFrame(x0=float(x_axis[0]), y0=float(y_axis[0]),
                               dx=abs(float(transform.a)), dy=abs(float(transform.e)),
                               units=crs.linear_units)
            units = crs.linear_units
        else:
            frame = GeographicFrame()
            units = ""
    else:
        x_axis = np.arange(nx, dtype=np.float64) + 0.5
        y_axis = np.arange(ny, dtype=np.float64) + 0.5
        frame = LocalFrame()
        units = ""
    sub_tail = target.split(":")[-1] if target != str(p) else None
    return RasterField(
        name=name or (f"{p.stem}:{sub_tail}" if sub_tail else p.stem),
        values=values, frame=frame, x_axis=x_axis, y_axis=y_axis, units=units,
        provenance={"source": str(p), "subdataset": (target if target != str(p) else None),
                    "crs": str(crs) if crs is not None else None,
                    **({} if georef else {"georef": "none (pixel-index axes)"})},
    )


def _load_hdf4(p: Path, dsname: str, name: "str | None"):
    """ASTER-class HDF-EOS2 via pyhdf: values + fill-value masking; pixel-index axes (the
    swath/geolocation solve is a recorded follow-up -- provenance says so honestly)."""
    from pyhdf.SD import SD, SDC

    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    sd = SD(str(p), SDC.READ)
    try:
        ds = sd.select(dsname)
        arr = np.asarray(ds.get(), dtype=np.float64)
        attrs = ds.attributes()
    finally:
        sd.end()
    fill = attrs.get("_FillValue")
    if fill is not None:
        arr = np.where(arr == fill, np.nan, arr)
    if arr.ndim == 3:
        # Heuristic, documented: the smallest axis is the band axis; move it last.
        band_ax = int(np.argmin(arr.shape))
        arr = np.moveaxis(arr, band_ax, -1)
    elif arr.ndim != 2:
        raise ValueError(f"{p}:{dsname}: expected a 2-D or 3-D dataset, got shape {arr.shape}")
    ny, nx = arr.shape[:2]
    return RasterField(
        name=name or f"{p.stem}:{dsname}",
        values=arr, frame=LocalFrame(),
        x_axis=np.arange(nx, dtype=np.float64) + 0.5,
        y_axis=np.arange(ny, dtype=np.float64) + 0.5,
        provenance={"source": str(p), "subdataset": dsname,
                    "georef": "none (HDF4 v1: pixel-index axes; geolocation follow-up)"},
    )
