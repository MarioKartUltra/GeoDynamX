# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The display-only whole-extent PICTURE of a raster too big to load.

Retires the decimated-overview DATASET as the default. The overview
was a second coordinate system -- its pixel indices were decimated -- and every seam that
touched it (pin-in-place, the child mapping, both ``x ov`` ROI mappings, the image rect) had
to translate, and each translation was a registration bug. The picture is not a coordinate
system at all:

- sample ``[k, j]`` is the file pixel at the CENTRE of native block ``[k*s, (k+1)*s) x
  [j*s, (j+1)*s)`` (clamped for the partial last block) -- we choose the pixel, GDAL's
  resampling never does, so what the picture shows is exactly knowable;
- the 2-D canvas draws sample k over exactly that block, in FILE pixels
  (``provenance["display_stride"]`` is read by the image draw and nothing else);
- the axes are the file transform at those same centre pixels, so the Vector view drapes it
  on the same ground the canvas block covers;
- the frame carries the NATIVE pixel size, so ``px_to_metres`` and the scale bar read file
  pixels, the unit every gesture now speaks;
- no tool runs on it (``engine.resolve`` refuses): analysis always reads native pixels, through
  an ROI.

Headless: numpy at module level, rasterio lazily (nothing here imports Qt).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["PICTURE_MAX_DIM", "block_centres", "display_stride", "native_shape",
           "file_pixel_grid", "picture_stride", "read_picture", "read_picture_hdf4",
           "sample_spacing"]

#: Long side of the picture, in samples. The canvas LOD shows at most 2048 per axis and the
#: Vector drape budgets ~2M points, so more would be read and never seen; BOEM West reads in
#: ~1.9 s at this size (measured with exact strided reads of its 64x128 tiles).
PICTURE_MAX_DIM = 4096


def picture_stride(height: int, width: int, max_dim: int = PICTURE_MAX_DIM) -> int:
    """Native pixels per picture sample: ``ceil(max(h, w) / max_dim)``, at least 1."""
    return max(1, -(-max(int(height), int(width)) // int(max_dim)))


def block_centres(n: int, s: int) -> np.ndarray:
    """The native index each of the ``ceil(n/s)`` samples reads: ``min(k*s + s//2, n-1)``."""
    k = np.arange(-(-int(n) // int(s)), dtype=np.int64)
    return np.minimum(k * int(s) + int(s) // 2, int(n) - 1)


def display_stride(field) -> int:
    """Native pixels per stored sample of a display picture; 1 for everything else."""
    prov = getattr(field, "provenance", None) or {}
    return int(prov.get("display_stride", 1) or 1)


def native_shape(field) -> tuple[int, int]:
    """``(rows, cols)`` in FILE pixels: ``full_dims`` for a picture, else the array's own."""
    if display_stride(field) > 1:
        h, w = field.provenance["full_dims"]
        return int(h), int(w)
    values = np.asarray(getattr(field, "values", field))
    return int(values.shape[0]), int(values.shape[1])


def file_pixel_grid(field):
    """``(x0, dx, y0, dy, nx, ny)`` of the FILE-pixel grid a display picture is drawn on --
    the pixel-centre coordinate of file column/row 0, the native step (signed, as the axes
    run) and the file's dims -- or ``None`` for an ordinary field (whose own axes ARE its
    pixel grid). Sample k sits at file pixel k*s + s//2, so the native step is the first two
    samples' spacing / s (never the endpoints: a partial last block's centre is clamped)."""
    s = display_stride(field)
    if s <= 1:
        return None
    xa = np.asarray(field.x_axis, dtype=np.float64)
    ya = np.asarray(field.y_axis, dtype=np.float64)
    frame = getattr(field, "frame", None)
    dx = (xa[1] - xa[0]) / s if xa.size > 1 else float(getattr(frame, "dx", 1.0) or 1.0)
    dy = (ya[1] - ya[0]) / s if ya.size > 1 else float(getattr(frame, "dy", 1.0) or 1.0)
    h, w = field.provenance["full_dims"]
    return (float(xa[0] - (s // 2) * dx), float(dx), float(ya[0] - (s // 2) * dy), float(dy),
            int(w), int(h))


def sample_spacing(field, view_stride: int = 1) -> tuple[float, float]:
    """Ground distance ``(dx, dy)`` between the samples a view actually draws: the frame's
    NATIVE pixel size x the picture stride x the view's own decimation. Hillshade needs it --
    a picture's samples are ``display_stride`` native pixels apart, and using the bare frame
    size would steepen every slope by that factor."""
    frame = getattr(field, "frame", None)
    total = int(view_stride) * display_stride(field)
    return (abs(float(getattr(frame, "dx", 1.0) or 1.0)) * total,
            abs(float(getattr(frame, "dy", 1.0) or 1.0)) * total)


def read_picture(path, *, max_dim: int = PICTURE_MAX_DIM, name: str | None = None):
    """Read the display picture of ``path`` (anything rasterio opens; band 1).

    Rows are gathered per BLOCK ROW of the file (``block_shapes``), so a tiled file is read
    once, in order, with no GDAL resampling anywhere. Declared nodata and the BOEM float32
    sentinel (``|v| >= 3e38``, the ``from_geotiff_window`` rule) become NaN.
    """
    import rasterio
    from rasterio.windows import Window

    from dynamix.core.frames import GeographicFrame, LocalFrame
    from dynamix.core.rasterfield import RasterField

    path = str(path)
    with rasterio.open(path) as src:
        H, W = int(src.height), int(src.width)
        s = picture_stride(H, W, max_dim)
        rows, cols = block_centres(H, s), block_centres(W, s)
        bh = max(1, int(src.block_shapes[0][0]))
        out = np.empty((rows.size, cols.size), dtype=np.float64)
        blocks = rows // bh
        for b in np.unique(blocks):
            sel = np.nonzero(blocks == b)[0]
            r0 = int(b) * bh
            r1 = min(H, r0 + bh)
            band = src.read(1, window=Window(0, r0, W, r1 - r0))
            out[sel] = band[rows[sel] - r0][:, cols]
        transform, crs, nodata = src.transform, src.crs, src.nodata
    if nodata is not None:
        out = np.where(out == nodata, np.nan, out)
    out = np.where(np.abs(out) >= 3e38, np.nan, out)
    x_axis, _ = transform * (cols + 0.5, np.zeros(cols.size))
    _, y_axis = transform * (np.zeros(rows.size), rows + 0.5)
    x_axis = np.asarray(x_axis, dtype=np.float64)
    y_axis = np.asarray(y_axis, dtype=np.float64)
    if crs is not None and crs.is_projected:
        frame = LocalFrame(x0=float(x_axis[0]), y0=float(y_axis[0]),
                           dx=abs(float(transform.a)), dy=abs(float(transform.e)),
                           units=crs.linear_units)
    elif crs is not None:
        frame = GeographicFrame()
    else:
        # No georeference (a bare netCDF/HDF5 grid): pixel-index axes in a local frame --
        # ``ingest._load_gdal``'s own convention, never a geographic claim.
        frame = LocalFrame()
    field = RasterField(name=name or f"{Path(path).stem}@pic{s}", values=out, frame=frame,
                        x_axis=x_axis, y_axis=y_axis)
    field.provenance.update({
        "source": path, "full_dims": (H, W), "crs": str(crs) if crs is not None else None,
        "window": {"row_off": 0, "col_off": 0}, "display_stride": s,
    })
    return field


def read_picture_hdf4(path, dsname: str, *, max_dim: int = PICTURE_MAX_DIM,
                      name: str | None = None):
    """The display picture of one 2-D HDF4 scientific dataset (pyhdf; ASTER-class files GDAL
    cannot open): the same block-centre samples, pixel-index axes in a local frame (the
    ``ingest._load_hdf4`` convention), ``_FillValue`` -> NaN. ``provenance["reader"] =
    "hdf4"`` routes the ROI runner's native window reads back through pyhdf."""
    from pyhdf.SD import SD, SDC

    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    from dynamix.core.ingest import hdf4_select

    sd = SD(str(path), SDC.READ)
    try:
        ds = hdf4_select(path, sd, dsname)
        _n, rank, dims, _t, _na = ds.info()
        if rank != 2:
            raise ValueError(f"{path}:{dsname}: a picture needs a 2-D dataset, got rank {rank}")
        H, W = int(dims[0]), int(dims[1])
        s = picture_stride(H, W, max_dim)
        rows, cols = block_centres(H, s), block_centres(W, s)
        out = np.empty((rows.size, cols.size), dtype=np.float64)
        for k, r in enumerate(rows):
            out[k] = np.asarray(ds.get(start=(int(r), 0), count=(1, W)),
                                dtype=np.float64)[0][cols]
        fill = ds.attributes().get("_FillValue")
    finally:
        sd.end()
    if fill is not None:
        out = np.where(out == fill, np.nan, out)
    field = RasterField(name=name or f"{Path(str(path)).stem}:{dsname}@pic{s}", values=out,
                        frame=LocalFrame(), x_axis=cols.astype(np.float64) + 0.5,
                        y_axis=rows.astype(np.float64) + 0.5)
    field.provenance.update({
        "source": str(path), "subdataset": dsname, "reader": "hdf4", "full_dims": (H, W),
        "window": {"row_off": 0, "col_off": 0}, "display_stride": s,
        "georef": "none (HDF4 v1: pixel-index axes; geolocation follow-up)",
    })
    return field
