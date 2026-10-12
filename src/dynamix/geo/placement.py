# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Where a field's image sits in map coordinates, display-only reprojection, and the frame
family that decides whether two native-frame rasters place together. Display only: analysis
never reads anything made here. ``rasterio`` is imported lazily."""
from __future__ import annotations

import numpy as np


def field_crs(field) -> "str | None":
    crs = (getattr(field, "provenance", None) or {}).get("crs")
    return str(crs) if crs else None


def image_transform(field, s: int = 1):
    """``(x0, y0, sx, sy)``: the map coordinate of item pixel (0, 0)'s corner and the step of
    one drawn pixel at decimation ``s``. A display picture registers on its native file grid."""
    from dynamix.roi.picture import display_stride, file_pixel_grid

    stride = display_stride(field)
    grid = file_pixel_grid(field) if stride > 1 else None
    if grid is not None:
        x0, dx, y0, dy, _nx, _ny = grid
        return x0 - 0.5 * dx, y0 - 0.5 * dy, dx * stride * s, dy * stride * s
    x = np.asarray(field.x_axis, dtype=np.float64)
    y = np.asarray(field.y_axis, dtype=np.float64)
    dx = (x[-1] - x[0]) / (x.size - 1) if x.size > 1 else 1.0
    dy = (y[-1] - y[0]) / (y.size - 1) if y.size > 1 else 1.0
    return x[0] - 0.5 * dx, y[0] - 0.5 * dy, dx * s, dy * s


def reproject_for_display(values, field, dst_crs: str, s: int = 1, max_dim: int = 1024):
    """``values`` (2-D, ``field``'s grid decimated by ``s``) reprojected into ``dst_crs`` at
    most ``max_dim`` pixels on the long side, nearest neighbour; NaN outside."""
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    v = np.asarray(values, dtype=np.float64)
    x0, y0, sx, sy = image_transform(field, s)
    ny, nx = v.shape
    src = Affine(sx, 0.0, x0, 0.0, sy, y0)
    xs, ys = (x0, x0 + nx * sx), (y0, y0 + ny * sy)
    dst, w, h = calculate_default_transform(field_crs(field), dst_crs, nx, ny,
                                            min(xs), min(ys), max(xs), max(ys))
    k = max(w, h) / max_dim
    if k > 1:
        dst = dst * Affine.scale(k)
        w, h = int(np.ceil(w / k)), int(np.ceil(h / k))
    out = np.full((h, w), np.nan)
    reproject(v, out, src_transform=src, src_crs=field_crs(field), dst_transform=dst,
              dst_crs=dst_crs, resampling=Resampling.nearest, src_nodata=np.nan,
              dst_nodata=np.nan)
    return out, (dst.c, dst.f, dst.a, dst.e)


def same_frame_family(a, b) -> bool:
    """Same frame type, units and pixel size; the anchors may differ (a fork is a slice of its
    parent's grid and places by its own axes)."""
    from dynamix.core.frames import LocalFrame

    if type(a) is not type(b):
        return False
    if isinstance(a, LocalFrame):
        return (a.units == b.units and np.isclose(a.dx, b.dx, rtol=1e-9)
                and np.isclose(a.dy, b.dy, rtol=1e-9))
    return True
