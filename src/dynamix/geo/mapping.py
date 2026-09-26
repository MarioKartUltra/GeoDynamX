# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Pixel -> lon/lat mapping for the arrangement view's draping.

Pure numpy + stdlib at module scope; ``rasterio`` (needed only for the actual CRS transform) is
imported LAZILY inside the functions that need it, matching :mod:`dynamix.core.rasterfield`'s own
convention -- the rest of ``dynamix.geo`` stays importable with pure-Python wheels. **No Qt, no
``dynamix.shell`` import, ever**: this module is consumed by both the headless test suite and
(eventually) ``dynamix/shell/arrangement/``'s scene manager, and the project's law is that nothing
outside ``dynamix/shell/`` may import PySide6/pyvista/pyqtgraph. ``tests/test_shell_boundaries.py
::test_devloop_is_the_only_shell_importer_outside_shell`` already AST-scans every module under
``src/dynamix`` (outside ``dynamix/shell/``) for those imports -- it walks the whole tree rather
than a hard-coded package list, so this module is covered automatically with nothing to extend
there; ``tests/test_geo_mapping.py`` also asserts it locally.

**Provenance schema consumed (bound against real code, not assumed):**
:class:`~dynamix.core.rasterfield.RasterField.from_geotiff_window` (and the whole-file open path
via :func:`dynamix.shell.opening.open_field`) stamp ``provenance["crs"]`` with ``str(rasterio's
CRS)`` -- the full WKT when the source CRS has no exact EPSG match (BOEM's custom NAD27 grids),
a bare ``"EPSG:<code>"`` when it does, or the key is absent/``None`` for a non-georeferenced field
(a plain image, an EBSD ``.npz``, a bare array). Neither loader stores the source ``rasterio``
Affine transform anywhere in ``provenance`` -- it is not JSON-safe and nothing needs it as an
object, since both loaders already convert it into ``x_axis``/``y_axis``: absolute pixel-CENTER
coordinates in the CRS's own units (``dynamix/shell/units.py``'s ``_native_px_size`` reads pixel
size the same way, off the axis arrays, for the same reason). This module follows that same
source of truth rather than trying to reconstruct or re-read a transform: pixel-center coordinates
come straight from ``field.x_axis``/``field.y_axis``, then :func:`rasterio.warp.transform` carries
them from the field's own CRS to ``EPSG:4326``.

**Sentinel masking:** a value is missing if it is already ``NaN`` (both loaders already convert a
correctly DECLARED nodata value to ``NaN`` at load time, so that case needs no extra handling
here -- ``NaN`` stays ``NaN`` through a copy) OR ``abs(v) >= 3e38`` -- the float32-max-magnitude
sentinel a WRONGLY declared nodata value can miss ("Live stretch reproducer + nodata-sentinel trap": BOEM East declares nodata 0.0 but actually fills
with -FLT_MAX, so the loader's own nodata check does not catch it). Both conditions are masked to
``NaN`` in the RETURNED array; ``field.values`` itself is never mutated.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["NoGeoreference", "field_lonlat_grid", "has_georeference", "lonlat_to_pixels", "points_lonlat"]

#: Anything at or above this magnitude is a float32-max-style fill sentinel, never real data
#: (e.g. BOEM East).
_SENTINEL_ABS = 3e38


class NoGeoreference(Exception):
    """Raised by :func:`field_lonlat_grid`/:func:`points_lonlat` for a field whose provenance
    carries no CRS. Callers (the arrangement scene manager) catch this to list the layer as
    "no georeference -- session view only" rather than crash or fabricate a placement ("Non-georeferenced layers")."""


def _field_crs(field):
    """``provenance["crs"]`` if present and non-empty/non-``None``, else ``None``."""
    provenance = getattr(field, "provenance", None) or {}
    crs_text = provenance.get("crs")
    return crs_text if crs_text else None


def _require_crs(field):
    crs_text = _field_crs(field)
    if crs_text is None:
        raise NoGeoreference(
            f"{getattr(field, 'name', field)!r} has no CRS in provenance; cannot place it on a "
            f"geography"
        )
    return crs_text


def has_georeference(field) -> bool:
    """Cheap, non-raising check: does ``field``'s provenance carry a usable CRS?

    Mirrors exactly what :func:`_require_crs` gates on (the ``field_lonlat_grid``/
    ``points_lonlat`` call it guards), so a caller that gets ``True`` here is guaranteed those
    functions will not raise :class:`NoGeoreference` for the same field -- e.g. the arrangement's
    multi-layer resolve (``MainWindow._sync_arrangement``), which needs to CLASSIFY a layer as
    "no-georeference" up front, before deciding whether a resolve/mapping call is even worth
    attempting, rather than mapping-and-catching for every visible layer on every sync.
    """
    return _field_crs(field) is not None


def _to_lonlat(crs_text, x, y):
    """``(lon, lat)`` float64 arrays for CRS-native ``x``/``y`` arrays, via a LAZILY-imported
    ``rasterio`` -- the only third-party dependency this module ever touches."""
    try:
        from rasterio.crs import CRS
        from rasterio.warp import transform as warp_transform
    except ImportError as exc:
        raise ValueError(f"lon/lat mapping needs rasterio: pip install rasterio ({exc})") from exc
    crs = CRS.from_user_input(crs_text)
    lon, lat = warp_transform(crs, "EPSG:4326", x, y)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _mask_sentinels(values) -> np.ndarray:
    """A float64 COPY of ``values`` with ``|v| >= 3e38`` set to ``NaN``. Pre-existing ``NaN``
    (already-masked declared nodata) passes through unchanged -- ``NaN`` compares false against
    everything, so the sentinel check never disturbs it."""
    out = np.asarray(values, dtype=np.float64).copy()
    return np.where(np.abs(out) >= _SENTINEL_ABS, np.nan, out)


def _stride_for(shape, max_points) -> int:
    """A small integer stride ``s >= 1`` with ``(ny // s) * (nx // s) <= max_points`` (the
    starting guess is only walked UPWARD from there -- for a very elongated ``shape`` this can
    overshoot the true minimal ``s``, so "small", not a proven minimum).

    Geo-side duplicate of the shell's display-side ``dynamix.shell.canvas.lod_stride``: that
    function bounds the LARGER dimension to a pixel count (display decimation for on-screen
    painting), this one bounds the total POINT count (a draping budget for the arrangement
    scene) -- different arithmetic for a different consumer, not a port of the same function.
    ``dynamix.geo`` cannot import ``dynamix.shell.canvas`` (Qt-free law), so this small integer
    computation is duplicated across that boundary rather than shared; accepted per the design.
    """
    ny, nx = int(shape[0]), int(shape[1])
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")
    n = ny * nx
    if n <= max_points:
        return 1
    s = max(1, math.isqrt(-(-n // max_points)))    # ceil(n / max_points), as a starting guess
    while (ny // s) * (nx // s) > max_points:
        s += 1
    return s


def field_lonlat_grid(field, max_points: int = 2_000_000):
    """Decimated ``(lon2d, lat2d, values2d, stride)`` for draping ``field`` on the arrangement's
    geography.

    ``lon2d``/``lat2d``/``values2d`` share shape ``(ny // stride, nx // stride)``, C-order, one
    entry per DECIMATED pixel center -- ``stride`` from :func:`_stride_for` so the point count
    never exceeds ``max_points`` (the spec's "~2M points per field" draping budget).
    ``values2d`` is a float64 COPY (see module docstring for the sentinel-masking rule);
    ``field.values`` is never mutated. Raises :class:`NoGeoreference` when ``field.provenance``
    carries no CRS.
    """
    crs_text = _require_crs(field)
    stride = _stride_for((field.ny, field.nx), max_points)
    # Exactly ny // stride / nx // stride samples -- not ceil(ny / stride) (what a plain
    # np.arange(0, ny, stride) would give) -- so the returned grid's point count matches the
    # (ny // s) * (nx // s) <= max_points budget _stride_for actually computed, never exceeding it.
    rows = np.arange(field.ny // stride, dtype=np.intp) * stride
    cols = np.arange(field.nx // stride, dtype=np.intp) * stride

    x, y = np.meshgrid(field.x_axis[cols], field.y_axis[rows])   # each (len(rows), len(cols))
    lon, lat = _to_lonlat(crs_text, x.ravel(), y.ravel())
    lon2d = lon.reshape(x.shape)
    lat2d = lat.reshape(x.shape)

    values2d = _mask_sentinels(field.values[np.ix_(rows, cols)])
    return lon2d, lat2d, values2d, stride


def points_lonlat(field, cols, rows, *, sub=None):
    """``(lon, lat)`` float64 arrays for the pixel centers at ``cols``/``rows`` (parallel integer
    arrays, vectorized) -- the same pixel-center source (``field.x_axis``/``field.y_axis``) and
    CRS transform as :func:`field_lonlat_grid`, so picking a pixel out of a decimated grid (or
    off any full-res list of extrema/chain vertices) lands at the identical lon/lat. Raises
    :class:`NoGeoreference` when ``field.provenance`` carries no CRS.

    ``sub``: optional ``(cols_f, rows_f)`` FRACTIONAL pixel positions (a maximum's subpixel
    refinement) placed linearly along the axes instead of the integer centers.
    """
    crs_text = _require_crs(field)
    if sub is not None:
        x = axis_at(field.x_axis, sub[0])
        y = axis_at(field.y_axis, sub[1])
        return _to_lonlat(crs_text, x, y)
    cols = np.asarray(cols, dtype=np.intp)
    rows = np.asarray(rows, dtype=np.intp)
    x = field.x_axis[cols]
    y = field.y_axis[rows]
    return _to_lonlat(crs_text, x, y)


def axis_at(axis, positions) -> np.ndarray:
    """Coordinates at FRACTIONAL pixel indices along ``axis`` (linear between pixel centers,
    extended linearly past the ends -- the regular-axis RasterField contract)."""
    axis = np.asarray(axis, dtype=np.float64)
    p = np.asarray(positions, dtype=np.float64)
    if axis.size < 2:
        return np.full(p.shape, float(axis[0]) if axis.size else 0.0)
    return axis[0] + p * ((axis[-1] - axis[0]) / (axis.size - 1))


def lonlat_to_pixels(field, lon, lat):
    """WGS84 lon/lat arrays -> fractional (cols, rows) on ``field``'s native grid.

    The exact inverse of :func:`points_lonlat`'s pipeline: one batched
    ``rasterio.warp.transform("EPSG:4326", crs, lon, lat)`` into the field's CRS, then the
    axis-derived inverse of the pixel-center mapping -- x_axis/y_axis are affine by
    construction (``from_geotiff_window`` builds them from the file transform), so
    ``col = (x - x_axis[0]) / dx`` with ``dx = (x_axis[-1] - x_axis[0]) / (n - 1)``.

    Raises :class:`NoGeoreference` when the field has no CRS. Returns float64 arrays;
    callers decide clipping/rounding -- a point off the raster is DATA (it lands outside
    [0, n)), not an error.
    """
    crs_text = _require_crs(field)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)

    # Lazy import: same pattern as _to_lonlat
    try:
        from rasterio.crs import CRS
        from rasterio.warp import transform as warp_transform
    except ImportError as exc:
        raise ValueError(f"lon/lat -> pixel mapping needs rasterio: pip install rasterio ({exc})") from exc

    # Transform from EPSG:4326 (WGS84) to the field's CRS
    crs = CRS.from_user_input(crs_text)
    x, y = warp_transform("EPSG:4326", crs, lon, lat)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Axis-derived inverse: pixel = (coord - coord_axis[0]) / delta_coord
    # where delta_coord = (coord_axis[-1] - coord_axis[0]) / (n - 1)
    # Guard against degenerate axes (n == 1)
    nx = field.nx
    ny = field.ny

    if nx == 1 or ny == 1:
        raise NoGeoreference(
            f"{getattr(field, 'name', field)!r} has degenerate axis (nx={nx}, ny={ny}); "
            f"cannot map to pixels"
        )

    # Compute pixel spacing from axis endpoints
    dx = (field.x_axis[-1] - field.x_axis[0]) / (nx - 1)
    dy = (field.y_axis[-1] - field.y_axis[0]) / (ny - 1)
    x0, y0 = field.x_axis[0], field.y_axis[0]
    # A display PICTURE answers in FILE pixels, the canvas's own space.
    from dynamix.roi.picture import file_pixel_grid

    grid = file_pixel_grid(field)
    if grid is not None:
        x0, dx, y0, dy = grid[:4]

    # Invert the linear transformation
    cols = (x - x0) / dx
    rows = (y - y0) / dy

    return np.asarray(cols, dtype=np.float64), np.asarray(rows, dtype=np.float64)
