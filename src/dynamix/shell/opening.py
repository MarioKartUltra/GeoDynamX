# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Opening rasters for the shell: GeoTIFF and zip-of-GeoTIFF, without ever extracting to disk.

BOEM's Gulf bathymetry ships as GeoTIFFs zipped to roughly 15% of their uncompressed size (10.6 GB
combined for the two Gulf halves) plus ``.ovr``/``.tfw``/``.aux.xml`` sidecars. Extracting either
zip before opening it would put that 10.6 GB on disk for no reason: GDAL's ``zip://`` virtual
filesystem (which ``rasterio.open`` understands directly) reads windows straight out of the
archive. This module never extracts anything -- the only thing it does with :mod:`zipfile` is list
an archive's member names.

Two entry points:

- :func:`resolve_openable` turns a user-given path into something
  :meth:`~dynamix.core.rasterfield.RasterField.from_file` (or
  :func:`~dynamix.core.rasterfield.geotiff_info`) can open directly: ``.npz``/``.tif``/``.tiff``
  (and anything else ``RasterField.from_file`` already handles) pass through unchanged; ``.zip``
  is inspected -- never extracted -- for its lone ``.tif``/``.tiff`` member and turned into a
  ``zip://{abspath}!{member}`` GDAL virtual path.
- :func:`open_field` resolves, then sizes the result with
  :func:`~dynamix.core.rasterfield.geotiff_info` (no pixels touched) and either loads the whole
  grid (small enough) or reads one centered, native-resolution window (too big) -- never
  resampled, per the project's standing WTMM scale-sensitivity rule.

Pure stdlib (``zipfile``, ``pathlib``) at module scope plus :mod:`dynamix.core.rasterfield` --
``rasterio`` itself is never imported here; it stays lazily imported exactly where it already was,
inside ``rasterfield.py``'s own functions, which this module calls into rather than duplicating.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from dynamix.core.rasterfield import RasterField, geotiff_info

__all__ = ["resolve_openable", "open_field"]


def resolve_openable(path: str) -> str:
    """A user-given path, turned into something rasterio/``RasterField`` can open directly.

    ``.npz``/``.tif``/``.tiff`` (and any other suffix ``RasterField.from_file`` already handles,
    e.g. ``.npy``, plain images) pass through unchanged. ``.zip`` is inspected with stdlib
    ``zipfile`` -- listing the archive's members, never extracting one -- for names ending
    ``.tif``/``.tiff``. Sidecars (``foo.tif.ovr``, ``foo.tfw``, ``foo.tif.aux.xml``) never match
    that suffix test, so they are never candidates. Exactly one match returns the GDAL virtual
    path ``zip://{abspath}!{member}``; zero or multiple matches raise ``ValueError`` naming what
    was found -- choosing among several members is a follow-up, not a silent pick.
    """
    p = Path(path)
    if p.suffix.lower() != ".zip":
        return str(path)
    with zipfile.ZipFile(p) as zf:
        names = zf.namelist()
    members = [n for n in names if n.lower().endswith((".tif", ".tiff"))]
    if not members:
        raise ValueError(f"{path}: no .tif/.tiff member found in zip; contains: {names}")
    if len(members) > 1:
        raise ValueError(
            f"{path}: multiple .tif/.tiff members in zip, choose one explicitly "
            f"(not yet supported): {members}"
        )
    abspath = str(p.resolve())
    return f"zip://{abspath}!{members[0]}"


def open_field(path: str, *, max_pixels: int = 64_000_000, window_size: int = 4096,
               mode: str = "picture",
               subdataset: "str | None" = None) -> RasterField:
    """Open ``path``: ``.npz``, a GeoTIFF, or a zip holding exactly one GeoTIFF.

    ``.npz`` and any small-enough GeoTIFF load whole through
    :meth:`~dynamix.core.rasterfield.RasterField.from_file` -- verified to work identically on a
    ``zip://...`` URL and on a bare path; GDAL's virtual filesystem doesn't care which it is
    given. A GeoTIFF whose ``width * height`` exceeds ``max_pixels`` (BOEM's two Gulf halves are
    ~8e8 / ~1.8e9 cells; even float32 that is gigabytes, and ``RasterField`` always coerces to
    float64) is never loaded whole: :func:`~dynamix.core.rasterfield.geotiff_info` sizes it
    WITHOUT reading pixels, and a single centered, native-resolution ``window_size``-square window
    (clamped to the raster's own bounds when it is smaller than that) is read via
    :meth:`~dynamix.core.rasterfield.RasterField.from_geotiff_window` -- the SAME windowed-read
    path the tiling machinery already uses, so there is no second window-reading implementation to
    drift from it. Never resampled: WTMM is scale-sensitive, so only the native grid is ever
    handed to analysis (the project's standing rule, not new here).

    Either GeoTIFF route comes back carrying ``provenance["source"]`` and ``provenance
    ["full_dims"]`` -- see :func:`_stamp_source`, and note that an ROI layer is not possible
    without the first of those.

    The window's offset is carried in the field's own name, ``{stem}@x{col_off}y{row_off}``,
    since a v1 window has no other honest way to say "this is not the whole raster" to whatever
    reads the name later (a region-picker UI is a follow-up; see the phase-4 follow-ups doc).
    """
    resolved = resolve_openable(str(path))
    from dynamix.core.ingest import GRID_SUFFIXES
    if resolved.lower().endswith(GRID_SUFFIXES):
        # Sensor formats (netCDF / HDF5 via rasterio's GDAL; HDF4/ASTER via pyhdf):
        # subdataset choice is the CALLER's (main_window probes and asks); a container with
        # exactly one grid loads it without ceremony.
        from dynamix.core.ingest import load_grid, probe

        if subdataset is None:
            info = probe(resolved)
            subs = info["subdatasets"]
            if len(subs) > 1:
                raise ValueError(
                    f"{path}: holds {len(subs)} grids -- pick one "
                    f"(subdataset=...): " + ", ".join(d for _s, d in subs[:8]))
            subdataset = subs[0][0] if subs else None
        # Every format honors the ONE open threshold -- a grid over
        # max_pixels opens as the display PICTURE of the chosen grid (GDAL subdataset strings
        # read directly; HDF4 through pyhdf), and ROI runs read that grid's native pixels.
        if mode == "picture":
            pic = _grid_picture_if_too_big(resolved, subdataset, max_pixels)
            if pic is not None:
                return pic
        field = load_grid(resolved, subdataset=subdataset, name=Path(path).stem
                          if subdataset is None else None)
        ny, nx = np.asarray(field.values).shape[:2]
        return _stamp_source(field, resolved, ny, nx)
    if resolved.lower().endswith((".tif", ".tiff")):
        info = geotiff_info(resolved)
        height, width = info["height"], info["width"]
        # Multiband GeoTIFF: the frozen rasterfield reads band 1 only; a stack
        # routes through ingest so every band lands ((ny, nx, nc) -- pca/tucker-ready).
        # Whole-file only; a too-big multiband window read is a recorded follow-up.
        from dynamix.core.ingest import load_grid, multiband_count
        if height * width <= max_pixels and multiband_count(resolved) > 1:
            field = load_grid(resolved, name=Path(path).stem)
            return _stamp_source(field, resolved, height, width)
        if height * width > max_pixels and mode == "picture":
            # The too-big DEFAULT
            # is the display-only PICTURE -- exact block-centre samples drawn over their
            # native blocks, in FILE pixels, never analysed (engine.resolve refuses it; tools
            # run on saved ROIs, which read native pixels). The overview dataset below stays
            # callable as mode="overview" (never-delete).
            from dynamix.roi.picture import read_picture

            s = read_picture(resolved)
            s.name = f"{Path(path).stem}@pic{int(s.provenance['display_stride'])}"
            return _stamp_source(s, resolved, height, width)
        if height * width > max_pixels and mode == "overview":
            # mode="overview": a decimated WHOLE-EXTENT overview -- display and navigation
            # live here; NATIVE pixels come back through the ROI child tool, which reads
            # windows straight off the source at full resolution (and through wtmm2d_roi,
            # which does the same). Nearest resampling:
            # every kept value is a REAL pixel of the file, never an average -- the
            # no-resampling law is about ANALYSIS, and analysis on an overview is already
            # refused/mapped-to-native by every consumer via provenance["overview"].
            # Axes are DISPLAY pixels; consumers map to native as r*ov + row_off (the
            # _on_roi_create convention).
            import rasterio
            from rasterio.enums import Resampling

            from dynamix.core.frames import GeographicFrame, LocalFrame

            stride = int(np.ceil(np.sqrt(height * width / max_pixels)))
            out_h = (height + stride - 1) // stride
            out_w = (width + stride - 1) // stride
            with rasterio.open(resolved) as src:
                vals = src.read(1, out_shape=(out_h, out_w),
                                resampling=Resampling.nearest).astype(np.float64)
                transform, crs, nodata = src.transform, src.crs, src.nodata
            if nodata is not None:
                vals = np.where(vals == nodata, np.nan, vals)   # the mask convention
            # BOEM-style undeclared fill: the real tifs declare nodata = 0.0
            # but fill empty areas with float32-lowest -- |v| >= 3e38 is nodata whatever the
            # header says (the geo _mask_sentinels convention, applied at READ so the ramp,
            # the stretch and the ANALYSIS pipeline all see NaN, not a sentinel).
            vals = np.where(np.abs(vals) >= 3e38, np.nan, vals)
            # Georeference SURVIVES the decimation (without it the geo view lists the
            # raster as "no-georeference" and never drapes it): axes are the PROJECTED
            # centers of the sampled native blocks --
            # the from_geotiff_window recipe with native pixel index (j + 0.5) * stride --
            # and the frame carries the metric block size, so px_to_metres reads true.
            cols = (np.arange(out_w, dtype=np.float64) + 0.5) * stride
            rows = (np.arange(out_h, dtype=np.float64) + 0.5) * stride
            x_axis, _ = transform * (cols, np.zeros_like(cols))
            _, y_axis = transform * (np.zeros_like(rows), rows)
            x_axis = np.asarray(x_axis, dtype=np.float64)
            y_axis = np.asarray(y_axis, dtype=np.float64)
            if crs is not None and crs.is_projected:
                frame = LocalFrame(x0=float(x_axis[0]), y0=float(y_axis[0]),
                                   dx=abs(float(transform.a)) * stride,
                                   dy=abs(float(transform.e)) * stride,
                                   units=crs.linear_units)
            else:
                frame = GeographicFrame()
            field = RasterField(
                name=f"{Path(path).stem}@ov{stride}", values=vals, frame=frame,
                x_axis=x_axis, y_axis=y_axis)
            field.provenance["overview"] = stride
            field.provenance["window"] = {"row_off": 0, "col_off": 0}
            field.provenance["crs"] = str(crs) if crs is not None else None
            return _stamp_source(field, resolved, height, width)
        if height * width > max_pixels:
            row_off = max(0, (height - window_size) // 2)
            col_off = max(0, (width - window_size) // 2)
            win_h = min(window_size, height)
            win_w = min(window_size, width)
            name = f"{Path(path).stem}@x{col_off}y{row_off}"
            field = RasterField.from_geotiff_window(
                resolved, row_off=row_off, col_off=col_off, height=win_h, width=win_w, name=name,
            )
            return _stamp_source(field, resolved, height, width)
        # Whole-file route: without an explicit name, RasterField.from_file derives it from
        # `resolved`'s OWN stem -- for a zip:// URL that is the mangled `field.zip!field`, which
        # wtmm_backend then uses verbatim as a cache directory / output filename. Name it from the
        # ORIGINAL user path instead, exactly as the windowed branch above already does.
        return _stamp_source(RasterField.from_file(resolved, name=Path(path).stem),
                             resolved, height, width)
    return RasterField.from_file(resolved)


def _grid_picture_if_too_big(resolved: str, subdataset, max_pixels: int):
    """The display picture of a netCDF/HDF grid over ``max_pixels``, else ``None``."""
    # ingest.load_grid's own routing rule, verbatim: an HDF4 SDS NAME on a .hdf file
    is_hdf4 = (subdataset is not None and not str(subdataset).startswith(
        ("NETCDF:", "HDF5:", "HDF4")) and Path(resolved).suffix.lower() == ".hdf")
    if is_hdf4:
        from pyhdf.SD import SD, SDC

        sd = SD(str(resolved), SDC.READ)
        try:
            _n, rank, dims, _t, _na = sd.select(str(subdataset)).info()
        finally:
            sd.end()
        if rank != 2 or int(dims[0]) * int(dims[1]) <= max_pixels:
            return None
        from dynamix.roi.picture import read_picture_hdf4

        return read_picture_hdf4(resolved, str(subdataset))
    import rasterio

    target = str(subdataset) if subdataset is not None else str(resolved)
    with rasterio.open(target) as src:
        h, w, count = int(src.height), int(src.width), int(src.count)
    if h * w <= max_pixels:
        return None
    from dynamix.roi.picture import read_picture

    pic = read_picture(target)
    pic.provenance["subdataset"] = subdataset
    if count > 1:
        pic.provenance["bands"] = count            # picture shows band 1 of the stack
    return pic


def _stamp_source(field, resolved: str, height: int, width: int):
    """Record which raster this field came out of, and how big that raster is.

    ``RasterField.from_geotiff_window`` already stamps ``provenance["source"]`` itself; the
    WHOLE-FILE path (``RasterField._from_geotiff``) stamps nothing at all, and that gap is a
    feature gap, not a cosmetic one: ``dynamix.devices.wtmm_roi.WTMM2DROI`` reads its per-scale
    halo windows straight off ``provenance["source"]`` and REFUSES a field that has none. Without
    this, an ROI could be drawn on any raster small enough to load whole and never be computable
    -- the gesture would work and Create would then fail on the strip.

    Stamped HERE rather than fixed at the source: ``dynamix/core/rasterfield.py`` is a verbatim
    copy of EQSelect (``tests/test_no_silent_drift.py`` hashes it), so it cannot grow a
    provenance key. ``setdefault``, never assignment -- whatever the loader recorded about its own
    read wins, and this only fills a hole.

    ``full_dims`` is stamped on BOTH branches or it would be worthless: the whole point of
    recording the FILE's dimensions is that they are the ones a windowed field cannot report from
    its own shape. It comes free from the ``geotiff_info`` call the size decision already made,
    which is the read a later consumer would otherwise repeat.
    """
    field.provenance.setdefault("source", resolved)
    field.provenance.setdefault("full_dims", (int(height), int(width)))
    return field
