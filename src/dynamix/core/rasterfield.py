# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Raster fields (EBSD scalar maps, plain images, GeoTIFF DEMs) as scene-placeable layers.

Pure numpy + stdlib (``json``, ``pathlib``, ``dataclasses``) + :mod:`dynamix.core.frames` at module
level -- no GUI, no matplotlib, no ``wtmm``/``wtmm_ebsd`` imports -- so this module is
unit-testable headless and safe to import from the reload path. ``rasterio`` (needed only for the
``.tif``/``.tiff`` loader) is imported LAZILY inside that one branch so the rest of the app works
with pure-Python wheels.

:class:`RasterField` wraps a 2-D (``(ny, nx)``) or multi-component (``(ny, nx, nc)``) array plus
the :class:`~dynamix.core.frames.CoordinateFrame` that places its pixel axes in scene coordinates. It
is the input type for the WTMM pipeline (:mod:`dynamix.core.wtmm_backend`) and the GUI's raster-field
layers: :meth:`to_scene_mesh` hands back plain arrays (points/dims/values) that the GUI turns into
a ``pyvista.StructuredGrid`` -- this module never touches pyvista itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dynamix.core.frames import CoordinateFrame, LocalFrame, GeographicFrame, frame_to_meta, frame_from_meta

__all__ = ["RASTERFIELD_NPZ_SCHEMA", "IMAGE_SUFFIXES", "RasterField",
           "geotiff_info", "geotiff_windows"]


def geotiff_info(path) -> dict:
    """Grid metadata WITHOUT reading any pixels -- for sizing a tiling run.

    ``{"height", "width", "n_cells", "dx", "dy", "projected", "units", "crs", "bounds"}``.
    ``dx``/``dy`` are ground pixel sizes in the CRS's own units (metres for a projected grid,
    degrees for a geographic one). ``rasterio`` is imported lazily here, as everywhere else.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ValueError(f"{path}: reading GeoTIFF metadata needs rasterio ({exc})") from exc
    with rasterio.open(path) as src:
        crs = src.crs
        projected = bool(crs is not None and crs.is_projected)
        units = (crs.linear_units if projected else "degree") if crs is not None else ""
        return {
            "height": int(src.height), "width": int(src.width),
            "n_cells": int(src.height) * int(src.width),
            "dx": abs(float(src.transform.a)), "dy": abs(float(src.transform.e)),
            "projected": projected, "units": units,
            "crs": str(crs) if crs is not None else None,
            "bounds": tuple(float(v) for v in src.bounds),
        }


def geotiff_windows(height, width, tile=2048, overlap=0):
    """Yield ``{"row_off", "col_off", "height", "width"}`` tiling a ``height x width`` grid.

    Pure integer maths -- no file access, so it is testable without rasterio and cheap to call on
    a billion-cell grid. With ``overlap=0`` the windows are an EXACT partition (every cell covered
    once); edge tiles are TRUNCATED rather than padded, so no fabricated data enters a WTMM run.
    ``overlap > 0`` steps by ``tile - overlap`` instead, for analyses that need margin.
    """
    tile = int(tile)
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    step = tile - int(overlap)
    if step <= 0:
        raise ValueError(f"overlap {overlap} must be smaller than tile {tile}")
    for row in range(0, int(height), step):
        for col in range(0, int(width), step):
            h = min(tile, int(height) - row)
            w = min(tile, int(width) - col)
            if h > 0 and w > 0:
                yield {"row_off": row, "col_off": col, "height": h, "width": w}

RASTERFIELD_NPZ_SCHEMA = "rasterfield/1"

#: Plain-image suffixes accepted by :meth:`RasterField.from_file` (read via Pillow, which ships
#: as a matplotlib dependency). These carry NO georeferencing -- they land in ``LocalFrame(px)``.
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".gif", ".pgm", ".ppm", ".webp"})


@dataclass(eq=False)
class RasterField:
    """A 2-D (or multi-component) raster field placed in scene coordinates via a ``frame``.

    Parameters
    ----------
    name : str
        Layer name (also used as the default cache-key prefix by ``wtmm_backend``).
    values : ndarray, shape (ny, nx) or (ny, nx, nc)
        Field data, always coerced to float64. ``nc`` > 1 marks a MULTI-COMPONENT field (e.g. the
        demeaned 3-component log-orientation field feeding the EBSD tensor WTMM). ``NaN`` marks a
        masked pixel (the ``recurrence_cpp`` convention).
    frame : CoordinateFrame
        Maps ``x_axis``/``y_axis`` (already in frame units) to scene coordinates.
    x_axis : ndarray, shape (nx,)
        Frame-x of each pixel COLUMN center, monotonic.
    y_axis : ndarray, shape (ny,)
        Frame-y of each pixel ROW center, monotonic.
    units : str
        Units of ``values`` (e.g. ``"deg"``, ``"1/µm²"``, ``""``).
    provenance : dict
        JSON-safe bookkeeping (source path, params, dates). ``eq=False`` on the dataclass -- two
        fields with identical arrays but different provenance dicts still compare unequal only if
        you compare provenance directly; ``RasterField`` itself has no ``__eq__`` (arrays aren't
        hashable/comparable that way), matching the contract's ``@dataclass(eq=False)``.
    """

    IMAGE_SUFFIXES = IMAGE_SUFFIXES

    name: str
    values: np.ndarray
    frame: CoordinateFrame
    x_axis: np.ndarray
    y_axis: np.ndarray
    units: str = ""
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float64)
        self.x_axis = np.asarray(self.x_axis, dtype=np.float64)
        self.y_axis = np.asarray(self.y_axis, dtype=np.float64)

    @property
    def ny(self) -> int:
        return self.values.shape[0]

    @property
    def nx(self) -> int:
        return self.values.shape[1]

    @property
    def n_components(self) -> int:
        """1 for a scalar ``(ny, nx)`` field; ``nc`` for a multi-component ``(ny, nx, nc)`` field."""
        return 1 if self.values.ndim == 2 else self.values.shape[2]

    @property
    def is_scalar(self) -> bool:
        return self.n_components == 1

    def to_scene_mesh(self) -> dict:
        """Plain-array mesh description for the GUI's ``pyvista.StructuredGrid`` builder.

        Returns
        -------
        dict
            ``"points"`` (ny*nx, 3) float64 scene coordinates from ``frame.to_scene`` on the
            pixel-center meshgrid (z=0); ``"dims"`` = ``(nx, ny, 1)``; ``"values"`` (ny*nx,)
            float64 -- for a scalar field this is ``values`` itself (NaN kept), for a
            multi-component field it is the per-pixel L2 norm over the last axis, forced to NaN
            wherever ANY component at that pixel is NaN.

            Flattening is row-major C-order: ``index = iy * nx + ix``, matching
            ``pv.StructuredGrid(points, dimensions=dims)``.
        """
        X, Y = np.meshgrid(self.x_axis, self.y_axis)          # each (ny, nx), C-order
        points = self.frame.to_scene(X.ravel(), Y.ravel())
        if self.is_scalar:
            values_flat = self.values.reshape(-1)
        else:
            values = self.values
            norm = np.linalg.norm(values, axis=-1)
            any_nan = np.isnan(values).any(axis=-1)
            values_flat = np.where(any_nan, np.nan, norm).reshape(-1)
        return {"points": points, "dims": (self.nx, self.ny, 1), "values": values_flat}

    def save_npz(self, path) -> None:
        """Write this field to ``path`` (``.npz``, compressed) per :data:`RASTERFIELD_NPZ_SCHEMA`."""
        np.savez_compressed(
            path,
            schema=RASTERFIELD_NPZ_SCHEMA,
            name=self.name,
            values=self.values,
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            units=self.units,
            frame_meta=json.dumps(frame_to_meta(self.frame)),
            provenance=json.dumps(self.provenance),
        )

    @classmethod
    def load_npz(cls, path) -> "RasterField":
        """Exact inverse of :meth:`save_npz`. Raises ``ValueError`` on a schema mismatch."""
        with np.load(path, allow_pickle=False) as d:
            schema = str(d["schema"])
            if schema != RASTERFIELD_NPZ_SCHEMA:
                raise ValueError(
                    f"{path}: unsupported rasterfield npz schema {schema!r}; "
                    f"expected {RASTERFIELD_NPZ_SCHEMA!r}"
                )
            return cls(
                name=str(d["name"]),
                values=np.asarray(d["values"], dtype=np.float64),
                frame=frame_from_meta(json.loads(str(d["frame_meta"]))),
                x_axis=np.asarray(d["x_axis"], dtype=np.float64),
                y_axis=np.asarray(d["y_axis"], dtype=np.float64),
                units=str(d["units"]),
                provenance=json.loads(str(d["provenance"])),
            )

    @classmethod
    def _from_plain_npz(cls, path, d, *, frame=None, name=None) -> "RasterField":
        """An npz that is NOT one of ours: accept it if it holds exactly one 2-D/3-D array
        (``np.savez(f, arr)`` -> key ``arr_0``), else name the keys it does have.

        Without this, every npz written by a notebook died on ``d["schema"]`` with a raw KeyError
        naming an npz internal -- useless to whoever picked the file."""
        keys = list(d.keys())
        cands = [k for k in keys if np.asarray(d[k]).ndim in (2, 3)]
        if len(cands) != 1:
            raise ValueError(
                f"{path}: not a RasterField .npz (no {RASTERFIELD_NPZ_SCHEMA!r} schema) and it does "
                f"not hold exactly one 2-D/3-D array; keys found: {keys}. Save it with "
                f"RasterField.save_npz, or hand over a single array (np.save / np.savez)."
            )
        return cls._from_bare_array(np.asarray(d[cands[0]]), path, frame=frame, name=name)

    @classmethod
    def _from_image(cls, path, *, frame=None, name=None, components=False) -> "RasterField":
        """A plain image (PNG/JPEG/BMP/TIFF/...) as a pixel-indexed field, via a LAZILY imported
        Pillow (already present as a matplotlib dependency -- no new requirement).

        RGB collapses to Rec.601 luminance by default, because a plain image is loaded to test the
        SCALAR pipeline. ``components=True`` keeps the three channels instead, giving the
        ``(ny, nx, 3)`` shape the tensor WTMM path consumes. Alpha is always dropped.
        """
        try:
            from PIL import Image
        except ImportError as exc:                      # pragma: no cover - Pillow ships with mpl
            raise ValueError(f"{path}: image support needs Pillow ({exc})") from exc
        try:
            with Image.open(path) as im:
                arr = np.asarray(im)
        except Exception as exc:
            raise ValueError(f"{path}: could not read image ({exc})") from exc
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[:, :, :3]                         # drop alpha
            if not components:
                arr = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
        elif arr.ndim != 2:
            raise ValueError(f"{path}: image must be 2-D or 3-channel, got shape {arr.shape}")
        return cls._from_bare_array(arr, path, frame=frame, name=name)

    @classmethod
    def from_file(cls, path, *, frame=None, name=None, components=False) -> "RasterField":
        """Dispatch by suffix: ``.npz`` -> :meth:`load_npz`; ``.npy``/``.txt`` -> bare ``(ny, nx)``
        values via :meth:`_from_bare_array`, with ``frame`` (default ``LocalFrame(units="px")``)
        and axes baked from that frame's own affine (see :meth:`_from_bare_array`'s own docstring,
        "Axis convention" -- pixel indices under the default frame, ``x0 + dx*arange(...)``
        otherwise); ``.tif``/``.tiff`` -> GeoTIFF via a LAZILY-imported ``rasterio``.

        Raises ``ValueError`` (naming the path and reason) for an unreadable file or an unknown
        suffix -- callers explicitly picked this file, so a silent ``None`` would be unhelpful.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        stem_name = name or path.stem

        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as d:
                if "schema" not in d:                   # a notebook's npz, not one of ours
                    return cls._from_plain_npz(path, d, frame=frame, name=stem_name)
            rf = cls.load_npz(path)
            if name is not None:
                rf.name = name
            if frame is not None:
                rf.frame = frame
            return rf

        if suffix == ".npy":
            try:
                values = np.load(path)
            except Exception as exc:
                raise ValueError(f"{path}: could not read .npy raster ({exc})") from exc
            return cls._from_bare_array(values, path, frame=frame, name=stem_name)

        if suffix == ".txt":
            try:
                values = np.loadtxt(path)
            except Exception as exc:
                raise ValueError(f"{path}: could not read .txt raster ({exc})") from exc
            return cls._from_bare_array(values, path, frame=frame, name=stem_name)

        if suffix in (".tif", ".tiff"):
            try:
                return cls._from_geotiff(path, name=stem_name)
            except ValueError:
                # No rasterio (or no georeferencing): a TIFF is still a perfectly good plain
                # raster. Fall back to Pillow rather than dead-ending -- the frame becomes
                # LocalFrame(px), so georeferencing is DROPPED, never fabricated.
                return cls._from_image(path, frame=frame, name=stem_name, components=components)

        if suffix in cls.IMAGE_SUFFIXES:
            return cls._from_image(path, frame=frame, name=stem_name, components=components)

        raise ValueError(
            f"{path}: unsupported raster suffix {suffix!r}; supported: .npz, .npy, .txt, "
            f".tif, .tiff, " + ", ".join(sorted(cls.IMAGE_SUFFIXES))
        )

    @classmethod
    def _from_bare_array(cls, values, path, *, frame=None, name=None) -> "RasterField":
        """A plain ``(ny, nx)``/``(ny, nx, nc)`` array with no georeferencing of its own -- the
        ``.npy``/``.txt``/plain-npz/plain-image loaders' shared tail.

        **Axis convention.** :class:`LocalFrame`'s
        own class docstring states the contract plainly: ``to_scene`` is a PURE pass-through (``x``,
        ``y`` straight into ``column_stack`` -- confirmed by reading ``frames.py`` directly, it
        never touches ``x0``/``y0``/``dx``/``dy`` at all), and "the affine pixel-index -> frame-unit
        mapping described by ``x0``/``y0``/``dx``/``dy`` is applied by ``RasterField`` when it
        builds its axis arrays" -- i.e. ``x_axis``/``y_axis`` must already BE frame-unit
        coordinates, not raw pixel indices, whenever the frame's own affine is non-default. Every
        OTHER construction path already honors this: :meth:`from_geotiff_window` bakes the affine
        straight from the GeoTIFF transform, :meth:`from_ebsd_fields` builds ``px_um * arange(...)``
        axes to match its own ``LocalFrame(dx=px_um, dy=px_um)``. This method was the one holdout --
        it built plain ``arange(nx)``/``arange(ny)`` axes regardless of what ``frame`` a caller
        passed in, so a NON-default ``LocalFrame`` (e.g. ``dx=2.0, x0=100.0`` on a ``.npy``/``.txt``
        load) silently produced a field whose displayed scale/origin disagreed with its own
        declared frame -- a real, silent correctness gap (confirmed: nothing in the shipped app
        exercises it today, ``opening.py`` never passes an explicit ``frame=``, so every existing
        test/caller uses the default ``LocalFrame(units="px")`` -- x0=0, dx=1 -- for which this fix
        is a no-op, ``x0 + dx*arange(nx) == arange(nx)`` exactly).

        Fixed here to match the documented convention: bake ``frame.x0 + frame.dx * arange(nx)`` /
        ``frame.y0 + frame.dy * arange(ny)`` whenever ``frame`` carries those attributes (every
        :class:`LocalFrame`); ``getattr(..., default)`` falls back to the pre-fix ``arange`` for any
        OTHER frame kind (e.g. a bare :class:`GeographicFrame`, which has no ``x0``/``dx`` of its
        own and is not a semantically meaningful pairing with a bare array in the first place) --
        never a crash, just the prior behavior for a combination this fix has no contract to apply
        to.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.ndim not in (2, 3):
            raise ValueError(f"{path}: raster array must be 2-D or 3-D, got shape {values.shape}")
        ny, nx = values.shape[0], values.shape[1]
        use_frame = frame if frame is not None else LocalFrame(units="px")
        x0 = getattr(use_frame, "x0", 0.0)
        y0 = getattr(use_frame, "y0", 0.0)
        dx = getattr(use_frame, "dx", 1.0)
        dy = getattr(use_frame, "dy", 1.0)
        return cls(
            name=name or path.stem,
            values=values,
            frame=use_frame,
            x_axis=x0 + dx * np.arange(nx, dtype=np.float64),
            y_axis=y0 + dy * np.arange(ny, dtype=np.float64),
        )

    @classmethod
    def from_geotiff_window(cls, path, *, row_off, col_off, height, width,
                            frame=None, name=None) -> "RasterField":
        """Read ONE window of a GeoTIFF -- the only way to touch a grid too large to hold.

        BOEM's Gulf bathymetry is 1.4e9 cells; as float64 that is 11 GB of values and another
        34 GB of scene points, so the whole-file path cannot be used at all. A windowed read
        materialises just the tile.

        **Frame choice matters for WTMM.** A PROJECTED source (e.g. UTM metres) yields a
        :class:`~dynamix.core.frames.LocalFrame` carrying the real ground pixel size, so wavelet
        scales are metric and equal-area everywhere in the tile. Reprojecting such a grid to
        lon/lat would make a fixed wavelet scale span different ground distances in x and y, and
        drift with latitude -- distorting the analysis to satisfy a display convention. A source
        that is already GEOGRAPHIC keeps :class:`~dynamix.core.frames.GeographicFrame`.

        Axes are ABSOLUTE projected/geographic coordinates for the window's own position, so
        adjacent tiles abut exactly rather than each starting at zero.
        """
        try:
            import rasterio
            from rasterio.windows import Window
        except ImportError as exc:
            raise ValueError(f"{path}: windowed GeoTIFF reads need rasterio ({exc})") from exc
        path = Path(path)
        try:
            with rasterio.open(path) as src:
                win = Window(int(col_off), int(row_off), int(width), int(height))
                band = np.asarray(src.read(1, window=win), dtype=np.float64)
                transform = src.window_transform(win)
                crs = src.crs
                nodata = src.nodata
        except Exception as exc:
            raise ValueError(f"{path}: could not read GeoTIFF window ({exc})") from exc
        if nodata is not None:
            band = np.where(band == nodata, np.nan, band)      # nodata -> NaN (the mask convention)
        # Undeclared float32-extreme fill (2026-09-22, the BOEM headers declare nodata = 0.0
        # while the empty areas hold -3.4028235e38): |v| >= 3e38 is nodata whatever the
        # header says -- geo.mapping._mask_sentinels' own threshold, applied at READ.
        band = np.where(np.abs(band) >= 3e38, np.nan, band)
        ny, nx = band.shape
        cols = np.arange(nx, dtype=np.float64) + 0.5           # pixel CENTRES
        rows = np.arange(ny, dtype=np.float64) + 0.5
        x_axis, _ = transform * (cols, np.zeros_like(cols))
        _, y_axis = transform * (np.zeros_like(rows), rows)
        x_axis = np.asarray(x_axis, dtype=np.float64)
        y_axis = np.asarray(y_axis, dtype=np.float64)
        if frame is None:
            if crs is not None and crs.is_projected:
                frame = LocalFrame(x0=float(x_axis[0]), y0=float(y_axis[0]),
                                   dx=abs(float(transform.a)), dy=abs(float(transform.e)),
                                   units=crs.linear_units)
            else:
                frame = GeographicFrame()
        return cls(
            name=name or f"{path.stem}_r{int(row_off)}c{int(col_off)}",
            values=band, frame=frame, x_axis=x_axis, y_axis=y_axis,
            units=(crs.linear_units if (crs is not None and crs.is_projected) else ""),
            provenance={"source": str(path), "window": {"row_off": int(row_off),
                                                        "col_off": int(col_off),
                                                        "height": int(ny), "width": int(nx)},
                        "crs": str(crs) if crs is not None else None},
        )

    @classmethod
    def _from_geotiff(cls, path, *, name=None) -> "RasterField":
        try:
            import rasterio
        except ImportError as exc:
            raise ValueError(
                f"{path}: GeoTIFF support needs rasterio: pip install rasterio ({exc})"
            ) from exc
        try:
            with rasterio.open(path) as src:
                band = np.asarray(src.read(1), dtype=np.float64)
                transform = src.transform
                crs = src.crs
                ny, nx = band.shape
                cols = np.arange(nx, dtype=np.float64) + 0.5
                rows = np.arange(ny, dtype=np.float64) + 0.5
                x_axis, _ = transform * (cols, np.zeros_like(cols))
                _, y_axis = transform * (np.zeros_like(rows), rows)
        except Exception as exc:
            raise ValueError(f"{path}: could not read GeoTIFF ({exc})") from exc
        # Frame by CRS KIND, mirroring from_geotiff_window's branch above (2026-08-19 fix; the
        # recorded b8cbdfd gap): a PROJECTED CRS means x_axis/y_axis are linear units (metres,
        # US survey feet, ...), and stamping GeographicFrame() sent them through projection.py's
        # longitude canonicalisation — np.mod(x, 360) — folding a UTM-like axis every 360 units
        # (measured on the BOEM west crop: 400 real columns collapsed onto 9 distinct x values,
        # 45 wrap-arounds — the "dateline-looking" smear on the Vector tab). The Globe path was
        # never affected: geo.mapping keys off provenance["crs"], not field.frame.
        if crs is not None and crs.is_projected:
            frame = LocalFrame(x0=float(x_axis[0]), y0=float(y_axis[0]),
                               dx=abs(float(transform.a)), dy=abs(float(transform.e)),
                               units=crs.linear_units)
        else:
            frame = GeographicFrame()
        return cls(
            name=name or path.stem,
            values=band,
            frame=frame,
            x_axis=np.asarray(x_axis, dtype=np.float64),
            y_axis=np.asarray(y_axis, dtype=np.float64),
            units=(crs.linear_units if (crs is not None and crs.is_projected) else ""),
            provenance={"crs": str(crs) if crs is not None else None},
        )

    @classmethod
    def from_ebsd_fields(cls, fields: dict, key: str, *, px_um: float, name=None) -> "RasterField":
        """Wrap one ``(ny, nx)`` output of ``wtmm_ebsd.scalar_fields.build_scalar_fields``.

        ``fields`` is the already-computed dict (this never calls ``orix``/``wtmm_ebsd`` itself).
        ``LocalFrame(dx=px_um, dy=px_um, units="µm")``; axes = ``px_um * arange``; bool arrays are
        coerced to float64 (0.0/1.0); ``provenance`` records ``key`` and
        ``fields["_norm_p99"].get(key)``.
        """
        values = np.asarray(fields[key], dtype=np.float64)
        ny, nx = values.shape[0], values.shape[1]
        norm_p99 = fields.get("_norm_p99", {}).get(key)
        return cls(
            name=name or key,
            values=values,
            frame=LocalFrame(dx=px_um, dy=px_um, units="µm"),
            x_axis=px_um * np.arange(nx, dtype=np.float64),
            y_axis=px_um * np.arange(ny, dtype=np.float64),
            provenance={"key": key, "norm_p99": norm_p99},
        )
