# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Coordinate frames mapping analysis-space (lon/lat, or lab-frame x/y) to 3-D scene coordinates.

Pure numpy + :mod:`dynamix.core.projection` -- no GUI, no scipy, no pandas, so this module is
unit-testable headless and safe to import from the reload path. It is the seam that lets
:class:`~dynamix.core.rasterfield.RasterField` and the WTMM chain overlays share one placement
convention with the existing earthquake point cloud: :class:`GeographicFrame` wraps
:mod:`dynamix.core.projection` (NO new math -- it delegates every coordinate straight to
``projection.project``), while :class:`LocalFrame` is a pass-through for lab-frame maps (EBSD
micrographs in microns, plain images in pixels) whose pixel-index-to-scene-unit mapping is applied
by the RasterField axis construction, not here.

Both frames implement the :class:`CoordinateFrame` protocol: ``to_scene`` (vectorized, accepts
scalars or arrays and broadcasts them), ``extent`` and ``axis_labels`` for the GUI's axis
plumbing. :func:`frame_to_meta` / :func:`frame_from_meta` give a JSON-safe round trip used by
``RasterField.save_npz`` / ``load_npz`` and the WTMM chain ``.npz`` exports.
"""
from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import numpy as np

from dynamix.core import projection

__all__ = [
    "CoordinateFrame",
    "GeographicFrame",
    "LocalFrame",
    "frame_to_meta",
    "frame_from_meta",
]


@runtime_checkable
class CoordinateFrame(Protocol):
    """The seam every scene-placeable coordinate system implements.

    Attributes
    ----------
    kind : str
        ``"geographic"`` or ``"local"`` -- discriminator used by :func:`frame_to_meta` /
        :func:`frame_from_meta` for the JSON round trip.
    units : str
        Horizontal units of the frame, e.g. ``"deg"``, ``"µm"``, ``"px"``.
    """

    kind: str
    units: str

    def to_scene(self, x, y, z=None) -> np.ndarray:
        """Vectorized ``(x, y, z) -> (N, 3)`` float64 scene coordinates; ``z=None`` -> zeros."""
        ...

    def extent(self, x_axis, y_axis) -> tuple:
        """``(xmin, xmax, ymin, ymax)`` in frame units for the given axis arrays."""
        ...

    def axis_labels(self) -> tuple[str, str]:
        """``(x_label, y_label)`` for plot/axis widgets, e.g. ``("lon (deg)", "lat (deg)")``."""
        ...


def _broadcast_xyz(x, y, z):
    """Coerce ``x``/``y``/``z`` to a common 1-D float64 length; ``z=None`` -> zeros like ``x``.

    Shared by both frame implementations so scalar or array inputs are handled identically.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if z is None:
        z = np.zeros_like(x)
    else:
        z = np.atleast_1d(np.asarray(z, dtype=float))
    x, y, z = np.broadcast_arrays(x, y, z)
    return x, y, z


def _extent(x_axis, y_axis) -> tuple:
    """Shared ``extent`` implementation: plain-float ``(xmin, xmax, ymin, ymax)``."""
    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)
    return (float(x_axis.min()), float(x_axis.max()), float(y_axis.min()), float(y_axis.max()))


@dataclasses.dataclass(frozen=True)
class GeographicFrame:
    """Earth-surface frame: wraps :func:`dynamix.core.projection.project` -- NO new math.

    The GUI constructs a fresh, cheap, immutable instance per rebuild with the CURRENT
    projection mode / vertical exaggeration (mirroring the existing ``_project_lld`` idiom);
    mixing several ``GeographicFrame`` instances (or with ``LocalFrame``) in one scene is
    allowed but only meaningful when their ``mode``/``vexag`` agree.

    Parameters
    ----------
    mode : str
        One of :data:`dynamix.core.projection.MODES` (``"pacific"``, ``"greenwich"``,
        ``"mercator"``, ``"globe"``).
    vexag : float
        Vertical exaggeration passed straight through to ``projection.project``.
    """

    mode: str = projection.DEFAULT_MODE
    vexag: float = 1.0

    kind = "geographic"
    units = "deg"

    def to_scene(self, x, y, z=None) -> np.ndarray:
        """``to_scene(lon, lat, height_km=None)`` == ``projection.project(lon, lat, height_km or 0,
        self.mode, self.vexag)``; height is signed and POSITIVE UP (a depth ``d`` is ``-d``).
        """
        lon, lat, height = _broadcast_xyz(x, y, z)
        return np.asarray(projection.project(lon, lat, height, self.mode, self.vexag), dtype=float)

    def extent(self, x_axis, y_axis) -> tuple:
        """``(lon_min, lon_max, lat_min, lat_max)`` for the given axis arrays."""
        return _extent(x_axis, y_axis)

    def axis_labels(self) -> tuple[str, str]:
        return ("lon (deg)", "lat (deg)")


@dataclasses.dataclass(frozen=True)
class LocalFrame:
    """Lab-frame maps: EBSD scalar fields (microns), plain images (pixels), and similar rasters.

    ``to_scene`` is a pure pass-through -- ``x``/``y`` are already expressed in frame (scene)
    units. The affine pixel-index -> frame-unit mapping described by ``x0``/``y0``/``dx``/``dy``
    is applied by :class:`~dynamix.core.rasterfield.RasterField` when it builds its axis arrays, so
    chains and extrema computed in frame units project 1:1 without a second transform here.

    Parameters
    ----------
    x0, y0 : float
        Scene coordinate of pixel-column-0 / pixel-row-0 CENTER.
    dx, dy : float
        Pixel size along x / y (frame units per pixel). ``dy`` is positive when y increases
        upward (screen/scene convention), i.e. it is NOT necessarily the raster row stride sign.
    units : str
        Horizontal/vertical units, e.g. ``"µm"``, ``"px"``.
    """

    x0: float = 0.0
    y0: float = 0.0
    dx: float = 1.0
    dy: float = 1.0
    units: str = "px"

    kind = "local"

    def to_scene(self, x, y, z=None) -> np.ndarray:
        """``to_scene(x, y, z=None)`` == ``column_stack([x, y, z or 0])`` -- pure pass-through."""
        x, y, z = _broadcast_xyz(x, y, z)
        return np.column_stack([x, y, z]).astype(float, copy=False)

    def extent(self, x_axis, y_axis) -> tuple:
        """``(xmin, xmax, ymin, ymax)`` for the given axis arrays."""
        return _extent(x_axis, y_axis)

    def axis_labels(self) -> tuple[str, str]:
        return (f"x ({self.units})", f"y ({self.units})")


_FRAME_CLASSES: dict[str, type] = {
    "geographic": GeographicFrame,
    "local": LocalFrame,
}


def frame_to_meta(frame: CoordinateFrame) -> dict:
    """JSON-safe ``{"kind": ..., "units": ..., <dataclass fields>}`` for ``frame``.

    ``kind`` (always) and ``units`` (for :class:`GeographicFrame`) are plain class attributes,
    not dataclass fields, so ``dataclasses.asdict`` never includes them -- both are added
    explicitly here so the round trip through :func:`frame_from_meta` can dispatch on ``kind``
    and every frame reports its ``units`` uniformly. For :class:`LocalFrame`, ``units`` is also
    a dataclass field; the explicit value below and the one from ``asdict`` always agree.
    """
    return {"kind": frame.kind, "units": frame.units, **dataclasses.asdict(frame)}


def frame_from_meta(meta: dict) -> CoordinateFrame:
    """Exact inverse of :func:`frame_to_meta`; raises ``ValueError`` on an unknown ``kind``."""
    kind = meta.get("kind")
    cls = _FRAME_CLASSES.get(kind)
    if cls is None:
        raise ValueError(f"unknown coordinate frame kind {kind!r}; choose from {sorted(_FRAME_CLASSES)}")
    field_names = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in meta.items() if k in field_names}
    return cls(**kwargs)


def frames_compatible(a: CoordinateFrame, b: CoordinateFrame) -> bool:
    """Predicate: whether two frames may co-display on the Vector tab.

    True iff ``a`` and ``b`` are the same concrete class AND same ``units`` AND
    numeric fields equal within ``rtol=1e-9``. Used to decide whether to display
    frames together in the Vector-tab slice view.

    Parameters
    ----------
    a, b : CoordinateFrame
        Either LocalFrame or GeographicFrame instances.

    Returns
    -------
    bool
        True if compatible for co-display, False otherwise (including Geographic-vs-Local).
    """
    if type(a) is not type(b):
        return False
    if a.units != b.units:
        return False
    for field in dataclasses.fields(a):
        val_a = getattr(a, field.name)
        val_b = getattr(b, field.name)
        if isinstance(val_a, (int, float)):
            if not np.isclose(val_a, val_b, rtol=1e-9):
                return False
        else:
            if val_a != val_b:
                return False
    return True
