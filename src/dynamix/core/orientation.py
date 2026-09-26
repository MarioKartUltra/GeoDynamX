# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Orientations as ``(angle, frame)`` -- never a bare float.

A WTMM argument lives in the PIXEL frame: ``arg = arctan2(dy, dx)`` over the array indices,
0 = toward increasing column, +90 = toward increasing row (clockwise on a row-down screen),
pointing uphill. It becomes a bearing only through the grid's own geometry:

- a map-projected grid gives a GRID azimuth (clockwise from grid north) from the directions
  its axes run -- rows running south is the usual north-up raster;
- a GRID azimuth becomes TRUE by adding the bearing of grid north at that point (the meridian
  convergence, :func:`dynamix.geo.mapping.grid_north_bearing`);
- a lon/lat grid's meridians ARE true north, so it goes straight to TRUE;
- a swath (unprojected sensor geometry) has no fixed north: it stays PIXEL until its
  geolocation lattice gives each pixel its own north.

MAGNETIC needs a declination model and a survey's recorded setting; nothing here guesses it.
"""
from __future__ import annotations

import dataclasses
import enum

import numpy as np

__all__ = ["Frame", "Orientation", "axial_sector", "from_arg", "grid_to_true", "to_azimuth"]


class Frame(str, enum.Enum):
    PIXEL = "pixel"
    GRID = "grid"
    TRUE = "true"
    MAGNETIC = "magnetic"


@dataclasses.dataclass(frozen=True)
class Orientation:
    """Angles in DEGREES (scalar or array) and the frame they are measured in. PIXEL angles
    are counter-clockwise-in-index-space from +column (the arctan2 convention); every other
    frame is an azimuth, clockwise from its north, in [0, 360)."""

    degrees: "np.ndarray | float"
    frame: Frame


def from_arg(arg_rad) -> Orientation:
    """A WTMM argument (radians, pixel frame) as an :class:`Orientation`."""
    return Orientation(np.degrees(np.asarray(arg_rad, dtype=np.float64)), Frame.PIXEL)


def to_azimuth(o: Orientation, *, col_east: bool, row_north: bool,
               north: Frame) -> Orientation:
    """A PIXEL orientation as an azimuth in the grid's ``north`` frame (GRID for a projected
    grid, TRUE for a lon/lat one). ``col_east`` / ``row_north``: whether increasing column
    runs east and increasing row runs north (a north-up raster has ``row_north=False``)."""
    if o.frame is not Frame.PIXEL:
        raise ValueError(f"to_azimuth converts PIXEL angles, not {o.frame.value}")
    if north not in (Frame.GRID, Frame.TRUE):
        raise ValueError("a grid's own north is GRID (projected) or TRUE (lon/lat)")
    a = np.radians(np.asarray(o.degrees, dtype=np.float64))
    east = np.cos(a) * (1.0 if col_east else -1.0)
    north_c = np.sin(a) * (1.0 if row_north else -1.0)
    return Orientation(np.mod(np.degrees(np.arctan2(east, north_c)), 360.0), north)


def grid_to_true(o: Orientation, grid_north_bearing_deg) -> Orientation:
    """A GRID azimuth as TRUE: add the true bearing of grid north at each point."""
    if o.frame is not Frame.GRID:
        raise ValueError(f"grid_to_true converts GRID azimuths, not {o.frame.value}")
    return Orientation(np.mod(np.asarray(o.degrees, dtype=np.float64)
                              + np.asarray(grid_north_bearing_deg, dtype=np.float64), 360.0),
                       Frame.TRUE)


def axial_sector(o: Orientation, n: int = 4) -> np.ndarray:
    """The AXIAL sector (orientation mod 180°) of each angle: ``n`` sectors centred on
    0, 180/n, 2·180/n, ... -- Arneodo, Decoster & Roux 2000's Fig. 26 bins for ``n = 4``
    (0, 45, 90, 135 ± 22.5°), which for an azimuth frame read N–S, NE–SW, E–W, SE–NW."""
    width = 180.0 / int(n)
    d = np.mod(np.asarray(o.degrees, dtype=np.float64) + width / 2.0, 180.0)
    return np.floor(d / width).astype(np.int64) % int(n)
