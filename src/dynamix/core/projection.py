# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Map (lon, lat, height) points to 3-D scene coordinates under several projections.

``height_km`` is signed and POSITIVE UP: elevation above the ellipsoid is positive, depth below it
is negative. EQSelect passed positive-down ``depth_km`` because a hypocentre is never above the
surface; DynamiX also carries DEMs and topography, so the convention is flipped at the boundary
(``height_km = -depth_km``). The geometry is unchanged.

Pure numpy, fully vectorized -- no GUI imports, so it is unit-testable headless. The GUI picks a
mode from :data:`MODES` and rebuilds the point cloud; selection is unaffected (it hit-tests in screen
space via the live camera, whatever the projection).

Modes
-----
``greenwich``  equirectangular, 0deg-centred -- ``[lon, lat, height/KM_PER_DEG*vexag]`` (the original
               view; splits arcs that straddle the +/-180 dateline, e.g. Kermadec/Tonga, Alaska).
``pacific``    equirectangular, Pacific-centred -- longitudes shifted to ``[0, 360)`` so the dateline
               is contiguous (the seam moves to the Atlantic). Fixes the split-arc problem.
``mercator``   Web-Mercator x/y, Pacific-centred (central meridian 180). Latitude is stretched by the
               Mercator term (clamped to +/-85 deg); x stays Pacific longitude.
``globe``      WGS84-ellipsoid ECEF (km): geodetic ``(lat, lon, h=height)`` -> Earth-centred Earth-fixed
               on the actual reference ellipsoid (equatorial a=6378.137, polar b=6356.752 km -- the
               ~21 km flattening), so there is *no* dateline seam and the placement is geodetically
               correct (catalog latitudes are geodetic). The whole Earth can be rotated.
"""
from __future__ import annotations

import numpy as np

KM_PER_DEG = 111.195                 # km per degree latitude -> height(km) to degree-equivalent Z
R_EARTH_KM = 6371.0                  # mean-sphere radius (kept for reference)

# WGS84 reference ellipsoid (km) -- used by the `globe` projection
WGS84_A = 6378.137                   # semi-major axis (equatorial radius)
WGS84_F = 1.0 / 298.257223563        # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared, e^2 = 2f - f^2
WGS84_B = WGS84_A * (1.0 - WGS84_F)   # semi-minor axis (polar radius) ~ 6356.752 km

# display order + human labels for the GUI dropdown
MODES = ("pacific", "greenwich", "mercator", "globe")
LABELS = {
    "pacific":   "Pacific (equirect)",
    "greenwich": "Greenwich (equirect)",
    "mercator":  "Web Mercator (Pacific)",
    "globe":     "Globe (ECEF)",
}
DEFAULT_MODE = "pacific"             # fixes the split-arc problem out of the box


def _pacific_lon(lon):
    """Longitudes shifted to ``[0, 360)`` so the +/-180 dateline is contiguous (Pacific-centred)."""
    return np.mod(lon, 360.0)


def _z_height(height_km, vexag):
    """Flat-view Z: signed height (km, positive up) -> degree-equivalent units, optionally exaggerated."""
    return np.asarray(height_km, float) / KM_PER_DEG * vexag


def display_lon(lon, mode):
    """The monotonic horizontal coordinate a flat projection lays longitudes out along.

    Used for SEAM handling of connected geometry (grids / polylines): in ``pacific``/``mercator`` the
    layout is ``lon mod 360`` (seam in the Atlantic), in ``greenwich`` it is ``lon`` (seam at +/-180).
    Meaningless for ``globe`` (a sphere has no seam) -- callers skip seam handling there.
    """
    lon = np.asarray(lon, float)
    if mode in ("pacific", "mercator"):
        return np.mod(lon, 360.0)
    return lon


def project(lon, lat, height_km, mode=DEFAULT_MODE, vexag=1.0):
    """Vectorized ``(lon, lat, height_km) -> (N, 3)`` scene coordinates for ``mode``.

    ``height_km`` is signed and POSITIVE UP; a depth of ``d`` below the surface is ``-d``.

    ``vexag`` exaggerates the vertical (flat views) or the radial displacement (globe). Raises
    ``ValueError`` on an unknown mode.
    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    height = np.asarray(height_km, float)

    if mode == "greenwich":
        return np.column_stack([lon, lat, _z_height(height, vexag)])

    if mode == "pacific":
        return np.column_stack([_pacific_lon(lon), lat, _z_height(height, vexag)])

    if mode == "mercator":
        phi = np.radians(np.clip(lat, -85.0, 85.0))
        merc_y = np.degrees(np.log(np.tan(np.pi / 4.0 + phi / 2.0)))   # degree-equivalent Mercator y
        return np.column_stack([_pacific_lon(lon), merc_y, _z_height(height, vexag)])

    if mode == "globe":
        # WGS84 geodetic (lat, lon, h) -> ECEF (km), h = height*vexag (positive up, along the
        # ellipsoid normal). N = prime-vertical radius of curvature. This is the standard
        # geodetic->ECEF transform; input latitudes are geodetic, so it is the correct placement.
        lo, la = np.radians(lon), np.radians(lat)
        sin_la, cos_la = np.sin(la), np.cos(la)
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_la ** 2)
        h = np.maximum(height * vexag, -0.9 * N)   # clamp so big exaggeration can't invert the sphere
        x = (N + h) * cos_la * np.cos(lo)
        y = (N + h) * cos_la * np.sin(lo)
        z = (N * (1.0 - WGS84_E2) + h) * sin_la
        return np.column_stack([x, y, z])

    raise ValueError(f"unknown projection mode {mode!r}; choose from {MODES}")


def unproject(xyz, mode):
    """Inverse of :func:`project` (horizontal only): scene xyz -> ``(lon, lat)`` in degrees [-180, 180).

    Depth/elevation is dropped (we only need where on the map a click landed, for transect endpoints).
    For ``globe`` the latitude is geocentric (negligibly off from geodetic for picking).
    """
    xyz = np.atleast_2d(np.asarray(xyz, float))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if mode == "globe":
        # WGS84 ECEF -> geodetic via Bowring's closed-form (exact inverse of the forward globe map)
        p = np.sqrt(x * x + y * y)
        ep2 = WGS84_E2 / (1.0 - WGS84_E2)                    # second eccentricity squared
        theta = np.arctan2(z * WGS84_A, p * WGS84_B)
        lon = np.degrees(np.arctan2(y, x))
        lat = np.degrees(np.arctan2(z + ep2 * WGS84_B * np.sin(theta) ** 3,
                                    p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3))
    elif mode == "mercator":
        lon = ((x + 180.0) % 360.0) - 180.0
        lat = np.degrees(2.0 * np.arctan(np.exp(np.radians(y))) - np.pi / 2)
    elif mode == "pacific":
        lon = ((x + 180.0) % 360.0) - 180.0
        lat = y
    else:  # greenwich
        lon, lat = x, y
    out = np.column_stack([lon, lat])
    return out[0] if out.shape[0] == 1 else out
