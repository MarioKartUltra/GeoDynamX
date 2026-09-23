"""Pure-numpy geometric selection primitives for the 3D earthquake viewer.

This module is the *math* behind click / box / lasso picking in the GUI. It is deliberately
free of any GUI or 3D-engine dependency (no PySide6 / pyvista / pyvistaqt / Qt): the GUI hands
us a camera matrix and a viewport size, we hand back pixel coordinates and boolean masks. That
keeps the selection logic unit-testable headlessly and reusable from the reload path.

Everything is fully vectorized over the (potentially millions of) events. The only Python loops
are over polygon *edges* (a handful), per the project's hard vectorization rule.

Functions
---------
project_points(pts3d, mvp, viewport) -> (pts2d, visible)
    World -> screen-pixel projection through a 4x4 model-view-projection matrix.
points_in_polygon(pts2d, polygon) -> mask
    Even-odd ray-cast point-in-polygon test (lasso selection).
points_in_box(pts2d, xmin, xmax, ymin, ymax) -> mask
    Axis-aligned rectangle test (box / rubber-band selection).
nearest_point(pts2d, xy, max_dist=None) -> int
    Closest point to a cursor (click selection); -1 if nothing is within ``max_dist``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["project_points", "points_in_polygon", "points_in_box", "nearest_point"]


def project_points(pts3d, mvp, viewport):
    """Project 3-D world points to 2-D screen pixels through an MVP matrix.

    Parameters
    ----------
    pts3d : array-like, shape (N, 3)
        World-space points (e.g. lon/lat/-depth or ECEF coordinates).
    mvp : array-like, shape (4, 4)
        Combined model-view-projection matrix in the usual column-major convention, i.e. clip
        coordinates for a column vector ``v`` are ``mvp @ v``. We therefore right-multiply our
        row-major point array by ``mvp.T``.
    viewport : (w, h)
        Render-window width and height in pixels.

    Returns
    -------
    pts2d : ndarray, shape (N, 2)
        Pixel coordinates ``(x_px, y_px)`` with the origin at the *top-left* of the window
        (y is flipped to screen convention). Values for non-visible / behind-camera points are
        still returned but are not meaningful (may be inf/nan).
    visible : ndarray of bool, shape (N,)
        True where the point is in front of the camera (clip-space ``w > 0``) *and* its
        normalized device coordinates fall inside the canonical view frustum ``[-1, 1]`` on all
        of x, y and z. Behind-camera points (``w <= 0``) are always False.

    Notes
    -----
    NDC -> pixel mapping (matches the contract / OpenGL screen convention)::

        x_px = (ndc_x * 0.5 + 0.5) * w
        y_px = (1 - (ndc_y * 0.5 + 0.5)) * h     # y flipped: NDC +1 (top) -> pixel row 0

    Fully vectorized; no Python loop over points.
    """
    pts3d = np.asarray(pts3d, dtype=float)
    if pts3d.ndim == 1:                       # tolerate a single point passed as (3,)
        pts3d = pts3d.reshape(1, 3)
    mvp = np.asarray(mvp, dtype=float)
    w, h = float(viewport[0]), float(viewport[1])

    n = pts3d.shape[0]
    if n == 0:                                # nothing to project -> well-shaped empties
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=bool)

    # Promote to homogeneous coordinates [x, y, z, 1] and transform to clip space.
    homog = np.column_stack([pts3d, np.ones(n)])      # (N, 4)
    clip = homog @ mvp.T                              # (N, 4) clip-space coordinates
    w_clip = clip[:, 3]

    # Perspective divide. w_clip <= 0 (behind camera or on the plane) produces inf/nan here; those
    # points are filtered out by the visibility test below, so suppress the spurious warnings.
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = clip[:, :3] / w_clip[:, None]           # (N, 3) normalized device coords

    x_px = (ndc[:, 0] * 0.5 + 0.5) * w
    y_px = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h        # flip y to top-left-origin screen pixels
    pts2d = np.column_stack([x_px, y_px])

    # Visible iff in front of the camera AND inside the [-1, 1] frustum on every axis. NaN values
    # (from w_clip == 0) compare False here, which is the desired behaviour.
    visible = (w_clip > 0) & np.all(np.abs(ndc) <= 1.0, axis=1)
    return pts2d, visible


def points_in_polygon(pts2d, polygon):
    """Even-odd (ray-casting) point-in-polygon test, vectorized over points.

    Implements the classic PNPOLY crossing-number algorithm (W. Randolph Franklin): for each
    point we count how many polygon edges a rightward horizontal ray crosses; an odd count means
    the point is inside. The point loop is fully vectorized -- the only Python loop is over the M
    polygon edges (M is small).

    Parameters
    ----------
    pts2d : array-like, shape (N, 2)
        Query points in the same pixel/screen space as ``polygon``.
    polygon : array-like, shape (M, 2)
        Lasso vertices in order. The polygon is auto-closed (the last->first edge is always
        included); an explicitly duplicated closing vertex is tolerated and ignored.

    Returns
    -------
    mask : ndarray of bool, shape (N,)
        True for points inside the polygon. Edge handling follows the standard PNPOLY half-open
        convention (lower/left edges count as inside, upper/right edges as outside), which keeps
        the result deterministic for points exactly on a boundary.
    """
    pts2d = np.asarray(pts2d, dtype=float)
    if pts2d.ndim == 1:
        pts2d = pts2d.reshape(1, 2)
    poly = np.asarray(polygon, dtype=float)

    n_pts = pts2d.shape[0]
    if n_pts == 0 or poly.shape[0] < 3:        # nothing to test / degenerate polygon
        return np.zeros(n_pts, dtype=bool)

    # Drop a duplicated closing vertex if present; the edge loop wraps around regardless.
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    m = poly.shape[0]
    if m < 3:
        return np.zeros(n_pts, dtype=bool)

    px = pts2d[:, 0]
    py = pts2d[:, 1]
    inside = np.zeros(n_pts, dtype=bool)

    j = m - 1
    for i in range(m):                         # loop over EDGES only (few), vectorized over points
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        # Does edge (j -> i) straddle the horizontal line y = py? Strict '>' on one side only gives
        # the half-open convention and guarantees (yj - yi) != 0 wherever `straddle` is True.
        straddle = (yi > py) != (yj > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (py - yi) / (yj - yi) + xi     # edge x at height py
        inside ^= straddle & (px < x_cross)    # toggle parity on each rightward crossing
        j = i

    return inside


def points_in_box(pts2d, xmin, xmax, ymin, ymax):
    """Axis-aligned rectangle (box / rubber-band) selection mask.

    Bounds are normalized so the test is robust to the drag direction (i.e. ``xmin > xmax`` and
    ``ymin > ymax`` are accepted and swapped). Inclusive on all four edges.

    Parameters
    ----------
    pts2d : array-like, shape (N, 2)
    xmin, xmax, ymin, ymax : float
        Rectangle bounds in the same space as ``pts2d``.

    Returns
    -------
    mask : ndarray of bool, shape (N,)
    """
    pts2d = np.asarray(pts2d, dtype=float)
    if pts2d.ndim == 1:
        pts2d = pts2d.reshape(1, 2)
    if pts2d.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    lo_x, hi_x = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
    lo_y, hi_y = (ymin, ymax) if ymin <= ymax else (ymax, ymin)

    x = pts2d[:, 0]
    y = pts2d[:, 1]
    return (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)


def nearest_point(pts2d, xy, max_dist=None):
    """Index of the point nearest to ``xy`` (click selection).

    Parameters
    ----------
    pts2d : array-like, shape (N, 2)
    xy : (x, y)
        Cursor position in the same space as ``pts2d``.
    max_dist : float or None
        If given, the nearest point is only accepted when it lies within this distance (in pixel
        units); otherwise the pick is a miss.

    Returns
    -------
    int
        Index of the nearest point, or ``-1`` if there are no points or the nearest one is farther
        than ``max_dist``.
    """
    pts2d = np.asarray(pts2d, dtype=float)
    if pts2d.ndim == 1:
        pts2d = pts2d.reshape(1, 2)
    if pts2d.shape[0] == 0:
        return -1

    xy = np.asarray(xy, dtype=float)
    d2 = (pts2d[:, 0] - xy[0]) ** 2 + (pts2d[:, 1] - xy[1]) ** 2   # squared distances
    idx = int(np.argmin(d2))

    if max_dist is not None and d2[idx] > float(max_dist) ** 2:    # compare in squared space
        return -1
    return idx
