# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.selection -- pure-numpy picking primitives.

Covers, per the contract:
  * points_in_polygon on a hand-built square (in / out / on-edge) and a concave (notched) polygon,
    plus auto-close equivalence,
  * points_in_box masks (including reversed drag bounds),
  * nearest_point picking incl. a max_dist miss (-1) and the empty-cloud case,
  * project_points on a unit cube through an identity MVP (exact pixel coords + all-visible),
    an out-of-frustum point, and a behind-camera point (clip w < 0 -> not visible).
"""
import numpy as np
import pytest

from dynamix.core.selection import (
    project_points,
    points_in_polygon,
    points_in_box,
    nearest_point,
)


# --------------------------------------------------------------------------- points_in_polygon
def test_points_in_polygon_square_in_out_on_edge():
    # CCW unit-ish square [0,4] x [0,4].
    square = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)

    # Clearly interior / clearly exterior points.
    inside_pts = np.array([[2, 2], [0.1, 0.1], [3.9, 3.9], [1.0, 3.0]])
    outside_pts = np.array([[5, 5], [-1, 2], [2, -1], [2, 5], [-0.1, -0.1]])

    assert points_in_polygon(inside_pts, square).all()
    assert not points_in_polygon(outside_pts, square).any()

    # On-edge points follow the standard PNPOLY half-open convention:
    #   left & bottom edges -> inside (True);  right & top edges -> outside (False).
    edge_pts = np.array([
        [0, 2],   # left edge   -> True
        [2, 0],   # bottom edge -> True
        [4, 2],   # right edge  -> False
        [2, 4],   # top edge    -> False
    ])
    expected = np.array([True, True, False, False])
    np.testing.assert_array_equal(points_in_polygon(edge_pts, square), expected)


def test_points_in_polygon_concave_notch():
    # Square with a downward V-notch cut out of the top, apex at (2,2): concave.
    concave = np.array([[0, 0], [4, 0], [4, 4], [2, 2], [0, 4]], dtype=float)

    pts = np.array([
        [2, 1],    # below the notch apex            -> inside
        [3, 1],    # lower-right interior            -> inside
        [0.5, 0.5],# lower-left interior             -> inside
        [2, 3],    # *inside the notch* (cut away)   -> outside
        [2, 3.9],  # high in the notch               -> outside
        [5, 5],    # far outside                     -> outside
    ])
    expected = np.array([True, True, True, False, False, False])
    np.testing.assert_array_equal(points_in_polygon(pts, concave), expected)


def test_points_in_polygon_autoclose_equivalence():
    # An explicitly closed polygon (first vertex duplicated at the end) must give the same mask
    # as the open one -- the function auto-closes either way.
    open_poly = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
    closed_poly = np.vstack([open_poly, open_poly[0]])

    pts = np.array([[2, 2], [5, 5], [1, 3], [-1, -1]])
    np.testing.assert_array_equal(
        points_in_polygon(pts, open_poly),
        points_in_polygon(pts, closed_poly),
    )


def test_points_in_polygon_degenerate_and_empty():
    square = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
    # No query points -> empty bool mask.
    assert points_in_polygon(np.empty((0, 2)), square).shape == (0,)
    # Degenerate polygon (< 3 vertices) -> everything outside.
    line = np.array([[0, 0], [1, 1]], dtype=float)
    assert not points_in_polygon(np.array([[0.5, 0.5]]), line).any()


# --------------------------------------------------------------------------------- points_in_box
def test_points_in_box_mask():
    pts = np.array([
        [2, 2],    # inside
        [1, 1],    # inside (corner, inclusive)
        [3, 3],    # inside (corner, inclusive)
        [0, 0],    # outside
        [5, 2],    # outside (x too big)
        [2, 5],    # outside (y too big)
    ])
    mask = points_in_box(pts, 1, 3, 1, 3)
    np.testing.assert_array_equal(mask, [True, True, True, False, False, False])


def test_points_in_box_reversed_bounds():
    # A box dragged from bottom-right to top-left passes xmin>xmax / ymin>ymax; must still work.
    pts = np.array([[2, 2], [0, 0], [5, 5]])
    forward = points_in_box(pts, 1, 3, 1, 3)
    reversed_ = points_in_box(pts, 3, 1, 3, 1)
    np.testing.assert_array_equal(forward, reversed_)
    np.testing.assert_array_equal(forward, [True, False, False])


# --------------------------------------------------------------------------------- nearest_point
def test_nearest_point_basic_and_max_dist():
    pts = np.array([[0, 0], [10, 0], [10, 10]], dtype=float)

    # Closest to (1, 0) is point 0 at distance 1.
    assert nearest_point(pts, (1, 0)) == 0
    # With a generous radius the pick stands.
    assert nearest_point(pts, (1, 0), max_dist=2.0) == 0
    # With a tight radius the nearest is too far -> miss.
    assert nearest_point(pts, (1, 0), max_dist=0.5) == -1
    # Exactly at a point.
    assert nearest_point(pts, (10, 10)) == 2


def test_nearest_point_empty():
    assert nearest_point(np.empty((0, 2)), (0, 0)) == -1
    assert nearest_point(np.empty((0, 2)), (0, 0), max_dist=10.0) == -1


# --------------------------------------------------------------------------------- project_points
def _unit_cube():
    """8 vertices of a cube at +/-0.5 on each axis (all strictly inside the [-1,1] frustum)."""
    return np.array([
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [-0.5,  0.5, -0.5],
        [-0.5,  0.5,  0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5, -0.5],
        [ 0.5,  0.5,  0.5],
    ], dtype=float)


def test_project_cube_identity_pixels_and_visibility():
    cube = _unit_cube()
    mvp = np.eye(4)
    w, h = 200, 100

    pts2d, visible = project_points(cube, mvp, (w, h))

    assert pts2d.shape == (8, 2)
    assert visible.shape == (8,)
    # Every vertex of the +/-0.5 cube is inside the frustum and in front of the camera.
    assert visible.all()

    # With identity MVP, NDC == world coords, so the pixel mapping is exact:
    #   x_px = (x*0.5 + 0.5) * w ;  y_px = (1 - (y*0.5 + 0.5)) * h
    expected_x = (cube[:, 0] * 0.5 + 0.5) * w
    expected_y = (1.0 - (cube[:, 1] * 0.5 + 0.5)) * h
    np.testing.assert_allclose(pts2d[:, 0], expected_x)
    np.testing.assert_allclose(pts2d[:, 1], expected_y)

    # Spot-check two concrete vertices (non-tautological hard numbers, incl. the y-flip):
    #   (0.5, 0.5, 0.5) -> (150, 25) ;  (-0.5, -0.5, -0.5) -> (50, 75)
    np.testing.assert_allclose(pts2d[7], [150.0, 25.0])
    np.testing.assert_allclose(pts2d[0], [50.0, 75.0])


def test_project_out_of_frustum_not_visible():
    # Identity MVP: a point with |ndc| > 1 on any axis must be flagged not visible.
    pts = np.array([
        [0.0, 0.0, 0.0],   # dead-center -> visible
        [2.0, 0.0, 0.0],   # x NDC = 2   -> not visible
        [0.0, -1.5, 0.0],  # y NDC = -1.5-> not visible
        [0.0, 0.0, 3.0],   # z NDC = 3   -> not visible
    ])
    _, visible = project_points(pts, np.eye(4), (256, 256))
    np.testing.assert_array_equal(visible, [True, False, False, False])


def test_project_behind_camera_w_negative_not_visible():
    # MVP whose 4th row maps clip-w := z. A point with z < 0 then has clip w < 0 (behind camera)
    # and must be invisible regardless of where it lands in NDC.
    mvp = np.eye(4)
    mvp[3] = [0.0, 0.0, 1.0, 0.0]   # w_clip = z

    pts = np.array([
        [0.0, 0.0,  2.0],   # in front: w = +2 -> visible
        [0.0, 0.0, -2.0],   # behind  : w = -2 -> NOT visible
    ])
    pts2d, visible = project_points(pts, mvp, (200, 100))

    # Sanity: the front point's clip w is positive, the behind point's is negative.
    homog = np.column_stack([pts, np.ones(2)])
    w_clip = (homog @ mvp.T)[:, 3]
    assert w_clip[0] > 0 and w_clip[1] < 0

    np.testing.assert_array_equal(visible, [True, False])


def test_project_empty_shapes():
    pts2d, visible = project_points(np.empty((0, 3)), np.eye(4), (100, 100))
    assert pts2d.shape == (0, 2)
    assert visible.shape == (0,)
    assert visible.dtype == bool


if __name__ == "__main__":  # convenience for direct invocation
    raise SystemExit(pytest.main([__file__, "-q"]))
