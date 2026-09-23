# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.extrema_io -- WTMM extrema .npz loading, height-positive-up.

The vertical convention is the point of these tests. EQSelect returned positive-DOWN depth here,
because a hypocentre is never above the surface. DynamiX carries DEMs and bathymetry, so the whole
package is positive-UP with zero at the reference surface, and a loader that disagreed with
``projection.project`` would mirror every chain through the ellipsoid.
"""
from __future__ import annotations

import numpy as np

from dynamix.core.extrema_io import flatten_extrema, load_topo_extrema


def _v2_npz(tmp_path, *, with_nodes: bool = False):
    """A miniature of topo_wtmm.export_extrema's v2 key set, with POSITIVE elev_m (above sea level)."""
    p = tmp_path / "v2.npz"
    arrays = dict(
        h_xyz=np.array([[10.0, 20.0, 1000.0], [10.1, 20.0, 1100.0]], np.float32),
        h_off=np.array([0, 2], np.int64),
        h_scale=np.array([0], np.int32),
        h_len=np.array([2], np.int32),
        v_xyz=np.array([[10.0, 20.0, -2500.0], [10.0, 20.1, -2400.0]], np.float32),  # bathymetry
        v_off=np.array([0, 2], np.int64),
        v_persist=np.array([2], np.int32),
        scales=np.array([1.0, 2.0]),
        n_scales=2,
        region="golden",
    )
    if with_nodes:
        arrays.update(
            schema_version=2,
            node_xyz=np.array([[10.0, 20.0, 8849.0]], np.float32),   # Everest, metres
            node_scale=np.array([0], np.int32),
            node_hchain=np.array([0], np.int64),
            node_parent=np.array([-1], np.int64),
            node_root=np.array([0], np.int64),
            node_depth=np.array([0], np.int64),
            n_nodes=1,
            n_trees=1,
        )
    np.savez_compressed(p, **arrays)
    return p


def test_elevation_above_sea_level_loads_as_positive_height(tmp_path):
    """1000 m above the reference surface is height_km = +1.0, not depth_km = -1.0."""
    te = load_topo_extrema(_v2_npz(tmp_path))
    z = te["h_segments"][0][:, 2]
    np.testing.assert_allclose(z, [1.0, 1.1])


def test_bathymetry_loads_as_negative_height(tmp_path):
    """2500 m below the reference surface is height_km = -2.5. Sign is preserved, not flipped."""
    te = load_topo_extrema(_v2_npz(tmp_path))
    z = te["v_segments"][0][:, 2]
    np.testing.assert_allclose(z, [-2.5, -2.4])


def test_node_points_use_the_same_convention(tmp_path):
    """The schema-v2 cascade node table must not disagree with the segments it indexes into."""
    te = load_topo_extrema(_v2_npz(tmp_path, with_nodes=True))
    np.testing.assert_allclose(te["node_pts"][:, 2], [8.849])
    assert te["n_nodes"] == 1 and te["n_trees"] == 1


def test_height_agrees_with_the_globe_projection(tmp_path):
    """The reason the convention matters: loader output fed straight to project() must place a
    peak ABOVE the ellipsoid. Under the old positive-down convention this sat below it."""
    from dynamix.core import projection as P

    te = load_topo_extrema(_v2_npz(tmp_path))
    lon, lat, height = te["h_segments"][0][0]
    r = np.linalg.norm(P.project([lon], [lat], [height], mode="globe")[0])
    assert r > P.WGS84_B, "a summit must sit outside the ellipsoid's polar radius"
    surface = np.linalg.norm(P.project([lon], [lat], [0.0], mode="globe")[0])
    assert r == np.float64(r) and r > surface


def test_flatten_preserves_the_chain_index_and_heights(tmp_path):
    """flatten_extrema only reshapes; it must not touch the vertical convention."""
    te = load_topo_extrema(_v2_npz(tmp_path))
    flat = flatten_extrema(te)
    np.testing.assert_allclose(flat["h_pts"][:, 2], [1.0, 1.1])
    np.testing.assert_array_equal(flat["h_chain"], [0, 0])
    assert flat["n_h_chains"] == 1 and flat["n_v_chains"] == 1


def test_short_segments_are_dropped(tmp_path):
    """A polyline needs >= 2 points; a 1-point chain is not a chain. (Behaviour carried over
    unchanged from the original -- pinned here because extrema_io now has its own tests.)"""
    p = tmp_path / "short.npz"
    np.savez_compressed(
        p,
        h_xyz=np.array([[1.0, 2.0, 100.0]], np.float32),
        h_off=np.array([0, 1], np.int64),
        h_scale=np.array([0], np.int32),
        h_len=np.array([1], np.int32),
        v_xyz=np.zeros((0, 3), np.float32),
        v_off=np.array([0], np.int64),
        v_persist=np.zeros(0, np.int32),
        scales=np.array([1.0]), n_scales=1, region="short",
    )
    te = load_topo_extrema(p)
    assert te["h_segments"] == [] and te["v_segments"] == []
