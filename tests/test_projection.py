# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.projection -- the 4 scene projections (pure numpy, headless)."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import projection as P


def test_modes_and_labels_consistent():
    assert P.DEFAULT_MODE in P.MODES
    assert set(P.LABELS) == set(P.MODES)


def test_greenwich_is_the_original_mapping():
    lon = np.array([-170.0, 0.0, 170.0])
    lat = np.array([10.0, -20.0, 30.0])
    hgt = np.array([0.0, -100.0, -700.0])                   # below the surface -> negative height
    xyz = P.project(lon, lat, hgt, mode="greenwich", vexag=2.0)
    assert xyz.shape == (3, 3)
    assert np.allclose(xyz[:, 0], lon)
    assert np.allclose(xyz[:, 1], lat)
    assert np.allclose(xyz[:, 2], hgt / P.KM_PER_DEG * 2.0)


def test_pacific_makes_the_dateline_contiguous():
    # two events either side of the +/-180 seam (Kermadec/Tonga). Greenwich puts them ~350 deg apart;
    # Pacific must place them within a few degrees.
    lon = np.array([-178.0, 179.0])
    out = P.project(lon, [0.0, 0.0], [0.0, 0.0], mode="pacific")
    assert np.allclose(out[:, 0], [182.0, 179.0])          # -178 -> 182
    assert abs(out[0, 0] - out[1, 0]) < 5.0                 # contiguous, not ~357 apart
    gw = P.project(lon, [0.0, 0.0], [0.0, 0.0], mode="greenwich")
    assert abs(gw[0, 0] - gw[1, 0]) > 350.0                 # the bug pacific fixes


def test_mercator_pacific_centered_and_monotonic_in_lat():
    lat = np.array([-60.0, 0.0, 60.0])
    out = P.project([200.0, 200.0, 200.0], lat, [0, 0, 0], mode="mercator")
    assert out[1, 1] == pytest.approx(0.0)                  # equator -> 0
    assert out[0, 1] < out[1, 1] < out[2, 1]               # y increases with latitude
    assert np.allclose(out[:, 0], P._pacific_lon(np.array([200.0, 200.0, 200.0])))
    # extreme latitudes are clamped (finite), not +/-inf
    assert np.isfinite(P.project([0], [89.9], [0], mode="mercator")).all()


def test_globe_is_wgs84_ellipsoid():
    # surface points: equatorial radius = a, polar radius = b (oblate, a != b)
    eq = P.project([0.0], [0.0], [0.0], mode="globe")[0]
    po = P.project([0.0], [90.0], [0.0], mode="globe")[0]
    assert np.linalg.norm(eq) == pytest.approx(P.WGS84_A, abs=1e-6)        # 6378.137 km
    assert np.linalg.norm(po) == pytest.approx(P.WGS84_B, abs=1e-6)        # 6356.752 km
    assert P.WGS84_A - P.WGS84_B == pytest.approx(21.385, abs=0.01)        # the ~21 km flattening
    # at the equator the geometry is a clean radius, so height adds straight on
    eq_deep = P.project([0.0], [0.0], [-100.0], mode="globe", vexag=1.0)[0]
    assert np.linalg.norm(eq_deep) == pytest.approx(P.WGS84_A - 100.0, abs=1e-6)
    # vexag scales the radial displacement at the equator
    eq_vex = P.project([0.0], [0.0], [-100.0], mode="globe", vexag=3.0)[0]
    assert np.linalg.norm(eq_vex) == pytest.approx(P.WGS84_A - 300.0, abs=1e-6)
    # a sphere would put a 45-deg surface point at radius a; the ellipsoid is strictly inside that
    mid = P.project([10.0], [45.0], [0.0], mode="globe")[0]
    assert P.WGS84_B < np.linalg.norm(mid) < P.WGS84_A


def test_display_lon_layout_per_mode():
    lon = np.array([-179.0, -1.0, 0.0, 1.0, 179.0])
    assert np.allclose(P.display_lon(lon, "greenwich"), lon)                  # seam at +/-180
    assert np.allclose(P.display_lon(lon, "pacific"), np.mod(lon, 360.0))     # seam in the Atlantic
    assert np.allclose(P.display_lon(lon, "mercator"), np.mod(lon, 360.0))


def test_unproject_round_trips():
    lon = np.array([-170.0, -40.0, 0.0, 60.0, 175.0])
    lat = np.array([-55.0, -10.0, 0.0, 25.0, 70.0])
    for mode in ("greenwich", "pacific", "mercator", "globe"):
        xyz = P.project(lon, lat, np.zeros_like(lon), mode=mode)
        ll = P.unproject(xyz, mode)
        # longitudes compared modulo 360 (pacific/mercator relabel the seam)
        dlon = (ll[:, 0] - lon + 180.0) % 360.0 - 180.0
        assert np.allclose(dlon, 0.0, atol=1e-6), mode
        assert np.allclose(ll[:, 1], lat, atol=1e-4), mode


def test_height_is_positive_up():
    """The rename is not cosmetic. Positive height must place a point ABOVE the ellipsoid, which is
    what lets a DEM or a peak sit on the globe at all -- EQSelect's depth_km could only go inward,
    because a hypocentre never has positive elevation."""
    surface = P.project([0.0], [0.0], [0.0], mode="globe")[0]
    peak = P.project([0.0], [0.0], [8.849], mode="globe")[0]          # Everest, km
    assert np.linalg.norm(peak) == pytest.approx(P.WGS84_A + 8.849, abs=1e-6)
    assert np.linalg.norm(peak) > np.linalg.norm(surface)
    # flat views agree that up is +Z
    assert P.project([0.0], [0.0], [5.0], mode="greenwich")[0, 2] > 0.0
    assert P.project([0.0], [0.0], [-5.0], mode="greenwich")[0, 2] < 0.0


def test_height_km_equals_negated_legacy_depth_km():
    """Numerically identical to the pre-rename behaviour for equivalent input: EQSelect's
    ``project(..., depth_km=d)`` is exactly this ``project(..., height_km=-d)``, in every mode.
    The rename changes the sign convention at the boundary, never the geometry."""
    lon = np.array([-170.0, 0.0, 60.0])
    lat = np.array([-30.0, 0.0, 45.0])
    depth = np.array([0.0, 100.0, 660.0])                            # legacy positive-down input
    for mode in ("greenwich", "pacific", "mercator"):
        out = P.project(lon, lat, -depth, mode=mode, vexag=2.0)
        legacy_z = -depth / P.KM_PER_DEG * 2.0                       # the old _z_depth formula
        assert np.allclose(out[:, 2], legacy_z), mode
    # globe: legacy h was -depth*vexag; with height_km it is height*vexag for height = -depth
    got = P.project([0.0], [0.0], [-660.0], mode="globe", vexag=1.0)[0]
    assert np.linalg.norm(got) == pytest.approx(P.WGS84_A - 660.0, abs=1e-6)


def test_globe_clamp_still_prevents_inversion():
    """The clamp that stopped a large vexag turning the ellipsoid inside out must survive the
    rename -- it now bounds height from below rather than depth from above."""
    out = P.project([0.0], [0.0], [-6000.0], mode="globe", vexag=100.0)[0]
    assert np.linalg.norm(out) > 0.0                                 # did not invert through centre
    assert np.linalg.norm(out) == pytest.approx(0.1 * P.WGS84_A, rel=1e-6)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        P.project([0.0], [0.0], [0.0], mode="orthographic")
