# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.transect -- pure numpy/scipy,
no Qt. Exercises the three math primitives directly: bilinear profile sampling (exact on a
synthetic affine ramp, NaN off-grid), the four smoothing kinds (shape + NaN-bridging), and the
point-to-segment swath selection. `orient_endpoints`'s auto-orientation rule is also covered here
at the pure-function level; `tests/test_canvas_pick.py` covers the same rule end to end through
the two-click canvas gesture.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.transect import (SMOOTHINGS, chains_in_buffer, orient_endpoints,
                                   sample_profile, smooth_profile)

# ------------------------------------------------------------------------------- orient_endpoints


def test_orient_endpoints_left_to_right_swaps_a_right_to_left_click():
    a, b = orient_endpoints((5.0, 5.0), (1.0, 5.0))
    assert a == (1.0, 5.0) and b == (5.0, 5.0)


def test_orient_endpoints_left_to_right_leaves_an_already_ordered_pair():
    a, b = orient_endpoints((1.0, 5.0), (5.0, 5.0))
    assert a == (1.0, 5.0) and b == (5.0, 5.0)


def test_orient_endpoints_bottom_to_top_swaps_when_clicked_top_first():
    """Canvas y grows DOWNWARD: 'bottom' (start) is the LARGER y, 'top' (end) the SMALLER y."""
    a, b = orient_endpoints((3.0, 2.0), (3.0, 9.0))
    assert a == (3.0, 9.0) and b == (3.0, 2.0)


def test_orient_endpoints_bottom_to_top_leaves_an_already_ordered_pair():
    a, b = orient_endpoints((3.0, 9.0), (3.0, 2.0))
    assert a == (3.0, 9.0) and b == (3.0, 2.0)


def test_orient_endpoints_tie_goes_to_the_left_right_branch():
    a, b = orient_endpoints((0.0, 0.0), (5.0, 5.0))            # |dx| == |dy|
    assert a == (0.0, 0.0) and b == (5.0, 5.0)


def test_orient_endpoints_degenerate_point_is_unchanged():
    a, b = orient_endpoints((2.0, 2.0), (2.0, 2.0))
    assert a == (2.0, 2.0) and b == (2.0, 2.0)


# ------------------------------------------------------------------------------- sample_profile


def _ramp(ny=5, nx=6):
    """An affine field, z[row, col] = 2*col + 3*row -- bilinear interpolation reproduces an
    affine function EXACTLY at every interior point, so the sampled profile can be checked
    against the analytic formula rather than an approximate/regression value."""
    return np.fromfunction(lambda r, c: 2.0 * c + 3.0 * r, (ny, nx), dtype=float)


def test_sample_profile_matches_the_analytic_line_on_an_affine_ramp():
    z = _ramp()
    a, b, n = (1.0, 1.0), (4.0, 3.0), 9
    dist, zvals = sample_profile(z, a, b, n=n)

    t = np.linspace(0.0, 1.0, n)
    px = a[0] + (b[0] - a[0]) * t
    py = a[1] + (b[1] - a[1]) * t
    expected_z = 2.0 * px + 3.0 * py
    expected_dist = np.hypot(px - a[0], py - a[1])

    assert dist.shape == (n,) and zvals.shape == (n,)
    np.testing.assert_allclose(zvals, expected_z)
    np.testing.assert_allclose(dist, expected_dist)
    assert dist[0] == 0.0


def test_sample_profile_scales_distance_by_spacing():
    z = _ramp()
    dist_px, _ = sample_profile(z, (0.0, 0.0), (4.0, 0.0), n=5, spacing=1.0)
    dist_scaled, _ = sample_profile(z, (0.0, 0.0), (4.0, 0.0), n=5, spacing=2.5)
    np.testing.assert_allclose(dist_scaled, dist_px * 2.5)


def test_sample_profile_is_nan_off_grid_and_finite_on_grid():
    """ny, nx = 5, 6 -> valid x in [0, 5], valid y in [0, 4]. A horizontal segment run from x=4 to
    x=8 stays on-grid through x=5 (the exact edge) and goes off-grid (NaN) for x > 5."""
    z = _ramp(ny=5, nx=6)
    dist, zvals = sample_profile(z, (4.0, 1.0), (8.0, 1.0), n=5)          # x = 4, 5, 6, 7, 8

    assert np.isfinite(zvals[0]) and np.isfinite(zvals[1])                # x = 4, 5: on-grid
    assert np.isnan(zvals[2:]).all()                                      # x = 6, 7, 8: off-grid
    np.testing.assert_allclose(zvals[0], 2.0 * 4.0 + 3.0 * 1.0)
    np.testing.assert_allclose(zvals[1], 2.0 * 5.0 + 3.0 * 1.0)
    assert np.isfinite(dist).all()                                        # distance is defined
                                                                            # regardless of NaN z


# ------------------------------------------------------------------------------- smooth_profile


def test_smooth_profile_none_returns_an_unmodified_copy():
    z = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    out = smooth_profile(z, "none")
    np.testing.assert_array_equal(out, z)
    out[0] = 99.0
    assert z[0] == 1.0                            # a copy, not the same array


def test_smooth_profile_unknown_kind_raises():
    with pytest.raises(ValueError):
        smooth_profile(np.array([1.0, 2.0, 3.0]), "bogus")


@pytest.mark.parametrize("kind", ["gaussian", "median", "savgol"])
def test_smooth_profile_round_trips_shape_and_bridges_nan(kind):
    z = np.array([1.0, 2.0, 3.0, np.nan, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    out = smooth_profile(z, kind, sigma=1.0, window=5)

    assert out.shape == z.shape
    nan_mask = ~np.isfinite(z)
    assert np.isnan(out[nan_mask]).all()          # originally-missing samples stay missing
    assert np.isfinite(out[~nan_mask]).all()       # every genuinely-sampled point is filtered,
                                                    # not itself turned into NaN by the gap beside it


def test_smooth_profile_all_smoothings_tuple_is_exhaustive_over_the_parametrized_kinds():
    assert SMOOTHINGS == ("none", "gaussian", "median", "savgol")


def test_smooth_profile_savgol_window_is_forced_odd_and_shorter_than_the_data():
    """window=200 on an 11-sample profile must not raise -- the ported EQSelect clamp keeps the
    window both ODD and no longer than the data."""
    z = np.linspace(0.0, 10.0, 11)
    out = smooth_profile(z, "savgol", window=200)
    assert out.shape == z.shape
    assert np.isfinite(out).all()


# ------------------------------------------------------------------------------- chains_in_buffer


def _chain(xs, ys):
    return {"x": np.array(xs, dtype=float), "y": np.array(ys, dtype=float)}


def test_chains_in_buffer_selects_only_chains_with_a_point_within_the_buffer():
    a, b = (0.0, 0.0), (10.0, 0.0)                 # a horizontal segment along y = 0
    chains = [
        _chain([5.0], [1.0]),                       # 0: 1 px away -- inside a 2px buffer
        _chain([5.0], [50.0]),                       # 1: far away -- outside
        _chain([50.0, 5.0], [50.0, 1.0]),             # 2: one far point, one close -- "any point"
        _chain([], []),                               # 3: empty -- never selected
    ]

    assert chains_in_buffer(chains, a, b, buffer_px=2.0) == [0, 2]


def test_chains_in_buffer_clamps_to_the_nearest_endpoint_past_the_segment():
    """A point beyond A' along the segment's own extension is measured against the ENDPOINT
    (round end-cap, the module's own documented v1 choice), not excluded outright."""
    a, b = (0.0, 0.0), (10.0, 0.0)
    chains = [_chain([12.0], [0.0])]                # 2 px past B, dead on the line

    assert chains_in_buffer(chains, a, b, buffer_px=3.0) == [0]
    assert chains_in_buffer(chains, a, b, buffer_px=1.0) == []


def test_chains_in_buffer_handles_a_degenerate_zero_length_segment():
    a = b = (5.0, 5.0)
    chains = [_chain([5.5], [5.0]), _chain([50.0], [50.0])]

    assert chains_in_buffer(chains, a, b, buffer_px=1.0) == [0]
