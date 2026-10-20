# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The CWT leaders' kernel (spec 30 §3.3): the exact disc maximum and box sum, and the cylinder
supremum as the disc dilation of the cumulative scale maximum, pinned against brute force."""
from __future__ import annotations

import numpy as np
import pytest


def _brute_disc(src, R, reduce):
    """``reduce`` over every pixel within |Δx|² ≤ R² of each pixel, NaN pixels left out; NaN where
    the disc holds no finite pixel."""
    ny, nx = src.shape
    out = np.full(src.shape, np.nan, dtype=np.float64)
    r = int(np.floor(R))
    for y in range(ny):
        for x in range(nx):
            vals = []
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy <= R * R and 0 <= y + dy < ny and 0 <= x + dx < nx:
                        v = src[y + dy, x + dx]
                        if np.isfinite(v):
                            vals.append(v)
            if vals:
                out[y, x] = reduce(vals)
    return out


@pytest.mark.parametrize("R", [1.0, 2.58, 7.3, 31.0])
def test_disc_dilate_is_the_exact_disc_maximum(R):
    from dynamix.core.cwt_leaders import disc_dilate

    rng = np.random.default_rng(int(R * 10))
    src = rng.standard_normal((40, 52)).astype(np.float32)
    src[3:6, 10:14] = np.nan                                  # a hole: left out, never spread
    got = disc_dilate(src, R)
    want = _brute_disc(src.astype(np.float64), R, max)
    assert got.dtype == np.float32 and got.shape == src.shape
    np.testing.assert_allclose(got, want, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("R", [1.0, 2.58, 7.3])
def test_disc_boxsum_is_the_exact_disc_sum_of_finite_pixels(R):
    from dynamix.core.cwt_leaders import disc_boxsum

    rng = np.random.default_rng(7)
    src = rng.random((33, 29)).astype(np.float32)
    src[20, 5] = np.nan
    got_sum, got_count = disc_boxsum(src, R)
    want_sum = _brute_disc(src.astype(np.float64), R, sum)
    want_count = _brute_disc(np.where(np.isfinite(src), 1.0, np.nan), R, sum)
    np.testing.assert_allclose(got_sum, np.nan_to_num(want_sum), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(got_count, np.nan_to_num(want_count).astype(np.int64))


def test_cylinder_leaders_are_the_disc_dilation_of_the_cumulative_scale_max():
    """L(x, a_j) = max over a_i ≤ a_j and |x' − x| ≤ R_j of M(a_i, x'): the running max over the
    finer scales first, one disc dilation per scale after; a point peak at the finest scale is seen
    by the coarse leader through the coarse disc even though it lies outside the fine disc (the
    cylinder, not the cone)."""
    from dynamix.core.cwt_leaders import cylinder_leaders, disc_dilate

    rng = np.random.default_rng(3)
    mod = rng.random((3, 24, 24)).astype(np.float32) * 0.1
    radii = np.array([1.0, 2.0, 4.0])
    mod[0, 12, 12] = 5.0                                      # a sharp peak at the finest scale
    got = cylinder_leaders(mod, radii, sup_from=0)
    cum = np.maximum.accumulate(mod, axis=0)
    for j in range(3):
        np.testing.assert_array_equal(got[j], disc_dilate(cum[j], radii[j]))
    # the coarse leader at (12, 15) is 3 px from the peak: inside R_2 = 4 (cylinder) and outside
    # R_0 = 1, the cone's radius at the peak's own scale
    assert got[2, 12, 15] == 5.0 and got[0, 12, 15] < 1.0
    # a scale cut from the running max is not seen at all
    cut = cylinder_leaders(mod, radii, sup_from=1)
    assert cut[2, 12, 12] < 1.0


def test_cylinder_leaders_refuse_a_cut_that_leaves_nothing():
    from dynamix.core.cwt_leaders import cylinder_leaders

    with pytest.raises(ValueError, match="sup_from"):
        cylinder_leaders(np.zeros((3, 8, 8), dtype=np.float32), np.array([1.0, 2.0, 4.0]), sup_from=3)
