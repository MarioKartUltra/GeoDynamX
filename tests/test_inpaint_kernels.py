# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The inpainting engine's numba kernels against numpy references."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from dynamix.core.inpaint import _kernels  # noqa: E402

K = _kernels.kernels()


def test_block_mean_and_spread_are_the_reshape_mean_and_its_adjoint():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((12, 18))
    out = np.empty((4, 6))
    K["block_mean"](x, 3, out)
    np.testing.assert_allclose(out, x.reshape(4, 3, 6, 3).mean(axis=(1, 3)))
    r = rng.standard_normal((4, 6))
    back = np.empty((12, 18))
    K["block_spread"](r, 3, back)
    assert abs(np.vdot(out, r) - np.vdot(x, back)) < 1e-12


def test_bilinear_sample_and_splat_are_adjoint_and_nan_out_of_range():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((20, 30))
    rows = rng.uniform(0, 19, 200)
    cols = rng.uniform(0, 29, 200)
    out = np.empty(200)
    K["bilinear_sample"](x, rows, cols, out)
    i0, j0 = np.floor(rows).astype(int), np.floor(cols).astype(int)
    i1, j1 = np.minimum(i0 + 1, 19), np.minimum(j0 + 1, 29)
    fr, fc = rows - i0, cols - j0
    ref = ((1 - fr) * (1 - fc) * x[i0, j0] + (1 - fr) * fc * x[i0, j1]
           + fr * (1 - fc) * x[i1, j0] + fr * fc * x[i1, j1])
    np.testing.assert_allclose(out, ref)
    r = rng.standard_normal(200)
    back = np.zeros((20, 30))
    K["bilinear_splat"](r, rows, cols, back)
    assert abs(np.vdot(out, r) - np.vdot(x, back)) < 1e-10
    K["bilinear_sample"](x, np.array([-0.5, 19.5]), np.array([3.0, 3.0]), out[:2])
    assert np.isnan(out[:2]).all()


def test_integral_and_window_any_find_a_mask_within_a_half_width():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5, 9] = 1
    integ = np.zeros((17, 17), dtype=np.int64)
    K["integral"](mask, integ)
    assert integ[-1, -1] == 1 and integ[6, 10] == 1 and integ[5, 10] == 0
    out = np.zeros((8, 8), dtype=np.uint8)
    K["window_any"](integ, 1, 2, 0, out)        # coefficient (i, j) covers pixels 2i..2i+1 +- 1
    hit = np.argwhere(out)
    assert set(map(tuple, hit)) == {(2, 4), (3, 4), (2, 5), (3, 5)}


def test_clear_pairs_collects_log_ratios_of_clear_parent_child_pairs():
    parent = np.array([[2.0, 0.0], [1.0, 4.0]])
    child = np.ones((4, 4))
    child[0, 0] = 4.0
    clear_p = np.ones((2, 2), dtype=np.uint8)
    clear_c = np.ones((4, 4), dtype=np.uint8)
    clear_c[2:, :] = 0                                     # the second row of parents: dirty children
    out = np.empty(16)
    n = K["clear_pairs"](parent, child, clear_p, clear_c, 1e-8 * 4.0, 1e-8 * 4.0, out)
    assert n == 4                                           # parent (0,0) only: (0,1) is zero
    assert sorted(np.round(out[:n], 6)) == sorted(np.round(
        [np.log(4.0) - np.log(2.0)] + [np.log(1.0) - np.log(2.0)] * 3, 6))
    child[0, 1] = 1e-12                                     # under the floor: counts as zero
    n2 = K["clear_pairs"](parent, child, clear_p, clear_c, 1e-8 * 4.0, 1e-8 * 4.0, out)
    assert n2 == 3
    assert sorted(np.round(out[:n2], 6)) == sorted(np.round(
        [np.log(4.0) - np.log(2.0)] + [np.log(1.0) - np.log(2.0)] * 2, 6))


def test_cascade_level_infers_a_parent_from_its_children_by_maximum_likelihood():
    # an abacus concentrated at ln eta = ln 0.5: a parent should be twice its children
    edges = np.linspace(-3.0, 3.0, 201)
    hist = np.full((3, 200), 1e-6)
    centre = np.searchsorted(edges, np.log(0.5)) - 1
    hist[:, centre] = 1.0
    log_hist = np.log(hist)
    parent = np.zeros((3, 2, 2))
    parent[0, 0, 0] = 1.0                                   # p0 = 1 with rho = 1: fully inferred
    parent[0, 1, 1] = -1.0
    child = np.full((3, 4, 4), 3.0)                        # children 3 -> parent 6
    rho = np.ones((2, 2))
    rho[1, 1] = 0.5
    cmax = np.array([3.0, 3.0, 3.0])
    K["cascade_level"](parent, child, rho, log_hist, edges, np.full(3, -3.0), np.full(3, 3.0),
                       cmax, 64)
    assert parent[0, 0, 0] == pytest.approx(6.0, rel=0.05)
    assert parent[0, 1, 1] == pytest.approx(0.5 * -1.0 + 0.5 * -6.0, rel=0.05)   # sign kept
    assert parent[1, 0, 0] == pytest.approx(6.0, rel=0.05)  # p0 = 0: the children's sum signs it
    assert parent[2, 1, 0] == pytest.approx(6.0, rel=0.05)


def test_synth_level_draws_children_under_set_parents_inside_the_region():
    rng = np.random.default_rng(4)
    edges = np.linspace(-3.0, 3.0, 201)
    cdf = np.tile(np.linspace(0.0, 1.0, 200), (3, 1))     # uniform on [-3, 3]
    parent = np.zeros((3, 2, 2))
    parent[:, 0, 0] = 2.0
    child = np.zeros((3, 4, 4))
    rho = np.zeros((3, 4, 4))                               # per orientation: each band is rolled by its shift
    rho[:, :2, :2] = 1.0                                    # the children of parent (0, 0)
    u1, u2 = rng.random((3, 4, 4)), rng.random((3, 4, 4))
    K["synth_level"](parent, child, rho, cdf, edges, np.ones(3), u1, u2)
    assert (child[:, :2, :2] != 0).all() and (child[:, 2:, :] == 0).all()
    assert (np.abs(child[:, :2, :2]) <= 2.0 * np.exp(3.0) + 1e-9).all()
    assert (np.sign(child[:, :2, :2]) == 1.0).all()         # sign_freq 1: the parent's sign


def test_heldout_blocks_are_clear_and_apart():
    valid = np.ones((64, 64), dtype=np.uint8)
    valid[20:30, :] = 0
    rng = np.random.default_rng(5)
    cand = rng.permutation(64 * 64).astype(np.int64)
    out = np.zeros((64, 64), dtype=np.uint8)
    k = K["heldout_blocks"](valid, cand, 8, 4, 6, out)
    assert 1 <= k <= 6 and out.sum() == 64 * k
    assert not (out[20:30, :]).any()
    from scipy.ndimage import label

    labels, n = label(out)
    assert n == k
    for a in range(1, n + 1):
        ra, ca = np.nonzero(labels == a)
        for b in range(a + 1, n + 1):
            rb, cb = np.nonzero(labels == b)
            row_gap = max(rb.min() - ra.max() - 1, ra.min() - rb.max() - 1)
            col_gap = max(cb.min() - ca.max() - 1, ca.min() - cb.max() - 1)
            assert max(row_gap, col_gap) >= 4           # apart by the margin on one axis


def test_fill_polygon_rasterises_even_odd_and_mask_edges_outline_it():
    out = np.zeros((10, 12), dtype=np.int32)
    pts = np.array([[2.0, 2.0], [8.0, 2.0], [8.0, 6.0], [2.0, 6.0]])   # (col, row)
    K["fill_polygon"](pts, out, 3)
    # even-odd at pixel centres: rows 2..5 and cols 2..7 are inside, the far edges are not
    assert out[2:6, 2:8].all() and out.sum() == 3 * 4 * 6
    assert out[1, 5] == 0 and out[6, 5] == 0 and out[3, 8] == 0
    xs, ys = np.empty(4 * out.size), np.empty(4 * out.size)
    n = K["mask_edges"](out != 0, xs, ys)
    assert n % 3 == 0 and n // 3 == 2 * 4 + 2 * 6
    assert np.isnan(xs[2]) and xs[0] == pytest.approx(1.5) and ys[0] == pytest.approx(1.5)


def test_missing_numba_names_it(monkeypatch):
    import sys

    monkeypatch.setattr(_kernels, "_KERNELS", None)
    monkeypatch.setitem(sys.modules, "numba", None)
    with pytest.raises(RuntimeError, match="numba"):
        _kernels.kernels()
