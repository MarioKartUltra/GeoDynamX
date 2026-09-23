# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Exact-port pins for dynamix.core.xsmurf_follow.

Every expected value below is hand-computed from the C in xSmurfMacPorts/xsmurf:
detection = ``w2_folow_contour`` + ``near_contour_line`` (wt2d/Extrema.c:1083/1105 -- kapap<0
AND a 4-neighbor kapa sign change with |kapa(p)| < |kapa(n)|); value channel =
``_get_interpolated_modulus_`` v1 (interpreter/wt2d_cmds.c:5170 -- the followVersion=1
DEFAULT: cardinal from ``_get_near_pos_`` by strongest opposite kapa, linear crossing ratio,
``_get_m2_`` cubic through 4 axis samples, sentinel/guard fallbacks); chaining =
``search_lines`` (wt2d/chain.c:605 -- phase 1 seeds line ENDS only until exhausted, phase 2
seeds the remaining CLOSED rings; ``_seek_neighbors_`` walks 8 neighbors in the verbatim
iPosArray/jPosArray priority, single successor, prepending; closed iff size>3 and the ends
are Chebyshev-adjacent). The wrapper parity file is the dataset-level oracle; these are the
unit semantics.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import xsmurf_follow as xf


# ------------------------------------------------------------------ detection

def test_near_contour_line_fires_on_the_smaller_side_of_a_cardinal_crossing():
    kapa = np.zeros((3, 3))
    kapa[1, 1] = 0.1
    kapa[0, 1] = -0.5                     # north neighbor: sign change, bigger |kapa|
    assert xf.near_contour_line(kapa, 1, 1)
    # the bigger side of the same crossing must NOT fire
    kapa2 = np.zeros((3, 3))
    kapa2[1, 1] = -0.5
    kapa2[0, 1] = 0.1
    assert not xf.near_contour_line(kapa2, 1, 1)


def test_near_contour_line_ignores_diagonal_crossings():
    kapa = np.zeros((3, 3))
    kapa[1, 1] = 0.1
    kapa[0, 0] = -0.5                     # diagonal only
    assert not xf.near_contour_line(kapa, 1, 1)


def test_follow_contour_detects_kapap_gate_and_borders():
    n = 5
    kapa = np.zeros((n, n)); kapap = np.full((n, n), -1.0)
    kapa[2, 2] = 0.1; kapa[1, 2] = -0.5           # crossing at (2,2), cardinal N
    ys, xs = xf.follow_contour(kapa, kapap)
    assert (2, 2) in set(zip(ys.tolist(), xs.tolist()))
    # kapap >= 0 kills it
    kapap2 = np.abs(kapap)
    ys2, _xs2 = xf.follow_contour(kapa, kapap2)
    assert ys2.size == 0
    # the same crossing moved to the border row is skipped (C loops 1..n-2)
    kapa3 = np.zeros((n, n)); kapa3[0, 2] = 0.1; kapa3[1, 2] = -0.5
    ys3, xs3 = xf.follow_contour(kapa3, kapap)
    assert (0, 2) not in set(zip(ys3.tolist(), xs3.tolist()))


# ------------------------------------------------------------- value channel

def test_get_m2_matches_the_c_cubic_on_a_linear_ramp():
    # y = (0, 1, 2, 3): k1 = (-0 + 6*2 - 3*1 - 3)/6 = 1, k2 = 0, k3 = 0 -> f(s) = 1 + s
    assert xf.get_m2(0.0, 1.0, 2.0, 3.0, 0.5) == pytest.approx(1.5)
    assert xf.get_m2(0.0, 1.0, 2.0, 3.0, 0.0) == pytest.approx(1.0)


def test_get_m2_negative_value_returns_the_c_sentinel():
    assert xf.get_m2(0.0, -5.0, -5.0, 0.0, 0.5) == -4.0


def test_interpolated_modulus_falls_back_at_borders_and_on_guard():
    n = 7
    mod = np.full((n, n), 2.0)
    kapa = np.zeros((n, n))
    # border pixel (x < 2): grid modulus verbatim
    assert xf.interpolated_modulus(mod, kapa, 1, 3) == 2.0
    # no crossing anywhere: _get_near_pos_ returns 0 -> C uses pos_incr[0] (the NW quirk);
    # flat mod means the cubic returns the flat value -- still 2.0 either way
    assert xf.interpolated_modulus(mod, kapa, 3, 3) == pytest.approx(2.0)


def test_interpolated_modulus_cubic_along_the_chosen_cardinal():
    n = 9
    mod = np.zeros((n, n))
    # along row 4 (the E-W axis): mod = 0,1,2,3,... -> linear ramp; crossing to the EAST
    for x in range(n):
        mod[4, x] = float(x)
    kapa = np.zeros((n, n))
    kapa[4, 4] = 0.1
    kapa[4, 5] = -0.3                     # east: ratio = |0.1 / (0.1 - (-0.3))| = 0.25
    # samples along E: y_1 = mod[4,3] = 3, y0 = 4, y1 = 5, y2 = 6 -> f(s) = 4 + s
    assert xf.interpolated_modulus(mod, kapa, 4, 4) == pytest.approx(4.25)


# ------------------------------------------------------------------ chaining

def test_open_line_is_seeded_at_an_end_and_ordered_by_the_prepend_walk():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 1] = mask[2, 2] = mask[2, 3] = True
    lines = xf.search_lines(mask)
    assert len(lines) == 1
    (pts, closed) = lines[0]
    assert not closed
    # seed = (2,1) (first end in raster order); _seek_neighbors_ PREPENDS, so the seed
    # ends the list: walk right, list built right-to-left.
    assert pts == [(2, 3), (2, 2), (2, 1)]


def test_closed_ring_is_one_closed_line():
    mask = np.zeros((5, 5), dtype=bool)
    for y, x in ((1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1)):
        mask[y, x] = True
    lines = xf.search_lines(mask)
    assert len(lines) == 1
    pts, closed = lines[0]
    assert closed and len(pts) == 8
    # every consecutive pair is 8-adjacent, and so are the ends (that IS the closed test)
    ring = pts + [pts[0]]
    for (y0, x0), (y1, x1) in zip(ring, ring[1:]):
        assert max(abs(y0 - y1), abs(x0 - x1)) == 1


def test_two_adjacent_neighbors_make_a_kink_end():
    # p at (2,2) with neighbors (2,1)-W and (1,1)-NW: those two are 4-adjacent to each
    # other, so p is a line END by the C's kink rule (wt2d/chain.c _is_line_end_).
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = mask[2, 1] = mask[1, 1] = True
    assert xf.is_line_end(mask, 2, 2)
    # but with neighbors W and E (not adjacent to each other) it is interior
    mask2 = np.zeros((5, 5), dtype=bool)
    mask2[2, 2] = mask2[2, 1] = mask2[2, 3] = True
    assert not xf.is_line_end(mask2, 2, 2)


def test_neighbor_priority_arrays_are_the_verbatim_c_tables():
    # wt2d/chain.c: iPosArray adds to pos%lx (COLUMN), jPosArray to pos/lx (ROW) --
    # ported verbatim, never renamed into compass words (the C's own comment mislabels
    # them; the arrays are the truth the walk executes).
    assert xf.I_POS_ARRAY == [-1, 0, 0, 1, -1, -1, 1, 1]
    assert xf.J_POS_ARRAY == [0, -1, 1, 0, -1, 1, -1, 1]


# ------------------------------------------------- the backend adapter (exact detector)

def _cross_grid(n=9):
    """kapa with a vertical zero-crossing between columns 3(+) and 4(-): the registered
    side is column 3 (|0.1| < |0.4|), rows 1..n-2 -- a vertical open line."""
    kapa = np.zeros((n, n)); kapap = np.full((n, n), -1.0)
    kapa[:, 3] = 0.1
    kapa[:, 4] = -0.4
    mod = np.full((n, n), 2.0)
    arg = np.zeros((n, n))
    return mod, arg, kapa, kapap


def test_exact_scale_schema_and_line():
    mod, arg, kapa, kapap = _cross_grid()
    e = xf.follow_extrema_scale_exact(mod, arg, kapa, kapap, thresh=0.0)
    assert set(e) >= {"x", "y", "mod", "arg", "line_id", "x_sub", "y_sub"}
    assert set(e["x"].tolist()) == {3}
    assert set(e["y"].tolist()) == set(range(1, 8))
    assert np.all(e["line_id"] == e["line_id"][0]) and e["line_id"][0] >= 0
    # crossing offset: ratio = |0.1/(0.1-(-0.4))| = 0.2 toward EAST -> x_sub = 3.2
    np.testing.assert_allclose(e["x_sub"], 3.2)
    np.testing.assert_allclose(e["y_sub"], e["y"].astype(float))
    # runs: one open run over the 7 points, xsmurf walk order (seed last), not closed
    assert len(e["_xs_runs"]) == 1 and e["_xs_closed"] == [False]
    run = e["_xs_runs"][0]
    assert sorted(run.tolist()) == list(range(7))


def test_exact_scale_moduli_match_the_scalar_port():
    rng = np.random.default_rng(3)
    n = 16
    mod = np.abs(rng.normal(2.0, 0.3, (n, n)))
    kapa = rng.normal(0.0, 0.3, (n, n))
    kapap = np.full((n, n), -1.0)
    arg = np.zeros((n, n))
    e = xf.follow_extrema_scale_exact(mod, arg, kapa, kapap, thresh=0.0)
    expect = [xf.interpolated_modulus(mod, kapa, int(x), int(y))
              for x, y in zip(e["x"], e["y"])]
    np.testing.assert_allclose(e["mod"], expect, rtol=1e-12)


def test_exact_scale_thresh_is_fraction_of_max():
    mod, arg, kapa, kapap = _cross_grid()
    mod[2, 3] = 0.01                             # one weak point on the line
    e = xf.follow_extrema_scale_exact(mod, arg, kapa, kapap, thresh=0.5)
    assert (3, 2) not in set(zip(e["x"].tolist(), e["y"].tolist()))
    assert len(e["x"]) == 6


def test_exact_scale_closed_ring_run():
    n = 13
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - 6.2, xx - 5.8)             # off-center: no knife-edge |kapa| ties
    kapa = 3.3 - r                               # zero circle of radius 3.3
    kapap = np.full((n, n), -1.0)
    mod = np.full((n, n), 1.0); arg = np.zeros((n, n))
    e = xf.follow_extrema_scale_exact(mod, arg, kapa, kapap, thresh=0.0)
    assert len(e["_xs_runs"]) >= 1
    big = max(range(len(e["_xs_runs"])), key=lambda i: e["_xs_runs"][i].size)
    assert e["_xs_closed"][big] is True
