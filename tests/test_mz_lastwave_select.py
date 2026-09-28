# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Reconstruction from selected maxima (``core.mz_lastwave.select``): the level states, the mirror
removals, the borrowed constraints, the α per chain and level 1's calibrated response."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import mz_lastwave as lw
from dynamix.core.mz_lastwave import select as S
from dynamix.devices.mz_edges import _line_ids


def _extrema(t, ex):
    out = []
    for l in range(1, ex.J + 1):
        mask, mag, arg = lw.primary_extrema(t, ex, l)
        y, x = np.nonzero(mask)
        out.append({"x": x.astype(np.int64), "y": y.astype(np.int64), "mod": mag[mask],
                    "arg": arg[mask], "line_id": _line_ids(mask)})
    return out


def _rough(ny=48, nx=40, seed=3):
    rng = np.random.default_rng(seed)
    return np.cumsum(np.cumsum(rng.standard_normal((ny, nx)), 0), 1)


def _table(J, **levels):
    table = S.parse_levels("", J)
    for key, (state, source) in levels.items():
        table["levels"][int(key[1:])] = {"state": state, "source": source}
    return table


def _all(extrema):
    return lambda l: np.ones(extrema[l - 1]["x"].size, bool)


def test_an_empty_table_is_every_level_own_and_round_trips():
    table = S.parse_levels("", 3)
    assert all(e == {"state": "own", "source": 2} for e in table["levels"].values())
    table["levels"][1] = {"state": "coder", "source": 2}
    table["filters"] = {2: {"hline_length": {"min_len": 5}}}
    assert S.parse_levels(S.dump_levels(table), 3) == table
    with pytest.raises(ValueError, match="unknown state"):
        S.parse_levels('{"levels": {"1": {"state": "maybe"}}}', 3)
    with pytest.raises(ValueError, match="the levels are 1..3"):
        S.parse_levels('{"levels": {"1": {"state": "near", "source": 5}}}', 3)


@pytest.mark.parametrize("border", ["mirror", "periodic"])
def test_a_pass_all_selection_reconstructs_bit_for_bit(border):
    f = _rough()
    J = 3
    t, ex = lw.analyze(f, J, border=border)
    extrema = _extrema(t, ex)
    table = S.parse_levels("", J)
    primary = S.primary_selection(extrema, f.shape, table, _all(extrema))
    sel = S.working_extrep(ex, extrema, f.shape, table, primary)
    for l in range(1, J + 1):
        assert np.array_equal(sel.mask[l], ex.mask[l])
        assert np.array_equal(sel.mag[l], ex.mag[l]) and np.array_equal(sel.arg[l], ex.arg[l])
    a = lw.e2recons(f, ex, t.S_full[J], J, k=3, border=border)[0]
    b = lw.e2recons(f, sel, t.S_full[J], J, k=3, border=border)[0]
    assert np.array_equal(a, b)


def test_a_removed_line_leaves_all_four_mirror_images():
    f = _rough()
    J = 3
    t, ex = lw.analyze(f, J, border="mirror")
    extrema = _extrema(t, ex)
    lid = extrema[2]["line_id"]
    longest = np.bincount(lid[lid >= 0]).argmax()
    keep = lid != longest

    def keep_of(l):
        return keep if l == 3 else np.ones(extrema[l - 1]["x"].size, bool)

    table = S.parse_levels("", J)
    sel = S.working_extrep(ex, extrema, f.shape, table,
                           S.primary_selection(extrema, f.shape, table, keep_of))
    Y, X = ex.mask[3].shape
    gone_y, gone_x = extrema[2]["y"][~keep], extrema[2]["x"][~keep]
    for ry, rx in ((gone_y, gone_x), (gone_y, -gone_x % X), (-gone_y % Y, gone_x),
                   (-gone_y % Y, -gone_x % X)):
        assert not sel.mask[3][ry, rx].any()
    assert sel.mask[3].sum() < ex.mask[3].sum()
    assert np.array_equal(sel.mask[2], ex.mask[2])


def test_near_at_radius_zero_onto_itself_is_its_own_selection():
    f = _rough()
    t, ex = lw.analyze(f, 3, border="periodic")
    extrema = _extrema(t, ex)
    rng = np.random.default_rng(0)
    own = [rng.random(e["x"].size) < 0.5 for e in extrema]
    table = _table(3, l2=("near", 2))
    primary = S.primary_selection(extrema, f.shape, table, lambda l: own[l - 1], radius=0)
    assert np.array_equal(primary[1], own[1])


def test_near_keeps_the_level_1_maxima_along_a_kept_level_2_edge_only():
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    f = (xx >= 32).astype(float) + ((yy - 16) ** 2 + (xx - 12) ** 2 <= 4) * 0.05
    t, ex = lw.analyze(f, 3, border="mirror")
    extrema = _extrema(t, ex)
    on_edge = [np.abs(e["x"] - 32) <= 1 for e in extrema]
    table = _table(3, l1=("near", 2), l2=("own", 2))
    primary = S.primary_selection(extrema, f.shape, table, lambda l: on_edge[l - 1])
    kept = primary[0]
    assert kept[on_edge[0]].all() and not kept[~on_edge[0]].any()


def test_a_coder_level_carries_its_own_transform_at_the_source_positions():
    f = _rough()
    J = 3
    t, ex = lw.analyze(f, J, border="mirror")
    extrema = _extrema(t, ex)
    table = _table(J, l1=("coder", 2))
    primary = S.primary_selection(extrema, f.shape, table, _all(extrema))
    sel = S.working_extrep(ex, extrema, f.shape, table, primary, transform=t)
    assert np.array_equal(sel.mask[1], sel.mask[2])
    hor, ver = sel.cartesian(1)
    m = sel.mask[1]
    np.testing.assert_allclose(hor[m], t.Wx_full[1][m], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(ver[m], t.Wy_full[1][m], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("degrees,level", [(0.0, 1), (90.0, 1), (180.0, 1), (270.0, 1),
                                           (30.0, 3), (45.0, 3), (-30.0, 3), (150.0, 3),
                                           (225.0, 3)])
def test_predicting_a_level_from_level_2_on_a_straight_step_gives_its_modulus(degrees, level):
    # Level 1 follows the decay law along the axes only (its staggered gradient; the module
    # docstring), so oblique steps are predicted at level 3, where levels 2-4 of a binary oblique
    # step still differ by a few percent from the continuous law.
    n = 128
    th = np.radians(degrees)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    f = ((xx - 64.6) * np.cos(th) + (yy - 63.3) * np.sin(th) >= 0).astype(float)
    t, ex = lw.analyze(f, 4, border="mirror")
    extrema = _extrema(t, ex)
    for e, a in zip(extrema, S.chain_alpha(extrema, f.shape)):
        e["alpha"] = a
    table = _table(4, **{f"l{level}": ("predict", 2)})
    primary = S.primary_selection(extrema, f.shape, table, _all(extrema))
    shown = S.shown_level(extrema, table, primary, level)

    def central(e):
        w = (e["x"] > 32) & (e["x"] < 96) & (e["y"] > 32) & (e["y"] < 96)
        return np.asarray(e["mod"], float)[w].mean()

    rel = 0.03 if degrees % 90 == 0 else 0.12
    assert central(shown) == pytest.approx(central(extrema[level - 1]), rel=rel)


def test_the_alpha_of_a_step_a_line_and_a_point():
    n, J = 128, 5
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    for f, want in (((xx >= 64).astype(float), 0.0), ((xx == 64).astype(float), -1.0),
                    (((xx == 64) & (yy == 64)).astype(float), -2.0)):
        t, ex = lw.analyze(f, J, border="mirror")
        extrema = _extrema(t, ex)
        for e, a in zip(extrema, S.chain_alpha(extrema, f.shape)):
            w = (e["x"] > 32) & (e["x"] < 96) & (e["y"] > 32) & (e["y"] < 96)
            assert np.nanmedian(a[w]) == pytest.approx(want, abs=0.06)


def test_the_alpha_check_drops_a_maximum_the_decay_does_not_predict():
    src = {"x": np.array([10]), "y": np.array([10]), "mod": np.array([1.0]),
           "arg": np.array([0.0]), "alpha": np.array([0.0]), "line_id": np.array([-1])}
    dst = {"x": np.array([10, 30]), "y": np.array([10, 30]), "mod": np.array([1.05, 4.0]),
           "arg": np.array([0.0, 0.0]), "line_id": np.array([-1, -1])}
    dst_far = dict(dst, x=np.array([10, 11]), y=np.array([10, 10]))
    table = _table(2, l1=("near", 2), l2=("all", 2))
    keep = S.primary_selection([dst_far, src], (40, 40), table, None, alpha_check=True,
                               alpha_tol=0.5)[0]
    assert keep.tolist() == [True, False]
    loose = S.primary_selection([dst_far, src], (40, 40), table, None)[0]
    assert loose.tolist() == [True, True]


def test_a_borrowing_source_and_a_self_borrow_are_refused():
    e = {"x": np.array([1]), "y": np.array([1]), "mod": np.array([1.0]), "arg": np.array([0.0]),
         "line_id": np.array([-1])}
    with pytest.raises(ValueError, match="which borrows itself"):
        S.primary_selection([e, e, e], (4, 4), _table(3, l1=("coder", 2), l2=("predict", 3)),
                            lambda l: np.ones(1, bool))
    with pytest.raises(ValueError, match="from itself"):
        S.primary_selection([e, e], (4, 4), _table(2, l1=("coder", 1)),
                            lambda l: np.ones(1, bool))
