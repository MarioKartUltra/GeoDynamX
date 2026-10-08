# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""core/dwt.py: PyWavelets behind one door, periodization only, with the Battle-Lemarie bank."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pywt")

from dynamix.core import dwt  # noqa: E402


def _img(ny=64, nx=96, seed=0):
    return np.random.default_rng(seed).standard_normal((ny, nx)).cumsum(0).cumsum(1)


@pytest.mark.parametrize("name", ["sym7", "db4", "bl3"])
def test_every_bank_is_orthogonal_and_reconstructs_under_periodization(name):
    w = dwt.wavelet(name)
    h = np.asarray(w.rec_lo)
    assert abs(h.sum() - np.sqrt(2)) < 1e-6
    for s in range(1, len(h) // 2):
        assert abs(np.dot(h, np.roll(h, 2 * s))) < 1e-6, s
    x = _img()
    c = dwt.pyramid(x, name, 3)
    np.testing.assert_allclose(dwt.inverse(c, name), x, atol=1e-5)
    assert dwt.dc_gain(name) == pytest.approx(np.sqrt(2), abs=1e-6)


def test_battle_lemarie_is_the_cubic_spline_construction():
    dec_lo, dec_hi, rec_lo, rec_hi = dwt.battle_lemarie_filters()
    h = np.asarray(rec_lo)
    assert 60 <= len(h) <= 140 and len(h) % 2 == 0
    p = int(np.argmax(h))                                   # whole-sample symmetric about its peak
    np.testing.assert_allclose(h[p - 5:p], h[p + 1:p + 6][::-1], atol=1e-12)
    np.testing.assert_allclose(h[p:p + 4], [0.766130, 0.433923, -0.050202, -0.110037], atol=1e-5)
    assert dwt.filter_length("bl3") == len(h)               # Mallat's table, A Wavelet Tour 7.2.3
    assert dwt.wavelet("bl3") is dwt.wavelet("bl3")         # built once


def test_an_unknown_bank_is_refused_by_name_and_haar_is_the_test_bank():
    with pytest.raises(ValueError, match="bior"):
        dwt.wavelet("bior2.2")
    assert dwt.wavelet("haar").dec_len == 2 and "haar" not in dwt.BANKS
    assert dwt.child_shift("haar") == 0


def test_pyramid_and_inverse_are_adjoint():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((64, 64))
    c = dwt.pyramid(x, "sym7", 3)
    r = [rng.standard_normal(c[0].shape)] + [tuple(rng.standard_normal(d.shape) for d in lev)
                                             for lev in c[1:]]
    lhs = np.dot(dwt._flat(c), dwt._flat(r))
    rhs = np.dot(x.ravel(), dwt.inverse(r, "sym7").ravel())
    assert abs(lhs - rhs) < 1e-9 * abs(lhs)


def test_stationary_is_a_tight_frame_with_its_own_adjoint():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((64, 64))
    s = dwt.stationary(x, "sym7", 3)
    assert len(s) == 4 and s[0].shape == x.shape and len(s[1]) == 3
    energy = (s[0] ** 2).sum() + sum((d ** 2).sum() for lev in s[1:] for d in lev)
    assert energy == pytest.approx((x ** 2).sum(), rel=1e-10)
    r = [rng.standard_normal(x.shape)] + [tuple(rng.standard_normal(x.shape) for _ in lev)
                                           for lev in s[1:]]
    lhs = np.vdot(dwt._flat(s), dwt._flat(r))
    rhs = np.vdot(x, dwt.stationary_inverse(r, "sym7"))
    assert abs(lhs - rhs) < 1e-9 * abs(lhs)


def test_the_working_shape_doubles_and_rounds_up_to_the_level():
    assert dwt.working_shape((63, 65), 3) == (128, 136)
    assert dwt.working_shape((64, 64), 3) == (128, 128)
    assert dwt.working_shape((5, 7), 4) == (16, 16)


def test_mirror_is_continuous_and_periodic_and_quarter_returns_the_original():
    x = _img(63, 65)
    m = dwt.mirror(x, dwt.working_shape(x.shape, 3))
    assert m.shape == (128, 136)
    np.testing.assert_array_equal(dwt.quarter(m, 63, 65), x)
    # even symmetry: the row after the last padded row mirrors it, and the torus closes
    assert np.array_equal(m[64], m[63]) and np.array_equal(m[0], m[-1])
    assert np.array_equal(m[:, 68], m[:, 67]) and np.array_equal(m[:, 0], m[:, -1])
    plain = dwt.mirror(x)
    assert plain.shape == (126, 130)


def test_full_depth_and_max_level():
    assert dwt.full_depth((128, 256)) == 6
    assert dwt.full_depth((2, 2)) == 1
    assert dwt.max_level((128, 128), "sym7") == 3
    assert dwt.max_level((256, 256), "bl3") == 1


def test_the_full_depth_pyramid_runs_past_max_level_without_a_warning(recwarn):
    x = _img(128, 128)
    c = dwt.pyramid(x, "bl3", 6)
    assert len(c) == 7 and c[0].shape == (2, 2)
    assert not [w for w in recwarn if "level" in str(w.message).lower()]
    np.testing.assert_allclose(dwt.inverse(c, "bl3"), x, atol=1e-4)


def test_a_constant_image_gives_two_to_the_level_in_the_approximation():
    one = np.ones((64, 64))
    for name in ("sym7", "bl3"):
        c = dwt.pyramid(one, name, 3)
        assert np.allclose(c[0], 2.0 ** 3, atol=1e-9)


@pytest.mark.parametrize("name,shift", [("sym7", -1), ("db4", -2), ("bl3", 0)])
def test_a_parent_atom_sits_over_its_children_after_the_banks_child_shift(name, shift):
    """Under periodization the diagonal band's parent k is centred over its children
    (2k, 2k+1) on both axes, and the horizontal and vertical bands are centred along their
    band-pass axis; along their low-pass axis the parent sits a bank-specific number of child
    cells off (the measured child shift the kernels roll by: sym7 -1.2, db4 -2.0, bl3 -0.5)."""
    assert dwt.child_shift(name) == shift
    n = 512
    zero = dwt.pyramid(np.zeros((n, n)), name, 4)

    def centre(level, band, k):
        c = [np.zeros_like(zero[0])] + [tuple(np.zeros_like(d) for d in lev) for lev in zero[1:]]
        c[-level][band][k, k] = 1.0
        atom = np.abs(dwt.inverse(c, name))
        w = atom / atom.sum()
        ii, jj = np.indices(atom.shape)
        return np.array([(w * ii).sum(), (w * jj).sum()])

    for j in (1, 2, 3):
        for band, lowpass_axis in ((0, 1), (1, 0), (2, None)):   # cH: columns; cV: rows; cD: none
            off = (centre(j + 1, band, 10) - 0.5 * (centre(j, band, 20) + centre(j, band, 21))) / 2.0 ** j
            for axis in (0, 1):
                if axis == lowpass_axis:
                    assert abs(off[axis] - shift) <= 0.55, (name, j, band, axis, off)
                else:
                    assert abs(off[axis]) <= 0.25, (name, j, band, axis, off)


def test_cycle_shift_is_bounded_and_undone_exactly():
    rng = np.random.default_rng(3)
    x = _img()
    for level, bound in ((2, 4), (6, 32), (9, 32)):
        z, shift = dwt.cycle_shift(x, rng, level)
        assert max(abs(shift[0]), abs(shift[1])) <= bound
        np.testing.assert_array_equal(dwt.cycle_unshift(z, shift), x)


def test_soft_threshold_shrinks_details_only_and_detail_l1_counts_them():
    c = dwt.pyramid(_img(), "sym7", 2)
    t = dwt.soft_threshold(c, 0.5)
    np.testing.assert_array_equal(t[0], c[0])
    for lev, tl in zip(c[1:], t[1:]):
        for d, td in zip(lev, tl):
            np.testing.assert_allclose(td, np.sign(d) * np.maximum(np.abs(d) - 0.5, 0.0))
    assert dwt.detail_l1(t) < dwt.detail_l1(c)


def test_only_band_keeps_one_band_and_the_bands_sum_to_the_image():
    x = _img(64, 64)
    c = dwt.pyramid(x, "sym7", 2)
    parts = [dwt.inverse(dwt.only_band(c, 2, "approximation"), "sym7")]   # level J: the stored approximation alone
    for level in (1, 2):
        for band in ("horizontal", "vertical", "diagonal"):
            parts.append(dwt.inverse(dwt.only_band(c, level, band), "sym7"))
    np.testing.assert_allclose(sum(parts), x, atol=1e-8)
    approx2 = dwt.inverse(dwt.only_band(c, 2, "approximation"), "sym7")
    assert approx2.shape == x.shape and abs(approx2.mean() - x.mean()) < 1e-8


def test_missing_pywt_names_it(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pywt", None)
    with pytest.raises(ImportError, match="PyWavelets"):
        dwt.require_pywt()


def test_importing_the_module_leaves_pywt_unimported():
    import pathlib
    import subprocess
    import sys

    code = "import sys, dynamix.core.dwt; sys.exit(1 if 'pywt' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code],
                          cwd=str(pathlib.Path(__file__).resolve().parent.parent)).returncode == 0
