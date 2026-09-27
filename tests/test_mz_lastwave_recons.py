# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""``e2recons``: reconstruction from the multiscale edges by alternating projections, bit for bit
with the authors' C (committed goldens), plus continuation, stopping, the coarse modes and a
non-square mirrored field."""
import pathlib

import numpy as np
import pytest

from dynamix.core import mz_lastwave as lw
from dynamix.core.mz_lastwave import detect, recons
from test_mz_lastwave_transform import _equal_through_libm   # the goldens' atan2 / cos / sin rule

GOLDEN = pathlib.Path(__file__).resolve().parent / "golden" / "lastwave"
GOLD = sorted(str(p) for p in GOLDEN.glob("*.npz"))


def _setup(g):
    J = int(g["J"])
    t = lw.dwt2d(g["input"], J, border="periodic")
    ex = detect.extrema2(t.Wx, t.Wy, J)
    return J, t, ex


def test_the_golden_files_are_found():
    assert len(GOLD) >= 6, GOLD


@pytest.mark.parametrize("path", GOLD)
def test_e2recons_is_bit_exact_at_init_and_after_1_5_20(path):
    g = np.load(path); J, t, ex = _setup(g)
    kappa = recons.KAPPA_LASTWAVE if str(g["kappa"]) == "lw" else 1.0
    clip = bool(int(g["clip"]))
    img, d, st = recons.e2recons(g["input"], ex, t.S[J], J, k=0, kappa=kappa, clip=clip,
                                 border="periodic")
    assert _equal_through_libm(img, g["it_0"])
    done = 0
    for k in (1, 5, 20):
        img, d, st = recons.e2recons(g["input"], ex, t.S[J], J, k=k - done, kappa=kappa,
                                     clip=clip, border="periodic", state=st)
        done = k
        assert _equal_through_libm(img, g[f"it_{k}"]), (path, k)
    assert d["iterations"] == 20 and d["stop"] == "fixed"


def test_continuing_equals_one_longer_run():
    g = np.load(GOLD[0]); J, t, ex = _setup(g)
    a, _, s = recons.e2recons(g["input"], ex, t.S[J], J, k=3, border="periodic")
    a, _, _ = recons.e2recons(g["input"], ex, t.S[J], J, k=4, border="periodic", state=s)
    b, _, _ = recons.e2recons(g["input"], ex, t.S[J], J, k=7, border="periodic")
    assert np.array_equal(a, b)


def test_continuing_leaves_the_given_state_unchanged():
    g = np.load(GOLD[0]); J, t, ex = _setup(g)
    first, d1, s = recons.e2recons(g["input"], ex, t.S[J], J, k=1, border="periodic")
    recons.e2recons(g["input"], ex, t.S[J], J, k=5, border="periodic", state=s)
    again, d2, _ = recons.e2recons(g["input"], ex, t.S[J], J, k=0, border="periodic", state=s)
    assert np.array_equal(again, first) and d2["iterations"] == d1["iterations"] == 1
    assert d2["resid"] == d1["resid"]


def test_converge_mode_stops_and_reports_why():
    g = np.load(GOLDEN / "disc64_k1_noclip.npz"); J, t, ex = _setup(g)
    img, d, s = recons.e2recons(g["input"], ex, t.S[J], J, k=500, mode="converge", tol=1e-3,
                                border="periodic")
    assert d["stop"] == "residual rising" and d["iterations"] < 500
    # the minimum-residual iterate is the one returned, the same image a fixed run of that length
    # gives
    best = d["best_iteration"]
    assert d["resid"][best] == min(d["resid"]) and best < d["iterations"]
    fixed, _, _ = recons.e2recons(g["input"], ex, t.S[J], J, k=best, border="periodic")
    assert np.array_equal(img, fixed)
    again, d2, _ = recons.e2recons(g["input"], ex, t.S[J], J, k=500, mode="converge", tol=1e-3,
                                   border="periodic", state=s)
    # a stopped state returns at once with the same image and count
    assert np.array_equal(again, img) and d2["iterations"] == d["iterations"]
    assert d2["stop"] == d["stop"] and d2["snr_db"] == d["snr_db"]


def test_a_converged_state_returns_at_once():
    g = np.load(GOLDEN / "disc64_lw_clip.npz"); J, t, ex = _setup(g)
    kw = dict(kappa=recons.KAPPA_LASTWAVE, clip=True, border="periodic", mode="converge",
              tol=1e-3)
    img, d, s = recons.e2recons(g["input"], ex, t.S[J], J, k=500, **kw)
    assert d["stop"] == "converged" and d["iterations"] < 500
    again, d2, s2 = recons.e2recons(g["input"], ex, t.S[J], J, k=500, state=s, **kw)
    assert d2["stop"] == "converged" and d2["iterations"] == d["iterations"]
    assert np.array_equal(again, img) and np.array_equal(s2.image, s.image)


def test_converge_mode_stops_at_the_cap():
    g = np.load(GOLD[0]); J, t, ex = _setup(g)
    img, d, s = recons.e2recons(g["input"], ex, t.S[J], J, k=2, mode="converge", tol=1e-12,
                                border="periodic")
    assert d["stop"] == "cap" and d["iterations"] == 2 and len(d["resid"]) == 3
    fixed, _, _ = recons.e2recons(g["input"], ex, t.S[J], J, k=2, border="periodic")
    assert np.array_equal(img, fixed)


def test_non_square_reconstructs():
    # a non-square, odd-sized field through analysis and reconstruction on the mirrored field
    x = np.random.default_rng(3).standard_normal((129, 100)).cumsum(0).cumsum(1)
    J = 4; t, ex = lw.analyze(x, J)                  # mirror mode: the 2N working field
    img, d, _ = recons.e2recons(x, ex, t.S_full[J], J, k=10)
    assert img.shape == x.shape and np.isfinite(d["snr_db"]) and d["snr_db"] > 10


def test_primary_extrema_crop_the_working_field():
    x = np.random.default_rng(4).standard_normal((40, 36))
    t, ex = lw.analyze(x, 3)
    assert ex.mask[1].shape == (80, 72)
    mask, mag, arg = lw.primary_extrema(t, ex, 2)
    assert mask.shape == mag.shape == arg.shape == (40, 36)
    assert np.array_equal(mask, ex.mask[2][:40, :36]) and np.array_equal(mag, ex.mag[2][:40, :36])


def test_a_cropped_coarse_in_mirror_mode_is_refused():
    x = np.random.default_rng(5).standard_normal((32, 32))
    t, ex = lw.analyze(x, 3)
    with pytest.raises(ValueError, match="working field"):
        recons.e2recons(x, ex, t.S[3], 3, k=1)


@pytest.mark.parametrize("border", ["mirror", "periodic"])
def test_coarse_modes(border):
    x = np.random.default_rng(6).standard_normal((64, 64)).cumsum(0).cumsum(1)
    J = 3; t, ex = lw.analyze(x, J, border=border)
    full, df, _ = recons.e2recons(x, ex, t.S_full[J], J, k=5, border=border)
    thumb, dt, _ = recons.e2recons(x, ex, t.S_full[J], J, k=5, coarse="thumbnail", border=border)
    none, dn, _ = recons.e2recons(x, ex, t.S_full[J], J, k=5, coarse="none", border=border)
    assert df["coarse"] == "full" and dt["coarse"] == "thumbnail" and dn["coarse"] == "none"
    assert not np.array_equal(full, thumb) and np.isfinite(dt["snr_db"]) and dt["snr_db"] > 5
    assert dt["snr_db"] > df["snr_db"] - 3                    # the decode loses little
    assert np.isfinite(dn["snr_db"]) and dn["snr_db"] > 0     # scored against the field's detail
    if border == "periodic":                                  # the details alone carry no mean
        assert abs(none.mean()) < 1e-9 * np.abs(x).max()
    # the pinned decode is the working field's own coarse at every thumbnail sample, the fold
    # row and column of the mirrored field included
    S, step = t.S_full[J], 2 ** J
    pin = recons._pinned_coarse(S, "thumbnail", J, border, x.shape)
    assert np.max(np.abs(pin[::step, ::step] - S[::step, ::step])) < 1e-5 * np.abs(S).max()
    y = x[:60, :60]; ty, exy = lw.analyze(y, J, border=border)
    with pytest.raises(ValueError, match="divisible"):
        recons.e2recons(y, exy, ty.S_full[J], J, k=1, coarse="thumbnail", border=border)
    # an odd thumbnail per axis (48 / 2**4 = 3, 80 / 2**4 = 5): refused on the periodic field,
    # even on the mirrored one
    z = np.random.default_rng(7).standard_normal((48, 80)).cumsum(0).cumsum(1)
    tz, exz = lw.analyze(z, 4, border=border)
    if border == "periodic":
        with pytest.raises(ValueError, match="even number"):
            recons.e2recons(z, exz, tz.S_full[4], 4, k=1, coarse="thumbnail", border=border)
    else:
        img, d, _ = recons.e2recons(z, exz, tz.S_full[4], 4, k=1, coarse="thumbnail",
                                    border=border)
        assert img.shape == z.shape and np.isfinite(d["snr_db"])


def test_kappa_selects_the_decay():
    g = np.load(GOLDEN / "disc64_lw_clip.npz"); J, t, ex = _setup(g)
    one, _, _ = recons.e2recons(g["input"], ex, t.S[J], J, k=2, border="periodic")
    lwk, d, _ = recons.e2recons(g["input"], ex, t.S[J], J, k=2, border="periodic",
                                kappa=recons.KAPPA_LASTWAVE)
    assert not np.array_equal(one, lwk) and d["kappa"] == recons.KAPPA_LASTWAVE


def test_cancel_and_progress():
    from dynamix.core.wtmm_backend import ComputeCancelled
    g = np.load(GOLD[0]); J, t, ex = _setup(g)
    with pytest.raises(ComputeCancelled):
        recons.e2recons(g["input"], ex, t.S[J], J, k=5, cancel=lambda: True, border="periodic")
    seen = []
    recons.e2recons(g["input"], ex, t.S[J], J, k=3, progress=lambda s, f: seen.append(f),
                    border="periodic")
    assert seen and seen[-1] == pytest.approx(1.0)
