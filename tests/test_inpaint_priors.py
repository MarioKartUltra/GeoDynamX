# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The priors: the clear masks and rho pyramid of a region, the abacus of ln|eta|, the
cycle-spun soft-threshold prox and the cascade inference."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("numba")
pywt = pytest.importorskip("pywt")

from dynamix.core import dwt  # noqa: E402
from dynamix.core.inpaint import priors as pr  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_microcanonical import WEIGHTS, _cascade2d  # noqa: E402


def _region(shape=(128, 128), box=(40, 72, 48, 80)):
    m = np.zeros(shape, dtype=bool)
    r0, r1, c0, c1 = box
    m[r0:r1, c0:c1] = True
    return m


def _haar_pyramid(x, level):
    return pywt.wavedec2(x, "haar", mode="periodization", level=level)


def _haar_inverse(c):
    return pywt.waverec2(c, "haar", mode="periodization")


def _haar_approximations(x, levels):
    out, a = [], np.asarray(x, dtype=np.float64)
    for _ in range(levels):
        a, _d = pywt.dwt2(a, "haar", mode="periodization")
        out.append(a)
    return out


def _haar_clear(mask, levels):
    """Haar: the atom is the block itself (L = 2), so a coefficient is clear when its block is."""
    out = []
    for j in range(1, levels + 1):
        s = 2 ** j
        ny, nx = mask.shape
        hit = mask[:ny // s * s, :nx // s * s].reshape(ny // s, s, nx // s, s).any(axis=(1, 3))
        out.append(np.ascontiguousarray((~hit).astype(np.uint8)))
    return out


def _ks(a, b) -> float:
    """Kolmogorov distance of two samples."""
    a, b = np.sort(np.ravel(a)), np.sort(np.ravel(b))
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.abs(fa - fb).max())


def test_approximations_are_the_running_low_pass():
    x = np.random.default_rng(0).standard_normal((64, 64))
    a = dwt.approximations(x, "sym7", 3)
    assert [v.shape for v in a] == [(32, 32), (16, 16), (8, 8)]
    np.testing.assert_allclose(a[2], dwt.pyramid(x, "sym7", 3)[0])


def test_rho_is_zero_far_from_the_region_near_one_inside_and_clipped():
    m = _region((256, 256), (96, 160, 96, 160))             # a 64 x 64 region
    for name in ("sym7", "bl3"):
        rho = pr.rho_pyramid(m, name, 4)
        assert all(0.0 <= r.min() and r.max() <= 1.0 for r in rho)
        assert rho[0][64, 64] >= 0.9                        # level 1: block (128, 128), the centre
        assert rho[1][32, 32] >= 0.9                        # level 2: block (128, 128)
        assert rho[0].shape == (128, 128) and rho[3].shape == (16, 16)
    rho = pr.rho_pyramid(m, "sym7", 4)
    assert rho[0][10, 10] < 1e-12 and rho[1][5, 5] < 1e-12  # beyond the atom's reach: exactly 0
    unclipped = dwt.approximations(m.astype(np.float64), "sym7", 1)[0] / 2.0
    assert unclipped.min() < -0.05 and unclipped.max() > 1.05   # why the clip exists


def test_clear_masks_exclude_the_atoms_support_and_clear_pixels_matches_the_full_grid():
    m = _region(box=(40, 72, 72, 80))                       # rows 40..71, cols 72..79 of 128^2
    clear = pr.clear_masks(m, "sym7", 3)
    assert clear[0].shape == (64, 64) and clear[0].dtype == np.uint8
    half1 = (dwt.filter_length("sym7") - 1) * 1               # level 1: 13 px each side
    assert clear[0][20, 36] == 0                            # its block (40..41, 72..73) is inside
    assert clear[0][13, 36] == 0                            # rows 26..27 + 13 reach row 40
    assert clear[0][12, 36] == 1 and clear[0][5, 36] == 1   # rows 24..25 + 13 stop at 38
    assert clear[2][0, 0] == 1 and clear[2][5, 9] == 0      # level 3: 52 px each side
    px = pr.clear_pixels(m, "sym7", 1)
    assert px.shape == m.shape and px[40 - half1, 60] == 0 and px[40 - half1 - 1, 60] == 1
    assert px[5, 5] == 1 and px[44, 76] == 0


def test_the_haar_abacus_of_the_exact_cascade_is_the_eight_point_law():
    dens, _ = _cascade2d()
    J = 6
    coeffs = _haar_pyramid(dens, J)
    clear = [np.ones(coeffs[J + 1 - j][0].shape, dtype=np.uint8) for j in range(1, J + 1)]
    ab = pr.Abacus.from_pyramid(coeffs, clear, "haar")
    support = np.log(np.array([w * r / 2.0 for w in WEIGHTS for r in (0.5, 1.0, 2.0)]))
    centres = 0.5 * (ab.edges[:-1] + ab.edges[1:])
    near = np.abs(centres[:, None] - support[None, :]).min(1) < 0.08
    for o in range(3):
        assert ab.hist[o][near].sum() / ab.hist[o].sum() > 0.97, o
        assert ab.n_pairs[o] > 1000
    assert ab.lo.shape == (3,) and (ab.lo < ab.hi).all()
    assert np.allclose(ab.cdf[:, -1], 1.0) and (np.diff(ab.cdf, axis=1) >= 0).all()
    assert (ab.log_hist > -20).all()                        # floored, never -inf
    assert 0.0 <= ab.sign_freq.min() and ab.sign_freq.max() <= 1.0


def test_an_abacus_round_trips_through_a_file(tmp_path):
    dens, _ = _cascade2d()
    coeffs = dwt.pyramid(dens, "sym7", 5)
    clear = [np.ones(coeffs[6 - j][0].shape, dtype=np.uint8) for j in range(1, 6)]
    ab = pr.Abacus.from_pyramid(coeffs, clear, "sym7")
    ab.scenes = ["cascade seed 0"]
    p = tmp_path / "abacus.npz"
    ab.save(p)
    back = pr.Abacus.load(p)
    np.testing.assert_array_equal(back.hist, ab.hist)
    assert back.wavelet == "sym7" and back.scenes == ["cascade seed 0"]
    with pytest.raises(ValueError, match="sym7"):
        pr.Abacus.load(p, wavelet="bl3")


def test_an_abacus_with_no_clear_pair_says_so():
    coeffs = _haar_pyramid(np.random.default_rng(0).standard_normal((32, 32)), 3)
    clear = [np.zeros(coeffs[4 - j][0].shape, dtype=np.uint8) for j in range(1, 4)]
    with pytest.raises(ValueError, match="no clear parent-child pair"):
        pr.Abacus.from_pyramid(coeffs, clear, "haar")


def test_the_sym7_abacus_pairs_by_the_banks_child_shift_and_concentrates():
    """With the child bands rolled by the bank's shift, the sym7 abacus of the exact cascade
    is sharper than the mis-paired one (the (2k, 2k+1) rule taken literally)."""
    dens, _ = _cascade2d()
    J = 5
    coeffs = dwt.pyramid(dens, "sym7", J)
    clear = [np.ones(coeffs[J + 1 - j][0].shape, dtype=np.uint8) for j in range(1, J + 1)]
    # the abacus bins differ between the two pairings (edges follow the pooled quantiles), so the
    # spreads are taken on the raw ratios, which the diagonal shares exactly
    spread = [np.std(r) for r in pr._collect_pairs(coeffs, clear, "sym7")[0]]
    spread0 = [np.std(r) for r in pr._collect_pairs(coeffs, clear, 0)[0]]
    assert spread[0] < spread0[0] and spread[1] < spread0[1]      # the two bands the shift touches
    assert abs(spread[2] - spread0[2]) < 1e-9                     # the diagonal is untouched


def test_make_abacus_pools_the_clear_pairs_of_several_fields():
    a, _ = _cascade2d(seed=0)
    b, _ = _cascade2d(seed=1)
    pooled = pr.make_abacus([a, b], [None, None], "haar", 6)
    ones = [pr.Abacus.from_pyramid(_haar_pyramid(dwt.mirror(f), 6), _haar_clear(np.zeros((256, 256), bool), 6), "haar")
            for f in (a, b)]
    one = ones[0]
    assert pooled.n_pairs.sum() == sum(o.n_pairs.sum() for o in ones)
    assert pooled.scenes == ["0", "1"] and pooled.wavelet == "haar"
    gap = np.zeros(a.shape, bool)
    gap[40:80, 40:80] = True
    fewer = pr.make_abacus([a], [gap], "haar", 6)
    assert fewer.n_pairs.sum() < one.n_pairs.sum()


def test_the_sparsity_prox_lowers_the_detail_l1_and_keeps_the_mean():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((64, 64)).cumsum(0)
    prior = pr.Sparsity("sym7", 3, lam=2.0, rng=np.random.default_rng(2))
    z = prior.prox(x, 0.5)
    assert dwt.detail_l1(dwt.pyramid(z, "sym7", 3)) < dwt.detail_l1(dwt.pyramid(x, "sym7", 3))
    assert abs(z.mean() - x.mean()) < 1e-9
    assert prior.penalty(z) < prior.penalty(x)


def test_auto_lambda_is_the_notebooks_rule():
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 0.5, (128, 128))
    clear1 = np.ones((64, 64), dtype=np.uint8)
    lam = pr.auto_lambda(y, clear1, "sym7", 0.2)
    details = dwt.pyramid(y, "sym7", 1)[1]
    sigma = np.median(np.abs(np.concatenate([d.ravel() for d in details]))) / 0.6745
    assert lam == pytest.approx(0.2 * sigma * np.sqrt(2 * np.log(128 * 128)))
    assert 0.4 < sigma < 0.6                                # white noise: the details keep sigma
