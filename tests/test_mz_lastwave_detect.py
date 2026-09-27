# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The LastWave ``extrema2`` port against the authors' C (golden files and a non-square case)."""
import pathlib
import sys

import numpy as np
import pytest

from dynamix.core import mz_lastwave as lw
from dynamix.core.mz_lastwave import detect

GOLDEN = pathlib.Path(__file__).resolve().parent / "golden" / "lastwave"
GOLD = sorted(str(p) for p in GOLDEN.glob("*_lw_clip.npz"))     # the extrema live in these


def _levels(g, J):
    Wx = [None] + [g[f"Wx_{l}"] for l in range(1, J + 1)]
    Wy = [None] + [g[f"Wy_{l}"] for l in range(1, J + 1)]
    return Wx, Wy


def test_the_golden_files_are_found():
    assert len(GOLD) == 4


@pytest.mark.parametrize("path", GOLD)
def test_extrema2_is_bit_exact_with_the_authors_c(path):
    g = np.load(path); J = int(g["J"])
    Wx, Wy = _levels(g, J)
    ex = detect.extrema2(Wx, Wy, J)
    for l in range(1, J + 1):
        assert np.array_equal(ex.mask[l], g[f"extmask_{l}"]), (path, l)
        assert np.array_equal(np.where(ex.mask[l], ex.mag[l], 0), g[f"extmagn_{l}"]), (path, l)
        dm, _ = ex.denormalised(l)
        assert np.array_equal(np.where(ex.mask[l], dm, 0), g[f"extmag_{l}"]), (path, l)
        assert np.array_equal(np.where(ex.mask[l], ex.arg[l], 0), g[f"extarg_{l}"]), (path, l)


def test_colocation_uses_the_two_tap_average_and_keeps_raw_anchors():
    g = np.load(GOLDEN / "disc64_lw_clip.npz"); J = int(g["J"])
    Wx, Wy = _levels(g, J)
    off, on = detect.extrema2(Wx, Wy, J), detect.extrema2(Wx, Wy, J, colocate_l1=True)
    for l in range(2, J + 1):                                  # only level 1 changes
        assert np.array_equal(off.mask[l], on.mask[l])
    wx1c = 0.5 * (np.roll(Wx[1], 1, 0) + Wx[1]); wy1c = 0.5 * (np.roll(Wy[1], 1, 1) + Wy[1])
    ref = detect.extrema2([None, wx1c] + Wx[2:], [None, wy1c] + Wy[2:], J)
    assert np.array_equal(on.mask[1], ref.mask[1])
    assert not np.array_equal(on.mask[1], off.mask[1])
    h, v = on.cartesian(1)
    raw = np.hypot(Wx[1], Wy[1])[on.mask[1]]
    assert np.allclose(np.hypot(h, v)[on.mask[1]], raw)       # anchors are the raw values
    assert np.allclose(h[on.mask[1]], Wx[1][on.mask[1]])
    assert np.allclose(v[on.mask[1]], Wy[1][on.mask[1]])


def _mirror2n(x):
    """The explicit 2N mirrored field ``dwt2d(border="mirror")`` runs the algorithm on."""
    return np.concatenate([np.concatenate([x, x[::-1]], 0),
                           np.concatenate([x, x[::-1]], 0)[:, ::-1]], 1)


def test_mirror_colocation_is_the_2n_fields_colocation():
    x = np.random.default_rng(1).standard_normal((48, 40)); ny, nx = x.shape; J = 3
    big = lw.dwt2d(_mirror2n(x), J, border="periodic")
    small = lw.dwt2d(x, J, border="mirror"); Wx, Wy = small.Wx, small.Wy
    wx1c = 0.5 * (np.concatenate([Wx[1][:1], Wx[1][:-1]], 0) + Wx[1])
    wy1c = 0.5 * (np.concatenate([Wy[1][:, :1], Wy[1][:, :-1]], 1) + Wy[1])
    # the first row (column) as its own predecessor is the 2N field's wrap, cropped, bit for bit
    assert np.array_equal(wx1c, (0.5 * (np.roll(big.Wx[1], 1, 0) + big.Wx[1]))[:ny, :nx])
    assert np.array_equal(wy1c, (0.5 * (np.roll(big.Wy[1], 1, 1) + big.Wy[1]))[:ny, :nx])
    on = detect.extrema2(Wx, Wy, J, colocate_l1=True, border="mirror")
    ref = detect.extrema2([None, wx1c] + Wx[2:], [None, wy1c] + Wy[2:], J)
    assert np.array_equal(on.mask[1], ref.mask[1])
    wrapped = detect.extrema2(Wx, Wy, J, colocate_l1=True, border="periodic")
    assert not np.array_equal(on.mask[1], wrapped.mask[1])
    # Away from the crop's own edge, whose skipped comparisons the level-1 gap-filling pass reads,
    # the extrema are those of the 2N field.
    full = detect.extrema2(big.Wx, big.Wy, J, colocate_l1=True).mask[1][:ny, :nx]
    assert np.array_equal(on.mask[1][2:-2, 2:-2], full[2:-2, 2:-2])


def test_an_unknown_border_is_refused():
    with pytest.raises(ValueError, match="border"):
        detect.extrema2([None, np.zeros((8, 8))], [None, np.zeros((8, 8))], 1, border="edge")


def test_cartesian_is_zero_off_the_extrema_and_the_c_formula_on_them():
    g = np.load(GOLD[0]); J = int(g["J"])
    ex = detect.extrema2(*_levels(g, J), J)
    for l in range(1, J + 1):
        h, v = ex.cartesian(l)
        assert not h[~ex.mask[l]].any() and not v[~ex.mask[l]].any()
        mag, arg = g[f"extmag_{l}"], g[f"extarg_{l}"]
        assert np.allclose(h, mag * np.cos(arg), rtol=0, atol=1e-15)
        assert np.allclose(v, mag * np.sin(arg), rtol=0, atol=1e-15)


# The authors' W2_compute_point (compiled verbatim) on a 24 x 37 field, levels 1 and 2: the masks
# as np.packbits hex. Level 1 adds the diagonal gap-filling pass, so the two differ.
_NONSQUARE_C = {
    1: "0aa08801061703dfca09060492504b0fa992226011724cdd0fbace512482545085100f0f2826de80048382"
       "958213d27491d2019496086346480e00938e194da532a4b8980b44c7504f88f8247c4738448c00050890ce"
       "cc418ce0119773140a432406c393c9a0001289498000110202",
    2: "0aa08801061703d74a09060092504b0ba992226011524cd50faace512482545084100f0f2806da80048382"
       "958213d27491d2019496086342480e00938c194da512a4b8880b44c7504f08f8247c4738448c000508908a"
       "cc418ce0109773140a0324068393c9a0001289498000110202",
}


def _nonsquare_field():
    f = np.random.default_rng(7).standard_normal((24, 37))
    for ax in (0, 1, 0, 1):
        f = 0.25 * (np.roll(f, 1, ax) + 2 * f + np.roll(f, -1, ax))
    return 0.5 * (f - np.roll(f, 1, 1)), 0.5 * (f - np.roll(f, 1, 0))


def test_non_square_matches_the_c_per_axis():
    hor, ver = _nonsquare_field()
    ex = detect.extrema2([None, hor, hor], [None, ver, ver], 2)
    for l in (1, 2):
        want = np.unpackbits(np.frombuffer(bytes.fromhex(_NONSQUARE_C[l]), np.uint8))
        assert ex.mask[l].shape == (24, 37)
        assert np.array_equal(ex.mask[l], want[: 24 * 37].reshape(24, 37).astype(bool)), l
    assert ex.mask[1].sum() == 318 and ex.mask[2].sum() == 298


def test_non_contiguous_input_gives_the_same_extrema():
    g = np.load(GOLD[0]); J = int(g["J"])
    Wx, Wy = _levels(g, J)
    big = [None] + [np.pad(w, ((0, 3), (0, 5))) for w in Wx[1:]]
    bigy = [None] + [np.pad(w, ((0, 3), (0, 5))) for w in Wy[1:]]
    ny, nx = Wx[1].shape
    views = detect.extrema2([None] + [w[:ny, :nx] for w in big[1:]],
                            [None] + [w[:ny, :nx] for w in bigy[1:]], J)
    ref = detect.extrema2(Wx, Wy, J)
    for l in range(1, J + 1):
        assert np.array_equal(views.mask[l], ref.mask[l])
        assert np.array_equal(views.mag[l], ref.mag[l])


def test_missing_numba_names_it(monkeypatch):
    monkeypatch.setattr(detect, "_KERNELS", {})
    monkeypatch.setattr(lw._kernels, "_KERNELS", None)
    monkeypatch.setitem(sys.modules, "numba", None)
    with pytest.raises(RuntimeError, match="numba"):
        detect.extrema2([None, np.zeros((8, 8))], [None, np.zeros((8, 8))], 1)
