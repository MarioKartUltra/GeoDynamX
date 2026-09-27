# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The LastWave ``dwt2d`` port: bit for bit with the authors' C on the committed golden files
(square, periodic), identity reconstruction on non-square grids and at J = 1 (the two C defects
fixed), the mirrored border equal to the algorithm on the explicit 2N field, registration by an
impulse, and the refusals."""
import pathlib
import platform
import subprocess
import sys

import numpy as np
import pytest

from dynamix.core import mz_lastwave as lw

GOLD_DIR = pathlib.Path(__file__).resolve().parent / "golden" / "lastwave"
GOLD = sorted(str(p) for p in GOLD_DIR.glob("*_lw_clip.npz"))     # the transform lives in these

#: The goldens were written by the C on Apple Silicon macOS, whose maths library the port's atan2,
#: cos and sin also call there.
LIBM_OF_THE_GOLDENS = sys.platform == "darwin" and platform.machine() == "arm64"


def _equal_through_libm(a, ref):
    """Equality for an array computed through atan2, cos or sin: bit for bit on the goldens'
    platform; elsewhere within 1e-12 of the array's peak, room for a C runtime that rounds those
    functions' last bit differently (a few ulp in every angle moves these arrays by under 1e-15
    of the peak). Arrays of + − × ÷ and sqrt alone are compared bit for bit everywhere."""
    if LIBM_OF_THE_GOLDENS:
        return bool(np.array_equal(a, ref))
    return bool(np.max(np.abs(a - ref)) <= 1e-12 * np.max(np.abs(ref)))


def test_the_golden_files_are_found():
    assert len(GOLD) >= 4, GOLD


@pytest.mark.parametrize("path", GOLD)
def test_dwt2d_is_bit_exact_with_the_authors_c(path):
    g = np.load(path); J = int(g["J"])
    t = lw.dwt2d(g["input"], J, border="periodic")
    for l in range(1, J + 1):
        assert np.array_equal(t.Wx[l], g[f"Wx_{l}"]), (path, l, "Wx")
        assert np.array_equal(t.Wy[l], g[f"Wy_{l}"]), (path, l, "Wy")
        assert np.array_equal(t.S[l], g[f"S_{l}"]), (path, l, "S")
        M, A = lw.polar(t, l)
        assert np.array_equal(M, g[f"M_{l}"]), (path, l, "M")
        assert _equal_through_libm(A, g[f"A_{l}"]), (path, l, "A")
    assert np.array_equal(t.S[J], g[f"S_{J}"])


@pytest.mark.parametrize("path", GOLD)
def test_idwt2d_matches_the_c_identity(path):
    g = np.load(path); t = lw.dwt2d(g["input"], int(g["J"]), border="periodic")
    assert np.array_equal(lw.idwt2d(t), g["rec"])


@pytest.mark.parametrize("path", GOLD)
def test_idwt2d_leaves_the_transform_intact(path):
    g = np.load(path); J = int(g["J"])
    t = lw.dwt2d(g["input"], J, border="periodic")
    lw.idwt2d(t)
    for l in range(1, J + 1):
        assert np.array_equal(t.Wx[l], g[f"Wx_{l}"]) and np.array_equal(t.S[l], g[f"S_{l}"])
    assert np.array_equal(lw.idwt2d(t), g["rec"])


@pytest.mark.parametrize("path", GOLD)
def test_the_projection_kernels_reproduce_the_c_loop(path):
    """LastWave's own settings (decay 1/5.8^(2/scale), clipping on), driven through the in-place
    workspace: the initial pass and one iteration equal the C's it_0 and it_1."""
    from dynamix.core.mz_lastwave.transform import _decompose, _recompose
    g = np.load(path); J = int(g["J"]); x = g["input"]; k = lw.kernels()
    t = lw.dwt2d(x, J, border="periodic")
    SJ = t.S_full[J].copy()
    E = [None] + [np.ascontiguousarray(g[f"extmask_{l}"], dtype=bool) for l in range(1, J + 1)]
    M = [None] + [g[f"extmag_{l}"] for l in range(1, J + 1)]
    H, V = [None], [None]
    for l in range(1, J + 1):
        h, v = np.empty(x.shape), np.empty(x.shape)
        k.cartesian(M[l], g[f"extarg_{l}"], h, v)
        H.append(h); V.append(v)
    m, a = np.empty(x.shape), np.empty(x.shape)

    def project():
        for l in range(1, J + 1):
            decay = 1.0 / 5.8 ** (2.0 / 2 ** l)
            h, v = t.Wx_full[l], t.Wy_full[l]
            k.proj1_rows(h, E[l], H[l], decay); k.proj1_cols(v, E[l], V[l], decay)
            k.polar(h, v, m, a); k.clip_rows(m, a, E[l], M[l]); k.clip_cols(m, a, E[l], M[l])
            k.cartesian(m, a, h, v)

    for l in range(1, J + 1):
        t.Wx_full[l][:] = 0.0; t.Wy_full[l][:] = 0.0
    img = np.empty(x.shape)
    project(); _recompose(t, SJ, img)
    assert _equal_through_libm(img, g["it_0"]), path
    _decompose(t, img); project(); _recompose(t, SJ, img)
    assert _equal_through_libm(img, g["it_1"]), path


@pytest.mark.parametrize("shape,J", [((129, 100), 4), ((64, 48), 3), ((65, 65), 1), ((64, 64), 1)])
def test_identity_reconstruction_on_non_square_and_j1(shape, J):
    x = np.random.default_rng(0).standard_normal(shape)
    for border in ("periodic", "mirror"):
        t = lw.dwt2d(x, J, border=border)
        assert t.shape == shape and all(a.shape == shape for a in t.Wx[1:] + t.Wy[1:] + t.S[1:])
        r = lw.idwt2d(t)
        assert r.shape == shape
        assert 10 * np.log10(np.sum(x**2) / np.sum((x - r) ** 2)) >= 300, (shape, J, border)


def test_mirror_equals_the_algorithm_on_the_explicit_2n_field():
    x = np.random.default_rng(1).standard_normal((48, 40))
    m = np.concatenate([np.concatenate([x, x[::-1]], 0),
                        np.concatenate([x, x[::-1]], 0)[:, ::-1]], 1)
    big, small = lw.dwt2d(m, 3, border="periodic"), lw.dwt2d(x, 3, border="mirror")
    for l in range(1, 4):
        assert np.array_equal(small.Wx[l], big.Wx[l][:48, :40])
        assert np.array_equal(small.Wy[l], big.Wy[l][:48, :40])
        assert np.array_equal(small.S[l], big.S[l][:48, :40])


def test_registration_by_impulse():
    x = np.zeros((64, 64)); x[32, 20] = 1.0
    t = lw.dwt2d(x, 4, border="periodic")
    # index j registers position j - 1/2, so the impulse at position p is centred at index p + 1/2
    # (levels >= 2 on both axes)
    for l in (2, 3, 4):
        assert lw._symmetry_centre(t.Wx[l][32, :]) == (20.5, "anti")
        assert lw._symmetry_centre(t.Wy[l][:, 20]) == (32.5, "anti")
    assert lw._symmetry_centre(t.Wx[1][32, :]) == (20.5, "anti")    # level 1: Wx at (j-1/2, i)
    assert lw._symmetry_centre(t.Wx[1][:, 20]) == (32.0, "sym")


def test_registration_by_impulse_places_each_channel_half_a_sample_after_its_position():
    # A sample at index j registers position j − ½, so an impulse at (row 32, column 20) is
    # centred at index 20 + ½ along an axis that channel is registered at − ½ on: both axes for
    # S_l and levels >= 2, the differentiated axis only for level 1.
    x = np.zeros((64, 64)); x[32, 20] = 1.0
    t = lw.dwt2d(x, 4, border="periodic")
    c = lw._symmetry_centre
    for l in (2, 3, 4):
        assert c(t.Wx[l][32, :]) == (20.5, "anti") and c(t.Wx[l][:, 20]) == (32.5, "sym"), l
        assert c(t.Wy[l][:, 20]) == (32.5, "anti") and c(t.Wy[l][32, :]) == (20.5, "sym"), l
    for l in (1, 2, 3):                                 # S_4's support wraps the 64-px torus
        assert c(t.S[l][32, :]) == (20.5, "sym") and c(t.S[l][:, 20]) == (32.5, "sym"), l
    assert c(t.Wx[1][32, :]) == (20.5, "anti") and c(t.Wx[1][:, 20]) == (32.0, "sym")
    assert c(t.Wy[1][:, 20]) == (32.5, "anti") and c(t.Wy[1][32, :]) == (20.0, "sym")


def test_refuses_a_grid_too_small_for_j():
    with pytest.raises(ValueError, match="2\\*\\*J"):
        lw.dwt2d(np.zeros((12, 40)), 4)


def test_refuses_j_below_one():
    with pytest.raises(ValueError, match="J"):
        lw.dwt2d(np.zeros((16, 16)), 0)


def test_missing_numba_names_it(monkeypatch):
    monkeypatch.setattr(lw._kernels, "_KERNELS", None)
    monkeypatch.setitem(sys.modules, "numba", None)
    with pytest.raises(RuntimeError, match="numba"):
        lw.kernels()


def test_importing_the_package_leaves_numba_unimported():
    code = ("import sys, dynamix.core.mz_lastwave; "
            "sys.exit(1 if 'numba' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code],
                          cwd=str(pathlib.Path(__file__).resolve().parent.parent)).returncode == 0
