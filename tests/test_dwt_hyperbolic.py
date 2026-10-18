# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The hyperbolic (fully separable) dyadic transform behind the directional formalism: every
pair of detail levels (j1, j2), in L¹ normalisation."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import dwt

pytest.importorskip("pywt")


def _work(n=128, seed=0):
    rng = np.random.default_rng(seed)
    return dwt.mirror(rng.standard_normal((n // 2, n // 2)))      # (n, n), periodic by mirror


@pytest.mark.parametrize("name", ["sym4", "db4", "bl3"])
def test_the_diagonal_is_the_pyramids_diagonal_band_in_l1(name):
    x, J = _work(), 3
    hyp = dwt.hyperbolic(x, name, J)
    pyr = dwt.pyramid(x, name, J)
    assert set(hyp) == {(j1, j2) for j1 in range(1, J + 1) for j2 in range(1, J + 1)}
    for j in range(1, J + 1):
        diagonal_band = pyr[J + 1 - j][2]                            # cD at level j (finest = 1)
        assert hyp[(j, j)].shape == diagonal_band.shape
        assert np.allclose(hyp[(j, j)], 2.0 ** (-j) * diagonal_band, atol=1e-12)
    for (j1, j2), c in hyp.items():
        assert c.shape == (x.shape[0] // 2 ** j2, x.shape[1] // 2 ** j1)


def test_a_product_field_separates_into_the_outer_product_of_its_one_dimensional_details():
    """u(x) v(y): the (j1, j2) pair is the outer product of v's level-j2 detail along y and
    u's level-j1 detail along x, each at its own L¹ scale 2^{−j/2}."""
    import pywt

    n, J = 128, 3
    rng = np.random.default_rng(1)
    u = np.concatenate([rng.standard_normal(n // 2)] * 2)[:n]       # periodic rows
    u = np.concatenate([u[:n // 2], u[:n // 2][::-1]])              # even: a mirrored line
    v = rng.standard_normal(n // 2)
    v = np.concatenate([v, v[::-1]])
    x = np.outer(v, u)                                               # x[y, x] = v(y) u(x)
    hyp = dwt.hyperbolic(x, "sym4", J)
    w = dwt.wavelet("sym4")
    du = pywt.wavedec(u, w, mode="periodization", level=J)          # [cA_J, cD_J, ..., cD_1]
    dv = pywt.wavedec(v, w, mode="periodization", level=J)
    for j1 in range(1, J + 1):
        for j2 in range(1, J + 1):
            expected = np.outer(dv[J + 1 - j2], du[J + 1 - j1]) * 2.0 ** (-(j1 + j2) / 2.0)
            assert np.allclose(hyp[(j1, j2)], expected, atol=1e-12), (j1, j2)


def test_a_field_constant_along_y_has_no_mixed_detail_and_a_sheet_has_energy_everywhere():
    n, J = 64, 2
    x_only = np.tile(np.sin(np.arange(n) * 2 * np.pi * 3 / n), (n, 1))
    for c in dwt.hyperbolic(x_only, "sym4", J).values():
        assert np.abs(c).max() < 1e-12                               # the y detail of a constant is 0
    rng = np.random.default_rng(2)
    sheet = dwt.mirror(rng.standard_normal((n // 2, n // 2)))
    assert all(np.abs(c).max() > 0 for c in dwt.hyperbolic(sheet, "sym4", J).values())
