# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import numpy as np

from dynamix.core import directional as dr, dwt


def test_literal_gamma_sets_and_closed_form_sheet_exponents():
    hh, hl, lh = dr.resolution_families(4, .5)
    assert set(hh) == {(x, y) for x in range(0, 4) for y in range(3, 8)}
    assert set(hl) == {(x, y) for x in range(0, 4) for y in range(3)}
    assert not lh
    J = 32
    mods = {(x, y): np.array([[2.**(.3*(x or J)+.7*(y or J))]])
            for x in range(J+1) for y in range(J+1)}
    coef = {"levels": J, "mods": mods, "valid": {k: np.ones((1, 1), bool) for k in mods}}
    scaling = dr.gamma_scaling(coef, [1., 2.], 2, step=.25)
    directions = dr.directional_scaling(scaling)
    np.testing.assert_allclose(directions["eta_e1"], [.3, .6], atol=.035)
    np.testing.assert_allclose(directions["eta_e2"], [.7, 1.4], atol=.06)


def test_complete_hyperbolic_keeps_mixed_branches_and_detail_coefficients():
    x = np.random.default_rng(2).normal(size=(64, 80))
    complete = dwt.hyperbolic_complete(x, "haar", 4)
    detail = dwt.hyperbolic(x, "haar", 4)
    assert (0, 0) in complete and (0, 3) in complete and (2, 0) in complete
    for key, value in detail.items():
        np.testing.assert_allclose(complete[key], value)


def test_real_block_origin_restricts_hyperbolic_sample_masks():
    c = dr.coefficients(np.ones((128, 128)), "haar", 4, edge="periodic", complete=True,
                        origin=(32, 16), block=(64, 64))
    v = c["valid"][(2, 3)]
    assert not v[:4].any() and not v[:, :4].any()
    assert v[4:12, 4:20].all() and v.sum() == 128
