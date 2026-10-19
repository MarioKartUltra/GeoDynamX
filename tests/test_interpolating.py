# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import numpy as np
import pytest

from dynamix.core import interpolating as iw


@pytest.mark.parametrize("bank", ["dd4", "gldaf"])
@pytest.mark.parametrize("edge", ["reflect", "periodic"])
@pytest.mark.parametrize("shape", [(32, 40), (31, 37), (8, 9)])
def test_lifting_reconstruction_and_additive_components(bank, edge, shape):
    x = np.random.default_rng(4).normal(size=shape)
    c = iw.pyramid(x, 3, bank=bank, edge=edge)
    np.testing.assert_allclose(iw.inverse(c), x, atol=2e-14)
    bands = iw.inverse(c, keep=(3, "low"))
    for j in range(1, 4):
        for name in ("horizontal", "vertical", "diagonal"):
            bands += iw.inverse(c, keep=(j, name))
    np.testing.assert_allclose(bands, x, atol=2e-14)


def test_dd_preserves_samples_and_reproduces_cubics_away_from_boundary():
    t = np.arange(30.)
    for degree in range(4):
        y = iw.refine(t**degree, 2)
        np.testing.assert_array_equal(y[::4], t**degree)
        np.testing.assert_allclose(y[16:-16], (np.arange(y.size)/4)[16:-16]**degree, atol=1e-11)
    x = np.arange(42.).reshape(6, 7)
    np.testing.assert_array_equal(iw.refine(x, 2)[::4, ::4], x)


@pytest.mark.parametrize("bank", ["dd4", "gldaf"])
def test_constant_field_has_no_detail_and_filter_preserves_dc(bank):
    c = iw.pyramid(np.full((31, 37), 2.3), 3, bank=bank)
    for bands in c["bands"]:
        for band in bands:
            np.testing.assert_allclose(band, 0, atol=1e-14)
    np.testing.assert_allclose(iw.inverse(c, threshold=100), 2.3, atol=1e-14)


def test_gl_daf_window_changes_the_predictor_and_invalid_inputs_fail():
    _, w1 = iw.predictor("gldaf", sigma=.5)
    _, w2 = iw.predictor("gldaf", sigma=2.)
    assert not np.allclose(w1, w2)
    assert w1.sum() == pytest.approx(1)
    with pytest.raises(ValueError, match="odd"):
        iw.predictor("gldaf", order=4)
    with pytest.raises(ValueError, match="nodata"):
        iw.pyramid(np.full((8, 8), np.nan), 2)
