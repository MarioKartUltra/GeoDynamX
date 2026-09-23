# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.sieve -- island size filtering for slices and band masks."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.sieve import sieve_classes, sieve_mask


def test_identity_when_disabled_and_input_untouched():
    m = np.zeros((8, 8), bool); m[2:4, 2:4] = True
    out = sieve_mask(m, 0, 0)
    np.testing.assert_array_equal(out, m)
    assert out is not m


def test_min_and_max_band_pass():
    m = np.zeros((16, 16), bool)
    m[0, 0] = True                       # 1-px speck
    m[4:6, 4:7] = True                   # 6-px ribbon
    m[10:16, 10:16] = True               # 36-px blob
    out = sieve_mask(m, min_px=2, max_px=0)
    assert not out[0, 0] and out[4, 4] and out[10, 10]
    out = sieve_mask(m, min_px=2, max_px=10)
    assert not out[0, 0] and out[4, 4] and not out[10, 10]


def test_connectivity_8_keeps_diagonal_ribbons_4_chops_them():
    m = np.zeros((8, 8), bool)
    for i in range(6):
        m[i, i] = True                   # a pure diagonal line, 6 px
    assert sieve_mask(m, min_px=4, connectivity=8).sum() == 6
    assert sieve_mask(m, min_px=4, connectivity=4).sum() == 0
    with pytest.raises(ValueError, match="connectivity"):
        sieve_mask(m, 2, connectivity=6)


def test_sieve_classes_filters_each_class_independently_to_nan():
    c = np.zeros((16, 16))
    c[0, 0] = 1.0                        # class-1 speck
    c[4:8, 4:8] = 1.0                    # class-1 blob (16 px)
    c[12, 12] = np.nan
    out = sieve_classes(c, n_classes=2, min_px=4)
    assert np.isnan(out[0, 0])           # speck suppressed -> transparent
    assert out[4, 4] == 1.0
    assert np.isnan(out[12, 12])         # NaN stays NaN
    # class 0 (the background) is huge and survives its own floor
    assert out[1, 1] == 0.0
