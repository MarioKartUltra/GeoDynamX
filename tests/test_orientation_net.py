# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Rose histograms retain angle frames, zero-seam observations and original counts."""
import numpy as np
import pytest
from dynamix.core.orientation import Orientation, Frame


def test_rose_zero_seam_and_axial_antipodes_keep_original_counts():
    from dynamix.core.orientation_net import rose_histogram, rose_coordinates
    o = Orientation(np.array([359., 1., 180., np.nan]), Frame.TRUE)
    directed = rose_histogram(o, bins=36)
    assert directed['total'] == 3 and directed['counts'][0] == 2
    axial = rose_histogram(o, bins=18, axial=True)
    assert axial['total'] == 3 and axial['counts'][0] == 3
    assert axial['fraction'].sum() == pytest.approx(1)
    x,y = rose_coordinates([0.,90.], 1., frame=Frame.TRUE)
    np.testing.assert_allclose(x, [0,1], atol=1e-14)
    np.testing.assert_allclose(y, [1,0], atol=1e-14)
    x,y = rose_coordinates([0.,90.], 1., frame=Frame.PIXEL)
    np.testing.assert_allclose(x, [1,0], atol=1e-14)
    np.testing.assert_allclose(y, [0,-1], atol=1e-14)
