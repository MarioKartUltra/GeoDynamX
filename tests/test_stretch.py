# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""dynamix.core.stretch — contrast stretches for display.

One function maps a field to [0, 1] for colouring: linear (min/max), percent (clip at the
p-th / (100-p)-th percentiles), stddev (mean ± k·σ), log (log1p over the positive-shifted
range), histogram (equalisation by rank). NaN stays NaN. Display only — never analysis."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.stretch import STRETCHES, stretch


@pytest.fixture
def skewed():
    rng = np.random.default_rng(0)
    v = rng.lognormal(0.0, 1.0, size=(64, 64))
    v[3, 3] = 1e4                    # one absurd outlier
    v[5, 5] = np.nan
    return v


def test_linear_uses_the_full_range_so_one_outlier_crushes_the_rest(skewed):
    n = stretch(skewed, "linear")
    assert np.nanmin(n) == 0.0 and np.nanmax(n) == 1.0
    assert np.isnan(n[5, 5])
    assert np.nanmedian(n) < 0.01                # everything but the outlier is near black


def test_percent_clips_the_tails_and_spreads_the_middle(skewed):
    n = stretch(skewed, "percent", percent=2.0)
    assert np.nanmedian(n) > 10 * np.nanmedian(stretch(skewed, "linear"))   # the middle is visible again
    assert n[3, 3] == 1.0                        # clipped to the top, not beyond


def test_stddev_stretch_is_mean_plus_minus_k_sigma(skewed):
    v = np.linspace(-10, 10, 400).reshape(20, 20)
    n = stretch(v, "stddev", k=1.0)
    m, s = v.mean(), v.std()
    inside = (v > m - s) & (v < m + s)
    assert np.all((n[inside] > 0) & (n[inside] < 1))
    assert n[v <= m - s].max() == 0.0 and n[v >= m + s].min() == 1.0


def test_log_stretch_lifts_the_dark_end(skewed):
    lin = stretch(skewed, "linear"); lg = stretch(skewed, "log")
    assert np.nanmedian(lg) > np.nanmedian(lin)
    assert 0.0 <= np.nanmin(lg) and np.nanmax(lg) <= 1.0


def test_histogram_equalisation_is_uniform_in_rank(skewed):
    n = stretch(skewed, "histogram")
    finite = n[np.isfinite(n)]
    hist, _ = np.histogram(finite, bins=10, range=(0, 1))
    assert hist.max() < 1.6 * hist.min()         # flat within noise
    assert np.isnan(n[5, 5])


def test_constant_field_and_unknown_mode_are_handled_honestly():
    flat = np.full((4, 4), 7.0)
    assert np.all(stretch(flat, "linear") == 0.0)
    with pytest.raises(ValueError, match="stretch"):
        stretch(flat, "cubist")
    assert set(STRETCHES) == {"linear", "percent", "stddev", "log", "histogram"}


# ------------------------------------------------------- density slice (ENVI, 2026-09-16)

def test_parse_levels_grammar():
    from dynamix.core.stretch import parse_levels
    assert parse_levels("") is None and parse_levels(None) is None
    assert parse_levels("5") == 5
    assert parse_levels(" -0.5, 0, 0.8 ") == [-0.5, 0.0, 0.8]
    assert parse_levels("-3") == [-3.0]          # a single NEGATIVE number is a break, not a count
    with pytest.raises(ValueError, match="levels"):
        parse_levels("h < 0.5")


def test_classify_breaks_are_data_unit_classes():
    from dynamix.core.stretch import classify
    v = np.array([[-1.0, -0.2], [0.4, 2.0], [np.nan, 0.9]])
    out = classify(v, [-0.5, 0.0, 0.8])          # 4 classes -> indices 0..3 over /3
    expect = np.array([[0.0, 1 / 3], [2 / 3, 1.0], [np.nan, 1.0]])
    np.testing.assert_allclose(out, expect, equal_nan=True)


def test_classify_int_is_quantile_classes():
    from dynamix.core.stretch import classify
    rng = np.random.default_rng(0)
    v = rng.normal(size=(64, 64))
    out = classify(v, 4)
    vals = np.unique(out[np.isfinite(out)])
    np.testing.assert_allclose(vals, [0.0, 1 / 3, 2 / 3, 1.0])
    # quantile slicing -> near-equal populations
    counts = [(out == x).sum() for x in vals]
    assert max(counts) - min(counts) < v.size * 0.02


def test_parse_class_colors_rgba_with_transparent_none():
    from dynamix.core.stretch import parse_class_colors
    assert parse_class_colors("") is None
    assert parse_class_colors("#ff0000,none,#0000ff") == [
        (255, 0, 0, 255), (0, 0, 0, 0), (0, 0, 255, 255)]
    import pytest as _pt
    with _pt.raises(ValueError, match="class color"):
        parse_class_colors("#ff00")
