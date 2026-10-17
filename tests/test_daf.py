# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The DAF periodic extension (spec 22 §3.4): the Hermite DAF (Frishman et al. 1996 Eq. 9), the gap
solve (Hoffman et al. 2000 Eq. 6), the self-tuning rule from the data's band limit, and its
fallback to reflection where the data are not band-limited or too short."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import daf, fft_policy


def _frishman_fig5():
    """Frishman et al. 1996 Fig. 5: 336 points on [-4.14, 4.09], the points in [-2, 2] removed."""
    x = -4.14 + 0.02457 * np.arange(336)
    f = 4 + 5 * np.exp(-(x - 1) ** 2) + 2 * sum(np.cos(k * x) for k in range(1, 5))
    return x, f, ~((x >= -2) & (x <= 2))


def _labyrinth(n, k0, seed=3):
    """Annulus-filtered white noise on a 2n grid, cropped to n: band-limited, not periodic."""
    rng = np.random.default_rng(seed)
    m = 2 * n
    k = np.hypot(np.fft.fftfreq(m)[:, None], np.fft.fftfreq(m)[None, :])
    spec = fft_policy.double().fft2(rng.standard_normal((m, m)))
    field = fft_policy.double().ifft2(spec * np.exp(-((k - k0) / (0.2 * k0)) ** 2)).real[:n, :n]
    return field / field.std()


def _power_above(a, f):
    """Share of the Hann-windowed power above radial frequency ``f`` (cycles/px)."""
    a = np.asarray(a, float) - np.mean(a)
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    p = np.abs(fft_policy.double().fft2(a * win)) ** 2
    k = np.hypot(np.fft.fftfreq(a.shape[0])[:, None], np.fft.fftfreq(a.shape[1])[None, :])
    return float(p[k > f].sum() / p.sum())


def test_the_taps_approximate_the_identity_and_the_gain_rule_holds():
    t = daf.hermite_daf(6.0, 12)
    assert t.sum() == pytest.approx(1.0, abs=1e-9) and np.allclose(t, t[::-1])
    s = daf.sigma_for_gain(0.05, 12)
    assert daf.daf_gain(2 * np.pi * 0.05, s, 12) == pytest.approx(0.99, abs=1e-9)
    assert daf.daf_gain(2 * np.pi * 0.05, 1.01 * s, 12) < 0.99


def test_frishman_fig5_gap_fill_within_its_published_error():
    x, f, known = _frishman_fig5()
    got = daf.fill_gap(np.where(known, f, 0.0), known, sigma=6.0, M=12, periodic=False)
    rel = np.abs(got[~known] - f[~known]) / np.abs(f[~known])
    assert abs(int((~known).sum()) - 163) <= 1           # the paper's 163; the grid's rounding
    assert rel.max() <= 0.017
    assert abs(x[~known][rel.argmax()] + 0.87) < 0.05     # where the paper puts its worst point


def test_patent_case_one_periodic_gap():
    """US 6,847,737 Case I without noise: sin(5πx/128) known on 0..219, period 256, W = 1."""
    x = np.arange(256.0)
    f = np.sin(5 * np.pi * x / 128)
    known = x < 220
    got = daf.fill_gap(np.where(known, f, 0.0), known, sigma=4.0, M=6, periodic=True,
                       weights=np.ones(256))
    assert np.abs(got - f)[~known].max() <= 1e-4


def test_lines_share_one_solve():
    x, f, known = _frishman_fig5()
    lines = np.stack([np.where(known, f, 0.0), np.where(known, 2 * f + 1, 0.0)])
    got = daf.fill_gap(lines, known, sigma=6.0, M=12, periodic=False)
    one = daf.fill_gap(lines[0], known, sigma=6.0, M=12, periodic=False)
    assert got.shape == lines.shape and np.allclose(got[0], one)
    assert np.allclose(got[1], 2 * one + 1, atol=1e-6)


@pytest.mark.parametrize("ppw", [6.5, 8.0, 12.0])   # exactly 6 measures 5.9: Hann leakage
def test_self_tuned_sine_extension_stays_inside_the_data_range(ppw):
    x = np.arange(300.0)
    g = np.sin(2 * np.pi * x / ppw + 0.3)
    t = daf.self_tune(g[None, :])
    assert t.fallback is None and t.ppw >= 6.0
    ext = daf.extend_periodic(g[None, :], t.gap, t.sigma, t.M, t.lam)[0, 300:]
    assert ext.size == t.gap
    assert ext.min() >= g.min() - 1e-6 and ext.max() <= g.max() + 1e-6


def test_labyrinth_leaks_less_above_one_and_a_half_band_limits_than_reflection():
    field = _labyrinth(256, k0=1 / 12)
    t = daf.self_tune(field)
    assert t.fallback is None
    ext = daf.extend_2d(field, t)
    assert ext.shape == (256 + t.gap, 256 + t.gap)
    assert np.array_equal(ext[:256, :256], field)
    refl = np.pad(field, ((0, t.gap), (0, t.gap)), mode="reflect")
    assert _power_above(ext, 1.5 * t.f_b) < _power_above(refl, 1.5 * t.f_b)


def test_five_points_per_wavelength_falls_back_to_reflect():
    x = np.arange(256.0)
    t = daf.self_tune(np.sin(2 * np.pi * x / 5.0)[None, :])
    assert t.fallback and "points per wavelength" in t.fallback


def test_a_thresholded_grain_image_falls_back():
    assert daf.self_tune(np.tanh(8 * _labyrinth(256, k0=1 / 20))).fallback


def test_a_line_shorter_than_the_stencil_falls_back_instead_of_raising():
    t = daf.self_tune(np.sin(np.arange(12.0) / 3.0)[None, :])
    assert t.fallback
