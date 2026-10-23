# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The directional viewer reads the exact fitted populations, branches and envelopes."""
import numpy as np
import pytest

from dynamix.core import directional as dr
from dynamix.core.hyperbolic_preview import axis_preview, tensor_preview


def test_gamma_diagnostics_retain_zero_counts_and_fit_details_only():
    J = 8
    mods = {(x, y): np.full((2, 2), 2.**(-.3*((x or J)+(y or J))))
            for x in range(J+1) for y in range(J+1)}
    mods[(0, 1)][0, 0] = 0
    valid = {k: np.ones((2, 2), bool) for k in mods}
    got = dr.gamma_scaling({'levels': J, 'mods': mods, 'valid': valid}, [1., 2.], 1, step=.25)
    pairs = [tuple(p) for p in got['moment_pairs']]
    col = pairs.index((0, 1))
    assert got['moment_counts'][col] == 4
    np.testing.assert_allclose(got['log2_moments'][:, col],
                               [np.log2(np.mean(mods[(0, 1)]**p)) for p in (1., 2.)])
    for ai, fit in enumerate(got['diagnostic_fits']):
        if not fit['admissible']:
            continue
        # The tables span the complete basis; the fitted Γ levels never reach the φ branch (key 0).
        assert all(0 not in pair for members in fit['members'] for pair in members[0])
        for p in range(2):
            for i, members in enumerate(fit['members']):
                cols = [pairs.index(tuple(pair)) for pair in members[0]]
                assert fit['envelopes'][p, 0, i] == pytest.approx(got['log2_moments'][p, cols].max())
            slope, intercept = np.polyfit(fit['x'], fit['envelopes'][p, 0], 1)
            assert slope == pytest.approx(got['eta_alpha'][p, ai])
            assert intercept == pytest.approx(fit['intercepts'][p, 0])


def test_legacy_diagnostics_keep_three_families_and_the_minimum_slope():
    j1, j2 = np.meshgrid(np.arange(1, 9), np.arange(1, 9), indexing='ij')
    table = np.stack([.3*j1+.7*j2, .6*j1+1.4*j2])
    got = dr.scaling_functions(table, 1, 8, step=.25)
    for ai, fit in enumerate(got['diagnostic_fits']):
        if fit['admissible']:
            assert fit['envelopes'].shape[1] == 3
            np.testing.assert_allclose(np.min(fit['slopes'], axis=1), got['eta_alpha'][:, ai])


@pytest.mark.parametrize('index', [0, 1, 3])
def test_preview_is_the_exact_analysis_filter_polynomial(index):
    p = axis_preview('db2', index, 4)
    taps, k = p['taps'], p['frequency'][::100]
    expected = np.abs(sum(tap*np.exp(-2j*np.pi*k*n) for n, tap in enumerate(taps)))
    # The sample grid includes the exact DC maximum for low-pass and a resolved band peak.
    full_peak = np.abs(sum(tap*np.exp(-2j*np.pi*p['frequency']*n) for n, tap in enumerate(taps))).max()
    np.testing.assert_allclose(p['magnitude'][::100], expected/full_peak, atol=2e-13)
    assert taps.size == 1 + 3*(2**(index or 4)-1)
    assert np.isnan(p['wavelength']) if index == 0 else np.isfinite(p['wavelength'])
    assert p['magnitude'][0] == pytest.approx(1 if index == 0 else 0, abs=1e-12)


def test_tensor_preview_keeps_independent_axis_support_with_bounded_display():
    x, y = axis_preview('db2', 2, 4), axis_preview('db2', 4, 4)
    out = tensor_preview(x, y, limit=16)
    sx, sy = out['pitch']
    np.testing.assert_array_equal(out['values'], np.outer(y['profile'][::sy], x['profile'][::sx]))
    assert max(out['values'].shape) <= 16


@pytest.mark.parametrize('bank, index', [('db4', 3), ('sym6', 2), ('db4', 0)])
def test_preview_profile_is_the_atom_a_coefficient_measures(bank, index):
    pywt = pytest.importorskip('pywt')
    depth = 3
    p = axis_preview(bank, index, depth)
    coeffs = pywt.wavedec(np.zeros(512), bank, mode='periodization', level=depth)
    band = coeffs[0] if index == 0 else coeffs[1 + depth - index]
    band[band.size//2] = 1.0
    # An orthogonal bank's synthesis atom is its analysis atom.
    atom = pywt.waverec(coeffs, bank, mode='periodization')
    nz = np.flatnonzero(np.abs(atom) > 1e-13)
    atom = atom[nz[0]:nz[-1]+1]
    np.testing.assert_allclose(p['profile'], atom/np.abs(atom).max(), atol=1e-12)
