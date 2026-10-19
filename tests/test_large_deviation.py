# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The large-deviation spectrum from per-scale moduli: the fixed-ε route reaches a non-concave
dip on the exact two-population fixture and the adaptive route reads the hull; on a binomial
cascade's leaders the adaptive (h(q), D(q)) agrees with the Legendre spectrum of the fitted τ;
the WTMM's maxima lines give a monofractal's narrow spectrum; the bound D_LD ≤ L holds."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from dynamix.core import large_deviation as ld
from dynamix.core.nonconcave import gaussian_ensemble
from dynamix.core.spectra import fit_spectra

sys.path.insert(0, "tests")
from conftest import fbm2d                                                      # noqa: E402
from test_nonconcave import L as L_FAR, SCALES, _f_dip, _sample, _upper_hull    # noqa: E402

Q = np.arange(-2.0, 4.01, 0.5)


def test_counts_window_each_bin_and_local_exponents_read_the_fixture_exactly():
    mods, counts, h = _sample(_f_dip)
    assert np.allclose(ld.local_exponents(mods[2], SCALES[2], L_FAR), h)
    n = ld.counts(mods, SCALES, [0.3, 0.5, 0.7], 0.0125, length_px=L_FAR, counts_per_scale=counts)
    assert n.shape == (SCALES.size, 3)
    for k in range(SCALES.size):                      # the five fixture points within ±0.0125
        assert n[k, 1] == pytest.approx(counts[k][np.abs(h - 0.5) <= 0.0125 + 1e-12].sum())
    with pytest.raises(ValueError, match="below the field extent"):
        ld.fixed(mods, SCALES, h_grid=h, log2_a_min=3, log2_a_max=8, length_px=100.0)


def test_the_fixed_route_reaches_the_dip_and_the_adaptive_route_reads_the_hull():
    mods, counts, h = _sample(_f_dip)
    grid = np.linspace(0.2, 0.8, 61)
    f = ld.fixed(mods, SCALES, h_grid=grid, log2_a_min=3, log2_a_max=8, length_px=L_FAR,
                 eps_list=(0.025, 0.0125), counts_per_scale=counts)
    assert f["D"].shape == (2, 61) and f["log2_counts"].shape == (2, 6, 61)
    assert np.allclose(f["D"][1], _f_dip(grid), atol=0.02)
    assert np.all(f["spread"] < 0.03)
    hull = np.interp(grid, h, _upper_hull(h, _f_dip(h)))
    assert np.allclose(f["D"][1], _f_dip(grid), atol=0.02) and (hull - f["D"][1]).max() > 0.05   # the dip
    a = ld.adaptive(mods, SCALES, q_list=np.linspace(-1, 1, 41), log2_a_min=3, log2_a_max=8,
                    length_px=L_FAR, counts_per_scale=counts)
    ok = np.isfinite(a["h"]) & (a["h"] > 0.2) & (a["h"] < 0.8)
    assert ok.sum() >= 15
    gap = a["D"][ok] - np.interp(a["h"][ok], h, _upper_hull(h, _f_dip(h)))
    assert np.abs(gap).max() < 0.05 and np.abs(gap).mean() < 0.03         # the hull, on a 0.005 grid
    assert np.all(a["eps"][:, ok] >= 0) and a["centres"].shape == (6, 41)


def test_the_bound_holds_against_the_legendre_spectrum_of_the_same_input():
    mods, counts, h = _sample(_f_dip)
    g = gaussian_ensemble(mods, SCALES, np.linspace(-6, 6, 97), [0.0], 3, 8,
                          counts_per_scale=counts, length_px=L_FAR)
    grid = np.linspace(0.2, 0.8, 61)
    f = ld.fixed(mods, SCALES, h_grid=grid, log2_a_min=3, log2_a_max=8, length_px=L_FAR,
                 eps_list=(0.0125,), counts_per_scale=counts)
    excess = ld.hull_excess(grid, f["D"][0], g["q"], g["tau"][:, 0])
    inside = np.isfinite(excess)
    assert inside.sum() > 40 and np.nanmax(excess) < 0.03            # never above the hull


def _cascade(n_levels, w, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.ones((1, 1))
    for _ in range(n_levels):
        ny, nx = mu.shape
        perm = np.stack([rng.permutation(w) for _ in range(ny * nx)]).reshape(ny, nx, 2, 2)
        mu = (mu[:, :, None, None] * perm).transpose(0, 2, 1, 3).reshape(2 * ny, 2 * nx)
    return mu


def test_the_adaptive_route_on_leaders_matches_the_tables_h_and_the_legendre_d():
    from dynamix.core import leaders as LE

    mu = _cascade(10, np.array([0.1, 0.2, 0.3, 0.4]))
    r = LE.run_leaders(mu, wavelet="haar", scale_min=4, scale_max=32, gamma=1.0, wse_rule="fixed")
    samples = [m[v] for m, v in zip(r["quantity"]["leaders"], r["valid"]["leaders"])]
    weight = [(1024 // 2 ** j) ** 2 / max(v.sum(), 1) for j, v in enumerate(r["valid"]["leaders"], start=1)]
    a = ld.adaptive(samples, r["scales"], q_list=Q, log2_a_min=2, log2_a_max=5, length_px=1024.0,
                    scale_weight=weight)
    fit = fit_spectra(LE.tables(r["samples"], Q)["leaders"], 2, 5)
    assert np.allclose(a["h"], fit["h"], atol=1e-9)                   # the same tilted means
    from dynamix.core.spectra import legendre_dh

    hull = legendre_dh(Q, fit["tau"])
    on_hull = np.interp(a["h"], hull["h"], hull["D"])
    mid = (Q >= -1.0) & (Q <= 0.5)
    assert np.allclose(a["D"][mid], on_hull[mid], atol=0.1)             # where the counts dominate
    assert np.all(a["D"] - on_hull < 0.3) and np.all(a["D"] - on_hull > -0.15)  # finite-bin excess, count noise
    assert a["D"][Q == 0.0] == pytest.approx(2.0, abs=0.1)
    shift = ld.prefactor_shift(samples, r["scales"], log2_a_min=2, log2_a_max=5, length_px=1024.0)
    unweighted = ld.adaptive(samples, r["scales"], q_list=Q, log2_a_min=2, log2_a_max=5, length_px=1024.0)
    assert unweighted["D"][Q == 0.0][0] > 2.03                        # the border's thinning, uncorrected
    assert shift.shape == (4,) and np.all(np.abs(shift) > 0.5)         # the fixed route's bias here


def test_the_wtmm_maxima_lines_give_a_monofractals_narrow_spectrum(clean_registry):
    """The WTMM route counts maxima LINES (one modulus per chain per scale, as the partition
    tables do), whose number scales as a^{-2} on a field singular everywhere; every maximum
    would count edge pixels along curves, a^{-1}."""
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import get_device, validate_params
    from dynamix.core.partition_tables import _modulus_matrix
    from conftest import FIXED_EDGES, make_field

    register_builtin_devices()
    dev = get_device("wtmm2d")
    p = validate_params(dev, {"n_oct": 3, **FIXED_EDGES})
    res = dev.compute(make_field(fbm2d(256, 0.5, seed=8)), p)
    scales = np.asarray(res["scales"], dtype=np.float64)
    M = _modulus_matrix(res["chains"], scales.size)
    mods = [M[np.isfinite(M[:, k]), k] for k in range(scales.size)]
    every = ld.adaptive([e["mod"] for e in res["extrema"]], scales, q_list=np.array([0.0]),
                        log2_a_min=np.log2(scales[1]), log2_a_max=np.log2(scales[-2]), length_px=256.0)
    assert every["D"][0] < 1.3                                          # edge pixels, not lines
    lo, hi = np.log2(scales[1]), np.log2(scales[-2])
    a = ld.adaptive(mods, scales, q_list=Q, log2_a_min=lo, log2_a_max=hi, length_px=256.0)
    ok = np.isfinite(a["D"])
    assert ok.sum() >= 8
    assert abs(a["h"][Q == 0.0][0] - 0.5) < 0.15
    assert 1.5 < a["D"][Q == 0.0][0] <= 2.1
    assert a["h"][ok].max() - a["h"][ok].min() < 0.6
