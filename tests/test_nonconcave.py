# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Spec 15's nonconcave-spectrum options (Touchette & Beck 2005), off by default.

The Legendre route returns the concave envelope of D(h); the kink check flags the jump in h(q) and
the straight D(h) segment it leaves, and the Gaussian (generalised canonical) ensemble,
W ∝ |T|^q exp(γ (ln|T|)² / ln(a/L)), reaches points inside a nonconcave dip. Fixtures are exact:
at every scale, n(h) = N0 (a/L)^-f(h) points of modulus (a/L)^h, so f is the true spectrum.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.nonconcave import gaussian_ensemble, generalized_legendre, kink_flags

SCALES = 2.0 ** np.arange(3, 9)                        # 8 .. 256 px
#: Far out, so ln(a/L) ≈ -50: the canonical fit's finite-size term, about 1/(2 ln(a/L)) on the
#: slopes (the weights' Gaussian spread around the saddle), drops to about 0.01.
L = 256.0 * np.exp(50.0)


def _f_dip(h):
    """A concave cap with a dip at h = 0.5, about 0.064 below the envelope (f'' = 10 there)."""
    return 0.9 - 1.0 * (h - 0.5) ** 2 - 0.12 * np.exp(-((h - 0.5) ** 2) / (2 * 0.1 ** 2))


def _f_cap(h):
    return 0.9 - 2.2 * (h - 0.5) ** 2


def _sample(f, h=np.linspace(0.15, 0.85, 141), n0=1e3):
    mods = [(a / L) ** h for a in SCALES]
    counts = [n0 * (a / L) ** (-f(h)) for a in SCALES]
    return mods, counts, h


def _upper_hull(x, y):
    """The concave envelope of the points (x, y), evaluated at x."""
    hull = []
    for p in zip(x, y):
        while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
                                  - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])) >= 0:
            hull.pop()
        hull.append(p)
    hx, hy = zip(*hull)
    return np.interp(x, hx, hy)


def test_gamma_zero_is_the_canonical_direct_fit():
    mods, counts, _h = _sample(_f_cap)
    q = np.linspace(-1.2, 1.2, 13)              # saddles h = 0.5 - q/4.4 stay inside the h grid
    g = gaussian_ensemble(mods, SCALES, q, [0.0], 3.0, 8.0, counts_per_scale=counts, length_px=L)
    # the canonical direct fit by hand: W ∝ n |T|^q, h = slope Σ W ln|T|, D = slope Σ W ln(W/n)
    la = np.log(SCALES)
    for i, qq in enumerate(q):
        sh, sd = [], []
        for m, n in zip(mods, counts):
            w = n * m ** qq
            w = w / w.sum()
            sh.append(np.sum(w * np.log(m)))
            sd.append(np.sum(w * np.log(w / n)))
        assert g["h"][i, 0] == pytest.approx(np.polyfit(la, sh, 1)[0], abs=1e-9)
        assert g["D"][i, 0] == pytest.approx(np.polyfit(la, sd, 1)[0], abs=1e-9)
    # an exact concave spectrum is read back exactly: D(h) on f
    ok = np.isfinite(g["h"][:, 0])
    assert np.allclose(g["D"][ok, 0], _f_cap(g["h"][ok, 0]), atol=0.015)


def test_the_consistency_identity_holds_at_every_q_and_gamma():
    mods, counts, _h = _sample(_f_dip)
    g = gaussian_ensemble(mods, SCALES, np.linspace(-2, 2, 9), [0.0, 2.0, 8.0, 20.0], 3.0, 8.0,
                          counts_per_scale=counts, length_px=L)
    assert np.nanmax(np.abs(g["residual"])) < 1e-9


def test_legendre_keeps_the_envelope_and_the_gaussian_ensemble_enters_the_dip():
    mods, counts, h = _sample(_f_dip)
    # The saddle f'(h) = q + 2γh: a point at h needs q ≈ f'(h) - 2γh, so q spans what γ asks.
    q = np.linspace(-20.0, 20.0, 161)
    gammas = np.linspace(0.0, 10.0, 21)
    g = gaussian_ensemble(mods, SCALES, q, gammas, 3.0, 8.0, counts_per_scale=counts, length_px=L)
    hull = lambda x: np.interp(x, h, _upper_hull(h, _f_dip(h)))          # noqa: E731
    std_h, std_d = g["h"][:, 0], g["D"][:, 0]
    ok = np.isfinite(std_h) & (np.abs(std_h - 0.5) < 0.3)              # saddles inside the grid
    assert np.all(std_d[ok] >= hull(std_h[ok]) - 0.015)                # γ = 0: the envelope
    depth = float(np.max(_upper_hull(h, _f_dip(h)) - _f_dip(h)))
    assert depth > 0.05
    inside = (np.abs(g["h"] - 0.5) < 0.06) & np.isfinite(g["D"])
    gap = hull(g["h"][inside]) - g["D"][inside]
    assert gap.max() > 0.6 * depth                                   # well inside the dip
    best = np.argmax(gap)
    assert g["D"][inside][best] == pytest.approx(_f_dip(g["h"][inside][best]), abs=0.015)


def test_the_kink_check_fires_on_a_first_order_transition_only():
    q = np.linspace(-3, 3, 61)
    lines = np.minimum(0.3 * q - 0.2, 0.7 * q - 0.2)                 # τ(q) with a kink at q = 0
    flags = kink_flags(q, lines)
    assert len(flags) == 1
    k = flags[0]
    assert k["q"] == pytest.approx(0.0, abs=0.1)
    assert (k["h_lo"], k["h_hi"]) == pytest.approx((0.3, 0.7), abs=1e-6)
    smooth = 0.5 * q - 0.05 * q ** 2                                  # a log-normal τ(q)
    assert kink_flags(q, smooth) == []


def test_the_nonconcave_fixture_shows_its_kink_at_gamma_zero():
    mods, counts, _h = _sample(_f_dip)
    q = np.linspace(-1.2, 1.2, 49)
    g = gaussian_ensemble(mods, SCALES, q, [0.0], 3.0, 8.0, counts_per_scale=counts, length_px=L)
    flags = kink_flags(q, g["tau"][:, 0])
    assert flags and abs(flags[0]["q"]) < 0.3 and flags[0]["h_hi"] - flags[0]["h_lo"] > 0.15


def test_paper_two_point_support_is_not_a_continuous_spectrum():
    q = np.linspace(-500, 500, 2001)
    g = np.linspace(0, 256, 65)
    # Touchette–Beck (30), exact free energy for f(1/2)=0 and f(1)=1.
    tau = np.minimum(q[:, None]/2 + g[None, :]/4, q[:, None]+g[None, :]-1)
    fit = generalized_legendre(tau, q, g, [.5, .75, 1.])
    np.testing.assert_allclose(fit["D"][[0, 2]], [0., 1.], atol=1e-12)
    assert fit["D"][1] < -10   # mixed weighted means must not fabricate support here
    canonical = generalized_legendre(tau[:, :1], q, [0.], [.75])
    assert canonical["D"][0] == pytest.approx(.5)


def test_empty_levels_and_amplitude_units_have_explicit_consistent_behavior():
    scales = np.array([2., 4., 8., 16.])
    mods = [np.array([s**.4, 2*s**.7]) for s in scales]
    mods[1] = np.array([])
    a = gaussian_ensemble(mods, scales, [-1, 0, 1], [0, 2], 1, 4, length_px=64., reference_amplitude=1.0)
    b = gaussian_ensemble([1000*m for m in mods], scales, [-1, 0, 1], [0, 2], 1, 4,
                          length_px=64., reference_amplitude=1000)
    assert a["n_scales"] == 3
    np.testing.assert_allclose(a["tau"], b["tau"], atol=1e-12)
    assert np.isfinite(a["tau"]).all()


# ------------------------------------------ the shift δ and the automatic amplitude reference

def test_the_shift_delta_is_a_third_axis_and_the_inversion_takes_its_infimum():
    """Leonarduzzi et al. 2019 (26): g = −γ(h − δ)², the weight (a/L)^{q h + γ(h−δ)²}, and
    the spectrum as the infimum over (q, γ, δ) of q h + γ(h − δ)² − τ. On an exact monofractal
    (every cell h₀, cells ∝ (a/L)^{−d}) τ is q h₀ + γ(h₀ − δ)² − d for every pair, so the
    inversion returns d at h₀ whatever the shift; without ``delta_list`` the tables are the
    two-axis ones they were."""
    h0, d = 0.6, 1.0
    mods = [np.full(1000, (a / L) ** h0) for a in SCALES]
    counts = [np.full(1000, (a / L) ** (-d) / 1000) for a in SCALES]
    q, g, dl = np.linspace(-2, 2, 5), np.array([0.0, 1.0, 4.0]), np.array([0.0, 0.3, 0.6, 0.9])
    e = gaussian_ensemble(mods, SCALES, q, g, 3, 8, counts_per_scale=counts, length_px=L,
                          delta_list=dl, reference_amplitude=1.0)
    assert e["tau"].shape == (5, 3, 4) and e["delta"].tolist() == dl.tolist()
    expect = q[:, None, None] * h0 + g[None, :, None] * (h0 - dl[None, None, :]) ** 2 - d
    np.testing.assert_allclose(e["tau"], expect, atol=1e-6)
    np.testing.assert_allclose(e["h"], h0, atol=1e-6)
    grid = np.linspace(0.3, 0.9, 7)
    fit = generalized_legendre(e["tau"], q, g, grid, delta_list=dl)
    assert fit["D"][3] == pytest.approx(d, abs=1e-6) and fit["delta_star"].shape == (7,)
    flat = gaussian_ensemble(mods, SCALES, q, g, 3, 8, counts_per_scale=counts, length_px=L,
                             reference_amplitude=1.0)
    assert flat["tau"].shape == (5, 3)
    np.testing.assert_allclose(flat["tau"], e["tau"][:, :, 0], atol=1e-9)
    # the family of shifted parabolas is a tighter bound than δ = 0 alone (Theorem 1): never
    # above it, equal where the spectrum lives
    alone = generalized_legendre(flat["tau"], q, g, grid)["D"]
    assert np.all(fit["D"] <= alone + 1e-9) and alone[3] == pytest.approx(d, abs=1e-6)
    assert np.any(fit["D"] < alone - 0.1)


def test_the_automatic_reference_amplitude_is_the_intercept_and_takes_an_amplitude_out():
    """Leonarduzzi et al. 2019 (18): the regression intercept c₁,₀ of the per-scale mean of
    log₂L is the amplitude the exponents are measured against; ``reference_amplitude="auto"``
    divides by 2^{c₁,₀}, so an amplitude on every modulus changes no table."""
    mods, counts, h = _sample(_f_dip)
    q, g = np.linspace(-2, 2, 9), np.array([0.0, 2.0, 8.0])
    base = gaussian_ensemble(mods, SCALES, q, g, 3, 8, counts_per_scale=counts, length_px=L,
                             reference_amplitude="auto")
    moved = gaussian_ensemble([m * 8.0 for m in mods], SCALES, q, g, 3, 8, counts_per_scale=counts,
                              length_px=L, reference_amplitude="auto")
    assert moved["reference_amplitude"] == pytest.approx(8.0 * base["reference_amplitude"])
    for key in ("h", "D", "tau", "h2"):
        np.testing.assert_allclose(moved[key], base[key], atol=1e-9)
    raw = gaussian_ensemble([m * 8.0 for m in mods], SCALES, q, g, 3, 8, counts_per_scale=counts,
                            length_px=L, reference_amplitude=1.0)
    assert not np.allclose(raw["D"][:, 1:], base["D"][:, 1:], atol=1e-3)     # the amplitude leaks at γ > 0


def _leaders_samples():
    from test_structure_functions_window import _samples
    return _samples(H=0.5, J=5, n=64)


def test_the_generalized_table_is_the_ordinary_table_at_gamma_zero():
    """At γ = 0 the generalized weights are |T|^q with the cell multiplicities: the table in the
    partition schema is leaders.tables's, entry for entry, with the reference amplitude at one."""
    from dynamix.core import leaders
    samples = _leaders_samples()
    q = np.arange(-2.0, 4.01, 0.5)
    plain = leaders.tables(samples, q)["leaders"]
    general = leaders.generalized_table(samples, "leaders", q, 0.0, 0.0, reference_amplitude=1.0)
    for key in ("tau_qa", "h_qa", "D_qa", "N_a", "log2_scales"):
        np.testing.assert_allclose(general[key], plain[key], atol=1e-9, equal_nan=True)
    assert general["set"] == "leaders" and general["gamma"] == 0.0 and general["delta"] == 0.0
    assert leaders.generalized_table({**samples, "dyadic": True}, "leaders", q, 0.0, 0.0)["dyadic"] is True


def test_the_generalized_table_s_slopes_are_the_gaussian_ensemble_s():
    """For γ > 0 the table's per-scale log2 Z are the ensemble's ln Z in log2 units: fitted over
    the same window they give the ensemble's τ(q, γ), h and D; the per-scale identity
    D = q h + γ h2 − τ holds; δ moves the table."""
    from dynamix.core import leaders
    from dynamix.core.nonconcave import gaussian_ensemble
    samples = _leaders_samples()
    q = np.arange(-2.0, 2.01, 0.5)
    gamma, L, lo, hi = 3.0, 64.0, 1.0, 5.0
    table = leaders.generalized_table(samples, "leaders", q, gamma, 0.0, length_px=L, reference_amplitude=1.0)
    fit = leaders.structure_fit(table, lo, hi)
    sets = samples["sets"]["leaders"]
    flat = [np.asarray(v)[m] for v, m in zip(sets["values"], sets["valid"])]
    counts = [np.full(x.size, float(n) / x.size) for n, x in zip(samples["cells"], flat)]
    g = gaussian_ensemble(flat, samples["scales"], q, [gamma], lo, hi, counts_per_scale=counts,
                          length_px=L, reference_amplitude=1.0)
    np.testing.assert_allclose(fit["tau"], g["tau"][:, 0], atol=1e-9)
    np.testing.assert_allclose(fit["h"], g["h"][:, 0], atol=1e-9)
    np.testing.assert_allclose(fit["D"], g["D"][:, 0], atol=1e-9)
    identity = q[:, None] * table["h_qa"] + gamma * table["h2_qa"] - table["tau_qa"]
    np.testing.assert_allclose(table["D_qa"], identity, atol=1e-9)
    shifted = leaders.generalized_table(samples, "leaders", q, gamma, 0.3, length_px=L, reference_amplitude=1.0)
    assert not np.allclose(shifted["tau_qa"], table["tau_qa"])


def test_the_ensemble_keeps_every_scale_and_fits_the_slopes_over_the_window_alone():
    """The per-scale sums do not depend on the window: ``log_partition`` and ``scales`` cover
    every nonempty scale, ``window`` marks the ones inside [log2_a_min, log2_a_max], and the
    slopes use those alone, so a run over a narrow window equals a run restricted to its
    scales; a window holding one scale still returns every scale's log partition."""
    scales = np.array([2., 4., 8., 16., 32.])
    rng = np.random.default_rng(3)
    mods = [s**.5 * rng.lognormal(0, .3, 50) for s in scales]
    q, gam = [-1, 0, 1, 2], [0, 1]
    full = gaussian_ensemble(mods, scales, q, gam, 2, 4, length_px=64., reference_amplitude=1.0)
    assert full["scales"].tolist() == scales.tolist() and full["log_partition"].shape[-1] == 5
    np.testing.assert_array_equal(full["window"], [False, True, True, True, False])
    assert full["n_scales"] == 3
    narrow = gaussian_ensemble(mods[1:4], scales[1:4], q, gam, 2, 4, length_px=64., reference_amplitude=1.0)
    np.testing.assert_allclose(full["tau"], narrow["tau"], atol=1e-12)
    np.testing.assert_allclose(full["h"], narrow["h"], atol=1e-12)
    np.testing.assert_allclose(full["D"], narrow["D"], atol=1e-12)
    np.testing.assert_allclose(full["log_partition"][..., 1:4], narrow["log_partition"], atol=1e-12)
    thin = gaussian_ensemble(mods, scales, q, gam, 2, 2.5, length_px=64., reference_amplitude=1.0)
    assert thin["n_scales"] == 1 and np.isnan(thin["tau"]).all()
    assert thin["log_partition"].shape[-1] == 5 and np.isfinite(thin["log_partition"]).all()
