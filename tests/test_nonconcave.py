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


def test_a_minimum_at_gamma_zero_sits_on_no_delta_limit():
    """At γ = 0 the inverted value q h − τ(q, 0, δ) is the same for every δ (the weights do not
    see the shift), so the minimiser's δ is a tie broken by the first grid value: that is no
    grid limit, and a Legendre-region minimum with q inside the grid is interior. The δ limits
    still count wherever γ > 0, and ``gamma_cap`` marks the minima bounded by the largest γ
    alone."""
    q, g, dl = np.linspace(-2, 2, 9), np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.5, 1.0])
    # τ = q h₀ − 1 − 0.1 q² + γ (h₀ − δ)² − 0.1 γ (δ − ½)² + c γ: at h₀ the inverted value is
    # 1 + 0.1 q² + 0.1 γ (δ − ½)² − c γ, smallest at q = 0, at δ = ½ once γ > 0, and at γ = 0
    # (c < 0) or at the largest γ (c > 0).
    h0 = 0.5
    Q, G, DL = q[:, None, None], g[None, :, None], dl[None, None, :]
    base = Q * h0 - 1.0 - 0.1 * Q ** 2 + G * (h0 - DL) ** 2 - 0.1 * G * (DL - 0.5) ** 2
    fit = generalized_legendre(base - 0.3 * G, q, g, [h0], delta_list=dl)
    assert fit["gamma_star"][0] == 0.0 and fit["q_star"][0] == 0.0 and fit["delta_star"][0] == dl.min()
    assert not fit["boundary"][0] and not fit["gamma_cap"][0]
    top = generalized_legendre(base + 0.3 * G, q, g, [h0], delta_list=dl)
    assert top["gamma_star"][0] == g.max() and top["delta_star"][0] == 0.5
    assert top["boundary"][0] and top["gamma_cap"][0]
    edge = generalized_legendre(base + 0.3 * G, q, g, [h0 + 1.0], delta_list=dl)   # q and δ limits too
    assert edge["q_star"][0] == q.min() and edge["boundary"][0] and not edge["gamma_cap"][0]


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


# ------------------------------------------------- diagnostics of the generalized estimate

def test_the_profile_in_gamma_flags_a_falling_estimate_and_passes_a_plateau():
    """D(h; γ), the minimum over (q, δ) at each γ, on a monofractal at h* = 0.5 (τ = q h* +
    γ h*² − 1): at h* every pair gives 1, a plateau, converged; at h = 0.4 the profile is
    0.9 − 0.09 γ, falling by 0.09 per step, above the tolerance; at h = 0.6 it rises with γ. The
    tolerance is max(0.01, 0.02 × the range of the finite estimate's values ≥ 0); the far h = 2,
    with a negative estimate, does not widen it."""
    from dynamix.core.nonconcave import gamma_convergence
    q, g, h = np.array([-1.0, 0.0, 1.0]), np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.4, 0.5, 0.6, 2.0])
    tau = q[:, None] * 0.5 + g[None, :] * 0.25 - 1.0
    c = gamma_convergence(tau, q, g, h)
    np.testing.assert_allclose(c["profile"][:, 0], 0.9 - 0.09 * g, atol=1e-12)
    np.testing.assert_allclose(c["profile"][:, 1], 1.0, atol=1e-12)
    np.testing.assert_allclose(c["D"], generalized_legendre(tau, q, g, h)["D"], atol=1e-12)
    assert c["decrease"][0] == pytest.approx(0.09) and c["decrease"][1] == pytest.approx(0.0, abs=1e-12)
    assert c["D"][3] < 0
    kept = c["D"][np.isfinite(c["D"]) & (c["D"] >= 0)]
    assert c["tol"] == pytest.approx(max(0.01, 0.02 * (kept.max() - kept.min())))
    assert c["converged"].tolist() == [False, True, True, True]          # h = 2 rises with γ
    single = gamma_convergence(tau[:, :1], q, g[:1], h)                    # one γ: nothing to extend
    assert single["converged"].all()


def test_the_tolerance_scales_with_the_spectrum_s_range():
    """A log-normal D(h) = top − (h − ½)²/(2λ), zero at h = 0 and 1, whose profile falls by 0.03
    from γ = 0 to γ = 1 at h = ½: converged against a spectrum 2 high (tol 0.02 × 1.97), not
    against one 1 high (tol 0.02 × 0.97)."""
    from dynamix.core.nonconcave import gamma_convergence
    q, g, h = np.arange(-10.0, 10.001, 0.25), np.array([0.0, 1.0]), np.array([0.0, 0.5, 1.0])
    for top, expect in ((2.0, True), (1.0, False)):
        lam = 0.125 / top
        tau = 0.5 * q[:, None] - top - 0.5 * lam * q[:, None] ** 2 + 0.28 * g[None, :]
        c = gamma_convergence(tau, q, g, h)
        np.testing.assert_allclose(c["D"], [-0.28, top - 0.03, 0.0], atol=1e-9)
        assert c["decrease"][1] == pytest.approx(0.03) and c["tol"] == pytest.approx(0.02 * (top - 0.03))
        assert bool(c["converged"][1]) is expect


def test_convergence_is_a_fall_per_unit_gamma_whatever_the_gamma_step():
    """The monofractal of the profile test on γ ∈ {0, 0.1, …, 1}: at h = 0.4 the profile falls
    steadily, 0.09 per unit γ, 0.009 per step, under the tolerance 0.01 per step but not per
    unit γ: still falling, not converged. At h* = 0.5 the plateau is converged. On the unit
    step grid the two readings agree."""
    from dynamix.core.nonconcave import gamma_convergence
    q, h = np.array([-1.0, 0.0, 1.0]), np.array([0.4, 0.5])
    for g in (np.linspace(0.0, 1.0, 11), np.arange(0.0, 4.0)):
        tau = q[:, None] * 0.5 + g[None, :] * 0.25 - 1.0
        c = gamma_convergence(tau, q, g, h)
        step = g[1] - g[0]
        assert c["tol"] == pytest.approx(0.01)
        assert c["decrease"][0] == pytest.approx(0.09 * step) and c["slope"][0] == pytest.approx(0.09)
        assert c["slope"][1] == pytest.approx(0.0, abs=1e-12)
        assert c["converged"].tolist() == [False, True]
    assert 0.09 * 0.1 < 0.01                                   # the per-step rule would have passed it


def test_a_gamma_listed_twice_is_one_gamma_in_the_slope():
    """A grid whose largest γ is repeated reads as the grid without the repeat: the slope over the
    two largest distinct γ, finite, the same verdicts; a grid of one γ repeated is a single γ."""
    import warnings

    from dynamix.core.nonconcave import gamma_convergence
    q, h = np.array([-1.0, 0.0, 1.0]), np.array([0.4, 0.5, 0.6])
    g = np.array([0.0, 1.0, 2.0, 3.0])
    tau = q[:, None] * 0.5 + g[None, :] * 0.25 - 1.0
    once = gamma_convergence(tau, q, g, h)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        twice = gamma_convergence(tau[:, [0, 1, 2, 3, 3]], q, g[[0, 1, 2, 3, 3]], h)
        alone = gamma_convergence(tau[:, [2, 2]], q, g[[2, 2]], h)
    assert np.isfinite(twice["slope"]).all() and np.isfinite(twice["decrease"]).all()
    np.testing.assert_allclose(twice["slope"], once["slope"], atol=1e-12)
    np.testing.assert_allclose(twice["decrease"], once["decrease"], atol=1e-12)
    assert twice["converged"].tolist() == once["converged"].tolist() == [False, True, True]
    assert alone["converged"].all() and not alone["slope"].any()


def test_a_concave_parabola_has_no_hollow():
    from dynamix.core.nonconcave import hollows
    h = np.linspace(0.0, 1.0, 51)
    out = hollows(h, 1.0 - 4.0 * (h - 0.5) ** 2, np.full(h.size, 0.02))
    np.testing.assert_allclose(out["gap"], 0.0, atol=1e-12)
    assert out["hollows"] == []


def test_two_humps_leave_one_hollow_between_them():
    """D = max of two caps at 0.3 and 0.7: the hull is flat at 1 between them and the gap
    1 − D peaks at 0.8 at h = 0.5. With a half-width of 0.09 the hollow is where the gap
    exceeds 0.18, h ∈ [0.40, 0.60], and it survives; with a half-width of 0.5 nothing clears
    the band, and the stretch above the 0.05 floor, h ∈ [0.36, 0.64], is reported as not
    surviving; with no jackknife the floor alone decides and survival is unknown."""
    from dynamix.core.nonconcave import hollows
    h = np.round(np.arange(0.0, 1.0001, 0.02), 10)
    D = np.maximum(1 - 20 * (h - 0.3) ** 2, 1 - 20 * (h - 0.7) ** 2)
    out = hollows(h, D, np.full(h.size, 0.09))
    np.testing.assert_allclose(out["hull"][(h >= 0.3) & (h <= 0.7)], 1.0, atol=1e-12)
    [hol] = out["hollows"]
    assert (hol["h_lo"], hol["h_hi"]) == pytest.approx((0.40, 0.60))
    assert hol["depth"] == pytest.approx(0.8) and hol["survives"] is True
    [weak] = hollows(h, D, np.full(h.size, 0.5))["hollows"]
    assert (weak["h_lo"], weak["h_hi"]) == pytest.approx((0.36, 0.64)) and weak["survives"] is False
    [bare] = hollows(h, D)["hollows"]
    assert (bare["h_lo"], bare["h_hi"]) == pytest.approx((0.36, 0.64)) and bare["survives"] is None


def test_a_bound_point_in_a_dip_keeps_one_hollow_and_adds_no_depth():
    """The two caps again, with the points at h = 0.46 and 0.5 not solid (minima on a grid
    limit: upper bounds, here 0.1 above the cap curve). The hull runs over the solid points
    alone and the gap is measured at the bound points too: one hollow, h ∈ [0.40, 0.60], as
    with every point solid; its depth, 0.7 from the bound point at h = 0.5, stays under the true
    0.8. Dropping the bound points instead splits the dip in two. A bound point far above the
    hull does not lift it; a point beyond the solid points' ends has no gap."""
    from dynamix.core.nonconcave import hollows
    h = np.round(np.arange(0.0, 1.0001, 0.02), 10)
    D = np.maximum(1 - 20 * (h - 0.3) ** 2, 1 - 20 * (h - 0.7) ** 2)
    bound = np.isin(h, [0.46, 0.5])
    up = np.where(bound, D + 0.1, D)
    out = hollows(h, up, np.full(h.size, 0.09), solid=~bound)
    np.testing.assert_allclose(out["hull"][(h >= 0.3) & (h <= 0.7)], 1.0, atol=1e-12)
    np.testing.assert_allclose(out["gap"][bound], 1.0 - up[bound], atol=1e-12)
    [hol] = out["hollows"]
    assert (hol["h_lo"], hol["h_hi"]) == pytest.approx((0.40, 0.60)) and hol["survives"] is True
    assert hol["depth"] == pytest.approx(0.7)
    split = hollows(h, np.where(bound, np.nan, D), np.full(h.size, 0.09))["hollows"]
    assert [v for x in split for v in (x["h_lo"], x["h_hi"])] == pytest.approx([0.40, 0.44, 0.52, 0.60])
    high = hollows(h, np.where(np.isclose(h, 0.3), 3.0, up), np.full(h.size, 0.09),
                   solid=~bound & ~np.isclose(h, 0.3))
    assert np.nanmax(high["hull"]) <= 1.0 + 1e-12 and high["gap"][np.isclose(h, 0.3)][0] < 0
    inner = (h > 0.1) & (h < 0.9)
    ends = hollows(h, D, solid=inner)
    assert np.isnan(ends["gap"][~inner]).all() and np.isfinite(ends["gap"][inner]).all()


def test_a_dip_shorter_than_three_points_is_no_hollow():
    """One or two low points under a concave curve: the gap is there but the stretch is
    shorter than three grid points. A missing point breaks a stretch."""
    from dynamix.core.nonconcave import hollows
    h = np.linspace(0.0, 1.0, 51)
    D = 1.0 - 4.0 * (h - 0.5) ** 2
    one = D.copy()
    one[20] -= 0.3
    out = hollows(h, one, np.zeros(h.size))
    assert out["gap"][20] == pytest.approx(0.3, abs=0.002) and out["hollows"] == []   # less the chord's sag
    two = D.copy()
    two[20:22] -= 0.3
    assert hollows(h, two, np.zeros(h.size))["hollows"] == []
    holed = D.copy()
    holed[20:25] -= 0.3
    holed[22] = np.nan
    assert hollows(h, holed, np.zeros(h.size))["hollows"] == []
    holed[22] = D[22] - 0.3
    assert len(hollows(h, holed, np.zeros(h.size))["hollows"]) == 1


def test_the_jackknife_refits_are_the_ensemble_over_the_other_scales():
    """Each leave-one-scale-out row is gaussian_ensemble over the window's other scales at the
    reference amplitude the full fit used, inverted on the same h grid; the band is the min
    and max over rows, the half-width half their difference."""
    from dynamix.core.nonconcave import scale_jackknife
    scales = 2.0 ** np.arange(1, 8)
    rng = np.random.default_rng(5)
    mods = [s ** 0.5 * rng.lognormal(0, 0.4, 200) for s in scales]
    q, g, dl, h = np.linspace(-2, 2, 5), np.array([0.0, 2.0]), np.array([0.0, 0.5, 1.0]), np.linspace(0, 1, 11)
    full = gaussian_ensemble(mods, scales, q, g, 2, 6, length_px=1024., delta_list=dl)
    jack = scale_jackknife(full, h)
    inside = np.flatnonzero(full["window"])
    assert jack["D"].shape == (inside.size, h.size) and inside.size == 5
    for row, k in zip(jack["D"], inside):
        keep = [j for j in inside if j != k]
        sub = gaussian_ensemble([mods[j] for j in keep], scales[keep], q, g, 2, 6, length_px=1024.,
                                delta_list=dl, reference_amplitude=full["reference_amplitude"])
        np.testing.assert_allclose(row, generalized_legendre(sub["tau"], q, g, h, delta_list=dl)["D"], atol=1e-9)
    np.testing.assert_allclose(jack["lo"], jack["D"].min(axis=0))
    np.testing.assert_allclose(jack["half_width"], 0.5 * (jack["D"].max(axis=0) - jack["D"].min(axis=0)))
    np.testing.assert_allclose(np.log2(jack["left_out"]), np.log2(scales[inside]))
    stop = iter([False, True])
    with pytest.raises(InterruptedError):
        scale_jackknife(full, h, cancel=lambda: next(stop))
