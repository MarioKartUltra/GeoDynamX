# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.spectra -- the numpy-only reimplementation of the wtmm_ebsd
multifractal-spectrum scale-window fitter.

Reference semantics: ``wtmm_ebsd.partition.fit_hq_Dq_weighted`` (the oracle -- installed here,
cross-checked at the bottom of this file) with one signature change: the window is given
in log2 units (``log2_a_min``/``log2_a_max`` against ``hd['log2_scales']``), matching the
interactive fitter's draggable-window knob, instead of the oracle's linear ``scale_range``.

The module must NEVER import wtmm_ebsd (core stays free of the optional backend) -- pinned below,
exactly as tests/test_chain_stats.py pins it for chain_stats.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import spectra


def _hd(slopes_tau, slopes_h, slopes_D, sigma=0.1, n_sc=6, intercepts=(1.0, -0.5, 2.0)):
    """Hand-built hd dict whose tau/h/D tables are exact lines in log2(a).

    One row per q; ``slope_*[qi]`` is the exact log-log slope the fitter must recover.
    ``sigma`` fills all three sigma tables uniformly (scalar or (n_q, n_sc) array).
    """
    q = np.asarray([-1.0, 0.0, 2.0][: len(slopes_tau)], dtype=np.float64)
    log2_s = np.arange(n_sc, dtype=np.float64)          # scales 1, 2, 4, ...
    scales = 2.0 ** log2_s

    def table(slopes, c):
        return np.asarray(slopes, dtype=np.float64)[:, None] * log2_s[None, :] + c

    sig = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (len(q), n_sc)).copy()
    return {
        "q_list": q, "scales": scales, "log2_scales": log2_s,
        "N_a": np.full(n_sc, 100, dtype=np.int64),
        "tau_qa": table(slopes_tau, intercepts[0]),
        "h_qa": table(slopes_h, intercepts[1]),
        "D_qa": table(slopes_D, intercepts[2]),
        "sigma_tau_qa": sig.copy(), "sigma_h_qa": sig.copy(), "sigma_D_qa": sig.copy(),
    }


# --------------------------------------------------------------------------- module hygiene

def test_spectra_never_imports_wtmm_ebsd():
    """core/ stays free of the optional backend -- the module reimplements the fitter precisely so
    it works WITHOUT wtmm_ebsd installed. Import STATEMENTS only; comments may name it."""
    with open(spectra.__file__) as f:
        lines = f.readlines()
    import_lines = [ln for ln in lines
                    if ln.strip().startswith(("import wtmm_ebsd", "from wtmm_ebsd",
                                              "import dynamix._vendor.wtmm_ebsd",
                                              "from dynamix._vendor.wtmm_ebsd"))]
    assert import_lines == []


# --------------------------------------------------------------------------- slope recovery

def test_exact_slopes_recovered_from_linear_tables():
    """tau/h/D tables that are exact lines in log2(a) -> the fitted slopes ARE those lines'
    slopes, per q, when the window spans the whole scale axis."""
    hd = _hd(slopes_tau=[-1.0, 0.0, 3.0], slopes_h=[0.3, 0.5, 0.7], slopes_D=[2.0, 2.0, 1.4])
    out = spectra.fit_spectra(hd, -0.5, 5.5)
    np.testing.assert_allclose(out["tau"], [-1.0, 0.0, 3.0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out["h"], [0.3, 0.5, 0.7], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out["D"], [2.0, 2.0, 1.4], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(out["q_list"], hd["q_list"])


# --------------------------------------------------------------------------- error propagation

def test_error_bars_propagate_sigma_through_the_ols_slope():
    """Var(slope) = sum((x_i - xbar)^2 sigma_i^2) / (sum((x_i - xbar)^2))^2 -- with xbar and the
    denominator taken over the FULL window (reference convention), so constant sigma gives the
    closed form sigma / sqrt(denom)."""
    sigma = 0.25
    hd = _hd([1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [2.0, 2.0, 2.0], sigma=sigma)
    out = spectra.fit_spectra(hd, -0.5, 5.5)
    x = hd["log2_scales"]
    denom = float(((x - x.mean()) ** 2).sum())
    expect = sigma / np.sqrt(denom)
    for key in ("tau_err", "h_err", "D_err"):
        np.testing.assert_allclose(out[key], expect, rtol=1e-12, err_msg=key)


# --------------------------------------------------------------------------- eta / frame

def test_frame_original_shifts_tau_and_h_back_by_eta():
    """Fractional-integration undo (Wendt 2009): frame='original', eta != 0 subtracts eta*q from
    tau and eta from h; D and ALL error bars are frame-invariant; frame='integrated' returns the
    as-computed spectra untouched. The result records eta and frame."""
    hd = _hd([-1.0, 0.0, 3.0], [0.3, 0.5, 0.7], [2.0, 2.0, 1.4])
    eta = 1.0
    integ = spectra.fit_spectra(hd, -0.5, 5.5, eta=eta, frame="integrated")
    orig = spectra.fit_spectra(hd, -0.5, 5.5, eta=eta, frame="original")
    q = hd["q_list"]
    np.testing.assert_allclose(orig["tau"], integ["tau"] - eta * q, rtol=1e-12)
    np.testing.assert_allclose(orig["h"], integ["h"] - eta, rtol=1e-12)
    np.testing.assert_allclose(orig["D"], integ["D"], rtol=1e-12)
    for key in ("tau_err", "h_err", "D_err"):
        np.testing.assert_allclose(orig[key], integ[key], rtol=1e-12, err_msg=key)
    assert integ["eta"] == eta and integ["frame"] == "integrated"
    assert orig["eta"] == eta and orig["frame"] == "original"
    # default frame is 'original' but eta defaults to 0 -> no shift
    default = spectra.fit_spectra(hd, -0.5, 5.5)
    np.testing.assert_allclose(default["tau"], integ["tau"], rtol=1e-12)
    assert default["eta"] == 0.0 and default["frame"] == "original"


def test_invalid_frame_raises_value_error():
    hd = _hd([1.0], [0.5], [2.0])
    with pytest.raises(ValueError, match="frame"):
        spectra.fit_spectra(hd, -0.5, 5.5, frame="raw")


# --------------------------------------------------------------------------- degenerate windows

def _assert_all_nan_result(out, n_q):
    for key in ("tau", "h", "D", "tau_err", "h_err", "D_err"):
        assert out[key].shape == (n_q,)
        assert np.isnan(out[key]).all(), key


def test_window_with_fewer_than_three_scales_returns_all_nan():
    """A slope needs >= 3 points (reference gate). 2-scale and 0-scale windows return the full
    key set, every spectrum value NaN, no warnings/crash."""
    hd = _hd([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], [2.0, 2.0, 2.0])
    with np.errstate(all="raise"):
        _assert_all_nan_result(spectra.fit_spectra(hd, 0.5, 2.5), 3)   # scales 2 and 4 only
        _assert_all_nan_result(spectra.fit_spectra(hd, 10.0, 12.0), 3)  # empty window


def test_zero_scale_spread_returns_all_nan():
    """>= 3 in-window scales but all at the SAME log2 value -> denominator 0 -> all NaN."""
    hd = _hd([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], [2.0, 2.0, 2.0])
    hd["log2_scales"] = np.zeros_like(hd["log2_scales"])
    hd["scales"] = np.ones_like(hd["scales"])
    with np.errstate(all="raise"):
        _assert_all_nan_result(spectra.fit_spectra(hd, -0.5, 0.5), 3)


def test_per_q_rows_with_too_few_finite_points_are_nan_only_for_that_q():
    """NaNs poison one q row (below 3 finite in-window values); other rows still fit. Sigma rows
    gate independently: non-positive sigmas don't count as usable."""
    hd = _hd([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], [2.0, 2.0, 2.0])
    hd["tau_qa"][1, :4] = np.nan                # 2 finite left in the full window
    hd["sigma_h_qa"][2, :] = -1.0               # positive-only gate -> row unusable
    out = spectra.fit_spectra(hd, -0.5, 5.5)
    assert np.isnan(out["tau"][1])
    np.testing.assert_allclose(out["tau"][[0, 2]], [1.0, 3.0], rtol=1e-12)
    np.testing.assert_allclose(out["h"], 0.5, rtol=1e-12)     # h table untouched
    assert np.isnan(out["h_err"][2]) and np.isfinite(out["h_err"][0])


def test_window_actually_restricts_the_fit():
    """A tau row that is piecewise-linear (slope 1 on the first half, slope 5 on the second)
    fits to ~1 when the window covers the first half only."""
    hd = _hd([1.0], [0.5], [2.0], n_sc=8)
    row = hd["tau_qa"][0].copy()
    row[4:] = row[3] + 5.0 * (hd["log2_scales"][4:] - hd["log2_scales"][3])
    hd["tau_qa"][0] = row
    out = spectra.fit_spectra(hd, -0.5, 3.5)
    np.testing.assert_allclose(out["tau"], [1.0], rtol=1e-12)
    out_late = spectra.fit_spectra(hd, 3.5, 7.5)
    np.testing.assert_allclose(out_late["tau"], [5.0], rtol=1e-12)


# --------------------------------------------------------------------------- method dispatch

def test_method_microcanonical_and_legendre_are_reserved_not_silent():
    """Both reserved names raise NotImplementedError -- never a silent fallback to canonical;
    unknown method names are a ValueError instead. Terminology: what we run
    is the CANONICAL (Arneodo WTMM) formalism; 'legendre' is its unimplemented Legendre-transform
    estimator; 'microcanonical' properly names Turiel's local-exponent formalism (future MSC
    work, not a method of this fitter) and is reserved only because the reference conflates
    the two."""
    hd = _hd([1.0], [0.5], [2.0])
    for m in ("microcanonical", "legendre"):
        with pytest.raises(NotImplementedError):
            spectra.fit_spectra(hd, -0.5, 5.5, method=m)
    with pytest.raises(ValueError, match="method"):
        spectra.fit_spectra(hd, -0.5, 5.5, method="thermodynamic")
    out = spectra.fit_spectra(hd, -0.5, 5.5, method="canonical")
    np.testing.assert_allclose(out["tau"], [1.0], rtol=1e-12)


# --------------------------------------------------------------------------- oracle cross-check

def _synthetic_chains(rng, n_chains=400, n_sc=12):
    """Chains over a 12-scale, 3-voice grid with the pathologies the real pipeline produces:
    varying vertical lengths (incl. sub-min_chain_len stubs), zero/negative mods (-> NaN in the
    padded matrix), and a spread of Hölder slopes so high-|q| moments are inequitably shared."""
    chains = []
    log2_s = np.arange(n_sc) / 3.0
    scales = 2.0 ** log2_s
    for _ in range(n_chains):
        k = int(rng.integers(1, n_sc + 1))
        h = rng.uniform(-0.2, 1.2)
        amp = rng.lognormal(0.0, 1.0)
        mod = amp * scales[:k] ** h * rng.lognormal(0.0, 0.15, k)
        if rng.random() < 0.08:
            mod[rng.integers(0, k)] = 0.0          # dead sample -> NaN in the matrix
        chains.append({"mod": mod})
    return chains, scales


def test_fit_spectra_matches_the_wtmm_ebsd_oracle_to_float_tolerance():
    """The reimplementation IS fit_hq_Dq_weighted: same slopes, same error bars,
    same NaN pattern, on both hd conventions (std/nosup + cmax/sup), across eta/frame combos and
    scale windows -- with the log2-window bounds mapped to the oracle's linear scale_range."""
    partition = pytest.importorskip("dynamix._vendor.wtmm_ebsd.partition")
    chains, scales = _synthetic_chains(np.random.default_rng(42))
    q_list = np.arange(-3.0, 6.0 + 1e-9, 0.5)
    hd_std, hd_cmax = partition.build_hd_from_chains(chains, scales, q_list)

    windows = [(-0.5, 4.0), (0.4, 3.2), (1.1, 2.6)]   # full, inner, narrow (bounds off-grid)
    for hd in (hd_std, hd_cmax):
        for lo, hi in windows:
            for eta, frame in ((0.0, "original"), (1.0, "original"), (1.0, "integrated")):
                got = spectra.fit_spectra(hd, lo, hi, eta=eta, frame=frame)
                ref = partition.fit_hq_Dq_weighted(hd, (2.0 ** lo, 2.0 ** hi),
                                                   eta=eta, frame=frame)
                for key in ("tau", "h", "D", "tau_err", "h_err", "D_err"):
                    np.testing.assert_allclose(
                        got[key], ref[key], rtol=1e-10, atol=1e-12, equal_nan=True,
                        err_msg=f"{key} window=({lo},{hi}) eta={eta} frame={frame}")
                np.testing.assert_array_equal(got["q_list"], ref["q_list"])


def test_fit_spectra_matches_the_oracle_on_nan_poisoned_tables():
    """The synthetic ensemble keeps every scale alive, so the plain cross-check never hits the
    partial-NaN paths (per-q finite-subset slope; the full-window xbar/denom sigma convention on
    rows with unusable entries). Poison the tables directly and re-compare -- both fitters see
    the SAME poisoned dict, so any divergence is a masking-math bug in the reimplementation."""
    partition = pytest.importorskip("dynamix._vendor.wtmm_ebsd.partition")
    chains, scales = _synthetic_chains(np.random.default_rng(7))
    q_list = np.arange(-3.0, 6.0 + 1e-9, 0.5)
    hd_std, _ = partition.build_hd_from_chains(chains, scales, q_list)

    rng = np.random.default_rng(0)
    hd = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in hd_std.items()}
    n_q, n_sc = hd["tau_qa"].shape
    for key in ("tau_qa", "h_qa", "D_qa"):
        hd[key][rng.random((n_q, n_sc)) < 0.25] = np.nan       # rows drop to 2-3 finite points
    for key in ("sigma_tau_qa", "sigma_h_qa", "sigma_D_qa"):
        hd[key][rng.random((n_q, n_sc)) < 0.2] = np.nan
        hd[key][rng.random((n_q, n_sc)) < 0.1] = 0.0           # the positive-only gate
    hd["tau_qa"][0, :] = np.nan                                 # a fully dead q row

    for lo, hi in ((-0.5, 4.0), (0.7, 2.9)):
        got = spectra.fit_spectra(hd, lo, hi)
        ref = partition.fit_hq_Dq_weighted(hd, (2.0 ** lo, 2.0 ** hi))
        for key in ("tau", "h", "D", "tau_err", "h_err", "D_err"):
            np.testing.assert_allclose(got[key], ref[key], rtol=1e-10, atol=1e-12,
                                       equal_nan=True, err_msg=f"{key} window=({lo},{hi})")


# --------------------------------------------------- Legendre-Fenchel hull (2026-09-20)


def test_legendre_hull_of_a_quadratic_tau_is_the_exact_parabola():
    """Log-normal closed form: tau(q) = -c0 + c1 q - (c2/2) q^2 has D(h) = c0 - (c1-h)^2/(2 c2),
    a parabola peaked at (c1, c0). The discrete Legendre-Fenchel transform must reproduce it on
    its h grid (exactly, up to the q-grid discretization of the inf)."""
    c0, c1, c2 = 2.0, 0.6, 0.15
    q = np.arange(-4.0, 4.01, 0.05)
    tau = -c0 + c1 * q - 0.5 * c2 * q * q
    out = spectra.legendre_dh(q, tau)
    h, D = out["h"], out["D"]
    expect = c0 - (c1 - h) ** 2 / (2.0 * c2)
    np.testing.assert_allclose(D, expect, atol=2e-4)
    # peak at (c1, c0)
    assert abs(h[np.nanargmax(D)] - c1) < 0.05 and abs(np.nanmax(D) - c0) < 1e-3


def test_legendre_hull_of_a_monofractal_tau_is_a_single_point():
    q = np.arange(-3.0, 3.01, 0.5)
    tau = 0.7 * q - 2.0                    # tau = qH - d: single (h, D) = (0.7, 2.0)
    out = spectra.legendre_dh(q, tau)
    np.testing.assert_allclose(out["h"], [0.7], atol=1e-9)
    np.testing.assert_allclose(out["D"], [2.0], atol=1e-9)


def test_legendre_hull_of_a_kinked_tau_is_the_chord():
    """Two lines meeting concavely at q* = 1: slopes 1.5 then 0.5. The transform gives the HULL
    -- between the endpoint exponents D(h) is the straight chord through the kink value, which
    is exactly the Touchette-Beck Thm 2-4 ambiguity (nonconcave OR affine, indistinguishable
    from tau alone). This test pins that the hull, not an invented interior, comes back."""
    q = np.arange(-2.0, 4.01, 0.25)
    q_star, s_lo, s_hi = 1.0, 1.5, 0.5
    tau = np.where(q <= q_star, s_lo * q - 1.0,
                   s_lo * q_star - 1.0 + s_hi * (q - q_star))
    out = spectra.legendre_dh(q, tau)
    h, D = out["h"], out["D"]
    interior = (h > s_hi + 0.05) & (h < s_lo - 0.05)
    # chord through the kink: D(h) = q* h - tau(q*)
    np.testing.assert_allclose(D[interior], q_star * h[interior] - (s_lo * q_star - 1.0),
                               atol=1e-9)


def test_legendre_hull_ignores_nan_tau_rows():
    q = np.arange(-2.0, 2.01, 0.25)
    tau = 0.5 * q - 1.0
    tau[0] = np.nan
    out = spectra.legendre_dh(q, tau)
    assert np.isfinite(out["h"]).all() and np.isfinite(out["D"]).all()


# --------------------------------------------- phase transition in tau(q) (cell 49 port)


def _kinked_lnq_tau(q, q_star=2.0, s_lo=2.0, s_hi=5.0):
    """tau linear in ln(q) with a slope break at q* -- the notebook's constant-specific-heat
    model ('tau(q) is linear in ln(q) within each phase')."""
    lnq = np.log(q)
    ln_star = np.log(q_star)
    return np.where(lnq <= ln_star, s_lo * lnq, s_lo * ln_star + s_hi * (lnq - ln_star))


def test_phase_transition_fit_recovers_both_slopes_exactly():
    q = np.geomspace(0.25, 16.0, 25)
    tau = _kinked_lnq_tau(q)
    out = spectra.phase_transition_fit(q, tau, q_break=2.0)
    np.testing.assert_allclose(out["slope_L"], 2.0, atol=1e-9)
    np.testing.assert_allclose(out["slope_R"], 5.0, atol=1e-9)
    np.testing.assert_allclose(out["ds"], -3.0, atol=1e-9)
    assert out["r2_L"] > 0.999999 and out["r2_R"] > 0.999999


def test_phase_transition_fit_uses_only_the_positive_branch():
    q = np.concatenate([np.arange(-3.0, 0.0, 0.5), np.geomspace(0.25, 16.0, 25)])
    tau = np.where(q > 0, _kinked_lnq_tau(np.maximum(q, 1e-9)), 99.0)   # garbage at q <= 0
    out = spectra.phase_transition_fit(q, tau, q_break=2.0)
    np.testing.assert_allclose(out["slope_L"], 2.0, atol=1e-9)


def test_phase_transition_fit_nans_a_side_that_is_too_short():
    q = np.geomspace(0.25, 16.0, 25)
    tau = _kinked_lnq_tau(q)
    out = spectra.phase_transition_fit(q, tau, q_break=q.max() + 1.0)
    assert np.isnan(out["slope_R"]) and np.isnan(out["ds"])
    assert np.isfinite(out["slope_L"])


def test_phase_transition_scan_finds_the_break():
    q = np.geomspace(0.25, 16.0, 40)
    tau = _kinked_lnq_tau(q, q_star=2.0)
    out = spectra.phase_transition_scan(q, tau)
    assert abs(out["best_q"] - 2.0) < 0.4
    assert out["q_grid"].size == out["r2w"].size > 10


# --------------------------------- negative-branch convexity loss (Pont 2006 truncation)


def test_convexity_loss_flags_the_negative_branch_inflection():
    """Pont/Turiel/Perez-Vicente 2006: experimental tau loses convexity (in OUR concave-tau
    convention: concavity) at large negative moments -- truncate at the inflection. A tau
    concave on q > q_i and convex below it must flag near q_i; a globally concave tau must
    return None."""
    q = np.arange(-5.0, 5.01, 0.25)
    clean = 0.6 * q - 0.1 * q * q          # concave everywhere
    assert spectra.negative_branch_convexity_loss(q, clean) is None
    broken = clean.copy()
    bad = q < -3.0
    # replace the tail with a CONVEX continuation (curvature flips sign at q = -3)
    broken[bad] = clean[q == -3.0][0] + (0.6 - 0.1 * -6.0) * (q[bad] + 3.0) \
        + 0.3 * (q[bad] + 3.0) ** 2
    q_loss = spectra.negative_branch_convexity_loss(q, broken)
    assert q_loss is not None and -3.6 < q_loss < -2.4


# ------------------------------------------------------------------ focus regression modes

def _focal_hd(x0=9.0, y0=-3.0, n_sc=8, log2_N=6.0):
    """hd whose per-q power-mean columns (tau - log2 N)/q form an EXACT cone at (x0, y0):
    tau(q, a) = q*(y0 + (log2 a - x0) h_q) + log2_N, constant N_a = 2**log2_N."""
    q = np.asarray([-3.0, -1.0, 0.0, 1.0, 2.0, 4.0], dtype=np.float64)
    log2_s = np.arange(n_sc, dtype=np.float64)
    h_q = np.linspace(1.3, 0.5, q.size)
    tau = np.empty((q.size, n_sc))
    for i, qq in enumerate(q):
        tau[i] = qq * (y0 + (log2_s - x0) * h_q[i]) + log2_N
    sig = np.full((q.size, n_sc), 0.1)
    return {
        "q_list": q, "scales": 2.0 ** log2_s, "log2_scales": log2_s,
        "N_a": np.full(n_sc, int(2 ** log2_N), dtype=np.int64),
        "tau_qa": tau, "h_qa": tau.copy(), "D_qa": tau.copy(),
        "sigma_tau_qa": sig, "sigma_h_qa": sig.copy(), "sigma_D_qa": sig.copy(),
    }, h_q, q


def test_focus_mode_recovers_tau_from_the_exact_cone():
    hd, h_q, q = _focal_hd(x0=9.0, log2_N=6.0)
    out = spectra.fit_spectra(hd, 0.0, 7.0, fit_mode="focus", log2_L=8.0)
    f = out["focus"]
    assert f["branch"] == "interior"
    assert f["x0"] == pytest.approx(9.0, abs=1e-8)
    # tau(q) = q*h_q + slope(log2 N) = q*h_q + 0 for constant N; tau(0) = 0 here
    nz = q != 0
    np.testing.assert_allclose(out["tau"][nz], q[nz] * h_q[nz], atol=1e-8)
    assert out["tau"][~nz][0] == pytest.approx(0.0, abs=1e-12)
    assert f["mse_q"].shape == (int(nz.sum()),)


def test_focus_mode_requires_the_anchor():
    hd, _h, _q = _focal_hd()
    with pytest.raises(ValueError, match="log2_L"):
        spectra.fit_spectra(hd, 0.0, 7.0, fit_mode="focus")


def test_fixed_focus_mode_never_uses_an_interior_root():
    hd, _h, _q = _focal_hd(x0=9.0)
    out = spectra.fit_spectra(hd, 0.0, 7.0, fit_mode="fixed_focus", log2_L=8.0)
    assert out["focus"]["branch"] in ("fixed", "mono")


def test_naive_mode_is_unchanged_and_carries_no_focus_key():
    hd, _h, _q = _focal_hd()
    out = spectra.fit_spectra(hd, 0.0, 7.0)
    assert "focus" not in out


def test_focus_falls_back_gracefully_on_a_tiny_window():
    hd, _h, _q = _focal_hd()
    out = spectra.fit_spectra(hd, 0.0, 3.0, fit_mode="focus", log2_L=8.0)  # 4 scales < 5
    assert out["focus"]["branch"] == "unavailable"
    assert np.all(np.isfinite(out["tau"]))          # the naive tau still stands
