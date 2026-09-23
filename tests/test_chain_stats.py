# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.chain_stats -- the numpy-only reimplementation of the wtmm_ebsd Hölder/
modulus/length estimators.

Reference semantics: ``wtmm_ebsd``'s own estimators. This module must
NEVER import wtmm_ebsd (core stays Qt-free AND free of the optional-dependency backend) -- the
first test below pins that.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import chain_stats


def _chain(log2_scales, log2_mod):
    return {"log2_scales": np.asarray(log2_scales, dtype=np.float64),
           "log2_mod": np.asarray(log2_mod, dtype=np.float64)}


# --------------------------------------------------------------------------- module hygiene

def test_chain_stats_never_imports_wtmm_ebsd():
    """core/ stays free of the optional backend -- a bug if this module ever imports it, even
    lazily, since its whole reason to exist is to work WITHOUT wtmm_ebsd installed. Checks for an
    actual import STATEMENT, not the bare substring -- the module's own docstring/comments
    legitimately name ``wtmm_ebsd`` throughout, explaining exactly why it is never imported."""
    with open(chain_stats.__file__) as f:
        lines = f.readlines()
    import_lines = [ln for ln in lines
                    if ln.strip().startswith(("import wtmm_ebsd", "from wtmm_ebsd",
                                              "import dynamix._vendor.wtmm_ebsd",
                                              "from dynamix._vendor.wtmm_ebsd"))]
    assert import_lines == []


# --------------------------------------------------------------------------- per_chain_stats

def test_per_chain_stats_all_nan_below_three_jointly_finite_points():
    ch = _chain([0.0, 1.0], [0.0, 1.0])       # n=2 raw, joint-finite=2 -- below the n<3 gate
    h, r2, mx, mn, n = chain_stats.per_chain_stats(ch)
    assert np.isnan(h) and np.isnan(r2) and np.isnan(mx) and np.isnan(mn)
    assert n == 2


def test_per_chain_stats_n_counts_jointly_finite_not_raw_length():
    """A deliberate decision (module docstring): n is the JOINT-finite count, not
    len(log2_scales) -- a chain with 4 raw points but only 2 finite pairs must gate as n=2, not
    n=4, even though the reference notebook's own len()-based n would have said 4 and NOT gated."""
    ch = _chain([0.0, 1.0, 2.0, 3.0], [0.0, np.nan, 2.0, np.nan])
    h, r2, mx, mn, n = chain_stats.per_chain_stats(ch)
    assert n == 2
    assert np.isnan(h)          # gated: 2 < 3


def test_per_chain_stats_matches_manual_ols_and_r2_above_the_gate():
    log2_s = np.array([0.0, 1.0, 2.0, 3.0])
    log2_m = np.array([0.0, 1.9, 4.2, 5.8])       # roughly slope-2, not exact -- exercises R² < 1
    ch = _chain(log2_s, log2_m)
    h, r2, mx, mn, n = chain_stats.per_chain_stats(ch)
    assert n == 4
    expected_slope = np.polyfit(log2_s, log2_m, 1)[0]
    assert h == pytest.approx(expected_slope)
    assert 0.0 < r2 < 1.0
    # local slopes: diff(m)/diff(s) = [1.9, 2.3, 1.6]
    assert mx == pytest.approx(2.3)
    assert mn == pytest.approx(1.6)


def test_per_chain_stats_r2_is_one_for_a_perfect_line():
    log2_s = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    log2_m = -1.0 * log2_s + 3.0
    ch = _chain(log2_s, log2_m)
    h, r2, mx, mn, n = chain_stats.per_chain_stats(ch)
    assert h == pytest.approx(-1.0)
    assert r2 == pytest.approx(1.0)
    assert mx == pytest.approx(-1.0)
    assert mn == pytest.approx(-1.0)


def test_per_chain_stats_missing_keys_all_nan_with_n_zero():
    h, r2, mx, mn, n = chain_stats.per_chain_stats({})
    assert np.isnan(h) and np.isnan(r2) and np.isnan(mx) and np.isnan(mn)
    assert n == 0


# --------------------------------------------------------------------------- stats_for

def test_stats_for_ols_matches_polyfit_on_a_synthetic_chain():
    log2_s = np.array([0.0, 1.0, 2.0, 3.0])
    log2_m = np.array([0.0, 1.9, 4.2, 5.8])
    ch = _chain(log2_s, log2_m)
    expected = np.polyfit(log2_s, log2_m, 1)[0]
    got = chain_stats.stats_for([ch], "ols")
    assert got.shape == (1,)
    assert got[0] == pytest.approx(expected)


def test_stats_for_ols_nan_below_two_jointly_finite_points_the_looser_gate():
    """stats_for uses the wtmm_ebsd chain_ols_holder gate (n<2), NOT per_chain_stats's n<3 -- a
    2-point chain is a valid OLS estimate here even though per_chain_stats would refuse it."""
    ch = _chain([0.0, 1.0], [0.0, 2.0])
    got = chain_stats.stats_for([ch], "ols")
    assert got[0] == pytest.approx(2.0)          # slope of the only two points
    one_point = _chain([0.0], [0.0])
    assert np.isnan(chain_stats.stats_for([one_point], "ols")[0])


def test_stats_for_max_is_the_largest_local_slope():
    log2_s = np.array([0.0, 1.0, 2.0])
    log2_m = np.array([0.0, 3.0, 1.0])            # local slopes: 3.0, -2.0
    ch = _chain(log2_s, log2_m)
    got = chain_stats.stats_for([ch], "max")
    assert got[0] == pytest.approx(3.0)


def test_stats_for_unknown_estimator_raises_value_error():
    with pytest.raises(ValueError):
        chain_stats.stats_for([_chain([0, 1], [0, 1])], "median")


def test_stats_for_vectorizes_over_multiple_chains():
    chains = [_chain([0.0, 1.0], [0.0, 1.0]), _chain([0.0, 1.0], [0.0, -1.0])]
    got = chain_stats.stats_for(chains, "ols")
    assert got.shape == (2,)
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(-1.0)


# --------------------------------------------------------------------------- max_log2_modulus_for / length_for

def test_max_log2_modulus_for_nan_when_missing():
    assert np.isnan(chain_stats.max_log2_modulus_for([{}])[0])


def test_max_log2_modulus_for_takes_the_finite_max():
    ch = {"log2_mod": np.array([1.0, np.nan, 3.0, -2.0])}
    assert chain_stats.max_log2_modulus_for([ch])[0] == pytest.approx(3.0)


def test_length_for_prefers_mod_then_log2_mod_then_log2_scales():
    assert chain_stats.length_for([{"mod": [1, 2, 3]}])[0] == 3
    assert chain_stats.length_for([{"log2_mod": [1, 2]}])[0] == 2
    assert chain_stats.length_for([{"log2_scales": [1]}])[0] == 1
    assert chain_stats.length_for([{}])[0] == 0


# --- vectorization equivalence + memo (2026-09-14 perf: Hölder-filter beach ball) --------------

def _chains_variety():
    """Chains covering the edge cases the vectorized path must match: empty, 1-point, clean,
    NaN-poisoned, and a repeated-scale (degenerate) chain."""
    import numpy as np
    rng = np.random.default_rng(7)
    out = []
    for k in (0, 1, 2, 3, 6, 10):
        s = np.sort(rng.uniform(0, 5, k)) if k else np.array([])
        m = rng.uniform(-3, 2, k) if k else np.array([])
        with np.errstate(divide="ignore"):
            out.append({"log2_scales": s, "log2_mod": m,
                        "mod": np.abs(rng.uniform(0.1, 4, k)) if k else np.array([])})
    bad = out[4]
    bad["log2_mod"] = bad["log2_mod"].copy(); bad["log2_mod"][0] = np.nan
    bad["log2_scales"] = bad["log2_scales"].copy(); bad["log2_scales"][2] = np.inf
    return out


def test_vectorized_stats_match_the_per_chain_reference():
    import numpy as np
    from dynamix.core import chain_stats as cs
    chains = _chains_variety()
    for est, ref_fn in (("ols", cs._chain_ols_holder), ("max", cs._chain_max_slope_holder)):
        got = cs.stats_for(chains, est)
        ref = np.array([ref_fn(c) for c in chains])
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10, equal_nan=True,
                                   err_msg=est)
    got_mod = cs.max_log2_modulus_for(chains)
    ref_mod = np.array([(lambda m: (m[np.isfinite(m)].max() if np.isfinite(m).any()
                                    else float("nan")))(np.asarray(c.get("log2_mod", ()), float))
                        for c in chains])
    np.testing.assert_allclose(got_mod, ref_mod, rtol=1e-10, atol=1e-10, equal_nan=True)


def test_stats_for_memoizes_on_the_same_chains_object():
    from dynamix.core import chain_stats as cs
    chains = _chains_variety()
    a = cs.stats_for(chains, "ols")
    b = cs.stats_for(chains, "ols")
    assert a is b                              # same object back -> the metric was not recomputed
    assert cs.stats_for(list(chains), "ols") is not a   # a different list recomputes
