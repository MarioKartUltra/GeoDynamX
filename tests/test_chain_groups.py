# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.core.chain_groups -- the EBSD-workbook chain grouping port (cell 52). Headless, numpy-only except the
subset-table oracle, which needs the installed wtmm_ebsd (skipped when absent, the
tests/test_spectra.py oracle pattern)."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import chain_groups as cg


def _chain(mods, scales):
    mod = np.asarray(mods, dtype=np.float64)
    k = mod.size
    with np.errstate(divide="ignore"):
        log2_mod = np.log2(np.abs(mod))
    return {"x": np.zeros(k, dtype=np.int64), "y": np.arange(k, dtype=np.int64),
            "mod": mod, "log2_mod": log2_mod,
            "log2_scales": np.log2(np.asarray(scales, dtype=np.float64))[:k]}


SCALES = np.array([1.0, 2.0, 4.0])


def _chains():
    """Three chains: one LOUD across all scales, one quiet full-length, one short."""
    return [
        _chain([8.0, 6.0, 9.0], SCALES),
        _chain([0.5, 0.4, 0.3], SCALES),
        _chain([2.0], SCALES),
    ]


# ---------------------------------------------------------------------------------- payload

def test_payload_matrices_and_lengths():
    chains = _chains()
    p = cg.grouping_payload(chains, len(SCALES))
    np.testing.assert_array_equal(p["chain_len"], [3, 3, 1])
    np.testing.assert_allclose(p["mod_matrix"][0], [8.0, 6.0, 9.0])
    np.testing.assert_allclose(p["mod_matrix"][2], [2.0, np.nan, np.nan])
    # chainmax = running sup finest -> coarse, NaN past the chain's own extent
    np.testing.assert_allclose(p["cmax_matrix"][0], [8.0, 8.0, 9.0])
    np.testing.assert_allclose(p["cmax_matrix"][2], [2.0, np.nan, np.nan])


def test_payload_is_memoized_by_identity():
    chains = _chains()
    assert cg.grouping_payload(chains, 3) is cg.grouping_payload(chains, 3)
    assert cg.grouping_payload(_chains(), 3) is not cg.grouping_payload(_chains(), 3)


# ---------------------------------------------------------------------------------- weights

def test_boltzmann_weights_are_a_softmax_of_q_log_mod():
    chains = _chains()
    p = cg.grouping_payload(chains, 3)
    w, alive = cg.boltzmann_weights(p, 0, 2.0)
    np.testing.assert_array_equal(alive, [0, 1, 2])
    expect = np.array([8.0, 0.5, 2.0]) ** 2.0
    np.testing.assert_allclose(w, expect / expect.sum(), rtol=1e-12)
    np.testing.assert_allclose(w.sum(), 1.0)


def test_weights_skip_scales_a_chain_never_reaches():
    p = cg.grouping_payload(_chains(), 3)
    w, alive = cg.boltzmann_weights(p, 2, 1.0)
    np.testing.assert_array_equal(alive, [0, 1])       # the short chain is gone at scale 2


# ----------------------------------------------------------------------------- classification

def test_positive_q_makes_the_loud_chain_dominant_and_negative_q_the_quiet_one():
    """The tilted measure's sign physics: q > 0 weights the strong moduli, q < 0 the weak."""
    p = cg.grouping_payload(_chains(), 3)
    dom_pos = cg.classify_chains(p, scale_idx=0, q=3.0, dom_percentile=40.0)
    dom_neg = cg.classify_chains(p, scale_idx=0, q=-3.0, dom_percentile=40.0)
    assert dom_pos[0] and not dom_pos[1]
    assert dom_neg[1] and not dom_neg[0]


def test_min_len_excludes_short_chains_from_dominance():
    p = cg.grouping_payload(_chains(), 3)
    dom = cg.classify_chains(p, scale_idx=0, q=0.5, dom_percentile=100.0, min_len=2)
    assert not dom[2]
    assert dom[0] and dom[1]                            # p=100: every alive chain qualifies


def test_log_threshold_modes_gate_on_q_ln_mod():
    p = cg.grouping_payload(_chains(), 3)
    # q=1: dominance needs ln|T| >= ln 4 -- only the loud chain's 8.0 passes at scale 0
    dom = cg.classify_chains(p, scale_idx=0, q=1.0, mode="mq_at_scale",
                             log_thresh=float(np.log(4.0)))
    np.testing.assert_array_equal(dom, [True, False, False])
    # sup mode reads the chainmax column instead
    dom_sup = cg.classify_chains(p, scale_idx=1, q=1.0, mode="mq_sup",
                                 log_thresh=float(np.log(7.0)))
    np.testing.assert_array_equal(dom_sup, [True, False, False])


def test_unknown_mode_is_refused():
    p = cg.grouping_payload(_chains(), 3)
    with pytest.raises(ValueError):
        cg.classify_chains(p, scale_idx=0, q=1.0, mode="lasso")


# ------------------------------------------------------------------------------ subset tables

def test_subset_hd_all_true_equals_the_full_partition_build():
    wtmm_ebsd = pytest.importorskip("dynamix._vendor.wtmm_ebsd")
    from dynamix._vendor.wtmm_ebsd.partition import build_hd_from_chains

    chains = _chains()
    q_list = np.arange(-2.0, 2.1, 0.5)
    ours_std, ours_cmax = cg.subset_hd(chains, SCALES, q_list,
                                       np.ones(len(chains), dtype=bool))
    ref_std, ref_cmax = build_hd_from_chains(chains, SCALES, q_list, min_chain_len=2)
    for ours, ref in ((ours_std, ref_std), (ours_cmax, ref_cmax)):
        np.testing.assert_allclose(ours["tau_qa"], ref["tau_qa"], equal_nan=True)
        np.testing.assert_allclose(ours["h_qa"], ref["h_qa"], equal_nan=True)


def test_subset_hd_is_memoized_per_membership():
    pytest.importorskip("dynamix._vendor.wtmm_ebsd")
    chains = _chains()
    q_list = np.arange(-1.0, 1.1, 0.5)
    mask = np.array([True, False, True])
    first = cg.subset_hd(chains, SCALES, q_list, mask)
    again = cg.subset_hd(chains, SCALES, q_list, mask.copy())
    assert first[0] is again[0] and first[1] is again[1]
    other = cg.subset_hd(chains, SCALES, q_list, np.array([False, True, True]))
    assert other[0] is not first[0]
