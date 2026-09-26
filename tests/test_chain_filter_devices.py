# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for the chain-filter devices that wrap ``wtmm_ebsd.chain_filters``.

Three things these pin. First, that the devices degrade rather than raise when the optional
``wtmm_ebsd`` package is absent -- a filter that throws inside a redraw loop takes the window down.
Second, that chain filters and extrema filters are genuinely different operations: chains thread
ACROSS scales, which is where the Hölder exponent lives, while extrema exist AT one scale. Third: "degrade" no longer
means "pass through unchanged" when ``wtmm_ebsd`` is absent -- it means fall back to
``dynamix.core.chain_stats``, a numpy-only reimplementation, and filter for real. The old silent
pass-through read as "the filter is broken"; see ``chain_filters.py``'s own module docstring for
the full B2 root cause and fix.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import chain_stats
from dynamix.core.chain_product import materialize_selection
from dynamix.core.rasterfield import RasterField
from dynamix.devices.chain_filters import (ChainHolderFilter, ChainLengthFilter,
                                           ChainModulusFilter, _chain_filters)
from dynamix.devices.wtmm import WTMM2D
from dynamix.model.device import defaults_for

FIXTURE = "tests/fixtures/kam_64.npz"
needs_backing = pytest.mark.skipif(_chain_filters() is None,
                                   reason="wtmm_ebsd not installed (documented local install)")


@pytest.fixture(scope="module")
def stack():
    """``defaults_for``, not a hand-spelled dict -- see test_real_devices.py's own fixture, which
    documents why.

    ``fracint_alpha=0`` pinned because the lift applies on the scalar path:
    every triage number below was MEASURED on the unlifted pipeline -- under the lift each
    chain's OLS slope shifts by exactly +alpha and the similitude band re-links chains, which is
    real physics but not what these reading-format/triage regressions pin. The lift itself is
    covered by tests/test_fracint_scalar.py."""
    field = RasterField.load_npz(FIXTURE)
    return WTMM2D().compute(field, dict(defaults_for(WTMM2D()), fracint_alpha=0.0))


def _dict_path(stack):
    """The stack without its product/selection -- forces the legacy dict path, which is what
    the degraded-mode (wtmm_ebsd-absent) tests exist to pin. On a product-stamped result the
    selection path reads the metric columns and neither backend nor fallback ever runs."""
    return {k: v for k, v in stack.items() if k not in ("chain_product", "_selection")}


# --------------------------------------------------------------------- graceful degradation

@pytest.mark.parametrize("device,params", [
    (ChainHolderFilter(), {"cutoff": 0.0, "estimator": "ols"}),
    (ChainModulusFilter(), {"threshold": 0.0}),
    (ChainLengthFilter(), {"min_len": 4}),
])
def test_devices_pass_through_when_there_are_no_chains(device, params):
    """An empty or chain-free result must come back untouched, not raise. A filter that throws in
    a redraw loop takes the window down with it."""
    assert device.apply({}, params) == {}
    assert device.apply({"chains": []}, params) == {"chains": []}


# --------------------------------------------------------------------- real behaviour

@needs_backing
def test_holder_cutoff_monotonically_reduces_the_kept_set(stack):
    """Raising the Hölder floor can only ever remove chains, never add them."""
    counts = [len(materialize_selection(
                  ChainHolderFilter().apply(stack, {"cutoff": c, "estimator": "ols"}))["chains"])
              for c in (-2.0, 0.0, 0.5, 1.0)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == len(stack["chains"])       # a floor below every value keeps everything
    assert counts[-1] < counts[0]


@needs_backing
def test_the_two_holder_estimators_disagree(stack):
    """OLS slope and max local slope are different estimators, and the choice changes the science
    -- which is exactly why this wraps a tested implementation instead of reimplementing one."""
    ols = materialize_selection(
        ChainHolderFilter().apply(stack, {"cutoff": 0.0, "estimator": "ols"}))["chains"]
    mx = materialize_selection(
        ChainHolderFilter().apply(stack, {"cutoff": 0.0, "estimator": "max"}))["chains"]
    assert len(ols) != len(mx)


@needs_backing
def test_length_filter_keeps_only_chains_spanning_enough_scales(stack):
    kept = materialize_selection(ChainLengthFilter().apply(stack, {"min_len": 4}))["chains"]
    assert 0 < len(kept) < len(stack["chains"])
    for chain in kept:
        assert len(chain["log2_scales"]) >= 4


@needs_backing
def test_filters_report_how_many_they_dropped(stack):
    out = materialize_selection(ChainHolderFilter().apply(stack, {"cutoff": 0.0,
                                                                  "estimator": "ols"}))
    assert out["_chains_dropped"] == len(stack["chains"]) - len(out["chains"])


@needs_backing
def test_chain_filters_do_not_mutate_the_cached_stack(stack):
    before = len(stack["chains"])
    ChainHolderFilter().apply(stack, {"cutoff": 0.5, "estimator": "ols"})
    assert len(stack["chains"]) == before, "filter mutated the cached transform result"


@needs_backing
def test_chain_and_extrema_filters_act_on_different_objects(stack):
    """The reason both families exist. A chain filter must leave the extrema layers alone."""
    n_extrema_layers = len(stack["extrema"])
    out = ChainHolderFilter().apply(stack, {"cutoff": 0.5, "estimator": "ols"})
    assert len(out["extrema"]) == n_extrema_layers


# --------------------------------------------------------------------- Default,
# degraded-mode fallback, reading, data_hints


def test_holder_default_cutoff_is_the_hard_floor_not_the_old_no_op_value():
    """Bug triage B2: -2.0 sat below the kam_64 fixture's whole real Hölder range (-1.28..+1.00),
    so a freshly dropped filter did nothing. -3.0 is the new default -- still pass-all (below the
    (-3, 2) histogram range), but now unambiguously "off" rather than "almost off"."""
    cutoff = [p for p in ChainHolderFilter().params if p.name == "cutoff"][0]
    assert cutoff.default == -3.0


def test_holder_filter_falls_back_to_chain_stats_when_wtmm_ebsd_is_absent(monkeypatch, stack):
    """The B2 fix: the fallback FILTERS FOR REAL, matching the real backend's own measured
    kept-counts from the triage (73 chains; cutoff -1.0 -> 64 kept) -- not the old silent
    pass-through (which this test would previously have measured as 73 kept at every cutoff)."""
    import dynamix.devices.chain_filters as cf_mod
    monkeypatch.setattr(cf_mod, "_chain_filters", lambda: None)
    bare = _dict_path(stack)

    out_off = ChainHolderFilter().apply(bare, {"cutoff": -3.0, "estimator": "ols"})
    assert len(out_off["chains"]) == len(stack["chains"]) == 73

    out = ChainHolderFilter().apply(bare, {"cutoff": -1.0, "estimator": "ols"})
    assert len(out["chains"]) == 64
    assert out["_chains_dropped"] == 73 - 64


def test_modulus_and_length_filters_also_fall_back_when_wtmm_ebsd_is_absent(monkeypatch, stack):
    import dynamix.devices.chain_filters as cf_mod
    monkeypatch.setattr(cf_mod, "_chain_filters", lambda: None)
    bare = _dict_path(stack)

    out = ChainModulusFilter().apply(bare, {"threshold": -99.0})
    assert len(out["chains"]) == len(stack["chains"])          # pass-all default, still honored

    lengths = chain_stats.length_for(stack["chains"])
    min_len = int(lengths.min()) + 1 if lengths.size else 2
    out = ChainLengthFilter().apply(bare, {"min_len": min_len})
    assert 0 < len(out["chains"]) < len(stack["chains"])
    for ch in out["chains"]:
        assert chain_stats.length_for([ch])[0] >= min_len


def test_holder_reading_matches_the_kam_64_triage_numbers(stack):
    """Exact string, 2 dp, from the measured fixture numbers (the design's pinned example)."""
    result = ChainHolderFilter().apply(stack, {"cutoff": -3.0, "estimator": "ols"})
    reading = ChainHolderFilter().reading(result, {"cutoff": -3.0, "estimator": "ols"})
    assert reading == "kept 73/73 · h ∈ [-1.28, 0.69]"


def test_holder_reading_reflects_a_real_cutoff_change(stack):
    result = materialize_selection(
        ChainHolderFilter().apply(stack, {"cutoff": -1.0, "estimator": "ols"}))
    reading = ChainHolderFilter().reading(result, {"cutoff": -1.0, "estimator": "ols"})
    assert reading.startswith("kept 64/73")


def test_holder_data_hints_cutoff_matches_the_computed_range(stack):
    result = ChainHolderFilter().apply(stack, {"cutoff": -3.0, "estimator": "ols"})
    hints = ChainHolderFilter().data_hints(result, {"cutoff": -3.0, "estimator": "ols"})
    lo, hi, snap = hints["cutoff"]
    vals = chain_stats.stats_for(stack["chains"], "ols")
    finite = vals[np.isfinite(vals)]
    assert lo == pytest.approx(float(finite.min()))
    assert hi == pytest.approx(float(finite.max()))
    assert snap == pytest.approx(lo)     # snapping preserves pass-all


def test_holder_data_hints_keys_are_a_subset_of_declared_params(stack):
    """Nothing a hint returns can leak a foreign param name into a caller that trusts it --
    display state stays scoped to params this device actually declares."""
    result = ChainHolderFilter().apply(stack, {"cutoff": -3.0, "estimator": "ols"})
    hints = ChainHolderFilter().data_hints(result, {"cutoff": -3.0, "estimator": "ols"})
    declared = {p.name for p in ChainHolderFilter().params}
    assert set(hints) <= declared


def test_holder_reading_and_hints_are_unavailable_without_log2_mod():
    """Chains present, but missing the key the Hölder estimate needs -- EQSelect's own
    "unavailable, don't apply, disable" degradation tier, not a crash and not a false
    kept-count."""
    chains = [{"x": np.array([0, 1]), "y": np.array([0, 1]),
              "log2_scales": np.array([0.0, 1.0])}]     # no "log2_mod" key at all
    result = {"chains": chains}
    reading = ChainHolderFilter().reading(result, {"cutoff": -3.0, "estimator": "ols"})
    assert reading == "unavailable — recompute the transform to enable"
    hints = ChainHolderFilter().data_hints(result, {"cutoff": -3.0, "estimator": "ols"})
    lo, hi, snap = hints["cutoff"]
    assert lo != lo and hi != hi and snap != snap     # NaN sentinel


def test_modulus_and_length_readings_are_present_and_kept_count_shaped(stack):
    """"Same additive treatment" for the other two chain filters -- just the shape,
    not a second full numeric pin."""
    out_mod = ChainModulusFilter().apply(stack, {"threshold": -99.0})
    reading = ChainModulusFilter().reading(out_mod, {"threshold": -99.0})
    assert reading.startswith("kept 73/73")
    hints = ChainModulusFilter().data_hints(out_mod, {"threshold": -99.0})
    assert "threshold" in hints

    out_len = ChainLengthFilter().apply(stack, {"min_len": 2})
    reading = ChainLengthFilter().reading(out_len, {"min_len": 2})
    assert reading.startswith(f"kept {len(out_len['chains'])}/{len(stack['chains'])}")
    hints = ChainLengthFilter().data_hints(out_len, {"min_len": 2})
    assert "min_len" in hints
