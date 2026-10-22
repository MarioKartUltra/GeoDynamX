# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.engine.cache."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.engine.cache import Cache


def test_cache_pin_unpin():
    c = Cache()
    c.put("k", 1)
    c.pin("k")
    assert c.is_pinned("k") and "k" in c.pinned
    c.unpin("k")
    assert not c.is_pinned("k")


def test_cache_pinned_property_is_frozenset():
    c = Cache()
    c.put("k1", 1)
    c.put("k2", 2)
    c.pin("k1")
    c.pin("k2")
    pinned = c.pinned
    assert isinstance(pinned, frozenset)
    assert pinned == frozenset(["k1", "k2"])


def test_cache_pin_nonexistent_key():
    c = Cache()
    c.pin("nonexistent")
    assert c.is_pinned("nonexistent")


def test_cache_unpin_nonexistent_key():
    c = Cache()
    c.unpin("nonexistent")
    assert not c.is_pinned("nonexistent")


# ------------------------------------------------- bounded LRU
# The noise device's seed/amplitude scrubs mint a ~134 MB field clone plus a full WTMM result PER
# STEP; an unbounded store retains them all -> memory pressure, beachball. Bounded LRU, pins
# exempt (the frozen-layer contract the docstring states).

def test_capacity_evicts_the_least_recently_used_unpinned_entry():
    c = Cache(maxsize=3)
    for k in ("a", "b", "c"):
        c.put(k, k.upper())
    assert c.get("a") == "A"                   # refresh a: b is now the LRU
    c.put("d", "D")
    assert "b" not in c and all(k in c for k in ("a", "c", "d"))


def test_get_or_compute_hit_refreshes_recency():
    c = Cache(maxsize=2)
    c.put("a", 1); c.put("b", 2)
    c.get_or_compute("a", lambda: 99)          # hit: refresh a
    c.put("c", 3)
    assert "b" not in c and "a" in c


def test_pinned_entries_survive_eviction_even_when_oldest():
    c = Cache(maxsize=2)
    c.put("old", 1); c.pin("old")
    c.put("b", 2); c.put("c", 3); c.put("d", 4)
    assert "old" in c and c.get("old") == 1
    assert len(c) <= 3                          # the pin rides above the bound; the rest is bounded


def test_default_capacity_is_finite_but_generous():
    c = Cache()
    assert 32 <= c.maxsize < 10_000


def test_replacing_a_cached_result_releases_its_landed_view_arrays():
    import weakref
    import numpy as np

    cache = Cache()
    values = np.ones((8, 8))
    tracked = weakref.ref(values)
    raw = {'h_map': values}
    cache.put('analysis', raw)
    view = cache.landed_result('analysis', raw,
                               {'analysis_key': 'analysis', 'filters': [], 'level': 0})
    del values, raw, view
    cache.put('analysis', {'h_map': None})
    assert tracked() is None


# ------------------------------------------------- retention of analysis results
# A transform knob mints a whole new lineage (a wtmm2d result, its chain topology) per value. The
# bounds keep a repeated-knob session at a bounded working set; what a layer shows, its upstream
# entries and the pins stay, and a result recomputed after eviction is the evicted one.

THRESHOLDS = [1e-3 * (i + 1) for i in range(10)]


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


@pytest.fixture
def fbm256():
    from conftest import fbm2d, make_field

    return make_field(fbm2d(256, 0.7, seed=0))


def _layer(thresh, layer_id=1):
    import dataclasses

    from conftest import make_layer

    layer = make_layer(("wtmm2d", {"n_oct": 2, "n_voice": 4, "thresh": thresh}),
                       ("chain_topology", {}))
    return dataclasses.replace(layer, layer_id=layer_id)


def _lineage(thresh) -> tuple[str, str]:
    """The wtmm2d key and the chain_topology key ``_layer(thresh)`` resolves under."""
    from dynamix.engine.resolve import source_identity, step_key
    from dynamix.model.device import get_device, validate_params

    layer = _layer(thresh)
    sid, upstream, keys = source_identity(layer), None, []
    for ref in layer.chain.steps:
        device = get_device(ref.device)
        upstream = step_key(device, sid, validate_params(device, ref.params), upstream)
        keys.append(upstream)
    return tuple(keys)


def _arrays(value) -> list:
    """Every ndarray inside ``value``, in a fixed order (dict keys sorted)."""
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, dict):
        return [a for k in sorted(value, key=str) for a in _arrays(value[k])]
    if isinstance(value, (list, tuple)):
        return [a for v in value for a in _arrays(v)]
    return []


@pytest.mark.parametrize("bound", ["results", "bytes"])
def test_a_repeated_knob_workload_stays_within_the_bounds(builtins, fbm256, bound):
    from dynamix.engine.resolve import resolve

    cache = Cache(max_results=8) if bound == "results" else Cache()
    resolve(_layer(THRESHOLDS[0]), fbm256, cache)
    if bound == "bytes":
        cache.max_bytes = int(2.5 * cache.report()["results"]["bytes"])
    for thresh in THRESHOLDS[1:]:
        resolve(_layer(thresh), fbm256, cache)
        held = cache.report()["results"]
        assert held["analysis"] <= (8 if bound == "results" else 64)
        assert held["bytes"] <= (cache.max_bytes or float("inf"))
        for t in THRESHOLDS:                     # a stored tail always finds its whole lineage
            upstream, tail = _lineage(t)
            assert tail not in cache or upstream in cache
    assert len(cache) < 2 * len(THRESHOLDS)
    assert all(k in cache for k in _lineage(THRESHOLDS[-1]))


def test_a_result_recomputed_after_eviction_equals_the_evicted_one(builtins, fbm256):
    from dynamix.engine.resolve import resolve

    cache = Cache(max_results=2)
    resolve(_layer(THRESHOLDS[0]), fbm256, cache)
    keys = _lineage(THRESHOLDS[0])
    evicted = [cache.get(k) for k in keys]
    resolve(_layer(THRESHOLDS[1]), fbm256, cache)
    assert not any(k in cache for k in keys)
    resolve(_layer(THRESHOLDS[0]), fbm256, cache)
    for key, old in zip(keys, evicted):
        new = cache.get(key)
        assert new is not old
        a, b = _arrays(old), _arrays(new)
        assert len(a) == len(b) > 0
        assert all(x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True)
                   for x, y in zip(a, b))
    assert evicted[1]["topology"].to_payload() == cache.get(keys[1])["topology"].to_payload()


def test_pinned_entries_survive_the_workload_above_the_bounds(builtins, fbm256):
    from dynamix.engine.resolve import resolve

    cache = Cache(max_results=2, max_bytes=1)
    pinned = _lineage(THRESHOLDS[0])
    for key in pinned:
        cache.pin(key)
    for thresh in THRESHOLDS[:4]:
        resolve(_layer(thresh), fbm256, cache)
    assert all(k in cache for k in pinned)
    held = cache.report()["results"]
    assert held["pinned"] == 2 and held["analysis"] <= 2


def test_what_each_layer_shows_survives_the_bounds_until_it_moves_on(builtins):
    from conftest import fbm2d, make_field
    from dynamix.engine.resolve import resolve

    field = make_field(fbm2d(64, 0.7, seed=0))
    cache = Cache(max_results=1, max_bytes=1)            # nothing else fits
    resolve(_layer(THRESHOLDS[0], layer_id=1), field, cache)
    resolve(_layer(THRESHOLDS[1], layer_id=2), field, cache)
    assert all(k in cache for t in THRESHOLDS[:2] for k in _lineage(t))
    resolve(_layer(THRESHOLDS[2], layer_id=1), field, cache)   # layer 1 shows another result
    assert not any(k in cache for k in _lineage(THRESHOLDS[0]))
    assert all(k in cache for t in THRESHOLDS[1:3] for k in _lineage(t))
    cache.mark_shown(2, None)                                  # layer 2 is gone
    assert not any(k in cache for k in _lineage(THRESHOLDS[1]))
    assert all(k in cache for k in _lineage(THRESHOLDS[2]))


def test_a_finished_resolve_trims_the_mlx_buffer_cache(builtins, monkeypatch):
    from conftest import fbm2d, make_field
    from dynamix.core import fft_policy
    from dynamix.engine.resolve import resolve

    calls = []
    monkeypatch.setattr(fft_policy, "trim_mlx_cache", lambda: calls.append(1))
    resolve(_layer(THRESHOLDS[0]), make_field(fbm2d(64, 0.7, seed=0)), Cache())
    assert calls == [1]


def test_the_report_keeps_the_three_budgets_apart(monkeypatch):
    import sys
    from types import SimpleNamespace

    from dynamix.engine.cache import retained_nbytes, stage_nbytes

    monkeypatch.setitem(sys.modules, "mlx.core", SimpleNamespace(
        get_active_memory=lambda: 7, get_cache_memory=lambda: 11))
    cache = Cache(max_bytes=10_000, max_results=3)
    result = {"chains": [{"x": np.zeros(4)}]}
    cache.put("result", result)
    cache.put("field", np.zeros(8))
    cache.pin("field")
    cache.stages["s"] = {"cwt": np.zeros(16)}
    report = cache.report()
    assert report["results"] == {
        "entries": 2, "analysis": 1, "max_results": 3, "bytes": retained_nbytes(result["chains"]),
        "max_bytes": 10_000, "pinned": 1, "pinned_bytes": retained_nbytes(np.zeros(8))}
    assert report["stages"] == {"entries": 1, "bytes": stage_nbytes({"cwt": np.zeros(16)}),
                                "max_bytes": 2_000_000_000}
    assert report["mlx"] == {"active": 7, "cache": 11}


def test_a_level_carried_as_its_cells_counts_as_an_output_and_not_against_the_analysis_bound():
    """A pyramid level that travels as its decimated cell grid is a lazy output like a raster
    one: it never counts toward the bound on analysis results, so scrubbing the levels of many
    quantities never evicts them by count."""
    import numpy as np
    from dynamix.engine.cache import Cache

    cache = Cache(max_results=2)
    for k in range(10):
        cache.put(f"level{k}", {"cells": np.zeros((8, 8), np.float32), "paint": {"level": 3, "pad": (0, 0), "crop": (0, 0, 64, 64)}})
    assert all(f"level{k}" in cache for k in range(10))
    assert cache.report()["results"]["analysis"] == 0
