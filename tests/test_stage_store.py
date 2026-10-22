# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The engine cache's stage store: a bounded, byte-counted mapping for a transform's
intermediate stages. Never pinned; cleared with the cache."""
from __future__ import annotations

import numpy as np

from dynamix.engine.cache import Cache, StageStore, stage_nbytes


def _blob(n_bytes: int):
    return {"records": [{"x": np.zeros(n_bytes // 8, dtype=np.float64)}]}


def test_a_cache_carries_a_stage_store_bounded_to_eight_entries():
    cache = Cache()
    assert isinstance(cache.stages, StageStore)
    for i in range(10):
        cache.stages[f"k{i}"] = i
    assert len(cache.stages) == 8
    assert "k0" not in cache.stages and "k1" not in cache.stages
    assert cache.stages["k9"] == 9


def test_a_hit_refreshes_recency_and_counts():
    store = StageStore(maxsize=2)
    store["a"] = 1
    store["b"] = 2
    assert store["a"] == 1
    store["c"] = 3
    assert "a" in store and "b" not in store
    assert store.get("zzz") is None and store.get("zzz", 0) == 0
    assert (store.hits, store.misses) == (1, 2)


def test_the_byte_budget_evicts_the_least_recent_entry_first():
    store = StageStore(maxsize=8, max_bytes=2000)
    store["a"] = _blob(800)
    store["b"] = _blob(800)
    store["c"] = _blob(800)
    assert "a" not in store and "b" in store and "c" in store


def test_the_newest_entry_stays_even_above_the_budget_alone():
    store = StageStore(maxsize=8, max_bytes=100)
    store["big"] = _blob(8000)
    assert "big" in store


def test_stage_nbytes_walks_arrays_in_dicts_lists_and_fields():
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    field = RasterField(name="f", values=np.zeros((10, 10)), frame=LocalFrame(),
                        x_axis=np.arange(10.0), y_axis=np.arange(10.0))
    assert stage_nbytes(field) == 800 + 2 * 80
    assert stage_nbytes({"a": [np.zeros(4), (np.zeros(2),)], "n": 3.0}) == 48
    assert stage_nbytes(3) == 0


def test_the_cache_byte_budget_is_the_stores_and_clear_empties_both():
    cache = Cache(max_bytes=4096)
    assert cache.stages.max_bytes == 2_000_000_000
    cache.put("k", {"raster": np.zeros(4)})
    cache.stages["s"] = 1
    cache.clear()
    assert len(cache) == 0 and len(cache.stages) == 0
