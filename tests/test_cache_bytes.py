# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The output byte budget: past ``Cache.max_bytes`` the least-recent unpinned raster outputs (dicts
holding a ``raster`` ndarray) are evicted first. Analysis results and fields -- a multiband stack
included -- are never counted or evicted by it, pins ride above it, the newest entry always stays,
and the count bound is unchanged. The window's budget is a Settings field, 2 GB by default."""
from __future__ import annotations

import json

import numpy as np

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache

MB = 1_000_000


def _raster(nbytes: int) -> dict:
    return {"raster": np.zeros(nbytes // 4, dtype=np.float32), "diag": {}}


def test_a_default_cache_has_no_byte_budget():
    cache = Cache()
    assert cache.max_bytes is None
    for i in range(10):
        cache.put(f"k{i}", _raster(4 * MB))
    assert all(f"k{i}" in cache for i in range(10))


def test_the_least_recent_raster_goes_first_past_the_budget():
    cache = Cache(max_bytes=10 * MB)
    for k in ("a", "b", "c"):
        cache.put(k, _raster(4 * MB))
    assert "a" not in cache and "b" in cache and "c" in cache


def test_get_or_compute_honours_the_budget_and_a_hit_refreshes_recency():
    cache = Cache(max_bytes=10 * MB)
    cache.get_or_compute("a", lambda: _raster(4 * MB))
    cache.get_or_compute("b", lambda: _raster(4 * MB))
    cache.get_or_compute("a", lambda: _raster(4 * MB))          # a hit: "a" is now the recent one
    cache.get_or_compute("c", lambda: _raster(4 * MB))
    assert "a" in cache and "b" not in cache and "c" in cache
    assert (cache.hits, cache.misses) == (1, 3)


def test_results_and_fields_are_never_counted_or_evicted_by_bytes():
    cache = Cache(max_bytes=1 * MB)
    stack = RasterField(name="stack", values=np.zeros((256, 256, 4)), frame=LocalFrame(),
                        x_axis=np.arange(256.0), y_axis=np.arange(256.0))
    cache.put("stack", stack)                                   # a 2 MB multiband field
    cache.put("result", {"extrema": [], "raster_out": np.zeros(MB, np.float32)})
    cache.put("out", _raster(4 * MB))
    cache.put("out2", _raster(4 * MB))
    assert "stack" in cache and "result" in cache
    assert "out" not in cache and "out2" in cache


def test_a_pinned_raster_is_never_evicted_and_rides_above_the_budget():
    cache = Cache(max_bytes=10 * MB)
    cache.put("a", _raster(8 * MB))
    cache.pin("a")
    cache.put("b", _raster(4 * MB))
    cache.put("c", _raster(4 * MB))
    assert "a" in cache and "b" in cache and "c" in cache     # 8 MB unpinned: within budget
    cache.put("d", _raster(4 * MB))
    assert "a" in cache and "b" not in cache and "c" in cache and "d" in cache


def test_the_newest_entry_stays_even_when_it_alone_exceeds_the_budget():
    cache = Cache(max_bytes=1 * MB)
    cache.put("big", _raster(4 * MB))
    assert "big" in cache
    cache.put("big2", _raster(4 * MB))
    assert "big" not in cache and "big2" in cache


def test_the_count_bound_is_unchanged():
    cache = Cache(maxsize=2, max_bytes=100 * MB)
    for k in ("a", "b", "c"):
        cache.put(k, _raster(MB))
    assert len(cache) == 2 and "a" not in cache


def test_settings_carry_the_output_budget_two_gigabytes_by_default(tmp_path, monkeypatch):
    from dynamix.shell.settings import Settings, load_settings, update_settings

    assert Settings().output_cache_bytes == 2_000_000_000
    assert load_settings().output_cache_bytes == 2_000_000_000
    update_settings(output_cache_bytes=123_456)
    assert load_settings().output_cache_bytes == 123_456
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"output_cache_bytes": -5}))
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(path))
    assert load_settings().output_cache_bytes == 2_000_000_000
