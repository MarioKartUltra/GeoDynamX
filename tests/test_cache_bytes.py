# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The byte budget: every entry counts, its arrays (``stage_nbytes``) plus ``OBJECT_BYTES`` for each
Python object it holds and a topology graph by its own ``nbytes``, and a value two entries share
counts once. Past ``Cache.max_bytes`` the least-recent unpinned entries are evicted first; pins ride
above it, the newest entry always stays, and the count bound is unchanged. The window's budget is a
Settings field, 2 GB by default, beside the bound on analysis results, 8 by default."""
from __future__ import annotations

import json

import numpy as np

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import OBJECT_BYTES, Cache, retained_nbytes, stage_nbytes

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


def test_every_entry_counts_toward_the_budget_results_and_fields_included():
    cache = Cache(max_bytes=5 * MB)
    stack = RasterField(name="stack", values=np.zeros((256, 256, 4)), frame=LocalFrame(),
                        x_axis=np.arange(256.0), y_axis=np.arange(256.0))
    cache.put("stack", stack)                                   # a 2 MB multiband field
    cache.put("result", {"extrema": [], "raster_out": np.zeros(MB, np.float32)})
    assert "stack" not in cache and "result" in cache
    cache.put("out", _raster(4 * MB))
    assert "result" not in cache and "out" in cache


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


def test_a_landing_keeps_the_entry_stored_last():
    cache = Cache(max_bytes=1)
    cache.put("big", _raster(4 * MB))
    cache.mark_shown(1, "analysis")
    assert "big" in cache


def test_the_count_bound_is_unchanged():
    cache = Cache(maxsize=2, max_bytes=100 * MB)
    for k in ("a", "b", "c"):
        cache.put(k, _raster(MB))
    assert len(cache) == 2 and "a" not in cache


def test_settings_carry_the_output_budget_two_gigabytes_by_default(tmp_path, monkeypatch):
    from dynamix.shell.settings import Settings, load_settings, update_settings

    assert Settings().cache_ram_bytes == 0
    assert load_settings().cache_ram_bytes == 0
    update_settings(cache_ram_bytes=123_456)
    assert load_settings().cache_ram_bytes == 123_456
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"output_cache_bytes": -5}), encoding="utf-8")
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(path))
    assert load_settings().cache_ram_bytes == 0


def test_an_entry_weighs_its_arrays_and_one_object_size_per_python_object():
    value = {"chains": [{"x": np.zeros(3), "y": np.zeros(3)} for _ in range(10)],
             "scales": np.zeros(4), "n": 3}
    objects = 1 + 1 + 10 * 3 + 1 + 1           # the dict, the list, each chain and its arrays, two more
    assert retained_nbytes(value) == stage_nbytes(value) + objects * OBJECT_BYTES


def test_a_topology_graph_weighs_its_nodes_and_edges():
    from dynamix.topology.model import ELEMENT_BYTES, Edge, Node, TopologyModel

    model = TopologyModel()
    model.nodes = {f"n{i}": Node(f"n{i}", "point") for i in range(3)}
    model.edges = [Edge("spatial", "n0", "n1"), Edge("spatial", "n1", "n2")]
    assert model.nbytes == 5 * ELEMENT_BYTES
    assert retained_nbytes({"topology": model}) == 5 * ELEMENT_BYTES + 2 * OBJECT_BYTES


def test_a_value_two_entries_share_counts_once():
    from dynamix.topology.model import TopologyModel

    product = np.zeros(MB // 8)
    cache = Cache()
    cache.put("wtmm", {"chain_product": product})
    cache.put("topology", {"chain_product": product, "topology": TopologyModel()})
    assert MB < cache.report()["results"]["bytes"] < MB + 1000


def test_the_estimate_follows_the_traced_memory_of_a_wtmm_result_and_its_topology(clean_registry):
    """The two measured constants hold on a result they were not fitted on (n_oct 2, 256²)."""
    import gc
    import tracemalloc

    from conftest import fbm2d, make_field
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import get_device, validate_params

    register_builtin_devices()
    field = make_field(fbm2d(256, 0.7, seed=0))
    wtmm, topology = get_device("wtmm2d"), get_device("chain_topology")
    params = validate_params(wtmm, {"n_oct": 2, "n_voice": 4})
    wtmm.compute(field, params)                  # compiles the kernels, plans the FFTs
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        result = wtmm.compute(field, params)
        graph = topology.compute(result, {})
        gc.collect()
        traced = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    cache = Cache()
    cache.put("wtmm", result)
    cache.put("topology", graph)
    counted = cache.report()["results"]["bytes"]
    assert abs(counted - traced) < 0.1 * traced


def test_settings_carry_the_analysis_result_bound_eight_by_default(tmp_path, monkeypatch):
    from dynamix.shell.settings import Settings, load_settings, update_settings

    assert Settings().result_cache_entries == 8
    assert load_settings().result_cache_entries == 8
    update_settings(result_cache_entries=3)
    assert load_settings().result_cache_entries == 3
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"result_cache_entries": 0}), encoding="utf-8")
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(path))
    assert load_settings().result_cache_entries == 8


def test_weighing_an_entry_never_builds_its_deferred_products():
    """A result's partition tables and topology graph are built by their first reader; the
    cache's weighing at insertion is not a reader: an unbuilt one weighs as one object, a built
    one as what it holds."""
    from dynamix.core.deferred import Deferred
    from dynamix.engine.cache import OBJECT_BYTES, retained_nbytes
    from dynamix.topology.model import DeferredTopology, TopologyModel

    calls = []

    def build_tables():
        calls.append("tables")
        return {"a": np.zeros(1000)}

    def build_graph():
        calls.append("graph")
        m = TopologyModel()
        m.add_node("n", "hchain")
        return m

    tables, graph = Deferred(build_tables), DeferredTopology(build_graph)
    assert retained_nbytes({"hd_std": tables, "topology": graph}) == 3 * OBJECT_BYTES
    assert calls == []
    tables["a"]
    assert retained_nbytes({"hd_std": tables}) == 3 * OBJECT_BYTES + 8000
    assert calls == ["tables"]
