# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A landing builds only what it reads. The partition tables, the CSR chain product and the chain
graph are built by their first reader, once, and then equal what an eager build gives."""
from __future__ import annotations

import collections
import gc
import threading
import time

import numpy as np
import pytest

from conftest import fbm2d, make_layer
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache, stage_nbytes
from dynamix.engine.resolve import resolve

#: The window's DEMO_CHAIN (``dynamix.shell.main_window``), restated so this module stays headless.
DEMO = (
    ("wtmm2d", {"n_oct": 3, "n_voice": 4}),
    ("chain_topology", {}),
    ("scale_select", {"scale_idx": 0}),
    ("min_vchains", {"min_vchains": 1}),
    ("orientation_wedge", {"centre": 0.0, "half_width": 90.0}),
    ("modulus_threshold", {"frac": 0.0}),
)


def _demo(scale_idx=0):
    return make_layer(*((n, dict(p, scale_idx=scale_idx) if n == "scale_select" else p)
                        for n, p in DEMO))


@pytest.fixture
def field():
    n = 96
    return RasterField(name="fbm", values=fbm2d(n, 0.7, seed=3), frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(n)), y_axis=np.arange(float(n)))


@pytest.fixture
def builds(monkeypatch, clean_registry):
    """Calls of the three builders a landing should not need, counted."""
    from dynamix.core import chain_product, wtmm_backend
    from dynamix.devices import register_builtin_devices
    from dynamix.topology import wtmm as topology_wtmm

    register_builtin_devices()
    counts = collections.Counter()
    for owner, name in ((wtmm_backend.PythonWTMMBackend, "partition2d"),
                        (chain_product, "build_chain_product"),
                        (topology_wtmm, "build_chain_topology")):
        def counted(*args, _name=name, _original=getattr(owner, name), **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(owner, name, counted)
    return counts


def _graph_objects() -> int:
    from dynamix.topology.model import Edge, Node

    gc.collect()
    return sum(isinstance(o, (Node, Edge)) for o in gc.get_objects())


def test_a_demo_chain_landing_builds_no_tables_product_or_graph(builds, field):
    graph_before = _graph_objects()
    cache = Cache()
    r = resolve(_demo(), field, cache)
    resolve(_demo(scale_idx=2), field, cache)            # a scale tick reads nothing more
    stage_nbytes(r.result)
    assert builds == {}
    assert _graph_objects() == graph_before
    result = r.result
    assert result["hd_std"] is not None and result["hd_cmax"] is not None
    assert "chain_product" in result and "topology" in result and "_topology_vids" in result
    assert r.result["_hchains_dropped"] > 0               # min_vchains still filtered


def test_each_product_is_built_once_by_its_first_reader_and_equals_an_eager_build(builds, field):
    from dynamix.core.chain_product import build_chain_product
    from dynamix.core.wtmm_backend import PythonWTMMBackend
    from dynamix.topology.wtmm import build_chain_topology, vchain_index

    cache = Cache()
    result = resolve(_demo(), field, cache).result
    raw = cache.get(resolve(make_layer(DEMO[0]), field, cache).analysis_key)
    params = raw["params"]

    tables = (result["hd_std"]["tau_qa"], result["hd_cmax"]["tau_qa"], dict(result["hd_std"]))
    product = dict(result["chain_product"])
    model = result["topology"]
    nodes, edges, vids = dict(model.nodes), list(model.edges), dict(result["_topology_vids"])
    assert builds == {"partition2d": 1, "build_chain_product": 1, "build_chain_topology": 1}
    assert result["hd_cmax"]["tau_qa"] is tables[1] and model.edges is model.edges
    assert builds == {"partition2d": 1, "build_chain_product": 1, "build_chain_topology": 1}

    std, cmax = PythonWTMMBackend().partition2d(raw["chains"], raw["scales"],
                                                q_list=params["q_list"],
                                                min_chain_len=params["min_chain_len"])
    assert set(tables[2]) == set(std)
    for key in std:
        np.testing.assert_array_equal(result["hd_std"][key], std[key])
        np.testing.assert_array_equal(result["hd_cmax"][key], cmax[key])
    eager = build_chain_product(raw["extrema"], raw["chains"], raw["scales"],
                                raw["_shape"][:2], runs=raw["_hline_runs"])
    assert set(product) == set(eager)
    for key, value in eager.items():
        np.testing.assert_array_equal(product[key], value)
    graph = build_chain_topology(raw)
    assert nodes == graph.nodes and edges == graph.edges
    assert vids == vchain_index(graph)


def test_min_vchains_drops_by_the_count_index_exactly_what_the_graph_drops(clean_registry, field):
    from dynamix.devices import register_builtin_devices
    from dynamix.devices.topology import MinVChains

    register_builtin_devices()
    cache = Cache()
    topo = resolve(make_layer(*DEMO[:2]), field, cache).result
    by_graph = {k: v for k, v in topo.items() if k != "_topology_counts"}
    for need in (1, 2, 3, 5):
        fast = MinVChains().apply(topo, {"min_vchains": need})
        slow = MinVChains().apply(by_graph, {"min_vchains": need})
        assert fast["_hchains_dropped"] == slow["_hchains_dropped"]
        for a, b in zip(fast["extrema"], slow["extrema"]):
            np.testing.assert_array_equal(a["x"], b["x"])
            np.testing.assert_array_equal(a["line_id"], b["line_id"])
    assert MinVChains().apply(topo, {"min_vchains": 1})["_hchains_dropped"] > 0


def test_vchain_counts_are_the_graph_index_counted(field, clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.topology.wtmm import build_chain_topology, vchain_counts, vchain_index

    register_builtin_devices()
    raw = resolve(make_layer(DEMO[0]), field, Cache()).result
    graph = build_chain_topology(raw)
    vids = vchain_index(graph)
    counts = vchain_counts(raw)
    assert len(counts) == len(raw["extrema"])
    lids_by_scale = collections.defaultdict(list)
    for node in graph.nodes_where(anchor0="hchain"):
        lids_by_scale[node.anchor[1]].append(node.anchor[2])
    for si, (lids, n) in enumerate(counts):
        assert lids.tolist() == sorted(lids_by_scale[si])
        assert n.tolist() == [len(vids.get(f"h{si}.{lid}", ())) for lid in lids.tolist()]
    assert any(n.max() > 1 for _lids, n in counts if n.size)


def test_a_deferred_value_is_built_once_when_two_threads_race():
    from dynamix.core.deferred import Deferred

    calls = []

    def build():
        calls.append(1)
        time.sleep(0.05)
        return {"a": np.arange(3)}

    d = Deferred(build)
    seen = []
    threads = [threading.Thread(target=lambda: seen.append(d["a"])) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == [1] and all(s is seen[0] for s in seen)


def test_the_stage_store_helpers_pass_a_deferred_through_unbuilt():
    from dynamix.core.deferred import Deferred
    from dynamix.core.wtmm_backend import _freeze, _own_structure

    d = Deferred(lambda: {"a": np.zeros(4)})
    holder = {"x": [d], "y": d}
    _freeze(holder)
    owned = _own_structure(holder)
    assert owned["y"] is d and owned["x"][0] is d
    assert stage_nbytes(holder) == 0 and not d.built
    assert d["a"].nbytes == 32 and stage_nbytes(holder) == 64     # built, it weighs its arrays
