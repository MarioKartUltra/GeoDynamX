# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.engine -- the cache and the resolve pass.

The headline property is the last test in this file: changing a FILTER parameter must not
invalidate a TRANSFORM's cached result. Everything about the application feeling responsive rests
on that, because a WTMM stack costs seconds to minutes and a slider drag costs microseconds.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from dynamix.engine.cache import Cache, cache_key
from dynamix.engine.resolve import resolve
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.project import Project


@pytest.fixture
def registry(clean_registry, stub_transform, stub_filter):
    from dynamix.model.device import register_device

    register_device(stub_transform)
    register_device(stub_filter)
    return clean_registry


def _layer(project, steps):
    src = project.add_source("/data/x.tif")
    return project.add_layer("x", src.source_id, Chain(steps))


# --------------------------------------------------------------------------- cache keys

def test_cache_key_is_stable_across_calls():
    a = cache_key("wtmm2d", "src0", {"n_oct": 3, "a_min": 1.0})
    b = cache_key("wtmm2d", "src0", {"n_oct": 3, "a_min": 1.0})
    assert a == b


def test_cache_key_ignores_param_dict_ordering():
    """The same recipe written two ways is one recipe -- otherwise a round trip through JSON could
    silently orphan a cached stack."""
    a = cache_key("wtmm2d", "src0", {"n_oct": 3, "a_min": 1.0})
    b = cache_key("wtmm2d", "src0", {"a_min": 1.0, "n_oct": 3})
    assert a == b


@pytest.mark.parametrize("field,value", [("device", "other"), ("source", "src9")])
def test_cache_key_separates_devices_and_sources(field, value):
    base = cache_key("wtmm2d", "src0", {"n_oct": 3})
    other = cache_key(value if field == "device" else "wtmm2d",
                      value if field == "source" else "src0", {"n_oct": 3})
    assert base != other


def test_cache_key_changes_with_params():
    assert cache_key("d", "s", {"n_oct": 3}) != cache_key("d", "s", {"n_oct": 4})


# --------------------------------------------------------------------------- the store

def test_get_or_compute_computes_once():
    cache, calls = Cache(), []
    for _ in range(3):
        cache.get_or_compute("k", lambda: calls.append(1) or "v")
    assert calls == [1] and cache.hits == 2 and cache.misses == 1


def test_invalidate_forces_a_recompute():
    cache, calls = Cache(), []

    def compute():
        calls.append(1)
        return "v"

    cache.get_or_compute("k", compute)
    assert cache.invalidate("k") is True
    cache.get_or_compute("k", compute)
    assert len(calls) == 2
    assert cache.invalidate("nope") is False


def test_invalidate_reports_presence_for_a_none_value():
    """Presence must be tested with `in`, not by inspecting the popped value -- otherwise an entry
    legitimately holding None reports absent and a caller cannot tell a miss from a stored null."""
    cache = Cache()
    cache.put("k", None)
    assert "k" in cache
    assert cache.invalidate("k") is True
    assert "k" not in cache


# --------------------------------------------------------------------------- resolve

def test_resolve_runs_transform_then_filters(registry):
    p = Project()
    layer = _layer(p, (DeviceRef("t", {"scale": 2}), DeviceRef("f", {"cut": 0.25})))
    r = resolve(layer, object(), Cache())
    assert r.transforms_run == ("t",) and r.filters_run == ("f",)
    assert r.result["scale"] == 2 and r.result["cut"] == 0.25


def test_resolve_reports_a_miss_on_first_run_and_a_hit_after(registry):
    p = Project()
    layer = _layer(p, (DeviceRef("t", {"scale": 2}),))
    cache = Cache()
    first = resolve(layer, object(), cache)
    second = resolve(layer, object(), cache)
    assert (first.cache_misses, first.from_cache) == (1, False)
    assert (second.cache_hits, second.from_cache) == (1, True)


def test_resolve_fills_defaults_from_the_declaration(registry):
    p = Project()
    layer = _layer(p, (DeviceRef("t", {}),))
    assert resolve(layer, object(), Cache()).result["scale"] == 4


def test_two_layers_over_one_source_share_a_cached_transform(registry):
    """Sharing a SourceRef is what makes this hold; it is why add_source dedupes by path."""
    p = Project()
    src = p.add_source("/data/x.tif")
    a = p.add_layer("a", src.source_id, Chain((DeviceRef("t", {"scale": 2}),)))
    b = p.add_layer("b", src.source_id, Chain((DeviceRef("t", {"scale": 2}),)))
    cache = Cache()
    resolve(a, object(), cache)
    assert resolve(b, object(), cache).from_cache is True


def test_changing_a_transform_param_does_invalidate(registry):
    p = Project()
    layer = _layer(p, (DeviceRef("t", {"scale": 2}),))
    cache = Cache()
    resolve(layer, object(), cache)
    layer.chain = Chain((DeviceRef("t", {"scale": 3}),)).materialized()
    assert resolve(layer, object(), cache).from_cache is False


def test_changing_a_FILTER_param_does_not_invalidate_the_transform(registry):
    """THE property the whole design rests on.

    A filter change must reuse the cached transform. If this ever fails, dragging a scale slider
    would recompute a WTMM stack -- seconds to minutes per frame -- and the application stops being
    an instrument and becomes a batch job with a progress bar.
    """
    p = Project()
    layer = _layer(p, (DeviceRef("t", {"scale": 2}), DeviceRef("f", {"cut": 0.1})))
    cache = Cache()
    resolve(layer, object(), cache)

    for cut in (0.2, 0.5, 0.9):
        layer.chain = Chain((DeviceRef("t", {"scale": 2}),
                             DeviceRef("f", {"cut": cut}))).materialized()
        r = resolve(layer, object(), cache)
        assert r.from_cache is True, f"cut={cut} recomputed the transform"
        assert r.result["cut"] == cut
    assert cache.misses == 1


# --------------------------------------------------------------------------- chained transforms

class _StageA:
    """A transform whose output depends on its params -- stands in for a preprocessing step."""

    name = "stage_a"

    from dynamix.model.param import Param as _P, ParamKind as _K
    params = (_P("n", _K.INT, default=1, min=1, max=99),)

    def compute(self, field, params, *, progress=None):
        return {"lineage": [("a", params["n"])]}

    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key as k
        return k(self.name, source_id, params)


class _StageB:
    """A second transform, consuming the first's output."""

    name = "stage_b"

    from dynamix.model.param import Param as _P, ParamKind as _K
    params = (_P("p", _K.INT, default=1, min=1, max=99),)

    def compute(self, field, params, *, progress=None):
        prior = field.get("lineage", []) if isinstance(field, dict) else []
        return {"lineage": list(prior) + [("b", params["p"])]}

    def cache_key(self, source_id, params):
        from dynamix.engine.cache import cache_key as k
        return k(self.name, source_id, params)


@pytest.fixture
def two_stage(clean_registry):
    from dynamix.model.device import register_device

    register_device(_StageA())
    register_device(_StageB())
    return clean_registry


def test_a_second_transform_consumes_the_first_result_not_the_raw_field(two_stage):
    p = Project()
    layer = _layer(p, (DeviceRef("stage_a", {"n": 3}), DeviceRef("stage_b", {"p": 7})))
    out = resolve(layer, object(), Cache()).result
    assert out["lineage"] == [("a", 3), ("b", 7)]


def test_chained_transform_keys_do_not_collide_across_different_upstreams(two_stage):
    """THE hazard this guards.

    Two chains differing only in the FIRST transform's params must not share the SECOND's cached
    result. A key of (device, source, params) alone is correct for a one-transform chain and
    silently wrong for a two-transform one -- returning numbers computed from a different input,
    with no error. It would first bite on any preprocessing step before WTMM, or on the
    embed -> recurrence -> RQA family, which is three transforms deep.
    """
    p = Project()
    cache = Cache()

    first = _layer(p, (DeviceRef("stage_a", {"n": 3}), DeviceRef("stage_b", {"p": 7})))
    assert resolve(first, object(), cache).result["lineage"] == [("a", 3), ("b", 7)]

    second = _layer(p, (DeviceRef("stage_a", {"n": 9}), DeviceRef("stage_b", {"p": 7})))
    got = resolve(second, object(), cache).result["lineage"]
    assert got == [("a", 9), ("b", 7)], f"stage_b served a stale result: {got}"


def test_an_identical_chain_still_hits_cache_at_every_stage(two_stage):
    """The fix must not defeat caching for a chain that genuinely repeats."""
    p = Project()
    cache = Cache()
    a = _layer(p, (DeviceRef("stage_a", {"n": 3}), DeviceRef("stage_b", {"p": 7})))
    b = _layer(p, (DeviceRef("stage_a", {"n": 3}), DeviceRef("stage_b", {"p": 7})))
    resolve(a, object(), cache)
    r = resolve(b, object(), cache)
    assert r.from_cache is True and r.cache_hits == 2


# --------------------------------------------------------------------------- The
# points.mapping column mapping must reach resolve identity.


def test_resolve_separates_cache_identity_for_two_layers_with_different_point_mappings(registry):
    """Two layers over ONE source (a CSV point catalogue, ``kind="points"``), same chain and
    params, but DIFFERENT ``layer.tags["points.mapping"]`` -- the column mapping that actually
    defines a PointSet's content. Before the fix these shared a cache key (source_id + params
    alone), so the second layer was silently served the first's cached result; the mapping must
    fold into resolve's own source identity so this is a genuine miss, not a stale hit."""
    p = Project()
    src = p.add_source("/data/quakes.csv", kind="points")
    a = p.add_layer("a", src.source_id, Chain((DeviceRef("t", {"scale": 2}),)))
    b = p.add_layer("b", src.source_id, Chain((DeviceRef("t", {"scale": 2}),)))
    a.tags["points.mapping"] = json.dumps({"lon": "x", "lat": "y"})
    b.tags["points.mapping"] = json.dumps({"lon": "x2", "lat": "y2"})
    cache = Cache()

    resolve(a, object(), cache)
    r = resolve(b, object(), cache)

    assert r.from_cache is False


def test_resolve_still_shares_cache_when_the_point_mapping_tag_is_identical(registry):
    """The fix must not defeat sharing when the mapping is genuinely the same -- including a
    reordered-but-equal mapping dict, since the fingerprint is over CANONICAL (sorted-key) JSON."""
    p = Project()
    src = p.add_source("/data/quakes.csv", kind="points")
    a = p.add_layer("a", src.source_id, Chain((DeviceRef("t", {"scale": 2}),)))
    b = p.add_layer("b", src.source_id, Chain((DeviceRef("t", {"scale": 2}),)))
    a.tags["points.mapping"] = json.dumps({"lon": "x", "lat": "y"})
    b.tags["points.mapping"] = json.dumps({"lat": "y", "lon": "x"})   # same mapping, reordered
    cache = Cache()

    resolve(a, object(), cache)
    r = resolve(b, object(), cache)

    assert r.from_cache is True


def test_resolve_ignores_an_absent_or_blank_mapping_tag(registry):
    """A raster layer (no ``points.mapping`` tag at all) must be completely unaffected -- the
    overwhelming majority of resolves. Also covers a layer that carries the tag as an empty
    string (never actually produced by ``point_import.py``, but tolerated the same way
    ``_display_style_of`` tolerates a garbled tag elsewhere)."""
    p = Project()
    src = p.add_source("/data/x.tif")
    a = p.add_layer("a", src.source_id, Chain((DeviceRef("t", {"scale": 2}),)))
    b = p.add_layer("b", src.source_id, Chain((DeviceRef("t", {"scale": 2}),)))
    b.tags["points.mapping"] = ""
    cache = Cache()

    resolve(a, object(), cache)
    r = resolve(b, object(), cache)

    assert r.from_cache is True


# --------------------------------------------------------------- selections through resolve

class _ProductTransform:
    """A Transform whose result carries the chain product + selection stamp, like wtmm2d."""

    name = "prodt"
    params = ()

    def compute(self, field, params, *, progress=None):
        from dynamix.core.chain_product import attach_chain_product

        def chain(xs, ys, mods, scales):
            mod = np.asarray(mods, dtype=np.float64)
            with np.errstate(divide="ignore"):
                log2_mod = np.log2(np.abs(mod))
            return {"x": np.asarray(xs, np.int64), "y": np.asarray(ys, np.int64),
                    "mod": mod, "log2_mod": log2_mod,
                    "log2_scales": np.log2(np.asarray(scales, float))[:mod.size]}

        scales = [1.0, 2.0]
        layer = {"x": np.arange(4, dtype=np.int64), "y": np.zeros(4, dtype=np.int64),
                 "mod": np.asarray([1.0, 0.8, 0.6, 0.9]), "arg": np.zeros(4),
                 "line_id": np.asarray([0, 0, 0, -1], np.int64)}
        res = {"extrema": [layer], "chains": [chain([0, 0], [0, 0], [1.0, 0.7], scales),
                                              chain([1], [0], [0.5], scales)],
               "scales": np.asarray(scales), "_shape": (4, 8)}
        return attach_chain_product(res)

    def cache_key(self, source_id, params):
        return f"prodt:{source_id}"


class _RogueFilter:
    """A legacy device with no selection awareness: reads and rewrites ``chains``."""

    name = "rogue"
    params = ()

    def apply(self, result, params):
        chains = result.get("chains") or []
        return dict(result, chains=list(chains), rogue_saw=len(chains))


def _register(*devices):
    from dynamix.model.device import register_device

    for d in devices:
        register_device(d)


@pytest.mark.skip(reason="selection mechanism disabled 2026-09-14 pending progressive-compute redesign")
def test_resolve_materializes_the_selection_for_the_terminal_result(clean_registry):
    """A chain of selection-aware filters narrows indices; the caller still receives honest
    filtered dicts (one materialization at the end), with the selection left live for views."""
    from dynamix.core.chain_product import selection_of
    from dynamix.devices.chain_filters import ChainLengthFilter

    _register(_ProductTransform(), ChainLengthFilter())
    p = Project()
    layer = _layer(p, (DeviceRef("prodt", {}), DeviceRef("chain_length", {"min_len": 2})))
    r = resolve(layer, None, Cache())
    assert len(r.result["chains"]) == 1
    assert r.result["chains"][0]["mod"].size == 2
    assert selection_of(r.result) is not None


def test_resolve_materializes_before_a_non_aware_device_runs(clean_registry):
    """A device outside the selection model must see the already-filtered dicts, exactly as the
    sequential dict path would have handed them over -- never the stale unfiltered ones."""
    from dynamix.core.chain_product import selection_of
    from dynamix.devices.chain_filters import ChainLengthFilter

    _register(_ProductTransform(), ChainLengthFilter(), _RogueFilter())
    p = Project()
    layer = _layer(p, (DeviceRef("prodt", {}), DeviceRef("chain_length", {"min_len": 2}),
                       DeviceRef("rogue", {})))
    r = resolve(layer, None, Cache())
    assert r.result["rogue_saw"] == 1                  # saw the filtered list, not all 2 chains
    assert selection_of(r.result) is None              # the rewrite invalidated the selection


@pytest.mark.skip(reason="selection mechanism disabled 2026-09-14 pending progressive-compute redesign")
def test_resolve_with_no_filters_returns_the_stamped_result_untouched(clean_registry):
    from dynamix.core.chain_product import selection_of

    _register(_ProductTransform())
    p = Project()
    layer = _layer(p, (DeviceRef("prodt", {}),))
    r = resolve(layer, None, Cache())
    assert len(r.result["chains"]) == 2
    sel = selection_of(r.result)
    assert sel is not None and sel["keep_v"] is None
