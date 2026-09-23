# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import numpy as np
import pytest

from dynamix.devices.topology import ChainTopology, MinVChains


def _result():
    """Duplicated from tests/test_topology_wtmm.py -- tests/ is not a package, so no
    cross-test import. One 5-point H-line at scale 0 along y=0, two V-chains through rows
    1 and 3, plus one isolated extremum (line_id -1)."""
    ext0 = {
        "x": np.array([0, 1, 2, 3, 4, 9], dtype=np.int64),
        "y": np.zeros(6, dtype=np.int64),
        "mod": np.array([1.0, 9.0, 2.0, 8.0, 1.0, 5.0]),
        "arg": np.zeros(6),
        "line_id": np.array([0, 0, 0, 0, 0, -1], dtype=np.int64),
    }
    chains = [
        {"x": np.array([1], dtype=np.int64), "y": np.array([0], dtype=np.int64),
         "mod": np.array([9.0])},
        {"x": np.array([3], dtype=np.int64), "y": np.array([0], dtype=np.int64),
         "mod": np.array([8.0])},
    ]
    return {"extrema": [ext0], "chains": chains, "scales": np.array([1.0]),
            "_shape": (1, 10)}


def _two_scale_result():
    """Two scales over the same 5-point H-line. Both V-chains die after scale 0 (their x/y
    arrays are one entry long), so scale 0's H-line is crossed twice and scale 1's not at all --
    the two scales disagree, which is what makes the post-ScaleSelect indexing observable."""
    def _layer(isolated_x):
        return {
            "x": np.array([0, 1, 2, 3, 4, isolated_x], dtype=np.int64),
            "y": np.zeros(6, dtype=np.int64),
            "mod": np.array([1.0, 9.0, 2.0, 8.0, 1.0, 5.0]),
            "arg": np.zeros(6),
            "line_id": np.array([0, 0, 0, 0, 0, -1], dtype=np.int64),
        }
    chains = [
        {"x": np.array([1], dtype=np.int64), "y": np.array([0], dtype=np.int64),
         "mod": np.array([9.0])},
        {"x": np.array([3], dtype=np.int64), "y": np.array([0], dtype=np.int64),
         "mod": np.array([8.0])},
    ]
    return {"extrema": [_layer(9), _layer(7)], "chains": chains,
            "scales": np.array([1.0, 2.0]), "_shape": (1, 10)}


def test_chain_topology_augments_without_mutating_input():
    r = _result()
    out = ChainTopology().compute(r, {})
    assert "topology" in out and "topology" not in r
    assert out["chains"] is r["chains"]


def test_chain_topology_raises_loudly_without_line_id():
    r = _result()
    del r["extrema"][0]["line_id"]
    with pytest.raises(ValueError):
        ChainTopology().compute(r, {})


def test_min_vchains_regression_never_touches_vchains():
    """THE EQSelect flaw, encoded: filtering H-chains by V-chain count must prune H-chains
    and must leave result['chains'] identical."""
    r = ChainTopology().compute(_result(), {})
    out = MinVChains().apply(r, {"min_vchains": 3})
    assert out["chains"] is r["chains"]                      # V-chains untouched, same object
    assert out["_hchains_dropped"] == 1                      # the only H-chain has 2 < 3
    assert out["_topology_stale"] is True                    # anchors no longer index extrema
    kept = out["extrema"][0]
    assert kept["x"].tolist() == [9]                         # only the isolated point survives
    assert kept["line_id"].tolist() == [-1]


def test_min_vchains_keeps_qualifying_hchains():
    r = ChainTopology().compute(_result(), {})
    out = MinVChains().apply(r, {"min_vchains": 2})
    assert out["_hchains_dropped"] == 0
    assert out["extrema"][0]["x"].tolist() == r["extrema"][0]["x"].tolist()
    assert "_topology_stale" not in out            # nothing rewritten, anchors still resolve


def test_min_vchains_prunes_the_scale_the_user_is_looking_at():
    """After ScaleSelect the displayed list is one layer long, so its position says nothing
    about which scale it is. Pruning by position drops scale 0's line ids out of scale k's
    layer -- silently, since both scales label their lines from 0."""
    from dynamix.devices.filters import ScaleSelect

    r = ChainTopology().compute(_two_scale_result(), {})
    sel = ScaleSelect().apply(r, {"scale_idx": 1})
    out = MinVChains().apply(sel, {"min_vchains": 2})
    assert out["_hchains_dropped"] == 1                       # scale 1's line has no V-chains
    assert out["extrema"][0]["x"].tolist() == [7]             # ...and it is gone from the view
    assert out["extrema"][0]["line_id"].tolist() == [-1]


def test_min_vchains_leaves_a_qualifying_selected_scale_alone():
    from dynamix.devices.filters import ScaleSelect

    r = ChainTopology().compute(_two_scale_result(), {})
    sel = ScaleSelect().apply(r, {"scale_idx": 0})
    out = MinVChains().apply(sel, {"min_vchains": 2})
    assert out["extrema"][0]["x"].tolist() == [0, 1, 2, 3, 4, 9]   # scale 0 qualifies


def test_chain_topology_prebuilds_the_vchain_index():
    """The incidence index is a pure function of the model, so it is built once in the
    transform (where it caches) rather than per redraw in the filter."""
    out = ChainTopology().compute(_result(), {})
    assert out["_topology_vids"] == {"h0.0": frozenset({"v0", "v1"})}


def test_min_vchains_consumes_the_prebuilt_index():
    """Sabotage the index: if the filter still reads it, the H-chain must fail a threshold its
    model-derived V-chain count would pass."""
    r = dict(ChainTopology().compute(_result(), {}))
    r["_topology_vids"] = {"h0.0": frozenset()}
    out = MinVChains().apply(r, {"min_vchains": 1})
    assert out["_hchains_dropped"] == 1


def test_min_vchains_rebuilds_the_index_when_it_is_missing():
    """A topology without an index -- an older cached result, or a model asserted by hand --
    still filters; it just pays for the index inline."""
    r = dict(ChainTopology().compute(_result(), {}))
    r.pop("_topology_vids")
    out = MinVChains().apply(r, {"min_vchains": 3})
    assert out["_hchains_dropped"] == 1
    assert out["extrema"][0]["x"].tolist() == [9]


def test_min_vchains_ignores_nodes_without_anchors():
    """A hand-asserted model names its elements, not array rows: anchor is (). Those nodes are
    unclassifiable here, and skipping them beats an IndexError."""
    from dynamix.topology.codes import LINE, POINT, point_code
    from dynamix.topology.model import TopologyModel

    m = TopologyModel()
    m.add_node("fault", LINE, space_dim=2)
    m.add_node("piercing", POINT, space_dim=2)
    m.assert_spatial(point_code(at="boundary"), "piercing", "fault", space_dim=2)

    out = MinVChains().apply({"extrema": [], "topology": m}, {"min_vchains": 2})
    assert out["_hchains_dropped"] == 0


def test_min_vchains_degrades_without_topology():
    r = _result()
    assert MinVChains().apply(r, {"min_vchains": 2}) is r


def test_registered_and_resolvable_through_engine(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.engine.cache import cache_key

    register_builtin_devices()
    # A stub-free two-transform pipeline needs a real field; instead pin the key-threading
    # contract directly, the way tests/test_engine.py does for two-transform chains:
    k1 = cache_key("wtmm2d", "src0", {"n_oct": 3})
    ka = cache_key("chain_topology", "src0", {}, upstream=k1)
    kb = cache_key("chain_topology", "src0", {}, upstream="other")
    assert ka != kb                                          # lineage reaches the key
    from dynamix.model.device import get_device, is_transform
    assert is_transform(get_device("chain_topology"))
    assert not is_transform(get_device("min_vchains"))
