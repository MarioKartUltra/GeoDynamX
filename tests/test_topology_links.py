# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.topology.links -- user-asserted links between chains. Headless: stdlib + numpy only, no Qt anywhere in this file."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.topology.codes import LINE, permitted
from dynamix.topology.links import LinkStore, ObjRef, suggest_code
from dynamix.topology.model import Edge, TopologyModel


def refs():
    return (ObjRef("L1", "wtmm2d", "line", 3), ObjRef("L2", "mz_edges", "line", 7))


def test_link_roundtrip():
    a, b = refs()
    s = LinkStore()
    s.link(a, b, code=suggest_code(np.array([[0, 0], [1, 0]]),
                                   np.array([[5, 5], [6, 5]]), contact_scale=2.0),
           scale_first_contact=2.0)
    assert len(s.links_for_layer("L1")) == 1
    restored = LinkStore.from_payload(s.to_payload())
    (lk,) = restored.links_for_layer("L2")
    assert lk.scale_first_contact == 2.0


def test_link_rejects_impossible_code():
    a, b = refs()
    s = LinkStore()
    import pytest
    with pytest.raises(ValueError):
        s.link(a, b, code=10 ** 9)


def test_suggest_code_three_regimes():
    far = suggest_code(np.array([[0, 0]]), np.array([[10, 10]]), contact_scale=1.0)
    near = suggest_code(np.array([[0, 0]]), np.array([[0.5, 0]]), contact_scale=1.0)
    assert far != near


def test_unlink():
    a, b = refs()
    s = LinkStore()
    s.link(a, b, code=suggest_code(np.array([[0, 0]]), np.array([[9, 9]]), 1.0))
    s.unlink(0)
    assert not s.links_for_layer("L1")


# --------------------------------------------------------------- extra coverage this task adds


def test_objref_rejects_unknown_kind():
    with pytest.raises(ValueError):
        ObjRef("L1", "wtmm2d", "polygon", 0)     # not a member of codes.KINDS


def test_suggest_code_all_three_regimes_are_permitted_line_line_2d():
    """The design's hard gate: every code suggest_code can ever return must be a member of
    permitted("line", "line", 2) -- proven directly here, not just trusted from the module-level
    self-check in links.py."""
    allowed = permitted(LINE, LINE, 2)
    far = suggest_code(np.array([[0, 0]]), np.array([[10, 10]]), contact_scale=1.0)
    near = suggest_code(np.array([[0, 0]]), np.array([[0.5, 0]]), contact_scale=1.0)
    overlapping = suggest_code(np.array([[0, 0], [10, 10]]), np.array([[0, 10], [10, 0]]),
                                contact_scale=0.1)
    assert {far, near, overlapping} <= allowed
    # the three regimes are genuinely distinct codes, not a heuristic that degenerates to one
    assert len({far, near, overlapping}) == 3


def test_unlink_leaves_the_other_layers_link_alone():
    a, b = refs()
    s = LinkStore()
    s.link(a, b, code=suggest_code(np.array([[0, 0]]), np.array([[9, 9]]), 1.0))
    s.unlink(0)
    assert not s.links_for_layer("L2")


def test_relinking_the_same_pair_reuses_nodes_not_duplicates():
    """LinkStore auto-adds a node for each ObjRef exactly once -- re-linking the same two chains
    (e.g. after the user changes their mind about the code) must not raise TopologyModel's own
    'node already exists' error."""
    a, b = refs()
    s = LinkStore()
    code = suggest_code(np.array([[0, 0]]), np.array([[9, 9]]), 1.0)
    s.link(a, b, code=code)
    s.link(a, b, code=code)    # must not raise
    assert len(s.links_for_layer("L1")) == 2


def test_from_payload_of_an_empty_dict_is_an_empty_store():
    """The absent-key case a Project without any saved links must tolerate."""
    store = LinkStore.from_payload({})
    assert store.all() == []


# --------------------------------------------------------------- Edge.scale_first_contact
# (topology/model.py, additive) -- payload round-trip, and old payloads without the key.


def test_edge_scale_first_contact_round_trips_through_topology_model():
    m = TopologyModel()
    m.add_node("a", LINE, space_dim=2)
    m.add_node("b", LINE, space_dim=2)
    m.assert_spatial(suggest_code(np.array([[0, 0]]), np.array([[9, 9]]), 1.0), "a", "b",
                     space_dim=2, scale_first_contact=4.5)
    m2 = TopologyModel.from_payload(m.to_payload())
    assert m2.edges[0].scale_first_contact == 4.5
    assert m2.to_payload() == m.to_payload()


def test_old_payload_without_scale_first_contact_key_still_loads():
    """A payload written before this field existed has no 'scale_first_contact' key on its edges
    at all -- Edge.scale_first_contact must default to None, not raise KeyError."""
    old_payload = {
        "nodes": [{"node_id": "a", "kind": "line", "space_dim": 2, "anchor": []},
                  {"node_id": "b", "kind": "line", "space_dim": 2, "anchor": []}],
        "edges": [{"family": "spatial", "a": "a", "b": "b", "code": 31, "space_dim": 2,
                   "via": None}],       # no "scale_first_contact" key
    }
    m = TopologyModel.from_payload(old_payload)
    assert m.edges[0].scale_first_contact is None


def test_edge_default_scale_first_contact_is_none():
    """A bare Edge built the old way (every existing call site) is unaffected by the new field."""
    e = Edge("derivation", "a", "b", via="x")
    assert e.scale_first_contact is None
