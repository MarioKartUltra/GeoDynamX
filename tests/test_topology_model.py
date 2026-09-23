# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import pytest

from dynamix.topology.codes import BODY, LINE, POINT, point_code
from dynamix.topology.model import TopologyModel


def _model():
    m = TopologyModel()
    m.add_node("h1", LINE, space_dim=2, anchor=("hchain", 0, 1))
    m.add_node("e1", POINT, space_dim=2, anchor=("extremum", 0, 4))
    m.add_node("ds1", "dataset", anchor=("source", "src0"))
    m.add_node("ds2", "dataset", anchor=("source", "src1"))
    return m


def test_assert_spatial_validates_against_permitted():
    m = _model()
    edge = m.assert_spatial(point_code(at="interior"), "e1", "h1", space_dim=2)
    assert edge.code == 92 and edge.family == "spatial" and edge.space_dim == 2
    with pytest.raises(ValueError):
        m.assert_spatial(511, "e1", "h1", space_dim=2)   # 511 impossible for point-line
    with pytest.raises(ValueError):
        m.assert_spatial(92, "e1", "nope", space_dim=2)  # unknown node


def test_duplicate_node_id_raises():
    m = _model()
    with pytest.raises(ValueError):
        m.add_node("h1", LINE, space_dim=2)


def test_assert_spatial_rejects_dataset_endpoint():
    """assert_spatial with a dataset node endpoint raises ValueError, not KeyError."""
    m = _model()
    with pytest.raises(ValueError):
        m.assert_spatial(point_code(at="interior"), "ds1", "h1", space_dim=2)


def test_assert_spatial_rejects_impossible_kind_dim_combo():
    """assert_spatial with impossible (kind_a, kind_b, dim) combo raises ValueError, not KeyError."""
    m = TopologyModel()
    m.add_node("b1", BODY, space_dim=3)
    m.add_node("l1", LINE, space_dim=2)
    # BODY/LINE at space_dim=2 is not in _PERMITTED (BODY only exists in R^3)
    with pytest.raises(ValueError):
        m.assert_spatial(31, "b1", "l1", space_dim=2)


def test_derivation_is_dataset_only():
    m = _model()
    e = m.add_derivation("ds2", "ds1", via="wtmm2d:abc123")
    assert e.family == "derivation" and e.via == "wtmm2d:abc123"
    with pytest.raises(ValueError):
        m.add_derivation("h1", "ds1", via="x")


def test_queries():
    m = _model()
    m.assert_spatial(point_code(at="interior"), "e1", "h1", space_dim=2)
    m.add_derivation("ds2", "ds1", via="k")
    assert m.neighbors("h1") == ["e1"]
    assert m.neighbors("ds1", family="derivation") == ["ds2"]
    assert [n.node_id for n in m.nodes_where(anchor0="hchain")] == ["h1"]
    assert len(m.edges_of("e1", family="spatial")) == 1


def test_payload_round_trip_with_zero_analysis():
    m = _model()
    m.assert_spatial(point_code(at="boundary"), "e1", "h1", space_dim=2)
    m.add_derivation("ds2", "ds1", via="k")
    m2 = TopologyModel.from_payload(m.to_payload())
    assert m2.to_payload() == m.to_payload()
    import json
    json.dumps(m.to_payload())   # payload must be pure-JSON
