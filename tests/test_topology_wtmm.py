# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import numpy as np
import pytest

from dynamix.topology.wtmm import build_chain_topology, split_walk


def test_split_walk_no_owner_is_one_span():
    assert split_walk(np.array([3.0, 2.0, 5.0]), np.array([], dtype=np.int64)) == [(0, 3)]


def test_split_walk_single_owner_no_split():
    assert split_walk(np.array([1.0, 9.0, 1.0]), np.array([1])) == [(0, 3)]


def test_split_walk_splits_at_local_min_between_owners():
    mod = np.array([1.0, 9.0, 4.0, 2.0, 5.0, 8.0, 1.0])
    #                     ^owner    min^       ^owner   -> cut AT index 3; min joins right span
    assert split_walk(mod, np.array([1, 5])) == [(0, 3), (3, 7)]


def test_split_walk_adjacent_owners():
    mod = np.array([9.0, 8.0])
    assert split_walk(mod, np.array([0, 1])) == [(0, 1), (1, 2)]


def test_split_walk_never_cuts_at_a_nan():
    """A NaN |T| is a hole in the walk, not the deepest point of it: np.argmin returns the
    NaN's index, which would put the cut in a place the data says nothing about."""
    mod = np.array([1.0, 9.0, 4.0, np.nan, 5.0, 8.0, 1.0])
    #                     ^owner  min^ NaN       ^owner
    assert split_walk(mod, np.array([1, 5])) == [(0, 2), (2, 7)]


def _result():
    """One 5-point H-line at scale 0 along y=0, two V-chains through rows 1 and 3,
    plus one isolated extremum (line_id -1)."""
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


def test_build_emits_expected_nodes_and_edges():
    m = build_chain_topology(_result())
    assert [n.node_id for n in m.nodes_where(anchor0="vchain")] == ["v0", "v1"]
    assert [n.node_id for n in m.nodes_where(anchor0="hchain")] == ["h0.0"]
    segs = m.nodes_where(anchor0="hseg")
    assert len(segs) == 2                      # two owners -> two segments
    assert len(m.nodes_where(anchor0="extremum")) == 2   # isolated row 5 is NOT an element
    # each extremum links to exactly one segment plus its v-chain (walk direction decides
    # WHICH segment, so pin the shape of the neighbourhood, not a specific segment id)
    nbrs = set(m.neighbors("e0.1"))
    assert "v0" in nbrs and len(nbrs) == 2
    assert any(n.startswith("h0.0.r0.s") for n in nbrs - {"v0"})
    assert all(m.neighbors(s.node_id, family="spatial") for s in segs)


def _cross_result():
    """A branching (plus-shaped) H-line: one line_id, but _order_lines restarts the walk at
    each arm, so the line arrives as three separately-ordered runs."""
    pts = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (0, 2), (1, 2), (3, 2), (4, 2)]
    x = np.array([p[0] for p in pts], dtype=np.int64)
    y = np.array([p[1] for p in pts], dtype=np.int64)
    ext0 = {
        "x": x, "y": y,
        "mod": np.arange(1.0, len(pts) + 1.0),
        "arg": np.zeros(len(pts)),
        "line_id": np.zeros(len(pts), dtype=np.int64),
    }
    chains = [{"x": np.array([2], dtype=np.int64), "y": np.array([2], dtype=np.int64),
               "mod": np.array([3.0])}]
    return {"extrema": [ext0], "chains": chains, "scales": np.array([1.0]),
            "_shape": (5, 5)}


def test_branching_line_asserts_equal_at_most_once():
    """R400 ``equal`` says "this segment IS the whole line". Three runs of one branching line
    cannot each be all of it -- that is a self-contradictory model, not an approximation."""
    m = build_chain_topology(_cross_result())
    runs = {n.anchor[3] for n in m.nodes_where(anchor0="hseg")}
    assert len(runs) > 1                      # the fixture really does branch
    for h in m.nodes_where(anchor0="hchain"):
        equal = [e for e in m.edges if e.b == h.node_id and e.code == 400]
        assert len(equal) <= 1
    # a full-span run of a branching line degrades to R476 covered_by, not disjoint/interior
    seg_codes = {e.code for e in m.edges
                 if e.a.startswith("h0.0.r") and e.b == "h0.0"}
    assert seg_codes == {476}


def test_single_run_line_still_asserts_equal():
    """The gate must not cost the honest case its R400: one unbranched line, one span."""
    r = _result()
    r["chains"] = []                          # no owners -> one span over the whole walk
    m = build_chain_topology(r)
    assert [e.code for e in m.edges if e.b == "h0.0"] == [400]


def test_build_requires_line_id_and_shape():
    r = _result()
    del r["extrema"][0]["line_id"]
    with pytest.raises(ValueError):
        build_chain_topology(r)
    r2 = _result()
    del r2["_shape"]
    with pytest.raises(ValueError):
        build_chain_topology(r2)
