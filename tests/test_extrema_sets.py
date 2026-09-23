# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Unit tests for the GUI-free extrema-selection core:

* ``reflayers.flatten_extrema`` -- chains -> selectable point cloud + per-point chain index.
* ``extrema_sets.ExtremaSetManager`` -- named H/V chain sets with an exact JSON round-trip.

No PySide6 / pyvista here (enforced separately by ``test_layering.py``).
"""
from __future__ import annotations

import numpy as np

from dynamix.core.extrema_sets import ExtremaSetManager
from dynamix.core.extrema_io import flatten_extrema


def _fake_te():
    """A tiny loaded-extrema dict: 2 H-chains (3,2 pts) + 2 V-chains (4,2 pts)."""
    h = [np.array([[10.0, 20.0, 1.0], [10.1, 20.1, 1.1], [10.2, 20.2, 1.2]]),
         np.array([[30.0, -5.0, 2.0], [30.1, -5.1, 2.1]])]
    v = [np.array([[10.0, 20.0, 1.0], [10.0, 20.0, 0.5], [10.0, 20.0, 0.2], [10.0, 20.0, 0.1]]),
         np.array([[30.0, -5.0, 2.0], [30.0, -5.0, 1.0]])]
    return dict(h_segments=h, v_segments=v)


def test_flatten_extrema_points_and_chain_index():
    f = flatten_extrema(_fake_te())
    assert f["h_pts"].shape == (5, 3) and f["v_pts"].shape == (6, 3)
    assert f["n_h_chains"] == 2 and f["n_v_chains"] == 2
    # per-point chain index groups the points back into their chains
    assert f["h_chain"].tolist() == [0, 0, 0, 1, 1]
    assert f["v_chain"].tolist() == [0, 0, 0, 0, 1, 1]
    # a "hit" on point 4 (index into h_pts) expands to its whole chain (chain 1 -> points {3,4})
    hit = 4
    chain = f["h_chain"][hit]
    assert np.flatnonzero(f["h_chain"] == chain).tolist() == [3, 4]


def test_flatten_extrema_carries_node_table():
    """When a schema-v2 node table is present, flatten passes the tree/incidence arrays through."""
    te = _fake_te()
    te.update(node_pts=np.zeros((3, 3)), node_scale=np.array([0, 1, 2]),
              node_mod=np.array([1.0, 2.0, 3.0]), node_hchain=np.array([0, 1, -1]),
              node_parent=np.array([-1, 0, 1]), node_root=np.array([0, 0, 0]),
              node_depth=np.array([0, 1, 2]), n_nodes=3, n_trees=1)
    f = flatten_extrema(te)
    assert f["n_nodes"] == 3 and f["n_trees"] == 1
    assert f["node_root"].tolist() == [0, 0, 0] and f["node_hchain"].tolist() == [0, 1, -1]
    assert f["node_mod"].tolist() == [1.0, 2.0, 3.0]


def test_flatten_extrema_empty():
    f = flatten_extrema(dict(h_segments=[], v_segments=[]))
    assert f["h_pts"].shape == (0, 3) and f["h_chain"].shape == (0,)
    assert f["n_h_chains"] == 0 and f["n_v_chains"] == 0


def test_set_manager_add_clean_and_range():
    m = ExtremaSetManager(n_h_chains=4, n_v_chains=3, region="Test", n_scales=5)
    sid = m.add_set(h_chains=[2, 2, 0, 99, -1], v_chains=[1])   # dupes/out-of-range dropped
    s = m.sets[0]
    assert s.set_id == sid and s.name == "Extrema 0"
    assert s.h_chains.tolist() == [0, 2] and s.v_chains.tolist() == [1]
    # assign unions more chains
    m.assign(sid, h_chains=[3, 0])
    assert m.sets[0].h_chains.tolist() == [0, 2, 3]


def test_set_manager_merge_and_delete():
    m = ExtremaSetManager(5, 5)
    a = m.add_set(h_chains=[0, 1])
    b = m.add_set(v_chains=[2, 3])
    merged = m.merge([a, b], name="AB")
    assert a not in m and b not in m and merged in m
    ms = m.sets[0]
    assert ms.name == "AB" and ms.h_chains.tolist() == [0, 1] and ms.v_chains.tolist() == [2, 3]
    m.delete(merged)
    assert len(m) == 0


def test_set_manager_json_roundtrip(tmp_path):
    m = ExtremaSetManager(6, 4, region="Alaska", n_scales=8)
    m.add_set(h_chains=[1, 3, 5], v_chains=[0], name="lineage", color="#abcdef")
    m.add_set(v_chains=[1, 2, 3])
    m.save(tmp_path)
    r = ExtremaSetManager.load(tmp_path)
    assert r.region == "Alaska" and r.n_scales == 8 and r.n_h_chains == 6 and r.n_v_chains == 4
    assert [s.name for s in r.sets] == ["lineage", "Extrema 1"]
    assert r.sets[0].color == "#abcdef" and r.sets[0].h_chains.tolist() == [1, 3, 5]
    assert r.sets[1].v_chains.tolist() == [1, 2, 3]
    assert r._next_id == 2                                   # counter restored above the max id


def test_extrema_set_tags_and_roundtrip(tmp_path):
    """A set carries arbitrary tags; set_tags updates named keys (others untouched), and they
    survive the JSON round-trip."""
    m = ExtremaSetManager(10, 5, region="global", n_scales=4)
    sid = m.add_set(h_chains=[1, 2], v_chains=[0], name="13a",
                    tags={"slab": "Kermadec", "group": "13a"})
    assert m.sets[0].tags == {"slab": "Kermadec", "group": "13a"}
    m.set_tags(sid, slab="Puysegur")                         # update one key; others unchanged
    assert m.sets[0].tags == {"slab": "Puysegur", "group": "13a"}
    m.set_tags(sid, slab="")                                 # empty value removes the key
    assert m.sets[0].tags == {"group": "13a"}
    m.set_tags(sid, slab="Vanuatu", group="v1")
    m.save(tmp_path)
    r = ExtremaSetManager.load(tmp_path)
    assert r.sets[0].tags == {"slab": "Vanuatu", "group": "v1"}


def test_add_set_drops_empty_tag_values():
    """'' means no tag, mirroring set_tags — an empty value must not persist as a dangling key."""
    m = ExtremaSetManager(5, 5)
    sid = m.add_set(h_chains=[0], tags={"slab": "", "group": "", "kind": "lineament"})
    assert m._get(sid).tags == {"kind": "lineament"}


def test_add_set_defaults_to_no_tags():
    """Tags are optional; omitting them yields an empty dict, never None."""
    m = ExtremaSetManager(5, 5)
    assert m._get(m.add_set(h_chains=[0])).tags == {}


def test_relabel_tag_updates_matching_sets_only():
    """Renaming a tag value must follow into every set carrying it, leaving other sets and other
    keys on the same set alone."""
    m = ExtremaSetManager(5, 5)
    a = m.add_set(h_chains=[0], name="a", tags={"group": "G1", "slab": "Alaska"})
    b = m.add_set(h_chains=[1], name="b", tags={"group": "G2"})
    c = m.add_set(h_chains=[2], name="c", tags={"slab": "G1"})   # same value, different key
    assert m.relabel_tag("group", "G1", "Kermadec deep") == 1
    assert m._get(a).tags == {"group": "Kermadec deep", "slab": "Alaska"}
    assert m._get(b).tags == {"group": "G2"}
    assert m._get(c).tags == {"slab": "G1"}                      # other key untouched


def test_load_tolerates_missing_tags_key(tmp_path):
    """An artifact with no tags key still loads, defaulting to an empty dict."""
    import json
    from pathlib import Path
    m = ExtremaSetManager(5, 5, region="R")
    m.add_set(h_chains=[0, 1], name="s0")
    m.save(tmp_path)
    p = Path(tmp_path) / "extrema_sets.json"
    payload = json.loads(p.read_text())
    for rec in payload["sets"]:
        rec.pop("tags", None)
    p.write_text(json.dumps(payload))
    r = ExtremaSetManager.load(tmp_path)                     # must not KeyError
    assert r.sets[0].tags == {}


def test_load_migrates_legacy_slab_group_keys(tmp_path):
    """EQSelect-era artifacts carry flat slab/group keys. They migrate into tags rather than being
    silently dropped — those files exist on disk in the slab repo today."""
    import json
    from pathlib import Path
    m = ExtremaSetManager(5, 5, region="R")
    m.add_set(h_chains=[0, 1], name="s0")
    m.save(tmp_path)
    p = Path(tmp_path) / "extrema_sets.json"
    payload = json.loads(p.read_text())
    for rec in payload["sets"]:
        rec.pop("tags", None)
        rec["slab"] = "Kermadec"
        rec["group"] = "13a"                                 # simulate an EQSelect artifact
    p.write_text(json.dumps(payload))
    r = ExtremaSetManager.load(tmp_path)
    assert r.sets[0].tags == {"slab": "Kermadec", "group": "13a"}
