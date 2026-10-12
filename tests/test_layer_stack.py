# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The layer stack tree: drawing order, inherited visibility and opacity, edits, payload."""
from __future__ import annotations

import pytest

from dynamix.model.layer_stack import Group, LayerStack, Row


def _stack():
    s = LayerStack()
    g = Group("ds:src0", "dem", auto=True)
    s.insert(g)
    s.insert(Row("src:src0", "raster"), "ds:src0", 0)
    s.insert(Row("prod:1", "product"), "ds:src0", 0)          # above the raster
    s.insert(Row("src:src1", "raster"))                         # top level, above the group
    return s


def test_drawn_is_bottom_first_through_groups():
    assert [r.id for r, _v, _o in _stack().drawn()] == ["src:src0", "prod:1", "src:src1"]


def test_a_hidden_group_hides_its_rows_and_opacity_multiplies():
    s = _stack()
    s.node("ds:src0").opacity = 0.5
    s.node("prod:1").opacity = 0.5
    assert {r.id: o for r, _v, o in s.drawn()}["prod:1"] == pytest.approx(0.25)
    s.node("ds:src0").visible = False
    assert [r.id for r, _v, _o in s.drawn()] == ["src:src1"]
    assert [(r.id, v) for r, v, _o in s.drawn(include_hidden=True)][:2] == [
        ("src:src0", False), ("prod:1", False)]


def test_move_marks_the_node_and_refuses_a_group_into_itself():
    s = _stack()
    s.move("src:src1", "ds:src0", 0)
    assert s.parent_of("src:src1").id == "ds:src0" and s.node("src:src1").moved
    with pytest.raises(ValueError):
        s.move("ds:src0", "ds:src0", 0)


def test_removing_the_last_row_of_an_auto_group_removes_the_group():
    s = _stack()
    s.remove("prod:1")
    s.remove("src:src0")
    assert s.node("ds:src0") is None
    g = s.new_group("mine")
    s.insert(Row("x:1", "product"), g.id, 0)
    s.remove("x:1")
    assert s.node(g.id) is not None                              # a user group stays


def test_ungroup_puts_the_children_where_the_group_was():
    s = _stack()
    s.ungroup("ds:src0")
    assert [c.id for c in s.root.children] == ["src:src1", "prod:1", "src:src0"]


def test_payload_round_trip_and_skip():
    s = _stack()
    s.node("prod:1").visible = False
    s.new_group("mine", index=1)
    back = LayerStack.from_payload(s.to_payload())
    assert back.to_payload() == s.to_payload()
    assert back.node("prod:1").visible is False
    assert "src:src0" not in LayerStack.from_payload(s.to_payload(skip={"src:src0"})).row_ids()
    assert back.new_group("again").id == "grp:1"                 # ids keep counting


def test_a_malformed_payload_entry_is_dropped():
    s = LayerStack.from_payload([{"id": "src:a", "kind": "raster"}, "junk", {"kind": "raster"}])
    assert s.row_ids() == ["src:a"]


def test_the_project_saves_and_loads_its_stack_and_map_crs():
    from dynamix.model.project import Project

    p = Project()
    src = p.add_source("/data/a.tif", label="a")
    p.layer_stack.insert(Row(f"src:{src.source_id}", "raster"))
    p.map_crs = "EPSG:32613"
    back = Project.from_payload(p.to_payload())
    assert back.layer_stack.row_ids() == [f"src:{src.source_id}"]
    assert back.map_crs == "EPSG:32613"


def test_an_old_project_opens_with_an_empty_stack():
    from dynamix.model.project import Project

    payload = Project().to_payload()
    del payload["layer_stack"], payload["map_crs"]
    back = Project.from_payload(payload)
    assert back.layer_stack.row_ids() == [] and back.map_crs is None


def test_a_temporary_source_leaves_its_rows_out_of_a_save():
    from dynamix.model.project import Project

    p = Project()
    keep = p.add_source("/data/a.tif")
    temp = p.add_source("/tmp/fork.npz")
    temp.temporary = True
    lay = p.add_layer("t", temp.source_id)
    for rid in (f"src:{keep.source_id}", f"src:{temp.source_id}", f"prod:{lay.layer_id}"):
        p.layer_stack.insert(Row(rid, "raster"))
    assert LayerStack.from_payload(p.to_payload()["layer_stack"]).row_ids() == [
        f"src:{keep.source_id}"]
