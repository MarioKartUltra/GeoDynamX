# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Default placement: rows for content that has none; a moved row keeps its place."""
from __future__ import annotations

from dynamix.model.layer_stack import LayerInfo, LayerStack, reconcile, rows_for

HOLDER = frozenset({"HMap"})
WTMM = frozenset({"ExtremaStack", "VChains"})


def _tree(stack, group=None):
    group = group or stack.root
    return [(c.id, _tree(stack, c)) if hasattr(c, "children") else c.id for c in group.children]


def test_rows_for_follow_the_contracts_top_first():
    assert rows_for(LayerInfo(3, "src0", WTMM | HOLDER)) == [
        ("chn:3", "chains"), ("ext:3", "extrema"), ("prod:3", "product")]
    assert rows_for(LayerInfo(4, "src0", frozenset({"Field"}))) == []


def test_a_layer_whose_show_puts_a_raster_up_has_a_product_row():
    """The product row is the layer's shown raster -- a lazy output (M-Z's coarse or
    reconstruction) as much as an h-map."""
    assert rows_for(LayerInfo(5, "src0", frozenset({"ExtremaStack"}), shows_raster=True)) == [
        ("ext:5", "extrema"), ("prod:5", "product")]


def test_open_then_tool_then_roi_tool():
    s = LayerStack()
    reconcile(s, [("src0", "dem")], [])
    assert _tree(s) == [("ds:src0", ["src:src0"])]
    reconcile(s, [("src0", "dem")], [LayerInfo(1, "src0", HOLDER)])
    assert _tree(s) == [("ds:src0", ["prod:1", "src:src0"])]
    reconcile(s, [("src0", "dem")], [LayerInfo(1, "src0", HOLDER),
                                     LayerInfo(2, "src0", WTMM, roi_id="roi0", roi_label="A")])
    assert _tree(s) == [("ds:src0", [("roig:roi0", ["chn:2", "ext:2"]), "prod:1", "src:src0"])]
    assert s.node("roig:roi0").name == "ROI A"


def test_a_second_dataset_goes_on_top_and_a_fork_goes_near_its_origin():
    s = LayerStack()
    layers = [LayerInfo(1, "src0", HOLDER, roi_id="roi0", roi_label="A")]
    reconcile(s, [("src0", "dem")], layers)
    reconcile(s, [("src0", "dem"), ("src1", "fork")], layers, near={"src:src1": "prod:1"})
    assert _tree(s) == [("ds:src0", [("roig:roi0", ["src:src1", "prod:1"]), "src:src0"])]
    reconcile(s, [("src0", "dem"), ("src1", "fork"), ("src2", "dem2")], layers)
    assert _tree(s)[0] == ("ds:src2", ["src:src2"])


def test_a_bus_is_a_stack_group_of_peer_channels_with_its_tool_rows_on_top():
    s = LayerStack()
    bus = LayerInfo(5, "src0", frozenset({"Components"}), channels=("B1", "B2", "B3"))
    reconcile(s, [("src0", "dem")], [bus])
    stack = s.node("stack:5")
    assert stack.kind == "stack" and stack.name == "Stack"
    assert [c.id for c in stack.children] == ["prod:5", "ch:5:0", "ch:5:1", "ch:5:2"]


def test_a_moved_row_keeps_its_place_and_removed_content_goes():
    s = LayerStack()
    layers = [LayerInfo(1, "src0", HOLDER)]
    reconcile(s, [("src0", "dem"), ("src1", "dem2")], layers)
    s.move("prod:1", "ds:src1", 0)
    reconcile(s, [("src0", "dem"), ("src1", "dem2")], layers)
    assert s.parent_of("prod:1").id == "ds:src1"
    dropped = reconcile(s, [("src1", "dem2")], [])
    assert dropped == 2 and s.node("ds:src0") is None and s.node("prod:1") is None


def test_chain_contracts_reads_the_contract_walk(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.chain import DeviceRef
    from dynamix.model.contracts import FIELD, chain_contracts

    register_builtin_devices()
    assert chain_contracts([DeviceRef("holder_measure", {})], FIELD) == {"HMap"}
    assert chain_contracts([], FIELD) == {FIELD}


def test_a_saved_roi_is_a_group_with_its_outline_on_top():
    """Saving an ROI gives it a group holding its outline row; its results go below the
    outline; a deleted ROI takes its outline and its emptied group."""
    s = LayerStack()
    rois = [("roi0", "src0", "A")]
    reconcile(s, [("src0", "dem")], [], rois=rois)
    assert _tree(s) == [("ds:src0", [("roig:roi0", ["roi:roi0"]), "src:src0"])]
    roi_layer = LayerInfo(1, "src0", HOLDER, roi_id="roi0", roi_label="A")
    reconcile(s, [("src0", "dem")], [roi_layer], rois=rois)
    assert _tree(s) == [("ds:src0", [("roig:roi0", ["roi:roi0", "prod:1"]), "src:src0"])]
    assert s.node("roi:roi0").kind == "outline"
    reconcile(s, [("src0", "dem")], [], rois=[])
    assert _tree(s) == [("ds:src0", ["src:src0"])]


def test_a_fork_near_an_roi_result_goes_below_the_outline():
    s = LayerStack()
    rois = [("roi0", "src0", "A")]
    layers = [LayerInfo(1, "src0", HOLDER, roi_id="roi0", roi_label="A")]
    reconcile(s, [("src0", "dem")], layers, rois=rois)
    reconcile(s, [("src0", "dem"), ("src1", "fork")], layers, near={"src:src1": "prod:1"},
              rois=rois)
    assert _tree(s)[0][1][0] == ("roig:roi0", ["roi:roi0", "src:src1", "prod:1"])
