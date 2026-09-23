# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.model.provenance -- the tree the per-source inspector renders.

The ordering IS the claim: sources in insertion order, layers in project order, a
layer's products in its chain's own step order, an ROI child nested under the layer it was drawn
on. Every assertion below is about ORDER or NESTING, because that is what "provenance is reflected in the hierarchy" means when it is executable rather than prose.
"""
from __future__ import annotations

import pytest

from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import register_device
from dynamix.model.layer import Layer
from dynamix.model.project import Project
from dynamix.model.provenance import (LayerNode, ProductNode, SourceNode, provenance_tree,
                                      raster_key, sublayer_key)


@pytest.fixture
def _registry(clean_registry, stub_transform, stub_filter):
    """Register stub devices for the test (same fixture shape as tests/test_project.py)."""
    register_device(stub_transform)
    register_device(stub_filter)


def _wtmm_chain() -> Chain:
    """Three steps -- two transforms then a filter, the legal transforms-then-filters shape."""
    return Chain((DeviceRef("t", {"scale": 2}), DeviceRef("t", {"scale": 4}),
                  DeviceRef("f", {"cut": 0.3})))


def _flatten(tree) -> list[tuple[int, str]]:
    """(depth, label) rows, in the literal order the inspector's tree will show them."""
    rows: list[tuple[int, str]] = []

    def walk_layer(node, depth):
        rows.append((depth, node.name))
        for product in node.products:
            rows.append((depth + 1, product.device))
        for child in node.children:
            walk_layer(child, depth + 1)

    for source in tree:
        rows.append((0, source.label))
        for layer in source.layers:
            walk_layer(layer, 1)
    return rows


def test_empty_project_yields_an_empty_tree():
    assert provenance_tree(Project()) == []


def test_tree_groups_layers_under_their_source(_registry):
    p = Project()
    a = p.add_source("/data/bathy.tif")
    b = p.add_source("/data/mag.tif")
    p.add_layer("bathy", a.source_id, _wtmm_chain())
    p.add_layer("mag", b.source_id, Chain((DeviceRef("t", {}),)))
    p.add_layer("bathy coarse", a.source_id, Chain())

    tree = provenance_tree(p)
    assert [s.source_id for s in tree] == [a.source_id, b.source_id]
    assert [n.name for n in tree[0].layers] == ["bathy", "bathy coarse"]
    assert [n.name for n in tree[1].layers] == ["mag"]
    assert isinstance(tree[0], SourceNode) and isinstance(tree[0].layers[0], LayerNode)


def test_products_are_in_chain_step_order(_registry):
    p = Project()
    s = p.add_source("/data/bathy.tif")
    layer = p.add_layer("bathy", s.source_id, _wtmm_chain())

    node = provenance_tree(p)[0].layers[0]
    assert [x.device for x in node.products] == ["t", "t", "f"]
    assert [x.index for x in node.products] == [0, 1, 2]
    assert [x.key for x in node.products] == [sublayer_key(layer.layer_id, i) for i in range(3)]
    assert isinstance(node.products[0], ProductNode)


def test_roi_child_nests_under_its_parent_layer_not_the_source(_registry):
    p = Project()
    s = p.add_source("/data/bathy.tif")
    parent = p.add_layer("bathy", s.source_id, _wtmm_chain())
    roi = p.add_roi(10, 20, 30, 40, layer_ids=(parent.layer_id,))
    child = p.add_layer("bathy ROI", s.source_id, Chain((DeviceRef("t", {}),)),
                        parent_id=parent.layer_id)
    child.roi_id = roi.roi_id

    source_node = provenance_tree(p)[0]
    assert [n.layer_id for n in source_node.layers] == [parent.layer_id]
    assert [n.layer_id for n in source_node.layers[0].children] == [child.layer_id]
    assert source_node.layers[0].children[0].roi_id == roi.roi_id
    assert source_node.layers[0].roi_id is None


def test_a_grandchild_roi_nests_two_deep(_registry):
    p = Project()
    s = p.add_source("/data/bathy.tif")
    parent = p.add_layer("bathy", s.source_id, Chain((DeviceRef("t", {}),)))
    child = p.add_layer("roi", s.source_id, Chain(), parent_id=parent.layer_id)
    grand = p.add_layer("roi of roi", s.source_id, Chain(), parent_id=child.layer_id)

    source_node = provenance_tree(p)[0]
    assert [n.layer_id for n in source_node.layers] == [parent.layer_id]
    kid = source_node.layers[0].children[0]
    assert kid.layer_id == child.layer_id
    assert [n.layer_id for n in kid.children] == [grand.layer_id]
    assert _flatten(provenance_tree(p)) == [
        (0, "bathy"),
        (1, "bathy"), (2, "t"),
        (2, "roi"),
        (3, "roi of roi"),
    ]


def test_row_order_is_source_then_layer_then_products_then_roi_child(_registry):
    """The acceptance criterion, asserted as the literal row order."""
    p = Project()
    s = p.add_source("/data/bathy.tif")
    parent = p.add_layer("bathy", s.source_id, _wtmm_chain())
    p.add_layer("bathy ROI", s.source_id, Chain((DeviceRef("f", {}),)),
                parent_id=parent.layer_id)

    tree = provenance_tree(p)
    assert len(tree) == 1
    assert _flatten(tree) == [
        (0, "bathy"),
        (1, "bathy"), (2, "t"), (2, "t"), (2, "f"),
        (2, "bathy ROI"), (3, "f"),
    ]


def test_tree_builds_with_an_empty_device_registry(clean_registry):
    """Registry-free by construction: provenance reads ``chain.steps``' device STRINGS, never
    ``Chain.transforms``/``filters``, which resolve through the registry and raise wherever a
    device is not registered -- which is every headless caller that never imported the devices."""
    p = Project()
    s = p.add_source("/data/bathy.tif")
    chain = Chain((DeviceRef("wtmm", {}), DeviceRef("skeleton", {})))
    p.layers.append(Layer(layer_id=0, name="bathy", source_id=s.source_id, chain=chain))

    with pytest.raises(KeyError):
        chain.transforms                                   # the path provenance must NOT take

    node = provenance_tree(p)[0].layers[0]
    assert [x.device for x in node.products] == ["wtmm", "skeleton"]


def test_sublayer_keys_are_unique_across_the_whole_tree(_registry):
    p = Project()
    a = p.add_source("/data/bathy.tif")
    b = p.add_source("/data/mag.tif")
    first = p.add_layer("bathy", a.source_id, _wtmm_chain())
    p.add_layer("bathy ROI", a.source_id, _wtmm_chain(), parent_id=first.layer_id)
    p.add_layer("mag", b.source_id, _wtmm_chain())

    keys: list[str] = []

    def walk(node):
        keys.extend(x.key for x in node.products)
        for child in node.children:
            walk(child)

    tree = provenance_tree(p)
    for source in tree:
        keys.append(raster_key(source.source_id))
        for layer in source.layers:
            walk(layer)

    assert len(keys) == 11                                  # 2 rasters + 3 layers x 3 products
    assert len(set(keys)) == len(keys)
