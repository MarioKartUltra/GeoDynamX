# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for Layer.parent_id -- grouped layers in the model.

The ROI lifecycle is a new layer grouped under the layer it targets (the Ableton group-track
model). parent_id is the field that records that relationship; None means top-level.
"""
from __future__ import annotations

import pytest

from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import register_device
from dynamix.model.layer import Layer
from dynamix.model.project import Project
from dynamix.model.projectfile import open_project, save_project


@pytest.fixture
def _registry(clean_registry, stub_transform):
    register_device(stub_transform)


def test_layer_defaults_to_no_parent():
    layer = Layer(layer_id=0, name="x", source_id="src0")
    assert layer.parent_id is None


def test_payload_round_trip_with_parent_id():
    layer = Layer(layer_id=1, name="roi", source_id="src0", parent_id=0)
    back = Layer.from_payload(layer.to_payload())
    assert back.parent_id == 0
    assert back.to_payload() == layer.to_payload()


def test_payload_round_trip_without_parent_id():
    layer = Layer(layer_id=0, name="bathy", source_id="src0")
    back = Layer.from_payload(layer.to_payload())
    assert back.parent_id is None
    assert back.to_payload() == layer.to_payload()


def test_to_payload_writes_parent_id_key():
    layer = Layer(layer_id=1, name="roi", source_id="src0", parent_id=0)
    assert layer.to_payload()["parent_id"] == 0


def test_legacy_payload_without_the_key_loads_as_none():
    """A payload saved before this field existed must load unchanged."""
    legacy = {"layer_id": 0, "name": "bathy", "source_id": "src0", "chain": {}}
    layer = Layer.from_payload(legacy)
    assert layer.parent_id is None


def test_add_layer_stores_parent_id(_registry):
    p = Project()
    s = p.add_source("/data/bathy.tif")
    parent = p.add_layer("bathy", s.source_id, Chain((DeviceRef("t", {"scale": 2}),)))
    roi = p.add_layer("bathy ROI", s.source_id, parent_id=parent.layer_id)
    assert roi.parent_id == parent.layer_id


def test_add_layer_defaults_parent_id_to_none(_registry):
    p = Project()
    s = p.add_source("/data/bathy.tif")
    layer = p.add_layer("bathy", s.source_id)
    assert layer.parent_id is None


def test_add_layer_accepts_parent_id_zero(_registry):
    """0 is a valid layer_id and must not be treated as falsy/missing."""
    p = Project()
    s = p.add_source("/data/bathy.tif")
    p.add_layer("bathy", s.source_id)  # layer_id 0
    roi = p.add_layer("bathy ROI", s.source_id, parent_id=0)
    assert roi.parent_id == 0


def test_project_payload_round_trip_preserves_grouping(_registry):
    p = Project()
    s = p.add_source("/data/bathy.tif")
    parent = p.add_layer("bathy", s.source_id, Chain((DeviceRef("t", {"scale": 2}),)))
    p.add_layer("bathy ROI", s.source_id, parent_id=parent.layer_id)
    back = Project.from_payload(p.to_payload())
    assert back.layers[0].parent_id is None
    assert back.layers[1].parent_id == parent.layer_id


def test_save_and_open_project_preserves_grouping(tmp_path, _registry):
    data = tmp_path / "data"
    data.mkdir()
    raster = data / "dem.tif"
    raster.write_bytes(b"not really a geotiff, but it hashes")
    p = Project(title="T")
    s = p.add_source(str(raster))
    parent = p.add_layer("dem", s.source_id, Chain((DeviceRef("t", {"scale": 2}),)))
    p.add_layer("dem ROI", s.source_id, parent_id=parent.layer_id)

    path = tmp_path / "proj.dynamix"
    save_project(p, path)
    back = open_project(path)

    assert back.layers[0].parent_id is None
    assert back.layers[1].parent_id == parent.layer_id
