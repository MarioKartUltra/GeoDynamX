# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for Project.topologies -- TopologyModel as a first-class .dynamix citizen, plus
provenance(project), the derivation-edge paper trail read off a project's layers."""
from __future__ import annotations

import pytest

from dynamix.devices import register_builtin_devices
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.project import Project
from dynamix.model.projectfile import open_project, save_project
from dynamix.topology.codes import LINE, POINT, point_code
from dynamix.topology.model import TopologyModel, provenance


@pytest.fixture
def _registry(clean_registry):
    """Register the real built-in devices -- provenance's test exercises the actual wtmm2d
    device so its ``via`` string reflects a real chain, not a stub."""
    register_builtin_devices()


def test_hand_asserted_topology_survives_dynamix_round_trip(tmp_path):
    """A geologist's asserted structure, zero analysis attached, survives save/open."""
    m = TopologyModel()
    m.add_node("fault", LINE, space_dim=2, anchor=("asserted", "fault-1"))
    m.add_node("piercing", POINT, space_dim=2, anchor=("asserted", "pp-1"))
    m.assert_spatial(point_code(at="boundary"), "piercing", "fault", space_dim=2)

    p = Project(title="topology only")
    p.add_topology(m)
    path = tmp_path / "t.dynamix"
    save_project(p, path)
    p2 = open_project(path)
    assert len(p2.topologies) == 1
    assert p2.topologies[0].to_payload() == m.to_payload()


def test_project_without_topologies_still_opens(tmp_path):
    p = Project(title="plain")
    path = tmp_path / "p.dynamix"
    save_project(p, path)
    assert open_project(path).topologies == []


def test_from_payload_predating_topologies_opens_with_an_empty_list(_registry):
    """A project file written before topologies existed has no "topologies" key at all."""
    p = Project.from_payload({
        "format": "dynamix-project", "schema": 1, "title": "old",
        "sources": [{"source_id": "src0", "path": "/tmp/dem.npz", "sha256": None,
                     "label": ""}],
        "layers": [{"layer_id": 0, "name": "wtmm of dem", "source_id": "src0",
                    "chain": {"steps": [{"device": "wtmm2d", "params": {"n_oct": 3}}]},
                    "visible": True, "tags": {}}],
    })
    assert p.topologies == []
    assert p.layers[0].name == "wtmm of dem"       # the rest of the legacy payload still loads


def test_provenance_survives_a_non_json_param_value():
    """provenance is duck-typed and takes params as it finds them: a device's params reach it
    unconverted, and a numpy scalar is not JSON-serialisable."""
    import types

    import numpy as np

    layer = types.SimpleNamespace(
        layer_id=0, source_id="src0",
        chain=types.SimpleNamespace(
            steps=(types.SimpleNamespace(device="wtmm2d", params={"n_oct": np.int64(3)}),)))
    project = types.SimpleNamespace(sources={"src0": None}, layers=[layer])

    (edge,) = provenance(project).edges_of("layer:0", family="derivation")
    assert "wtmm2d" in edge.via and "3" in edge.via


def test_provenance_reads_the_paper_trail_off_the_project(tmp_path, _registry):
    """'Dataset 2 is dataset 1 through these operations', as derivation edges."""
    p = Project(title="trail")
    src = p.add_source(str(tmp_path / "dem.npz"))
    layer = p.add_layer("wtmm of dem", src.source_id,
                        Chain((DeviceRef("wtmm2d", {"n_oct": 3}),)))
    m = provenance(p)
    src_node = f"src:{src.source_id}"
    layer_node = f"layer:{layer.layer_id}"
    assert m.nodes[src_node].kind == "dataset"
    assert m.nodes[layer_node].kind == "dataset"
    (edge,) = m.edges_of(layer_node, family="derivation")
    assert edge.a == layer_node and edge.b == src_node
    assert "wtmm2d" in edge.via and "n_oct" in edge.via
