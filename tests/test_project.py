# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.model.project -- the document users save and open."""
from __future__ import annotations

import json

import pytest

from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import register_device
from dynamix.model.layer import Layer
from dynamix.model.project import (AnnotationRecord, FORMAT, SCHEMA, Project, RoiRecord,
                                   SourceRef)
from dynamix.topology.codes import KINDS
from dynamix.topology.links import ObjRef


@pytest.fixture
def _registry(clean_registry, stub_transform, stub_filter):
    """Register stub devices for the test."""
    register_device(stub_transform)
    register_device(stub_filter)


@pytest.fixture
def project_with_source(_registry):
    """A project with one source, ready for layer operations."""
    p = Project(title="Test Project")
    p.add_source("/data/test.tif")
    return p


def _project():
    p = Project(title="Kermadec fabric")
    s = p.add_source("/data/bathy.tif")
    p.add_layer("bathy", s.source_id,
                Chain((DeviceRef("t", {"scale": 2}), DeviceRef("f", {"cut": 0.3}))))
    p.add_layer("bathy coarse", s.source_id, Chain((DeviceRef("t", {}),)))
    return p


def test_add_source_assigns_stable_ids(_registry):
    p = Project()
    a = p.add_source("/data/a.tif")
    b = p.add_source("/data/b.tif")
    assert a.source_id != b.source_id
    assert p.sources[a.source_id].path == "/data/a.tif"


def test_adding_the_same_path_twice_reuses_the_source(_registry):
    """Two layers over one raster must share a source, or relocating it means N edits and the
    cache keys diverge for identical data."""
    p = Project()
    a = p.add_source("/data/a.tif")
    b = p.add_source("/data/a.tif")
    assert a.source_id == b.source_id
    assert len(p.sources) == 1


def test_add_source_defaults_to_raster_kind(_registry):
    p = Project()
    a = p.add_source("/data/a.tif")
    assert a.kind == "raster"


def test_add_source_accepts_points_kind(_registry):
    p = Project()
    a = p.add_source("/data/quakes.csv", kind="points")
    assert a.kind == "points"
    assert p.sources[a.source_id].kind == "points"


def test_layers_reference_sources_by_id(_registry):
    p = _project()
    assert len({l.source_id for l in p.layers}) == 1
    assert [l.layer_id for l in p.layers] == [0, 1]


def test_add_layer_rejects_an_unknown_source_id(_registry):
    p = Project()
    with pytest.raises(KeyError, match="no source"):
        p.add_layer("x", "src_missing", Chain(()))


def test_source_ref_kind_round_trips_through_payload():
    ref = SourceRef(source_id="src0", path="/data/quakes.csv", kind="points")
    back = SourceRef.from_payload(ref.to_payload())
    assert back.kind == "points"
    assert back.to_payload() == ref.to_payload()


def test_source_ref_from_payload_tolerates_a_missing_kind_key():
    """A ``.dynamix`` file saved before ``kind`` existed has no such key at all -- it must open as
    ``"raster"``, not raise a ``KeyError``."""
    old_payload = {"source_id": "src0", "path": "/data/dem.tif"}
    ref = SourceRef.from_payload(old_payload)
    assert ref.kind == "raster"


def test_payload_round_trip_is_exact(_registry):
    p = _project()
    assert Project.from_payload(p.to_payload()).to_payload() == p.to_payload()


def test_payload_carries_format_and_schema(_registry):
    """A bare JSON blob must be identifiable as a DynamiX project without guessing."""
    payload = _project().to_payload()
    assert payload["format"] == FORMAT and payload["schema"] == SCHEMA


def test_metadata_survives_the_round_trip(_registry):
    p = _project()
    p.description = "BOEM Gulf grids, 040 wedge"
    back = Project.from_payload(p.to_payload())
    assert back.title == "Kermadec fabric"
    assert back.description == "BOEM Gulf grids, 040 wedge"


def test_validate_rejects_a_layer_whose_chain_was_assigned_directly(_registry):
    """There is no set_chain, so Phase 4 assigns layer.chain -- and add_layer's materialisation is
    bypassed. Project.validate is the check that save_project runs before touching disk."""
    p = Project()
    s = p.add_source("/data/a.tif")
    layer = p.add_layer("x", s.source_id, Chain(()))
    layer.chain = Chain((DeviceRef("nope", {}),))
    with pytest.raises(KeyError, match="no device"):
        p.validate()


def test_validate_names_the_offending_layer(_registry):
    """With twenty layers, 'no device named x' alone does not tell you where to look."""
    p = Project()
    s = p.add_source("/data/a.tif")
    p.add_layer("fine", s.source_id, Chain(()))
    p.layers.append(Layer(layer_id=7, name="broken", source_id=s.source_id,
                          chain=Chain((DeviceRef("t", {"scale": 99}),))))
    with pytest.raises(ValueError, match="layer 7 'broken'"):
        p.validate()


def test_validate_accepts_a_healthy_project_and_returns_self(_registry):
    p = _project()
    assert p.validate() is p


def test_layer_visibility_and_tags_survive(_registry):
    p = Project()
    s = p.add_source("/data/x.tif")
    p.add_layer("x", s.source_id, Chain(()), visible=False, tags={"campaign": "2026"})
    back = Project.from_payload(p.to_payload())
    assert back.layers[0].visible is False
    assert back.layers[0].tags == {"campaign": "2026"}


def test_remove_layer_cascades_to_children(project_with_source):
    p = project_with_source
    src_id = next(iter(p.sources.keys()))
    parent = p.add_layer("a", src_id)
    child = p.add_layer("b", src_id, parent_id=parent.layer_id)
    removed = p.remove_layer(parent.layer_id)
    assert removed == [child.layer_id, parent.layer_id]
    assert all(l.layer_id not in (parent.layer_id, child.layer_id) for l in p.layers)


def test_remove_layer_unknown_raises(project_with_source):
    with pytest.raises(KeyError):
        project_with_source.remove_layer(999)


def test_remove_layer_no_cascade_raises_when_children_exist(project_with_source):
    p = project_with_source
    src_id = next(iter(p.sources.keys()))
    parent = p.add_layer("a", src_id)
    child = p.add_layer("b", src_id, parent_id=parent.layer_id)
    with pytest.raises(ValueError):
        p.remove_layer(parent.layer_id, cascade=False)


def test_source_collapsed_round_trips(project_with_source):
    p = project_with_source
    next(iter(p.sources.values())).collapsed = True
    q = type(p).from_payload(p.to_payload())
    assert next(iter(q.sources.values())).collapsed is True


def test_source_collapsed_defaults_false_on_old_payloads(project_with_source):
    p = project_with_source
    payload = p.to_payload()
    # Simulate an old payload that doesn't have collapsed field
    payload["sources"][0].pop("collapsed", None)
    q = type(p).from_payload(payload)
    assert next(iter(q.sources.values())).collapsed is False


# -- ROI records ---------------------------------------
# Both forms must exist: an ROI drawn FOR one layer (``layer_ids`` names it) and a GENERAL ROI
# (empty ``layer_ids``) applicable to any layer. ``Layer.roi_id`` is additive -- a file saved
# before it existed has no such key and must open with ``roi_id is None``.


def test_layer_roi_id_defaults_to_none_and_round_trips(_registry):
    p = Project()
    s = p.add_source("/data/x.tif")
    layer = p.add_layer("x", s.source_id)
    assert layer.roi_id is None

    layer.roi_id = "roi0"
    back = Project.from_payload(p.to_payload())
    assert back.layers[0].roi_id == "roi0"


def test_layer_from_payload_tolerates_a_missing_roi_id_key():
    """A layer written before ``roi_id`` existed has no such key -- it opens as ``None``."""
    old_payload = {"layer_id": 0, "name": "dem", "source_id": "src0"}
    assert Layer.from_payload(old_payload).roi_id is None


def test_add_roi_mints_monotonic_ids(_registry):
    p = Project()
    a = p.add_roi(10, 20, 30, 40)
    b = p.add_roi(1, 2, 3, 4)
    assert (a.roi_id, b.roi_id) == ("roi0", "roi1")
    assert [r.roi_id for r in p.rois] == ["roi0", "roi1"]

    # The counter is re-derived on load, so the next mint never collides with a saved id.
    back = Project.from_payload(p.to_payload())
    assert back.add_roi(5, 6, 7, 8).roi_id == "roi2"


def test_a_general_roi_has_no_layer_ids_and_a_drawn_roi_names_one(_registry):
    p = Project()
    s = p.add_source("/data/x.tif")
    layer = p.add_layer("x", s.source_id)

    general = p.add_roi(0, 0, 64, 64, label="the 040 wedge")
    drawn = p.add_roi(8, 9, 16, 16, layer_ids=(layer.layer_id,))

    assert general.layer_ids == ()          # applies to any layer
    assert general.label == "the 040 wedge"
    assert drawn.layer_ids == (layer.layer_id,)

    back = Project.from_payload(p.to_payload())
    assert back.rois[0].layer_ids == ()
    assert back.rois[1].layer_ids == (layer.layer_id,)


def test_roi_record_round_trip_is_exact():
    r = RoiRecord(roi_id="roi7", row=1200, col=3400, h=256, w=512,
                  layer_ids=(3, 5), label="wedge", visible=False)
    back = RoiRecord.from_payload(r.to_payload())
    assert back == r                                    # tuples re-tupled, not left as lists
    assert back.to_payload() == r.to_payload()


# -- camera memory --------------------------------------
# G2: ``Scene._camera_memory`` dies with the process. Keys are ``Scene.camera_key()`` strings,
# values the ``Scene._camera_snapshot()`` shape. Per PROJECT, not per machine: a
# saved viewpoint is closer to a measurement than to ``Settings.center_view``.

_SNAPSHOT = {
    "position": (1.0, 2.0, 3.0),
    "focal_point": (0.0, 0.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "parallel_scale": 42.5,
    "parallel_projection": True,
    "clipping_range": (0.5, 500.0),
}
_KEY = "frame:0.00,0.00,64.00,64.00"


def test_cameras_defaults_to_empty_dict():
    assert Project().cameras == {}


def test_cameras_round_trip_preserves_tuples():
    p = Project()
    p.cameras[_KEY] = dict(_SNAPSHOT)

    payload = p.to_payload()
    assert payload["cameras"][_KEY]["position"] == [1.0, 2.0, 3.0]   # JSON lists on the way out
    json.dumps(payload)                                              # and the blob is JSON-safe

    back = Project.from_payload(payload)
    assert back.cameras == p.cameras                                 # re-tupled on the way in
    assert back.cameras[_KEY]["position"] == (1.0, 2.0, 3.0)
    assert back.cameras[_KEY]["clipping_range"] == (0.5, 500.0)
    assert back.cameras[_KEY]["parallel_scale"] == 42.5
    assert back.cameras[_KEY]["parallel_projection"] is True
    assert back.to_payload()["cameras"] == payload["cameras"]        # and stable across a cycle


def test_project_from_payload_without_cameras_key_opens_with_an_empty_dict():
    """A payload written before camera memory existed has no "cameras" key at all."""
    p = Project.from_payload({"format": FORMAT, "schema": SCHEMA, "title": "old"})
    assert p.cameras == {}
    assert p.title == "old"                     # the rest of the legacy payload still loads


# -- annotations ----------------------------------------
# G3: a note pinned to a place in the world is a research artifact and belongs in the project
# document. Point-pinned first, but the TARGET is an ``ObjRef``-style reference of ANY
# dimension -- ``codes.KINDS`` is the whole vocabulary, so a label attaches to a point, a chain
# ("line") or a whole hierarchy of chains ("surface"/"body") alike.


def test_annotation_with_a_point_only_round_trips(_registry):
    p = Project()
    note = p.add_annotation("the 040 wedge starts here", point=(12.5, 33.0))

    assert note.annotation_id == 0
    assert note.point == (12.5, 33.0)
    assert note.target is None
    assert note.visible is True

    payload = p.to_payload()
    json.dumps(payload)                                   # the blob stays JSON-safe
    assert payload["annotations"][0]["point"] == [12.5, 33.0]
    assert payload["annotations"][0]["target"] is None

    back = Project.from_payload(payload)
    (restored,) = back.annotations
    assert restored == note                               # re-tupled, not left as a list
    assert restored.point == (12.5, 33.0)


def test_annotation_targeting_a_line_objref_round_trips(_registry):
    p = Project()
    s = p.add_source("/data/x.tif")
    layer = p.add_layer("wtmm of x", s.source_id)
    ref = ObjRef(layer_id=layer.layer_id, transform="mz_edges", kind="line", obj_id=7)
    note = p.add_annotation("chain 7 is the fault trace", target=ref)

    payload = p.to_payload()
    json.dumps(payload)
    assert payload["annotations"][0]["target"] == {
        "layer_id": layer.layer_id, "transform": "mz_edges", "kind": "line", "obj_id": 7}
    assert payload["annotations"][0]["point"] is None

    back = Project.from_payload(payload)
    (restored,) = back.annotations
    assert restored.target == ref
    assert restored.target.node_id() == ref.node_id()     # the identity string is unchanged
    assert restored.text == note.text


@pytest.mark.parametrize("kind", KINDS)
def test_annotation_target_accepts_every_topology_kind(kind):
    """"A feature of ANY dimension", asserted rather than asserted-in-prose."""
    ref = ObjRef(layer_id=2, transform="mz_edges", kind=kind, obj_id=7)
    note = AnnotationRecord(annotation_id=0, text="here", target=ref)

    back = AnnotationRecord.from_payload(note.to_payload())
    assert back.target.kind == kind
    assert back == note


def test_annotation_with_an_illegal_target_kind_raises():
    """``ObjRef``'s own gate (``links.py:69-73``) is inherited, never bypassed by the payload."""
    with pytest.raises(ValueError, match="not a topology kind"):
        ObjRef(layer_id=2, transform="mz_edges", kind="blob", obj_id=7)

    with pytest.raises(ValueError, match="not a topology kind"):
        AnnotationRecord.from_payload({
            "annotation_id": 0, "text": "here", "point": None,
            "target": {"layer_id": 2, "transform": "mz_edges", "kind": "blob", "obj_id": 7}})


def test_annotation_ids_are_monotonic_and_survive_a_delete(_registry):
    p = Project()
    a = p.add_annotation("a", point=(0.0, 0.0))
    b = p.add_annotation("b", point=(1.0, 1.0))
    c = p.add_annotation("c", point=(2.0, 2.0))
    assert [n.annotation_id for n in (a, b, c)] == [0, 1, 2]

    p.annotations.remove(b)                               # the collision lesson: no reuse
    assert p.add_annotation("d", point=(3.0, 3.0)).annotation_id == 3

    # The counter is re-derived on load, so the next mint never collides with a saved id.
    back = Project.from_payload(p.to_payload())
    assert back.add_annotation("e", point=(4.0, 4.0)).annotation_id == 4


def test_annotation_with_neither_point_nor_target_is_refused():
    with pytest.raises(ValueError, match="point"):
        AnnotationRecord(annotation_id=0, text="floating")

    with pytest.raises(ValueError, match="target"):
        Project().add_annotation("floating")


def test_project_from_payload_without_annotations_key_opens_with_an_empty_list():
    """A payload written before annotations existed has no "annotations" key at all."""
    p = Project.from_payload({"format": FORMAT, "schema": SCHEMA, "title": "old"})
    assert p.annotations == []
    assert p.title == "old"                     # the rest of the legacy payload still loads


# ------------------------------------------------------------ reference layers (2026-08-29)

def test_reference_layer_records_are_additive_and_round_trip(_registry):
    from dynamix.model.project import ReferenceLayerRecord
    p = Project()
    r = p.add_reference_layer("/data/anomaly_slumps.shp", name="anomaly_slumps", color="#ff8800")
    assert isinstance(r, ReferenceLayerRecord) and r.ref_id == "ref0" and r.visible is True
    r2 = p.add_reference_layer("/data/seep_positives.shp")
    assert r2.ref_id == "ref1" and r2.name == "seep_positives" and r2.color != r.color
    payload = p.to_payload()
    assert [x["path"] for x in payload["reference_layers"]] == ["/data/anomaly_slumps.shp", "/data/seep_positives.shp"]
    q = Project.from_payload(payload)
    assert [x.ref_id for x in q.reference_layers] == ["ref0", "ref1"] and q.reference_layers[0].color == "#ff8800"
    assert q.add_reference_layer("/data/x.shp").ref_id == "ref2"          # counter re-derived


def test_a_payload_without_reference_layers_opens_with_an_empty_list(_registry):
    p = Project()
    payload = p.to_payload(); payload.pop("reference_layers", None)
    assert Project.from_payload(payload).reference_layers == []


def test_source_hidden_flag_round_trips_and_defaults_false():
    # 2026-08-29: hide the DATASET (raster) from the source header row, products stay.
    from dynamix.model.project import SourceRef
    s = SourceRef(source_id="s0", path="/x.tif", hidden=True)
    assert SourceRef.from_payload(s.to_payload()).hidden is True
    assert SourceRef.from_payload({"source_id": "s0", "path": "/x.tif"}).hidden is False


def test_source_kind_vocabulary_covers_the_typed_source_model():
    """Raster / points / vector /
    object are the kinds the panel will type rows by; the model owns the vocabulary ahead of
    the layout mockups, and every kind round-trips through the payload."""
    from dynamix.model.project import SOURCE_KINDS, SourceRef

    assert SOURCE_KINDS == ("raster", "points", "vector", "object")
    for kind in SOURCE_KINDS:
        ref = SourceRef(source_id="s", path="/x", kind=kind)
        assert SourceRef.from_payload(ref.to_payload()).kind == kind


def test_a_saved_roi_names_its_source_and_old_payloads_load_without_one(_registry):
    """A saved ROI belongs to ONE source (a project can hold BOEM West and
    East); additive -- a payload saved before the key existed loads with ``None``."""
    p = Project()
    s = p.add_source("/data/west.tif")
    r = p.add_roi(10, 20, 30, 40, source_id=s.source_id, label="A")
    assert r.source_id == s.source_id
    back = Project.from_payload(p.to_payload())
    assert back.rois[0].source_id == s.source_id
    legacy = r.to_payload()
    del legacy["source_id"]
    assert RoiRecord.from_payload(legacy).source_id is None


def test_derivative_flags_round_trip_and_default_off():
    """A derivative dataset may be TEMPORARY (this session only) and may show nested inside the
    dataset it came from; both additive, so older payloads load with neither."""
    from dynamix.model.project import SourceRef

    s = SourceRef(source_id="s0", path="/d.npz", temporary=True, nest_under="s9")
    back = SourceRef.from_payload(s.to_payload())
    assert back.temporary is True and back.nest_under == "s9"
    old = SourceRef.from_payload({"source_id": "s0", "path": "/x.tif"})
    assert old.temporary is False and old.nest_under is None


def test_a_saved_project_leaves_temporary_datasets_out(_registry):
    p = Project()
    keep = p.add_source("/data/a.tif")
    temp = p.add_source("/tmp/derived.npz")
    temp.temporary = True
    p.add_layer("a", keep.source_id)
    d = p.add_layer("derived", temp.source_id)
    p.add_layer("derived child", temp.source_id, parent_id=d.layer_id)
    p.add_roi(0, 0, 4, 4, source_id=temp.source_id)
    p.add_roi(1, 1, 4, 4, source_id=keep.source_id)
    payload = p.to_payload()
    assert [s["source_id"] for s in payload["sources"]] == [keep.source_id]
    assert [l["name"] for l in payload["layers"]] == ["a"]
    assert [r["source_id"] for r in payload["rois"]] == [keep.source_id]
    assert temp.source_id in p.sources and len(p.layers) == 3      # the live project is intact
    assert p.temporary_sources() == [temp]
