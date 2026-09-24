# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.point_import -- CSV point import, the suffix glue, and the shell-side
tolerances a point layer (no raster field, empty chain) needs from ``MainWindow``/``WorkflowZone``/
``LayerPanel``.

Offscreen Qt, same convention as the rest of the shell suite. ``window``/``loaded``/``stub_devices``
come from ``test_shell_window.py`` (the established cross-file precedent -- see
``test_right_panel.py``'s own docstring for why: no separate parallel fixtures).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PySide6 import QtWidgets

from dynamix.core.pointset import PointSet, read_csv_points
from dynamix.model.chain import Chain
from dynamix.model.project import Project
from dynamix.model.projectfile import open_project, save_project
from dynamix.shell import point_import
from dynamix.shell.layer_panel import LayerPanel
from dynamix.shell.main_window import MainWindow
from dynamix.shell.point_import import load_points
from dynamix.shell.workflow_zone import WorkflowZone
from tests.test_shell_window import _FIELD, loaded, stub_devices, window  # noqa: F401

_CSV = "lon,lat,depth,mag\n1.0,2.0,3.0,4.0\n5.0,6.0,7.0,8.0\n"
_AMBIGUOUS_CSV = "x,y,z\n1.0,2.0,3.0\n4.0,5.0,6.0\n"


def _write_csv(tmp_path, text: str = _CSV, name: str = "quakes.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


# --------------------------------------------------------------------------- load_points


def test_load_points_auto_detects_and_registers_a_points_source(qtbot, window, tmp_path,
                                                                 monkeypatch):
    def _boom(headers):
        raise AssertionError("auto-detection succeeded; the dialog seam must not be touched")

    monkeypatch.setattr(point_import, "_ask_mapping", _boom)
    path = _write_csv(tmp_path)

    load_points(window, str(path))

    assert len(window.project.sources) == 1
    source = next(iter(window.project.sources.values()))
    assert source.kind == "points"
    assert source.path == str(path)

    assert len(window.project.layers) == 1
    layer = window.project.layers[0]
    assert layer.chain.steps == ()                     # point layers start chainless
    assert layer.name == "quakes"


def test_load_points_registers_the_pointset_in_the_fields_registry(qtbot, window, tmp_path,
                                                                    monkeypatch):
    """The device receives the PointSet through ``resolve()``'s field argument -- which
    means it has to already be sitting in ``self._fields[layer_id]``, exactly like a raster
    field."""
    monkeypatch.setattr(point_import, "_ask_mapping",
                        lambda headers: pytest.fail("dialog touched"))
    path = _write_csv(tmp_path)
    load_points(window, str(path))
    layer = window.project.layers[0]
    field = window._fields[layer.layer_id]
    assert isinstance(field, PointSet)
    assert field.lon.shape == (2,)


def test_load_points_selects_the_new_layer_without_crashing_the_canvas(qtbot, window, tmp_path,
                                                                        monkeypatch):
    """The whole point of ``MainWindow._select_layer``'s point-layer tolerance: selecting a point
    layer must not blow up on ``Canvas.set_field`` (a PointSet is not array-like)."""
    monkeypatch.setattr(point_import, "_ask_mapping",
                        lambda headers: pytest.fail("dialog touched"))
    path = _write_csv(tmp_path)
    load_points(window, str(path))          # raises if _select_layer chokes on the PointSet
    layer = window.project.layers[0]
    assert window.layer is layer
    assert window.layer_list.current_layer_id() == layer.layer_id


def test_load_points_stores_the_resolved_mapping_as_json_tags(qtbot, window, tmp_path,
                                                               monkeypatch):
    monkeypatch.setattr(point_import, "_ask_mapping",
                        lambda headers: pytest.fail("dialog touched"))
    path = _write_csv(tmp_path)
    load_points(window, str(path))
    layer = window.project.layers[0]
    mapping = json.loads(layer.tags["points.mapping"])
    assert mapping == {"lon": "lon", "lat": "lat", "depth": "depth", "mag": "mag"}


def test_load_points_opens_the_mapping_dialog_only_when_ambiguous(qtbot, window, tmp_path,
                                                                   monkeypatch):
    path = _write_csv(tmp_path, _AMBIGUOUS_CSV)
    calls = []

    def _fake_ask(headers):
        calls.append(list(headers))
        return {"lon": "x", "lat": "y", "depth": "z"}

    monkeypatch.setattr(point_import, "_ask_mapping", _fake_ask)
    load_points(window, str(path))

    assert calls == [["x", "y", "z"]]
    layer = window.project.layers[0]
    assert json.loads(layer.tags["points.mapping"]) == {"lon": "x", "lat": "y", "depth": "z"}
    field = window._fields[layer.layer_id]
    assert field.lon.tolist() == [1.0, 4.0]


def test_load_points_cancelled_dialog_adds_nothing(qtbot, window, tmp_path, monkeypatch):
    path = _write_csv(tmp_path, _AMBIGUOUS_CSV)
    monkeypatch.setattr(point_import, "_ask_mapping", lambda headers: None)

    load_points(window, str(path))

    assert window.project.sources == {}
    assert window.project.layers == []


# --------------------------------------------------------------------------- open_path suffix glue


def test_open_path_dispatches_csv_to_load_points(qtbot, window, tmp_path, monkeypatch):
    path = _write_csv(tmp_path)
    monkeypatch.setattr(point_import, "_ask_mapping",
                        lambda headers: pytest.fail("dialog touched"))

    window.open_path(str(path))

    assert len(window.project.sources) == 1
    assert next(iter(window.project.sources.values())).kind == "points"


def test_open_button_filter_string_includes_csv(qtbot, window, monkeypatch):
    captured = {}

    def fake_get_open_file_name(*args, **kwargs):
        captured["filter"] = args[3] if len(args) > 3 else kwargs.get("filter")
        return "", ""

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", fake_get_open_file_name)
    window._on_open_clicked()
    assert "*.csv" in captured["filter"]


# --------------------------------------------------------------------------- resolve/worker safety


def test_selecting_a_chainless_point_layer_resolves_safely(qtbot, window, tmp_path, monkeypatch):
    """Point layers start chainless -- the worker still gets dispatched (the ordinary
    ``_select_layer`` path), but ``resolve()`` over an empty chain is a cheap no-op: this must
    land through ``resolved``, not ``errored``, with an empty result."""
    monkeypatch.setattr(point_import, "_ask_mapping",
                        lambda headers: pytest.fail("dialog touched"))
    path = _write_csv(tmp_path)

    errors = []
    window.errored.connect(errors.append)
    with qtbot.waitSignal(window.resolved, timeout=10000):
        load_points(window, str(path))

    assert errors == []


# --------------------------------------------------------------------------- WorkflowZone: field=None


def test_workflow_zone_set_steps_accepts_a_pointset_field_with_no_descriptors(qtbot):
    """A point layer opens with an empty chain -- ``WorkflowZone.set_steps([], field=<PointSet>)``
    builds no ``DeviceBox`` at all, so the field is never dereferenced as an array. Pinning this
    directly (not only through the full ``MainWindow`` path above) documents that the tolerance is
    structural, not incidental."""
    zone = WorkflowZone()
    qtbot.addWidget(zone)
    pset = PointSet(lon=np.array([1.0]), lat=np.array([2.0]))
    zone.set_steps([], field=pset)          # must not raise
    assert zone._descriptors == []


# --------------------------------------------------------------------------- LayerPanel: kind-agnostic


def test_layer_panel_groups_a_point_layer_under_its_own_source_header():
    """layer_panel.py's grouping (``_ensure_source_header``) reads only the source's file stem --
    it never inspects ``SourceRef.kind`` at all, so a point layer's row lands under its own header
    exactly like a raster's. Pinned directly against a bare ``LayerPanel``, no ``MainWindow``."""
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    project = Project(title="t")
    source = project.add_source("/data/quakes.csv", kind="points")
    layer = project.add_layer("quakes", source.source_id, Chain())
    panel = LayerPanel(project)
    panel.add_layer_row(layer, None)

    assert panel.topLevelItemCount() == 1
    header = panel.topLevelItem(0)
    assert header.text(0) == "quakes"
    # The dataset's first layer IS its row (2026-09-23) -- no second "quakes" row under it.
    assert header.childCount() == 0
    assert panel._layer_items[layer.layer_id] is header
    assert panel.count() == 1


# --------------------------------------------------------------------------- project round trip


def test_project_round_trip_reopens_a_point_source_without_the_dialog(qtbot, window, tmp_path,
                                                                       monkeypatch):
    """Save a project with one raster layer + one CSV point layer, reopen it, and confirm the
    point layer reconstructs through its stored ``points.mapping`` tag WITHOUT ever touching the
    mapping dialog -- even for a header ``resolve_columns`` alone could not have auto-detected."""
    monkeypatch.setattr(point_import, "_ask_mapping",
                        lambda headers: pytest.fail("dialog touched during import"))
    csv_path = _write_csv(tmp_path, _AMBIGUOUS_CSV)
    # A real mapping-driven import, exercising the dialog path once (with a monkeypatch answer)
    # so the saved tag is exactly what a user's own dialog pick would have produced.
    monkeypatch.setattr(point_import, "_ask_mapping",
                        lambda headers: {"lon": "x", "lat": "y", "depth": "z"})
    with qtbot.waitSignal(window.resolved, timeout=10000):
        load_points(window, str(csv_path))

    with qtbot.waitSignal(window.resolved, timeout=10000):
        window.load_field(_FIELD, str(tmp_path / "raster.mem"))

    proj_path = tmp_path / "proj.dynamix"
    save_project(window.project, proj_path)

    # The reopen path must never construct the dialog -- fail hard if it does.
    monkeypatch.setattr(point_import, "_ask_mapping",
                        lambda headers: pytest.fail("dialog touched during reopen"))

    window._open_project_path(proj_path)

    kinds = {s.path: s.kind for s in window.project.sources.values()}
    assert kinds[str(csv_path)] == "points"

    point_layer = next(l for l in window.project.layers
                       if window.project.sources[l.source_id].kind == "points")
    field = window._fields[point_layer.layer_id]
    assert isinstance(field, PointSet)
    assert field.lon.tolist() == [1.0, 4.0]
    assert field.lat.tolist() == [2.0, 5.0]
    assert field.depth.tolist() == [3.0, 6.0]


def test_project_round_trip_restores_each_layers_own_mapping_when_sharing_a_source(
        qtbot, window, tmp_path):
    """Two layers over the SAME CSV
    (one deduped source), each carrying its OWN ``points.mapping`` tag. Before the fix,
    ``_open_project_path``'s ``fields_by_source`` was keyed by ``sid`` alone, so the SECOND
    layer's distinct mapping was silently ignored on reopen -- both layers reconstructed against
    whichever one the loop happened to encounter first, while the second layer's own tags kept
    claiming a mapping its reopened field no longer had."""
    csv_path = _write_csv(tmp_path, "lonA,latA,lonB,latB\n1.0,2.0,30.0,40.0\n")
    mapping1 = {"lon": "lonA", "lat": "latA"}
    mapping2 = {"lon": "lonB", "lat": "latB"}

    source = window.project.add_source(str(csv_path), kind="points")
    layer1 = window.project.add_layer("pts1", source.source_id)
    layer1.tags["points.mapping"] = json.dumps(mapping1)
    window.add_layer_row(layer1, read_csv_points(str(csv_path), mapping=mapping1))

    layer2 = window.project.add_layer("pts2", source.source_id)
    layer2.tags["points.mapping"] = json.dumps(mapping2)
    window.add_layer_row(layer2, read_csv_points(str(csv_path), mapping=mapping2))

    window.layer_list.select_layer(layer1.layer_id)
    proj_path = tmp_path / "proj.dynamix"
    save_project(window.project, proj_path)

    window._open_project_path(proj_path)

    reopened = {l.name: l for l in window.project.layers}
    field1 = window._fields[reopened["pts1"].layer_id]
    field2 = window._fields[reopened["pts2"].layer_id]
    assert field1.lon.tolist() == [1.0]
    assert field1.lat.tolist() == [2.0]
    assert field2.lon.tolist() == [30.0]
    assert field2.lat.tolist() == [40.0]


def test_old_dynamix_file_with_no_kind_key_reopens_every_source_as_raster(qtbot, window,
                                                                          tmp_path):
    """An old project file (pre-Task-11) has no ``"kind"`` key on any source at all -- it must
    still open cleanly through the full ``MainWindow`` reopen path, every source treated as a
    raster."""
    raster_path = tmp_path / "dem.npz"
    np.savez(raster_path, values=_FIELD)

    payload = {
        "format": "dynamix-project", "schema": 1, "app_version": "", "created": "",
        "modified": "", "title": "old", "description": "",
        "sources": [{"source_id": "src0", "path": str(raster_path)}],
        "layers": [{"layer_id": 0, "name": "dem", "source_id": "src0",
                   "chain": {"steps": []}, "visible": True, "tags": {}, "parent_id": None}],
        "topologies": [],
    }
    proj_path = tmp_path / "old.dynamix"
    proj_path.write_text(json.dumps(payload))

    window._open_project_path(proj_path)

    assert window.project.sources["src0"].kind == "raster"
    assert len(window.project.layers) == 1
