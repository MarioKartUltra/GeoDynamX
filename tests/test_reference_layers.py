# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Reference layers on the 2-D canvas and in the 3-D scene: interpretation drawn over the
data -- BOEM's anomaly polygons over the bathymetry they were mapped from."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets  # noqa: F401

from dynamix.shell.canvas import Canvas


def _entry(ref_id="ref0", kind="polygon", color="#ff8800", visible=True, parts=None):
    parts = parts if parts is not None else [np.array([[1.0, 1.0], [5.0, 1.0], [5.0, 4.0], [1.0, 1.0]]),
                                             np.array([[7.0, 7.0], [9.0, 7.0], [9.0, 9.0], [7.0, 7.0]])]
    return {"ref_id": ref_id, "name": "slumps", "kind": kind, "color": color, "visible": visible,
            "features": [parts]}


def test_canvas_draws_a_polygon_layer_as_one_nan_separated_polyline_item(qtbot):
    c = Canvas(); qtbot.addWidget(c)
    c.set_reference_layers([_entry()])
    item = c.reference_items["ref0"]
    x, y = item.getData()
    assert np.isnan(x).sum() == 2                       # one NaN after each part
    assert x[0] == 1.0 and x[3] == 1.0 and y[5] == 7.0    # part 1 then part 2 vertices
    assert item.opts["pen"].color().name() == "#ff8800"
    assert item.isVisible()


def test_canvas_draws_a_point_layer_as_a_scatter_and_honours_visibility(qtbot):
    c = Canvas(); qtbot.addWidget(c)
    pts = [[np.array([[2.0, 3.0]])], [np.array([[4.0, 5.0]])]]
    c.set_reference_layers([{"ref_id": "ref1", "name": "plumes", "kind": "point", "color": "#00ffcc",
                             "visible": False, "features": pts}])
    item = c.reference_items["ref1"]
    assert item.data["x"].tolist() == [2.0, 4.0] and item.data["y"].tolist() == [3.0, 5.0]
    assert not item.isVisible()
    c.set_reference_visible("ref1", True)
    assert item.isVisible()


def test_setting_a_new_list_replaces_old_items(qtbot):
    c = Canvas(); qtbot.addWidget(c)
    c.set_reference_layers([_entry("ref0"), _entry("ref1")])
    c.set_reference_layers([_entry("ref1")])
    assert set(c.reference_items) == {"ref1"}


# ---------------------------------------------------------------------------------- the window

from pathlib import Path  # noqa: E402

from tests.test_shell_window import loaded, stub_devices, window  # noqa: E402,F401
from tests.test_vectors import _write_shapefile  # noqa: E402


def _shp(tmp_path, name="anomaly_slumps"):
    from rasterio.crs import CRS
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / name
    ring = [(2.0, 2.0), (10.0, 2.0), (10.0, 9.0), (2.0, 2.0)]
    _write_shapefile(base, 15, [[ring]], [("NAME", "C", 8, 0)], [("slump",)], CRS.from_epsg(32750).to_wkt())
    return base.with_suffix(".shp")


def test_open_reference_layer_registers_it_lists_it_and_draws_it_on_the_canvas(loaded, tmp_path, monkeypatch):
    shp = _shp(tmp_path)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: ([str(shp)], "")))
    loaded._on_open_reference_clicked()
    assert [r.name for r in loaded.project.reference_layers] == ["anomaly_slumps"]
    ref_id = loaded.project.reference_layers[0].ref_id
    assert ref_id in loaded._reference_layers and ref_id in loaded.canvas.reference_items
    panel = loaded.reference_panel
    assert panel.count() == 1 and panel.item(0).text() == "anomaly_slumps"
    assert "1 reference layer" in loaded.statusBar().currentMessage()


def test_toggling_the_panel_row_hides_the_layer_and_persists_visibility(loaded, tmp_path):
    shp = _shp(tmp_path)
    loaded._open_reference_layers([str(shp)])
    ref_id = loaded.project.reference_layers[0].ref_id
    item = loaded.reference_panel.item(0)
    item.setCheckState(QtCore.Qt.Unchecked)
    assert not loaded.canvas.reference_items[ref_id].isVisible()
    assert loaded.project.reference_layers[0].visible is False
    assert loaded.project.to_payload()["reference_layers"][0]["visible"] is False


def test_reopening_a_project_re_reads_reference_files_and_reports_a_missing_one(loaded, tmp_path, monkeypatch):
    shp = _shp(tmp_path)
    loaded._open_reference_layers([str(shp)])
    loaded.project.add_reference_layer(str(tmp_path / "gone.shp"))
    payload = loaded.project.to_payload()
    from dynamix.model.project import Project
    loaded.project = Project.from_payload(payload)
    loaded._reload_reference_layers()
    assert set(loaded._reference_layers) == {"ref0"}                 # gone.shp is kept as a record, not loaded
    assert "gone.shp" in loaded.statusBar().currentMessage()


def test_status_reports_how_many_features_fall_inside_the_raster(loaded, tmp_path):
    # A layer with no features inside the raster's window opens and shows nothing. The status
    # line must say so instead of leaving the user guessing.
    from rasterio.crs import CRS
    far = tmp_path / "far_away"
    ring = [(500.0, 500.0), (600.0, 500.0), (600.0, 600.0), (500.0, 500.0)]
    _write_shapefile(far, 15, [[ring]], [("NAME", "C", 8, 0)], [("x",)], CRS.from_epsg(32750).to_wkt())
    loaded._open_reference_layers([str(_shp(tmp_path)), str(far.with_suffix(".shp"))])
    msg = loaded.statusBar().currentMessage()
    assert "anomaly_slumps 1/1" in msg and "far_away 0/1" in msg and "inside this raster" in msg


def test_open_path_routes_a_shapefile_to_the_reference_layers_and_keeps_the_field(loaded, tmp_path):
    # A shapefile can arrive through the raster Open dialog too. Whichever dialog a shapefile
    # comes through, it is a reference layer.
    before = loaded.field
    loaded.open_path(str(_shp(tmp_path)))
    assert [r.name for r in loaded.project.reference_layers] == ["anomaly_slumps"]
    assert loaded.field is before


def test_opening_a_lyr_opens_every_shapefile_under_the_packages_commondata(loaded, tmp_path):
    # An Esri layer package: v10/<name>.lyr (binary, unreadable) + commondata/<folders>/*.shp.
    pkg = tmp_path / "BOEM_pkg"
    (pkg / "v10").mkdir(parents=True)
    lyr = pkg / "v10" / "0000BOEM.lyr"
    lyr.write_bytes(b"\x00" * 64)
    _shp(pkg / "commondata" / "products", "anomaly_slumps")
    _shp(pkg / "commondata" / "2014_mar", "plumes")
    loaded.open_path(str(lyr))
    assert sorted(r.name for r in loaded.project.reference_layers) == ["anomaly_slumps", "plumes"]


# ------------------------------------------------------------------ zoom to a layer's full extent
# The layers are drawn whole; the VIEW is fitted to the raster. Double-click a row.

def test_double_clicking_a_reference_row_asks_to_zoom_to_it(qtbot):
    from dynamix.model.project import ReferenceLayerRecord
    from dynamix.shell.layer_panel import ReferencePanel
    panel = ReferencePanel(); qtbot.addWidget(panel)
    panel.set_records([ReferenceLayerRecord(ref_id="ref7", path="/x.shp", name="slumps", color="#ff8800")])
    with qtbot.waitSignal(panel.zoomRequested, timeout=1000) as sig:
        panel.itemDoubleClicked.emit(panel.item(0))
    assert sig.args == ["ref7"]


def test_zoom_to_reference_fits_the_canvas_to_the_layers_pixel_bounds(loaded, tmp_path):
    loaded._open_reference_layers([str(_shp(tmp_path))])         # ring x 2..10, y 2..9 in field units
    ref_id = loaded.project.reference_layers[0].ref_id
    loaded._on_reference_zoom(ref_id)
    (x0, x1), (y0, y1) = loaded.canvas.view.viewRange()
    assert x0 <= 2.0 and x1 >= 10.0 and y0 <= 2.0 and y1 >= 9.0
    assert x1 - x0 < 40.0                                          # fitted to the ring, not the raster


def test_scene_entries_carry_the_layer_in_the_fields_pixel_frame_for_surface_riding(loaded, tmp_path):
    loaded._open_reference_layers([str(_shp(tmp_path))])
    e = loaded._scene_reference_entries()[0]
    canvas_parts = loaded.canvas.reference_items[e["ref_id"]].getData()
    assert e["pixels"] is not None and len(e["pixels"]) == 1
    np.testing.assert_allclose(e["pixels"][0][0][:, 0], [2.0, 10.0, 10.0, 2.0])   # the ring's cols


def test_multi_select_unchecks_every_selected_row_at_once(qtbot):
    """A shp package opens many layers: rows are extended-select (shift+click, ctrl+click),
    and toggling one selected row's checkbox propagates to every selected row (one
    visibilityToggled per row)."""
    from PySide6 import QtCore
    from dynamix.model.project import ReferenceLayerRecord
    from dynamix.shell.layer_panel import ReferencePanel

    panel = ReferencePanel(); qtbot.addWidget(panel)
    panel.set_records([ReferenceLayerRecord(ref_id=f"r{i}", path=f"/{i}.shp",
                                            name=f"L{i}", color="#ff8800") for i in range(5)])
    assert panel.selectionMode() == panel.SelectionMode.ExtendedSelection

    # select rows 1,2,3 (a range), then uncheck one of them -> all three go off
    for i in (1, 2, 3):
        panel.item(i).setSelected(True)
    emitted = {}
    panel.visibilityToggled.connect(lambda rid, vis: emitted.__setitem__(rid, vis))
    panel.item(2).setCheckState(QtCore.Qt.Unchecked)     # toggle one of the selection

    assert emitted == {"r1": False, "r2": False, "r3": False}
    assert all(panel.item(i).checkState() == QtCore.Qt.Unchecked for i in (1, 2, 3))
    assert panel.item(0).checkState() == QtCore.Qt.Checked   # untouched rows stay
    assert panel.item(4).checkState() == QtCore.Qt.Checked


def test_toggling_an_unselected_row_affects_only_it(qtbot):
    from PySide6 import QtCore
    from dynamix.model.project import ReferenceLayerRecord
    from dynamix.shell.layer_panel import ReferencePanel

    panel = ReferencePanel(); qtbot.addWidget(panel)
    panel.set_records([ReferenceLayerRecord(ref_id=f"r{i}", path=f"/{i}.shp",
                                            name=f"L{i}", color="#ff8800") for i in range(3)])
    panel.item(0).setSelected(True)
    emitted = {}
    panel.visibilityToggled.connect(lambda rid, vis: emitted.__setitem__(rid, vis))
    panel.item(2).setCheckState(QtCore.Qt.Unchecked)     # NOT part of the selection
    assert emitted == {"r2": False}
    assert panel.item(0).checkState() == QtCore.Qt.Checked


def test_opening_a_small_raster_frames_it_not_the_reference_vectors(loaded, qtbot, tmp_path):
    """A far-flung reference layer must not drive the camera when a new raster opens: the
    view frames the dataset; vectors merely overlay it. (Seen as: forking a small
    derivative zoomed all the way out to a basin-wide shapefile's extent.)"""
    from rasterio.crs import CRS

    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    far = tmp_path / "far_ring"
    ring = [(900.0, 900.0), (1500.0, 900.0), (1500.0, 1500.0), (900.0, 900.0)]
    _write_shapefile(far, 15, [[ring]], [("NAME", "C", 8, 0)], [("x",)],
                     CRS.from_epsg(32750).to_wkt())
    loaded._open_reference_layers([str(far.with_suffix(".shp"))])
    small = RasterField(name="fork", values=np.zeros((16, 20)), frame=LocalFrame(),
                        x_axis=np.arange(20.0), y_axis=np.arange(16.0))
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.load_field(small, "mem:small")
    (x0, x1), (y0, y1) = loaded.canvas.view.viewRange()
    assert x1 - x0 < 60.0 and y1 - y0 < 60.0        # framed to the 20 x 16 raster, not the ring
    assert x0 <= 0.0 <= x1 and x1 >= 19.0            # and the raster is inside the frame


def test_context_menu_remove_emits_for_every_selected_row(qtbot, monkeypatch):
    from dynamix.model.project import ReferenceLayerRecord
    from dynamix.shell.layer_panel import ReferencePanel

    panel = ReferencePanel(); qtbot.addWidget(panel)
    panel.set_records([ReferenceLayerRecord("ref0", "/a.shp"),
                       ReferenceLayerRecord("ref1", "/b.shp"),
                       ReferenceLayerRecord("ref2", "/c.shp")])
    got = []
    panel.removeRequested.connect(got.append)
    panel.item(0).setSelected(True)
    panel.item(1).setSelected(True)
    monkeypatch.setattr(ReferencePanel, "_exec_menu",
                        lambda self, menu, pos: menu.actions()[0])
    panel._on_context_menu(panel.visualItemRect(panel.item(0)).center())
    assert got == ["ref0", "ref1"]                    # the selection, not the third row
    got.clear()
    panel._on_context_menu(panel.visualItemRect(panel.item(2)).center())
    assert got == ["ref2"]                            # an unselected row removes only itself


def test_removing_a_reference_layer_unloads_it_but_keeps_the_file(loaded, tmp_path):
    shp = _shp(tmp_path)
    loaded._open_reference_layers([str(shp)])
    ref_id = loaded.project.reference_layers[0].ref_id
    loaded._on_reference_remove(ref_id)
    assert loaded.project.reference_layers == []
    assert ref_id not in loaded._reference_layers
    assert ref_id not in loaded.canvas.reference_items
    assert ref_id not in loaded._reference_pixels and ref_id not in loaded._reference_inside
    assert loaded.reference_panel.count() == 0
    assert shp.exists()                               # never touch the file
    assert "untouched" in loaded.statusBar().currentMessage()


def test_layer_switches_reuse_the_reference_pixel_warp(loaded, qtbot, tmp_path, monkeypatch):
    """Selecting a layer must not re-warp reference vectors for a georeference already
    seen: over a big shapefile that warp is a visible pause on every click."""
    import dynamix.shell.main_window as mw

    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    loaded._open_reference_layers([str(_shp(tmp_path))])     # warps for the loaded grid
    calls = []
    real = mw.to_field_pixels

    def counting(layer, field):
        calls.append(1)
        return real(layer, field)

    monkeypatch.setattr(mw, "to_field_pixels", counting)
    other = RasterField(name="other", values=np.zeros((8, 9)), frame=LocalFrame(),
                        x_axis=np.arange(9.0) + 100.0, y_axis=np.arange(8.0) + 100.0)
    with qtbot.waitSignal(loaded.resolved, timeout=10000):
        loaded.load_field(other, "mem:other")                # a NEW grid: warps once
    assert len(calls) == 1
    a, b = loaded.project.layers[0], loaded.layer
    loaded.layer_list.select_layer(a.layer_id)               # back to the first grid: cached
    loaded.layer_list.select_layer(b.layer_id)               # and the new grid again: cached
    loaded.layer_list.select_layer(a.layer_id)
    assert len(calls) == 1
