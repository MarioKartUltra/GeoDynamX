# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Map tab in the main window: the stack follows the project; the Map draws it."""
from __future__ import annotations

import numpy as np
import pytest

from test_roi_shell_flow import _fork, _roi_child, _save, registered_builtins  # noqa: F401


@pytest.fixture
def geo_window(qtbot, registered_builtins, tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from dynamix.shell.main_window import MainWindow

    H, W = 60, 80
    vals = np.random.default_rng(1).standard_normal((H, W)).cumsum(0).astype("float32")
    path = tmp_path / "geo.tif"
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
                       crs="EPSG:32613", transform=from_origin(500000, 4400000, 10, 10)) as ds:
        ds.write(vals, 1)
    win = MainWindow(steps=())
    qtbot.addWidget(win)
    win.open_path(str(path))
    return win


def test_opening_a_dataset_makes_its_group_and_sets_the_map_crs(geo_window):
    win = geo_window
    sid = win.layer.source_id
    assert win.project.layer_stack.node(f"ds:{sid}") is not None
    assert win.project.map_crs == "EPSG:32613"


def test_the_map_tab_draws_a_fork_inside_its_roi(geo_window, qtbot, monkeypatch):
    win = geo_window
    _save(win, row=30, col=40, h=24, w=30)
    child = _roi_child(win, qtbot, "holder_measure")
    src = _fork(win, monkeypatch, child.layer_id, nest=True)
    assert win.project.layer_stack.parent_of(f"src:{src.source_id}").id == "roig:" + \
        win.project.rois[0].roi_id
    win._set_center_view("map")
    r = win.map_view.item_rect(f"src:{src.source_id}")
    assert (r.left(), r.right()) == pytest.approx((500400.0, 500700.0))


def test_the_layers_section_shows_only_on_stack_views(geo_window):
    win = geo_window
    section = win.right_panel._sections["Layers"]
    assert not section.isVisibleTo(win)
    win._set_center_view("map")
    assert section.isVisibleTo(win)


def test_a_layer_waiting_for_run_reads_not_run(geo_window, qtbot, monkeypatch):
    import dynamix.shell.main_window as mw
    from test_roi_shell_flow import _drop

    win = geo_window
    monkeypatch.setattr(mw, "_AUTO_RUN_MAX_PX", 10)
    _drop(win, qtbot, "holder_measure")
    win._set_center_view("map")
    assert win.map_view.reasons().get(f"prod:{win.layer.layer_id}") == "not run"


def test_removing_a_layer_while_the_map_shows_drops_its_rows(geo_window, qtbot):
    win = geo_window
    child = _roi_child(win, qtbot, "holder_measure")
    win._set_center_view("map")
    win._on_remove_requested(child.layer_id, confirm=False)
    assert win.project.layer_stack.node(f"prod:{child.layer_id}") is None
    assert f"prod:{child.layer_id}" not in win.map_view._items


def test_an_old_project_gets_a_default_stack_and_dangling_ids_drop(geo_window, tmp_path):
    from dynamix.model.layer_stack import Row

    win = geo_window
    win.project.layer_stack.insert(Row("src:gone", "raster"))
    dropped = win._reconcile_stack()
    assert dropped == 1 and win.project.layer_stack.node("src:gone") is None


@pytest.mark.skip(reason="an ROI's outline is a row of its group in the stack "
                         "(test_an_roi_outline_is_a_row_of_its_group)")
def test_roi_outlines_draw_on_the_map(geo_window):
    win = geo_window
    _save(win, row=30, col=40, h=24, w=30)
    win._set_center_view("map")
    (outline,) = [i for i in win.map_view._overlay_items]
    xs, _ys = outline.getData()
    assert min(xs) == pytest.approx(500400.0) and max(xs) == pytest.approx(500700.0)


def test_a_lazy_output_on_show_is_its_layers_product_row(geo_window, qtbot):
    """A layer's shown raster -- a lazy output too -- is its product row, named after the
    output; with no raster on show the row says so."""
    win = geo_window
    layer = _roi_child(win, qtbot, "mz_edges")
    rid = f"prod:{layer.layer_id}"
    assert win.project.layer_stack.node(rid) is not None
    win._set_center_view("map")
    assert win.map_view.reasons().get(rid) == "nothing shown"
    win.layer_list.outputHideToggled.emit(layer.layer_id, "coarse", False)
    key = win._output_keys(layer)["coarse"]
    qtbot.waitUntil(lambda: key in win.cache and not win.is_computing
                    and win._out_request is None, timeout=60000)
    win._redraw_map()
    assert rid not in win.map_view.reasons() and win.map_view._items[rid].isVisible()
    assert win._row_label(rid).endswith("· coarse")
    np.testing.assert_array_equal(win.map_view._items[rid].image.shape,
                                  np.shape(win.cache.get(key)["raster"])[::-1])


def test_a_view_knob_updates_the_map_row_without_a_landing(geo_window, qtbot):
    win = geo_window
    child = _roi_child(win, qtbot, "ssa2d")
    win._set_center_view("map")
    before = win.map_view._items[f"prod:{child.layer_id}"].image.copy()
    i = win._names.index("ssa2d")
    win._on_param_changed(i, "show", "residual")                 # a view knob: no compute
    after = win.map_view._items[f"prod:{child.layer_id}"].image
    assert not np.array_equal(before, after)


def test_a_picked_chain_highlights_on_the_map(geo_window, qtbot):
    win = geo_window
    child = _roi_child(win, qtbot, "wtmm2d")
    win._set_center_view("map")
    assert win._active_result.get("chains")
    win._group_palette.add_pick((child.layer_id, 0), shift=False)   # the Raster tab's pick
    assert win.map_view._vectors[f"chn:{child.layer_id}"][2].isVisible()


def test_the_live_band_preview_draws_on_the_map_at_the_roi(geo_window, qtbot, monkeypatch):
    """A stride-s preview places block k centred on its sample, as the Raster tab does."""
    import dynamix.shell.main_window as mw

    win = geo_window
    _save(win, row=30, col=40, h=24, w=30)
    child = _roi_child(win, qtbot, "holder_measure")
    win._set_center_view("map")
    monkeypatch.setattr(mw, "_PREVIEW_MAX_DIM", 8)
    hm = np.asarray(win._active_result["h_map"])
    win._on_band_preview_requested(float(np.nanmin(hm)), float(np.nanmax(hm)) + 1.0, "mask")
    s = -(-30 // 8)
    rid = f"prod:{child.layer_id}"
    assert win.map_view._items[rid].image.shape[:2] == (-(-30 // s), -(-24 // s))
    r = win.map_view.item_rect(rid)
    assert r.left() == pytest.approx(500000.0 + 10 * 40 + 5 - 10 * s / 2)


def test_a_switched_layer_never_shows_the_previous_layers_result(geo_window, qtbot,
                                                                 monkeypatch):
    """Between a selection and its landing the active result is still the previous layer's:
    the new layer's product row reads the cache, never that result."""
    import dynamix.shell.main_window as mw

    win = geo_window
    a = _roi_child(win, qtbot, "holder_measure")
    master = next(l for l in win.project.layers if l.parent_id is None)
    win.layer_list.select_layer(master.layer_id)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=30000)
    monkeypatch.setattr(mw, "_AUTO_RUN_MAX_PX", 10)
    from dynamix.model.device import defaults_for, get_device

    desc = [{"device": "holder_measure",                  # another recipe: its own cache key
             "params": {**defaults_for(get_device("holder_measure")), "n_scales": 5}}]
    win.strips.set_steps(desc, field=win.field)
    win._on_chain_edited(desc)
    qtbot.wait(30)
    b = win.layer
    win._pending_layers.discard(b.layer_id)               # waiting no more, not yet computed
    win.layer_list.select_layer(a.layer_id)
    qtbot.waitUntil(lambda: win.layer is a and not win.is_computing, timeout=30000)
    win.layer = b                                         # the switch, before b's landing
    assert win._row_content(f"prod:{b.layer_id}").reason == "not run"


def test_stack_flags_for_the_3d_entries(geo_window, qtbot):
    win = geo_window
    child = _roi_child(win, qtbot, "wtmm2d")
    stack = win.project.layer_stack
    stack.node(f"ext:{child.layer_id}").visible = False
    flags = win._stack_entry_flags(child)
    assert flags["show_extrema"] is False and flags["show_chains"] is True
    assert flags["extrema_style"] == "blocks" and flags["rank"] >= 0
    stack.node(f"chn:{child.layer_id}").visible = False
    assert win._stack_entry_flags(child) is None or not win._stack_entry_flags(child)["show_chains"]


def test_an_roi_outline_is_a_row_of_its_group(geo_window):
    win = geo_window
    roi = _save(win, row=30, col=40, h=24, w=30)
    rid = f"roi:{roi.roi_id}"
    stack = win.project.layer_stack
    assert stack.parent_of(rid).id == f"roig:{roi.roi_id}"
    assert win._row_label(rid) == f"ROI {roi.label} outline"
    win._set_center_view("map")
    (outline,) = win.map_view._vectors[rid]
    xs, _ys = outline.getData()
    assert outline.isVisible()
    assert min(xs) == pytest.approx(500400.0) and max(xs) == pytest.approx(500700.0)


def test_the_outline_rows_eye_is_the_rois_own_h(geo_window):
    """One switch: the Layers panel's outline row and the ROI row's H in the layer list."""
    from PySide6 import QtCore

    win = geo_window
    roi = _save(win, row=30, col=40, h=24, w=30)
    rid = f"roi:{roi.roi_id}"
    win._set_center_view("map")
    win.layers_panel.item_for(rid).setCheckState(0, QtCore.Qt.Unchecked)
    assert roi.visible is False
    assert win.layer_list._roi_rows[roi.roi_id].hide_button.isChecked()
    assert not win.map_view._vectors[rid][0].isVisible()
    win.layer_list._roi_rows[roi.roi_id].hide_button.setChecked(False)     # the H, back on
    assert roi.visible is True
    assert win.project.layer_stack.node(rid).visible is True
    assert win.layers_panel.item_for(rid).checkState(0) == QtCore.Qt.Checked
    assert win.map_view._vectors[rid][0].isVisible()


def test_the_roi_groups_eye_hides_its_outline_and_its_results(geo_window, qtbot):
    from PySide6 import QtCore

    win = geo_window
    roi = _save(win, row=30, col=40, h=24, w=30)
    child = _roi_child(win, qtbot, "holder_measure")
    win._set_center_view("map")
    win.layers_panel.item_for(f"roig:{roi.roi_id}").setCheckState(0, QtCore.Qt.Unchecked)
    assert not win.map_view._vectors[f"roi:{roi.roi_id}"][0].isVisible()
    assert not win.map_view._items[f"prod:{child.layer_id}"].isVisible()
    assert win.map_view._items[f"src:{child.source_id}"].isVisible()      # the dataset stays
    assert win._stack_entry_flags(child) is None


def test_the_3d_entry_follows_the_outline_row(geo_window, qtbot):
    win = geo_window
    roi = _save(win, row=30, col=40, h=24, w=30)
    child = _roi_child(win, qtbot, "holder_measure")
    assert win._stack_entry_flags(child)["show_outline"] is True
    win.project.layer_stack.node(f"roi:{roi.roi_id}").visible = False
    assert win._stack_entry_flags(child)["show_outline"] is False


def test_a_click_on_the_map_keeps_the_selected_layer(geo_window, qtbot):
    """The Raster tab shows the selected layer: a click on the Map (to pan, to look) never
    switches it."""
    win = geo_window
    child = _roi_child(win, qtbot, "holder_measure")
    win._set_center_view("map")
    win.map_view.rowClicked.emit(f"src:{child.source_id}")
    assert win.layer is child


def _centre(view):
    (x0, x1), (y0, y1) = view.viewRange()
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def test_the_map_opens_on_the_raster_tabs_extent(geo_window, qtbot):
    """Raster pixel (col, row) is map (500005 + 10 col, 4399995 - 10 row) on this grid."""
    win = geo_window
    win.resize(1200, 800)
    win.show()
    win.canvas.view.setRange(xRange=(10, 30), yRange=(5, 25), padding=0)
    cx, cy, cw, ch = _centre(win.canvas.view)
    win._set_center_view("map")
    mx, my, mw, mh = _centre(win.map_view.view)
    assert (mx, my) == pytest.approx((500005 + 10 * cx, 4399995 - 10 * cy))
    assert mw >= 10 * cw - 1e-6 or mh >= 10 * ch - 1e-6


def test_raster_returns_on_the_maps_extent(geo_window, qtbot):
    win = geo_window
    win.resize(1200, 800)
    win.show()
    win._set_center_view("map")
    win.map_view.view.setRange(xRange=(500200.0, 500400.0), yRange=(4399700.0, 4399900.0),
                               padding=0)
    mx, my, _mw, _mh = _centre(win.map_view.view)
    win._set_center_view("raster")
    cx, cy, _cw, _ch = _centre(win.canvas.view)
    assert (500005 + 10 * cx, 4399995 - 10 * cy) == pytest.approx((mx, my))


def test_the_extent_crosses_a_map_crs(geo_window, qtbot):
    from rasterio.warp import transform

    win = geo_window
    win.resize(1200, 800)
    win.show()
    win.project.map_crs = "EPSG:4326"
    win.canvas.view.setRange(xRange=(10, 30), yRange=(5, 25), padding=0)
    cx, cy, _cw, _ch = _centre(win.canvas.view)
    win._set_center_view("map")
    mx, my, _mw, _mh = _centre(win.map_view.view)
    (lon,), (lat,) = transform("EPSG:32613", "EPSG:4326", [500005 + 10 * cx],
                               [4399995 - 10 * cy])
    assert (mx, my) == pytest.approx((lon, lat), abs=1e-4)


def test_the_opacity_slider_fades_the_map_row_as_it_moves(geo_window, qtbot):
    win = geo_window
    child = _roi_child(win, qtbot, "holder_measure")
    win._set_center_view("map")
    rid = f"prod:{child.layer_id}"
    win.layers_panel.select_row(rid)
    win.layers_panel.opacity_slider.setValue(30)
    assert win.map_view._items[rid].opacity() == pytest.approx(0.3)
    assert win.project.layer_stack.node(rid).opacity == pytest.approx(0.3)


from test_bus_shell import _Route, three_datasets  # noqa: F401,E402
from tests.test_shell_window import stub_devices, window  # noqa: F401,E402


def test_a_bus_is_a_stack_group_of_its_channels(three_datasets, qtbot, monkeypatch):
    import dynamix.shell.main_window as mw

    win, (host, tex, _far) = three_datasets
    route = _Route(lambda cands: [s for c in cands if c["label"] in ("tex", "host")
                                  for s in c["sends"]][::-1])
    monkeypatch.setattr(mw, "BusDialog", route)
    with qtbot.waitSignal(win.resolved, timeout=20000):
        win._on_bus_requested(host.layer_id)
    bus = win.layer
    group = win.project.layer_stack.node(f"stack:{bus.layer_id}")
    assert group is not None and group.kind == "stack" and group.name == "Stack"
    assert [c.id for c in group.children] == [f"ch:{bus.layer_id}:0", f"ch:{bus.layer_id}:1"]
    assert win._bus_plane(bus, 1).values.shape == (36, 36)
    assert win._bus_plane(bus, 1) is win._bus_plane(bus, 1)     # one plane per stack


def test_a_tool_dropped_on_a_child_dataset_gets_its_rows(geo_window, qtbot):
    """A child dataset's rack commits in place (it never forks): the stack follows the commit."""
    from test_roi_shell_flow import _drop

    win = geo_window
    win._on_roi_child_create({"roi_row": 10, "roi_col": 10, "roi_h": 24, "roi_w": 30})
    child = win.layer
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "wtmm2d")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    assert win.layer is child
    stack = win.project.layer_stack
    assert all(stack.node(f"{k}:{child.layer_id}") is not None for k in ("ext", "chn", "prod"))


def test_editing_a_bus_s_sends_follows_in_the_stack(three_datasets, qtbot, monkeypatch):
    import json

    import dynamix.shell.main_window as mw
    from dynamix.core.bus import parse_sends

    win, (host, _tex, _far) = three_datasets
    route = _Route(lambda cands: [s for c in cands if c["label"] in ("tex", "host")
                                  for s in c["sends"]][::-1])
    monkeypatch.setattr(mw, "BusDialog", route)
    with qtbot.waitSignal(win.resolved, timeout=20000):
        win._on_bus_requested(host.layer_id)
    bus = win.layer
    sends = parse_sends(bus.chain.steps[0].params.get("_sends"))
    with qtbot.waitSignal(win.resolved, timeout=20000):
        win._on_param_changed(0, "_sends", json.dumps(sends[:1]))
    group = win.project.layer_stack.node(f"stack:{bus.layer_id}")
    assert [c.id for c in group.children] == [f"ch:{bus.layer_id}:0"]


def test_a_non_active_layers_map_rows_follow_its_shown_level(geo_window, qtbot):
    from test_roi_shell_flow import _drop

    from dynamix.engine.resolve import resolve

    win = geo_window
    master = win.layer
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "wtmm2d")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    wt = win.layer
    wt.tags["ui.scale_idx"] = "2"                       # the Transport's shown level
    win.layer_list.select_layer(master.layer_id)
    qtbot.waitUntil(lambda: win.layer is master and not win.is_computing, timeout=30000)
    content = win._row_content(f"ext:{wt.layer_id}")
    shown = resolve(wt, win._fields[wt.layer_id], win.cache, source_id=wt.source_id,
                    level=2).result
    assert content.result.get("_scale_idx") == 2
    np.testing.assert_array_equal(content.result["extrema"][content.scale_idx]["x"],
                                  shown["extrema"][0]["x"])


def test_the_lift_rank_is_dense_over_the_layers_that_draw_vectors(geo_window):
    from dynamix.model.layer import Layer
    from dynamix.model.layer_stack import Row

    win = geo_window
    stack = win.project.layer_stack
    for rid, kind in (("ext:8", "extrema"), ("chn:7", "chains"), ("ext:7", "extrema")):
        stack.insert(Row(rid, kind), "root", 0)            # ext:7 on top, then chn:7, ext:8
    rank = {lid: win._stack_entry_flags(Layer(layer_id=lid, name="", source_id="x"))["rank"]
            for lid in (7, 8)}
    assert rank == {8: 0, 7: 1}


def test_a_channel_row_of_an_emptied_bus_still_has_a_label(geo_window):
    from dynamix.model.layer_stack import Row

    win = geo_window
    win.project.layer_stack.insert(Row(f"ch:{win.layer.layer_id}:0", "channel"), "root", 0)
    assert win._row_label(f"ch:{win.layer.layer_id}:0") == "band 1"


def test_a_point_layer_the_map_cannot_place_stays_off_the_map(geo_window, monkeypatch):
    import types

    import rasterio.warp

    from dynamix.model.chain import Chain

    win = geo_window
    src = win.project.add_source("mem:pts", kind="points")
    layer = win.project.add_layer("pts", src.source_id, Chain())
    win._fields[layer.layer_id] = types.SimpleNamespace(lon=[1.0], lat=[2.0])

    def boom(*_a, **_k):
        raise ValueError("no transform")

    monkeypatch.setattr(rasterio.warp, "transform", boom)
    assert win._map_overlays()["points"] == []


def test_file_axes_of_a_sample_anchored_stride_field_start_at_its_first_sample():
    import types

    from dynamix.shell.main_window import MainWindow

    f = types.SimpleNamespace(x_axis=100.0 + 40.0 * np.arange(5),
                              y_axis=200.0 - 40.0 * np.arange(4), frame=None,
                              provenance={"display_stride": 4, "display_anchor": "sample",
                                          "full_dims": (16, 20)})
    axes = MainWindow._file_axes(f)
    assert (axes.x_axis[0], axes.x_axis[1], len(axes.x_axis)) == (100.0, 110.0, 20)
    assert (axes.y_axis[0], axes.y_axis[1]) == (200.0, 190.0)


def test_a_reference_layer_is_reprojected_once_per_map_crs(geo_window, monkeypatch):
    import dynamix.geo.vectors as gv
    from dynamix.geo.vectors import Feature, VectorLayer

    win = geo_window
    layer = VectorLayer(name="r", kind="polyline", crs="EPSG:4326", bounds=(0, 0, 1, 1),
                        features=[Feature(parts=[np.array([[-105.0, 39.7], [-104.9, 39.8]])])])
    rec = win.project.add_reference_layer("mem:ref", name="r")
    win._reference_layers[rec.ref_id] = layer
    calls = []
    real = gv.to_crs
    monkeypatch.setattr(gv, "to_crs", lambda l, c: (calls.append(1), real(l, c))[1])
    win._map_overlays()
    win._map_overlays()
    assert calls == [1]
