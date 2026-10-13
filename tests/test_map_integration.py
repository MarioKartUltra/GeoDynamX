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
    assert flags["extrema_style"] == "polylines" and flags["rank"] >= 0
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


def test_both_bars_read_lambda_of_the_shown_level_in_metres(geo_window, qtbot):
    from test_roi_shell_flow import _drop

    from dynamix.core.scale_units import lambda_peak_px
    from dynamix.core.wavelet_profiles import format_length

    win = geo_window
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "wtmm2d")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    res = win._active_result
    a = float(np.asarray(res["scales"])[int(res.get("_scale_idx", 0))])
    lam = lambda_peak_px(a, res["params"]["wavelet"], 1)
    want = format_length(lam * 10.0, "m")                        # 10 m pixels
    assert win.canvas.wavelet_bar.isVisible() and win.canvas.wavelet_bar.label == want
    win._set_center_view("map")
    assert win.map_view.wavelet_bar.isVisible() and win.map_view.wavelet_bar.label == want
    (x0, x1), _y = win.map_view.view.viewRange()
    m_per_px = (x1 - x0) / win.map_view.view.width()
    assert win.map_view.wavelet_bar.bracket_px() == pytest.approx(lam * 10.0 / m_per_px, rel=0.05)


def test_a_bar_drag_is_remembered_per_view(geo_window, qtbot):
    from dynamix.shell.settings import load_settings

    win = geo_window
    win.canvas.wavelet_bar.drag_by(-30.0, -10.0)
    assert load_settings().view_options["wavelet_bar"]["raster"] == [-140.0, -54.0]
    win._set_center_view("map")
    win.map_view.scale_bar.drag_by(-5.0, -5.0)
    assert load_settings().view_options["map_scale_bar"] == [-25.0, -25.0]


def test_the_transport_reading_quotes_lambda_in_metres(geo_window, qtbot):
    from test_roi_shell_flow import _drop

    from dynamix.core.scale_units import lambda_peak_px
    from dynamix.core.wavelet_profiles import format_length

    win = geo_window
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "wtmm2d")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    a = float(win._scales[1])
    lam = lambda_peak_px(a, win._active_result["params"]["wavelet"], 1)
    assert win._scale_reading(1) == (f"a = {a:.1f} · λ* {lam:.1f} px ≈ "
                                     f"{format_length(lam * 10.0, 'm')}")


def test_the_cascade_follows_the_racks_knobs_before_a_run(geo_window, qtbot):
    from test_roi_shell_flow import _drop

    win = geo_window
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "holder_measure")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    cascade = win.canvas.wavelet_cascade
    n = int(win._active_result["params"]["n_scales"])
    assert cascade.isVisible() and len(cascade.widths_px()) == n
    last_before = cascade.labels[1]
    i = win._names.index("holder_measure")
    win._on_param_changed(i, "kappa", float(win._params[i]["kappa"]) * 4.0)
    assert cascade.labels[1] != last_before                      # moved at once, before any landing
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)


def test_the_cascade_hides_without_a_microcanonical_tool_and_with_the_toggle(geo_window, qtbot):
    from test_roi_shell_flow import _drop

    win = geo_window
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "wtmm2d")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    assert not win.canvas.wavelet_cascade.isVisible()
    master = next(l for l in win.project.layers if l.parent_id is None)
    win.layer_list.select_layer(master.layer_id)
    qtbot.waitUntil(lambda: win.layer is master and not win.is_computing, timeout=30000)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "holder_multiaffine")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    assert win.canvas.wavelet_cascade.isVisible()
    win._on_display_style_changed("wavelet_cascade", False)
    assert not win.canvas.wavelet_cascade.isVisible()
    win._on_display_style_changed("wavelet_cascade", True)
    assert win.canvas.wavelet_cascade.isVisible()


def test_the_spectrum_windows_follow_the_racks_chain_filter_and_share_the_fit_mode(geo_window, qtbot):
    from test_roi_shell_flow import _drop

    from dynamix.model.device import defaults_for, get_device

    win = geo_window
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "wtmm2d")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    i = win._names.index("wtmm2d")
    desc = [{"device": "wtmm2d", "params": dict(win._params[i])},
            {"device": "chain_length", "params": {**defaults_for(get_device("chain_length")),
                                                  "min_len": 6}}]
    win.strips.set_steps(desc, field=win.field)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    res = win._active_result
    assert res["_chains_total"] > len(res["chains"])
    std, _cmax, all_tables, note = win._spectrum_tables(res)
    assert note.startswith("filtered:") and std is not res["hd_std"] and all_tables[0] is res["hd_std"]
    win._on_spectrum_button_clicked()
    fitter = win._multifractal_window
    assert "filtered" in fitter.windowTitle() and fitter.all_chains_check is not None
    assert fitter._fitmode_combo.isEnabled()
    win._on_dh_button_clicked()
    construction = win._spectrum_construction_window
    fitter._fitmode_combo.setCurrentText("fixed focus")
    assert construction._fitmode_combo.currentText() == "fixed focus"
    assert "filtered" in construction.windowTitle()


def test_a_v_trail_colour_turns_the_trails_on_and_isolated_maxima_toggle(geo_window, qtbot):
    from test_roi_shell_flow import _drop

    win = geo_window
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "wtmm2d")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    assert not win.canvas.vtrail_item.isVisible()
    win._on_display_style_changed("color_vtrail", "#00ff00")
    assert win.layer.tags.get("ui.show_trails") == "True" and win.canvas.vtrail_item.isVisible()
    assert not win.canvas.isolated_raster_item.isVisible()        # isolated maxima hidden by default
    win._on_display_style_changed("show_isolated", True)
    assert win.canvas.isolated_raster_item.isVisible() and win.right_panel.isolated_check.isChecked()


def test_the_chain_grouping_selects_the_dominant_chains_on_the_views(geo_window, qtbot):
    from test_roi_shell_flow import _drop

    win = geo_window
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "wtmm2d")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    win._on_dh_button_clicked()
    construction = win._spectrum_construction_window
    lid = win.layer.layer_id
    dom = construction.current_grouping()
    assert dom is not None and construction.select_views_check.isChecked()
    assert win._group_palette.selection() == {(lid, int(i)) for i in np.flatnonzero(dom)}
    construction._gpct_spin.setValue(construction._gpct_spin.value() - 20.0)   # a retune follows
    dom2 = construction.current_grouping()
    assert not np.array_equal(dom, dom2)
    assert win._group_palette.selection() == {(lid, int(i)) for i in np.flatnonzero(dom2)}
    construction.select_views_check.setChecked(False)                     # off: the views keep theirs
    construction._gpct_spin.setValue(construction._gpct_spin.value() + 20.0)
    assert win._group_palette.selection() == {(lid, int(i)) for i in np.flatnonzero(dom2)}


def test_a_microcanonical_tool_shows_its_cascade_and_no_single_bar_on_the_raster(geo_window, qtbot):
    from test_roi_shell_flow import _drop

    win = geo_window
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "holder_measure")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    assert win.canvas.wavelet_cascade.isVisible()
    assert not win.canvas.wavelet_bar.isVisible()                 # the view shows h, not coefficients
    assert win._wavelet_bar_state[0] is not None                  # the Transport and the Map still read λ*


def test_removing_the_effect_takes_the_wavelet_bars_away(geo_window, qtbot):
    from test_roi_shell_flow import _drop

    win = geo_window
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "wtmm2d")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    assert win.canvas.wavelet_bar.isVisible()
    win._ensure_map_view()
    win._to_map_live()
    assert win.map_view.wavelet_bar.isVisible()
    win.strips.strip(0).removeRequested.emit()                # the strip's own x
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    qtbot.wait(50)
    assert not win.canvas.wavelet_bar.isVisible()
    assert win._wavelet_bar_state == (None, "")
    assert not win.map_view.wavelet_bar.isVisible()


def test_a_bar_style_change_applies_everywhere_and_is_remembered(geo_window, qtbot):
    from test_roi_shell_flow import _drop

    from dynamix.shell.settings import load_settings

    win = geo_window
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "wtmm2d")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    win._ensure_map_view()
    assert win.canvas.wavelet_bar.label.endswith(" m")
    win.canvas.wavelet_bar.set_style({"units": "px"})
    win.canvas.wavelet_bar.styleChanged.emit(win.canvas.wavelet_bar.style())   # the menu's path
    assert win.canvas.wavelet_bar.label.endswith(" px")
    assert win.map_view.wavelet_bar.label.endswith(" px")
    assert win.map_view.wavelet_bar.style()["units"] == "px"
    assert load_settings().view_options["bar_style"]["wavelet"]["units"] == "px"
    win.map_view.scale_bar.set_style({"units": "km", "color": "#ff0000"})
    win.map_view.scale_bar.styleChanged.emit(win.map_view.scale_bar.style())
    assert win.canvas.scale_bar_label.text().endswith(" km")
    assert win.canvas.scale_bar_line.opts["pen"].color().name() == "#ff0000"
    assert load_settings().view_options["bar_style"]["scale"]["color"] == "#ff0000"
