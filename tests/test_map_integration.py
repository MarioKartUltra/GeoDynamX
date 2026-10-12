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
