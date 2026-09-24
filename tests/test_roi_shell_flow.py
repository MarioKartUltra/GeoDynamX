# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The shell side of the ROI runner.

One coordinate system: every gesture, panel number and saved ROI is in FILE pixels; a display
picture is drawn over its native blocks and never analysed.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField


def _picture(ny=10, nx=14, s=5, full=(50, 70), source="/nonexistent/pic.tif"):
    f = RasterField(name="pic@pic5", values=np.arange(ny * nx, dtype=float).reshape(ny, nx),
                    frame=LocalFrame(), x_axis=np.arange(nx, dtype=np.float64),
                    y_axis=np.arange(ny, dtype=np.float64))
    f.provenance.update({"display_stride": s, "full_dims": full, "source": source,
                         "window": {"row_off": 0, "col_off": 0}})
    return f


@pytest.fixture
def registered_builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


@pytest.fixture
def pic_window(qtbot, registered_builtins):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=())
    qtbot.addWidget(win)
    win.load_field(_picture(), "/nonexistent/pic.tif")
    return win


def test_roi_tool_seeds_its_box_in_file_pixels_on_a_picture(pic_window):
    """The ROI tool's seed box is centred in the FILE (50 x 70), not in the 10 x 14 picture."""
    pic_window._on_roi_tool_clicked()
    spec = pic_window.roi_panel.values()
    assert (spec["roi_h"], spec["roi_w"]) == (50, 50)
    assert (spec["roi_row"], spec["roi_col"]) == (0, 10)


def test_transect_on_a_picture_samples_the_blocks_under_its_file_pixel_line(pic_window,
                                                                              monkeypatch):
    """A transect's endpoints are FILE pixels; on a picture they sample the block-centre
    samples under them, with the distance axis in file pixels."""
    import dynamix.shell.main_window as mw
    from dynamix.model.project import TransectRecord

    got = {}

    class _Dialog:
        def __init__(self, dist, z, **kw):
            got["dist"], got["z"] = np.asarray(dist), np.asarray(z)

        def show(self):
            pass

        raise_ = activateWindow = show

    monkeypatch.setattr(mw, "ProfileDialog", _Dialog)
    # row 27 is the centre of block 5 (rows 25..29); cols 2 .. 67 are block centres 0 .. 13
    rec = TransectRecord(transect_id=1, a=(2.0, 27.0), b=(67.0, 27.0))
    pic_window._on_transect_plot_requested(rec)
    assert got["z"][0] == pytest.approx(5 * 14 + 0)          # picture[5, 0]
    assert got["z"][-1] == pytest.approx(5 * 14 + 13)        # picture[5, 13]
    assert got["dist"][-1] == pytest.approx(65.0)            # file pixels, not samples


def test_the_reported_misplacement_is_gone_roi_numbers_stay_file_pixels(pic_window):
    """The 2026-09-22 bug: a box on the decimated overview landed stride-x off. On the
    picture the box's numbers ARE file pixels -- nothing multiplies them."""
    # Only the recipe is under test: no worker (the fixture's source does not exist, and an
    # in-flight worker's error would land in a LATER test's event loop).
    pic_window._start_worker = lambda: None
    pic_window._on_roi_create({"roi_row": 12, "roi_col": 23, "roi_h": 16, "roi_w": 20,
                               "boundary": "auto"})
    step = next(r for r in pic_window.layer.chain.steps if r.device == "wtmm2d_roi")
    assert (step.params["roi_row"], step.params["roi_col"]) == (12, 23)
    assert (step.params["roi_h"], step.params["roi_w"]) == (16, 20)


def test_a_tool_dropped_on_the_picture_without_an_roi_is_refused(pic_window, qtbot):
    """No tool runs on the picture itself: the drop spawns nothing and says why."""
    from dynamix.model.device import defaults_for, get_device

    notes = []
    pic_window._notify = lambda msg, *a, **k: notes.append(msg)
    n_layers = len(pic_window.project.layers)
    desc = [{"device": "holder_measure",
             "params": defaults_for(get_device("holder_measure"))}]
    pic_window.strips.set_steps(desc, field=pic_window.field)
    pic_window._on_chain_edited(desc)
    qtbot.wait(20)                                   # a deferred fork would land here
    assert len(pic_window.project.layers) == n_layers
    assert any("ROI" in n for n in notes)


def _drop(win, qtbot, device):
    from dynamix.model.device import defaults_for, get_device

    desc = [{"device": device, "params": defaults_for(get_device(device))}]
    win.strips.set_steps(desc, field=win.field)
    win._on_chain_edited(desc)
    qtbot.wait(30)                                    # the fork spawns one tick later


def test_save_roi_stores_the_file_rect_on_the_source_and_makes_it_active(pic_window):
    master = pic_window.layer
    pic_window._on_roi_save({"roi_row": 12, "roi_col": 23, "roi_h": 16, "roi_w": 20,
                             "boundary": "auto"})
    (roi,) = pic_window.project.rois
    assert (roi.row, roi.col, roi.h, roi.w) == (12, 23, 16, 20)
    assert roi.source_id == master.source_id and roi.label == "A"
    assert pic_window._active_roi_id == roi.roi_id
    assert pic_window.roi_panel.saved_list.count() == 1
    pic_window._on_roi_save({"roi_row": 0, "roi_col": 0, "roi_h": 8, "roi_w": 8,
                             "boundary": "auto"})
    assert [r.label for r in pic_window.project.rois] == ["A", "B"]
    assert pic_window._active_roi_id == pic_window.project.rois[1].roi_id
    x, _ = pic_window.canvas.saved_roi_item.getData()
    assert 22.5 in set(np.asarray(x)[np.isfinite(x)].tolist())       # A's left edge, file px


def test_a_tool_dropped_with_an_active_roi_spawns_its_result_child_on_that_roi(pic_window,
                                                                             qtbot):
    pic_window._start_worker = lambda: None
    pic_window._on_roi_save({"roi_row": 12, "roi_col": 23, "roi_h": 16, "roi_w": 20,
                             "boundary": "auto"})
    roi = pic_window.project.rois[0]
    _drop(pic_window, qtbot, "holder_measure")
    child = pic_window.layer
    assert child.parent_id is not None
    assert child.tags.get("roi.window") == "12,23,16,20"
    assert child.roi_id == roi.roi_id
    assert child.name.endswith("@A")
    assert [s.device for s in child.chain.steps] == ["holder_measure"]


def test_activation_and_deactivation_follow_the_panel(pic_window, qtbot):
    pic_window._on_roi_save({"roi_row": 12, "roi_col": 23, "roi_h": 16, "roi_w": 20,
                             "boundary": "auto"})
    pic_window._on_roi_save({"roi_row": 0, "roi_col": 0, "roi_h": 8, "roi_w": 8,
                             "boundary": "auto"})
    first = pic_window.project.rois[0].roi_id
    pic_window.roi_panel.roiActivated.emit(first)
    assert pic_window._active_roi_id == first
    pic_window.roi_panel.roiActivated.emit("")          # the panel's Deselect
    assert pic_window._active_roi_id is None
    n = len(pic_window.project.layers)
    _drop(pic_window, qtbot, "holder_measure")          # the picture refuses again
    assert len(pic_window.project.layers) == n


def test_the_legacy_one_click_also_saves_the_roi_it_runs_on(pic_window):
    pic_window._start_worker = lambda: None
    pic_window._on_roi_create({"roi_row": 12, "roi_col": 23, "roi_h": 16, "roi_w": 20,
                               "boundary": "auto"})
    (roi,) = pic_window.project.rois
    assert (roi.row, roi.col, roi.h, roi.w) == (12, 23, 16, 20)
    assert pic_window.layer.roi_id == roi.roi_id


@pytest.fixture
def real_pic_window(qtbot, registered_builtins, tmp_path):
    """A picture of a REAL GeoTIFF, so the worker's ROI runner can read native pixels."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from dynamix.roi.picture import read_picture
    from dynamix.shell.main_window import MainWindow

    v = np.random.default_rng(2).standard_normal((64, 80)).cumsum(0).cumsum(1)
    path = tmp_path / "r.tif"
    with rasterio.open(path, "w", driver="GTiff", height=64, width=80, count=1,
                       dtype="float32", crs="EPSG:32615",
                       transform=from_origin(500000.0, 3200000.0, 2.0, 2.0)) as d:
        d.write(v.astype(np.float32), 1)
    win = MainWindow(steps=())
    qtbot.addWidget(win)
    win.load_field(read_picture(path, max_dim=16), str(path))
    return win


def test_an_roi_hmap_is_shown_pinned_at_the_roi(real_pic_window, qtbot):
    win = real_pic_window
    win._on_roi_save({"roi_row": 20, "roi_col": 24, "roi_h": 16, "roi_w": 20,
                      "boundary": "auto"})
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "holder_measure")
    qtbot.wait(50)
    r = win.canvas.image_item.mapRectToView(win.canvas.image_item.boundingRect())
    assert (r.x(), r.y(), r.width(), r.height()) == (23.5, 19.5, 20.0, 16.0)
    assert win.canvas._field.values.shape == (16, 20)


def test_roi_edge_extrema_are_drawn_over_the_roi_on_the_picture(real_pic_window, qtbot):
    """An ROI edge result is ROI-local; its pixel overlay must sit on the ROI's FILE pixels
    over the picture (display_offset = the ROI origin, the picture's window being (0, 0))."""
    win = real_pic_window
    win._on_roi_save({"roi_row": 20, "roi_col": 24, "roi_h": 16, "roi_w": 20,
                      "boundary": "auto"})
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "cdf_edges")
    qtbot.wait(50)
    item = win.canvas.extrema_raster_item
    r = item.mapRectToView(item.boundingRect())
    assert (r.x(), r.y()) == (23.5, 19.5)
    assert (r.width(), r.height()) == (20.0, 16.0)


class _NoMarginPlugin:
    """An outside plugin that declares no roi_margin."""
    name = "no_margin_plugin"
    params = ()

    def compute(self, field, params, *, progress=None):
        v = np.asarray(field.values)
        return {"raster_out": v.copy(), "extrema": [], "chains": [], "_shape": v.shape}

    def cache_key(self, source_id, params):
        return f"nmp:{source_id}"


def test_an_undeclared_plugin_on_an_roi_runs_and_the_window_says_so(real_pic_window, qtbot):
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.device import register_device

    win = real_pic_window
    register_device(_NoMarginPlugin())
    notes = []
    win._notify = lambda msg, *a, **k: notes.append(msg)
    master = win.layer
    layer = win.project.add_layer("plugin @A", master.source_id,
                                  Chain((DeviceRef("no_margin_plugin", {}),)),
                                  parent_id=master.layer_id, tags={"roi.window": "20,24,16,20"})
    win.add_layer_row(layer, win._fields.get(master.layer_id, win.field))
    with qtbot.waitSignal(win.resolved, timeout=60000) as sig:
        win.layer_list.select_layer(layer.layer_id)
    assert sig.args[0].result["_roi"]["margin_declared"] is False
    assert any("margin" in n for n in notes)


def test_the_vector_scene_gets_an_roi_result_on_the_rois_own_grid(real_pic_window, qtbot):
    """The scene places result pixels as field.x_axis[cols]; an ROI result's cols are
    ROI-LOCAL, so its entry must carry a field on the ROI's own grid (native values + axes)
    and an ROI outline in that grid -- never the parent picture indexed by local pixels."""
    from unittest.mock import MagicMock

    win = real_pic_window
    win._on_roi_save({"roi_row": 20, "roi_col": 24, "roi_h": 16, "roi_w": 20,
                      "boundary": "auto"})
    with qtbot.waitSignal(win.resolved, timeout=60000) as sig:
        _drop(win, qtbot, "holder_measure")
    res = sig.args[0].result
    captured = []
    win._arrangement = MagicMock()
    win._arrangement.set_layers.side_effect = captured.extend
    win._sync_arrangement(frame_mode=True)
    entry = next(e for e in captured if e["layer"] is win.layer)
    assert entry["field"].values.shape == (16, 20)
    np.testing.assert_allclose(entry["field"].x_axis, res["_roi_axes"][0])
    assert entry["result"]["_roi"]["roi"] == (0, 0, 16, 20)
    assert entry["drape"].shape == (16, 20)


def test_repeated_scene_syncs_hand_over_the_same_roi_entry_objects(real_pic_window, qtbot):
    """The scene rebuilds a layer's actors whenever id(result) changes: the ROI-grid entry
    must be memoised per cached result, or every resync would redraw the ROI child."""
    from unittest.mock import MagicMock

    win = real_pic_window
    win._on_roi_save({"roi_row": 20, "roi_col": 24, "roi_h": 16, "roi_w": 20,
                      "boundary": "auto"})
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "holder_measure")
    seen = []
    win._arrangement = MagicMock()
    win._arrangement.set_layers.side_effect = lambda e: seen.append(
        next(x for x in e if x["layer"] is win.layer))
    win._sync_arrangement(frame_mode=True)
    win._sync_arrangement(frame_mode=True)
    assert seen[0]["result"] is seen[1]["result"]
    assert seen[0]["field"] is seen[1]["field"]


# ------------------------------------------- Backproject + reference on a picture

def test_backproject_targets_a_pictures_file_grid(tmp_path):
    """Final review: a picture target stamped its SAMPLE grid (nx = samples,
    dx = s x native), so backprojected points landed s x off on the file-pixel canvas."""
    from dynamix.roi.picture import read_picture
    from dynamix.shell.main_window import MainWindow

    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / "t.tif"
    with rasterio.open(path, "w", driver="GTiff", height=48, width=69, count=1,
                       dtype="float32", crs="EPSG:32615",
                       transform=from_origin(500000.0, 3200000.0, 2.0, 2.0)) as d:
        d.write(np.zeros((48, 69), dtype=np.float32), 1)
    sc = MainWindow._backproject_scalars_for(read_picture(path, max_dim=16))
    assert (sc["_target_nx"], sc["_target_ny"]) == (69, 48)
    assert sc["_target_dx"] == pytest.approx(2.0) and sc["_target_dy"] == pytest.approx(-2.0)
    assert sc["_target_x0"] == pytest.approx(500001.0)
    assert sc["_target_y0"] == pytest.approx(3199999.0)


def test_reference_inside_count_uses_the_files_extent_on_a_picture(real_pic_window):
    """The vertices are now FILE pixels; the 'how much can this raster show' count must use
    the file's extent, not the picture's sample count."""
    from dynamix.geo.vectors import Feature, VectorLayer

    win = real_pic_window                                  # 64 x 80 file, stride 5 picture
    x, y = 500000.0 + 2.0 * (70 + 0.5), 3200000.0 - 2.0 * (50 + 0.5)   # file col 70, row 50
    rec = win.project.add_reference_layer("/nonexistent/seeps.shp", name="seeps")
    win._reference_layers[rec.ref_id] = VectorLayer(
        name="seeps", kind="point", crs="EPSG:32615",
        features=[Feature(parts=[np.array([[x, y]])])], bounds=(x, y, x, y))
    win._push_reference_layers()
    assert win._reference_inside[rec.ref_id] == (1, 1)
    (pt,) = win._reference_pixels[rec.ref_id][0]
    assert tuple(pt[0]) == pytest.approx((70.0, 50.0))


# ------------------------------------------- Any tool on the active ROI

@pytest.fixture
def plain_window(qtbot, registered_builtins, tmp_path):
    """A small raster loaded WHOLE (no picture) -- ROIs work the same on it."""
    from dynamix.shell.main_window import MainWindow

    v = np.random.default_rng(4).standard_normal((40, 50)).cumsum(0)
    f = RasterField(name="plain", values=v, frame=LocalFrame(), x_axis=np.arange(50.0),
                    y_axis=np.arange(40.0))
    win = MainWindow(steps=())
    qtbot.addWidget(win)
    win.load_field(f, "/nonexistent/plain.npz")
    return win


@pytest.mark.parametrize("fixture", ["pic_window", "plain_window"])
def test_a_non_primary_analyzer_dropped_with_an_roi_active_runs_on_that_roi(fixture, request,
                                                                          qtbot):
    """Final review: pca/tucker are not 'primary analyzers', so they never forked
    -- on a picture the drop errored, on a whole-loaded file it silently ran on the WHOLE
    field. With an ROI active, any analyzing transform spawns on the ROI."""
    win = request.getfixturevalue(fixture)
    win._start_worker = lambda: None
    master = win.layer
    win._on_roi_save({"roi_row": 4, "roi_col": 5, "roi_h": 16, "roi_w": 20,
                      "boundary": "auto"})
    _drop(win, qtbot, "pca")
    child = win.layer
    assert child is not master and child.parent_id == master.layer_id
    assert child.tags.get("roi.window") == "4,5,16,20" and child.name.endswith("@A")
    assert [s.device for s in master.chain.steps] == []          # the master is untouched


def test_a_field_stage_alone_on_a_picture_is_refused_with_how_to(pic_window, qtbot):
    win = pic_window
    notes = []
    win._notify = lambda msg, *a, **k: notes.append(msg)
    master = win.layer
    win._on_roi_save({"roi_row": 4, "roi_col": 5, "roi_h": 16, "roi_w": 20,
                      "boundary": "auto"})
    n = len(win.project.layers)
    _drop(win, qtbot, "noise")
    assert len(win.project.layers) == n
    assert [s.device for s in master.chain.steps] == []
    assert any("analyzer" in m for m in notes)


# ------------------------------------------- Removing a result takes its drawing with it

def _overlay_counts(canvas):
    """How much of a result's overlay is still drawn: extrema raster pixels, extrema points,
    chain-trail points."""
    img = canvas.extrema_raster_item.image
    ex, _ = canvas.extrema_item.getData()
    hx, _ = canvas.hchain_item.getData()
    return (0 if img is None else int(np.count_nonzero(np.nan_to_num(img))),
            0 if ex is None else len(ex), 0 if hx is None else len(hx))


@pytest.mark.parametrize("fixture", ["real_pic_window", "plain_window"])
def test_removing_an_roi_result_takes_its_overlay_off_the_canvas(fixture, request, qtbot):
    """Removing ``cdf_edges @A`` selects the parent, whose (empty-chain) result carries no
    extrema -- the removed child's extrema and trails must not stay drawn over it."""
    win = request.getfixturevalue(fixture)
    win._on_roi_save({"roi_row": 4, "roi_col": 5, "roi_h": 16, "roi_w": 20,
                      "boundary": "auto"})
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "cdf_edges")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    child = win.layer
    assert _overlay_counts(win.canvas) != (0, 0, 0)            # the child is drawn
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win._on_remove_requested(child.layer_id)
    assert win.layer is not child
    assert _overlay_counts(win.canvas) == (0, 0, 0)
    assert win.canvas._pick_chains is None


def test_an_extrema_less_landing_behind_the_vector_view_clears_on_flip_back(plain_window,
                                                                           qtbot):
    """The canvas is not redrawn while the Vector view is up; a landing with no extrema must
    still leave it clean when the user flips back."""
    from PySide6 import QtWidgets

    win = plain_window
    win._on_roi_save({"roi_row": 4, "roi_col": 5, "roi_h": 16, "roi_w": 20,
                      "boundary": "auto"})
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "cdf_edges")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    child = win.layer
    stand_in = QtWidgets.QWidget()                     # stands in for the Vector view's 3-D scene
    win._center_stack.addWidget(stand_in)
    win._center_stack.setCurrentWidget(stand_in)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win._on_remove_requested(child.layer_id)
    win._center_stack.setCurrentIndex(0)
    win._redraw_canvas_overlay_if_dirty()
    assert _overlay_counts(win.canvas) == (0, 0, 0)


# ------------------------------------------- The ROI panel closes

def test_the_roi_panels_close_button_hides_it_and_clears_only_the_drawn_box(pic_window):
    """× hides the panel, takes the unsaved amber box off the canvas and disarms a pending
    Place box; the saved ROIs and the active one are untouched."""
    win = pic_window
    win._on_roi_save({"roi_row": 12, "roi_col": 23, "roi_h": 16, "roi_w": 20,
                      "boundary": "auto"})
    active = win._active_roi_id
    win._on_roi_tool_clicked()
    win.roi_panel.place_button.click()
    assert not win.roi_panel.isHidden()
    x, _ = win.canvas.roi_band_item.getData()
    assert x is not None and len(x)                        # the drawn box is up
    win.roi_panel.close_button.click()
    assert win.roi_panel.isHidden()
    x, _ = win.canvas.roi_band_item.getData()
    assert x is None or len(x) == 0
    assert win.canvas._roi_place is None
    assert win._active_roi_id == active and len(win.project.rois) == 1


# ------------------------------------------- Switching what a tool shows never recomputes

def test_flipping_show_redraws_from_the_cache_without_a_worker_or_a_pending_run(
        plain_window, qtbot):
    """cdf_edges' Show (edges -> filtered) is a display choice. With auto-run
    OFF (the case that used to demand a manual Run) the flip is drawn at once from the cache."""
    win = plain_window
    win._on_roi_save({"roi_row": 4, "roi_col": 5, "roi_h": 16, "roi_w": 20,
                      "boundary": "auto"})
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, "cdf_edges")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    win._auto_run_action.setChecked(False)
    misses = win.cache.misses
    i = win._names.index("cdf_edges")
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win._on_param_changed(i, "show", "filtered")
    assert win._thread is None and not win.is_computing
    assert win.layer.layer_id not in win._pending_layers
    assert win.cache.misses == misses
    np.testing.assert_array_equal(win.canvas._field.values, win._active_result["filtered"])


# ------------------------------------------- The dataset never takes an analyzer directly

def test_tucker_dropped_on_the_dataset_without_an_roi_spawns_a_child(plain_window, qtbot):
    """After the dataset-row merge: tucker_havok dropped on the dataset with no ROI replaced the
    dataset -- the analyzer fork only knew the primary analyzers, and the any-analyzer spawn
    only ran with an ROI active. The master is the raw dataset; every analysis is a child."""
    win = plain_window
    win._start_worker = lambda: None
    master = win.layer
    assert win._active_roi_for(master.source_id) is None
    _drop(win, qtbot, "tucker_havok")
    child = win.layer
    assert child is not master and child.parent_id == master.layer_id
    assert child.name == f"{master.name} · tucker_havok"
    assert not child.tags.get("roi.window")
    assert [s.device for s in child.chain.steps] == ["tucker_havok"]
    assert [s.device for s in master.chain.steps] == []          # the dataset stays raw


def test_flipping_back_from_the_vector_view_before_any_result_does_not_raise(
        qtbot, registered_builtins):
    """``_canvas_overlay_dirty`` was only ever set by a landing or a project open, so a fresh
    window (every devloop rebuild) flipping back from the Vector view raised AttributeError."""
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=())
    qtbot.addWidget(win)
    win._redraw_canvas_overlay_if_dirty()


# ------------------------------------------- ROI rows drive which region a tool runs on

def _save(win, row=4, col=5, h=16, w=20):
    win._on_roi_save({"roi_row": row, "roi_col": col, "roi_h": h, "roi_w": w,
                      "boundary": "auto"})
    return win.project.rois[-1]


def test_saving_an_roi_selects_its_row_and_makes_it_active(plain_window):
    win = plain_window
    roi = _save(win)
    item = win.layer_list.currentItem()
    assert item is win.layer_list._roi_items[roi.roi_id]
    assert win._active_roi_id == roi.roi_id


def test_selecting_an_roi_row_then_dropping_runs_on_that_roi(plain_window, qtbot):
    win = plain_window
    win._start_worker = lambda: None
    a = _save(win)
    _save(win, row=20, col=22, h=12, w=12)                   # B is now active
    win.layer_list.setCurrentItem(win.layer_list._roi_items[a.roi_id])
    assert win._active_roi_id == a.roi_id
    _drop(win, qtbot, "tucker_havok")
    child = win.layer
    assert child.name.endswith("@A") and child.tags.get("roi.window") == "4,5,16,20"
    assert win.layer_list._layer_items[child.layer_id].parent() is \
        win.layer_list._roi_items[a.roi_id]


def test_selecting_the_dataset_row_runs_the_next_tool_on_the_whole_field(plain_window, qtbot):
    """The dataset row means the whole field -- it replaces the panel's Deselect."""
    win = plain_window
    win._start_worker = lambda: None
    master = win.layer
    _save(win)
    win.layer_list.setCurrentItem(win.layer_list.topLevelItem(0))
    assert win._active_roi_id is None
    _drop(win, qtbot, "tucker_havok")
    child = win.layer
    assert child.parent_id == master.layer_id and not child.tags.get("roi.window")


def test_hiding_an_roi_row_takes_only_its_outline_off_the_canvas(plain_window):
    win = plain_window
    a = _save(win)
    b = _save(win, row=20, col=22, h=12, w=12)
    win.layer_list._roi_rows[a.roi_id].hide_button.click()
    assert a.visible is False and b.visible is True
    xs = []
    for item in (win.canvas.saved_roi_item, win.canvas.active_roi_item):
        x, _ = item.getData()
        if x is not None:
            xs += [v for v in np.asarray(x) if np.isfinite(v)]
    assert b.col - 0.5 in xs and a.col - 0.5 not in xs


# ------------------------------------------- H never cascades -- each row hides its own drawing

def _roi_child(win, qtbot, device="cdf_edges"):
    with qtbot.waitSignal(win.resolved, timeout=60000):
        _drop(win, qtbot, device)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    return win.layer


def test_the_dataset_rows_h_hides_the_raster_but_not_an_roi_results_drawing(plain_window,
                                                                            qtbot):
    win = plain_window
    roi = _save(win)
    child = _roi_child(win, qtbot)
    drawn = _overlay_counts(win.canvas)
    assert drawn != (0, 0, 0)
    win._on_source_hide_toggled(child.source_id, True)
    assert child.visible is True
    assert _overlay_counts(win.canvas) == drawn                 # the result stays drawn


def test_the_roi_rows_h_leaves_the_results_on_it_drawn(plain_window, qtbot):
    win = plain_window
    roi = _save(win)
    child = _roi_child(win, qtbot)
    drawn = _overlay_counts(win.canvas)
    win.layer_list._roi_rows[roi.roi_id].hide_button.click()
    assert roi.visible is False and child.visible is True
    assert _overlay_counts(win.canvas) == drawn


def test_hiding_a_result_leaves_its_sibling_and_the_dataset_visible(plain_window, qtbot):
    win = plain_window
    master = win.layer
    a = _save(win)
    first = _roi_child(win, qtbot)
    win.layer_list.select_roi(a.roi_id)
    second = _roi_child(win, qtbot, "pm_edges")
    win._on_hide_toggled(first.layer_id, True)
    assert first.visible is False
    assert second.visible is True and master.visible is True
    assert _overlay_counts(win.canvas) != (0, 0, 0)            # second (active) still drawn


# ------------------------------------------- Delete layer / and children / ROI / many

def _whole_field_child(win, qtbot, device="cdf_edges"):
    win.layer_list.select_layer(win.project.layers[0].layer_id)
    return _roi_child(win, qtbot, device)


def test_delete_layer_moves_its_children_up_to_its_parent(plain_window, qtbot):
    win = plain_window
    master = win.layer
    first = _whole_field_child(win, qtbot)
    win._on_refined_run(first.layer_id)
    grandchild = next(l for l in win.project.layers if l.parent_id == first.layer_id)
    win._on_remove_layer_only_requested(first.layer_id)
    assert first not in win.project.layers and grandchild in win.project.layers
    assert grandchild.parent_id == master.layer_id
    assert win.layer_list._layer_items[grandchild.layer_id].parent() is \
        win.layer_list.topLevelItem(0)


def test_an_roi_that_still_has_results_is_not_deleted(plain_window, qtbot):
    win = plain_window
    notes = []
    win._notify = lambda msg, *a, **k: notes.append(msg)
    roi = _save(win)
    child = _roi_child(win, qtbot)
    win._on_remove_roi_requested(roi.roi_id)
    assert roi in win.project.rois and roi.roi_id in win.layer_list._roi_items
    assert any(child.name in n for n in notes)


def test_an_roi_without_results_is_deleted(plain_window):
    win = plain_window
    roi = _save(win)
    win._on_remove_roi_requested(roi.roi_id)
    assert roi not in win.project.rois
    assert roi.roi_id not in win.layer_list._roi_items
    assert win._active_roi_id is None


def test_deleting_several_rows_asks_once_and_takes_an_roi_with_its_results(plain_window, qtbot,
                                                                         monkeypatch):
    from PySide6 import QtWidgets

    win = plain_window
    asked = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a, **k: asked.append(a) or QtWidgets.QMessageBox.Yes)
    one = _whole_field_child(win, qtbot)
    roi = _save(win)
    on_roi = _roi_child(win, qtbot)
    win._on_remove_many_requested([one.layer_id, on_roi.layer_id], [roi.roi_id], [])
    assert len(asked) == 1
    assert one not in win.project.layers and on_roi not in win.project.layers
    assert roi not in win.project.rois


def test_deleting_several_empty_rois_at_once_takes_them_all(plain_window, monkeypatch):
    from PySide6 import QtWidgets

    win = plain_window
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a, **k: QtWidgets.QMessageBox.Yes)
    rois = [_save(win), _save(win, row=20, col=22, h=12, w=12), _save(win, row=2, col=30,
                                                                     h=10, w=10)]
    win._on_remove_many_requested([], [r.roi_id for r in rois], [])
    assert win.project.rois == []


# ------------------------------------------- the dataset's H hides the dataset, not a result's raster

class _ArrangementStub:
    """Stands in for the Vector view: captures the entries the window would draw."""

    def __init__(self):
        self.entries = None

    def set_layers(self, entries):
        self.entries = entries


def test_hiding_the_dataset_keeps_a_results_own_raster(plain_window, qtbot):
    """A whole-field tucker child shows its reconstruction; hiding the DATASET hid it too (the
    image followed the dataset's flag whatever it showed). The dataset's H hides the dataset's
    raster; a result's own raster follows the result's H."""
    win = plain_window
    master = win.layer
    child = _whole_field_child(win, qtbot, "tucker_havok")
    assert win.layer is child and win._holder_raster_ref is not None   # its recon is up
    win._on_source_hide_toggled(master.source_id, True)
    assert win.canvas.image_item.isVisible()
    stub = _ArrangementStub()
    win._arrangement = stub
    win._sync_arrangement(frame_mode=True)                  # the Vector view's own placement
    shown = {e["layer"].layer_id: e["show_raster"] for e in stub.entries}
    assert shown[child.layer_id] is True                    # the recon drape stays
    assert shown[master.layer_id] is False                  # the dataset's own raster goes


def test_hiding_the_dataset_still_hides_it_under_an_extrema_only_result(plain_window, qtbot):
    win = plain_window
    master = win.layer
    child = _whole_field_child(win, qtbot, "cdf_edges")     # Show = edges: no raster product
    assert win.layer is child and win._holder_raster_ref is None
    win._on_source_hide_toggled(master.source_id, True)
    assert not win.canvas.image_item.isVisible()            # the dataset's raster is hidden
    assert _overlay_counts(win.canvas) != (0, 0, 0)          # the result's extrema stay


def test_stepping_tucker_components_relabels_the_row_and_never_recomputes(plain_window, qtbot):
    win = plain_window
    child = _whole_field_child(win, qtbot, "tucker_havok")
    misses = win.cache.misses
    i = win._names.index("tucker_havok")
    win._on_param_changed(i, "show", "component")
    win._on_param_changed(i, "component", 2)
    assert win.cache.misses == misses
    text = win.layer_list.layer_text(child.layer_id)
    assert "· C2 (" in text and text.endswith("%)")
    np.testing.assert_array_equal(win.canvas._field.values,
                                  win._active_result["tucker_components"][1])
