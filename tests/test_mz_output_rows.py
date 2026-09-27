# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""mz_edges' lazily computed outputs in the window.

A lazy output row (coarse, thumbnail, the two reconstructions, the residual) computes its raster
the first time it is shown, as a job on the worker slot the analysis uses (its progress, the
Stop button), and from then on displays it from the cache in place of the field. A recon row
reads its iterations, status and SNR. The thumbnail draws on its own coarse grid with each block
centred on its sample. Rows that cannot compute grey out with the reason; a stopped job caches
nothing and leaves the analysis alone; a landing for a layer that is gone, or for a key the
layer no longer shows, is ignored."""
from __future__ import annotations

import math
import pathlib
import re
import shutil

import numpy as np
import pytest

from dynamix.core import mz_edges as mz
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.resolve import output_key, resolve
from dynamix.model.device import declared_outputs, defaults_for, get_device

DEM = pathlib.Path(__file__).resolve().parents[1] / "docs" / "demo" / "dem_crop.npz"
NOTE = re.compile(r"(\d+) it · (converged|still improving|rising|diverging) · -?\d+\.\d dB")
LAZY = ("coarse", "thumbnail", "recon", "recon_edges_only", "residual")


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _window(qtbot, field, path):
    from dynamix.shell.main_window import MainWindow

    w = MainWindow(steps=())
    qtbot.addWidget(w)
    w.load_field(field, path)
    qtbot.waitUntil(lambda: not w.is_computing, timeout=30000)
    return w


@pytest.fixture
def dem_path(tmp_path):
    if not DEM.is_file():
        pytest.skip("the demo raster is not in this checkout")
    path = tmp_path / "dem_crop.npz"
    shutil.copy(DEM, path)
    return path


@pytest.fixture
def win(qtbot, builtins, dem_path):
    return _window(qtbot, RasterField.load_npz(str(dem_path)), str(dem_path))


@pytest.fixture
def dispatches(monkeypatch):
    """The names of the output jobs the window starts, in order."""
    import dynamix.shell.main_window as mw

    seen: list = []
    real = mw.OutputWorker

    class _Counted(real):
        def __init__(self, layer, field, cache, source_id, name):
            seen.append(name)
            super().__init__(layer, field, cache, source_id, name)

    monkeypatch.setattr(mw, "OutputWorker", _Counted)
    return seen


def _mz_layer(win, qtbot, **params):
    """Drop mz_edges on the dataset (it spawns a child one tick later) and let it land."""
    desc = [{"device": "mz_edges", "params": {**defaults_for(get_device("mz_edges")), **params}}]
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.strips.set_steps(desc, field=win.field)
        win._on_chain_edited(desc)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    return win.layer


def _step(layer):
    return next(s for s in layer.chain.steps if s.device == "mz_edges")


def _key(win, layer, name):
    """The engine's own key for ``name`` of ``layer`` (its analysis must be cached)."""
    r = resolve(layer, win._fields[layer.layer_id], win.cache, source_id=layer.source_id)
    out = next(o for o in declared_outputs(get_device("mz_edges")) if o.name == name)
    return output_key(r.analysis_device, out, r.analysis_params, r.analysis_key)


def _show(win, qtbot, layer, name):
    """Un-hide ``name``'s row and wait until its output is computed and the window is idle."""
    win.layer_list.outputHideToggled.emit(layer.layer_id, name, False)
    qtbot.waitUntil(lambda: not win.is_computing and win._out_request is None, timeout=60000)
    key = _key(win, layer, name)
    qtbot.waitUntil(lambda: key in win.cache and not win.is_computing, timeout=60000)
    return key


def _row(win, layer, name):
    return win.layer_list._output_items[layer.layer_id][name][0]


def _hidden(win, layer, name):
    return win.layer_list._output_rows[layer.layer_id][name].hide_button.isChecked()


def _close32(got, want) -> None:
    want = np.asarray(want, dtype=np.float64)
    tol = 1e-6 * float(np.abs(want).max())
    np.testing.assert_allclose(np.asarray(got, dtype=np.float64), want, rtol=1e-6, atol=tol)


def _rect(canvas):
    item = canvas.image_item
    r = item.mapRectToView(item.boundingRect())
    return r.x(), r.y(), r.width(), r.height()


# --------------------------------------------------------------------------- rows and display
def test_an_mz_layer_gets_its_output_rows(win, qtbot):
    layer = _mz_layer(win, qtbot)
    group = win.layer_list._output_groups[layer.layer_id]
    assert [group.child(i).text(0) for i in range(group.childCount())] == [
        "edges", "coarse", "thumbnail", "recon (edges + coarse)", "recon (edges only)",
        "residual"]
    assert all(not group.child(i).isDisabled() for i in range(group.childCount()))


def test_showing_recon_computes_it_once_and_displays_it_with_its_reading(win, qtbot,
                                                                        dispatches):
    layer = _mz_layer(win, qtbot)
    analysis_key = win._cache_keys_for(layer)[-1]
    key = _show(win, qtbot, layer, "recon")
    assert dispatches == ["recon"]
    values = win.field.values
    _close32(win.canvas._field.values, mz.reconstruct(values, mz.analyze(values, 4),
                                                      n_iter=10)[0])
    assert win._active_result["raster_out"] is win.cache.get(key)["raster"]
    assert "raster_out" not in win.cache.get(analysis_key)        # the cached result is intact
    text = _row(win, layer, "recon").text(0)
    assert text.startswith("recon (edges + coarse) · ")
    assert NOTE.search(text).group(1) == "10"
    assert not _hidden(win, layer, "recon") and _hidden(win, layer, "coarse")
    assert win.layer_list.layer_text(layer.layer_id).endswith("· recon (edges + coarse)")
    assert _step(layer).params["show"] == "recon"


def test_hiding_and_reshowing_recon_is_a_cache_hit(win, qtbot, dispatches):
    layer = _mz_layer(win, qtbot)
    lid = layer.layer_id
    key = _show(win, qtbot, layer, "recon")
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win.layer_list.outputHideToggled.emit(lid, "recon", True)
    np.testing.assert_array_equal(win.canvas._field.values, win.field.values)
    assert NOTE.search(_row(win, layer, "recon").text(0))          # the reading stays
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win.layer_list.outputHideToggled.emit(lid, "recon", False)
    qtbot.wait(50)
    assert dispatches == ["recon"] and not win.is_computing
    np.testing.assert_array_equal(win.canvas._field.values, win.cache.get(key)["raster"])


def test_changing_iterations_while_shown_recomputes_the_output_only(win, qtbot, dispatches):
    layer = _mz_layer(win, qtbot)
    analysis = win._cache_keys_for(layer)
    first = _show(win, qtbot, layer, "recon")
    win._on_param_changed(win._names.index("mz_edges"), "iterations", 5)
    qtbot.waitUntil(lambda: not win.is_computing and win._out_request is None, timeout=60000)
    assert win._cache_keys_for(layer) == analysis                  # view-only: same analysis
    second = _key(win, layer, "recon")
    assert second != first and second in win.cache
    assert dispatches == ["recon", "recon"]
    assert NOTE.search(_row(win, layer, "recon").text(0)).group(1) == "5"
    assert win.cache.get(second)["diag"]["n_iter"] == 5
    np.testing.assert_array_equal(win.canvas._field.values, win.cache.get(second)["raster"])


def test_a_knob_turn_mid_job_supersedes_it(win, qtbot, dispatches):
    layer = _mz_layer(win, qtbot)
    first = _key(win, layer, "recon")
    win.layer_list.outputHideToggled.emit(layer.layer_id, "recon", False)
    qtbot.waitUntil(lambda: win._out_job is not None, timeout=10000)
    win._on_param_changed(win._names.index("mz_edges"), "iterations", 5)
    qtbot.waitUntil(lambda: not win.is_computing and win._out_request is None
                    and NOTE.search(_row(win, layer, "recon").text(0)) is not None,
                    timeout=60000)
    second = _key(win, layer, "recon")
    assert first not in win.cache and second in win.cache
    assert dispatches == ["recon", "recon"]
    assert NOTE.search(_row(win, layer, "recon").text(0)).group(1) == "5"
    np.testing.assert_array_equal(win.canvas._field.values, win.cache.get(second)["raster"])


def test_stop_during_a_reconstruction_caches_nothing_and_leaves_the_analysis(win, qtbot,
                                                                             dispatches):
    layer = _mz_layer(win, qtbot)
    lid = layer.layer_id
    analysis_state = (win._dispatched, win._stopped_sig, win._active_pending)
    win.layer_list.outputHideToggled.emit(lid, "recon", False)
    qtbot.waitUntil(lambda: win._out_job is not None, timeout=10000)
    assert _row(win, layer, "recon").text(0) == "recon (edges + coarse) · computing…"
    key = _key(win, layer, "recon")
    win._stop_compute()
    qtbot.waitUntil(lambda: not win.is_computing, timeout=30000)
    qtbot.wait(100)
    assert not win.is_computing and dispatches == ["recon"]        # no redispatch
    assert key not in win.cache
    assert _row(win, layer, "recon").text(0) == "recon (edges + coarse)"
    assert (win._dispatched, win._stopped_sig, win._active_pending) == analysis_state
    assert not win._user_stopped
    np.testing.assert_array_equal(win.canvas._field.values, win.field.values)
    # a later show computes it
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win.layer_list.outputHideToggled.emit(lid, "recon", True)
    _show(win, qtbot, layer, "recon")
    assert dispatches == ["recon", "recon"] and key in win.cache


def test_an_analysis_knob_change_mid_reconstruction_never_shows_the_old_recon(win, qtbot,
                                                                              dispatches):
    layer = _mz_layer(win, qtbot)
    old_key = _key(win, layer, "recon")
    values = np.array(win.field.values)
    old = mz.reconstruct(values, mz.analyze(values, 4), n_iter=10)[0]
    shown = []
    real = win.canvas.set_field

    def spy(field, *a, **k):
        shown.append(np.array(getattr(field, "values", field), dtype=np.float64))
        return real(field, *a, **k)

    win.canvas.set_field = spy
    win.layer_list.outputHideToggled.emit(layer.layer_id, "recon", False)
    qtbot.waitUntil(lambda: win._out_job is not None, timeout=10000)
    win._on_param_changed(win._names.index("mz_edges"), "n_levels", 3)
    qtbot.waitUntil(lambda: not win.is_computing and win._out_request is None
                    and NOTE.search(_row(win, layer, "recon").text(0)) is not None,
                    timeout=60000)
    assert _step(layer).params["n_levels"] == 3
    key = _key(win, layer, "recon")
    assert key in win.cache
    assert old_key not in win.cache                      # the superseded job was cancelled
    assert dispatches == ["recon", "recon"]
    scale = float(np.abs(old).max())
    assert all(np.abs(v - old).max() > 1e-3 * scale for v in shown)
    _close32(win.canvas._field.values, mz.reconstruct(values, mz.analyze(values, 3),
                                                      n_iter=10)[0])


def test_removing_the_layer_mid_job_ignores_the_landing(win, qtbot):
    master = win.project.layers[0]
    layer = _mz_layer(win, qtbot)
    win.layer_list.outputHideToggled.emit(layer.layer_id, "recon", False)
    qtbot.waitUntil(lambda: win._out_job is not None, timeout=10000)
    win._on_remove_requested(layer.layer_id, confirm=False)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=30000)
    qtbot.wait(50)
    assert win.layer is master and win._out_job is None
    assert layer.layer_id not in win.layer_list._output_rows
    np.testing.assert_array_equal(win.canvas._field.values, win.field.values)


def test_reopening_a_project_mid_job_ignores_the_landing(win, qtbot, tmp_path):
    layer = _mz_layer(win, qtbot)
    path = win._save_project_to(tmp_path / "session.dynamix")
    win.layer_list.outputHideToggled.emit(layer.layer_id, "recon", False)
    qtbot.waitUntil(lambda: win._out_job is not None, timeout=10000)
    win._open_project_path(path)
    qtbot.waitUntil(lambda: not win.is_computing, timeout=30000)
    qtbot.wait(50)
    assert win._out_job is None and win._out_keys == {}
    assert win.layer is win.project.layers[0] and win.layer is not layer
    np.testing.assert_array_equal(win.canvas._field.values, win.field.values)


def test_a_finished_job_for_a_layer_that_is_gone_is_ignored(win, qtbot):
    from dynamix.model.layer import Layer

    layer = _mz_layer(win, qtbot)
    shown = np.array(win.canvas._field.values)

    class _Thread:
        def quit(self):
            pass

        def wait(self):
            pass

    ghost = Layer(layer_id=layer.layer_id, name="gone", source_id=layer.source_id,
                  chain=layer.chain)
    win._out_job = (ghost, "recon", "a-key-nobody-shows")
    win._thread = _Thread()
    win._on_output_finished({"raster": np.zeros_like(shown, dtype=np.float32), "diag": {}})
    assert win._out_job is None and win._thread is None
    np.testing.assert_array_equal(win.canvas._field.values, shown)


def test_a_finished_job_for_a_key_the_live_layer_no_longer_shows_is_not_displayed(win, qtbot):
    layer = _mz_layer(win, qtbot)
    key = _show(win, qtbot, layer, "recon")
    shown = win.cache.get(key)["raster"]

    class _Thread:
        def quit(self):
            pass

        def wait(self):
            pass

    stale = {"raster": np.zeros_like(shown),
             "diag": {"n_iter": 99, "status": "converged", "snr_db": 0.0}}
    win._out_job = (layer, "recon", "a-key-the-layer-no-longer-shows")
    win._thread = _Thread()
    win._on_output_finished(stale)
    assert win._out_job is None and not win.is_computing
    assert win.layer is layer
    assert win._active_result["raster_out"] is shown        # re-derived from the current key
    np.testing.assert_array_equal(win.canvas._field.values, shown)
    assert NOTE.search(_row(win, layer, "recon").text(0)).group(1) == "10"


def test_a_request_dropped_by_a_layer_switch_clears_its_computing_note(win, qtbot, dispatches):
    master = win.project.layers[0]
    layer = _mz_layer(win, qtbot)
    win.layer_list.outputHideToggled.emit(layer.layer_id, "recon", False)
    assert win._out_request is not None and win._out_request[0] is layer
    assert _row(win, layer, "recon").text(0) == "recon (edges + coarse) · computing…"
    win.layer_list.select_layer(master.layer_id)
    qtbot.waitUntil(lambda: win.layer is master and not win.is_computing, timeout=30000)
    qtbot.wait(50)
    assert win._out_request is None and dispatches == []
    assert _row(win, layer, "recon").text(0) == "recon (edges + coarse)"


def test_the_thumbnail_draws_on_its_own_grid_centred_on_its_samples(win, qtbot):
    from dynamix.core.derivative import raster_choices

    layer = _mz_layer(win, qtbot)
    key = _show(win, qtbot, layer, "thumbnail")
    S = 16
    ny, nx = win.field.values.shape
    value = win.cache.get(key)
    assert value["display_stride"] == S
    np.testing.assert_array_equal(win.canvas._field.values, value["raster"])
    assert win.canvas._field.values.shape == (math.ceil(ny / S), math.ceil(nx / S))
    assert _rect(win.canvas) == (-S / 2, -S / 2, math.ceil(nx / S) * S, math.ceil(ny / S) * S)
    assert "display_stride" not in win.field.provenance           # the shared dict is untouched
    assert layer.layer_id not in win._derived_fields              # a drawing, never a dataset
    assert all(not label.startswith("as shown")                   # the fork does not offer it
               for label, _a in raster_choices(win._active_result))
    # back to the data: the field's own registration
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win.layer_list.outputHideToggled.emit(layer.layer_id, "thumbnail", True)
    assert _rect(win.canvas) == (-0.5, -0.5, nx, ny)


def test_a_sample_anchored_strided_field_centres_each_block_on_its_sample(qtbot):
    from dynamix.shell.canvas import Canvas

    canvas = Canvas()
    qtbot.addWidget(canvas)
    f = RasterField(name="t", values=np.zeros((4, 5)), frame=LocalFrame(),
                    x_axis=np.arange(5.0) * 16, y_axis=np.arange(4.0) * 16)
    f.provenance.update({"display_stride": 16, "full_dims": (60, 70),
                         "display_anchor": "sample"})
    canvas.set_field(f)
    assert _rect(canvas) == (-8.0, -8.0, 80.0, 64.0)
    del f.provenance["display_anchor"]                  # a picture: its sample IS the centre
    canvas.set_field(f)
    assert _rect(canvas) == (-0.5, -0.5, 80.0, 64.0)


# --------------------------------------------------------------------------- greyed-out rows
def test_an_roi_result_greys_out_its_lazy_rows(win, qtbot, dispatches):
    win._on_roi_save({"roi_row": 64, "roi_col": 64, "roi_h": 64, "roi_w": 64,
                      "boundary": "auto"})
    child = _mz_layer(win, qtbot)
    assert child.tags.get("roi.window")
    rows = win.layer_list._output_rows[child.layer_id]
    for name in LAZY:
        item = _row(win, child, name)
        assert item.isDisabled() and item.toolTip(0) == "not on ROI results yet"
        assert not rows[name].hide_button.isEnabled()
    assert not _row(win, child, "edges").isDisabled()
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win._on_param_changed(win._names.index("mz_edges"), "show", "recon")
    qtbot.wait(50)
    assert dispatches == [] and not win.is_computing


def test_a_thumbnail_coarse_on_a_non_divisible_grid_greys_recon_but_draws_the_thumbnail(
        qtbot, builtins, dispatches):
    v = np.random.default_rng(3).standard_normal((60, 60)).cumsum(0).cumsum(1)
    field = RasterField(name="odd", values=v, frame=LocalFrame(), x_axis=np.arange(60.0),
                        y_axis=np.arange(60.0))
    win = _window(qtbot, field, "/nonexistent/odd.npz")
    layer = _mz_layer(win, qtbot, n_levels=4, coarse="thumbnail")
    for name in ("recon", "residual"):
        item = _row(win, layer, name)
        assert item.isDisabled()
        assert item.toolTip(0) == "needs the grid divisible by 2^J (J = 4)"
    for name in ("coarse", "thumbnail", "recon_edges_only"):
        assert not _row(win, layer, name).isDisabled()
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win._on_param_changed(win._names.index("mz_edges"), "show", "recon")
    qtbot.wait(50)
    assert dispatches == [] and not win.is_computing
    np.testing.assert_array_equal(win.canvas._field.values, win.field.values)
    _show(win, qtbot, layer, "thumbnail")                 # the partial last block draws
    assert win.canvas._field.values.shape == (4, 4)
    assert _rect(win.canvas) == (-8.0, -8.0, 64.0, 64.0)
    # Coarse back to full: recon computes
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win._on_param_changed(win._names.index("mz_edges"), "coarse", "full")
    assert not _row(win, layer, "recon").isDisabled()


def test_the_thumbnail_row_greys_out_in_the_vector_views(win, qtbot):
    layer = _mz_layer(win, qtbot)
    win._center_view = "vector"            # the switcher's view (the scene itself is not built)
    win._sync_output_rows(layer)
    item = _row(win, layer, "thumbnail")
    assert item.isDisabled() and item.toolTip(0) == "shown in the 2-D view"
    assert not _row(win, layer, "recon").isDisabled()
    win._center_view = "raster"
    win._sync_output_rows(layer)
    assert not item.isDisabled()


def test_in_the_vector_views_the_active_output_goes_before_background_layers(
        win, qtbot, dispatches, monkeypatch):
    from PySide6 import QtWidgets

    master = win.project.layers[0]
    layer = _mz_layer(win, qtbot)
    background = []
    monkeypatch.setattr(win, "_start_worker_for", background.append)
    monkeypatch.setattr(win, "_arr_tail_key", lambda l: "not-cached")
    win._center_stack.addWidget(QtWidgets.QWidget())       # stands in for the arrangement
    win._center_stack.setCurrentIndex(1)
    win.layer_list.outputHideToggled.emit(layer.layer_id, "recon", False)
    win._arr_queue = [master.layer_id]
    win._dispatch_next()
    assert dispatches == ["recon"] and background == []
    assert win._arr_queue == [master.layer_id]
    win._center_stack.setCurrentIndex(0)
    qtbot.waitUntil(lambda: not win.is_computing and win._out_request is None, timeout=60000)


def test_leaving_the_vector_views_restores_the_thumbnail_row_of_every_layer(win, qtbot):
    master = win.project.layers[0]
    layer = _mz_layer(win, qtbot)
    win._center_view = "vector"            # the switcher's view (the scene itself is not built)
    win._sync_output_rows(layer)
    assert _row(win, layer, "thumbnail").isDisabled()
    win.layer_list.select_layer(master.layer_id)
    qtbot.waitUntil(lambda: win.layer is master and not win.is_computing, timeout=30000)
    win._set_center_view("raster")
    assert not _row(win, layer, "thumbnail").isDisabled()


# --------------------------------------------------------------------------- the rest of the app
def test_the_vector_drape_takes_a_cached_full_resolution_output_only(win, qtbot):
    layer = _mz_layer(win, qtbot)
    seen = []

    class _Arrangement:
        def set_layers(self, entries):
            seen.append({e["layer"].layer_id: e.get("drape") for e in entries
                         if e["status"] == "ok"})

    key = _show(win, qtbot, layer, "recon")
    win._arrangement = _Arrangement()
    win._sync_arrangement(frame_mode=True)
    assert seen[-1][layer.layer_id] is win.cache.get(key)["raster"]
    _show(win, qtbot, layer, "thumbnail")
    win._sync_arrangement(frame_mode=True)
    assert seen[-1][layer.layer_id] is None                        # the raw field drapes


def test_a_shown_recon_forks_like_raster_out(win, qtbot, monkeypatch):
    import dynamix.shell.main_window as mw

    layer = _mz_layer(win, qtbot)
    key = _show(win, qtbot, layer, "recon")
    offered = {}

    class _Fork:
        def __init__(self, **kw):
            offered.update(kw)

        def exec(self):
            return 1

        def choices(self):
            return {"name": "recon", "bands": [0], "vectors": False, "nest": False,
                    "temporary": True}

    monkeypatch.setattr(mw, "ForkDialog", _Fork)
    before = set(win.project.sources)
    win._on_fork_derivative(layer.layer_id)
    (new,) = set(win.project.sources) - before
    assert offered["raster_labels"][0] == "as shown (recon (edges + coarse))"
    forked = RasterField.from_file(win.project.sources[new].path).values
    np.testing.assert_allclose(forked, win.cache.get(key)["raster"])
    qtbot.waitUntil(lambda: not win.is_computing, timeout=30000)


def test_pinning_a_layer_pins_its_computed_outputs(win, qtbot):
    layer = _mz_layer(win, qtbot)
    key = _show(win, qtbot, layer, "recon")
    win._on_freeze_toggled(layer.layer_id, True)
    assert win.cache.is_pinned(key)
    assert win.cache.is_pinned(win._cache_keys_for(layer)[-1])
    win._on_freeze_toggled(layer.layer_id, False)
    assert not win.cache.is_pinned(key)


def test_reopening_restores_the_rows_and_recomputes_the_shown_recon(win, qtbot, tmp_path,
                                                                    dispatches):
    from dynamix.shell.main_window import MainWindow

    layer = _mz_layer(win, qtbot)
    lid = layer.layer_id
    _show(win, qtbot, layer, "recon")
    win.layer_list.outputHideToggled.emit(lid, "edges", True)
    path = win._save_project_to(tmp_path / "session.dynamix")

    win2 = MainWindow(steps=())
    qtbot.addWidget(win2)
    win2._open_project_path(path)
    qtbot.waitUntil(lambda: not win2.is_computing, timeout=30000)
    back = next(l for l in win2.project.layers if l.layer_id == lid)
    assert back.tags["ui.edges_hidden"] == "1" and _step(back).params["show"] == "recon"
    assert _hidden(win2, back, "edges") and not _hidden(win2, back, "recon")
    assert not win2.layer_list._output_groups[lid].isExpanded()
    win2.layer_list.select_layer(lid)
    qtbot.waitUntil(lambda: not win2.is_computing and win2._out_request is None
                    and NOTE.search(_row(win2, back, "recon").text(0)) is not None,
                    timeout=60000)
    key = _key(win2, back, "recon")
    assert dispatches == ["recon", "recon"] and key in win2.cache
    np.testing.assert_array_equal(win2.canvas._field.values, win2.cache.get(key)["raster"])
    assert win2.canvas.extrema_raster_item.isVisible() is False
