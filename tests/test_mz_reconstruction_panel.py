# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The right panel's Reconstruction section for mz_edges on the LastWave engine.

The section shows while the active chain's output step declares reconstruction knobs, with the
knobs that apply to its algorithm (the others hidden), Run, Stop and the reading of what the recon
row draws. Showing the recon row computes the one-iteration preview; a decay, clipping or coarse
change previews again; Run computes the full reconstruction, continuing from the preview's state.
Stop caches nothing and keeps the preview; an analysis knob turned mid-run cancels it, so a recon
of the old analysis never lands; Run on a cached converged result lands at once, unchanged."""
from __future__ import annotations

import re
import threading

import numpy as np
import pytest

from dynamix.core import mz_lastwave as lw
from dynamix.core.frames import LocalFrame
from dynamix.core.mz_lastwave import recons
from dynamix.core.rasterfield import RasterField
from dynamix.engine.resolve import output_key, resolve
from dynamix.model.device import declared_outputs, defaults_for, get_device

READING = re.compile(r"(\d+) it · (fixed|converged|cap|residual rising) · -?\d+\.\d dB")
SECTION = "Reconstruction"
RECON_KNOBS = ("kappa", "clip", "run_mode", "iterations", "tolerance", "coarse", "mode")
J = 4


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


@pytest.fixture
def values():
    return np.random.default_rng(7).standard_normal((128, 128)).cumsum(0).cumsum(1)


@pytest.fixture
def win(qtbot, builtins, values):
    from dynamix.shell.main_window import MainWindow

    w = MainWindow(steps=())
    qtbot.addWidget(w)
    field = RasterField(name="fbm", values=values, frame=LocalFrame(),
                        x_axis=np.arange(128.0), y_axis=np.arange(128.0))
    w.load_field(field, "/nonexistent/fbm.npz")
    qtbot.waitUntil(lambda: not w.is_computing, timeout=30000)
    return w


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


@pytest.fixture
def held_run(monkeypatch, win):
    """Holds every reconstruction that continues from a state (a Run; the preview starts from
    none) until the returned Event is set; teardown sets it and joins a worker thread still
    running."""
    gate = threading.Event()
    real = lw.e2recons

    def held(*a, **k):
        if k.get("state") is not None:
            gate.wait(30)
        return real(*a, **k)

    monkeypatch.setattr(lw, "e2recons", held)
    yield gate
    gate.set()
    thread = win._thread
    if thread is not None:
        thread.quit()
        thread.wait()


def _mz_layer(win, qtbot, **params):
    """Drop mz_edges (LastWave, J = 4) on the dataset; it forks a child layer and lands."""
    params = {**defaults_for(get_device("mz_edges")), "n_levels": J, **params}
    desc = [{"device": "mz_edges", "params": params}]
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


def _row(win, layer, name):
    return win.layer_list._output_items[layer.layer_id][name][0]


def _section_shown(win):
    return not win.right_panel._sections[SECTION].isHidden()


def _idle(win):
    return not win.is_computing and win._out_request is None


def _show_preview(win, qtbot, layer):
    """Un-hide the recon row and wait for its preview to land; returns the preview's key."""
    win.layer_list.outputHideToggled.emit(layer.layer_id, "recon", False)
    key = _key(win, layer, "recon_preview")
    qtbot.waitUntil(lambda: key in win.cache and _idle(win), timeout=60000)
    return key


def _run(win, qtbot, layer):
    """Press Run and wait for the full reconstruction to land; returns its key."""
    win._recon_panel.run_button.click()
    key = _key(win, layer, "recon")
    qtbot.waitUntil(lambda: key in win.cache and _idle(win), timeout=60000)
    return key


def _preview_of(values, **knobs):
    t, ex = lw.analyze(values, J)
    return lw.e2recons(values, ex, t.S_full[J], J, k=1, mode="fixed", **knobs)


# --------------------------------------------------------------------------- the section
def test_the_section_shows_while_mz_edges_is_in_the_rack_with_the_knobs_that_apply(win, qtbot):
    master = win.layer
    assert not _section_shown(win)
    layer = _mz_layer(win, qtbot)
    assert _section_shown(win)
    panel = win._recon_panel
    assert panel.visible_knobs() == ("kappa", "clip", "run_mode", "iterations", "tolerance",
                                     "coarse")
    assert panel.run_button.isEnabled() and panel.stop_button.isEnabled()
    i = win._names.index("mz_edges")
    assert not set(win.strips.strip(i).controls) & set(RECON_KNOBS)   # the strip shows none
    win.strips.strip(i)._on_control_changed("algorithm", "printed")
    qtbot.waitUntil(lambda: not win.is_computing, timeout=60000)
    assert panel.visible_knobs() == ("iterations", "coarse", "mode")
    assert not panel.run_button.isEnabled()
    assert panel.run_button.toolTip() == "the printed algorithm runs on show"
    win.layer_list.select_layer(master.layer_id)
    qtbot.waitUntil(lambda: win.layer is master and not win.is_computing, timeout=30000)
    assert not _section_shown(win)
    win.layer_list.select_layer(layer.layer_id)
    qtbot.waitUntil(lambda: win.layer is layer and not win.is_computing, timeout=30000)
    assert _section_shown(win) and panel.visible_knobs()[-1] == "mode"


def test_a_section_knob_writes_through_the_steps_box(win, qtbot):
    layer = _mz_layer(win, qtbot)
    i = win._names.index("mz_edges")
    win._recon_panel.controls["iterations"].valueChanged.emit(7)
    assert win._params[i]["iterations"] == 7
    assert win.strips.strip(i)._params["iterations"] == 7
    assert _step(layer).params["iterations"] == 7
    assert win._cache_keys_for(layer)[-1] in win.cache          # view-only: the analysis stays


# --------------------------------------------------------------------------- live preview
def test_showing_recon_previews_one_iteration_and_a_kappa_change_previews_again(
        win, qtbot, dispatches, values):
    layer = _mz_layer(win, qtbot)
    first = _show_preview(win, qtbot, layer)
    assert dispatches == ["recon_preview"]
    assert _key(win, layer, "recon") not in win.cache              # the full recon waits for Run
    want = _preview_of(values)[0].astype(np.float32)
    np.testing.assert_array_equal(win.cache.get(first)["raster"], want)
    np.testing.assert_array_equal(win.canvas._field.values, want)
    note = READING.search(_row(win, layer, "recon").text(0))
    assert note is not None and note.group(1) == "1" and note.group(2) == "fixed"
    assert READING.fullmatch(win._recon_panel.readout.text())
    assert win.layer_list.layer_text(layer.layer_id).endswith("· recon preview")

    win._recon_panel.controls["kappa"].valueChanged.emit(2.0)
    second = _key(win, layer, "recon_preview")
    qtbot.waitUntil(lambda: second in win.cache and _idle(win), timeout=60000)
    assert second != first and dispatches == ["recon_preview", "recon_preview"]
    assert _step(layer).params["kappa"] == 2.0
    want = _preview_of(values, kappa=2.0)[0].astype(np.float32)
    np.testing.assert_array_equal(win.canvas._field.values, want)


def test_run_continues_from_the_preview_state(win, qtbot, dispatches, values):
    layer = _mz_layer(win, qtbot)
    state = win.cache.get(_show_preview(win, qtbot, layer))["state"]
    key = _run(win, qtbot, layer)
    assert dispatches == ["recon_preview", "recon"]
    value = win.cache.get(key)
    assert value["diag"]["iterations"] > 1
    t, ex = lw.analyze(values, J)
    want, diag, _ = lw.e2recons(values, ex, t.S_full[J], J, k=20, mode="converge", tol=1e-3,
                                state=state)
    np.testing.assert_array_equal(value["raster"], want.astype(np.float32))
    np.testing.assert_array_equal(win.canvas._field.values, value["raster"])
    reading = READING.fullmatch(win._recon_panel.readout.text())
    assert reading is not None and int(reading.group(1)) == diag["iterations"]
    assert win.layer_list.layer_text(layer.layer_id).endswith("· recon (edges + coarse)")


def test_run_from_another_row_shows_the_recon(win, qtbot, dispatches):
    layer = _mz_layer(win, qtbot)
    assert _step(layer).params["show"] == "edges"
    _run(win, qtbot, layer)
    assert _step(layer).params["show"] == "recon"
    assert dispatches == ["recon"]                  # its compute runs the preview's pass itself
    assert win.cache.get(_key(win, layer, "recon_preview")) is not None


# --------------------------------------------------------------------------- Stop and supersede
def test_stop_mid_run_caches_nothing_and_keeps_the_preview(win, qtbot, dispatches, held_run):
    layer = _mz_layer(win, qtbot)
    preview = np.array(win.cache.get(_show_preview(win, qtbot, layer))["raster"])
    note = _row(win, layer, "recon").text(0)
    reading = win._recon_panel.readout.text()
    win._recon_panel.run_button.click()
    qtbot.waitUntil(lambda: win._out_job is not None and win._out_job[1] == "recon",
                    timeout=10000)
    key = _key(win, layer, "recon")
    assert _row(win, layer, "recon").text(0) == "recon (edges + coarse) · computing…"
    assert win._recon_panel.readout.text() == "computing…"
    win._recon_panel.stop_button.click()
    held_run.set()
    qtbot.waitUntil(lambda: not win.is_computing, timeout=30000)
    qtbot.wait(100)
    assert not win.is_computing and dispatches == ["recon_preview", "recon"]
    assert key not in win.cache
    np.testing.assert_array_equal(win.canvas._field.values, preview)
    assert _row(win, layer, "recon").text(0) == note
    assert win._recon_panel.readout.text() == reading


def test_changing_levels_mid_run_cancels_it_and_never_shows_the_old_recon(
        win, qtbot, dispatches, held_run, values):
    layer = _mz_layer(win, qtbot)
    state = win.cache.get(_show_preview(win, qtbot, layer))["state"]
    old_key = _key(win, layer, "recon")
    t, ex = lw.analyze(values, J)
    # The engine's own function: the package attribute is held by the fixture.
    old = recons.e2recons(values, ex, t.S_full[J], J, k=20, mode="converge", tol=1e-3,
                          state=state)[0]
    shown = []
    real = win.canvas.set_field

    def spy(field, *a, **k):
        shown.append(np.array(getattr(field, "values", field), dtype=np.float64))
        return real(field, *a, **k)

    win.canvas.set_field = spy
    win._recon_panel.run_button.click()
    qtbot.waitUntil(lambda: win._out_job is not None and win._out_job[1] == "recon",
                    timeout=10000)
    win.strips.strip(win._names.index("mz_edges"))._on_control_changed("n_levels", 3)
    held_run.set()
    qtbot.waitUntil(lambda: _idle(win) and win._cache_keys_for(layer)[-1] in win.cache,
                    timeout=60000)
    new = _key(win, layer, "recon_preview")
    qtbot.waitUntil(lambda: new in win.cache and _idle(win), timeout=60000)
    assert _step(layer).params["n_levels"] == 3
    assert old_key not in win.cache
    assert dispatches == ["recon_preview", "recon", "recon_preview"]
    scale = float(np.abs(old).max())
    assert shown and all(np.abs(v - old).max() > 1e-3 * scale for v in shown)
    np.testing.assert_array_equal(win.canvas._field.values, win.cache.get(new)["raster"])
    assert READING.fullmatch(win._recon_panel.readout.text()).group(1) == "1"


# --------------------------------------------------------------------------- converged
def test_run_on_a_converged_result_lands_at_once_with_the_same_image_and_reading(
        win, qtbot, dispatches):
    layer = _mz_layer(win, qtbot)
    _show_preview(win, qtbot, layer)
    key = _run(win, qtbot, layer)
    assert win.cache.get(key)["diag"]["stop"] == "converged"
    image = np.array(win.canvas._field.values)
    reading = win._recon_panel.readout.text()
    assert "converged" in reading
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win._recon_panel.run_button.click()
    assert not win.is_computing and win._out_request is None
    qtbot.wait(50)
    assert dispatches == ["recon_preview", "recon"]
    np.testing.assert_array_equal(win.canvas._field.values, image)
    assert win._recon_panel.readout.text() == reading
    # nothing dispatched, so no Run stays pending: an evicted recon shown again previews first
    assert win._recon_run is None
