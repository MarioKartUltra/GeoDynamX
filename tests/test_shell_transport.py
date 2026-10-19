# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.transport -- the scale sweep as a first-class control.

``SweepClock`` is pure Python (no Qt import in its own code path) and is exercised fully
headless. ``Transport`` is a QFrame exercised offscreen via pytest-qt: scrubbing and playback
both drive ``scaleChanged`` through the same code path, and the window supplies the reading
text via an injected formatter -- the transport itself holds no unit logic.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PySide6 import QtCore

from dynamix.shell.transport import PAUSE_GLYPH, PLAY_GLYPH, SweepClock, Transport
from tests.test_shell_window import STUB_CHAIN, _FIELD, loaded, stub_devices, window  # noqa: F401


# --- SweepClock (pure, headless) -------------------------------------------------------------

def test_tick_sequence_loops():
    clock = SweepClock(4, loop=True)
    assert [clock.tick() for _ in range(9)] == [1, 2, 3, 0, 1, 2, 3, 0, 1]


def test_scrub_resets_position_and_next_tick_continues_from_there():
    clock = SweepClock(5, loop=True)
    clock.scrub(3)
    assert clock.tick() == 4
    assert clock.tick() == 0


def test_tick_always_advances_regardless_of_playing():
    """DECISION: .tick() is the raw advance the timer calls -- the WIDGET only
    ticks while playing; the clock's own .tick() always advances when called, whether paused
    or playing."""
    clock = SweepClock(3, loop=True)
    assert clock.playing is False           # starts paused
    assert clock.tick() == 1                # tick still advances even though paused
    clock.pause()
    assert clock.tick() == 2


def test_play_pause_toggle():
    clock = SweepClock(3)
    assert clock.playing is False
    clock.play()
    assert clock.playing is True
    clock.pause()
    assert clock.playing is False
    clock.toggle()
    assert clock.playing is True
    clock.toggle()
    assert clock.playing is False


def test_interval_ms_from_fps():
    assert SweepClock(3, fps=8.0).interval_ms == pytest.approx(125.0)
    assert SweepClock(3, fps=50.0).interval_ms == pytest.approx(20.0)


def test_no_loop_clamps_at_last_index():
    clock = SweepClock(3, loop=False)
    assert clock.tick() == 1
    assert clock.tick() == 2
    assert clock.tick() == 2       # stays clamped, does not wrap
    assert clock.tick() == 2


# --- Transport widget (offscreen, qtbot) ------------------------------------------------------

@pytest.fixture
def scale_reading():
    from dynamix.core.scale_units import SIGMA_PER_SCALE_EXACT
    return lambda idx: f"a = {idx} · σ {idx * SIGMA_PER_SCALE_EXACT:.1f} px"


def test_scrub_emits_scale_changed_and_updates_reading(qtbot, scale_reading):
    from dynamix.core.scale_units import SIGMA_PER_SCALE_EXACT
    t = Transport(5, scale_reading)
    qtbot.addWidget(t)
    with qtbot.waitSignal(t.scaleChanged) as sig:
        t.slider.setValue(3)
    assert sig.args == [3]
    expected_reading = f"a = 3 · σ {3 * SIGMA_PER_SCALE_EXACT:.1f} px"
    assert t.reading_label.text() == expected_reading
    assert t.reading_label.property("reading") == "true"


def test_play_button_toggles_text_and_checked_state(qtbot, scale_reading):
    t = Transport(5, scale_reading)
    qtbot.addWidget(t)
    assert t.play_button.text() == PLAY_GLYPH
    assert t.play_button.isChecked() is False

    qtbot.mouseClick(t.play_button, QtCore.Qt.LeftButton)
    assert t.play_button.isChecked() is True
    assert t.play_button.text() == PAUSE_GLYPH

    qtbot.mouseClick(t.play_button, QtCore.Qt.LeftButton)
    assert t.play_button.isChecked() is False
    assert t.play_button.text() == PLAY_GLYPH


def test_playing_ticks_emit_scale_changed_through_timer(qtbot, scale_reading):
    """High fps so a couple of intervals reliably elapse within qtbot.wait's budget; asserts at
    least one scaleChanged fired via the timer -> clock.tick() -> same emit path as scrubbing."""
    t = Transport(5, scale_reading, fps=50.0)
    qtbot.addWidget(t)

    with qtbot.waitSignal(t.scaleChanged, timeout=1000):
        qtbot.mouseClick(t.play_button, QtCore.Qt.LeftButton)   # starts playback
    qtbot.wait(60)                                              # past ~2-3 intervals at 50fps

    assert t.slider.value() != 0 or t.clock.playing is True


def test_sync_to_adopts_an_index_silently(qtbot, scale_reading):
    """``sync_to`` is for a scale change that came from somewhere ELSE (the strip's own knob).
    The slider, the reading and the clock all follow -- and nothing is emitted, because the
    caller has already applied the change and an echo would re-run the chain for nothing."""
    from dynamix.core.scale_units import SIGMA_PER_SCALE_EXACT
    t = Transport(5, scale_reading)
    qtbot.addWidget(t)

    emissions = []
    t.scaleChanged.connect(emissions.append)
    t.sync_to(3)

    assert emissions == []
    assert t.slider.value() == 3
    expected_reading = f"a = 3 · σ {3 * SIGMA_PER_SCALE_EXACT:.1f} px"
    assert t.reading_label.text() == expected_reading
    assert t.clock.index == 3
    assert t.clock.tick() == 4          # playback continues from HERE, not from the stale 0


def test_set_n_scales_clamps_current_position(qtbot, scale_reading):
    t = Transport(5, scale_reading)
    qtbot.addWidget(t)
    t.slider.setValue(4)
    assert t.slider.value() == 4

    t.set_n_scales(3)                 # new count no longer reaches index 4
    assert t.slider.maximum() == 2
    assert t.slider.value() == 2      # clamped into range


def test_set_n_scales_reranges_slider_upward(qtbot, scale_reading):
    t = Transport(3, scale_reading)
    qtbot.addWidget(t)
    t.set_n_scales(8)
    assert t.slider.minimum() == 0
    assert t.slider.maximum() == 7


def test_set_n_scales_emits_scale_changed_exactly_once_when_clamping(qtbot, scale_reading):
    """set_n_scales(3) on a Transport scrubbed to index 4 must clamp the slider's out-of-range
    value AND emit scaleChanged exactly once -- not once from Qt auto-clamping setMaximum() on
    an unblocked slider, and once more from the explicit clamp call."""
    t = Transport(5, scale_reading)
    qtbot.addWidget(t)
    t.slider.setValue(4)

    emissions = []
    t.scaleChanged.connect(emissions.append)
    t.set_n_scales(3)

    assert emissions == [2]
    assert t.slider.maximum() == 2


def test_set_n_scales_emits_at_most_once_when_growing(qtbot, scale_reading):
    t = Transport(5, scale_reading)
    qtbot.addWidget(t)
    t.slider.setValue(1)

    emissions = []
    t.scaleChanged.connect(emissions.append)
    t.set_n_scales(8)

    assert len(emissions) <= 1


# --- The shown level: worker, window, project (MainWindow wiring) ----------------------------

def test_the_resolve_worker_resolves_at_the_level_it_is_handed(qtbot, stub_devices):
    from dynamix.engine.cache import Cache
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.layer import Layer
    from dynamix.shell.worker import ResolveWorker

    chain = Chain((DeviceRef("stub_stack", {}),
                   DeviceRef("modulus_threshold", {"frac": 0.0}))).materialized()
    layer = Layer(layer_id=1, name="stub", source_id="mem:worker", chain=chain)
    landed = []
    worker = ResolveWorker(layer, _FIELD, Cache(), "mem:worker", level=2)
    worker.finished.connect(landed.append)
    worker._run()                              # on this thread: the signal is delivered directly

    (renderable,) = landed
    assert renderable.result["_scale_idx"] == 2
    assert len(renderable.result["extrema"]) == 1
    assert "ui.scale_idx" not in layer.tags    # the worker never writes view state


def test_the_transport_sets_the_shown_level_and_leaves_the_chain(qtbot, loaded):
    from dynamix.model.contracts import StepState

    chain = loaded.layer.chain
    with qtbot.waitSignal(loaded.resolved, timeout=5000) as sig:
        loaded.transport.slider.setValue(2)
    renderable = sig.args[0]
    assert loaded.layer.tags["ui.scale_idx"] == "2"
    assert loaded.layer.chain is chain                         # a scrub never edits the chain
    assert loaded._params[1]["scale_idx"] == 0                 # the saved step keeps its value
    assert renderable.result["_scale_idx"] == 2
    assert renderable.cache_misses == 0
    assert renderable.step_states[1] == StepState(False, "the scale is set by the Transport")
    assert loaded._shown_level() == 3                          # 1-based, for per-level filters
    assert loaded._current_scale_px() == loaded._scales[2]


def test_the_wtmm_holder_rack_scrubs(qtbot, fbm64, clean_registry):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.devices import register_builtin_devices
    from dynamix.model.presets import PRESETS
    from dynamix.shell.main_window import MainWindow

    register_builtin_devices()
    field = RasterField(name="fbm", values=fbm64, frame=LocalFrame(units="px"),
                        x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    win = MainWindow(steps=PRESETS["WTMM + Hölder"])
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=60000) as sig:
        win.load_field(field, "mem:wtmm-holder")
    first = sig.args[0].result
    assert len(first["extrema"]) == 1 and first["_scale_idx"] == 0
    assert win.transport.clock.n_scales == len(first["scales"]) > 3

    with qtbot.waitSignal(win.resolved, timeout=10000) as swept:
        win.transport.slider.setValue(3)

    r = swept.args[0]
    assert r.cache_misses == 0
    assert r.result["_scale_idx"] == 3 and len(r.result["extrema"]) == 1
    assert len(r.result["chains"]) == len(first["chains"])     # V-chains are not a level's
    assert win._active_result["_scale_idx"] == 3                # what the canvas was given
    assert win.layer.tags["ui.scale_idx"] == "3"
    assert [s.device for s in win.layer.chain.steps] == [n for n, _ in PRESETS["WTMM + Hölder"]]


def test_a_derivative_fork_writes_the_shown_level(qtbot, stub_devices, monkeypatch):
    import dynamix.core.derivative as derivative
    import dynamix.shell.main_window as mw
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.shell.main_window import MainWindow

    field = RasterField(name="fork", values=np.asarray(_FIELD), frame=LocalFrame(units="px"),
                        x_axis=np.arange(16.0), y_axis=np.arange(16.0))
    win = MainWindow(steps=(("stub_stack", {}), ("modulus_threshold", {"frac": 0.0})))
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(field, "mem:fork-level")
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win.transport.slider.setValue(2)
    offered, written = {}, {}

    class _Fork:
        def __init__(self, **kw):
            offered.update(kw)

        def exec(self):
            return 1

        def choices(self):
            return {"name": "level", "bands": [], "vectors": True, "nest": False,
                    "temporary": True}

    def _write(path, **kw):
        written.update(kw)
        raise ValueError("held")                  # stop before the file opens as a dataset

    monkeypatch.setattr(mw, "ForkDialog", _Fork)
    monkeypatch.setattr(derivative, "write_derivative", _write)
    win._on_fork_derivative(win.layer.layer_id)

    (shown,) = win._active_result["extrema"]
    assert offered["vector_counts"][0] == len(shown["x"])
    (kept,) = written["vectors"]["extrema"]
    np.testing.assert_array_equal(kept["x"], shown["x"])
    assert written["vectors"]["_scale_idx"] == 2


def _saved_project(qtbot, tmp_path, name, level=None):
    """A one-layer project over a real raster file (``STUB_CHAIN``), shown at ``level`` when
    given, saved; returns the project file."""
    from dynamix.shell.main_window import MainWindow

    raster = tmp_path / f"{name}.npz"
    np.savez(raster, np.asarray(_FIELD, dtype=np.float64))
    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.open_path(str(raster))
    if level is not None:
        with qtbot.waitSignal(win.resolved, timeout=5000):
            win.transport.slider.setValue(level)
    return win._save_project_to(tmp_path / f"{name}.dynamix")


def test_the_shown_level_persists_through_save_and_open(qtbot, stub_devices, tmp_path):
    from dynamix.shell.main_window import MainWindow

    written = _saved_project(qtbot, tmp_path, "level", level=2)
    assert json.loads(written.read_text(encoding="utf-8"))["layers"][0]["tags"]["ui.scale_idx"] == "2"

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000) as sig:
        win._open_project_path(written)
    assert win.layer.tags["ui.scale_idx"] == "2"
    assert win.transport.slider.value() == 2
    assert sig.args[0].result["_scale_idx"] == 2


def test_an_old_project_with_scale_select_opens_at_its_scale_and_shows_the_step_inactive(
        qtbot, stub_devices, tmp_path):
    from dynamix.model.contracts import StepState
    from dynamix.shell.main_window import MainWindow

    written = _saved_project(qtbot, tmp_path, "old")
    payload = json.loads(written.read_text(encoding="utf-8"))
    (saved,) = payload["layers"]
    saved["tags"].pop("ui.scale_idx", None)              # written before the Transport owned it
    (step,) = [s for s in saved["chain"]["steps"] if s["device"] == "scale_select"]
    step["params"]["scale_idx"] = 2
    written.write_text(json.dumps(payload), encoding="utf-8")

    idle = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(idle)
    idle._start_worker = lambda: None
    idle._open_project_path(written)
    assert idle.layer.tags["ui.scale_idx"] == "2"        # seeded on open, before any landing

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000) as sig:
        win._open_project_path(written)
    assert win.layer.tags["ui.scale_idx"] == "2"
    assert win.transport.slider.value() == 2
    assert sig.args[0].result["_scale_idx"] == 2
    assert win.layer.chain.steps[1].params["scale_idx"] == 2      # the chain is not edited
    assert sig.args[0].step_states[1] == StepState(False, "the scale is set by the Transport")


def test_the_anisotropy_window_reads_every_level_the_chain_keeps(qtbot, stub_devices):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=(("stub_stack", {}), ("modulus_threshold", {"frac": 0.0})))
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD, "mem:aniso-stack")
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win.transport.slider.setValue(1)
    assert len(win._active_result["extrema"]) == 1          # the canvas shows the shown level
    fed = {}

    class _Window:
        def set_result(self, result, field, *, offset, scale_idx, **extras):
            fed.update(result=result, scale_idx=scale_idx, **extras)

    win._feed_anisotropy(_Window())

    assert len(fed["result"]["extrema"]) == 3                # the window reads across scales
    assert fed["scale_idx"] == 1
    assert fed["fit_window"] is None and fed["directional"] == []    # no fitter open, no leaders
    assert win.is_computing is False


def test_the_anisotropy_window_reads_every_level_past_a_saved_scale_select(qtbot, stub_devices):
    """A chain that still holds ``scale_select`` (the default chain, an old project) keeps its
    pick for nothing: the window reads every level, as it does without the step."""
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=(("stub_stack", {}), ("scale_select", {"scale_idx": 0}),
                            ("modulus_threshold", {"frac": 0.0})))
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_FIELD, "mem:aniso-scale-select")
    with qtbot.waitSignal(win.resolved, timeout=5000):
        win.transport.slider.setValue(1)
    fed = {}

    class _Window:
        def set_result(self, result, field, *, offset, scale_idx, **extras):
            fed.update(result=result, scale_idx=scale_idx, **extras)

    win._feed_anisotropy(_Window())

    assert len(fed["result"]["extrema"]) == 3
    assert fed["scale_idx"] == 1
    assert any(ref.device == "scale_select" for ref in win.layer.chain.steps)   # chain untouched
    assert win.is_computing is False


def test_a_pending_layers_scale_select_strip_shows_inactive(qtbot, loaded):
    """A pending layer's strips are judged on their own recipe (``_strip_states``' fallback),
    under the layer's shown level as the engine judges a landing: the scale_select strip stays
    inactive, with the Transport's reason, and is never bypassed."""
    from dynamix.model.contracts import SCALE_BY_TRANSPORT

    with qtbot.waitSignal(loaded.resolved, timeout=5000):
        loaded.transport.slider.setValue(2)
    assert loaded.strips.strip(1).inactive_reason == SCALE_BY_TRANSPORT
    loaded._auto_run_action.setChecked(False)
    loaded._on_param_changed(0, "n_scales", 2)                # a transform edit waits for Run
    assert loaded.layer.layer_id in loaded._pending_layers
    with qtbot.waitSignal(loaded.resolved, timeout=5000) as sig:
        loaded._on_param_changed(3, "frac", 0.1)              # a filter edit lands at once
    assert sig.args[0].result["_scale_idx"] == 2
    box = loaded.strips.strip(1)
    assert box.inactive_reason == SCALE_BY_TRANSPORT
    assert box.reading_label.text() == SCALE_BY_TRANSPORT
    assert loaded._bypassed[1] is False


def test_the_browser_greys_scale_while_the_layer_has_a_shown_level(qtbot, loaded):
    from dynamix.model.contracts import SCALE_BY_TRANSPORT
    from dynamix.model.device import get_device
    from dynamix.model.families import display_name, tooltip_of

    row = loaded.browser._rows["scale_select"][0]
    assert loaded.layer.tags["ui.scale_idx"] == "0"           # the landing wrote the level
    assert row.toolTip(0) == SCALE_BY_TRANSPORT
    assert row.foreground(0).color() == loaded.browser._muted.color()
    del loaded.layer.tags["ui.scale_idx"]                     # no shown level
    loaded._sync_browser_states()
    assert row.toolTip(0) == tooltip_of(get_device("scale_select"))
