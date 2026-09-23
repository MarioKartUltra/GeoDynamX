# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for dynamix.shell.transport -- the scale sweep as a first-class control.

``SweepClock`` is pure Python (no Qt import in its own code path) and is exercised fully
headless. ``Transport`` is a QFrame exercised offscreen via pytest-qt: scrubbing and playback
both drive ``scaleChanged`` through the same code path, and the window supplies the reading
text via an injected formatter -- the transport itself holds no unit logic.
"""
from __future__ import annotations

import pytest
from PySide6 import QtCore

from dynamix.shell.transport import PAUSE_GLYPH, PLAY_GLYPH, SweepClock, Transport


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
