# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Transport: the scale sweep, a first-class control.

``SweepClock`` is the pure stepping logic -- no Qt import anywhere in its code path, so it is
constructible and testable fully headless. ``Transport`` is the QFrame that drives a
``SweepClock`` from a QTimer while playing, and from a QSlider drag at any time; both paths
funnel through the same ``_set_index`` -- there is exactly one place that emits ``scaleChanged``
and updates the reading, per the "same code path" requirement (scrub bar and
playback filter the SAME scale-index).

The window supplies ``scale_reading`` (scale px + physical units); the transport itself holds no
unit logic (The Reading Rule lives one level up, where the units are known).
"""
from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtWidgets

PLAY_GLYPH = "▶"
PAUSE_GLYPH = "❚❚"


class SweepClock:
    """Pure scale-index stepper. No Qt import in this class's code path.

    ``tick()`` is the raw advance -- it always advances the position, whether ``playing`` is
    True or False. The caller (a QTimer, in ``Transport``) is what decides WHETHER to call
    ``tick()`` while paused; the clock's own ``playing`` flag is state a UI reads/toggles, not a
    guard ``tick()`` itself enforces.
    """

    def __init__(self, n_scales: int, fps: float = 8.0, loop: bool = True):
        self.n_scales = n_scales
        self.fps = fps
        self.loop = loop
        self.playing = False
        self._index = 0

    @property
    def interval_ms(self) -> float:
        return 1000.0 / self.fps

    @property
    def index(self) -> int:
        return self._index

    def tick(self) -> int:
        """Advance one step and return the new index. Loops back to 0 past the last index when
        ``loop`` is True; otherwise clamps at ``n_scales - 1``."""
        if self._index + 1 < self.n_scales:
            self._index += 1
        elif self.loop:
            self._index = 0
        # else: clamped, stays put
        return self._index

    def scrub(self, idx: int) -> None:
        self._index = idx

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def toggle(self) -> None:
        self.playing = not self.playing


class Transport(QtWidgets.QFrame):
    """Play/pause + scrub + mono reading, one strip-zone-left-edge control.

    ``scale_reading`` formats the current index into the reading text (e.g. ``"a = 8.0 px ≈
    240 m"``) -- supplied by the window, since only it knows the physical units. Scrubbing the
    slider and the QTimer's playback ticks both funnel through ``_set_index``, so there is one
    code path that emits ``scaleChanged`` and refreshes the reading label.
    """

    scaleChanged = QtCore.Signal(int)

    def __init__(self, n_scales: int, scale_reading: Callable[[int], str],
                 fps: float = 8.0, parent=None):
        super().__init__(parent)
        self.clock = SweepClock(n_scales, fps=fps)
        self._scale_reading = scale_reading

        self.play_button = QtWidgets.QPushButton(PLAY_GLYPH)
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._on_play_toggled)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(n_scales - 1, 0))
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider_moved)

        self.reading_label = QtWidgets.QLabel(scale_reading(0))
        self.reading_label.setProperty("reading", "true")

        row = QtWidgets.QHBoxLayout(self)
        row.addWidget(self.play_button)
        row.addWidget(self.slider, 1)
        row.addWidget(self.reading_label)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(round(self.clock.interval_ms)))
        self._timer.timeout.connect(self._on_timer_tick)

    # -- the one code path: scrub and playback both land here ---------------------------------
    def _set_index(self, idx: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.reading_label.setText(self._scale_reading(idx))
        self.scaleChanged.emit(idx)

    def _on_slider_moved(self, value: int) -> None:
        self.clock.scrub(value)
        self._set_index(value)

    def _on_timer_tick(self) -> None:
        self._set_index(self.clock.tick())

    def _on_play_toggled(self, checked: bool) -> None:
        if checked:
            self.clock.play()
            self.play_button.setText(PAUSE_GLYPH)
            self._timer.start()
        else:
            self.clock.pause()
            self.play_button.setText(PLAY_GLYPH)
            self._timer.stop()

    def sync_to(self, idx: int) -> None:
        """Adopt a scale index the transport did NOT originate, without emitting.

        The same scale is addressable from two places -- this widget and the ``scale_select``
        strip's own knob -- and only one of them can be the origin of any given change. When the
        knob is, the transport must still follow: otherwise the slider and its reading show the
        previous scale (so the display contradicts the canvas), and the next playback tick
        advances from the STALE position, snapping the view backwards. Deliberately silent: the
        caller has already acted on this index, so emitting ``scaleChanged`` here would echo it
        back and re-run the filter chain for a change that has already been applied.
        """
        self.clock.scrub(idx)
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.reading_label.setText(self._scale_reading(idx))

    def set_n_scales(self, n: int) -> None:
        """Re-range after a recompute. Clamps the current position into the new range.

        ``setMaximum`` on a slider whose current value sits above the new maximum makes Qt
        auto-clamp the value AND emit ``valueChanged`` synchronously -- which would otherwise
        reach ``_on_slider_moved`` and fire a first ``scaleChanged`` before the explicit clamp
        below fires a second one for the same index. Blocking signals across the ``setMaximum``
        call keeps ``_set_index`` the single place this method emits from.
        """
        self.clock.n_scales = n
        clamped = min(self.clock.index, max(n - 1, 0))
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(n - 1, 0))
        self.slider.blockSignals(False)
        self.clock.scrub(clamped)
        self._set_index(clamped)
