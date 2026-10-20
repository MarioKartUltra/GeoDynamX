# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""One wheel gesture on every plot of the spectra and figures: the wheel zooms x, Shift + wheel
zooms y. A source scan keeps it that way for the next plot anyone adds."""
from __future__ import annotations

import pathlib
import re

import pyqtgraph as pg
from PySide6 import QtCore

from dynamix.shell.plot_zoom import WheelAxisViewBox, zoom_plot


class _Wheel:
    """What ``ViewBox.wheelEvent`` reads off a scene wheel event."""

    def __init__(self, delta: float, shift: bool) -> None:
        self._delta, self._shift = delta, shift

    def delta(self):
        return self._delta

    def modifiers(self):
        return QtCore.Qt.ShiftModifier if self._shift else QtCore.Qt.NoModifier

    def pos(self):
        return QtCore.QPointF(0.0, 0.0)

    def accept(self):
        pass


def test_the_wheel_zooms_one_axis_at_a_time(qtbot):
    plot = zoom_plot(title="t")
    qtbot.addWidget(plot)
    vb = plot.getViewBox()
    assert isinstance(vb, WheelAxisViewBox)
    vb.setRange(xRange=(0, 10), yRange=(0, 10), padding=0)
    vb.wheelEvent(_Wheel(120, shift=False))
    (x0, x1), (y0, y1) = vb.viewRange()
    assert (x1 - x0) < 10 and abs((y1 - y0) - 10) < 1e-9          # x zoomed, y untouched
    vb.setRange(xRange=(0, 10), yRange=(0, 10), padding=0)
    vb.wheelEvent(_Wheel(120, shift=True))
    (x0, x1), (y0, y1) = vb.viewRange()
    assert abs((x1 - x0) - 10) < 1e-9 and (y1 - y0) < 10          # y zoomed, x untouched


#: Plots that zoom both axes on purpose: the raster canvas (its own ViewBox), the map (a
#: locked-aspect projection), the levels picker's histogram and the demo.
_BOTH_AXES_FILES = {"canvas.py", "map_view.py", "levels_dialog.py", "demo.py"}
_BOTH_AXES_MARK = "# both-axes zoom"


def test_every_plot_in_the_shell_takes_the_gesture_or_says_why_not():
    shell = pathlib.Path(__file__).resolve().parents[1] / "src" / "dynamix" / "shell"
    offenders = []
    for path in sorted(shell.glob("*.py")):
        if path.name in _BOTH_AXES_FILES:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"pg\.PlotWidget\(|\.addPlot\(", line) and "viewBox=" not in line \
                    and _BOTH_AXES_MARK not in line:
                offenders.append(f"{path.name}:{number}")
    assert offenders == [], offenders
