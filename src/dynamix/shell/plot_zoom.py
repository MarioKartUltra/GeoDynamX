# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The one wheel gesture of every spectrum and figure plot: the wheel zooms the x axis, with
Shift held the y axis. A plot whose two axes must scale together (a locked aspect) keeps
pyqtgraph's own ViewBox and says so beside its construction."""
from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore

__all__ = ["WheelAxisViewBox", "zoom_plot"]


class WheelAxisViewBox(pg.ViewBox):
    """The wheel zooms the x axis; with Shift held, the y axis."""

    def wheelEvent(self, ev, axis=None):
        if axis is None:
            axis = 1 if bool(ev.modifiers() & QtCore.Qt.ShiftModifier) else 0
        super().wheelEvent(ev, axis)


def zoom_plot(**kwargs) -> pg.PlotWidget:
    """A ``PlotWidget`` on a :class:`WheelAxisViewBox`; ``kwargs`` as ``PlotWidget``'s own."""
    return pg.PlotWidget(viewBox=WheelAxisViewBox(), **kwargs)
