# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The one wheel gesture of every spectrum and figure plot: the wheel zooms the x axis, with
Shift held the y axis. A plot whose two axes must scale together (a locked aspect) keeps
pyqtgraph's own ViewBox and says so beside its construction."""
from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore

__all__ = ["WheelAxisViewBox", "take_in_outside_items", "zoom_plot"]


class WheelAxisViewBox(pg.ViewBox):
    """The wheel zooms the x axis; with Shift held, the y axis. ``outside_bounds`` lists items
    added outside the bounds (``ignoreBounds``, so they never set the view) that View All
    (:meth:`autoRange`) and the auto-range button (:meth:`auto_button_clicked`, see
    :func:`take_in_outside_items`) still take in."""

    outside_bounds: tuple = ()

    def wheelEvent(self, ev, axis=None):
        if axis is None:
            axis = 1 if bool(ev.modifiers() & QtCore.Qt.ShiftModifier) else 0
        super().wheelEvent(ev, axis)

    def autoRange(self, padding=None, items=None, item=None):
        """View All: the items in the bounds and the ones ``outside_bounds`` lists."""
        if items is None and item is None and self.outside_bounds:
            items = [*self.addedItems, *self.outside_bounds]
        super().autoRange(padding=padding, items=items, item=item)

    def auto_button_clicked(self, button) -> None:
        """The auto-range button. With items outside the bounds listed, View All: the view
        stays where it lands, since an auto-range keeps to the bounded items and would drop the
        outside ones at the next redraw; without, auto-range on, as pyqtgraph's own button."""
        if self.outside_bounds:
            self.autoRange()
            return
        self.enableAutoRange()
        button.hide()


def take_in_outside_items(plot) -> None:
    """``plot``'s auto-range button takes in its view box's ``outside_bounds`` too."""
    plot.autoBtn.clicked.disconnect(plot.autoBtnClicked)
    plot.autoBtn.clicked.connect(plot.getViewBox().auto_button_clicked)


def zoom_plot(**kwargs) -> pg.PlotWidget:
    """A ``PlotWidget`` on a :class:`WheelAxisViewBox`; ``kwargs`` as ``PlotWidget``'s own."""
    return pg.PlotWidget(viewBox=WheelAxisViewBox(), **kwargs)
