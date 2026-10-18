# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Curve window: the curves a tool publishes (:mod:`dynamix.model.curves`), one at a time.

A floating window with one plot: a combo names the curve, two checks switch each axis between
linear and log (each curve brings its own defaults), and the curve's marks are vertical lines
with their labels. The first feed is the Periodic extension's power spectrum with its band limit;
the measures' curves against β and the lineal path follow the same way.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.shell.theme import current

__all__ = ["CurveWindow"]


class CurveWindow(QtWidgets.QWidget):
    """A floating window; :meth:`set_curves` feeds it the active result's curves."""

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Curves")
        from dynamix.shell.window_fit import fit_to_screen

        fit_to_screen(self, 720, 520)
        self._curves: list = []
        outer = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Curve"))
        self.curve_combo = QtWidgets.QComboBox()
        bar.addWidget(self.curve_combo, 1)
        self.logx_check = QtWidgets.QCheckBox("log x")
        self.logy_check = QtWidgets.QCheckBox("log y")
        bar.addWidget(self.logx_check)
        bar.addWidget(self.logy_check)
        outer.addLayout(bar)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(current().ground)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        for side in ("bottom", "left"):           # plain decimals, never an "(x0.001)" prefix
            self.plot.plotItem.getAxis(side).enableAutoSIPrefix(False)
        outer.addWidget(self.plot, 1)
        self.note = QtWidgets.QLabel("")
        self.note.setProperty("reading", "true")
        self.note.setWordWrap(True)
        outer.addWidget(self.note)
        self.curve_combo.currentIndexChanged.connect(self._on_curve_picked)
        self.logx_check.toggled.connect(self.refresh)
        self.logy_check.toggled.connect(self.refresh)

    def set_curves(self, curves: list, title: str = "") -> None:
        """The curves on offer; the pick is kept by name across feeds when it still exists."""
        picked = self.curve_combo.currentText()
        self._curves = list(curves)
        self.setWindowTitle(f"Curves — {title}" if title else "Curves")
        self.curve_combo.blockSignals(True)
        self.curve_combo.clear()
        for c in self._curves:
            self.curve_combo.addItem(c.name)
        names = [c.name for c in self._curves]
        self.curve_combo.setCurrentIndex(names.index(picked) if picked in names else 0)
        self.curve_combo.blockSignals(False)
        self._adopt_defaults()
        self.refresh()

    def current(self):
        i = self.curve_combo.currentIndex()
        return self._curves[i] if 0 <= i < len(self._curves) else None

    def _adopt_defaults(self) -> None:
        c = self.current()
        if c is None:
            return
        for check, value in ((self.logx_check, c.logx), (self.logy_check, c.logy)):
            check.blockSignals(True)
            check.setChecked(bool(value))
            check.blockSignals(False)

    def _on_curve_picked(self, *_a) -> None:
        self._adopt_defaults()
        self.refresh()

    def refresh(self, *_a) -> None:
        self.plot.clear()
        c = self.current()
        if c is None:
            self.note.setText("")
            return
        logx, logy = self.logx_check.isChecked(), self.logy_check.isChecked()
        x, y = np.asarray(c.x, dtype=float), np.asarray(c.y, dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        if logx:
            keep &= x > 0
        if logy:
            keep &= y > 0
        self.plot.setLogMode(x=logx, y=logy)
        self.plot.plot(x[keep], y[keep], pen=pg.mkPen(current().ink, width=1.6), symbol="o",
                       symbolSize=4, symbolPen=None, symbolBrush=current().ink)
        for xm, label in c.marks:
            if not np.isfinite(xm) or (logx and not xm > 0):
                continue
            line = pg.InfiniteLine(pos=np.log10(xm) if logx else xm, angle=90, movable=False,
                                   pen=pg.mkPen(current().amber), label=label,
                                   labelOpts={"position": 0.9, "color": current().amber})
            self.plot.addItem(line)
        self.plot.setLabel("bottom", c.xlabel)
        self.plot.setLabel("left", c.ylabel)
        self.note.setText(c.note)
