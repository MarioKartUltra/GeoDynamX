# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""InpaintHistogramsWindow: did the fill keep the field's cascade? Per level and orientation,
the ``ln|eta|`` histogram of the parent-child pairs inside the regions against the clear
pairs (and the abacus the cascade route used), and the ``ln|c|`` amplitude histograms, with
their Kolmogorov distances (Pottier, Turiel & Garcon 2008, Fig. 14). Reads the active result's
``_hists`` and ``_abacus``; nothing recomputes. Lifecycle is main_window's: one window at a
time, refreshed from the active result."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

__all__ = ["InpaintHistogramsWindow"]

_ORIENTATIONS = ("horizontal", "vertical", "diagonal")


class InpaintHistogramsWindow(QtWidgets.QWidget):
    def __init__(self, result: dict, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Inpainting — ln|η| and amplitude histograms")
        self.resize(960, 420)
        outer = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Level · orientation"))
        self.level_combo = QtWidgets.QComboBox()
        bar.addWidget(self.level_combo)
        self.note = QtWidgets.QLabel("")
        bar.addWidget(self.note, 1)
        outer.addLayout(bar)
        plots = QtWidgets.QHBoxLayout()
        self.eta_plot = pg.PlotWidget(title="ln|η| = ln|c| − ln|p|: inside vs clear")
        self.eta_plot.addLegend(offset=(-5, 5))
        self.amp_plot = pg.PlotWidget(title="ln|c|: inside vs clear")
        self.amp_plot.addLegend(offset=(-5, 5))
        plots.addWidget(self.eta_plot, 1)
        plots.addWidget(self.amp_plot, 1)
        outer.addLayout(plots, 1)
        outer.addWidget(QtWidgets.QLabel(
            "A fill that kept the cascade has inside histograms on top of the clear ones; "
            "consistent ratios alone do not show the amplitudes are right, so both are read."))
        self._result = {}
        self._keys: list = []
        self.level_combo.currentIndexChanged.connect(self.refresh)
        self.set_result(result)

    def set_result(self, result: dict) -> None:
        self._result = result or {}
        hists = self._result.get("_hists") or {}
        self._keys = sorted(hists)
        self.level_combo.blockSignals(True)
        self.level_combo.clear()
        for j, o in self._keys:
            self.level_combo.addItem(f"level {j} · {_ORIENTATIONS[o]}")
        self.level_combo.blockSignals(False)
        self.refresh()

    def refresh(self) -> None:
        self.eta_plot.clear()
        self.amp_plot.clear()
        i = self.level_combo.currentIndex()
        if i < 0 or i >= len(self._keys):
            self.note.setText("no histograms on this result")
            return
        key = self._keys[i]
        entry = self._result["_hists"][key]
        for plot, part in ((self.eta_plot, entry["eta"]), (self.amp_plot, entry["amp"])):
            edges = np.asarray(part["edges"], dtype=float)
            for name, hist, color in (("inside", part["inside"], "#e15759"),
                                      ("clear", part["clear"], "#4e79a7")):
                h = np.asarray(hist, dtype=float)
                plot.plot(edges, h / max(h.sum(), 1.0), stepMode="center", name=name,
                          pen=pg.mkPen(color, width=2))
        abacus = self._result.get("_abacus")
        if abacus is not None:
            edges = np.asarray(abacus["edges"], dtype=float)
            self.eta_plot.plot(edges, np.asarray(abacus["hist"])[key[1]], stepMode="center",
                               name="abacus", pen=pg.mkPen("#59a14f", style=QtCore.Qt.DashLine))
        self.note.setText(f"Kolmogorov distance: ln|η| {entry['eta']['distance']:.3f} "
                          f"(inside {entry['eta']['n_inside']}, clear {entry['eta']['n_clear']}) · "
                          f"ln|c| {entry['amp']['distance']:.3f}")
