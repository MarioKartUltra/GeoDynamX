# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A range regression and a two-click manual segment over the same measured series."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core.figure_fits import manual_slope, range_fit
from dynamix.shell.theme import current
from dynamix.shell.plot_zoom import zoom_plot


class SlopePlot(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, title, *, quantity="slope", factor=1., parent=None):
        super().__init__(parent)
        self.quantity, self.factor = quantity, factor
        self.x, self.y = np.array([]), np.array([])
        self.measurement = {}
        self._anchor = None
        self._points = None
        self._seeded = False
        self._editing = False
        self.paper_mode = False
        layout = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Range", "Manual"])
        bar.addWidget(self.mode)
        self.redraw = QtWidgets.QPushButton("Draw / clear")
        bar.addWidget(self.redraw)
        self.readout = QtWidgets.QLabel()
        bar.addWidget(self.readout, 1)
        layout.addLayout(bar)
        self.numeric = QtWidgets.QWidget()
        numbers = QtWidgets.QHBoxLayout(self.numeric)
        numbers.setContentsMargins(0, 0, 0, 0)
        self.endpoints = []
        for label in ("X₁", "Y₁", "X₂", "Y₂"):
            numbers.addWidget(QtWidgets.QLabel(label))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(6)
            spin.valueChanged.connect(self._numbers)
            numbers.addWidget(spin)
            self.endpoints.append(spin)
        layout.addWidget(self.numeric)
        self.numeric.hide()
        self.plot = zoom_plot(title=title)
        self.plot.setBackground(current().ground)
        self.plot.setMinimumSize(230, 200)
        layout.addWidget(self.plot, 1)
        self.region = pg.LinearRegionItem([0, 1], brush=(160, 160, 160, 18))
        self.region.sigRegionChangeFinished.connect(self._fit)
        self.segment = pg.PlotDataItem(pen=pg.mkPen(current().ink, width=1.8))
        self.preview = pg.PlotDataItem(pen=pg.mkPen(current().amber, style=QtCore.Qt.DashLine))
        self.roi = None
        self.plot.scene().sigMouseClicked.connect(self._click)
        self.plot.scene().sigMouseMoved.connect(self._move)
        self.plot.installEventFilter(self)
        self.mode.currentTextChanged.connect(self._mode)
        self.redraw.clicked.connect(self.clear_manual)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Escape:
            self._anchor = None
            self.preview.setData([], [])
            return True
        return super().eventFilter(obj, event)

    def set_data(self, x, y):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.plot.clear()
        ink = current().ground if self.paper_mode else current().ink
        self.plot.plot(self.x, self.y, pen=None, symbol="o", symbolSize=4,
                       symbolBrush=None, symbolPen=pg.mkPen(ink))
        self.segment.setPen(pg.mkPen(ink, width=1.8))
        if not self._seeded and self.x.size:
            finite = self.x[np.isfinite(self.x)]
            if finite.size:
                self.region.setRegion([float(finite.min()), float(finite.max())])
                self._seeded = True
        for item in (self.region, self.segment, self.preview):
            self.plot.addItem(item)
        if self.roi is not None:
            self.plot.addItem(self.roi)
        self._mode()

    def _mode(self, *_):
        manual = self.mode.currentText() == "Manual"
        self.numeric.setVisible(manual)
        self.region.setVisible(not manual)
        self.preview.setVisible(manual)
        if self.roi is not None:
            self.roi.setVisible(manual)
        self._fit()

    def _fit(self, *_):
        if self.mode.currentText() == "Manual":
            if self._points is None:
                self.readout.setText("Click start, then end · Esc cancels")
                self.segment.setData([], [])
                self.measurement = {}
                return
            result = manual_slope(self._points, factor=self.factor)
            xx = np.asarray(self._points)[:, 0]
            suffix = "manual slope"
        else:
            result = range_fit(self.x, self.y, *self.region.getRegion(), factor=self.factor)
            xx = np.asarray(result.get("included",result["bounds"]))
            suffix = f"SE {result['se']:.3g} · n={result['count']}"
        self.measurement = result
        if np.isfinite(result["slope"]):
            self.segment.setData(xx, result["slope"]*xx+result["intercept"])
            self.readout.setText(f"{self.quantity} {result['value']:.4g} · {suffix}")
        else:
            self.segment.setData([], [])
            self.readout.setText("Undefined slope" if self.mode.currentText() == "Manual" else "Need ≥3 finite bins")
        self.changed.emit()

    def clear_manual(self):
        self._anchor = self._points = None
        if self.roi is not None:
            self.plot.removeItem(self.roi)
            self.roi = None
        self.preview.setData([], [])
        self._fit()

    def _position(self, scene_pos):
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            return None
        p = vb.mapSceneToView(scene_pos)
        return [p.x(), p.y()]

    def _click(self, event):
        if self.mode.currentText() != "Manual" or event.button() != QtCore.Qt.LeftButton or self._points is not None:
            return
        p = self._position(event.scenePos())
        if p is None:
            return
        event.accept()
        if self._anchor is None:
            self._anchor = p
        else:
            self.set_segment([self._anchor, p])

    def _move(self, pos):
        if self.mode.currentText() == "Manual" and self._anchor is not None:
            p = self._position(pos)
            if p is not None:
                points = np.asarray([self._anchor, p])
                self.preview.setData(points[:, 0], points[:, 1])
                m = manual_slope(points, factor=self.factor)
                self.readout.setText(f"{self.quantity} {m['value']:.4g} · click to finish")

    def set_segment(self, points):
        self.clear_manual()
        self._points = np.asarray(points, float).tolist()
        self.roi = pg.LineSegmentROI(self._points, pen=pg.mkPen(current().amber, width=1.2))
        self.plot.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self._dragged)
        self._anchor = None
        self.preview.setData([], [])
        self.mode.setCurrentText("Manual")
        self._fit()
        self._sync_numbers()

    def _dragged(self):
        self._points = [[p.x(), p.y()] for _, p in self.roi.getLocalHandlePositions()]
        origin = self.roi.pos()
        self._points = [[x+origin.x(), y+origin.y()] for x, y in self._points]
        self._fit()
        self._sync_numbers()

    def _sync_numbers(self):
        if self._points is None:
            return
        self._editing = True
        for spin,value in zip(self.endpoints, np.asarray(self._points).ravel()):
            spin.setValue(float(value))
        self._editing = False

    def _numbers(self):
        if self._editing or self._points is None:
            return
        self.set_segment(np.array([s.value() for s in self.endpoints]).reshape(2,2))
