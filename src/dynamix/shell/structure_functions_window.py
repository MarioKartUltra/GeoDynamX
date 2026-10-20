# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Structure functions window: the wavelet leaders' own fitter.

One set of the ``Samples`` contract at a time (leaders, p-leaders, WSE leaders or the bare
coefficients): the structure functions log₂ S(q, j) against log₂ a for the chosen moment
orders, the fit range as a draggable region, and off that range ζ(q) by weighted least squares
(Wendt, Abry and Jaffard 2007, eq. 7; uniform weights or the valid counts n_j), τ(q) = ζ(q) − d,
h(q) and D(q) from the tilted means (their eqs. 12 and 13) and the Legendre transform of τ as
the dashed hull on D(h). The pseudo-fractional integration γ the transform applied is taken back
out in the original frame. Every change re-fits and publishes the fit (``fitChanged``) for the
construction window, where the methods' spectra are compared.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core import leaders, spectra
from dynamix.model.samples import LABELS
from dynamix.shell.multifractal_window import PhysicalAxis, WheelAxisViewBox, _q_color, add_zero_lines
from dynamix.shell.theme import current

__all__ = ["StructureFunctionsWindow"]

#: The moment orders' defaults, the WTMM's.
Q_DEFAULT = (-3.0, 6.0, 0.5)
WEIGHTS = ("uniform", "counts n_j")


class StructureFunctionsWindow(QtWidgets.QDialog):
    """ζ(q), τ(q), h(q) and D(h) off one sample set, over a chosen fit range."""

    #: The current fit, published on every change: ``name``, ``kind``, ``q``, ``tau``, ``h``,
    #: ``D`` with their σ's, ``hull`` (the Legendre transform of τ), ``cq`` (the specific heat).
    fitChanged = QtCore.Signal(object)

    def __init__(self, samples: dict, *, gamma: float = 0.0, levels=None, source: str = "",
                 parent=None, physical_of_log2a=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setModal(False)
        self._samples = samples
        self._gamma = float(gamma)
        self._source = source
        self._physical = physical_of_log2a
        self._tables: dict = {}
        self._fit: dict | None = None
        self._family_items: list = []
        self._fit_items: list = []
        self.setWindowTitle(f"Structure functions — {source}" if source else "Structure functions")
        # The span is the set's own log2 scales: the pyramid's levels 1..J for a dyadic set, the
        # ladder's log2 a for a continuous one.
        log2_scales = np.log2(np.asarray(samples["scales"], dtype=np.float64))
        self._span = (float(log2_scales[0]), float(log2_scales[-1]))
        lo, hi = levels if levels is not None else self._span
        self._build_ui(float(lo), float(hi))
        self._refit()

    # -- public -----------------------------------------------------------------------------
    def current_fit(self) -> dict | None:
        """The latest :func:`~dynamix.core.leaders.structure_fit`, with the set's label."""
        return self._fit

    def set_scale_window(self, log2_a_min: float, log2_a_max: float) -> None:
        self._region.setRegion((float(log2_a_min), float(log2_a_max)))

    # -- ui ---------------------------------------------------------------------------------
    def _build_ui(self, lo: float, hi: float) -> None:
        from dynamix.shell.window_fit import fit_to_screen, remember_geometry

        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Set"))
        self.set_combo = QtWidgets.QComboBox()
        for name in self._samples["sets"]:
            self.set_combo.addItem(LABELS.get(name, name), name)
        self.set_combo.currentIndexChanged.connect(lambda _i: self._refit(rebuild=True))
        controls.addWidget(self.set_combo)
        controls.addWidget(QtWidgets.QLabel("q"))
        self.qmin_spin, self.qmax_spin, self.qstep_spin = (QtWidgets.QDoubleSpinBox() for _ in range(3))
        for spin, value, rng, step in ((self.qmin_spin, Q_DEFAULT[0], (-20.0, 20.0), 0.5),
                                       (self.qmax_spin, Q_DEFAULT[1], (-20.0, 20.0), 0.5),
                                       (self.qstep_spin, Q_DEFAULT[2], (0.05, 5.0), 0.05)):
            spin.setRange(*rng)
            spin.setDecimals(2)
            spin.setSingleStep(step)
            spin.setValue(value)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(lambda _v: self._refit(rebuild=True))
            controls.addWidget(spin)
        controls.addWidget(QtWidgets.QLabel("Weights"))
        self.weights_combo = QtWidgets.QComboBox()
        self.weights_combo.addItems(list(WEIGHTS))
        self.weights_combo.currentIndexChanged.connect(lambda _i: self._refit())
        controls.addWidget(self.weights_combo)
        controls.addWidget(QtWidgets.QLabel("Frame"))
        self.frame_combo = QtWidgets.QComboBox()
        self.frame_combo.addItems(["original", "integrated"])
        self.frame_combo.setEnabled(bool(self._gamma))
        self.frame_combo.setToolTip(f"γ = {self._gamma:g}: the original frame takes γ q out of τ and γ out of h"
                                    if self._gamma else "γ = 0: both frames coincide")
        self.frame_combo.currentIndexChanged.connect(lambda _i: self._refit())
        controls.addWidget(self.frame_combo)
        controls.addWidget(QtWidgets.QLabel("log₂a"))
        self.lo_spin, self.hi_spin = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for spin, seed in ((self.lo_spin, lo), (self.hi_spin, hi)):
            spin.setRange(*self._span)
            spin.setDecimals(2)
            spin.setSingleStep(0.25)
            spin.setValue(seed)
            spin.valueChanged.connect(self._on_spins)
            controls.addWidget(spin)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(current().ground)
        layout.addWidget(self._glw, 1)
        self.family_plot = self._glw.addPlot(row=0, col=0, colspan=3, viewBox=WheelAxisViewBox())
        self.family_plot.setLabel("bottom", "log₂ a")
        self.family_plot.setLabel("left", "log₂ S(q, a)")
        if self._physical is not None:
            self.family_plot.setAxisItems({"top": PhysicalAxis(self._physical)})
            self.family_plot.showAxis("top")
        band = pg.mkColor(current().amber)
        band.setAlpha(28)
        self._region = pg.LinearRegionItem(values=(lo, hi), orientation="vertical", brush=pg.mkBrush(band),
                                           pen=pg.mkPen(current().amber))
        self._region.setBounds(self._span)
        self._region.sigRegionChanged.connect(self._on_region)
        self.family_plot.addItem(self._region)
        self.plots = {}
        for col, (name, xlabel, ylabel) in enumerate((("tau", "q", "τ(q) = ζ(q) − d"), ("h", "q", "h(q)"),
                                                      ("D", "h", "D(h)"))):
            plot = self._glw.addPlot(row=1, col=col, viewBox=WheelAxisViewBox())
            plot.setLabel("bottom", xlabel)
            plot.setLabel("left", ylabel)
            for side in ("bottom", "left"):             # plain decimals, never an "(x0.001)" prefix
                plot.getAxis(side).enableAutoSIPrefix(False)
            add_zero_lines(plot)
            self.plots[name] = plot
        muted = pg.mkPen(current().ink_muted)
        self._err = {name: pg.ErrorBarItem(pen=muted, beam=0.05) for name in self.plots}
        self._curves = {name: pg.PlotDataItem(pen=pg.mkPen(current().ink), symbol="o", symbolSize=5,
                                              symbolPen=None, symbolBrush=current().ink, connect="finite")
                        for name in self.plots}
        for name, plot in self.plots.items():
            plot.addItem(self._err[name])
            plot.addItem(self._curves[name])
        self._hull = pg.PlotDataItem(pen=pg.mkPen(current().amber, style=QtCore.Qt.DashLine, width=1.4),
                                     connect="finite")
        self.plots["D"].addItem(self._hull)
        self._d_line = pg.InfiniteLine(angle=0, pos=float(self._samples.get("d", 2)), movable=False,
                                       pen=pg.mkPen(current().ink_muted, style=QtCore.Qt.DashLine))
        self.plots["D"].addItem(self._d_line)
        legend = self.plots["D"].addLegend(offset=(10, -10))          # bottom left: the points sit top right
        legend.addItem(self._curves["D"], "tilted means (eqs. 12, 13)")
        legend.addItem(self._hull, "Legendre of ζ")
        self.readout = QtWidgets.QLabel("")
        self.readout.setProperty("reading", "true")
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)
        fit_to_screen(self, 1100, 760)
        remember_geometry(self, "structure_functions")

    # -- the fit ----------------------------------------------------------------------------
    def _q_list(self) -> np.ndarray:
        lo, hi, step = self.qmin_spin.value(), self.qmax_spin.value(), self.qstep_spin.value()
        if hi < lo:
            lo, hi = hi, lo
        from dynamix.core.partition_tables import moment_grid

        return moment_grid(lo, hi, step)

    def _table(self) -> dict:
        name = self.set_combo.currentData()
        key = (name, tuple(self._q_list()))
        if key not in self._tables:
            self._tables[key] = leaders.tables({**self._samples, "sets": {name: self._samples["sets"][name]}},
                                               self._q_list())[name]
        return self._tables[key]

    def _on_spins(self) -> None:
        lo, hi = self.lo_spin.value(), self.hi_spin.value()
        if (lo, hi) != tuple(self._region.getRegion()):
            self._region.setRegion((lo, hi))

    def _on_region(self) -> None:
        lo, hi = self._region.getRegion()
        for spin, v in ((self.lo_spin, lo), (self.hi_spin, hi)):
            if spin.value() != v:
                spin.blockSignals(True)
                spin.setValue(float(v))
                spin.blockSignals(False)
        self._refit()

    def _refit(self, rebuild: bool = False) -> None:
        table = self._table()
        lo, hi = sorted(float(v) for v in self._region.getRegion())
        weights = None if self.weights_combo.currentIndex() == 0 else table["N_a"]
        d = int(self._samples.get("d", 2))
        fit = leaders.structure_fit(table, lo, hi, weights=weights, gamma=self._gamma,
                                    frame=self.frame_combo.currentText(), d=d)
        label = self.set_combo.currentText()
        fit["set"] = label
        fit["kind"] = "structure functions"
        fit["name"] = f"{self._source} · {label}, {fit['kind']}" if self._source else f"{label}, {fit['kind']}"
        fit["hull"] = spectra.legendre_dh(fit["q_list"], fit["tau"])
        fit["cq"] = spectra.specific_heat(fit["q_list"], fit["tau"])
        fit["q"] = fit["q_list"]
        self._fit = fit
        if rebuild or not self._family_items:
            self._rebuild_family(table)
        self._draw_fit_lines(table, fit, lo, hi)
        q = fit["q_list"]
        for name, x in (("tau", q), ("h", q), ("D", fit["h"])):
            y, err = fit[name], fit[name + "_err"]
            ok = np.isfinite(x) & np.isfinite(y)
            self._curves[name].setData(x[ok], y[ok])
            self._err[name].setData(x=x[ok], y=y[ok], height=np.where(np.isfinite(err[ok]), 2 * err[ok], 0.0))
        self._hull.setData(fit["hull"]["h"], fit["hull"]["D"])
        self._read(fit, lo, hi)
        self.fitChanged.emit(fit)

    def _rebuild_family(self, table: dict) -> None:
        for item in self._family_items:
            self.family_plot.removeItem(item)
        self._family_items = []
        q, log2_s = table["q_list"], np.asarray(table["log2_scales"], dtype=np.float64)
        for i, row in enumerate(np.asarray(table["tau_qa"])):
            item = pg.PlotDataItem(log2_s, row, pen=pg.mkPen(_q_color(i, len(q)), width=1.0), connect="finite")
            self.family_plot.addItem(item)
            self._family_items.append(item)

    def _draw_fit_lines(self, table: dict, fit: dict, lo: float, hi: float) -> None:
        """The fitted line of every q across the fit range, dashed, in the frame as computed
        (what the family panel shows)."""
        for item in self._fit_items:
            self.family_plot.removeItem(item)
        self._fit_items = []
        x = np.array([lo, hi])
        gamma = self._gamma if fit["frame"] == "original" else 0.0
        for i, (q, slope, at_zero) in enumerate(zip(fit["q_list"], fit["tau"] + gamma * fit["q_list"], fit["tau_0"])):
            if not np.isfinite(slope):
                continue
            item = pg.PlotDataItem(x, at_zero + slope * x,
                                   pen=pg.mkPen(_q_color(i, len(fit["q_list"])), style=QtCore.Qt.DashLine, width=1.6))
            self.family_plot.addItem(item)
            self._fit_items.append(item)

    def _read(self, fit: dict, lo: float, hi: float) -> None:
        q = fit["q_list"]
        at = {v: int(np.argmin(np.abs(q - v))) for v in (0.0, 1.0, 2.0)}
        parts = [f"{fit['set']} · log₂a {lo:.2f}–{hi:.2f} ({len(fit['levels'])} levels) · {fit['weights']} weights"]
        if np.isclose(q[at[2.0]], 2.0) and np.isfinite(fit["zeta"][at[2.0]]):
            parts.append(f"ζ(2) = {fit['zeta'][at[2.0]]:.3f} ± {fit['zeta_err'][at[2.0]]:.3f}")
        if np.isclose(q[at[1.0]], 1.0) and np.isfinite(fit["h"][at[1.0]]):
            parts.append(f"h(1) = {fit['h'][at[1.0]]:.3f} ± {fit['h_err'][at[1.0]]:.3f}")
        if np.isclose(q[at[0.0]], 0.0) and np.isfinite(fit["tau"][at[0.0]]):
            parts.append(f"τ(0) = {fit['tau'][at[0.0]]:.3f}")
        D = fit["hull"]["D"]
        if np.isfinite(D).any():
            parts.append(f"hull: h* = {fit['hull']['h'][np.nanargmax(D)]:.3f}, D* = {np.nanmax(D):.3f}")
        if self._gamma:
            parts.append(f"γ = {self._gamma:g}, {fit['frame']} frame")
        self.readout.setText(" · ".join(parts))
