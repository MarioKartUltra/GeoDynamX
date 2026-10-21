# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Log-cumulants window: the cumulants of ln ℓ against log₂ a, one regression each.

One per-level sample set at a time (:mod:`dynamix.model.samples`: the wavelet leaders' sets, the
WTMM's maxima lines or every maximum). Four panels, C₁(a) to C₄(a) (the mean, variance,
third and fourth cumulant of ln ℓ per level), each with its own draggable fit range and its own
regression against ln a: c_m is that slope, in the leaders literature's sign (c₂ < 0 for a
multifractal, the variance of ln ℓ growing towards fine scales), shown with its standard error, R²
and the number of levels fitted (Wendt, Abry and Jaffard 2007, eq. 8; Castaing, Gagne and
Hopfinger 1990). The
ranges move together while "sync ranges" is on. In the original frame c₁ loses the
pseudo-fractional integration γ. Every change publishes the cumulant expansion (``fitChanged``)
for the construction window: τ(q) = −d + c₁ q + c₂ q²/2 + c₃ q³/6 + c₄ q⁴/24 and its h(q),
D(h) and c(q).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core import spectra
from dynamix.shell.multifractal_window import PhysicalAxis, WheelAxisViewBox
from dynamix.shell.theme import current
from dynamix.shell.shared_range import SharedRangeFollower

__all__ = ["LogCumulantsWindow"]

ORDERS = (1, 2, 3, 4)
_NAMES = {1: "C₁(a): mean of ln ℓ", 2: "C₂(a): variance of ln ℓ", 3: "C₃(a): third cumulant",
          4: "C₄(a): fourth cumulant"}
_SUB = {1: "₁", 2: "₂", 3: "₃", 4: "₄"}
#: The q grid the published expansion is evaluated on.
Q_GRID = np.round(np.arange(-3.0, 6.001, 0.25), 6)


class LogCumulantsWindow(SharedRangeFollower, QtWidgets.QDialog):
    """c₁..c₄ off one sample set, each over its own fit range."""

    #: The current expansion, published on every change: ``name``, ``kind``, ``cumulants``
    #: (``c1``..``c4`` and per order ``sigma``, ``r2``, ``n``), ``q``, ``tau``, ``h``, ``D``, ``cq``.
    fitChanged = QtCore.Signal(object)

    def __init__(self, sets: dict, *, gamma: float = 0.0, levels=None, source: str = "", d: float = 2.0,
                 parent=None, physical_of_log2a=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setModal(False)
        self._sets = {k: v for k, v in sets.items() if v.get("mods") and v.get("scales") is not None}
        if not self._sets:
            raise ValueError("LogCumulantsWindow needs at least one per-level sample set")
        self._d = float(d)
        self._gamma = float(gamma)
        self._source = source
        self._physical = physical_of_log2a
        self._moments: dict = {}
        self._fit: dict | None = None
        self._syncing = False
        self.setWindowTitle(f"Log-cumulants — {source}" if source else "Log-cumulants")
        log2_s = np.log2(np.asarray(next(iter(self._sets.values()))["scales"], dtype=np.float64))
        self._span = (float(log2_s.min()), float(log2_s.max()))
        lo, hi = levels if levels is not None and None not in levels else self._span
        self._build_ui(float(lo), float(hi))
        self._refit()

    # -- public -----------------------------------------------------------------------------
    def current_fit(self) -> dict | None:
        return self._fit

    def ranges(self) -> list:
        """``[(log2_a_min, log2_a_max)]`` per order."""
        return [tuple(float(v) for v in r.getRegion()) for r in self.regions]

    def _set_regions(self, log2_a_min: float, log2_a_max: float) -> None:
        """Every order's region at once (the three take a shared range together), one re-fit."""
        self._syncing = True
        try:
            for region in self.regions:
                region.setRegion((float(log2_a_min), float(log2_a_max)))
        finally:
            self._syncing = False
        self._refit()

    def set_scale_window(self, log2_a_min: float, log2_a_max: float) -> None:
        self._set_regions(log2_a_min, log2_a_max)
        self._publish_range(float(log2_a_min), float(log2_a_max))

    _apply_scale_window = _set_regions

    # -- ui ---------------------------------------------------------------------------------
    def _build_ui(self, lo: float, hi: float) -> None:
        from dynamix.shell.window_fit import fit_to_screen, remember_geometry

        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Set"))
        self.set_combo = QtWidgets.QComboBox()
        self.set_combo.addItems(list(self._sets))
        self.set_combo.currentIndexChanged.connect(lambda _i: self._refit())
        controls.addWidget(self.set_combo)
        controls.addWidget(QtWidgets.QLabel("Frame"))
        self.frame_combo = QtWidgets.QComboBox()
        self.frame_combo.addItems(["original", "integrated"])
        self.frame_combo.setEnabled(bool(self._gamma))
        self.frame_combo.setToolTip(f"γ = {self._gamma:g}: the original frame takes γ out of c₁"
                                    if self._gamma else "γ = 0: both frames coincide")
        self.frame_combo.currentIndexChanged.connect(lambda _i: self._refit())
        controls.addWidget(self.frame_combo)
        self.sync_check = QtWidgets.QCheckBox("sync ranges")
        self.sync_check.setChecked(True)
        self.sync_check.toggled.connect(self._on_sync_toggled)
        controls.addWidget(self.sync_check)
        controls.addWidget(self._make_follow_check())
        controls.addStretch(1)
        layout.addLayout(controls)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(current().ground)
        layout.addWidget(self._glw, 1)
        band = pg.mkColor(current().amber)
        band.setAlpha(28)
        self.plots, self.regions, self.labels = {}, [], {}
        self._points, self._lines = {}, {}
        for m in ORDERS:
            plot = self._glw.addPlot(row=(m - 1) // 2, col=(m - 1) % 2, viewBox=WheelAxisViewBox())
            plot.setLabel("bottom", "log₂ a")
            plot.setLabel("left", _NAMES[m])
            plot.getAxis("left").enableAutoSIPrefix(False)
            if self._physical is not None:
                plot.setAxisItems({"top": PhysicalAxis(self._physical)})
                plot.showAxis("top")
            region = pg.LinearRegionItem(values=(lo, hi), orientation="vertical", brush=pg.mkBrush(band),
                                         pen=pg.mkPen(current().amber))
            region.setBounds(self._span)
            region.sigRegionChanged.connect(lambda _r, m=m: self._on_region(m))
            plot.addItem(region)
            self._points[m] = pg.PlotDataItem(pen=None, symbol="o", symbolSize=6, symbolPen=None,
                                              symbolBrush=current().ink)
            self._lines[m] = pg.PlotDataItem(pen=pg.mkPen(current().amber, style=QtCore.Qt.DashLine, width=1.6))
            plot.addItem(self._points[m])
            plot.addItem(self._lines[m])
            label = pg.LabelItem(justify="left")
            self._glw.addItem(label, row=(m - 1) // 2 + 2, col=(m - 1) % 2)
            self.plots[m], self.labels[m] = plot, label
            self.regions.append(region)
        # The readouts sit under their panels, no taller than a line.
        for row in (2, 3):
            self._glw.ci.layout.setRowStretchFactor(row, 0)
        for row in (0, 1):
            self._glw.ci.layout.setRowStretchFactor(row, 1)
        self.readout = QtWidgets.QLabel("")
        self.readout.setProperty("reading", "true")
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)
        fit_to_screen(self, 1000, 760)
        remember_geometry(self, "log_cumulants")

    # -- the fit ----------------------------------------------------------------------------
    def _rows(self) -> np.ndarray:
        """``(4, n_scales)``: the cumulants of ln ℓ per scale of the chosen set."""
        name = self.set_combo.currentText()
        if name not in self._moments:
            self._moments[name] = spectra.ln_moments(self._sets[name]["mods"])
        return self._moments[name]

    def _on_sync_toggled(self, on: bool) -> None:
        if on:
            self._on_region(1)

    def _on_region(self, m: int) -> None:
        if self._syncing:
            return
        if self.sync_check.isChecked():
            self._syncing = True
            lo, hi = self.regions[m - 1].getRegion()
            for region in self.regions:
                if region is not self.regions[m - 1]:
                    region.setRegion((lo, hi))
            self._syncing = False
        self._refit()
        self._publish_range(*self.regions[m - 1].getRegion())

    def _refit(self) -> None:
        C, scales = self._rows(), np.asarray(self._sets[self.set_combo.currentText()]["scales"], dtype=np.float64)
        log2_s = np.log2(scales)
        frame = self.frame_combo.currentText()
        cums: dict = {"sigma": {}, "r2": {}, "n": {}, "range": {}}
        for m in ORDERS:
            lo, hi = sorted(float(v) for v in self.regions[m - 1].getRegion())
            reg = spectra.cumulant_regression(C, scales, lo, hi, m)
            c = reg["c"] - (self._gamma if m == 1 and frame == "original" else 0.0)
            cums[f"c{m}"], cums["sigma"][m], cums["r2"][m], cums["n"][m] = c, reg["sigma"], reg["r2"], reg["n"]
            cums["range"][m] = (lo, hi)
            ok = np.isfinite(C[m - 1])
            self._points[m].setData(log2_s[ok], C[m - 1][ok])
            if np.isfinite(reg["slope"]):
                x = np.array([lo, hi])
                self._lines[m].setData(x, reg["intercept"] + reg["slope"] * x * np.log(2.0))
            else:
                self._lines[m].setData([], [])
            self.labels[m].setText(self._label(m, c, reg))
        label = self.set_combo.currentText()
        q = Q_GRID
        tau = spectra.cumulant_tau(cums, q, c0=self._d)
        c1, c2, c3, c4 = (cums[f"c{m}"] for m in ORDERS)          # an order without a fit: no expansion
        h = c1 + c2 * q + c3 * q ** 2 / 2.0 + c4 * q ** 3 / 6.0
        name = f"{self._source} · {label}, log-cumulants" if self._source else f"{label}, log-cumulants"
        self._fit = {"name": name, "kind": "log-cumulants",
                     "set": label, "cumulants": cums, "frame": frame, "gamma": self._gamma,
                     "q": q, "tau": tau, "h": h, "D": q * h - tau, "cq": spectra.cumulant_heat(cums, q)}
        parts = [f"{label} · " + " · ".join(f"c{_SUB[m]} = {cums[f'c{m}']:+.4f}" for m in ORDERS)]
        if self._gamma:
            parts.append(f"γ = {self._gamma:g}, {frame} frame")
        self.readout.setText(" · ".join(parts))
        self.fitChanged.emit(self._fit)

    @staticmethod
    def _label(m: int, c: float, reg: dict) -> str:
        if not np.isfinite(c):
            return f"c{_SUB[m]}: fewer than two levels in range"
        sigma = f" ± {reg['sigma']:.4f}" if np.isfinite(reg["sigma"]) else ""
        return f"c{_SUB[m]} = {c:+.4f}{sigma} · R² {reg['r2']:.3f} · N {reg['n']}"
