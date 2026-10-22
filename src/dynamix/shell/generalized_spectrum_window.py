# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Touchette–Beck generalized inversion as an independent fitting tool."""
from __future__ import annotations

import threading

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core import spectra
from dynamix.core.nonconcave import gaussian_ensemble, generalized_legendre
from dynamix.core.partition_tables import moment_grid
from dynamix.shell.theme import current
from dynamix.shell.multifractal_window import WheelAxisViewBox, _q_color, add_zero_lines
from dynamix.shell.plot_zoom import zoom_plot  # noqa: F401  (the source scan's marker; the panels take WheelAxisViewBox)
from dynamix.shell.shared_range import SharedRangeFollower
from dynamix.shell.window_fit import TOOL_WINDOW, show_while_inactive


class _Signals(QtCore.QObject):
    done = QtCore.Signal(int, object)


class _Job(QtCore.QRunnable):
    def __init__(self, generation, entry, settings):
        super().__init__()
        self.generation, self.entry, self.settings = generation, entry, settings
        self.signals = _Signals()
        self.cancel = threading.Event()

    def run(self):
        try:
            e, s = self.entry, self.settings
            counts = ([np.full(np.size(m), w) for m, w in zip(e["mods"], e["weights"])]
                      if e.get("weights") is not None else None)
            g = gaussian_ensemble(e["mods"], e["scales"], s["q"], s["gamma"], s["lo"], s["hi"],
                                  counts_per_scale=counts, length_px=s["L"],
                                  reference_amplitude=s["amplitude"], delta_list=s["delta"],
                                  cancel=self.cancel.is_set)
            fit = generalized_legendre(g["tau"], s["q"], s["gamma"], s["h"], delta_list=s["delta"])
            result = {**fit, "ensemble": g, "settings": s}
        except Exception as exc:
            result = {"error": str(exc)}
        self.signals.done.emit(self.generation, result)


class GeneralizedSpectrumWindow(SharedRangeFollower, QtWidgets.QDialog):
    fitChanged = QtCore.Signal(object)

    def __init__(self, sets, *, log2_L=None, levels=None, source="", parent=None,
                 physical_of_log2a=None):
        super().__init__(parent, TOOL_WINDOW)
        show_while_inactive(self)
        self.setModal(False)
        self.setWindowTitle(f"Generalized spectrum — {source}")
        self._sets, self._source = sets, source
        self._fit, self._generation, self._jobs = None, 0, {}
        self._last_name = None
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(list(sets))
        row.addWidget(QtWidgets.QLabel("Samples"))
        row.addWidget(self.source_combo)
        self.apply_button = QtWidgets.QPushButton("Compute generalized spectrum")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        row.addWidget(self.apply_button)
        row.addWidget(self.cancel_button)
        row.addWidget(self._make_follow_check())
        # Leonarduzzi et al. 2019 (18): the amplitude reference c₁,₀ from the data, unless set by hand.
        self.auto_amplitude_check = QtWidgets.QCheckBox("c₁,₀ auto")
        self.auto_amplitude_check.setChecked(True)
        self.auto_amplitude_check.setToolTip("The amplitude the exponents are read against: the intercept of the "
                                             "per-scale mean of ln|T| on ln(a/L) (Leonarduzzi et al. 2019), or the "
                                             "log₂ reference spin when off")
        row.addWidget(self.auto_amplitude_check)
        layout.addLayout(row)
        self.spins = {}
        a = np.asarray(next(iter(sets.values()))["scales"], float)
        lo, hi = (levels if levels and None not in levels else (np.log2(a).min(), np.log2(a).max()))
        grid = QtWidgets.QGridLayout()
        # The narrow q with a sweep of the shift δ over the spectrum's support is Leonarduzzi et al.
        # 2019 §III-C3's recipe; a wide q with few parabolas is their Fig. 2's unstable case.
        specs = [("q min", -4., -200., 200., .5), ("q max", 4., -200., 200., .5),
                 ("Δq", .5, .01, 10., .1), ("γ min", 0., -256., 256., 1.),
                 ("γ max", 16., -256., 256., 1.), ("Δγ", 1., .01, 64., .5),
                 ("h min", -1., -100., 100., .1), ("h max", 2., -100., 100., .1),
                 ("Δh", .025, .001, 1., .01), ("δ min", -1., -100., 100., .1),
                 ("δ max", 2., -100., 100., .1), ("Δδ", .25, .01, 10., .05),
                 ("log₂a min", lo, -20., 30., .5), ("log₂a max", hi, -20., 30., .5),
                 ("log₂L", float(log2_L) if log2_L is not None else np.log2(a.max())+2, -20., 50., 1.),
                 ("log₂ amplitude reference", 0., -100., 100., 1.)]
        for k, (label, value, minimum, maximum, step) in enumerate(specs):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setValue(value)
            spin.setKeyboardTracking(False)
            self.spins[label] = spin
            r, c = divmod(k, 3)
            grid.addWidget(QtWidgets.QLabel(label), r, 2*c)
            grid.addWidget(spin, r, 2*c+1)
        layout.addLayout(grid)
        self.note = QtWidgets.QLabel("Finite-grid generalized Legendre inversion over (q, γ, δ), g = −γ(h − δ)² "
                                   "(Leonarduzzi et al. 2019): keep q narrow and sweep δ over the spectrum's support; "
                                   "compare scale windows. A small algebraic residual does not validate a recovered "
                                   "spectrum.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        # What the inf is built from, beside it: for the γ and δ picked below, log₂ Z_g(q, a)
        # against log₂ a per q with the fitted slope through the points (Leonarduzzi et al. 2019
        # Fig. 3's log-scale diagrams), and τ_g(q) for every γ at that δ.
        pick = QtWidgets.QHBoxLayout()
        pick.addWidget(QtWidgets.QLabel("diagrams at γ"))
        self.gamma_combo = QtWidgets.QComboBox()
        pick.addWidget(self.gamma_combo)
        pick.addWidget(QtWidgets.QLabel("δ"))
        self.delta_combo = QtWidgets.QComboBox()
        pick.addWidget(self.delta_combo)
        pick.addStretch(1)
        layout.addLayout(pick)
        # Four panels: D(h) and the partition function above, τ_g(q) and the specific heat
        # c_γ(q) = −q² τ_g″ below; the partition panel carries the scale-window sliders.
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(current().ground)
        self.plot = self._glw.addPlot(row=0, col=0, viewBox=WheelAxisViewBox())     # wheel: x; Shift: y
        self.plot.setLabel("bottom", "h")
        self.plot.setLabel("left", "generalized spectrum upper estimate")
        self.curve = self.plot.plot(pen=pg.mkPen(current().ink, width=2), connect="finite")
        self.boundary_points = self.plot.plot(pen=None, symbol="o", symbolSize=4)
        # h_min and h_max: the last interior minima; outside them the minima sit on the grid's
        # boundary, Touchette & Beck's −∞ floor with an unbounded grid.
        floor_pen = pg.mkPen(current().ink_muted, style=QtCore.Qt.DashLine)
        self.h_min_line = pg.InfiniteLine(angle=90, movable=False, pen=floor_pen, label="h_min",
                                          labelOpts={"position": 0.9, "color": current().ink_muted})
        self.h_max_line = pg.InfiniteLine(angle=90, movable=False, pen=floor_pen, label="h_max",
                                          labelOpts={"position": 0.9, "color": current().ink_muted})
        for line in (self.h_min_line, self.h_max_line):
            line.setVisible(False)
            self.plot.addItem(line, ignoreBounds=True)
        self.diagram_plot = self._glw.addPlot(row=0, col=1, viewBox=WheelAxisViewBox())
        self.diagram_plot.setLabel("bottom", "log₂ a")
        self.diagram_plot.setLabel("left", "log₂ Z_g(q, a)")
        band = pg.mkColor(current().amber)
        band.setAlpha(28)
        self._region = pg.LinearRegionItem(values=(lo, hi), orientation="vertical", brush=pg.mkBrush(band),
                                           pen=pg.mkPen(current().amber))
        self._region.setBounds((float(np.log2(a).min()) - 1.0, float(np.log2(a).max()) + 1.0))
        self._region.sigRegionChanged.connect(self._on_region)
        self.diagram_plot.addItem(self._region, ignoreBounds=True)
        self.tau_plot = self._glw.addPlot(row=1, col=0, viewBox=WheelAxisViewBox())
        self.tau_plot.setLabel("bottom", "q")
        self.tau_plot.setLabel("left", "τ_g(q)")
        add_zero_lines(self.tau_plot)
        self.cq_plot = self._glw.addPlot(row=1, col=1, viewBox=WheelAxisViewBox())
        self.cq_plot.setLabel("bottom", "q")
        self.cq_plot.setLabel("left", "c_γ(q) = −q² τ_g″")
        add_zero_lines(self.cq_plot)
        self._panel_items: dict = {self.diagram_plot: [], self.tau_plot: [], self.cq_plot: []}
        self._ensemble, self._fit_lines = None, []      # the last Compute's per-scale log Z; its fitted lines
        for plot in (self.plot, self.diagram_plot, self.tau_plot, self.cq_plot):
            for side in ("bottom", "left"):
                plot.getAxis(side).enableAutoSIPrefix(False)
        layout.addWidget(self._glw, 1)
        self.gamma_combo.currentIndexChanged.connect(lambda _i: self._draw_diagrams())
        self.delta_combo.currentIndexChanged.connect(lambda _i: self._draw_diagrams())
        self.readout = QtWidgets.QLabel("Choose controls and Compute. No transform is rerun.")
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)
        self.apply_button.clicked.connect(self._compute)
        self.cancel_button.clicked.connect(self._cancel)
        for spin in self.spins.values():
            spin.valueChanged.connect(self._mark_pending)
        for label in ("log₂a min", "log₂a max"):
            self.spins[label].valueChanged.connect(lambda _v: (self._sync_region(), self._publish_range(
                self.spins["log₂a min"].value(), self.spins["log₂a max"].value())))
        self.spins["log₂ amplitude reference"].setEnabled(False)
        self.auto_amplitude_check.toggled.connect(
            lambda on: (self.spins["log₂ amplitude reference"].setEnabled(not on), self._mark_pending()))
        self.source_combo.currentTextChanged.connect(self._mark_pending)
        self.resize(1180, 700)
        from dynamix.shell.window_fit import fit_to_screen
        fit_to_screen(self, 1180, 700)

    def _sync_region(self) -> None:
        """The sliders follow the spins."""
        lo, hi = sorted((self.spins["log₂a min"].value(), self.spins["log₂a max"].value()))
        if tuple(self._region.getRegion()) != (lo, hi):
            self._region.blockSignals(True)
            self._region.setRegion((lo, hi))
            self._region.blockSignals(False)
        self._draw_fit_lines()

    def _on_region(self) -> None:
        """The spins follow the sliders; the next Compute runs over the new window."""
        lo, hi = sorted(float(v) for v in self._region.getRegion())
        for label, v in (("log₂a min", lo), ("log₂a max", hi)):
            spin = self.spins[label]
            if spin.value() != v:
                spin.blockSignals(True)
                spin.setValue(v)
                spin.blockSignals(False)
        self._mark_pending()
        self._draw_fit_lines()
        self._publish_range(lo, hi)

    def _fill_pick(self, combo: QtWidgets.QComboBox, values) -> None:
        """The grid's values on a picker, 0 picked when the grid holds it, else the first."""
        combo.blockSignals(True)
        combo.clear()
        for v in values:
            combo.addItem(f"{float(v):g}")
        zero = [k for k, v in enumerate(values) if float(v) == 0.0]
        combo.setCurrentIndex(zero[0] if zero else 0)
        combo.blockSignals(False)

    def _picked_log2_z(self):
        """The last Compute's log₂ Z_g(q, a) at the picked γ and δ: q, log₂ a per scale, the table."""
        g = self._ensemble
        gi, di = max(self.gamma_combo.currentIndex(), 0), max(self.delta_combo.currentIndex(), 0)
        q, x = np.asarray(g["q"], float), np.log2(np.asarray(g["scales"], float))
        return q, x, np.asarray(g["log_partition"], float)[:, gi, di, :] / np.log(2.0)

    def _draw_fit_lines(self) -> None:
        """The dashed line per q is the least-squares slope of log₂ Z_g over the scales inside
        the current range, refitted on every move of the sliders or the spins; it is τ_g(q) at the
        picked γ and δ once Compute has run over that range."""
        for item in self._fit_lines:
            self.diagram_plot.removeItem(item)
        self._fit_lines.clear()
        if self._ensemble is None:
            return
        q, x, log2_z = self._picked_log2_z()
        lo, hi = sorted((self.spins["log₂a min"].value(), self.spins["log₂a max"].value()))
        inside = (x >= lo - 1e-9) & (x <= hi + 1e-9)
        for i in range(q.size):
            ok = inside & np.isfinite(log2_z[i])
            if ok.sum() < 2:
                continue
            slope, intercept = np.polyfit(x[ok], log2_z[i][ok], 1)
            ends = np.array([x[ok].min(), x[ok].max()])
            item = pg.PlotDataItem(ends, intercept + slope * ends,
                                   pen=pg.mkPen(_q_color(i, q.size), style=QtCore.Qt.DashLine, width=1.4))
            self.diagram_plot.addItem(item)
            self._fit_lines.append(item)

    def _draw_diagrams(self) -> None:
        if self._ensemble is None:
            return
        g = self._ensemble
        gi, di = max(self.gamma_combo.currentIndex(), 0), max(self.delta_combo.currentIndex(), 0)
        q, x, log2_z = self._picked_log2_z()
        tau = np.asarray(g["tau"], float)
        for plot, items in self._panel_items.items():          # the sliders and the zero lines stay
            for item in items:
                plot.removeItem(item)
            items.clear()

        def put(plot, item):
            plot.addItem(item)
            self._panel_items[plot].append(item)
        for i in range(q.size):                                  # every scale of the set, the window or not
            put(self.diagram_plot, pg.PlotDataItem(x, log2_z[i], pen=None, symbol="o", symbolSize=5,
                                                   symbolPen=None, symbolBrush=_q_color(i, q.size)))
        self._draw_fit_lines()
        gammas = np.asarray(g["gamma"], float)
        for k in range(gammas.size):
            hue = _q_color(k, max(gammas.size, 2))
            pen = pg.mkPen(hue, width=2.5 if k == gi else 1.0)
            put(self.tau_plot, pg.PlotDataItem(q, tau[:, k, di], pen=pen, name=f"γ = {gammas[k]:g}"))
            put(self.cq_plot, pg.PlotDataItem(q, spectra.specific_heat(q, tau[:, k, di]), pen=pen,
                                              connect="finite", name=f"γ = {gammas[k]:g}"))

    def current_fit(self):
        return self._fit

    def _set_scale_spins(self, log2_a_min: float, log2_a_max: float) -> None:
        for label, v in (("log₂a min", log2_a_min), ("log₂a max", log2_a_max)):
            spin = self.spins[label]
            spin.blockSignals(True)
            spin.setValue(float(v))
            spin.blockSignals(False)
        self._sync_region()
        self._mark_pending()

    def set_scale_window(self, log2_a_min: float, log2_a_max: float) -> None:
        """The scale window the next Compute runs over, published to the other fitting windows."""
        self._set_scale_spins(log2_a_min, log2_a_max)
        self._publish_range(float(log2_a_min), float(log2_a_max))

    _apply_scale_window = _set_scale_spins

    def _mark_pending(self, *_):
        self._cancel()
        self._retract()
        self.readout.setText("Controls changed — Compute to apply; upstream samples are reused.")

    def _retract(self):
        if self._last_name:
            self.fitChanged.emit({"name": self._last_name, "valid": False})
        self._fit = None
        self.curve.setData([], [])
        self.boundary_points.setData([], [])

    def _cancel(self):
        self._generation += 1
        for job in self._jobs.values():
            job.cancel.set()
        self.cancel_button.setEnabled(False)
        self.readout.setText("Cancelled; no stale result will be published.")

    def _compute(self):
        self._cancel()
        self._retract()
        v = {k: s.value() for k, s in self.spins.items()}
        try:
            q = moment_grid(v["q min"], v["q max"], v["Δq"], max_points=501)
            g = moment_grid(v["γ min"], v["γ max"], v["Δγ"], max_points=129)
            h = moment_grid(v["h min"], v["h max"], v["Δh"], max_points=4001)
            d = moment_grid(v["δ min"], v["δ max"], v["Δδ"], max_points=129)
            if q.size*g.size*d.size > 65536:
                raise ValueError("q/γ/δ grid exceeds 65536 triples; widen increments")
            if v["log₂a max"] <= v["log₂a min"]:
                raise ValueError("scale maximum must exceed minimum")
        except ValueError as exc:
            self.readout.setText(str(exc))
            return
        settings = {"q": q, "gamma": g, "h": h, "delta": d, "lo": v["log₂a min"], "hi": v["log₂a max"],
                    "L": 2**v["log₂L"],
                    "amplitude": "auto" if self.auto_amplitude_check.isChecked() else 2**v["log₂ amplitude reference"]}
        self._last_name = f"{self._source} · {self.source_combo.currentText()}, generalized spectrum"
        job = _Job(self._generation, self._sets[self.source_combo.currentText()], settings)
        job.signals.done.connect(self._completed)
        self._jobs[self._generation] = job
        self.cancel_button.setEnabled(True)
        self.readout.setText("Computing generalized partition fits…")
        QtCore.QThreadPool.globalInstance().start(job)

    @QtCore.Slot(int, object)
    def _completed(self, generation, result):
        self._jobs.pop(generation, None)
        if generation != self._generation:
            return
        self.cancel_button.setEnabled(False)
        if "error" in result:
            self.readout.setText(result["error"])
            return
        g = result["ensemble"]
        if g["n_scales"] < 3:
            self.readout.setText("At least three nonempty scales are required for a spectrum fit.")
            return
        # A minimum on the grid's boundary is a bound, never a value: the line connects the
        # interior minima only, the boundary ones stay as loose dots (Touchette & Beck's floor
        # outside [h_min, h_max] is where those would fall with an unbounded grid).
        b = result["boundary"] & np.isfinite(result["D"])
        self.curve.setData(result["h"], np.where(b, np.nan, result["D"]))
        self.boundary_points.setData(result["h"][b], result["D"][b])
        inside = np.asarray(result["h"], float)[~b & np.isfinite(result["D"])]
        for line, value in ((self.h_min_line, inside.min() if inside.size else None),
                            (self.h_max_line, inside.max() if inside.size else None)):
            line.setVisible(value is not None)
            if value is not None:
                line.setValue(float(value))
        self.plot.setXRange(-1.0, 3.0, padding=0.0)              # the spectrum's own window
        self.plot.setYRange(-0.5, 2.5, padding=0.0)
        self._fit = {**result, "name": self._last_name, "kind": "generalized spectrum",
                     "valid": bool(np.isfinite(result["D"]).any()), "window": (result["settings"]["lo"], result["settings"]["hi"])}
        self._ensemble = g
        self._fill_pick(self.gamma_combo, g["gamma"])
        self._fill_pick(self.delta_combo, g["delta"])
        self._draw_diagrams()
        r2 = g["tau_r2"][np.isfinite(g["tau_r2"])]
        quality = f"min partition R² {r2.min():.3f}" if r2.size else "partition R² unavailable"
        d = g["delta"]
        self.readout.setText(f"{g['n_scales']} of {np.size(g['scales'])} scales · {quality} · δ {d.min():g}–{d.max():g} ({d.size}) · "
                             f"c₁,₀ = log₂ {g['reference_amplitude']:.3g} = {np.log2(g['reference_amplitude']):.2f}"
                             f"{'' if result['settings']['amplitude'] == 'auto' else ' (set by hand)'} · "
                             f"{b.sum()}/{b.size} grid-boundary minima (dots). Negative values retained; "
                             "finite-grid bounds do not prove occupied h.")
        self.fitChanged.emit(self._fit)

    def closeEvent(self, event):
        self._cancel()
        super().closeEvent(event)
