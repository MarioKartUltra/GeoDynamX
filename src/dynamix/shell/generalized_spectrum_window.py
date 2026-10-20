# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Touchette–Beck generalized inversion as an independent fitting tool."""
from __future__ import annotations

import threading

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core.nonconcave import gaussian_ensemble, generalized_legendre
from dynamix.core.partition_tables import moment_grid
from dynamix.shell.theme import current


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


class GeneralizedSpectrumWindow(QtWidgets.QDialog):
    fitChanged = QtCore.Signal(object)

    def __init__(self, sets, *, log2_L=None, levels=None, source="", parent=None,
                 physical_of_log2a=None):
        super().__init__(parent, QtCore.Qt.Window)
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
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "h")
        self.plot.setLabel("left", "generalized spectrum upper estimate")
        self.curve = self.plot.plot(pen=pg.mkPen(current().ink, width=2))
        self.boundary_points = self.plot.plot(pen=None, symbol="o", symbolSize=4)
        layout.addWidget(self.plot, 1)
        self.readout = QtWidgets.QLabel("Choose controls and Compute. No transform is rerun.")
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)
        self.apply_button.clicked.connect(self._compute)
        self.cancel_button.clicked.connect(self._cancel)
        for spin in self.spins.values():
            spin.valueChanged.connect(self._mark_pending)
        self.spins["log₂ amplitude reference"].setEnabled(False)
        self.auto_amplitude_check.toggled.connect(
            lambda on: (self.spins["log₂ amplitude reference"].setEnabled(not on), self._mark_pending()))
        self.source_combo.currentTextChanged.connect(self._mark_pending)
        self.resize(840, 700)
        from dynamix.shell.window_fit import fit_to_screen
        fit_to_screen(self, 840, 700)

    def current_fit(self):
        return self._fit

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
        self.curve.setData(result["h"], result["D"])
        b = result["boundary"] & np.isfinite(result["D"])
        self.boundary_points.setData(result["h"][b], result["D"][b])
        self._fit = {**result, "name": self._last_name, "kind": "generalized spectrum",
                     "valid": bool(np.isfinite(result["D"]).any()), "window": (result["settings"]["lo"], result["settings"]["hi"])}
        r2 = g["tau_r2"][np.isfinite(g["tau_r2"])]
        quality = f"min partition R² {r2.min():.3f}" if r2.size else "partition R² unavailable"
        d = g["delta"]
        self.readout.setText(f"{g['n_scales']} scales · {quality} · δ {d.min():g}–{d.max():g} ({d.size}) · "
                             f"c₁,₀ = log₂ {g['reference_amplitude']:.3g} = {np.log2(g['reference_amplitude']):.2f}"
                             f"{'' if result['settings']['amplitude'] == 'auto' else ' (set by hand)'} · "
                             f"{b.sum()}/{b.size} grid-boundary minima (dots). Negative values retained; "
                             "finite-grid bounds do not prove occupied h.")
        self.fitChanged.emit(self._fit)

    def closeEvent(self, event):
        self._cancel()
        super().closeEvent(event)
