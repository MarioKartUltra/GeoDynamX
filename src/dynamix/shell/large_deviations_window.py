# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Large deviations window: the microcanonical route's own tool.

One per-level sample set at a time (:mod:`dynamix.model.samples`: the WTMM's maxima lines or
every maximum, the wavelet leaders' sets): every sample's local exponent h_λ = log₂ M / log₂(a/L)
per scale (the left panel, their histograms), and the large-deviation spectrum counted on them
over the chosen scales (:mod:`dynamix.core.large_deviation`): the adaptive bins' (h(q), D(q))
curve, or the fixed-ε sweep's three curves whose spread is the convergence check. No Legendre
transform is taken. Fixed bins can reveal nonconcavity; adaptive q-tilted bins can skip it.
Every change publishes the curve
(``fitChanged``) for the construction window, where it is read against the other methods'.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core import large_deviation as ld
from dynamix.shell.multifractal_window import WheelAxisViewBox, _q_color
from dynamix.shell.theme import current
from dynamix.shell.shared_range import SharedRangeFollower

__all__ = ["LD_RULES", "LargeDeviationsWindow"]

#: The bin rules: adaptive tilted bins (a finite-scale heuristic), or a fixed-ε sweep.
LD_RULES = ("adaptive ε", "fixed ε sweep")
H_GRID = np.linspace(-1.5, 2.5, 161)


class LargeDeviationsWindow(SharedRangeFollower, QtWidgets.QDialog):
    """D_LD(h) off one sample set over a chosen scale window, under either bin rule."""

    #: The current curve, published on every change: ``name``, ``kind``, ``h``, ``D``.
    fitChanged = QtCore.Signal(object)

    def __init__(self, sets: dict, *, log2_L: "float | None" = None, levels=None, source: str = "",
                 parent=None, physical_of_log2a=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setModal(False)
        self._sets = {k: v for k, v in sets.items() if v.get("mods") and v.get("scales") is not None}
        if not self._sets:
            raise ValueError("LargeDeviationsWindow needs at least one per-level sample set")
        self._log2_L = float(log2_L) if log2_L is not None else None
        self._source = source
        self._physical = physical_of_log2a
        self._fit: dict | None = None
        self._hist_items: list = []
        self.setWindowTitle(f"Large deviations — {source}" if source else "Large deviations")
        self._build_ui(levels)
        self._recompute()

    # -- public -----------------------------------------------------------------------------
    def current_fit(self) -> dict | None:
        return self._fit

    def set_scale_window(self, log2_a_min: float, log2_a_max: float) -> None:
        """Both spins at once, one recount; the recount publishes the range unless it is being taken."""
        for spin, v in ((self.lo_spin, log2_a_min), (self.hi_spin, log2_a_max)):
            spin.blockSignals(True)
            spin.setValue(float(v))
            spin.blockSignals(False)
        self._recompute()

    _apply_scale_window = set_scale_window

    # -- ui ---------------------------------------------------------------------------------
    def _entry(self) -> dict:
        return self._sets[self.source_combo.currentText()]

    def _build_ui(self, levels) -> None:
        from dynamix.shell.window_fit import fit_to_screen, remember_geometry

        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Source"))
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(list(self._sets))
        self.source_combo.currentIndexChanged.connect(lambda _i: self._on_source_changed())
        controls.addWidget(self.source_combo)
        controls.addWidget(QtWidgets.QLabel("Bins"))
        self.rule_combo = QtWidgets.QComboBox()
        self.rule_combo.addItems(list(LD_RULES))
        self.rule_combo.currentIndexChanged.connect(lambda _i: self._recompute())
        controls.addWidget(self.rule_combo)
        controls.addWidget(QtWidgets.QLabel("log₂a"))
        self.lo_spin, self.hi_spin = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        log2_s = np.log2(np.asarray(self._entry()["scales"], dtype=np.float64))
        lo, hi = (float(levels[0]), float(levels[1])) if levels and None not in levels else (log2_s.min(), log2_s.max())
        for spin, seed in ((self.lo_spin, lo), (self.hi_spin, hi)):
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            spin.setRange(float(log2_s.min()), float(log2_s.max()))
            spin.setValue(seed)
            spin.valueChanged.connect(lambda _v: self._recompute())
            controls.addWidget(spin)
        controls.addWidget(QtWidgets.QLabel("q"))
        self.qmin_spin, self.qmax_spin, self.qstep_spin = (QtWidgets.QDoubleSpinBox() for _ in range(3))
        for spin, value, rng, step in ((self.qmin_spin, -3.0, (-20.0, 20.0), 0.5), (self.qmax_spin, 6.0, (-20.0, 20.0), 0.5),
                                       (self.qstep_spin, 0.5, (0.05, 5.0), 0.05)):
            spin.setRange(*rng)
            spin.setDecimals(2)
            spin.setSingleStep(step)
            spin.setValue(value)
            spin.valueChanged.connect(lambda _v: self._recompute())
            controls.addWidget(spin)
        controls.addWidget(self._make_follow_check())
        # Leonarduzzi et al. 2019 (18)–(19): the amplitude intercept c₁,₀ taken off every exponent.
        self.prefactor_check = QtWidgets.QCheckBox("prefactor c₁,₀")
        self.prefactor_check.setChecked(True)
        self.prefactor_check.setToolTip("Read every exponent against c₁,₀, the intercept of the per-scale mean of "
                                        "log₂M on log₂(a/L) (Leonarduzzi et al. 2019; Struzik's mean-field reference)")
        self.prefactor_check.toggled.connect(lambda _c: self._recompute())
        controls.addWidget(self.prefactor_check)
        # Bastin, Esser & Jaffard 2016: the increasing and decreasing leaders profiles, the
        # wavelet-robust bounds, beside the fixed sweep's density.
        self.profiles_check = QtWidgets.QCheckBox("profiles θ±")
        self.profiles_check.setToolTip("The cumulative counts θ⁺ (h_λ ≤ h + ε) and θ⁻ (h_λ ≥ h − ε) of Bastin, Esser "
                                       "& Jaffard 2016, each the running maximum of the density from its side")
        self.profiles_check.toggled.connect(lambda _c: self._recompute())
        controls.addWidget(self.profiles_check)
        controls.addStretch(1)
        layout.addLayout(controls)

        grids = QtWidgets.QHBoxLayout()
        self.hmin_spin, self.hmax_spin, self.hstep_spin = (QtWidgets.QDoubleSpinBox() for _ in range(3))
        self.eps_spins = [QtWidgets.QDoubleSpinBox() for _ in range(3)]
        for label, spin, value, bounds in (
                ("h min", self.hmin_spin, -1.5, (-100., 100.)),
                ("h max", self.hmax_spin, 2.5, (-100., 100.)),
                ("Δh", self.hstep_spin, .025, (.001, 10.)),
                *((f"ε {i + 1}", spin, e, (.001, 10.))
                  for i, (spin, e) in enumerate(zip(self.eps_spins, ld.EPS_SWEEP)))):
            grids.addWidget(QtWidgets.QLabel(label))
            spin.setDecimals(3)
            spin.setRange(*bounds)
            spin.setValue(value)
            spin.setKeyboardTracking(False)
            grids.addWidget(spin)
        self.apply_grid_button = QtWidgets.QPushButton("Apply grid")
        self.apply_grid_button.clicked.connect(self._recompute)
        grids.addWidget(self.apply_grid_button)
        layout.addLayout(grids)
        self.target_label = QtWidgets.QLabel("")
        self.target_label.setWordWrap(True)
        layout.addWidget(self.target_label)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(current().ground)
        layout.addWidget(self._glw, 1)
        self.hist_plot = self._glw.addPlot(row=0, col=0, viewBox=WheelAxisViewBox())
        self.hist_plot.setLabel("bottom", "h_λ = log₂ M / log₂(a/L)")
        self.hist_plot.setLabel("left", "cells")
        self.hist_plot.setLogMode(y=True)
        self._hist_legend = self.hist_plot.addLegend(offset=(-10, 10))
        self.dh_plot = self._glw.addPlot(row=0, col=1, viewBox=WheelAxisViewBox())
        self.dh_plot.setLabel("bottom", "h")
        self.dh_plot.setLabel("left", "D_LD(h)")
        for plot in (self.hist_plot, self.dh_plot):
            for side in ("bottom", "left"):
                plot.getAxis(side).enableAutoSIPrefix(False)
        self._legend = self.dh_plot.addLegend(offset=(10, -10))
        # One curve per ε of the fixed sweep, the finest ε last and darkest; the adaptive rule
        # draws on the last alone.
        self.curves = []
        for i in range(3):
            colour = pg.mkColor(current().amber)
            colour.setAlpha(90 + 80 * i)
            curve = pg.PlotDataItem(pen=pg.mkPen(colour, width=1.4), symbol="t", symbolSize=5,
                                    symbolPen=None, symbolBrush=colour, connect="finite")
            self.dh_plot.addItem(curve)
            self.curves.append(curve)
        # The two profiles, dashed, on the narrowest ε of the sweep.
        self.profile_curves = []
        for _ in range(2):
            curve = pg.PlotDataItem(pen=pg.mkPen(current().ink, style=QtCore.Qt.DashLine, width=1.2),
                                    connect="finite")
            self.dh_plot.addItem(curve)
            self.profile_curves.append(curve)
        self.readout = QtWidgets.QLabel("")
        self.readout.setProperty("reading", "true")
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)
        fit_to_screen(self, 1100, 640)
        remember_geometry(self, "large_deviations")

    # -- the spectrum -----------------------------------------------------------------------
    def _length_px(self, scales) -> float:
        """The integral scale the exponents are measured against: the field's extent, else
        four times the largest scale."""
        return 2.0 ** self._log2_L if self._log2_L is not None else 4.0 * float(np.max(scales))

    def _q_list(self) -> np.ndarray:
        lo, hi, step = self.qmin_spin.value(), self.qmax_spin.value(), self.qstep_spin.value()
        if hi < lo:
            lo, hi = hi, lo
        from dynamix.core.partition_tables import moment_grid

        return moment_grid(lo, hi, step)

    def _on_source_changed(self) -> None:
        log2_s = np.log2(np.asarray(self._entry()["scales"], dtype=np.float64))
        for spin in (self.lo_spin, self.hi_spin):
            spin.blockSignals(True)
            spin.setRange(float(log2_s.min()), float(log2_s.max()))
            spin.blockSignals(False)
        self._recompute()

    def _recompute(self) -> None:
        entry = self._entry()
        scales = np.asarray(entry["scales"], dtype=np.float64)
        L = self._length_px(scales)
        lo, hi = sorted((float(self.lo_spin.value()), float(self.hi_spin.value())))
        self._publish_range(lo, hi)
        rule = self.rule_combo.currentText()
        adaptive = rule == LD_RULES[0]
        for spin in (self.qmin_spin, self.qmax_spin, self.qstep_spin):
            spin.setEnabled(adaptive)
            spin.setKeyboardTracking(False)
        for spin in (self.hmin_spin, self.hmax_spin, self.hstep_spin, *self.eps_spins):
            spin.setEnabled(not adaptive)
        self.apply_grid_button.setEnabled(not adaptive)
        self.profiles_check.setEnabled(not adaptive)
        prefactor = "intercept" if self.prefactor_check.isChecked() else "none"
        self.target_label.setText(entry.get("target", "Sample large deviations") +
                                  (" · adaptive heuristic; q tilts may skip nonconcave regions" if adaptive else
                                   " · finite-bin estimates; compare bin widths and resolutions"))
        name = f"{self._source} · {self.source_combo.currentText()}, large deviations" if self._source \
            else f"{self.source_combo.currentText()}, large deviations"
        common = dict(log2_a_min=lo, log2_a_max=hi, length_px=L, scale_weight=entry.get("weights"),
                      prefactor=prefactor)
        histogram_key = (self.source_combo.currentText(), L, lo, hi, prefactor)
        if histogram_key != getattr(self, "_histogram_key", None):
            try:
                offset = ld.intercept(entry["mods"], scales, log2_a_min=lo, log2_a_max=hi, length_px=L) \
                    if prefactor == "intercept" else 0.0
            except ValueError:
                offset = 0.0
            self._draw_histograms(entry["mods"], scales, L, lo, hi, offset)
            self._histogram_key = histogram_key
        for curve in self.curves + self.profile_curves:
            curve.setData([], [])
        try:
            if rule == LD_RULES[0]:
                q = self._q_list()
                a = ld.adaptive(entry["mods"], scales, q_list=q, **common)
                ok = np.isfinite(a["h"]) & np.isfinite(a["D"])
                h, D = a["h"][ok], a["D"][ok]
                self.curves[-1].setData(h, D)
                note = f"{int(ok.sum())} of {q.size} q · adaptive half-widths (ln 2)^½ × the tilted sd"
                extra = {"q": q[ok], "eps": a["eps"], "intercept": a["intercept"]}
            else:
                h_lo, h_hi, dh = self.hmin_spin.value(), self.hmax_spin.value(), self.hstep_spin.value()
                if h_hi <= h_lo:
                    raise ValueError("h max must exceed h min")
                if (h_hi - h_lo) / dh > 10000:
                    raise ValueError("h grid exceeds 10000 bins; increase Δh or narrow its range")
                from dynamix.core.partition_tables import moment_grid

                grid = moment_grid(h_lo, h_hi, dh)
                eps = tuple(spin.value() for spin in self.eps_spins)
                if len(set(eps)) != len(eps):
                    raise ValueError("epsilon widths must be distinct for a bin-width comparison")
                f = ld.fixed(entry["mods"], scales, h_grid=grid, eps_list=eps, **common)
                for curve, D in zip(self.curves, f["D"]):
                    finite = np.flatnonzero(np.isfinite(D))
                    if finite.size:
                        span = slice(finite[0], finite[-1] + 1)
                        curve.setData(grid[span], D[span])
                narrowest = int(np.argmin(eps))
                finite = np.isfinite(f["D"][narrowest])
                h, D = grid[finite], f["D"][narrowest][finite]
                spread = f["spread"][np.isfinite(f["spread"])]
                note = (f"ε {', '.join(f'{e:g}' for e in f['eps'])} · spread across ε ≤ {spread.max():.2f}"
                        if spread.size else "insufficient common ε support; spread unavailable")
                extra = {"eps": f["eps"], "D_by_eps": f["D"], "h_grid": grid,
                         "epsilon_count": f["epsilon_count"], "intercept": f["intercept"]}
                if self.profiles_check.isChecked():
                    p = ld.profiles(entry["mods"], scales, h_grid=grid, eps=float(eps[narrowest]), **common)
                    for curve, Dp in zip(self.profile_curves, (p["D_plus"], p["D_minus"])):
                        span = np.flatnonzero(np.isfinite(Dp))
                        if span.size:
                            curve.setData(grid[span[0]:span[-1] + 1], Dp[span[0]:span[-1] + 1])
                    extra.update(D_plus=p["D_plus"], D_minus=p["D_minus"], profiles_eps=p["eps"])
                    note += f" · θ± profiles at ε {eps[narrowest]:g}"
        except ValueError as exc:
            self._fit = None
            for curve in self.curves:
                curve.setData([], [])
            self.readout.setText(str(exc))
            self._sync_legend(rule)
            self.fitChanged.emit({"name": name, "valid": False})
            return
        kept = int(np.sum((np.log2(scales) >= lo - 1e-9) & (np.log2(scales) <= hi + 1e-9)))
        self._fit = {"name": name, "kind": "large deviations", "rule": rule, "set": self.source_combo.currentText(),
                     "h": h, "D": D, "window": (lo, hi), "length_px": L, **extra}
        self._fit["target"] = entry.get("target", "sample large deviations")
        self._fit["valid"] = bool(D.size)
        peak = f" · peak D {np.nanmax(D):.2f} at h {h[np.nanargmax(D)]:.3f}" if D.size else ""
        c10 = f" · c₁,₀ = {extra['intercept']:.2f}" if prefactor == "intercept" else ""
        self.readout.setText(f"{self.source_combo.currentText()} · log₂a {lo:g}–{hi:g} ({kept} scales), "
                             f"L = {L:g} px · {note}{peak}{c10}")
        self._sync_legend(rule)
        self.fitChanged.emit(self._fit)

    def _sync_legend(self, rule: str) -> None:
        self._legend.clear()
        if rule == LD_RULES[0]:
            self._legend.addItem(self.curves[-1], "adaptive bins: (h(q), D(q))")
        else:
            for curve, eps in zip(self.curves, (s.value() for s in self.eps_spins)):
                self._legend.addItem(curve, f"ε = {eps:g}")
            if self.profiles_check.isChecked():
                self._legend.addItem(self.profile_curves[0], "θ⁺ (h_λ ≤ h + ε)")
                self._legend.addItem(self.profile_curves[1], "θ⁻ (h_λ ≥ h − ε)")

    def _draw_histograms(self, mods, scales, L, lo, hi, offset: float = 0.0) -> None:
        """The local exponents' histogram at every scale inside the window, cold to warm from
        the finest scale, on one grid of bins, read against the prefactor ``offset``."""
        for item in self._hist_items:
            self.hist_plot.removeItem(item)
        self._hist_items = []
        self._hist_legend.clear()
        inside = [k for k, a in enumerate(scales) if lo - 1e-9 <= np.log2(a) <= hi + 1e-9 and a < L]
        exps = [ld.local_exponents(mods[k], scales[k], L, offset) for k in inside]
        finite = np.concatenate([e[np.isfinite(e)] for e in exps]) if exps else np.array([])
        if finite.size < 2:
            return
        edges = np.linspace(*np.percentile(finite, (0.5, 99.5)), 61)
        for i, (k, e) in enumerate(zip(inside, exps)):
            counts, _ = np.histogram(e[np.isfinite(e)], bins=edges)
            item = pg.PlotDataItem(edges, counts.astype(np.float64), stepMode="center",
                                   pen=pg.mkPen(_q_color(i, len(inside)), width=1.2), connect="finite")
            self.hist_plot.addItem(item)
            self._hist_items.append(item)
            self._hist_legend.addItem(item, f"a = {scales[k]:g} px")
