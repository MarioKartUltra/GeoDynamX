# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Structure functions window: the wavelet leaders' own fitter.

One set of the ``Samples`` contract at a time (leaders, p-leaders, WSE leaders or the bare
coefficients): the structure functions log₂ S(q, j), the mean of the quantity's q-th power over
the valid cells (the leaders' are the wavelet leader structure functions, WLSF), against log₂ a
for the chosen moment orders, the fit range as a draggable region, and off that range ζ(q), their
slopes, by weighted least squares
(Wendt, Abry and Jaffard 2007, eq. 7; uniform weights or the valid counts n_j), τ(q) = ζ(q) − d,
h(q) and D(q) from the tilted means (their eqs. 12 and 13) and the Legendre transform of τ as
the dashed hull on D(h). The pseudo-fractional integration γ the transform applied is taken back
out in the original frame. Every change re-fits and publishes the fit (``fitChanged``) for the
construction window, where the methods' spectra are compared; the fit is named for its source,
set and kind, so a change of γ, δ, the moments or the range replaces it there.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core import leaders, spectra
from dynamix.model.samples import LABELS
from dynamix.shell.multifractal_window import (PhysicalAxis, PhysicalLog2Axis, WheelAxisViewBox, _q_color,
                                               add_zero_lines)
from dynamix.shell.theme import current
from dynamix.shell.shared_range import SharedRangeFollower
from dynamix.shell.window_fit import TOOL_WINDOW, show_while_inactive

__all__ = ["StructureFunctionsWindow"]

#: The moment orders' defaults, the WTMM's.
Q_DEFAULT = (-3.0, 6.0, 0.5)
WEIGHTS = ("uniform", "counts n_j")
#: The tables kept, least recently used first out: a sweep of γ, δ or q computes one per value.
TABLES_KEPT = 8


#: The fit modes as the Partition functions window names them, and the core's own names.
FIT_MODES = {"naive": "naive", "fixed focus": "fixed_focus", "focus": "focus"}


#: The structure functions each set of the ``Samples`` contract gives, by the set's name: the
#: leaders' are the wavelet leader structure functions (WLSF).
SF_KINDS = {"leaders": "WLSF", "p-leaders": "p-leader structure functions",
            "wse": "WSE-leader structure functions", "coefficients": "wavelet coefficient structure functions",
            "cwt-leaders": "CWT leader structure functions", "cwt-p-leaders": "CWT p-leader structure functions"}


class StructureFunctionsWindow(SharedRangeFollower, QtWidgets.QDialog):
    """ζ(q), τ(q), h(q) and D(h) off one sample set, over a chosen fit range."""

    #: The current fit, published on every change: ``name`` (the source, the set and the kind,
    #: the fit's identity), ``label`` (the name with γ and δ), ``kind``, ``q``, ``tau``, ``h``,
    #: ``D`` with their σ's, ``hull`` (the Legendre transform of τ), ``cq`` (the specific heat).
    fitChanged = QtCore.Signal(object)

    def __init__(self, samples: dict, *, gamma: float = 0.0, levels=None, source: str = "",
                 parent=None, physical_of_log2a=None, log2_L=None):
        super().__init__(parent, TOOL_WINDOW)
        show_while_inactive(self)
        self.setModal(False)
        self._samples = samples
        self._gamma = float(gamma)
        self._source = source
        self._physical = physical_of_log2a
        self._dyadic = bool(samples.get("dyadic"))      # a pyramid's levels: the axis reads j
        self._log2_L = None if log2_L is None else float(log2_L)    # the focus anchor, the field extent
        self._tables: OrderedDict = OrderedDict()
        self._fit: dict | None = None
        self._family_items: list = []
        self._fit_items: list = []
        title = "Wavelet leader structure functions"
        self.setWindowTitle(f"{title} — {source}" if source else title)
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
        """The user's path: the region moves, the fit follows and the range is published."""
        self._region.setRegion((float(log2_a_min), float(log2_a_max)))

    _apply_scale_window = set_scale_window

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
        # Leonarduzzi et al. 2019's generalized structure functions (Touchette & Beck at δ = 0): γ
        # the quadratic penalty, δ its centre; γ = 0 is the ordinary table.
        controls.addWidget(QtWidgets.QLabel("γ"))
        self.gamma_spin = QtWidgets.QDoubleSpinBox()
        self.gamma_spin.setRange(-256.0, 256.0)
        self.gamma_spin.setDecimals(2)
        self.gamma_spin.setSingleStep(1.0)
        self.gamma_spin.setToolTip("Generalized structure functions: the weight |T|^q 2^{j γ (φ − δ)²} per leader "
                                   "(Leonarduzzi et al. 2019; Touchette & Beck at δ = 0); 0 is the ordinary table")
        self.gamma_spin.valueChanged.connect(lambda _v: self._refit(rebuild=True))
        controls.addWidget(self.gamma_spin)
        controls.addWidget(QtWidgets.QLabel("δ"))
        self.delta_spin = QtWidgets.QDoubleSpinBox()
        self.delta_spin.setRange(-100.0, 100.0)
        self.delta_spin.setDecimals(3)
        self.delta_spin.setSingleStep(0.1)
        self.delta_spin.setToolTip("The centre of the quadratic penalty, a local exponent; with a narrow q a sweep of "
                                   "δ over the spectrum's support reaches its far parts (Leonarduzzi et al. 2019 §III-C3)")
        self.delta_spin.valueChanged.connect(lambda _v: self._refit(rebuild=True))
        controls.addWidget(self.delta_spin)
        controls.addWidget(QtWidgets.QLabel("Weights"))
        self.weights_combo = QtWidgets.QComboBox()
        self.weights_combo.addItems(list(WEIGHTS))
        self.weights_combo.currentIndexChanged.connect(lambda _i: self._refit())
        controls.addWidget(self.weights_combo)
        controls.addWidget(QtWidgets.QLabel("Fit"))
        self.fit_combo = QtWidgets.QComboBox()
        self.fit_combo.addItems(list(FIT_MODES))
        self.fit_combo.setEnabled(self._log2_L is not None)
        self.fit_combo.setToolTip("naive: one slope per q over the window; fixed focus / focus: the focus regression "
                                  "of the Partition functions window, anchored at the field extent"
                                  if self._log2_L is not None else "The focus modes need the field's extent")
        self.fit_combo.currentIndexChanged.connect(lambda _i: self._refit())
        controls.addWidget(self.fit_combo)
        controls.addWidget(QtWidgets.QLabel("Frame"))
        self.frame_combo = QtWidgets.QComboBox()
        self.frame_combo.addItems(["original", "integrated"])
        self.frame_combo.setEnabled(bool(self._gamma))
        self.frame_combo.setToolTip(f"γ = {self._gamma:g}: the original frame takes γ q out of τ and γ out of h"
                                    if self._gamma else "γ = 0: both frames coincide")
        self.frame_combo.currentIndexChanged.connect(lambda _i: self._refit())
        controls.addWidget(self.frame_combo)
        self._scale_label = QtWidgets.QLabel(self._scale_name())
        controls.addWidget(self._scale_label)
        self.lo_spin, self.hi_spin = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for spin, seed in ((self.lo_spin, lo), (self.hi_spin, hi)):
            spin.setRange(*self._span)
            spin.setDecimals(2)
            spin.setSingleStep(0.25)
            spin.setValue(seed)
            spin.valueChanged.connect(self._on_spins)
            controls.addWidget(spin)
        controls.addWidget(self._make_follow_check())
        # The family's abscissa: the ladder's log₂ a (or j) in pixels, or log₂ of the physical
        # λ* in the bars' unit, as the Partition functions window offers.
        controls.addWidget(QtWidgets.QLabel("x"))
        self._xaxis_combo = QtWidgets.QComboBox()
        self._xaxis_combo.addItems(["log₂ a (px)", "log₂ λ* (unit)"])
        self._xaxis_combo.setEnabled(self._physical is not None)
        self._xaxis_combo.currentTextChanged.connect(self._on_xaxis_changed)
        controls.addWidget(self._xaxis_combo)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(current().ground)
        layout.addWidget(self._glw, 1)
        self.family_plot = self._glw.addPlot(row=0, col=0, colspan=3, viewBox=WheelAxisViewBox())
        self.family_plot.setLabel("bottom", self._scale_name())
        self.family_plot.setLabel("left", "log₂ S(q, a)")
        if self._physical is not None:
            self.family_plot.setAxisItems({"top": PhysicalAxis(self._physical),
                                           "bottom": PhysicalLog2Axis(self._physical, physical_ticks=False)})
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
        # xsmurf's local-cubic Legendre of the fitted τ(q) beside the hull (spectra.legendre_local).
        self._local = pg.PlotDataItem(pen=pg.mkPen(current().amber, style=QtCore.Qt.DotLine, width=1.4),
                                      connect="finite")
        self.plots["D"].addItem(self._local)
        self._d_line = pg.InfiniteLine(angle=0, pos=float(self._samples.get("d", 2)), movable=False,
                                       pen=pg.mkPen(current().ink_muted, style=QtCore.Qt.DashLine))
        self.plots["D"].addItem(self._d_line)
        legend = self.plots["D"].addLegend(offset=(10, -10))          # bottom left: the points sit top right
        legend.addItem(self._curves["D"], "tilted means (eqs. 12, 13)")
        legend.addItem(self._hull, "D**(h), Legendre of ζ")
        self.readout = QtWidgets.QLabel("")
        self.readout.setProperty("reading", "true")
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)
        fit_to_screen(self, 1100, 760)
        remember_geometry(self, "structure_functions")

    def _scale_name(self) -> str:
        return "j" if self._dyadic else "log₂ a"

    def _log2_cells(self, table: dict) -> np.ndarray:
        """Per level of ``table``, log₂ of the field's cell count its sums run over (the set's own
        count where it carries one): what turns log₂ Z into log₂ S."""
        s = self._samples["sets"].get(self.set_combo.currentData(), {})
        cells = np.asarray(s.get("cells", self._samples["cells"]), dtype=np.float64)
        return np.log2(cells[:len(table["log2_scales"])])

    def _physical_unit(self) -> str:
        """The unit the physical conversion reads at the middle of the ladder."""
        if self._physical is None:
            return ""
        log2_s = np.log2(np.asarray(self._samples["scales"], dtype=np.float64))
        for v in (float(np.median(log2_s)), float(log2_s.min()), float(log2_s.max())):
            got = self._physical(v)
            if got is not None:
                return str(got[1])
        return ""

    def _on_xaxis_changed(self, text: str) -> None:
        physical = text.startswith("log₂ λ*") and self._physical is not None
        axis = self.family_plot.getAxis("bottom")
        if isinstance(axis, PhysicalLog2Axis):
            axis.set_physical_ticks(physical)
        if physical:
            self.family_plot.setLabel("bottom", f"log₂ λ* ({self._physical_unit()})")
        else:
            self.family_plot.setLabel("bottom", self._scale_name())

    # -- the fit ----------------------------------------------------------------------------
    def _q_list(self) -> np.ndarray:
        lo, hi, step = self.qmin_spin.value(), self.qmax_spin.value(), self.qstep_spin.value()
        if hi < lo:
            lo, hi = hi, lo
        from dynamix.core.partition_tables import moment_grid

        return moment_grid(lo, hi, step)

    def _generalized(self) -> tuple:
        """``(γ, δ)`` of the generalized structure functions; ``(0, δ)`` is the ordinary table."""
        return float(self.gamma_spin.value()), float(self.delta_spin.value())

    def _table(self) -> dict:
        name = self.set_combo.currentData()
        gamma, delta = self._generalized()
        key = (name, tuple(self._q_list()), gamma, delta if gamma else 0.0)
        if key in self._tables:
            self._tables.move_to_end(key)
        else:
            if gamma:
                length = 2.0 ** self._log2_L if self._log2_L is not None else None
                self._tables[key] = leaders.generalized_table(self._samples, name, self._q_list(), gamma, delta,
                                                              length_px=length)
            else:
                self._tables[key] = leaders.tables({**self._samples, "sets": {name: self._samples["sets"][name]}},
                                                   self._q_list())[name]
            if len(self._tables) > TABLES_KEPT:
                self._tables.popitem(last=False)
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
        self._publish_range(float(lo), float(hi))

    def _refit(self, rebuild: bool = False) -> None:
        table = self._table()
        lo, hi = sorted(float(v) for v in self._region.getRegion())
        weights = None if self.weights_combo.currentIndex() == 0 else table["N_a"]
        d = int(self._samples.get("d", 2))
        fit = leaders.structure_fit(table, lo, hi, weights=weights, gamma=self._gamma,
                                    frame=self.frame_combo.currentText(), d=d)
        mode = FIT_MODES[self.fit_combo.currentText()]
        self.weights_combo.setEnabled(mode == "naive")
        if mode != "naive":
            fit = self._focus_fit(table, fit, lo, hi, mode, d)
        label = self.set_combo.currentText()
        gamma, delta = self._generalized()
        fit["set"] = label
        fit["kind"] = SF_KINDS.get(self.set_combo.currentData(), "structure functions")
        fit["generalized"] = {"gamma": gamma, "delta": delta} if gamma else None
        fit["name"] = (f"{self._source} · {label}, {fit['kind']}" if self._source
                       else f"{label}, {fit['kind']}")
        fit["label"] = fit["name"] + (f", γ = {gamma:g} δ = {delta:g}" if gamma else "")
        # The Legendre routes of a generalized τ carry the penalty back: L_g(h) = ζ_g*(h) − g(h),
        # g = −γ(h − δ)² (Leonarduzzi et al. 2019 (23); Touchette & Beck's f(α) at fixed γ).
        fit["hull"] = spectra.legendre_dh(fit["q_list"], fit["tau"])
        fit["legendre_local"] = spectra.legendre_local(fit["q_list"], fit["tau"])
        if gamma:
            for route in (fit["hull"], fit["legendre_local"]):
                route["D"] = route["D"] + gamma * (route["h"] - delta) ** 2
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
        self._local.setData(fit["legendre_local"]["h"], fit["legendre_local"]["D"])
        self._read(fit, lo, hi)
        self.fitChanged.emit(fit)

    def _rebuild_family(self, table: dict) -> None:
        for item in self._family_items:
            self.family_plot.removeItem(item)
        self._family_items = []
        q, log2_s = table["q_list"], np.asarray(table["log2_scales"], dtype=np.float64)
        # The table is the partition function, Σ over the field's cells; less log₂ of the cell
        # count it is the structure function, the mean, whose slopes are ζ(q).
        for i, row in enumerate(np.asarray(table["tau_qa"]) - self._log2_cells(table)[None, :]):
            # Each scale a dot; the fitted line through them is _draw_fit_lines's.
            item = pg.PlotDataItem(log2_s, row, pen=None, symbol="o", symbolSize=5, symbolPen=None,
                                   symbolBrush=_q_color(i, len(q)))
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
        # The partition function's line, less the cell count's own (−d per octave): slope ζ(q).
        d = int(self._samples.get("d", 2))
        c0 = float(np.nanmean(self._log2_cells(table) + d * np.asarray(table["log2_scales"], dtype=np.float64)))
        for i, (q, slope, at_zero) in enumerate(zip(fit["q_list"], fit["tau"] + gamma * fit["q_list"], fit["tau_0"])):
            if not np.isfinite(slope):
                continue
            item = pg.PlotDataItem(x, at_zero - c0 + (slope + d) * x,
                                   pen=pg.mkPen(_q_color(i, len(fit["q_list"])), style=QtCore.Qt.DashLine, width=1.6))
            self.family_plot.addItem(item)
            self._fit_items.append(item)

    def _focus_fit(self, table: dict, naive: dict, lo: float, hi: float, mode: str, d: int) -> dict:
        """The focus regression (:func:`spectra.fit_spectra`, Mukli's anchor at the field
        extent) over the same table and window: τ, h, D and their errors replace the per-q
        slopes, ζ = τ + d and the (3.6) check follow, and the fit lines pass through the focus,
        their intercept from the focus point and the cell count's own line over the live
        window. When the focus is unavailable (too few live scales) the naive fit stays and
        the reading says why."""
        focus = spectra.fit_spectra(table, lo, hi, eta=self._gamma, frame=self.frame_combo.currentText(),
                                    fit_mode=mode, log2_L=float(self._log2_L))
        info = focus["focus"]
        fit = dict(naive, focus=info, weights="focus-regression")
        if info.get("branch") == "unavailable":
            fit["weights"] = f"{naive['weights']} weights; {self.fit_combo.currentText()} unavailable: {info.get('reason', '')}"
            return fit
        q = fit["q_list"]
        for k in ("tau", "tau_err", "h", "h_err", "D", "D_err"):
            fit[k] = np.asarray(focus[k], dtype=np.float64)
        fit["zeta"], fit["zeta_err"] = fit["tau"] + d, fit["tau_err"]
        fit["sobolev"] = leaders.sobolev_condition(q, fit["zeta"], d)
        x = np.asarray(table["log2_scales"], dtype=np.float64)
        n_a = np.asarray(table["N_a"], dtype=np.float64)
        alive = (x >= lo) & (x <= hi) & (n_a > 0)
        s_n = float(info["s_N"])
        c_n = float(np.log2(n_a[alive]).mean() - s_n * x[alive].mean())
        gamma = self._gamma if fit["frame"] == "original" else 0.0
        slope = fit["tau"] + gamma * q                      # the integrated-frame slopes the panel shows
        h_q = np.where(q != 0, (slope - s_n) / np.where(q != 0, q, 1.0), 0.0)
        fit["tau_0"] = q * (float(info["y0"]) - h_q * float(info["x0"])) + c_n
        return fit

    def _read(self, fit: dict, lo: float, hi: float) -> None:
        q = fit["q_list"]
        at = {v: int(np.argmin(np.abs(q - v))) for v in (0.0, 1.0, 2.0)}
        parts = [f"{fit['set']} · log₂a {lo:.2f}–{hi:.2f} ({len(fit['levels'])} levels) · {fit['weights']} weights"]
        if fit.get("focus") and fit["focus"].get("branch") not in (None, "unavailable"):
            parts.append(f"focus {fit['focus']['branch']} at log₂a = {fit['focus']['x0']:.2f}")
        if fit.get("generalized"):
            g = fit["generalized"]
            parts.append(f"generalized γ = {g['gamma']:g}, δ = {g['delta']:g}: the Legendre routes carry +γ(h − δ)²")
        if np.isclose(q[at[2.0]], 2.0) and np.isfinite(fit["zeta"][at[2.0]]):
            parts.append(f"ζ(2) = {fit['zeta'][at[2.0]]:.3f} ± {fit['zeta_err'][at[2.0]]:.3f}")
        if np.isclose(q[at[1.0]], 1.0) and np.isfinite(fit["h"][at[1.0]]):
            parts.append(f"h(1) = {fit['h'][at[1.0]]:.3f} ± {fit['h_err'][at[1.0]]:.3f}")
        if np.isclose(q[at[0.0]], 0.0) and np.isfinite(fit["tau"][at[0.0]]):
            parts.append(f"τ(0) = {fit['tau'][at[0.0]]:.3f}")
        D = fit["hull"]["D"]
        if np.isfinite(D).any():
            parts.append(f"hull: h* = {fit['hull']['h'][np.nanargmax(D)]:.3f}, D* = {np.nanmax(D):.3f}")
        s = fit.get("sobolev")
        if s and np.isfinite(s["margin"]):
            parts.append(f"Jaffard (3.6) holds (margin {s['margin']:+.2f} at p = {s['at_q']:g})" if s["holds"]
                         else f"Jaffard (3.6) FAILS at p = {s['at_q']:g} (margin {s['margin']:+.2f}): the fit is wrong")
        if self._gamma:
            parts.append(f"γ = {self._gamma:g}, {fit['frame']} frame")
        self.readout.setText(" · ".join(parts))
