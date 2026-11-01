# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Touchette–Beck generalized inversion as an independent fitting tool."""
from __future__ import annotations

import threading

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.core import spectra
from dynamix.core.nonconcave import (gamma_convergence, gaussian_ensemble, generalized_legendre, hollows,
                                     scale_jackknife)
from dynamix.core.partition_tables import moment_grid
from dynamix.shell.theme import current
from dynamix.shell.multifractal_window import WheelAxisViewBox, _q_color, add_zero_lines
from dynamix.shell.plot_zoom import zoom_plot  # noqa: F401  (the source scan's marker; the panels take WheelAxisViewBox)
from dynamix.shell.shared_range import SharedRangeFollower
from dynamix.shell.window_fit import TOOL_WINDOW, show_while_inactive


#: The fewest scales in the window for the leave-one-scale-out band.
JACKKNIFE_MIN_SCALES = 5


def _diagnose(result: dict, cancel=None) -> dict:
    """The diagnostics drawn with the estimate. A minimum on a q or δ limit is a bound and stays
    a loose dot; one held by the largest γ alone is judged by the γ profile: converged, it joins
    the curve, still falling, it is an open circle. Beneath: the Legendre spectrum from the γ = 0
    column of the same τ table, and the leave-one-scale-out band. The hollow test's hull runs
    over the curve's points; its gap is measured at the dots and open circles too, against
    that band."""
    g = result["ensemble"]
    h, D = np.asarray(result["h"], float), np.asarray(result["D"], float)
    tau = np.asarray(g["tau"], float)
    tau = tau[:, :, None] if tau.ndim == 2 else tau
    gammas, deltas = np.asarray(g["gamma"], float), np.asarray(g["delta"], float)
    convergence = gamma_convergence(tau, g["q"], gammas, h, delta_list=deltas)
    boundary = np.asarray(result["boundary"], bool)
    cap = np.asarray(result.get("gamma_cap", np.zeros(h.size, bool)), bool)
    finite = np.isfinite(D)
    pinned = finite & boundary & ~cap
    solid = finite & ~pinned & (convergence["converged"] | ~cap)
    zero = np.flatnonzero(gammas == 0.0)
    legendre = None
    if zero.size:
        # At γ = 0 every δ column is the same table: the δ = 0 one, or the first.
        di = int(np.flatnonzero(deltas == 0.0)[0]) if (deltas == 0.0).any() else 0
        legendre = generalized_legendre(tau[:, zero[:1], di], g["q"], [0.0], h)["D"]
    band = scale_jackknife(g, h, cancel=cancel) if g["n_scales"] >= JACKKNIFE_MIN_SCALES else None
    test = hollows(h, D, None if band is None else band["half_width"], solid=solid)
    return {**result, "convergence": convergence, "pinned": pinned, "solid": solid,
            "open": finite & ~pinned & ~solid, "legendre": legendre, "band": band, "hollow_test": test}


def _hollow_text(hol: dict) -> str:
    verdict = {True: "survives the jackknife", False: "does not survive the jackknife",
               None: "jackknife skipped"}[hol["survives"]]
    return f"hollow: h ∈ [{hol['h_lo']:.3g}, {hol['h_hi']:.3g}], depth {hol['depth']:.2f}, {verdict}"


def _grid_bracketed(h, D, boundary):
    """The h range whose minima lie inside the (q, γ, δ) grid, or None."""
    inside = np.asarray(h, float)[~np.asarray(boundary, bool) & np.isfinite(D)]
    return (float(inside.min()), float(inside.max())) if inside.size else None


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
            if g["n_scales"] >= 3:
                result = _diagnose(result, cancel=self.cancel.is_set)
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
        self.legend = self.plot.addLegend(offset=(-5, 5))
        ink, muted = current().ink, current().ink_muted
        # Beneath the estimate: the leave-one-scale-out band (min to max over the refits) and the
        # Legendre spectrum of the same τ table.
        shade = pg.mkColor(ink)
        shade.setAlpha(45)
        edge = pg.mkColor(ink)
        edge.setAlpha(70)
        self.band_lo = self.plot.plot(pen=pg.mkPen(edge, width=0.5))
        self.band_hi = self.plot.plot(pen=pg.mkPen(edge, width=0.5))
        self.band = pg.FillBetweenItem(self.band_lo, self.band_hi, brush=pg.mkBrush(shade))
        self.plot.addItem(self.band)
        self._band_sample = pg.PlotDataItem(pen=pg.mkPen(shade, width=8))      # the legend's swatch
        self.legendre_curve = self.plot.plot(pen=pg.mkPen(muted, width=1), connect="finite",
                                             name="Legendre (γ = 0)")
        for z, item in ((-3, self.band), (-2, self.band_lo), (-2, self.band_hi), (-1, self.legendre_curve)):
            item.setZValue(z)
        self.curve = self.plot.plot(pen=pg.mkPen(ink, width=2), connect="finite", name="generalized, γ-converged")
        # Scatter items: their legend swatches show the symbol before any point is drawn.
        self.open_points = pg.ScatterPlotItem(symbol="o", size=6, pen=pg.mkPen(ink), brush=pg.mkBrush(None),
                                              name="still falling at γ max")
        self.boundary_points = pg.ScatterPlotItem(symbol="o", size=4, pen=pg.mkPen(None), brush=pg.mkBrush(muted),
                                                  name="on a q or δ limit")
        for item in (self.open_points, self.boundary_points):
            self.plot.addItem(item)
        # The h range whose minima lie inside the (q, γ, δ) grid: two dashed lines and one caption
        # between them, so a narrow range never stacks two labels. The caption hangs from the
        # view's top edge, inside the view, clear of the title row above it.
        floor_pen = pg.mkPen(muted, style=QtCore.Qt.DashLine)
        self.bracket_lines = tuple(pg.InfiniteLine(angle=90, movable=False, pen=floor_pen) for _ in range(2))
        self.bracket_caption = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(None), label="grid-bracketed",
            labelOpts={"position": 1.0, "color": muted, "anchors": [(0.5, 0.0), (0.5, 0.0)]})
        for line in (*self.bracket_lines, self.bracket_caption):
            line.setVisible(False)
            self.plot.addItem(line, ignoreBounds=True)
        # The hollows' shading and dotted sides stop at the caption's foot: their span, a fraction
        # of the view's height, is refitted whenever the view resizes.
        self._hollow_items: list = []
        self.plot.vb.sigResized.connect(lambda _vb: self._fit_hollow_span())
        # The title row carries the hollow captions on one line of fixed height, so the view keeps
        # its place whatever the row says. Its text claims no width, so the text's length never
        # moves the split between the panel columns: a zero maximum width bounds the minimum the
        # label sets from its text on every resize, and the label sits at the view's centre with
        # its text centred on it, cut to the view's width.
        title = self.plot.titleLabel
        title.setMaximumWidth(0)
        self.plot.layout.setAlignment(title, QtCore.Qt.AlignHCenter)
        self._set_title("")
        height = title.item.boundingRect().height() + 4.0      # one line, 2 px clear of the view's top edge
        title.setVisible(True)
        title.setMaximumHeight(height)
        self.plot.layout.setRowFixedHeight(0, height)
        self.plot.vb.sigResized.connect(lambda _vb: self._fit_title())
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
        for item in (self.curve, self.open_points, self.boundary_points, self.legendre_curve, self.band_lo, self.band_hi):
            item.setData([], [])
        self._show_band_entry(False)
        for line in (*self.bracket_lines, self.bracket_caption):
            line.setVisible(False)
        self._draw_hollows([])
        self._set_title("")

    def _show_band_entry(self, on: bool) -> None:
        self.legend.removeItem(self._band_sample)
        if on:
            self.legend.addItem(self._band_sample, "leave-one-scale-out")

    def _draw_bracket(self, bracket) -> None:
        for line in (*self.bracket_lines, self.bracket_caption):
            line.setVisible(bracket is not None)
        if bracket is not None:
            for line, value in zip((*self.bracket_lines, self.bracket_caption), (*bracket, sum(bracket) / 2)):
                line.setValue(value)

    def _draw_hollows(self, found) -> None:
        """A shaded h range per hollow inside the view; the captions go to the title row."""
        for item in self._hollow_items:
            self.plot.removeItem(item)
        self._hollow_items.clear()
        shade = pg.mkColor(current().ink)
        shade.setAlpha(22)
        for hol in found:
            region = pg.LinearRegionItem(values=(hol["h_lo"], hol["h_hi"]), movable=False, brush=pg.mkBrush(shade),
                                         pen=pg.mkPen(current().ink_muted, style=QtCore.Qt.DotLine),
                                         span=self._hollow_span())
            region.setZValue(-4)
            self.plot.addItem(region, ignoreBounds=True)
            self._hollow_items.append(region)

    def _hollow_span(self) -> tuple:
        """From the view's bottom to 1 px below the caption hanging from its top edge."""
        view = self.plot.vb.height()
        caption = self.bracket_caption.label.textItem.boundingRect().height() + 1.0
        return (0.0, max(0.0, 1.0 - caption / view) if view > 0 else 1.0)

    def _fit_hollow_span(self) -> None:
        span = self._hollow_span()
        for region in self._hollow_items:
            region.setSpan(*span)
            for line in region.lines:
                line.viewTransformChanged()           # a line caches its extent until the view changes

    def _set_title(self, text: str) -> None:
        """The spectrum panel's title row, above the view, where a caption never covers the data."""
        self._title_text = text
        self._fit_title()

    def _fit_title(self) -> None:
        """The title on one line: text wider than the view is cut with an ellipsis, refitted as
        the view resizes (the readout carries it whole)."""
        title = self.plot.titleLabel
        font = QtGui.QFont(title.item.font())
        font.setPointSizeF(9.0)
        room = max(self.plot.vb.width() - 2.0 * title.item.document().documentMargin(), 0.0)
        title.setText(QtGui.QFontMetricsF(font).elidedText(self._title_text, QtCore.Qt.ElideRight, room),
                      color=current().ink, size="9pt")

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
        # A minimum on a q or δ limit is a bound, never a value: it stays a loose dot. One held
        # by the largest γ alone joins the curve once the γ profile has stopped falling and is an
        # open circle while it still falls (Touchette & Beck's −∞ with an unbounded γ).
        h, D = np.asarray(result["h"], float), np.asarray(result["D"], float)
        solid, falling, pinned = (np.asarray(result[k], bool) for k in ("solid", "open", "pinned"))
        self.curve.setData(h, np.where(solid, D, np.nan))
        self.open_points.setData(h[falling], D[falling])
        self.boundary_points.setData(h[pinned], D[pinned])
        if result["legendre"] is not None:
            self.legendre_curve.setData(h, result["legendre"])
        band = result["band"]
        if band is not None:
            both = np.isfinite(band["lo"]) & np.isfinite(band["hi"])
            self.band_lo.setData(h[both], band["lo"][both])
            self.band_hi.setData(h[both], band["hi"][both])
        self._show_band_entry(band is not None)
        bracket = _grid_bracketed(h, D, result["boundary"])
        self._draw_bracket(bracket)
        found = result["hollow_test"]["hollows"]
        self._draw_hollows(found)
        self.plot.setXRange(-1.0, 3.0, padding=0.0)              # the spectrum's own window
        self.plot.setYRange(-0.5, 2.5, padding=0.0)
        conv = result["convergence"]
        n_conv, n_finite = int((conv["converged"] & np.isfinite(D)).sum()), int(np.isfinite(D).sum())
        half = band["half_width"][solid & np.isfinite(band["half_width"])] if band is not None else np.empty(0)
        jackknife = (None if band is None else
                     {"n_refits": int(band["D"].shape[0]), "max_half_width": float(half.max()) if half.size else None})
        self._fit = {**result, "name": self._last_name, "kind": "generalized spectrum",
                     "valid": bool(np.isfinite(D).any()), "window": (result["settings"]["lo"], result["settings"]["hi"]),
                     "grid_bracketed": bracket,
                     "gamma_converged": {"n": n_conv, "of": n_finite, "tol": float(conv["tol"])},
                     "jackknife": jackknife, "hollows": found}
        self._ensemble = g
        self._fill_pick(self.gamma_combo, g["gamma"])
        self._fill_pick(self.delta_combo, g["delta"])
        self._draw_diagrams()
        r2 = g["tau_r2"][np.isfinite(g["tau_r2"])]
        quality = f"min partition R² {r2.min():.3f}" if r2.size else "partition R² unavailable"
        d = g["delta"]
        where = (f"minima inside the grid for h in [{bracket[0]:g}, {bracket[1]:g}]; outside, the drawn values are "
                 "upper bounds at the grid's edge" if bracket is not None else
                 "no minimum inside the grid: every drawn value is an upper bound at the grid's edge")
        if jackknife is None:
            spread = (f"leave-one-scale-out skipped: {g['n_scales']} scales in the window, "
                      f"{JACKKNIFE_MIN_SCALES} needed")
        else:
            widest = jackknife["max_half_width"]
            spread = (f"leave-one-scale-out band over {jackknife['n_refits']} refits (c₁,₀ held)"
                      + (f", max half-width {widest:.3f} on the curve" if widest is not None else ""))
        verdicts = ([_hollow_text(hol) for hol in found] if found else
                    ["no hollow beyond the jackknife band" if band is not None else "no hollow deeper than 0.05"])
        self._set_title(" · ".join(verdicts))
        verdict = "; ".join(verdicts)
        self.readout.setText(
            f"{g['n_scales']} of {np.size(g['scales'])} scales · {quality} · δ {d.min():g}–{d.max():g} ({d.size}) · "
            f"c₁,₀ = log₂ {g['reference_amplitude']:.3g} = {np.log2(g['reference_amplitude']):.2f}"
            f"{'' if result['settings']['amplitude'] == 'auto' else ' (set by hand)'}\n"
            f"{where} · {n_conv} of {n_finite} h converged in γ (tol {conv['tol']:.3g}) · "
            f"{int(pinned.sum())} on a q or δ limit (dots), {int(falling.sum())} still falling at γ max (open circles)\n"
            f"{spread} · {verdict}. Negative values retained; finite-grid bounds do not prove occupied h.")
        self.fitChanged.emit(self._fit)

    def closeEvent(self, event):
        self._cancel()
        super().closeEvent(event)
