# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""MultifractalWindow: the interactive tau(q)/D(h) scale-window fitter.

A non-modal ``Qt.Window`` over the two per-(q, scale) partition tables every scalar WTMM run
already stamps (``result["hd_std"]`` / ``result["hd_cmax"]``). The expensive half of the spectrum
is already in those tables -- what this window adds is the CHOICE: which convention to fit
(sup/no-sup), over which scale window, in which frame. The fit itself is
:func:`dynamix.core.spectra.fit_spectra`, a cheap vectorized slope re-fit (ms), so every drag of
the scale window re-fits live; nothing here ever recomputes WTMM.

**Layout.** One ``pyqtgraph.GraphicsLayoutWidget``, three rows. Top: the THREE partition-function
families side by side -- ``log2 Z(q,a)`` (``tau_qa``), ``H(q,a)`` (``h_qa``) and ``D(q,a)``
(``D_qa``) against ``log2 a``, one q-graded curve family each, every panel carrying a draggable
:class:`~pyqtgraph.LinearRegionItem` mirrored across all three (the books' presentation: Arneodo reads H(q,a) and D(q,a) directly instead of Legendre-transforming tau(q) -- the workbook cell 47's own top row, click-to-bracket
on any panel). Middle: tau(q), D(h), h(q) with propagated 1-sigma error bars; the tau panel also
carries the Pont-2006 negative-branch convexity-loss marker (the right-tail truncation
indicator -- large negative q where concavity breaks is numerical artifact, flagged never
auto-truncated). Bottom: the cell-49 phase-transition surface -- tau(q) vs ln q with the
two-segment fit split at q* (constant-specific-heat approximation: linear-in-ln-q per phase; the
slope break signals freezing of singularities), and the count-weighted R² scan with the auto-best
marker; q* is settable by the spin or by clicking either panel. pyqtgraph, not matplotlib,
deliberately: the region-drag -> live-refit loop is this window's whole point, and pyqtgraph's
interaction primitives are what the rest of the shell already runs on.

**eta / frame (the two forward lifts).** The reference family applies per-scale
``a**power`` multiplies on the way IN (``expo`` on the 1D CWT coefficients directly;
``fracint_alpha`` on the 2D tensor path's WT derivatives) and undoes them at the FIT stage via
``eta``/``frame``. Which lift (if any) was actually applied is the CALLER's knowledge, not this
window's: DynamiX's scalar 2D path applies none (``cwt2d`` has no scale-power multiply), the
tensor path applies ``fracint_alpha`` -- so ``main_window`` derives the seed from the result's own
``params`` (the ``_skeleton_px_size`` derive-honestly pattern) and passes ``eta_seed`` plus a
human-readable ``forward_note`` this window only displays. The eta control stays editable; the
note is what keeps an edited value an informed decision rather than a guess.

**Method label.** What this fits is the CANONICAL (Arneodo WTMM)
formalism -- the window title says so. "Microcanonical" (Turiel's local-exponent/MSC formalism)
is a different, deferred pipeline and deliberately appears nowhere in this UI.

**q-subrange is a display mask only.** The fit is vectorized over the full ``q_list`` regardless
(it costs nothing); the two q spins only choose which rows the four plots SHOW. ``current_fit``
returns the full-q engine truth, so a number read off this window can always be reproduced by
calling :func:`~dynamix.core.spectra.fit_spectra` with the displayed window/eta/frame.

**Theme.** Background and curve inks resolve through ``dynamix.shell.theme`` (the ``canvas.py``
precedent -- this is a real ``dynamix.shell.*`` dependency, declared in devloop's
``_DEPENDS_ON``). The q-gradation endpoints below are DATA identity (which moment order a curve
belongs to), not chrome -- the same Theme-Rule carve-out ``canvas.py``'s palette constants
document -- and are numeric RGB tuples, never hex literals.

**Lifecycle** is owned by ``main_window.py`` (the ``SkeletonDialog`` pattern): a fresh window per
"Multifractal spectrum…" click, the previous one closed first, parented to the main window.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core import spectra
from dynamix.shell.theme import RESTRAINED_DARK

__all__ = ["MultifractalWindow"]

#: q-gradation endpoints for the partition-family curves, most negative q -> most positive q.
#: DATA identity (the curve's moment order), not chrome -- the Theme-Rule carve-out canvas.py's
#: palette constants already document. Cool-to-warm, echoing the skeleton dialog's coolwarm ramp.
Q_COLD = (80, 130, 235)
Q_WARM = (235, 95, 60)

#: The two partition-table conventions, in combo order. sup (``hd_cmax``, the running-supremum /
#: chainmax textbook WTMM convention) is the default; no-sup (``hd_std``) is the raw ``|T(a)|``.
TABLE_CHOICES = ("sup (cmax)", "no-sup (std)")


def _q_color(i: int, n: int) -> tuple:
    """Linear cold->warm ramp over the q index -- plain lerp, no colormap dependency."""
    t = i / (n - 1) if n > 1 else 0.5
    return tuple(int(round(c + t * (w - c))) for c, w in zip(Q_COLD, Q_WARM))


#: The three partition-function families, plotted in this order (the books' presentation).
FAMILY_TABLES = (("tau_qa", "log₂ Z(q,a)"), ("h_qa", "H(q,a)"), ("D_qa", "D(q,a)"))


class MultifractalWindow(QtWidgets.QDialog):
    """Interactive canonical (Arneodo WTMM) spectrum fitter over ``hd_std``/``hd_cmax``."""

    #: Emitted on every re-fit with the current (log2_a_min, log2_a_max) -- the coupled-state
    #: channel the spectrum-construction window inherits its window from (the workbook's own
    #: cell-49-inherits-cell-47 pattern).
    scaleWindowChanged = QtCore.Signal(float, float)

    def __init__(self, hd_std: dict, hd_cmax: dict, *, eta_seed: float = 0.0,
                 forward_note: str = "", parent=None):
        # Qt.Window, not the QDialog default -- a real, independently resizable top-level window,
        # the same choice SkeletonDialog/ProfileDialog made and for the same reason.
        super().__init__(parent, QtCore.Qt.Window)
        self.setModal(False)
        self.setWindowTitle("Multifractal spectrum — canonical (Arneodo WTMM)")

        self._hd_by_choice = {"sup (cmax)": hd_cmax, "no-sup (std)": hd_std}
        self._fit: dict | None = None
        #: Per family table, its PlotDataItems; ``_family_items`` stays the tau list (the
        #: original single-panel attribute, kept as the visibility contract's face).
        self._family_items_by_table: dict[str, list] = {k: [] for k, _ in FAMILY_TABLES}
        self._region_syncing = False

        self._build_ui(eta_seed, forward_note)
        self._rebuild_family()
        self._refit()

    @property
    def _family_items(self) -> list:
        return self._family_items_by_table["tau_qa"]

    # -- public -----------------------------------------------------------------------------
    def current_fit(self) -> dict | None:
        """The last :func:`~dynamix.core.spectra.fit_spectra` result, over the FULL q_list
        (the q-subrange spins mask the display only -- see the module docstring)."""
        return self._fit

    def set_scale_window(self, log2_a_min: float, log2_a_max: float) -> None:
        """Programmatic region set -- same path as a drag (fires the live re-fit)."""
        self._region.setRegion((log2_a_min, log2_a_max))

    # -- construction -----------------------------------------------------------------------
    def _current_hd(self) -> dict:
        return self._hd_by_choice[self._table_combo.currentText()]

    def _build_ui(self, eta_seed: float, forward_note: str) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Table"))
        self._table_combo = QtWidgets.QComboBox()
        self._table_combo.addItems(list(TABLE_CHOICES))
        self._table_combo.currentTextChanged.connect(self._on_table_changed)
        controls.addWidget(self._table_combo)

        controls.addWidget(QtWidgets.QLabel("η"))
        self._eta_spin = QtWidgets.QDoubleSpinBox()
        self._eta_spin.setRange(-5.0, 5.0)
        self._eta_spin.setSingleStep(0.1)
        self._eta_spin.setDecimals(2)
        self._eta_spin.setValue(eta_seed)
        self._eta_spin.valueChanged.connect(lambda _v: self._refit())
        controls.addWidget(self._eta_spin)

        controls.addWidget(QtWidgets.QLabel("frame"))
        self._frame_combo = QtWidgets.QComboBox()
        self._frame_combo.addItems(["original", "integrated"])
        self._frame_combo.currentTextChanged.connect(lambda _t: self._refit())
        controls.addWidget(self._frame_combo)

        q = np.asarray(self._current_hd()["q_list"], dtype=np.float64)
        controls.addWidget(QtWidgets.QLabel("q"))
        self._qmin_spin = QtWidgets.QDoubleSpinBox()
        self._qmax_spin = QtWidgets.QDoubleSpinBox()
        for spin, seed in ((self._qmin_spin, q.min()), (self._qmax_spin, q.max())):
            spin.setRange(float(q.min()), float(q.max()))
            spin.setSingleStep(0.5)
            spin.setDecimals(2)
            spin.setValue(float(seed))
            spin.valueChanged.connect(lambda _v: self._on_q_range_changed())
            controls.addWidget(spin)

        # The phase-transition break point (cell 49): settable here or by clicking either
        # bottom-row panel. Positive branch only, so the floor sits just above zero.
        controls.addWidget(QtWidgets.QLabel("q*"))
        self._qstar_spin = QtWidgets.QDoubleSpinBox()
        self._qstar_spin.setRange(0.02, max(float(q.max()), 0.02))
        self._qstar_spin.setSingleStep(0.1)
        self._qstar_spin.setDecimals(2)
        self._qstar_spin.setValue(min(1.0, max(0.02, float(q.max()))))
        self._qstar_spin.valueChanged.connect(lambda _v: self._update_phase_transition())
        controls.addWidget(self._qstar_spin)

        #: The two-segment fit's live numbers -- slope_L / slope_R / ds + the scan's best q*.
        self._pt_label = QtWidgets.QLabel("")
        self._pt_label.setStyleSheet(f"color: {RESTRAINED_DARK.ink_muted};")
        controls.addWidget(self._pt_label)

        #: What the forward transform actually multiplied in -- display only, derived by the
        #: caller (see the module docstring's eta/frame section). Muted: context, not a control.
        self._forward_label = QtWidgets.QLabel(forward_note)
        self._forward_label.setStyleSheet(f"color: {RESTRAINED_DARK.ink_muted};")
        controls.addStretch(1)
        controls.addWidget(self._forward_label)
        layout.addLayout(controls)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(RESTRAINED_DARK.ground)
        layout.addWidget(self._glw, 1)

        # Row 0: the THREE partition-function families (Z, H, D vs log2 a -- the books' top
        # row), each carrying a mirrored copy of the fit window so click-and-drag works on any
        # panel, exactly like the workbook's click-to-bracket on any of its three panels.
        band = pg.mkColor(RESTRAINED_DARK.amber)
        band.setAlpha(36)
        log2_s = np.asarray(self._current_hd()["log2_scales"], dtype=np.float64)
        span = (float(log2_s.min()), float(log2_s.max()))
        self._family_plots: dict[str, pg.PlotItem] = {}
        self._regions: list[pg.LinearRegionItem] = []
        for col, (key, ylabel) in enumerate(FAMILY_TABLES):
            plot = self._glw.addPlot(row=0, col=col)
            plot.setLabel("bottom", "log₂ a")
            plot.setLabel("left", ylabel)
            region = pg.LinearRegionItem(values=span, orientation="vertical",
                                         brush=pg.mkBrush(band),
                                         pen=pg.mkPen(RESTRAINED_DARK.amber))
            region.setBounds(span)
            region.sigRegionChanged.connect(self._on_region_changed)
            plot.addItem(region)
            self._family_plots[key] = plot
            self._regions.append(region)
        # The tau panel/region remain the canonical names the public API and older callers use.
        self._partition_plot = self._family_plots["tau_qa"]
        self._region = self._regions[0]

        ink = pg.mkPen(RESTRAINED_DARK.ink, width=1.4)
        muted = pg.mkPen(RESTRAINED_DARK.ink_muted)
        self._spectrum_plots = {}
        self._spectrum_curves = {}
        self._spectrum_errbars = {}
        for col, (name, xlabel, ylabel) in enumerate((("tau", "q", "τ(q)"),
                                                      ("D", "h", "D(h)"),
                                                      ("h", "q", "h(q)"))):
            plot = self._glw.addPlot(row=1, col=col)
            plot.setLabel("bottom", xlabel)
            plot.setLabel("left", ylabel)
            self._spectrum_plots[name] = plot
            errbar = pg.ErrorBarItem(pen=muted, beam=0.05)
            plot.addItem(errbar)
            self._spectrum_errbars[name] = errbar
            curve = pg.PlotDataItem(pen=ink, connect="finite", symbol="o", symbolSize=4,
                                    symbolPen=None, symbolBrush=RESTRAINED_DARK.ink)
            plot.addItem(curve)
            self._spectrum_curves[name] = curve

        # Pont-2006 truncation indicator on the tau(q) panel: the negative-branch abscissa where
        # concavity breaks (the right-tail regime's numerical-artifact zone) -- shown as a dashed
        # marker, never an automatic truncation.
        self._convexity_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(RESTRAINED_DARK.ink_muted, style=QtCore.Qt.DashLine),
            label="convexity loss", labelOpts={"position": 0.92,
                                               "color": RESTRAINED_DARK.ink_muted})
        self._convexity_line.setVisible(False)
        self._spectrum_plots["tau"].addItem(self._convexity_line)

        # Row 2: the cell-49 phase-transition surface. Left: fitted tau(q) against ln q with the
        # two-segment fit; right: the count-weighted R² scan over candidate q*.
        muted_dash = pg.mkPen(RESTRAINED_DARK.ink_muted, style=QtCore.Qt.DashLine)
        self._pt_plot = self._glw.addPlot(row=2, col=0, colspan=2)
        self._pt_plot.setLabel("bottom", "ln q  (positive branch)")
        self._pt_plot.setLabel("left", "τ(q)")
        self._pt_points = pg.PlotDataItem(pen=None, symbol="o", symbolSize=5,
                                          symbolPen=None,
                                          symbolBrush=RESTRAINED_DARK.ink)
        self._pt_plot.addItem(self._pt_points)
        self._pt_fit_left = pg.PlotDataItem(pen=pg.mkPen(Q_COLD, width=1.6))
        self._pt_fit_right = pg.PlotDataItem(pen=pg.mkPen(Q_WARM, width=1.6))
        self._pt_plot.addItem(self._pt_fit_left)
        self._pt_plot.addItem(self._pt_fit_right)
        self._pt_qstar_line = pg.InfiniteLine(angle=90, movable=False,
                                              pen=pg.mkPen(RESTRAINED_DARK.amber))
        self._pt_plot.addItem(self._pt_qstar_line)

        self._pt_scan_plot = self._glw.addPlot(row=2, col=2)
        self._pt_scan_plot.setLabel("bottom", "q*")
        self._pt_scan_plot.setLabel("left", "weighted R²")
        self._pt_scan_curve = pg.PlotDataItem(pen=pg.mkPen(RESTRAINED_DARK.ink),
                                              connect="finite")
        self._pt_scan_plot.addItem(self._pt_scan_curve)
        self._pt_best_line = pg.InfiniteLine(angle=90, movable=False, pen=muted_dash,
                                             label="best", labelOpts={"position": 0.9})
        self._pt_scan_plot.addItem(self._pt_best_line)
        self._pt_scan_qstar_line = pg.InfiniteLine(angle=90, movable=False,
                                                   pen=pg.mkPen(RESTRAINED_DARK.amber))
        self._pt_scan_plot.addItem(self._pt_scan_qstar_line)

        # Click-to-set q* on either panel: ln-q abscissa on the left panel, plain q on the scan
        # (the workbook's exact gesture pair, cell 49).
        self._glw.scene().sigMouseClicked.connect(self._on_scene_clicked)

        self.resize(1150, 950)

    # -- redraw -----------------------------------------------------------------------------
    def _q_mask(self, q: np.ndarray) -> np.ndarray:
        return (q >= self._qmin_spin.value()) & (q <= self._qmax_spin.value())

    def _rebuild_family(self) -> None:
        """One PlotDataItem per q row of each of the CURRENT table's three families -- rebuilt
        only on a table switch; a q-subrange change just toggles visibility."""
        hd = self._current_hd()
        log2_s = np.asarray(hd["log2_scales"], dtype=np.float64)
        for key, _label in FAMILY_TABLES:
            plot = self._family_plots[key]
            for item in self._family_items_by_table[key]:
                plot.removeItem(item)
            self._family_items_by_table[key] = []
            table = np.asarray(hd[key], dtype=np.float64)
            n_q = table.shape[0]
            for i in range(n_q):
                item = pg.PlotDataItem(log2_s, table[i], pen=pg.mkPen(_q_color(i, n_q)),
                                       connect="finite")
                plot.addItem(item)
                self._family_items_by_table[key].append(item)
        self._update_family_visibility()

    def _update_family_visibility(self) -> None:
        q = np.asarray(self._current_hd()["q_list"], dtype=np.float64)
        mask = self._q_mask(q)
        for items in self._family_items_by_table.values():
            for item, visible in zip(items, mask):
                item.setVisible(bool(visible))

    def _refit(self) -> None:
        lo, hi = self._region.getRegion()
        self._fit = spectra.fit_spectra(
            self._current_hd(), float(lo), float(hi),
            eta=self._eta_spin.value(), frame=self._frame_combo.currentText())
        self._update_spectrum_plots()
        self._update_phase_transition()
        self._update_convexity_marker()
        self.scaleWindowChanged.emit(float(lo), float(hi))

    def _update_spectrum_plots(self) -> None:
        fit = self._fit
        q = np.asarray(fit["q_list"], dtype=np.float64)
        mask = self._q_mask(q)
        panels = (("tau", q, fit["tau"], None, fit["tau_err"]),
                  ("D", fit["h"], fit["D"], fit["h_err"], fit["D_err"]),
                  ("h", q, fit["h"], None, fit["h_err"]))
        for name, x, y, xerr, yerr in panels:
            x, y = np.asarray(x)[mask], np.asarray(y)[mask]
            self._spectrum_curves[name].setData(x, y)
            finite = np.isfinite(x) & np.isfinite(y)
            kwargs = {"x": x[finite], "y": y[finite]}
            yerr_f = np.asarray(yerr)[mask][finite]
            kwargs["height"] = np.where(np.isfinite(yerr_f), 2.0 * yerr_f, 0.0)
            if xerr is not None:
                xerr_f = np.asarray(xerr)[mask][finite]
                kwargs["width"] = np.where(np.isfinite(xerr_f), 2.0 * xerr_f, 0.0)
            self._spectrum_errbars[name].setData(**kwargs)

    def _update_phase_transition(self) -> None:
        """The cell-49 surface off the CURRENT fit's tau(q): two-segment ln-q fit at the spin's
        q*, the R² scan, and the live readout. All list-index/polyfit cheap -- nothing here ever
        touches the tables, let alone WTMM."""
        fit = self._fit
        if fit is None:
            return
        q = np.asarray(fit["q_list"], dtype=np.float64)
        tau = np.asarray(fit["tau"], dtype=np.float64)
        q_star = float(self._qstar_spin.value())
        pt = spectra.phase_transition_fit(q, tau, q_star)
        scan = spectra.phase_transition_scan(q, tau)

        keep = np.isfinite(q) & np.isfinite(tau) & (q > 0.01)
        lnq, t = np.log(q[keep]), tau[keep]
        self._pt_points.setData(lnq, t)
        for line_item, side_mask, slope_key in (
                (self._pt_fit_left, lnq <= np.log(q_star), "slope_L"),
                (self._pt_fit_right, lnq > np.log(q_star), "slope_R")):
            slope = pt[slope_key]
            if np.isfinite(slope) and side_mask.sum() >= 2:
                xs = lnq[side_mask]
                ys = t[side_mask]
                # anchor the drawn line on the side's own mean (the OLS line passes through it)
                line_item.setData(xs, slope * (xs - xs.mean()) + ys.mean())
            else:
                line_item.setData([], [])
        self._pt_qstar_line.setPos(float(np.log(q_star)))
        self._pt_scan_qstar_line.setPos(q_star)
        self._pt_scan_curve.setData(scan["q_grid"], scan["r2w"])
        best = scan["best_q"]
        self._pt_best_line.setVisible(bool(np.isfinite(best)))
        if np.isfinite(best):
            self._pt_best_line.setPos(float(best))
        self._pt_fit = pt
        self._pt_scan = scan
        self._pt_label.setText(
            f"sₗ={pt['slope_L']:.3f}  sᵣ={pt['slope_R']:.3f}  Δs={pt['ds']:.3f}"
            + (f"  ·  auto q*={best:.2f}" if np.isfinite(best) else ""))

    def _update_convexity_marker(self) -> None:
        fit = self._fit
        if fit is None:
            return
        q_loss = spectra.negative_branch_convexity_loss(fit["q_list"], fit["tau"])
        self._convexity_line.setVisible(q_loss is not None)
        if q_loss is not None:
            self._convexity_line.setPos(float(q_loss))

    # -- gestures ---------------------------------------------------------------------------
    def _on_region_changed(self) -> None:
        """Any panel's region drag mirrors onto the other two, then one re-fit -- guarded so the
        programmatic mirroring never recurses into more re-fits."""
        if self._region_syncing:
            return
        sender = self.sender()
        lo, hi = sender.getRegion() if sender in self._regions else self._region.getRegion()
        self._region_syncing = True
        try:
            for region in self._regions:
                if region is not sender:
                    region.setRegion((lo, hi))
        finally:
            self._region_syncing = False
        self._refit()

    def _on_scene_clicked(self, ev) -> None:
        """Click-to-set q* (cell 49's gesture pair): the tau-vs-ln-q panel maps the click's x
        back through exp; the scan panel takes it as q* directly."""
        for plot, is_log in ((self._pt_plot, True), (self._pt_scan_plot, False)):
            vb = plot.getViewBox()
            if vb.sceneBoundingRect().contains(ev.scenePos()):
                x = float(vb.mapSceneToView(ev.scenePos()).x())
                q_star = float(np.exp(x)) if is_log else x
                lo, hi = self._qstar_spin.minimum(), self._qstar_spin.maximum()
                self._qstar_spin.setValue(min(max(q_star, lo), hi))
                return

    def _on_table_changed(self, _text: str) -> None:
        self._rebuild_family()
        self._refit()

    def _on_q_range_changed(self) -> None:
        # Display mask only -- the fit (and with it the phase-transition surface, which always
        # reads the full positive branch) is already over the full q_list (module docstring).
        self._update_family_visibility()
        self._update_spectrum_plots()
