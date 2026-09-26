# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""SpectrumWindow: the singularity-spectrum CONSTRUCTION window.

A non-modal ``Qt.Window`` sibling of :class:`~dynamix.shell.multifractal_window.
MultifractalWindow`, showing HOW a D(h) is constructed, three ways on one panel:

- **Canonical (Arneodo direct)** -- the parametric (h(q), D(q)) points from
  :func:`dynamix.core.spectra.fit_spectra` over the hd tables, with propagated error bars. The
  Arneodo route never differentiates a fitted tau: it fits the three partition-function families
  and reads D(h) parametrically. This is the construction of
  record; everything else on the panel is commentary on it.
- **Legendre hull of the fitted tau(q)** -- :func:`~dynamix.core.spectra.legendre_dh`, an
  OPTIONAL dashed overlay: hull-only by theorem, so
  where it departs from the canonical points is exactly the tau-kink / phase-transition tell
  (Touchette-Beck 2006 Thm 2-4; the generalized Gaussian ensemble that would RESOLVE the
  ambiguity is a designed follow-up, not this overlay).
- **Microcanonical histogram (eq. 21)** -- :func:`dynamix.core.microcanonical.dh_histogram`
  over an ``h_map``-bearing result, when one is supplied. Labeled "microcanonical", never
  "Legendre" (the naming law). When the h-map came from the ``punctual`` estimator its label
  carries the Pont-2006 right-limb caveat: the single-scale estimator linearizes the right
  tail; only multiscale methods resolve it.

**Chain grouping (the cell-52 port).** When the result's ``chains`` ride along, the grouping
strip classifies the chain population dominant / non-dominant through the partition function's
tilted measure (:mod:`dynamix.core.chain_groups`) and -- behind its own checkbox, the workbook's
own gating for the expensive half -- overlays the two subsets' canonical D(h) alongside the
all-chains one (the three-spectra overlay). Subset tables delegate to ``wtmm_ebsd`` and are
membership-memoized, so repeated gestures are dict hits. ``groupingChanged`` broadcasts the
boolean membership for any future canvas highlight (not wired tonight -- the selection-layer
regression history says that wiring is a decision, not a default).

**Precompute posture.** Everything a gesture touches is a cached-table mask + polyfit
(fit window, q, table, eta), a vectorized softmax (classification), or a memoized delegation
(subset tables). Nothing here ever recomputes WTMM or an h-map; the histogram is computed once
at construction. The scale window can also be DRIVEN by an open MultifractalWindow
(``set_scale_window`` -- main_window connects the fitter's ``scaleWindowChanged``), the
workbook's cell-49-inherits-cell-47 coupling.

**Lifecycle** is main_window's, the SkeletonDialog pattern: fresh window per click, previous
one closed first, parented to the main window.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core import chain_groups, spectra
from dynamix.shell.multifractal_window import Q_COLD, Q_WARM, TABLE_CHOICES
from dynamix.shell.theme import RESTRAINED_DARK

__all__ = ["SpectrumWindow"]

#: Human labels for the membership rules, in ``chain_groups.MODES`` order (cell 52's dropdown).
MODE_LABELS = ("Top percentile", "log|M|^q at scale", "log|M|^q on sup (all scales)")


class SpectrumWindow(QtWidgets.QDialog):
    """D(h) construction: canonical points, optional Legendre hull, microcanonical histogram,
    and the dominant/non-dominant chain-group overlay."""

    #: The latest dominant/non-dominant membership (bool per chain), emitted on every change.
    groupingChanged = QtCore.Signal(object)

    def __init__(self, hd_std: "dict | None" = None, hd_cmax: "dict | None" = None, *,
                 chains: "list | None" = None, scales=None, h_map=None,
                 h_map_estimator: str = "", eta_seed: float = 0.0,
                 forward_note: str = "", min_chain_len: int = 2,
                 log2_L: "float | None" = None, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setModal(False)
        self.setWindowTitle("Singularity spectrum — construction")

        if (hd_std is None or hd_cmax is None) and h_map is None:
            raise ValueError("SpectrumWindow needs the partition tables (hd_std AND hd_cmax) "
                             "or an h_map -- got neither")
        self._hd_by_choice = ({"sup (cmax)": hd_cmax, "no-sup (std)": hd_std}
                              if hd_std is not None and hd_cmax is not None else None)
        self._chains = chains if chains else None
        #: The focus anchor -- log2 of the field extent (Mukli Eq. 7); None disables the
        #: focus fit modes (the anchor is the SIGNAL LENGTH, unknowable from tables alone).
        self._log2_L = float(log2_L) if log2_L is not None else None
        self._scales = np.asarray(scales, dtype=np.float64) if scales is not None else None
        self._min_chain_len = int(min_chain_len)
        self._fit: dict | None = None
        self._grouping: "np.ndarray | None" = None

        # The histogram half is static -- one compute at construction (precompute posture).
        self._hist: "tuple | None" = None
        self._hist_label = ""
        if h_map is not None:
            from dynamix.core.microcanonical import dh_histogram
            h_centers, D = dh_histogram(np.asarray(h_map))
            if h_centers.size:
                self._hist = (h_centers, D)
                self._hist_label = "microcanonical histogram (eq. 21)"
                if h_map_estimator == "punctual":
                    # Pont/Turiel/Perez-Vicente 2006: the single-scale estimator linearizes
                    # the right tail; only multiscale methods resolve it.
                    self._hist_label += " — right limb unreliable (punctual estimator)"

        self._build_ui(eta_seed, forward_note)
        self._recompute()

    # -- public -----------------------------------------------------------------------------
    def current_fit(self) -> dict | None:
        """The latest full-q :func:`~dynamix.core.spectra.fit_spectra` result (None when the
        window is histogram-only)."""
        return self._fit

    def current_grouping(self) -> "np.ndarray | None":
        """The latest dominant/non-dominant membership, or None without chains."""
        return self._grouping

    def set_scale_window(self, log2_a_min: float, log2_a_max: float) -> None:
        """Programmatic fit-window set -- the fitter-window coupling channel."""
        if self._hd_by_choice is None:
            return
        for spin, value in ((self._lo_spin, log2_a_min), (self._hi_spin, log2_a_max)):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        self._recompute()

    # -- construction -----------------------------------------------------------------------
    def _current_hd(self) -> "dict | None":
        if self._hd_by_choice is None:
            return None
        return self._hd_by_choice[self._table_combo.currentText()]

    def _build_ui(self, eta_seed: float, forward_note: str) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        has_hd = self._hd_by_choice is not None

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Table"))
        self._table_combo = QtWidgets.QComboBox()
        self._table_combo.addItems(list(TABLE_CHOICES))
        self._table_combo.currentTextChanged.connect(lambda _t: self._recompute())
        controls.addWidget(self._table_combo)

        # Focus regression (Schadner/Mukli -- core/spectra.focus_regression):
        # joint fit of every q through one focus. The readout is the honesty display the
        # method demands: branch, x-hat_0, and Delta-h naive vs focus SIDE BY SIDE -- the
        # estimator buys valid spectra by narrowing Delta-h (bias ~ 1/|xi|), and the reader
        # must see the price.
        controls.addWidget(QtWidgets.QLabel("Fit"))
        self._fitmode_combo = QtWidgets.QComboBox()
        self._fitmode_combo.addItems(["naive", "fixed focus", "focus"])
        if self._log2_L is None:
            self._fitmode_combo.setEnabled(False)
            self._fitmode_combo.setToolTip(
                "focus fitting needs the field extent (log2 L) -- open from a computed result")
        self._fitmode_combo.currentTextChanged.connect(lambda _t: self._recompute())
        controls.addWidget(self._fitmode_combo)
        self._focus_readout = QtWidgets.QLabel("")
        controls.addWidget(self._focus_readout)

        if has_hd:
            log2_s = np.asarray(self._current_hd()["log2_scales"], dtype=np.float64)
            lo0, hi0 = float(log2_s.min()), float(log2_s.max())
        else:
            lo0, hi0 = 0.0, 1.0
        controls.addWidget(QtWidgets.QLabel("log₂a"))
        self._lo_spin = QtWidgets.QDoubleSpinBox()
        self._hi_spin = QtWidgets.QDoubleSpinBox()
        for spin, seed in ((self._lo_spin, lo0), (self._hi_spin, hi0)):
            spin.setRange(lo0 - 1.0, hi0 + 1.0)
            spin.setSingleStep(0.25)
            spin.setDecimals(2)
            spin.setValue(seed)
            spin.valueChanged.connect(lambda _v: self._recompute())
            controls.addWidget(spin)

        controls.addWidget(QtWidgets.QLabel("η"))
        self._eta_spin = QtWidgets.QDoubleSpinBox()
        self._eta_spin.setRange(-5.0, 5.0)
        self._eta_spin.setSingleStep(0.1)
        self._eta_spin.setDecimals(2)
        self._eta_spin.setValue(eta_seed)
        self._eta_spin.valueChanged.connect(lambda _v: self._recompute())
        controls.addWidget(self._eta_spin)

        self._hull_check = QtWidgets.QCheckBox("Legendre hull (τ)")
        self._hull_check.toggled.connect(lambda _c: self._recompute())
        controls.addWidget(self._hull_check)

        for w in (self._table_combo, self._lo_spin, self._hi_spin, self._eta_spin,
                  self._hull_check):
            w.setEnabled(has_hd)

        self._forward_label = QtWidgets.QLabel(forward_note)
        self._forward_label.setStyleSheet(f"color: {RESTRAINED_DARK.ink_muted};")
        controls.addStretch(1)
        controls.addWidget(self._forward_label)
        layout.addLayout(controls)

        # Grouping strip (cell 52): only when the chains AND the tables rode along -- subset
        # spectra need both the population and the q grid.
        self._group_controls: list[QtWidgets.QWidget] = []
        if self._chains is not None and has_hd and self._scales is not None:
            group = QtWidgets.QHBoxLayout()
            group.addWidget(QtWidgets.QLabel("Group: scale"))
            # The chains' own scale ladder bounds the classification scale -- in a real result
            # it equals the tables' ladder, but the chains are the classification's authority.
            n_sc = int(self._scales.size)
            self._gscale_spin = QtWidgets.QSpinBox()
            self._gscale_spin.setRange(0, max(0, n_sc - 1))
            self._gscale_spin.setValue(n_sc // 2)
            group.addWidget(self._gscale_spin)
            group.addWidget(QtWidgets.QLabel("q"))
            self._gq_spin = QtWidgets.QDoubleSpinBox()
            self._gq_spin.setRange(-10.0, 10.0)
            self._gq_spin.setSingleStep(0.25)
            self._gq_spin.setValue(2.0)
            group.addWidget(self._gq_spin)
            self._gmode_combo = QtWidgets.QComboBox()
            self._gmode_combo.addItems(list(MODE_LABELS))
            group.addWidget(self._gmode_combo)
            group.addWidget(QtWidgets.QLabel("top %"))
            self._gpct_spin = QtWidgets.QDoubleSpinBox()
            self._gpct_spin.setRange(1.0, 100.0)
            self._gpct_spin.setValue(20.0)
            group.addWidget(self._gpct_spin)
            group.addWidget(QtWidgets.QLabel("log θ"))
            self._gthresh_spin = QtWidgets.QDoubleSpinBox()
            self._gthresh_spin.setRange(-50.0, 50.0)
            self._gthresh_spin.setValue(0.0)
            group.addWidget(self._gthresh_spin)
            self._gsup_check = QtWidgets.QCheckBox("classify on sup")
            group.addWidget(self._gsup_check)
            group.addWidget(QtWidgets.QLabel("min len"))
            self._gminlen_spin = QtWidgets.QSpinBox()
            self._gminlen_spin.setRange(1, 30)
            self._gminlen_spin.setValue(3)
            group.addWidget(self._gminlen_spin)
            #: The workbook's own gate on the expensive half (subset partition builds).
            self._gspectra_check = QtWidgets.QCheckBox("Group spectra")
            group.addWidget(self._gspectra_check)
            self._gcount_label = QtWidgets.QLabel("")
            self._gcount_label.setStyleSheet(f"color: {RESTRAINED_DARK.ink_muted};")
            group.addWidget(self._gcount_label)
            group.addStretch(1)
            layout.addLayout(group)
            for w in (self._gscale_spin, self._gq_spin, self._gpct_spin, self._gthresh_spin,
                      self._gminlen_spin):
                w.valueChanged.connect(lambda _v: self._recompute())
            self._gmode_combo.currentIndexChanged.connect(lambda _i: self._recompute())
            self._gsup_check.toggled.connect(lambda _c: self._recompute())
            self._gspectra_check.toggled.connect(lambda _c: self._recompute())
            self._group_controls = [self._gscale_spin]

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(RESTRAINED_DARK.ground)
        layout.addWidget(self._glw, 1)
        self._plot = self._glw.addPlot()
        self._plot.setLabel("bottom", "h")
        self._plot.setLabel("left", "D(h)")
        self._plot.addLegend(offset=(10, 10))

        muted = pg.mkPen(RESTRAINED_DARK.ink_muted)
        self._d_line = pg.InfiniteLine(
            angle=0, pos=2.0, movable=False,
            pen=pg.mkPen(RESTRAINED_DARK.ink_muted, style=QtCore.Qt.DashLine))
        self._plot.addItem(self._d_line)
        self._canon_err = pg.ErrorBarItem(pen=muted, beam=0.01)
        self._plot.addItem(self._canon_err)
        self._canon_curve = pg.PlotDataItem(
            pen=None, symbol="o", symbolSize=6, symbolPen=None,
            symbolBrush=RESTRAINED_DARK.ink, name="canonical (Arneodo direct)")
        self._plot.addItem(self._canon_curve)
        self._hull_curve = pg.PlotDataItem(
            pen=pg.mkPen(Q_WARM, style=QtCore.Qt.DashLine, width=1.6),
            connect="finite", name="Legendre hull of fitted τ(q)")
        self._plot.addItem(self._hull_curve)
        if self._hist is not None:
            hist_curve = pg.PlotDataItem(*self._hist, pen=pg.mkPen(Q_COLD, width=1.4),
                                         connect="finite", name=self._hist_label)
            self._plot.addItem(hist_curve)
        self._dom_curve = pg.PlotDataItem(pen=pg.mkPen(Q_WARM, width=1.2),
                                          connect="finite", name="dominant chains")
        self._non_curve = pg.PlotDataItem(pen=pg.mkPen(Q_COLD, width=1.2),
                                          connect="finite", name="non-dominant chains")
        self._plot.addItem(self._dom_curve)
        self._plot.addItem(self._non_curve)
        self._peak_text = pg.TextItem(color=RESTRAINED_DARK.ink, anchor=(0.5, 1.2))
        self._plot.addItem(self._peak_text)

        self.resize(950, 700)

    # -- recompute --------------------------------------------------------------------------
    def _recompute(self) -> None:
        hd = self._current_hd()
        if hd is not None:
            lo, hi = float(self._lo_spin.value()), float(self._hi_spin.value())
            mode = {"naive": "naive", "fixed focus": "fixed_focus",
                    "focus": "focus"}[self._fitmode_combo.currentText()]
            self._fit = spectra.fit_spectra(
                hd, lo, hi, eta=self._eta_spin.value(), frame="original",
                fit_mode=mode, log2_L=self._log2_L if mode != "naive" else None)
            f = self._fit.get("focus")
            if f is None:
                self._focus_readout.setText("")
            elif f.get("branch") == "unavailable":
                self._focus_readout.setText(f"focus: unavailable ({f.get('reason', '')})")
            else:
                hn = np.asarray(f["h_q_mean_naive"]); hf = np.asarray(f["h_q_mean"])
                dhn = float(hn.max() - hn.min()); dhf = float(hf.max() - hf.min())
                x0 = f["x0"]
                x0_s = "inf" if not np.isfinite(x0) else f"{x0:.2f}"
                self._focus_readout.setText(
                    f"{f['branch']} · x₀={x0_s} · Δh {dhn:.3f}→{dhf:.3f} · "
                    f"SSE {f['sse']:.3g} (naive {f['sse_naive']:.3g})")
            h = np.asarray(self._fit["h"])
            D = np.asarray(self._fit["D"])
            finite = np.isfinite(h) & np.isfinite(D)
            self._canon_curve.setData(h[finite], D[finite])
            h_err = np.asarray(self._fit["h_err"])[finite]
            D_err = np.asarray(self._fit["D_err"])[finite]
            self._canon_err.setData(
                x=h[finite], y=D[finite],
                width=np.where(np.isfinite(h_err), 2.0 * h_err, 0.0),
                height=np.where(np.isfinite(D_err), 2.0 * D_err, 0.0))
            if finite.any():
                i = int(np.nanargmax(np.where(finite, D, -np.inf)))
                self._peak_text.setPos(float(h[i]), float(D[i]))
                self._peak_text.setText(f"h*={h[i]:.3f}  D*={D[i]:.3f}")
            if self._hull_check.isChecked():
                hull = spectra.legendre_dh(self._fit["q_list"], self._fit["tau"])
                self._hull_curve.setData(hull["h"], hull["D"])
            else:
                self._hull_curve.setData([], [])
        self._update_grouping()

    def _update_grouping(self) -> None:
        if self._chains is None or self._current_hd() is None or self._scales is None:
            return
        payload = chain_groups.grouping_payload(self._chains, self._scales.size)
        mode = chain_groups.MODES[self._gmode_combo.currentIndex()]
        dom = chain_groups.classify_chains(
            payload, scale_idx=int(self._gscale_spin.value()),
            q=float(self._gq_spin.value()), mode=mode,
            dom_percentile=float(self._gpct_spin.value()),
            log_thresh=float(self._gthresh_spin.value()),
            chainmax=self._gsup_check.isChecked(),
            min_len=int(self._gminlen_spin.value()))
        changed = self._grouping is None or not np.array_equal(dom, self._grouping)
        self._grouping = dom
        self._gcount_label.setText(f"{int(dom.sum())} dominant / {dom.size}")
        if changed:
            self.groupingChanged.emit(dom)
        if not self._gspectra_check.isChecked():
            self._dom_curve.setData([], [])
            self._non_curve.setData([], [])
            return
        hd = self._current_hd()
        q_list = np.asarray(hd["q_list"], dtype=np.float64)
        lo, hi = float(self._lo_spin.value()), float(self._hi_spin.value())
        sup = self._table_combo.currentText() == TABLE_CHOICES[0]
        for curve, mask in ((self._dom_curve, dom), (self._non_curve, ~dom)):
            if not mask.any():
                curve.setData([], [])
                continue
            tables = chain_groups.subset_hd(self._chains, self._scales, q_list, mask,
                                            min_chain_len=self._min_chain_len)
            table = tables[1] if sup else tables[0]
            fit = spectra.fit_spectra(table, lo, hi, eta=self._eta_spin.value(),
                                      frame="original")
            h = np.asarray(fit["h"])
            D = np.asarray(fit["D"])
            finite = np.isfinite(h) & np.isfinite(D)
            curve.setData(h[finite], D[finite])
