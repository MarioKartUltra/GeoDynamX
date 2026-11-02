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

**The D(h) comparison.** Every spectrum sent to the D(h) comparison
(:mod:`dynamix.shell.spectra_board`: each fitting window's Send to D(h) comparison, this
window's own Send buttons for its canonical points, with the tangent at q_c when asked, its
histogram and its box-counting construction, so a state of the construction can be frozen and
compared with a later one, and its "Send landed results…" for the other layers' landed tables)
draws on D(h) beside this window's own construction, its τ(q) on the τ panel, named in the legend by its display name with
the full name as tooltip; the board's inspector is a tool window in front of this one, shown while
the board holds entries and by the Comparison… button. Nothing arrives by itself: the comparison shows what was sent. Three panels, τ(q), h(q)
and c(q), sit beside D(h); ``set_fits`` draws fits handed to it on all four panels in their own
hues, with a Show menu and a Spectra… list while it holds any.

**Lifecycle** is main_window's, the SkeletonDialog pattern: fresh window per click, previous
one closed first, parented to the main window.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.core import chain_groups, spectra
from dynamix.model.spectra_board import plain
from dynamix.shell.gc_guard import weak_method
from dynamix.shell.multifractal_window import (LEADERS_METHODS, _SOURCE_HUES, Q_COLD, Q_WARM, TABLE_CHOICES,
                                                WheelAxisViewBox, add_zero_lines, tip_sample_sets)
from dynamix.shell.spectra_board import (BoardComparison, ReplaceButton, ancestor_with, board_of, dh_view_range,
                                         from_fit, kind_of_method, send, send_button, send_landed)
from dynamix.shell.theme import current
from dynamix.shell.window_fit import WrappingLayout, TOOL_WINDOW, show_while_inactive

__all__ = ["SpectrumWindow"]


def _tangent_html(intercept: float, q_c: float) -> str:
    """The tangent's label: its intercept at h = 0 is −τ(q), and q_c carries a subscript."""
    from dynamix.shell.theme import current
    import pyqtgraph as pg
    colour = pg.mkColor(current().ink_muted).name()
    return (f'<span style="color: {colour}">−τ(q) = {intercept:.3f} · q<sub>crit</sub> = {q_c:g}</span>')

#: Human labels for the membership rules, in ``chain_groups.MODES`` order (cell 52's dropdown).
MODE_LABELS = ("Top percentile", "log|M|^q at scale", "log|M|^q on sup (all scales)")


_WrappingLayout = WrappingLayout      # the construction window's rows, from window_fit

def _row(*widgets) -> QtWidgets.QWidget:
    """``widgets`` side by side as one item of a wrapping strip."""
    row = QtWidgets.QWidget()
    box = QtWidgets.QHBoxLayout(row)
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(4)
    for widget in widgets:
        box.addWidget(widget)
    return row


class SpectraInspector(QtWidgets.QWidget):
    """The construction's published spectra, one checkable row each in its colour, kept in front
    of the construction: a row shows or hides its spectrum on every panel, in step with the Show
    menu."""

    def __init__(self, window):
        super().__init__(window, TOOL_WINDOW | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        show_while_inactive(self)
        self.setWindowTitle("Spectra")
        self.list = QtWidgets.QListWidget()
        QtWidgets.QVBoxLayout(self).addWidget(self.list)
        self.list.itemChanged.connect(self._on_item_changed)
        self.resize(320, 260)

    def refresh(self, rows) -> None:
        """``rows``: ``(name, label, shown, colour)`` per spectrum, in drawing order; the list is
        rebuilt when the spectra change and only re-checked otherwise."""
        role = QtCore.Qt.ItemDataRole.UserRole
        names = [r[0] for r in rows]
        self.list.blockSignals(True)
        if [self.list.item(i).data(role) for i in range(self.list.count())] != names:
            self.list.clear()
            for name, label, _shown, colour in rows:
                item = QtWidgets.QListWidgetItem(label)
                item.setData(role, name)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                swatch = QtGui.QPixmap(12, 12)
                swatch.fill(pg.mkColor(colour))
                item.setIcon(QtGui.QIcon(swatch))
                self.list.addItem(item)
        for i, (_name, label, shown, _colour) in enumerate(rows):
            item = self.list.item(i)
            item.setText(label)
            item.setCheckState(QtCore.Qt.CheckState.Checked if shown else QtCore.Qt.CheckState.Unchecked)
        self.list.blockSignals(False)

    def _on_item_changed(self, item) -> None:
        action = self.parentWidget()._show_actions.get(item.data(QtCore.Qt.ItemDataRole.UserRole))
        if action is not None:
            action.setChecked(item.checkState() == QtCore.Qt.CheckState.Checked)


class SpectrumWindow(BoardComparison, QtWidgets.QDialog):
    """D(h) construction: canonical points, optional Legendre hull, microcanonical histogram,
    the dominant/non-dominant chain-group overlay, and the D(h) comparison."""

    #: The latest dominant/non-dominant membership (bool per chain), emitted on every change.
    groupingChanged = QtCore.Signal(object)

    def __init__(self, hd_std: "dict | None" = None, hd_cmax: "dict | None" = None, *,
                 chains: "list | None" = None, scales=None, h_map=None,
                 h_map_estimator: str = "", h_map_method: str = "", eta_seed: float = 0.0,
                 forward_note: str = "", min_chain_len: int = 2,
                 log2_L: "float | None" = None, parent=None, all_tables=None,
                 filter_note: str = "", dh_curves=None, gh_field=None,
                 source: str = "canonical (Arneodo direct)", kind: str = "partition",
                 per_scale: "dict | None" = None, tables: "dict | None" = None, fits=None,
                 sources=None):
        super().__init__(parent, TOOL_WINDOW)
        show_while_inactive(self)
        #: Other layers' landed results, each a dict with ``name`` and ``hd_std``/``hd_cmax``
        #: and/or ``dh_hist``: listed by "Send landed results…".
        self._landed = [dict(s) for s in (sources or [])]
        self._h_map_method = str(h_map_method or "")
        #: The published fits of the methods' fitting tools, by name; their items by name.
        self._fits: dict = {}
        self._fit_items: dict = {}
        #: Per-scale sets of moduli for the Gaussian ensemble:
        #: name -> {"mods": [per scale], "scales": px, "weights": per-scale count factors or None}.
        self._per_scale = {k: v for k, v in (per_scale or {}).items()
                           if v.get("mods") and v.get("scales") is not None}
        #: What the canonical points are: the WTMM's direct construction, or the moment route.
        self._source = str(source)
        self._kind = str(kind or "partition")
        #: One table under two names (the moment route), or one named table: the choice means
        #: nothing and is hidden.
        self._one_table = (len(tables) == 1 if tables
                           else hd_std is not None and hd_std is hd_cmax)
        #: The legend lists what is drawn: (item, name) pairs synced after every change.
        self._legend_entries: list = []
        self._in_legend: dict = {}
        self.setModal(False)
        self.setWindowTitle("Singularity spectrum — construction")
        #: ``(hd_std, hd_cmax)`` over EVERY chain of the transform when the tables given are
        #: over the rack's filtered chains, with the note that says so; None when they are the same.
        self._all_tables = tuple(all_tables) if all_tables is not None else None
        self._filter_note = str(filter_note)
        if self._filter_note:
            self.setWindowTitle(self.windowTitle() + f" — {self._filter_note}")
        self.all_chains_check = None

        if (hd_std is None or hd_cmax is None) and h_map is None:
            raise ValueError("SpectrumWindow needs the partition tables (hd_std AND hd_cmax) "
                             "or an h_map -- got neither")
        self._hd_by_choice = (dict(tables) if tables else
                              {"sup (cmax)": hd_cmax, "no-sup (std)": hd_std}      # TABLE_CHOICES order
                              if hd_std is not None and hd_cmax is not None else None)
        self._filtered_tables = (hd_std, hd_cmax)
        self._chains = chains if chains else None
        #: The focus anchor -- log2 of the field extent (Mukli Eq. 7); None disables the
        #: focus fit modes (the anchor is the SIGNAL LENGTH, unknowable from tables alone).
        self._log2_L = float(log2_L) if log2_L is not None else None
        self._scales = np.asarray(scales, dtype=np.float64) if scales is not None else None
        self._min_chain_len = int(min_chain_len)
        self._fit: dict | None = None
        self._grouping: "np.ndarray | None" = None

        # The histogram construction: eq. 21 with the eq. 39 bars, over the bins and the
        # resolution the controls choose (recomputed from the h-map, cheap); the per-scale
        # curves (Turiel 2008 test ii) come precomputed with the result; the gradient
        # histogram (GH) is computed from the field on request.
        self._h_map = np.asarray(h_map) if h_map is not None else None   # its own dtype: the same bins
        self._dh_curves = list(dh_curves) if dh_curves else []
        self._gh_field = np.asarray(gh_field, dtype=np.float64) if gh_field is not None else None
        self._bins = 80
        self._r0_choice = "1/√N"
        self._hist: "tuple | None" = None
        self._hist_dD = None
        self._hist_label = ""
        if h_map is not None:
            self._compute_hist()
            if self._hist is not None:
                self._hist_label = "microcanonical histogram (eq. 21)"
                if str(h_map_method or "") in LEADERS_METHODS:
                    # Cardo & Corvalán 2009: Turiel's density formula on the leaders' extrapolated
                    # exponents, at one resolution; the across-scale D_LD lives in the Large
                    # deviations window and the box counting.
                    r0 = self._r0_rel() or 1.0 / np.sqrt(np.asarray(h_map).size)
                    self._hist_label = (f"GMWLP histogram (Cardo & Corvalán; the leaders' local exponents, "
                                        f"eq. 21 at r₀ = {r0:.2g})")
                if h_map_estimator == "punctual":
                    # Pont/Turiel/Perez-Vicente 2006: the single-scale estimator linearizes
                    # the right tail; only multiscale methods resolve it.
                    self._hist_label += " — right limb unreliable (punctual estimator)"

        self._build_ui(eta_seed, forward_note)
        self._recompute()
        self.set_fits(fits or [])
        self.attach_board(board_of(self))

    # -- public -----------------------------------------------------------------------------
    def current_fit(self) -> dict | None:
        """The latest full-q :func:`~dynamix.core.spectra.fit_spectra` result (None when the
        window is histogram-only)."""
        return self._fit

    def set_fits(self, fits) -> None:
        """The methods' published fits to draw on the four panels: each ``{"name", "label"?, "q",
        "tau", "h", "D", "cq", "hull"?}``, or ``{"name", "h", "D"}`` for a method that constructs
        D(h) alone; a name already drawn is redrawn, one no longer listed is removed. The name is
        the fit's identity; the legend reads the label when the tool gives one."""
        wanted = {f["name"]: f for f in fits}
        for name in [n for n in self._fit_items if n not in wanted]:
            self.show_menu.removeAction(self._show_actions.pop(name))
            self._fit_hues.pop(name, None)
            gone = self._fit_items.pop(name).values()
            for item in gone:
                self._plot.removeItem(item)
                for plot in self.method_plots.values():
                    plot.removeItem(item)
            self._legend_entries = [(it, n) for it, n in self._legend_entries if it not in gone]
        for i, (name, fit) in enumerate(wanted.items()):
            if name not in self._fit_items:
                hue = _SOURCE_HUES[len(self._fit_items) % len(_SOURCE_HUES)]
                self._fit_hues[name] = hue
                pen = pg.mkPen(hue, width=1.4)
                items = {k: pg.PlotDataItem(pen=pen, connect="finite", symbol="o", symbolSize=3,
                                            symbolPen=None, symbolBrush=hue) for k in ("tau", "h", "cq")}
                items["D"] = pg.PlotDataItem(pen=pen, connect="finite", symbol="o", symbolSize=3,
                                             symbolPen=None, symbolBrush=hue)
                items["hull"] = pg.PlotDataItem(pen=pg.mkPen(hue, width=1.0, style=QtCore.Qt.DashLine),
                                                connect="finite")
                # Outside the auto-range: a published fit (a log-cumulant extrapolation past a
                # phase transition shoots far out) never sets the panels' extent.
                for k in ("tau", "h", "cq"):
                    self.method_plots[k].addItem(items[k], ignoreBounds=True)
                self._plot.addItem(items["D"], ignoreBounds=True)
                self._plot.addItem(items["hull"], ignoreBounds=True)
                self._fit_items[name] = items
                action = self.show_menu.addAction(name)
                action.setCheckable(True)
                action.setChecked(True)
                action.toggled.connect(lambda _on, n=name: self._apply_shown(n))
                self._show_actions[name] = action
                label_of = weak_method(self._fit_label)
                self._legend_entries.append((items["D"], partial(label_of, name)))
                self._legend_entries.append((items["hull"], partial(label_of, name, " (D**, envelope)")))
            items = self._fit_items[name]
            if fit.get("tau") is not None:               # a D(h)-only method (large deviations) has no τ
                q = np.asarray(fit["q"], dtype=np.float64)
                items["tau"].setData(q, np.asarray(fit["tau"], dtype=np.float64))
                items["h"].setData(q, np.asarray(fit["h"], dtype=np.float64))
                items["cq"].setData(q, np.asarray(fit["cq"], dtype=np.float64))
            items["D"].setData(np.asarray(fit["h"], dtype=np.float64), np.asarray(fit["D"], dtype=np.float64))
            hull = fit.get("hull")
            items["hull"].setData(np.asarray(hull["h"]) if hull else [], np.asarray(hull["D"]) if hull else [])
            self._apply_shown(name, sync=False)
        self._fits = wanted
        for button in (self.show_button, self.spectra_button):
            button.setVisible(bool(wanted))
        self._sync_legend()
        self._refresh_spectra_inspector()
        if hasattr(self, "_chord"):
            self._update_kink_geometry()

    def _fit_label(self, name: str, suffix: str = "") -> str:
        return self._fits[name].get("label", name) + suffix

    def _apply_shown(self, name: str, *, sync: bool = True) -> None:
        """One published spectrum's items on or off (its Show action), the hull also behind the
        hull switch."""
        items = self._fit_items.get(name)
        action = self._show_actions.get(name)
        if items is None or action is None:
            return
        on = action.isChecked()
        for k in ("tau", "h", "cq", "D"):
            items[k].setVisible(on)
        items["hull"].setVisible(on and self._hull_check.isChecked())
        if sync:
            self._sync_legend()
            self._refresh_spectra_inspector()

    def _open_spectra_inspector(self) -> None:
        if self.spectra_inspector is None:
            self.spectra_inspector = SpectraInspector(self)
        self._refresh_spectra_inspector()
        self.spectra_inspector.show()
        self.spectra_inspector.raise_()

    def _refresh_spectra_inspector(self) -> None:
        if self.spectra_inspector is None:
            return
        self.spectra_inspector.refresh([(name, self._fit_label(name), action.isChecked(), self._fit_hues.get(name, "w"))
                                        for name, action in self._show_actions.items() if name in self._fits])

    def fit_mode(self) -> str:
        """The fit mode as :func:`dynamix.core.spectra.fit_spectra` names it."""
        return {"naive": "naive", "fixed focus": "fixed_focus",
                "focus": "focus"}[self._fitmode_combo.currentText()]

    def _table_key(self) -> str:
        """The table the construction reads: ``std`` or ``cmax``, or a named table's name."""
        text = self._table_combo.currentText()
        return {"no-sup (std)": "std", "sup (cmax)": "cmax"}.get(text, text)

    def _pinned_table(self) -> "str | None":
        """The table the histogram is pinned to, None while it is not."""
        return self._table_key() if self.pin_check.isChecked() else None

    def _fit_settings(self) -> dict:
        return {"eta": float(self._eta_spin.value()), "frame": self._frame_combo.currentText(),
                "all_chains": bool(self.all_chains_check is not None and self.all_chains_check.isChecked())}

    def send_landed_results(self, names=None) -> int:
        """Send the other layers' landed results named in ``names`` (asked in a list when None),
        fitted under this window's scale window, fit mode, η, frame and table. Returns how many
        spectra were sent."""
        lo, hi = self._scale_window()
        return send_landed(self, self._landed, names, lo=lo, hi=hi, mode=self.fit_mode(),
                           eta=float(self._eta_spin.value()), frame=self._frame_combo.currentText(),
                           log2_L=self._log2_L, table=self._table_key(), settings=self._fit_settings())

    def send_primary(self, with_tangent: "bool | None" = None, replace: bool = False) -> bool:
        """The canonical points to the D(h) comparison: every q row with the error bars D(h)
        draws (the fit's standard errors over the scale window), τ(q), and the tangent at q_c
        drawn on D(h) when ``with_tangent`` (the "with tangent" box by default); with ``replace``,
        in place of the canonical points this window sent last."""
        if self._fit is None:
            return send(self, None, "no canonical points to send")
        if with_tangent is None:
            with_tangent = self.with_tangent_check.isChecked()
        name = self._source if self._one_table else f"{self._source} · {self._table_combo.currentText()}"
        entry = from_fit(self, {**self._fit, "name": name}, kind_of_method(self._h_map_method),
                         scale_window=self._scale_window(), fit_mode=self.fit_mode(), table=self._table_key(),
                         settings=self._fit_settings(), tangent=self._tangent_now() if with_tangent else None)
        return send(self, entry, "no finite D(h) to send", replace=replace)

    def send_histogram(self, replace: bool = False) -> bool:
        """The histogram construction as D(h) draws it (pinned to the fit's D₀ when the pin is
        on), with its eq. 39 bars, under its legend name."""
        if self._hist is None:
            return send(self, None, "no histogram construction to send")
        h, D = (np.asarray(a, dtype=np.float64) for a in self._hist_curve.getData())
        pinned = self.pin_check.isChecked()
        settings = {"pin_to_D0": pinned, "bins": self._bins, "r0": self._r0_choice,
                    **(self._fit_settings() if pinned else {})}
        entry = from_fit(self, {"name": self._hist_curve_name(), "h": h, "D": D, "D_err": self._hist_dD}, "histogram",
                         scale_window=self._scale_window() if pinned else None,
                         fit_mode=self.fit_mode() if pinned else None,
                         table=self._pinned_table(), settings=settings)
        return send(self, entry, "the histogram construction has no finite D(h)", replace=replace)

    def _boxcount(self) -> tuple:
        """Eq. 20 across r over the h-map, on coarser bins than the histogram's: a thin bin is
        a sparse set below its spacing."""
        from dynamix.core.microcanonical import box_counting_spectrum

        return box_counting_spectrum(self._h_map, n_bins=max(8, self._bins // 4))

    def send_boxcount(self, replace: bool = False) -> bool:
        """The box-counting construction across r (eq. 20) with its slopes' standard errors."""
        if self._h_map is None:
            return send(self, None, "no box-counting construction to send")
        h, D, dD = self._boxcount()
        entry = from_fit(self, {"name": "box-counting across r (eq. 20)", "h": h, "D": D, "D_err": dD},
                         "boxcount", settings={"bins": max(8, self._bins // 4)})
        return send(self, entry, "the box-counting construction has no finite D(h)", replace=replace)

    def _export_figure(self) -> None:
        main = ancestor_with(self.parentWidget(), "_export_comparison_figure")
        if main is not None:
            main._export_comparison_figure(self, "dh_comparison")

    def drawn_fit(self) -> "dict | None":
        """The construction's canonical points as a figure record names them: the name and kind,
        the scale window, the fit mode, the table, the settings, its error bars and the tangent
        drawn on D(h); None for a histogram-only window."""
        if self._fit is None:
            return None
        tangent = self._tangent_now()
        return plain({"name": self._source, "kind": kind_of_method(self._h_map_method),
                      "scale_window": list(self._scale_window()), "fit_mode": self.fit_mode(),
                      "table": self._table_key(), "settings": self._fit_settings(), "errors": True,
                      "tangent": tangent})

    def figure_record(self) -> dict:
        """What an exported figure of this window records: the D(h) range on show, the
        construction's own canonical fit and every spectrum drawn from the D(h) comparison."""
        return {"view": "D(h) construction", "view_range": dh_view_range(self._plot),
                "fit": self.drawn_fit(), "spectra": self.drawn_spectra()}

    def current_grouping(self) -> "np.ndarray | None":
        """The latest dominant/non-dominant membership, or None without chains."""
        return self._grouping

    def broadcast_grouping(self) -> None:
        """Emit :attr:`groupingChanged` with the current membership while the strip's "Select on
        views" box is checked -- a no-op without chains or with the box off."""
        if self._grouping is not None and self.select_views_check.isChecked():
            self.groupingChanged.emit(self._grouping)

    def _on_all_chains_toggled(self, checked: bool) -> None:
        std, cmax = self._all_tables if checked else self._filtered_tables
        self._hd_by_choice = {"sup (cmax)": cmax, "no-sup (std)": std}
        self._recompute()

    def set_frame(self, text: str) -> None:
        """Follow the fitter's frame (``original`` / ``integrated``)."""
        if text in ("original", "integrated"):
            self._frame_combo.setCurrentText(text)

    def set_fit_mode(self, text: str) -> None:
        """Follow the fitter's fit mode (``naive`` / ``fixed focus`` / ``focus``); ignored
        when the focus modes are unavailable here."""
        if text in ("naive", "fixed focus", "focus") and self._fitmode_combo.isEnabled():
            self._fitmode_combo.setCurrentText(text)

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

        controls = _WrappingLayout()
        self._table_label = QtWidgets.QLabel("Table")
        controls.addWidget(self._table_label)
        self._table_combo = QtWidgets.QComboBox()
        self._table_combo.addItems(list(self._hd_by_choice) if self._hd_by_choice else list(TABLE_CHOICES))
        self._table_combo.currentTextChanged.connect(lambda _t: self._recompute())
        controls.addWidget(self._table_combo)
        for w in (self._table_label, self._table_combo):
            w.setVisible(not self._one_table)
        if self._all_tables is not None and self._hd_by_choice is not None:
            # The rack's filters narrowed the chains: the tables are over the survivors; this
            # switch shows the whole transform's instead.
            self.all_chains_check = QtWidgets.QCheckBox("all chains")
            self.all_chains_check.setToolTip(self._filter_note)
            self.all_chains_check.toggled.connect(self._on_all_chains_toggled)
            controls.addWidget(self.all_chains_check)

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
        self._focus_readout.setWordWrap(True)
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
        # The frame the exponents read in: original takes the η lift back out, integrated keeps it.
        self._frame_combo = QtWidgets.QComboBox()
        self._frame_combo.addItems(["original", "integrated"])
        self._frame_combo.currentTextChanged.connect(lambda _t: self._recompute())
        controls.addWidget(self._frame_combo)

        self._hull_check = QtWidgets.QCheckBox("D**(h), concave envelope (Legendre of τ)")
        self._hull_check.toggled.connect(lambda _c: self._recompute())
        controls.addWidget(self._hull_check)
        self._local_check = QtWidgets.QCheckBox("Legendre (local cubic, xsmurf)")
        self._local_check.setToolTip("xsmurf's legendre2: a cubic through the five τ(q) points around each q, "
                                     "h = its derivative there, D = q h − τ; a kink survives except within two "
                                     "steps of it")
        self._local_check.toggled.connect(lambda _c: self._recompute())
        controls.addWidget(self._local_check)
        # A nonconcave point has no supporting line (Touchette & Beck's point b): the kink
        # geometry can be left out.
        self._geometry_check = QtWidgets.QCheckBox("kink geometry (tangents at q_c, chord)")
        self._geometry_check.setChecked(True)
        self._geometry_check.setToolTip("The two one-sided tangents of τ at q_c and the envelope's chord across "
                                        "(h_l, h_h); leave them out where the kink is no phase transition, a "
                                        "nonconcave point admits no supporting line")
        self._geometry_check.toggled.connect(lambda _c: self._recompute())
        controls.addWidget(self._geometry_check)
        self.tangent_check = QtWidgets.QCheckBox("tangent at q_c")
        self.tangent_check.setChecked(True)
        self.tangent_check.setToolTip("The tangent to D(h) at the phase transition, at the convexity-loss q_c of "
                                      "the negative branch; off, no tangent is drawn")
        self.tangent_check.toggled.connect(lambda _c: self._update_tangent())
        controls.addWidget(self.tangent_check)
        # The canonical points frozen on the D(h) comparison, to compare with a later state.
        self.send_button = QtWidgets.QPushButton("Send to D(h) comparison")
        self.send_button.setToolTip("The canonical points with the error bars D(h) draws (the fit's standard "
                                    "errors over the scale window), to the D(h) comparison")
        self.send_button.setEnabled(has_hd)
        self.send_button.clicked.connect(lambda _c=False: self.send_primary())
        controls.addWidget(self.send_button)
        self.replace_button = ReplaceButton(self, lambda w: kind_of_method(w._h_map_method), self.send_primary,
                                            "canonical points", self.send_button, table=SpectrumWindow._table_key)
        self.replace_button.watch(self._table_combo.currentTextChanged)
        controls.addWidget(self.replace_button)
        self.with_tangent_check = QtWidgets.QCheckBox("with tangent")
        self.with_tangent_check.setToolTip("Send the tangent at q_c drawn on D(h) with the points")
        self.with_tangent_check.setEnabled(has_hd)
        self.tangent_check.toggled.connect(lambda on: self.with_tangent_check.setEnabled(on and has_hd))
        controls.addWidget(self.with_tangent_check)
        self.send_landed_button = QtWidgets.QPushButton("Send landed results…")
        self.send_landed_button.setToolTip("Fit the other layers' landed results under this window's scale "
                                           "window, fit mode, η and frame, and send the chosen ones to the "
                                           "D(h) comparison")
        self.send_landed_button.setEnabled(bool(self._landed))
        self.send_landed_button.clicked.connect(lambda _c=False: self.send_landed_results())
        controls.addWidget(self.send_landed_button)
        self.export_button = QtWidgets.QPushButton("Export figure…")
        self.export_button.setToolTip("These plots as a PNG, with a record naming the construction and every "
                                      "spectrum drawn from the D(h) comparison")
        self.export_button.clicked.connect(lambda _c=False: self._export_figure())
        controls.addWidget(self.export_button)
        # Show: each published spectrum on or off, on every panel.
        self.show_button = QtWidgets.QToolButton()
        self.show_button.setText("Show")
        self.show_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.show_menu = QtWidgets.QMenu(self.show_button)
        self.show_button.setMenu(self.show_menu)
        self._show_actions: dict = {}
        controls.addWidget(self.show_button)
        # The same choice as a small window kept in front, one row per spectrum.
        self.spectra_button = QtWidgets.QPushButton("Spectra…")
        self.spectra_button.setToolTip("A list of the published spectra to show or hide, kept in front")
        self.spectra_button.clicked.connect(self._open_spectra_inspector)
        self.spectra_inspector = None
        self._fit_hues: dict = {}
        controls.addWidget(self.spectra_button)
        # The D(h) comparison's inspector, the same kind of window: it opens by itself when a
        # spectrum is sent; this brings it back.
        self.comparison_button = QtWidgets.QPushButton("Comparison…")
        self.comparison_button.setToolTip("The spectra sent to the D(h) comparison: rename (R), recolour, reorder, "
                                          "remove; a window kept in front")
        self.comparison_button.clicked.connect(self.show_board_inspector)
        controls.addWidget(self.comparison_button)
        # The construction's own series on D(h) can step aside for the comparison.
        self.points_check = QtWidgets.QCheckBox("own points")
        self.points_check.setChecked(True)
        self.points_check.setToolTip("The construction's own canonical points and their bars on D(h)")
        self.points_check.toggled.connect(self._on_points_toggled)
        controls.addWidget(self.points_check)
        self.hist_check = None

        # The histogram construction's own controls, with an h-map to build it from.
        self.bins_spin = self.r0_combo = self.per_scale_check = self.gh_check = None
        self.boxcount_check = self.pin_check = self.send_hist_button = self.send_box_button = None
        self.replace_hist_button = self.replace_box_button = None
        if self._h_map is not None:
            self.hist_check = QtWidgets.QCheckBox("histogram")
            self.hist_check.setChecked(True)
            self.hist_check.setToolTip("The histogram construction on D(h): its points, bars, peak label and "
                                       "per-scale curves")
            self.hist_check.toggled.connect(self._on_hist_toggled)
            controls.addWidget(self.hist_check)
            controls.addWidget(QtWidgets.QLabel("bins"))
            self.bins_spin = QtWidgets.QSpinBox()
            self.bins_spin.setRange(10, 400)
            self.bins_spin.setValue(self._bins)
            self.bins_spin.valueChanged.connect(lambda _v: self._on_hist_controls_changed())
            controls.addWidget(self.bins_spin)
            controls.addWidget(QtWidgets.QLabel("r₀"))
            self.r0_combo = QtWidgets.QComboBox()
            self.r0_combo.addItem("1/√N")
            if self._scales is not None and self._scales.size:
                self.r0_combo.addItem("r_min/L")
            self.r0_combo.currentTextChanged.connect(lambda _t: self._on_hist_controls_changed())
            controls.addWidget(self.r0_combo)
            self.per_scale_check = QtWidgets.QCheckBox("per-scale histograms")
            self.per_scale_check.setEnabled(bool(self._dh_curves))
            self.per_scale_check.toggled.connect(self._on_per_scale_toggled)
            controls.addWidget(self.per_scale_check)
            self.gh_check = QtWidgets.QCheckBox("gradient histogram (GH)")
            self.gh_check.setEnabled(self._gh_field is not None)
            self.gh_check.toggled.connect(self._on_gh_toggled)
            controls.addWidget(self.gh_check)
            # Eq. 20 across resolutions, no pinning: the level sets' box-counting dimensions.
            self.boxcount_check = QtWidgets.QCheckBox("box-counting across r (eq. 20)")
            self.boxcount_check.toggled.connect(self._on_boxcount_toggled)
            self.send_box_button = send_button("box-counting construction", self.send_boxcount)
            self.replace_box_button = ReplaceButton(self, "boxcount", self.send_boxcount, "box-counting construction",
                                                    self.send_box_button)
            # One row, so the wrapping never parts the Send and Replace from what they send.
            controls.addWidget(_row(self.boxcount_check, self.send_box_button, self.replace_box_button))
            # Eq. 21 with the fit's own D₀ = -tau(0) in place of d, to compare shapes.
            self.pin_check = QtWidgets.QCheckBox("pin peak to D₀ of the fit")
            self.pin_check.setEnabled(has_hd)
            self.pin_check.toggled.connect(lambda _c: self._update_hist_items())
            controls.addWidget(self.pin_check)
            self.send_hist_button = send_button("histogram construction", self.send_histogram)
            self.send_hist_button.setEnabled(self._hist is not None)
            self.replace_hist_button = ReplaceButton(self, "histogram", self.send_histogram,
                                                     "histogram construction", self.send_hist_button,
                                                     table=SpectrumWindow._pinned_table)
            self.replace_hist_button.watch(self._table_combo.currentTextChanged, self.pin_check.toggled)
            # Named, so the histogram's Send never reads as part of the pin's label.
            controls.addWidget(_row(QtWidgets.QLabel("histogram"), self.send_hist_button, self.replace_hist_button))

        for w in (self._table_combo, self._lo_spin, self._hi_spin, self._eta_spin,
                  self._hull_check):
            w.setEnabled(has_hd)

        self._forward_label = QtWidgets.QLabel(forward_note)
        self._forward_label.setWordWrap(True)
        self._forward_label.setStyleSheet(f"color: {current().ink_muted};")
        controls.addWidget(self._forward_label)
        layout.addLayout(controls)

        # Grouping strip (cell 52): only when the chains AND the tables rode along -- subset
        # spectra need both the population and the q grid.
        self._group_controls: list[QtWidgets.QWidget] = []
        if self._chains is not None and has_hd and self._scales is not None:
            group = _WrappingLayout()
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
            #: The dominant chains become the shared selection (the views colour them).
            self.select_views_check = QtWidgets.QCheckBox("Select on views")
            self.select_views_check.setChecked(True)
            group.addWidget(self.select_views_check)
            self._gcount_label = QtWidgets.QLabel("")
            self._gcount_label.setStyleSheet(f"color: {current().ink_muted};")
            group.addWidget(self._gcount_label)
            layout.addLayout(group)
            for w in (self._gscale_spin, self._gq_spin, self._gpct_spin, self._gthresh_spin,
                      self._gminlen_spin):
                w.valueChanged.connect(lambda _v: self._recompute())
            self._gmode_combo.currentIndexChanged.connect(lambda _i: self._recompute())
            self._gsup_check.toggled.connect(lambda _c: self._recompute())
            self._gspectra_check.toggled.connect(lambda _c: self._recompute())
            self.select_views_check.toggled.connect(lambda _c: self.broadcast_grouping())
            self._group_controls = [self._gscale_spin]

        # Beyond the hull (spec 15): the Gaussian ensemble's points with their γ over a
        # per-scale set, and the kink shading of τ(q). The large-deviation spectrum has its own
        # tool (large_deviations_window) and arrives here as a published fit.
        self.gaussian_check = self.gamma_max_spin = self.gamma_steps_spin = None
        self.kink_check = None
        self._gamma_readout = QtWidgets.QLabel("")
        self._gamma_readout.setWordWrap(True)
        self._gamma_readout.setStyleSheet(f"color: {current().ink_muted};")
        if self._per_scale or has_hd:
            beyond = _WrappingLayout()
            if self._per_scale:
                # The ensemble's set, kept off the layout: a hidden child of the window, deleted
                # with it together with its connection.
                self.ld_source_combo = QtWidgets.QComboBox(self)
                self.ld_source_combo.hide()
                self.ld_source_combo.addItems(list(self._per_scale))
                tip_sample_sets(self.ld_source_combo)
                self.ld_source_combo.currentTextChanged.connect(lambda _t: self._recompute())
                self.gaussian_check = QtWidgets.QCheckBox("Ensemble candidates γ")
                self.gaussian_check.setToolTip("Weighted-mean candidates can include mixed populations. Use the separate Generalized spectrum tool for Touchette–Beck inversion.")
                self.gaussian_check.setEnabled(has_hd)
                self.gaussian_check.toggled.connect(lambda _c: self._recompute())
                beyond.addWidget(self.gaussian_check)
                beyond.addWidget(QtWidgets.QLabel("γ max"))
                self.gamma_max_spin = QtWidgets.QDoubleSpinBox()
                self.gamma_max_spin.setRange(0.25, 64.0)
                self.gamma_max_spin.setValue(8.0)
                self.gamma_max_spin.setSingleStep(1.0)
                self.gamma_max_spin.valueChanged.connect(lambda _v: self._recompute())
                beyond.addWidget(self.gamma_max_spin)
                beyond.addWidget(QtWidgets.QLabel("steps"))
                self.gamma_steps_spin = QtWidgets.QSpinBox()
                self.gamma_steps_spin.setRange(1, 12)
                self.gamma_steps_spin.setValue(5)
                self.gamma_steps_spin.valueChanged.connect(lambda _v: self._recompute())
                beyond.addWidget(self.gamma_steps_spin)
                beyond.addWidget(self._gamma_readout)
            if has_hd:
                self.kink_check = QtWidgets.QCheckBox("kinks")
                self.kink_check.toggled.connect(lambda _c: self._recompute())
                beyond.addWidget(self.kink_check)
            layout.addLayout(beyond)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(current().ground)
        layout.addWidget(self._glw, 1)
        # Four panels: τ(q) top left, D(h) top right, h(q) bottom left, c(q) bottom right; the
        # methods' published fits draw on all four (set_fits).
        self._plot = self._glw.addPlot(row=0, col=1, viewBox=WheelAxisViewBox())   # wheel: x; Shift: y
        self._plot.setLabel("bottom", "h")
        self._plot.setLabel("left", "D(h)")
        self.method_plots = {}
        for name, ylabel, (row, col) in (("tau", "τ(q)", (0, 0)), ("h", "h(q)", (1, 0)),
                                         ("cq", "c(q) = −q² τ″", (1, 1))):
            plot = self._glw.addPlot(row=row, col=col, viewBox=WheelAxisViewBox())
            plot.setLabel("bottom", "q")
            plot.setLabel("left", ylabel)
            plot.getAxis("left").enableAutoSIPrefix(False)
            add_zero_lines(plot)
            self.method_plots[name] = plot
        self._plot.getAxis("bottom").enableAutoSIPrefix(False)
        self._legend = self._plot.addLegend(offset=(10, 10))
        add_zero_lines(self._plot)
        # The D(h) tangent at the phase transition off the current fit (the automatic q_c of
        # spectra.phase_transition_tangent), across the whole view; hidden while the negative
        # branch stays concave.
        self._tangent_line = pg.PlotDataItem(
            pen=pg.mkPen(current().ink_muted, style=QtCore.Qt.DashLine), connect="finite")
        self._tangent_marker = pg.ScatterPlotItem(size=7, pen=None, brush=current().ink_muted)
        self._tangent_label = pg.TextItem(anchor=(0, 1), color=current().ink_muted)
        for item in (self._tangent_line, self._tangent_marker, self._tangent_label):
            item.setVisible(False)
            # Out of the auto-range: the line spans the view, and a view that grew to hold its
            # ends would make it span a larger view, without end.
            self._plot.addItem(item, ignoreBounds=True)
        self._plot.getViewBox().sigXRangeChanged.connect(lambda *_a: self._update_tangent())
        self._build_board(self._plot, self.method_plots["tau"])  # its inspector is a tool window, in front
        # The window's own fit on the method panels, in the ink, under the published ones.
        self._own_items = {k: pg.PlotDataItem(pen=pg.mkPen(current().ink, width=1.6), connect="finite")
                           for k in ("tau", "h", "cq")}
        for k, item in self._own_items.items():
            self.method_plots[k].addItem(item)
        # Touchette & Beck 2006 Figs. 2 and 3: the two one-sided tangents of τ at q_c, slopes h_h
        # (left) and h_l (right), and on D(h) the envelope's chord of slope q_c across (h_l, h_h),
        # the open interval on which the spectrum lies below D**.
        muted = pg.mkPen(current().ink_muted, style=QtCore.Qt.DashLine)
        self._tangent_left = pg.PlotDataItem(pen=muted, connect="finite")
        self._tangent_right = pg.PlotDataItem(pen=muted, connect="finite")
        self._kink_label = pg.TextItem(anchor=(0, 1), color=current().ink_muted)
        for item in (self._tangent_left, self._tangent_right, self._kink_label):
            item.setVisible(False)
            self.method_plots["tau"].addItem(item, ignoreBounds=True)
        self._chord = pg.PlotDataItem(pen=pg.mkPen(current().ink_muted, style=QtCore.Qt.DashLine, width=2.4),
                                      symbol="o", symbolSize=7, symbolPen=None, symbolBrush=current().ink_muted,
                                      connect="finite")
        self._chord.setVisible(False)
        self._plot.addItem(self._chord, ignoreBounds=True)

        muted = pg.mkPen(current().ink_muted)
        self._d_line = pg.InfiniteLine(
            angle=0, pos=2.0, movable=False,
            pen=pg.mkPen(current().ink_muted, style=QtCore.Qt.DashLine))
        self._plot.addItem(self._d_line)
        self._canon_err = pg.ErrorBarItem(pen=muted, beam=0.01)
        self._plot.addItem(self._canon_err)
        # Items carry no name of their own: the legend lists what is drawn (_sync_legend).
        self._canon_curve = pg.PlotDataItem(
            pen=None, symbol="o", symbolSize=6, symbolPen=None,
            symbolBrush=current().ink)
        self._plot.addItem(self._canon_curve)
        self._hull_curve = pg.PlotDataItem(
            pen=pg.mkPen(Q_WARM, style=QtCore.Qt.DashLine, width=1.6), connect="finite")
        self._plot.addItem(self._hull_curve)
        self._local_curve = pg.PlotDataItem(
            pen=pg.mkPen(Q_WARM, style=QtCore.Qt.DotLine, width=1.6), symbol="o", symbolSize=3,
            symbolPen=None, symbolBrush=Q_WARM, connect="finite")
        self._plot.addItem(self._local_curve)
        self._hist_err = pg.ErrorBarItem(pen=pg.mkPen(Q_COLD), beam=0.0)
        self._plot.addItem(self._hist_err)
        # The histogram's bins as points, the way the paper draws them.
        self._hist_curve = pg.PlotDataItem(pen=None, symbol="o", symbolSize=4, symbolPen=None,
                                           symbolBrush=Q_COLD)
        self._plot.addItem(self._hist_curve)
        self._scale_curves: list = []
        self._gh_curve = pg.PlotDataItem(pen=None, symbol="d", symbolSize=5, symbolPen=None,
                                         symbolBrush=current().amber)
        self._plot.addItem(self._gh_curve)
        self._box_err = pg.ErrorBarItem(pen=pg.mkPen(Q_WARM), beam=0.0)
        self._plot.addItem(self._box_err)
        self._box_curve = pg.PlotDataItem(pen=None, symbol="s", symbolSize=4, symbolPen=None,
                                          symbolBrush=Q_WARM)
        self._plot.addItem(self._box_curve)
        self._dom_curve = pg.PlotDataItem(pen=pg.mkPen(Q_WARM, width=1.2), connect="finite")
        self._non_curve = pg.PlotDataItem(pen=pg.mkPen(Q_COLD, width=1.2), connect="finite")
        self._plot.addItem(self._dom_curve)
        self._plot.addItem(self._non_curve)
        # The Gaussian ensemble's points carrying (q, γ), and the kink bands.
        self._gaussian_points = pg.ScatterPlotItem(size=5, pen=None, hoverable=True)
        self._gaussian_points.sigClicked.connect(self._on_gaussian_point)
        self._gaussian_points.sigHovered.connect(self._on_gaussian_point)
        self._plot.addItem(self._gaussian_points)
        self._gaussian_name = ""
        self._kink_bands: list = []
        self._legend_entries = [
            (self._canon_curve, lambda n=self._source: n),           # no self: no cycle
            (self._hull_curve, lambda: "D**(h): concave envelope of fitted τ(q)"),
            (self._local_curve, lambda: "Legendre of fitted τ(q), local cubic (xsmurf legendre2)"),
            (self._hist_curve, weak_method(self._hist_curve_name)),
            (self._gh_curve, lambda: "gradient histogram (GH)"),
            (self._box_curve, lambda: "box-counting across r (eq. 20)"),
            (self._gaussian_points, weak_method(self._gaussian_curve_name)),
            (self._dom_curve, lambda: "dominant chains"),
            (self._non_curve, lambda: "non-dominant chains")]
        self._update_hist_items()
        self._peak_text = pg.TextItem(color=current().ink, anchor=(0.5, 1.2))
        self._plot.addItem(self._peak_text)

        from dynamix.shell.window_fit import fit_to_screen, remember_geometry

        fit_to_screen(self, 950, 700)
        remember_geometry(self, "spectrum")

    # -- the histogram construction ---------------------------------------------------------
    def _r0_rel(self) -> "float | None":
        """The resolution eq. 21 divides by: the paper's 1/sqrt(N), or the finest scale over
        the field's extent."""
        from dynamix.core.microcanonical import relative_scale

        if self._r0_choice == "r_min/L" and self._scales is not None and self._scales.size:
            r = relative_scale(float(self._scales[0]), self._h_map.shape)
            return r if 0.0 < r < 1.0 else None
        return None

    def _compute_hist(self) -> None:
        from dynamix.core.microcanonical import dh_histogram_errors

        h, D, dD, _counts = dh_histogram_errors(self._h_map, r0_rel=self._r0_rel(),
                                                n_bins=self._bins)
        self._hist = (h, D) if h.size else None
        self._hist_dD = dD if h.size else None

    def _pin_shift(self) -> float:
        """Eq. 21 pinned to the fit's D₀ = -tau(0) instead of d: a constant shift of the
        histogram curve; 0 when the pin is off or the fit has no q = 0."""
        if self.pin_check is None or not self.pin_check.isChecked() or self._fit is None:
            return 0.0
        q = np.asarray(self._fit["q_list"], dtype=np.float64)
        at0 = np.flatnonzero(q == 0.0)
        if at0.size == 0:
            return 0.0
        tau0 = float(np.asarray(self._fit["tau"])[at0[0]])
        return (-tau0 - 2.0) if np.isfinite(tau0) else 0.0

    def _update_hist_items(self) -> None:
        if not hasattr(self, "_hist_curve"):
            return
        if self._hist is None:
            self._hist_curve.setData([], [])
            self._hist_err.setData(x=np.zeros(0), y=np.zeros(0), height=np.zeros(0))
            return
        h, D = self._hist
        shift = self._pin_shift()
        self._hist_curve.setData(h, D + shift)
        height = np.where(np.isfinite(self._hist_dD), 2.0 * self._hist_dD, 0.0)
        self._hist_err.setData(x=h, y=np.where(np.isfinite(D), D + shift, 0.0), height=height)
        self._sync_legend()

    def _hist_curve_name(self) -> str:
        shift = self._pin_shift()
        return self._hist_label + (f" — pinned to D₀ = {2.0 + shift:.2f}" if shift else "")

    @staticmethod
    def _drawn(item) -> bool:
        if isinstance(item, pg.ScatterPlotItem):
            return item.isVisible() and len(item.data) > 0
        x = getattr(item, "xData", None)
        return item.isVisible() and x is not None and len(x) > 0

    def _sync_legend(self) -> None:
        """The legend lists exactly the items that are drawn, under their current names, then
        every spectrum drawn from the D(h) comparison by its display name, its full name and its
        bars' kind the tooltip, then one "board bars: ±1σ" row while any of them draws bars (the
        construction's own histogram keeps its eq. 39 bars)."""
        if not hasattr(self, "_legend"):
            return
        entries = list(self._legend_entries) + [
            (curve, lambda r=r_rel: f"histogram at r = {r:.3g} L")
            for curve, (r_rel, _h, _D) in zip(self._scale_curves, self._dh_curves)]
        wanted = [(item, name_of()) for item, name_of in entries if self._drawn(item)]
        board = [(curve, name) for curve, name, _tip in self._board_legend if self._drawn(curve)]
        tips = {curve: tip for curve, _name, tip in self._board_legend}
        wanted += board
        if board and self._board_bars:
            wanted.append((self._board_bars_note, "board bars: ±1σ"))
            tips[self._board_bars_note] = ("Every error bar drawn from the D(h) comparison spans ±1σ; "
                                           "each entry's tooltip names its kind. The construction's own "
                                           "histogram keeps its eq. 39 bars.")
        if wanted != self._in_legend.get("order"):
            self._legend.clear()
            for item, name in wanted:
                self._legend.addItem(item, name)
                if item in tips:
                    self._legend.items[-1][1].setToolTip(tips[item])
            self._in_legend = {"order": wanted}

    def _on_board_drawn(self) -> None:
        self._sync_legend()
        self._update_kink_geometry()

    def _on_boxcount_toggled(self, checked: bool) -> None:
        if not checked:
            self._box_curve.setData([], [])
            self._box_err.setData(x=np.zeros(0), y=np.zeros(0), height=np.zeros(0))
            self._sync_legend()
            return
        h, D, dD = self._boxcount()
        self._box_curve.setData(h, D)
        self._box_err.setData(x=h, y=np.where(np.isfinite(D), D, 0.0),
                              height=np.where(np.isfinite(dD), 2.0 * dD, 0.0))
        self._sync_legend()

    def _on_hist_controls_changed(self) -> None:
        self._bins = int(self.bins_spin.value())
        self._r0_choice = self.r0_combo.currentText()
        self._compute_hist()
        self._update_hist_items()
        self.send_hist_button.setEnabled(self._hist is not None)
        if self.gh_check is not None and self.gh_check.isChecked():
            self._on_gh_toggled(True)
        if self.boxcount_check is not None and self.boxcount_check.isChecked():
            self._on_boxcount_toggled(True)

    def _on_per_scale_toggled(self, checked: bool) -> None:
        """The construction at every resolution of the stack (Turiel 2008 test ii): on a
        multifractal the curves coincide."""
        if checked and not self._scale_curves:
            n = max(len(self._dh_curves), 1)
            for i, (r_rel, h, D) in enumerate(self._dh_curves):
                colour = pg.mkColor(Q_COLD)
                colour.setAlpha(70 + int(150 * i / n))
                curve = pg.PlotDataItem(h, D, pen=pg.mkPen(colour, width=1.0), connect="finite")
                self._plot.addItem(curve)
                self._scale_curves.append(curve)
        for curve in self._scale_curves:
            curve.setVisible(bool(checked) and self._hist_shown())
        self._sync_legend()

    def _on_points_toggled(self, checked: bool) -> None:
        """The construction's own canonical points and their bars on D(h), shown or stepped aside."""
        for item in (self._canon_curve, self._canon_err):
            item.setVisible(bool(checked))
        self._sync_legend()

    def _hist_shown(self) -> bool:
        return self.hist_check is None or self.hist_check.isChecked()

    def _on_hist_toggled(self, checked: bool) -> None:
        """The histogram construction on D(h): its points, bars, peak label and per-scale curves."""
        for item in (self._hist_curve, self._hist_err, self._peak_text):
            item.setVisible(bool(checked))
        per_scale = self.per_scale_check is not None and self.per_scale_check.isChecked()
        for curve in self._scale_curves:
            curve.setVisible(bool(checked) and per_scale)
        self._sync_legend()

    def _on_gh_toggled(self, checked: bool) -> None:
        """Turiel 2006's gradient histogram, the crude fast construction from the field."""
        if not checked or self._gh_field is None:
            self._gh_curve.setData([], [])
            self._sync_legend()
            return
        from dynamix.core.microcanonical import gradient_histogram_spectrum

        h, D, _dD = gradient_histogram_spectrum(self._gh_field, r0_rel=self._r0_rel(),
                                                n_bins=self._bins)
        self._gh_curve.setData(h, D)
        self._sync_legend()

    # -- recompute --------------------------------------------------------------------------
    def _recompute(self) -> None:
        hd = self._current_hd()
        if hd is not None:
            lo, hi = float(self._lo_spin.value()), float(self._hi_spin.value())
            mode = self.fit_mode()
            self._fit = spectra.fit_spectra(
                hd, lo, hi, eta=self._eta_spin.value(), frame=self._frame_combo.currentText(),
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
            q_own = np.asarray(self._fit["q_list"], dtype=np.float64)
            self._own_items["tau"].setData(q_own, np.asarray(self._fit["tau"], dtype=np.float64))
            self._own_items["h"].setData(q_own, np.asarray(self._fit["h"], dtype=np.float64))
            self._own_items["cq"].setData(q_own, spectra.specific_heat(q_own, self._fit["tau"]))
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
            if self._local_check.isChecked():
                local = spectra.legendre_local(self._fit["q_list"], self._fit["tau"])
                self._local_curve.setData(local["h"], local["D"])
            else:
                self._local_curve.setData([], [])
        for name in self._fit_items:                            # every hull follows the one switch
            self._apply_shown(name, sync=False)
        self._update_tangent()
        self._update_kink_geometry()
        self._update_grouping()
        self._update_hist_items()
        self._update_ensemble()
        self._update_kinks()
        self._sync_legend()

    # -- beyond the hull: the Gaussian ensemble, the kinks ------------------------------------
    def _scale_window(self) -> tuple:
        return float(self._lo_spin.value()), float(self._hi_spin.value())

    def _length_px(self, scales) -> float:
        """The integral scale the per-scale exponents are measured against: the field's extent,
        else four times the largest scale (the Gaussian ensemble's own fallback)."""
        return 2.0 ** self._log2_L if self._log2_L is not None else 4.0 * float(np.max(scales))

    def _gaussian_curve_name(self) -> str:
        return self._gaussian_name

    def _gammas(self) -> np.ndarray:
        """γ max and the steps below it by halving: 8 and 5 steps give ½, 1, 2, 4, 8."""
        top, n = float(self.gamma_max_spin.value()), int(self.gamma_steps_spin.value())
        return top / 2.0 ** np.arange(n - 1, -1, -1)

    def _update_ensemble(self) -> None:
        """The Gaussian ensemble's D(h) points over the fit window from the chosen per-scale
        set (spec 15), the q grid widened with γ so the saddles reach inside a dip; each point
        carries its (q, γ) for the readout, and its brush darkens with γ."""
        on = (self.gaussian_check is not None and self.gaussian_check.isChecked()
              and self._fit is not None)
        self._gaussian_points.setVisible(on)
        if not on:
            self._gaussian_points.setData([], [])
            self._gamma_readout.setText("")
            return
        from dynamix.core.nonconcave import gaussian_ensemble

        name = self.ld_source_combo.currentText()
        entry = self._per_scale[name]
        scales = np.asarray(entry["scales"], dtype=np.float64)
        gammas = self._gammas()
        q = np.asarray(self._fit["q_list"], dtype=np.float64)
        grid = np.linspace(q.min() - 2.0 * gammas.max(), q.max() + 2.0 * gammas.max(), 81)
        lo, hi = self._scale_window()
        try:
            g = gaussian_ensemble(entry["mods"], scales, grid, gammas, lo, hi,
                                  length_px=self._length_px(scales))
        except ValueError as exc:
            self._gaussian_points.setData([], [])
            self._gamma_readout.setText(str(exc))
            return
        keep = np.isfinite(g["h"]) & np.isfinite(g["D"]) & (g["D"] >= 0.0)     # a dimension
        qi, gi = np.nonzero(keep)
        spots = []
        for i, j in zip(qi, gi):
            colour = pg.mkColor(*Q_WARM)
            colour.setAlpha(int(90 + 160 * (j + 1) / gammas.size))
            spots.append({"pos": (float(g["h"][i, j]), float(g["D"][i, j])),
                          "brush": pg.mkBrush(colour), "data": (float(grid[i]), float(gammas[j]))})
        self._gaussian_points.setData(spots)
        self._gaussian_name = f"Ensemble candidates ({name}, γ ≤ {gammas.max():g})"
        self._gamma_readout.setText(f"{len(spots)} points · click or hover one for its γ")

    def _on_gaussian_point(self, _item, points, *_rest) -> None:
        """The per-point readout: the (q, γ) a clicked or hovered ensemble point came from."""
        if len(points) == 0:
            return
        pt = points[0]
        q, gamma = pt.data()
        pos = pt.pos()
        self._gamma_readout.setText(f"γ = {gamma:g}, q = {q:.2f}: h = {pos.x():.3f}, D = {pos.y():.3f}")

    def _update_kinks(self) -> None:
        """The kink check on τ(q) (spec 15): the h interval the Legendre spectrum bridges at
        each kink shaded on D(h); cleared when off or when τ is smooth."""
        for band in self._kink_bands:
            self._plot.removeItem(band)
        self._kink_bands = []
        if self.kink_check is None or not self.kink_check.isChecked() or self._fit is None:
            return
        from dynamix.core.nonconcave import kink_flags

        warm = pg.mkColor(*Q_WARM)
        fill = pg.mkColor(*Q_WARM)
        fill.setAlpha(40)
        for k in kink_flags(self._fit["q_list"], self._fit["tau"]):
            band = pg.LinearRegionItem(values=(k["h_lo"], k["h_hi"]), orientation="vertical",
                                       brush=pg.mkBrush(fill), pen=pg.mkPen(warm), movable=False)
            band.setToolTip(f"kink at q* = {k['q']:.2f}: h jumps {k['h_lo']:.2f} → {k['h_hi']:.2f}")
            self._plot.addItem(band)
            self._kink_bands.append(band)

    def _update_kink_geometry(self) -> None:
        """The two one-sided tangents at q_c on τ(q), the envelope's chord across (h_l, h_h) on
        D(h) when D** is drawn, and the reading, with where a published generalized spectrum (on
        the board, its converged estimate) leaves the envelope by the other route."""
        fit = getattr(self, "_fit", None)
        wanted = fit is not None and self._geometry_check.isChecked()
        got = spectra.kink_interval(fit["q_list"], fit["tau"]) if wanted else None
        for item in (self._tangent_left, self._tangent_right, self._kink_label, self._chord):
            item.setVisible(got is not None)
        if got is None:
            return
        q = np.asarray(fit["q_list"], dtype=np.float64)
        span = np.array([np.nanmin(q), np.nanmax(q)])
        self._tangent_left.setData(span, got["tau_c"] + got["h_h"] * (span - got["q_c"]))
        self._tangent_right.setData(span, got["tau_c"] + got["h_l"] * (span - got["q_c"]))
        text = f"q_c = {got['q_c']:g}: h_h = {got['h_h']:.3f} (left slope), h_l = {got['h_l']:.3f} (right slope)"
        hull = spectra.legendre_dh(q, fit["tau"])
        if hull["h"].size >= 2 and self._hull_check.isChecked():
            ends = np.array([got["h_l"], got["h_h"]])
            self._chord.setData(ends, np.interp(ends, hull["h"], hull["D"]))
        else:
            self._chord.setVisible(False)
        generalized = [(f["h"], f["D"]) for f in self._fits.values()
                       if f.get("kind") == "generalized spectrum" and f.get("h") is not None and f.get("D") is not None]
        # The converged estimate alone: the Legendre (γ = 0) reference sent beside it is the
        # concave spectrum of the same τ table, the envelope it is read against.
        generalized += [(e.h, e.D) for e in self._board_drawn if e.kind == "generalized"
                        and e.identity.get("settings", {}).get("curve") != "legendre"]
        if generalized:
            left = spectra.envelope_departure(hull["h"], hull["D"], *generalized[0])
            if left is not None:
                text += f" · the generalized spectrum leaves D** on ({left[0]:.2f}, {left[1]:.2f})"
        self._kink_label.setText(text)
        self._kink_label.setPos(got["q_c"], got["tau_c"])

    def _phase_tangent(self) -> "dict | None":
        """The tangent at the phase transition off the current fit while the tangent is on."""
        fit = getattr(self, "_fit", None)
        if fit is None or not self.tangent_check.isChecked():
            return None
        return spectra.phase_transition_tangent(fit["q_list"], fit["tau"], fit["h"], fit["D"])

    def _tangent_now(self) -> "dict | None":
        """The tangent drawn on D(h) now, in the board's keys; None when none is drawn."""
        got = self._phase_tangent()
        if got is None:
            return None
        return {"q_c": got["q_c"], "h_c": got["h_c"], "D_c": got["D_c"], "slope": got["q_c"],
                "intercept": got["intercept"]}

    def _update_tangent(self) -> None:
        """The tangent at the phase transition off the current fit, drawn across the view;
        hidden while the tangent is off."""
        items = (self._tangent_line, self._tangent_marker, self._tangent_label)
        got = self._phase_tangent()
        for item in items:
            item.setVisible(got is not None)
        if got is None:
            return
        x = np.asarray(self._plot.getViewBox().viewRange()[0], dtype=np.float64)
        self._tangent_line.setData(x, got["intercept"] + got["q_c"] * x)
        self._tangent_marker.setData([0.0], [got["intercept"]])
        self._tangent_label.setHtml(_tangent_html(got['intercept'], got['q_c']))
        self._tangent_label.setToolTip("the tangent to D(h) at q_c, the convexity-loss order; its intercept at h = 0 is −τ(q_c)")
        self._tangent_label.setPos(0.0, got["intercept"])

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
            self.broadcast_grouping()
        if not self._gspectra_check.isChecked():
            self._dom_curve.setData([], [])
            self._non_curve.setData([], [])
            return
        hd = self._current_hd()
        q_list = np.asarray(hd["q_list"], dtype=np.float64)
        lo, hi = float(self._lo_spin.value()), float(self._hi_spin.value())
        sup = self._table_combo.currentText() == "sup (cmax)"
        for curve, mask in ((self._dom_curve, dom), (self._non_curve, ~dom)):
            if not mask.any():
                curve.setData([], [])
                continue
            tables = chain_groups.subset_hd(self._chains, self._scales, q_list, mask,
                                            min_chain_len=self._min_chain_len)
            table = tables[1] if sup else tables[0]
            fit = spectra.fit_spectra(table, lo, hi, eta=self._eta_spin.value(),
                                      frame=self._frame_combo.currentText())
            h = np.asarray(fit["h"])
            D = np.asarray(fit["D"])
            finite = np.isfinite(h) & np.isfinite(D)
            curve.setData(h[finite], D[finite])
