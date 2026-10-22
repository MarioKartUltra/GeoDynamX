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
shares a legend row under D(h) with the c(q) panel; D(h) carries the tangent at the phase
transition (the Pont-2006 negative-branch convexity-loss abscissa q_c: slope q_c through the
fitted point, drawn to the y axis, where it reads -tau(q_c)). Bottom: the cell-49 phase-transition surface -- tau(q) vs ln q with the
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
"Partition functions…" click, the previous one closed first, parented to the main window.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core import spectra
from dynamix.shell.theme import current
from dynamix.shell.plot_zoom import WheelAxisViewBox  # noqa: F401  (re-exported for the fitting windows)
from dynamix.shell.shared_range import SharedRangeFollower
from dynamix.shell.window_fit import TOOL_WINDOW, show_while_inactive

__all__ = ["MultifractalWindow"]

#: q-gradation endpoints for the partition-family curves, most negative q -> most positive q.
#: DATA identity (the curve's moment order), not chrome -- the Theme-Rule carve-out canvas.py's
#: palette constants already document. Cool-to-warm, echoing the skeleton dialog's coolwarm ramp.
Q_COLD = (80, 130, 235)
Q_WARM = (235, 95, 60)

#: The two partition-table conventions, in combo order. no-sup (``hd_std``, each line's modulus at
#: the current scale, xsmurf's own default) is the default; sup (``hd_cmax``, the running supremum
#: over finer scales along the chain) is the option -- it flattens any part of the spectrum with
#: h < 0 toward 0, so it is wrong where negative exponents are expected.
#: Devices whose h-map is the leaders' local exponents: their eq-21 histogram is Cardo &
#: Corvalán 2009's GMWLP (Turiel's single-resolution density formula on extrapolated exponents),
#: a different target from the large-deviation routes across scales, and named as such.
LEADERS_METHODS = ("wavelet_leaders", "cwt_leaders")

TABLE_CHOICES = ("no-sup (std)", "sup (cmax)")
#: What a source's tables are, by ``hd_kind``: partition functions over maxima lines (WTMM, box
#: counting), or over the field's cells (moments over the projections, wavelet leaders: the sum,
#: so τ(q) = ζ(q) − d and D(q = 0) = d, the structure functions' mean being the Structure
#: functions window's). Both are Z.
VOCABULARY = {"partition": ("Partition functions", "log₂ Z(q,a)"),
              "structure": ("Partition functions", "log₂ Z(q,a)")}


def vocabulary(kind) -> tuple:
    """``(window title, label of the first family)`` for ``kind`` (partition by default)."""
    return VOCABULARY.get(str(kind or "partition"), VOCABULARY["partition"])


def add_zero_lines(plot) -> tuple:
    """Thin lines at x = 0 and y = 0 on a D(h) plot, behind the data."""
    pen = pg.mkPen(current().ink_muted, width=0.8)
    lines = (pg.InfiniteLine(angle=90, pos=0.0, movable=False, pen=pen),
             pg.InfiniteLine(angle=0, pos=0.0, movable=False, pen=pen))
    for line in lines:
        line.setZValue(-10)
        plot.addItem(line)
    return lines


#: The grid has twelve columns: the three families span four each, the four spectra three each.
_SPECTRUM_CELLS = {"tau": (1, 0), "D": (1, 3), "h": (1, 6), "cq": (1, 9)}
#: D(h)'s opening view: h from −2 to 2, D from −½ to 1½, the same on every dataset until zoomed.
DH_RANGE = ((-2.0, 2.0), (-0.5, 1.5))
POWER_MEAN_LABEL = "(log₂Z − log₂N_a)/q"
#: One hue per secondary source (another method's tables beside the primary's).
_SOURCE_HUES = ((242, 169, 59), (76, 201, 240), (241, 91, 181), (158, 240, 26), (255, 107, 107), (179, 136, 255))


def _q_color(i: int, n: int) -> tuple:
    """Linear cold->warm ramp over the q index -- plain lerp, no colormap dependency."""
    t = i / (n - 1) if n > 1 else 0.5
    return tuple(int(round(c + t * (w - c))) for c, w in zip(Q_COLD, Q_WARM))


#: The three partition-function families, plotted in this order (the books' presentation).
FAMILY_TABLES = (("tau_qa", "log₂ Z(q,a)"), ("h_qa", "H(q,a)"), ("D_qa", "D(q,a)"))


class PhysicalAxis(pg.AxisItem):
    """A top axis whose ticks sit at the panel's log₂ a positions and read the physical
    length λ* there (the wavelet scale bar's conversion); a scale without a band reads blank."""

    def __init__(self, physical, orientation: str = "top", **kw) -> None:
        super().__init__(orientation, **kw)
        self._physical = physical

    def tickValues(self, minVal, maxVal, size):
        """One label per octave, at integer log₂ a, whatever the panel's width: the three
        family panels then tick alike instead of each by its own measured text widths."""
        lo, hi = int(np.ceil(min(minVal, maxVal))), int(np.floor(max(minVal, maxVal)))
        return [(1.0, [float(k) for k in range(lo, hi + 1)])]

    def tickStrings(self, values, scale, spacing):
        from dynamix.core.wavelet_profiles import format_length

        out = []
        for v in values:
            got = self._physical(float(v))
            out.append(format_length(*got) if got is not None else "")
        return out


class PhysicalLog2Axis(pg.AxisItem):
    """A bottom axis whose ticks sit at the panel's log₂ a positions and read log₂ of the
    physical length λ* there -- the partition functions against the length the bars quote."""

    def __init__(self, physical, orientation: str = "bottom", **kw) -> None:
        super().__init__(orientation, **kw)
        self._physical = physical

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            got = self._physical(float(v))
            out.append(f"{np.log2(got[0]):.2f}" if got is not None and got[0] > 0 else "")
        return out


class MultifractalWindow(SharedRangeFollower, QtWidgets.QDialog):
    """Interactive canonical (Arneodo WTMM) spectrum fitter over ``hd_std``/``hd_cmax``."""

    #: Emitted on every re-fit with the current (log2_a_min, log2_a_max) -- the coupled-state
    #: channel the spectrum-construction window inherits its window from (the workbook's own
    #: cell-49-inherits-cell-47 pattern).
    scaleWindowChanged = QtCore.Signal(float, float)

    #: The fit mode chosen here (``naive`` / ``fixed focus`` / ``focus``), for the construction
    #: window to follow: one fit mode, chosen where the fit happens.
    fitModeChanged = QtCore.Signal(str)

    #: The current fit as the construction window compares it with the other methods':
    #: ``name``, ``kind``, ``q``, ``tau``, ``h``, ``D`` with their σ's, ``hull``, ``cq``.
    fitChanged = QtCore.Signal(object)

    def __init__(self, hd_std: dict, hd_cmax: dict, *, eta_seed: float = 0.0,
                 forward_note: str = "", parent=None, physical_of_log2a=None,
                 log2_L: "float | None" = None, all_tables=None, filter_note: str = "",
                 extrema_mods=None, scales=None, dh_hist=None, dh_box=None,
                 source: str = "canonical (Arneodo WTMM)", sources=None, h_map=None,
                 kind: str = "partition", tables: "dict | None" = None, h_map_method: str = ""):
        #: What the tables are (``VOCABULARY``): names the window and its first family.
        self._kind = str(kind or "partition")
        #: The h-map, for the box-counting construction on request, and the device that made
        #: it: a leaders device's h-map makes the eq-21 histogram Cardo & Corvalán's GMWLP.
        self._h_map = np.asarray(h_map) if h_map is not None else None
        self._h_map_method = str(h_map_method or "")
        self._dh_legend_entries: list = []
        self._in_dh_legend: dict = {}
        #: The box-counting construction ``(h, D, dD)`` of an h-map result (eq. 20 across r).
        self._dh_box = tuple(dh_box) if dh_box is not None else None
        #: Other methods' results drawn beside the primary's, each a dict with ``name`` and
        #: ``hd_std``/``hd_cmax`` and/or ``dh_hist``; the Show menu toggles each.
        self._sources = [dict(s) for s in (sources or [])]
        self._source_items: dict = {}
        self._primary_visible = True
        #: The histogram construction ``(h, D, dD)`` of an h-map result, drawn beside the
        #: canonical points; None without one.
        self._dh_hist = tuple(dh_hist) if dh_hist is not None else None
        self._source = str(source)
        #: The maxima's moduli per scale and the scales (px) they sit at, for the cascade's
        #: cumulants under the fit window; None leaves the c(q) panel to the fit's own curve.
        self._extrema_mods = ([np.asarray(m, dtype=np.float64) for m in extrema_mods]
                              if extrema_mods is not None else None)
        self._scales_px = np.asarray(scales, dtype=np.float64) if scales is not None else None
        self._cumulants: dict | None = None
        #: log2 of the field extent, the focus anchor the focus fit modes need; None leaves
        #: them unavailable.
        self._log2_L = float(log2_L) if log2_L is not None else None
        #: ``(hd_std, hd_cmax)`` over EVERY chain of the transform when the tables given are
        #: over the rack's filtered chains, with the note that says so; None when they are the same.
        self._all_tables = tuple(all_tables) if all_tables is not None else None
        self._filter_note = str(filter_note)
        #: ``log2 a -> (length, unit) | None``: λ* of the analysing wavelet at that scale in
        #: the field's physical unit (the wavelet scale bar's own conversion), for a top axis
        #: on the three family panels and the fit window's readout; None leaves them out.
        self._physical = physical_of_log2a
        # Qt.Window, not the QDialog default -- a real, independently resizable top-level window,
        # the same choice SkeletonDialog/ProfileDialog made and for the same reason.
        super().__init__(parent, TOOL_WINDOW)
        show_while_inactive(self)
        self.setModal(False)
        self.setWindowTitle(f"{vocabulary(self._kind)[0]} — {self._source}")

        #: The Table choice's entries: the WTMM's two modes, or the named tables a result carries
        #: (the wavelet leaders' methods), the first entry the default.
        self._hd_by_choice = (dict(tables) if tables
                              else {"no-sup (std)": hd_std, "sup (cmax)": hd_cmax})   # TABLE_CHOICES order
        #: One table under two names (the moment route), or one named table: the choice means
        #: nothing and is hidden.
        self._one_table = len(self._hd_by_choice) == 1 or (not tables and hd_std is hd_cmax)
        self._filtered_tables = (hd_std, hd_cmax)
        if self._filter_note:
            self.setWindowTitle(self.windowTitle() + f" — {self._filter_note}")
        self._fit: dict | None = None
        #: Per family table, its PlotDataItems; ``_family_items`` stays the tau list (the
        #: original single-panel attribute, kept as the visibility contract's face).
        self._family_items_by_table: dict[str, list] = {k: [] for k, _ in FAMILY_TABLES}
        #: Per family table, the best-fit line of each q row (``partition_fit_lines``).
        self._family_lines_by_table: dict[str, list] = {k: [] for k, _ in FAMILY_TABLES}
        self._fit_lines: dict | None = None
        self._region_syncing = False
        self._cum_syncing = False

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

    def current_cumulants(self) -> dict | None:
        """The cascade's cumulants (:func:`~dynamix.core.spectra.cascade_cumulants`) under the
        current fit window, in the fit's frame; None without maxima moduli."""
        return self._cumulants

    def set_scale_window(self, log2_a_min: float, log2_a_max: float) -> None:
        """Programmatic region set -- same path as a drag (fires the live re-fit)."""
        self._region.setRegion((log2_a_min, log2_a_max))

    _apply_scale_window = set_scale_window

    # -- construction -----------------------------------------------------------------------
    def _current_hd(self) -> dict:
        return self._hd_by_choice[self._table_combo.currentText()]

    def _build_ui(self, eta_seed: float, forward_note: str) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QHBoxLayout()
        self._table_label = QtWidgets.QLabel("Table")
        controls.addWidget(self._table_label)
        self._table_combo = QtWidgets.QComboBox()
        self._table_combo.addItems(list(self._hd_by_choice))
        self._table_combo.currentTextChanged.connect(self._on_table_changed)
        controls.addWidget(self._table_combo)
        for w in (self._table_label, self._table_combo):
            w.setVisible(not self._one_table)

        controls.addWidget(QtWidgets.QLabel("η"))
        self._eta_spin = QtWidgets.QDoubleSpinBox()
        self._eta_spin.setRange(-100.0, 100.0)
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

        # The fit mode (core/spectra.focus_regression): plain per-q slopes, or every q fitted
        # through one focus at the field extent. Chosen here, where the fit happens; the
        # construction window follows (fitModeChanged).
        controls.addWidget(QtWidgets.QLabel("Fit"))
        self._fitmode_combo = QtWidgets.QComboBox()
        self._fitmode_combo.addItems(["naive", "fixed focus", "focus"])
        if self._log2_L is None:
            self._fitmode_combo.setEnabled(False)
            self._fitmode_combo.setToolTip(
                "focus fitting needs the field extent (log2 L) -- open from a computed result")
        self._fitmode_combo.currentTextChanged.connect(self._on_fit_mode_changed)
        controls.addWidget(self._fitmode_combo)
        self._focus_readout = QtWidgets.QLabel("")          # shown on the readouts line below

        # The families' abscissa: log₂ a in pixels, or log₂ of the physical λ* in the bars' unit.
        controls.addWidget(QtWidgets.QLabel("x"))
        self._xaxis_combo = QtWidgets.QComboBox()
        self._xaxis_combo.addItems(["log₂ a (px)", "log₂ λ* (unit)"])
        self._xaxis_combo.setEnabled(self._physical is not None)
        self._xaxis_combo.currentTextChanged.connect(self._on_xaxis_changed)
        controls.addWidget(self._xaxis_combo)

        # Show: which method's spectra and partition functions are drawn, the primary included.
        self.show_button = QtWidgets.QToolButton()
        self.show_button.setText("Show")
        self.show_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.show_menu = QtWidgets.QMenu(self.show_button)
        for name in [self._source] + [s["name"] for s in self._sources]:
            action = self.show_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(True)
            action.toggled.connect(lambda checked, n=name: self._on_source_toggled(n, checked))
        self.show_button.setMenu(self.show_menu)
        controls.addWidget(self.show_button)
        # The strip wraps onto a second row here: one row of everything had a minimum width
        # wider than a laptop screen, and a window can never be narrower than its layout.
        controls.addStretch(1)
        layout.addLayout(controls)
        controls = QtWidgets.QHBoxLayout()
        # Eq. 21 pinned to this fit's D₀ = -tau(0) instead of d, to compare shapes.
        self.pin_check = QtWidgets.QCheckBox("pin peak to D₀")
        self.pin_check.setEnabled(self._dh_hist is not None)
        self.pin_check.toggled.connect(lambda _c: self._update_dh_hist())
        controls.addWidget(self.pin_check)
        self.boxcount_check = QtWidgets.QCheckBox("box-counting")
        self.boxcount_check.setEnabled(self._h_map is not None)
        self.boxcount_check.toggled.connect(self._on_boxcount_toggled)
        controls.addWidget(self.boxcount_check)
        # tau, h and D(h) from the cascade's cumulants, over the fitted ones.
        self.cumulant_check = QtWidgets.QCheckBox("cumulant spectrum")
        self.cumulant_check.toggled.connect(self._on_cumulant_requested)
        controls.addWidget(self.cumulant_check)
        # The cumulants' own scale range: a second window on the log2 a panels, in the
        # cumulant curve's colour; off, they read the fit window.
        self.cumulant_range_check = QtWidgets.QCheckBox("cumulant range")
        self.cumulant_range_check.setEnabled(
            (self._extrema_mods is not None and self._scales_px is not None)
            or self._current_hd().get("lnT_moments") is not None)
        self.cumulant_range_check.toggled.connect(self._on_cumulant_range_toggled)
        # Nonconcave spectra (core/nonconcave.py), off by default: the kink check on tau(q) and
        # the Gaussian ensemble's D(h) points from the maxima moduli.
        self.kink_check = QtWidgets.QCheckBox("kinks")
        self.kink_check.toggled.connect(lambda _c: self._update_kinks())
        self.gaussian_check = QtWidgets.QCheckBox("Ensemble candidates γ")
        self.gaussian_check.setToolTip("Weighted-mean candidates can mix populations. Use Generalized spectrum for the generalized Legendre inversion.")
        self.gaussian_check.setEnabled(self._extrema_mods is not None and self._scales_px is not None)
        self.gaussian_check.toggled.connect(lambda _c: self._update_gaussian())
        self._kink_label = QtWidgets.QLabel("")
        self._kink_label.setStyleSheet(f"color: {current().ink_muted};")
        self._kink_lines: list = []
        self._kink_bands: list = []

        # The rack's filters narrowed the chains: the tables are over the survivors; this
        # switch shows the whole transform's instead.
        self.all_chains_check = None
        if self._all_tables is not None:
            self.all_chains_check = QtWidgets.QCheckBox("all chains")
            self.all_chains_check.setToolTip(self._filter_note)
            self.all_chains_check.toggled.connect(self._on_all_chains_toggled)
            controls.addWidget(self.all_chains_check)

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
        # The Bershadskii specific heat (tau against ln q, the phase-transition row): shown
        # on request, off by default.
        self.bershadskii_check = QtWidgets.QCheckBox("Bershadskii c(q)")
        self.bershadskii_check.toggled.connect(self._set_bershadskii_visible)
        controls.addWidget(self.bershadskii_check)
        self._qstar_label = QtWidgets.QLabel("q*")
        controls.addWidget(self._qstar_label)
        self._qstar_spin = QtWidgets.QDoubleSpinBox()
        self._qstar_spin.setRange(0.02, max(float(q.max()), 0.02))
        self._qstar_spin.setSingleStep(0.1)
        self._qstar_spin.setDecimals(2)
        self._qstar_spin.setValue(min(1.0, max(0.02, float(q.max()))))
        self._qstar_spin.valueChanged.connect(lambda _v: self._update_phase_transition())
        controls.addWidget(self._qstar_spin)

        controls.addStretch(1)
        layout.addLayout(controls)

        #: The two-segment fit's live numbers -- slope_L / slope_R / ds + the scan's best q*.
        self._pt_label = QtWidgets.QLabel("")
        self._pt_label.setStyleSheet(f"color: {current().ink_muted};")
        #: The cascade's cumulants c₁..c₄ under the fit window (the c(q) panel's readout).
        self._cq_label = QtWidgets.QLabel("")
        self._cq_label.setStyleSheet(f"color: {current().ink_muted};")
        #: What the forward transform actually multiplied in -- display only, derived by the
        #: caller (see the module docstring's eta/frame section). Muted: context, not a control.
        self._forward_label = QtWidgets.QLabel(forward_note)
        self._forward_label.setStyleSheet(f"color: {current().ink_muted};")
        # The readouts' texts change with every fit; on their own line, wrapping, with an
        # ignored width policy, they can never raise the window's minimum width.
        readouts = QtWidgets.QHBoxLayout()
        for label in (self._focus_readout, self._pt_label, self._cq_label, self._forward_label):
            label.setWordWrap(True)
            label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                QtWidgets.QSizePolicy.Policy.Preferred)
            if label is self._cq_label:
                readouts.addWidget(self.cumulant_range_check)      # beside what it refits
            readouts.addWidget(label, 1)
        self._kink_label.setWordWrap(True)
        self._kink_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                       QtWidgets.QSizePolicy.Policy.Preferred)
        readouts.addWidget(self.kink_check)
        readouts.addWidget(self.gaussian_check)
        readouts.addWidget(self._make_follow_check())
        readouts.addWidget(self._kink_label, 1)
        layout.addLayout(readouts)

        # The spectra are squared on the event loop's next turn (_square_spectra), by a timer the
        # window owns, so a window destroyed first takes the pending call with it.
        self._square_timer = QtCore.QTimer(self)
        self._square_timer.setSingleShot(True)
        self._square_timer.setInterval(0)
        self._square_timer.timeout.connect(self._square_spectra)
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(current().ground)
        layout.addWidget(self._glw, 1)
        #: The fit window in log₂ a and, through ``physical_of_log2a``, in physical length.
        self.range_label = QtWidgets.QLabel("")
        self.range_label.setProperty("reading", "true")
        layout.addWidget(self.range_label)

        # Row 0: the THREE partition-function families (Z, H, D vs log2 a -- the books' top
        # row), each carrying a mirrored copy of the fit window so click-and-drag works on any
        # panel, exactly like the workbook's click-to-bracket on any of its three panels.
        band = pg.mkColor(current().amber)
        band.setAlpha(36)
        log2_s = np.asarray(self._current_hd()["log2_scales"], dtype=np.float64)
        span = (float(log2_s.min()), float(log2_s.max()))
        self._family_plots: dict[str, pg.PlotItem] = {}
        self._regions: list[pg.LinearRegionItem] = []
        self._cum_regions: list[pg.LinearRegionItem] = []
        warm = pg.mkColor(*Q_WARM)
        warm.setAlpha(28)
        for col, (key, ylabel) in enumerate(FAMILY_TABLES):
            plot = self._glw.addPlot(row=0, col=4 * col, colspan=4, viewBox=WheelAxisViewBox())
            plot.setLabel("bottom", self._scale_axis_label())
            plot.setLabel("left", vocabulary(self._kind)[1] if key == "tau_qa" else ylabel)
            if self._physical is not None:
                plot.setAxisItems({"top": PhysicalAxis(self._physical)})
                plot.showAxis("top")
            region = pg.LinearRegionItem(values=span, orientation="vertical",
                                         brush=pg.mkBrush(band),
                                         pen=pg.mkPen(current().amber))
            region.setBounds(span)
            region.sigRegionChanged.connect(self._on_region_changed)
            plot.addItem(region)
            cum = pg.LinearRegionItem(values=span, orientation="vertical", brush=pg.mkBrush(warm),
                                      pen=pg.mkPen(Q_WARM, style=QtCore.Qt.DashLine))
            cum.setBounds(span)
            cum.setVisible(False)
            cum.sigRegionChanged.connect(self._on_cumulant_region_changed)
            plot.addItem(cum)
            self._cum_regions.append(cum)
            self._family_plots[key] = plot
            self._regions.append(region)
        # The tau panel/region remain the canonical names the public API and older callers use.
        self._partition_plot = self._family_plots["tau_qa"]
        self._region = self._regions[0]
        # A focus mode's focus on each family panel, and the field extent L ruled.
        dash = pg.mkPen(current().ink_muted, style=QtCore.Qt.DashLine)
        self._focus_markers: dict[str, pg.ScatterPlotItem] = {}
        self._L_lines: dict[str, pg.InfiniteLine] = {}
        for key, plot in self._family_plots.items():
            marker = pg.ScatterPlotItem(symbol="o", size=11,
                                        pen=pg.mkPen(current().amber, width=1.5),
                                        brush=pg.mkBrush(None))
            marker.setVisible(False)
            plot.addItem(marker)
            self._focus_markers[key] = marker
            line = pg.InfiniteLine(angle=90, movable=False, pen=dash, label="L",
                                   labelOpts={"position": 0.92})
            line.setVisible(False)
            plot.addItem(line)
            self._L_lines[key] = line

        ink = pg.mkPen(current().ink, width=1.4)
        muted = pg.mkPen(current().ink_muted)
        self._spectrum_plots = {}
        self._spectrum_curves = {}
        self._spectrum_errbars = {}
        for col, (name, xlabel, ylabel) in enumerate((("tau", "q", "τ(q)"),
                                                      ("D", "h", "D(h)"),
                                                      ("h", "q", "h(q)"))):
            row, col0 = _SPECTRUM_CELLS[name]
            plot = self._glw.addPlot(row=row, col=col0, colspan=3, viewBox=WheelAxisViewBox())
            plot.setLabel("bottom", xlabel)
            plot.setLabel("left", ylabel)
            self._spectrum_plots[name] = plot
            errbar = pg.ErrorBarItem(pen=muted, beam=0.0)      # whiskers alone, no caps
            plot.addItem(errbar)
            self._spectrum_errbars[name] = errbar
            curve = pg.PlotDataItem(pen=ink, connect="finite", symbol="o", symbolSize=4,
                                    symbolPen=None, symbolBrush=current().ink)
            plot.addItem(curve)
            self._spectrum_curves[name] = curve
        # h is read as a plain decimal: pyqtgraph's automatic SI prefix would otherwise label
        # these axes "(x0.001)" whenever the exponents are small.
        self._spectrum_plots["D"].getAxis("bottom").enableAutoSIPrefix(False)
        self._spectrum_plots["h"].getAxis("left").enableAutoSIPrefix(False)
        self._spectrum_plots["D"].setXRange(*DH_RANGE[0], padding=0.0)
        self._spectrum_plots["D"].setYRange(*DH_RANGE[1], padding=0.0)
        add_zero_lines(self._spectrum_plots["D"])

        # The tangent at the phase transition on D(h): through the fitted point at the Pont-2006
        # negative-branch convexity-loss abscissa q_c with slope q_c, drawn from the y axis past
        # the point, the intercept marked and read (spectra.phase_transition_tangent). An
        # indicator, never an automatic truncation.
        muted_dash = pg.mkPen(current().ink_muted, style=QtCore.Qt.DashLine)
        self._tangent_line = pg.PlotDataItem(pen=muted_dash, connect="finite")
        self._tangent_marker = pg.ScatterPlotItem(size=7, pen=None,
                                                  brush=current().ink_muted)
        self._tangent_label = pg.TextItem(anchor=(0, 1), color=current().ink_muted)
        for item in (self._tangent_line, self._tangent_marker, self._tangent_label):
            item.setVisible(False)
            self._spectrum_plots["D"].addItem(item, ignoreBounds=True)   # spans the view: no auto-range
        self._spectrum_plots["D"].getViewBox().sigXRangeChanged.connect(
            lambda *_a: self._update_phase_tangent())

        # Row 2: the cell-49 phase-transition surface. Left: fitted tau(q) against ln q with the
        # two-segment fit; right: the count-weighted R² scan over candidate q*.
        muted_dash = pg.mkPen(current().ink_muted, style=QtCore.Qt.DashLine)
        self._pt_plot = pg.PlotItem(viewBox=WheelAxisViewBox())
        self._pt_plot.setLabel("bottom", "ln q  (positive branch)")
        self._pt_plot.setLabel("left", "τ(q)")
        self._pt_points = pg.PlotDataItem(pen=None, symbol="o", symbolSize=5,
                                          symbolPen=None,
                                          symbolBrush=current().ink)
        self._pt_plot.addItem(self._pt_points)
        self._pt_fit_left = pg.PlotDataItem(pen=pg.mkPen(Q_COLD, width=1.6))
        self._pt_fit_right = pg.PlotDataItem(pen=pg.mkPen(Q_WARM, width=1.6))
        self._pt_plot.addItem(self._pt_fit_left)
        self._pt_plot.addItem(self._pt_fit_right)
        self._pt_qstar_line = pg.InfiniteLine(angle=90, movable=False,
                                              pen=pg.mkPen(current().amber))
        self._pt_plot.addItem(self._pt_qstar_line)

        self._pt_scan_plot = pg.PlotItem(viewBox=WheelAxisViewBox())
        self._pt_scan_plot.setLabel("bottom", "q*")
        self._pt_scan_plot.setLabel("left", "weighted R²")
        self._pt_scan_curve = pg.PlotDataItem(pen=pg.mkPen(current().ink),
                                              connect="finite")
        self._pt_scan_plot.addItem(self._pt_scan_curve)
        self._pt_best_line = pg.InfiniteLine(angle=90, movable=False, pen=muted_dash,
                                             label="best", labelOpts={"position": 0.9})
        self._pt_scan_plot.addItem(self._pt_best_line)
        self._pt_scan_qstar_line = pg.InfiniteLine(angle=90, movable=False,
                                                   pen=pg.mkPen(current().amber))
        self._pt_scan_plot.addItem(self._pt_scan_qstar_line)

        # An h-map result's histogram construction (eq. 21, the eq. 39 bars) beside the points,
        # as points; the D(h) panel carries the legend of what is drawn (_sync_dh_legend).
        d_plot = self._spectrum_plots["D"]
        # The legend is a row of (sample, short name) pairs under the panel's bottom axis, in
        # the PlotItem's own grid (row 4 sits below the axis row), never over the data.
        self._dh_legend_row = pg.GraphicsLayout()
        self._dh_legend_row.setContentsMargins(0, 0, 0, 0)
        self._glw.ci.addItem(self._dh_legend_row, row=2, col=0, colspan=12)
        self._dh_hist_err = pg.ErrorBarItem(pen=pg.mkPen(Q_COLD), beam=0.0)
        d_plot.addItem(self._dh_hist_err)
        self._dh_hist_curve = pg.PlotDataItem(pen=None, symbol="o", symbolSize=4, symbolPen=None,
                                              symbolBrush=Q_COLD)
        d_plot.addItem(self._dh_hist_curve)
        self._dh_box_err = pg.ErrorBarItem(pen=pg.mkPen(Q_WARM), beam=0.0)
        d_plot.addItem(self._dh_box_err)
        self._dh_box_curve = pg.PlotDataItem(pen=None, symbol="s", symbolSize=4, symbolPen=None,
                                             symbolBrush=Q_WARM)
        d_plot.addItem(self._dh_box_curve)
        amber_dash = pg.mkPen(current().amber, width=1.4, style=QtCore.Qt.DashLine)
        self._cum_curves = {name: pg.PlotDataItem(pen=amber_dash, connect="finite")
                            for name in ("tau", "D", "h")}
        for name, curve in self._cum_curves.items():
            self._spectrum_plots[name].addItem(curve)
        hist_name = ("GMWLP histogram (leaders)" if self._h_map_method in LEADERS_METHODS
                     else "microcanonical histogram")
        self._dh_legend_entries = [
            (self._spectrum_curves["D"], lambda n=self._source: n),   # no self: no cycle
            (self._dh_hist_curve, lambda n=hist_name: n),
            (self._dh_box_curve, lambda: "box-counting across r"),
            (self._cum_curves["D"], lambda: "cumulant spectrum")]
        if self._dh_box is not None:
            h, D, dD = (np.asarray(a, dtype=np.float64) for a in self._dh_box)
            self._dh_box_curve.setData(h, D)
            self._dh_box_err.setData(x=h, y=np.where(np.isfinite(D), D, 0.0),
                                     height=np.where(np.isfinite(dD), 2.0 * dD, 0.0))
        self._update_dh_hist()

        # Row 3: the specific heat c(q) = -q² τ'' off the fit (points), the cumulant expansion's
        # curve from the maxima moduli, and the log-normal reference c₂ q² (the workbook's
        # specific-heat figure).
        self._cq_plot = self._glw.addPlot(*_SPECTRUM_CELLS["cq"], colspan=3, viewBox=WheelAxisViewBox())
        self._cq_plot.setLabel("bottom", "q")
        self._cq_plot.setLabel("left", "c(q) = −q² τ″")
        # Points the fit, the solid line the cumulant expansion, dotted the log-normal −c₂q²:
        # three entries of the legend row under D(h), beside the spectra it shares that row with.
        self._cq_curve = pg.PlotDataItem(pen=None, symbol="o", symbolSize=5, symbolPen=None,
                                         symbolBrush=current().ink)
        self._cq_plot.addItem(self._cq_curve)
        self._cq_cum_curve = pg.PlotDataItem(pen=pg.mkPen(Q_WARM, width=1.6))
        self._cq_plot.addItem(self._cq_cum_curve)
        self._cq_ln_curve = pg.PlotDataItem(pen=pg.mkPen(current().ink_muted,
                                                         style=QtCore.Qt.DotLine))
        self._cq_plot.addItem(self._cq_ln_curve)
        # The q of the D(h) tangent, picked by dragging this line on c(q) (snapped to the fitted
        # q grid); until it is moved it sits at the automatic convexity-loss q_c.
        self._tangent_q: "float | None" = None
        self._placing_pick = False
        self._q_pick_line = pg.InfiniteLine(
            pos=0.0, angle=90, movable=True, pen=pg.mkPen(Q_WARM, width=1.5),
            hoverPen=pg.mkPen(Q_WARM, width=3), label="tangent q",
            labelOpts={"position": 0.9, "color": Q_WARM})
        self._q_pick_line.sigPositionChanged.connect(self._on_q_pick_moved)
        self._q_pick_line.sigPositionChangeFinished.connect(self._on_q_pick_finished)
        self._cq_plot.addItem(self._q_pick_line)
        self._dh_legend_entries += [(self._cq_curve, lambda: "c(q) fit"),
                                    (self._cq_cum_curve, lambda: "cumulant expansion"),
                                    (self._cq_ln_curve, lambda: "log-normal −c₂q²")]

        # The families take half the height; the 2 x 2 grid the rest (the Bershadskii row,
        # when shown, a fifth).
        self._set_bershadskii_visible(False)

        self._build_source_items()

        # Click-to-set q* on either panel: ln-q abscissa on the left panel, plain q on the scan
        # (the workbook's exact gesture pair, cell 49).
        self._glw.scene().sigMouseClicked.connect(self._on_scene_clicked)

        from dynamix.shell.window_fit import fit_to_screen, remember_geometry

        fit_to_screen(self, 1150, 1000)
        remember_geometry(self, "multifractal")

    # -- redraw -----------------------------------------------------------------------------
    def _q_mask(self, q: np.ndarray) -> np.ndarray:
        return (q >= self._qmin_spin.value()) & (q <= self._qmax_spin.value())

    def _rebuild_family(self) -> None:
        """One PlotDataItem per q row of each of the CURRENT table's three families -- rebuilt
        only on a table switch; a q-subrange change just toggles visibility."""
        hd = self._current_hd()
        log2_s = np.asarray(hd["log2_scales"], dtype=np.float64)
        for key, label in FAMILY_TABLES:
            plot = self._family_plots[key]
            for item in self._family_items_by_table[key] + self._family_lines_by_table[key]:
                plot.removeItem(item)
            self._family_items_by_table[key] = []
            self._family_lines_by_table[key] = []
            table = self._family_table(key)
            if key == "tau_qa":
                label = POWER_MEAN_LABEL if self._power_mean_view() else vocabulary(self._kind)[1]
            plot.setLabel("left", label)
            n_q = table.shape[0]
            for i in range(n_q):
                # The data as points, the fit the only line (the EBSD figures' look).
                item = pg.PlotDataItem(log2_s, table[i], pen=None, symbol="o", symbolSize=4,
                                       symbolPen=None, symbolBrush=_q_color(i, n_q),
                                       connect="finite")
                plot.addItem(item)
                self._family_items_by_table[key].append(item)
                line = pg.PlotDataItem(pen=pg.mkPen(_q_color(i, n_q), width=1.2),
                                       connect="finite")
                plot.addItem(line)
                self._family_lines_by_table[key].append(line)
        self._rebuild_source_rows()
        self._update_family_visibility()

    # -- the other methods' sources -------------------------------------------------------
    def _source_table(self, source: dict) -> "dict | None":
        """The source's table under the primary's Table choice (one table when it has one)."""
        cmax = self._table_combo.currentText() == "sup (cmax)"
        return source.get("hd_cmax" if cmax else "hd_std") or source.get("hd_std") \
            or source.get("hd_cmax")

    def _build_source_items(self) -> None:
        """One set of items per source: its rows on the family panels (filled by
        ``_rebuild_source_rows``), its fitted tau/D/h and c(q) curves, its histogram on D(h)."""
        for i, source in enumerate(self._sources):
            hue = _SOURCE_HUES[i % len(_SOURCE_HUES)]
            pen = pg.mkPen(hue, width=1.4)
            items = {"hue": hue, "visible": True, "rows": {k: [] for k, _ in FAMILY_TABLES},
                     "curves": {}, "fit": None}
            if self._source_table(source) is not None:
                for name in ("tau", "D", "h"):
                    curve = pg.PlotDataItem(pen=pen, connect="finite", symbol="o", symbolSize=3,
                                            symbolPen=None, symbolBrush=hue)
                    self._spectrum_plots[name].addItem(curve)
                    items["curves"][name] = curve
                cq = pg.PlotDataItem(pen=pen, connect="finite")
                self._cq_plot.addItem(cq)
                items["curves"]["cq"] = cq
                self._dh_legend_entries.append((items["curves"]["D"],
                                                lambda n=source["name"]: n))
            if source.get("dh_hist") is not None:
                h, D, _dD = (np.asarray(a, dtype=np.float64) for a in source["dh_hist"])
                hist = pg.PlotDataItem(h, D, pen=None, symbol="o", symbolSize=3, symbolPen=None,
                                       symbolBrush=hue)
                self._spectrum_plots["D"].addItem(hist)
                items["curves"]["dh"] = hist
                self._dh_legend_entries.append((hist, lambda n=source["name"]: f"{n} (histogram)"))
            self._source_items[source["name"]] = items
        self._sync_dh_legend()

    def _rebuild_source_rows(self) -> None:
        for source in self._sources:
            items = self._source_items.get(source["name"])
            hd = self._source_table(source)
            if items is None or hd is None:
                continue
            colour = pg.mkColor(items["hue"])
            colour.setAlpha(150)
            log2_s = np.asarray(hd["log2_scales"], dtype=np.float64)
            for key, _label in FAMILY_TABLES:
                plot = self._family_plots[key]
                for item in items["rows"][key]:
                    plot.removeItem(item)
                table = self._family_table_of(hd, key)
                items["rows"][key] = []
                for row in table:
                    item = pg.PlotDataItem(log2_s, row, pen=pg.mkPen(colour, width=0.9),
                                           connect="finite")
                    plot.addItem(item)
                    items["rows"][key].append(item)

    def _refit_sources(self) -> None:
        """Every source's fit under the primary's window, mode, eta and frame."""
        lo, hi = self._region.getRegion()
        mode = self.fit_mode()
        for source in self._sources:
            items = self._source_items[source["name"]]
            hd = self._source_table(source)
            if hd is None or "tau" not in items["curves"]:
                continue
            fit = spectra.fit_spectra(hd, float(lo), float(hi), eta=self._eta_spin.value(),
                                      frame=self._frame_combo.currentText(), fit_mode=mode,
                                      log2_L=self._log2_L if mode != "naive" else None)
            items["fit"] = fit
            q = np.asarray(fit["q_list"], dtype=np.float64)
            mask = self._q_mask(q)
            items["curves"]["tau"].setData(q[mask], np.asarray(fit["tau"])[mask])
            items["curves"]["D"].setData(np.asarray(fit["h"])[mask], np.asarray(fit["D"])[mask])
            items["curves"]["h"].setData(q[mask], np.asarray(fit["h"])[mask])
            items["curves"]["cq"].setData(q[mask], spectra.specific_heat(q, fit["tau"])[mask])

    @staticmethod
    def _drawn(item) -> bool:
        x = getattr(item, "xData", None)
        return item.isVisible() and x is not None and len(x) > 0

    @staticmethod
    def _short_name(name: str) -> str:
        """A source's legend name: what follows the last " · " (the layer, not the method
        sentence); the full name is the entry's tooltip."""
        return str(name).rsplit(" · ", 1)[-1].strip()

    def _dh_legend_names(self) -> list:
        """The short names in the legend row, in order."""
        return [self._short_name(name) for _item, name in self._in_dh_legend.get("order", [])]

    def _sync_dh_legend(self) -> None:
        """The legend row under D(h) lists exactly what is drawn on D(h) and c(q): the primary,
        the histogram and box-counting constructions, the cumulant spectrum, every source, the
        c(q) fit, its cumulant expansion and the log-normal reference. Four entries per line."""
        if not hasattr(self, "_dh_legend_row"):
            return
        wanted = [(item, name_of()) for item, name_of in self._dh_legend_entries
                  if self._drawn(item)]
        if wanted == self._in_dh_legend.get("order"):
            return
        from pyqtgraph.graphicsItems.LegendItem import ItemSample

        row = self._dh_legend_row
        row.clear()
        for i, (item, name) in enumerate(wanted):
            line, col = divmod(i, 4)
            sample = ItemSample(item)
            sample.setMinimumSize(20, 20)
            sample.setMaximumSize(20, 20)
            label = pg.LabelItem(self._short_name(name), size="9pt",
                                 color=current().ink_muted, justify="left")
            label.setToolTip(str(name))
            row.addItem(sample, line, 2 * col)
            row.addItem(label, line, 2 * col + 1)
        # Entries pack to the left: a stretch column after them takes the slack.
        n_cols = 2 * min(len(wanted), 4)
        for col in range(n_cols):
            row.layout.setColumnStretchFactor(col, 0)
        row.addItem(pg.LabelItem(""), 0, n_cols)
        row.layout.setColumnStretchFactor(n_cols, 1)
        self._in_dh_legend = {"order": wanted}

    def _on_boxcount_toggled(self, checked: bool) -> None:
        if not checked or self._h_map is None:
            self._dh_box_curve.setData([], [])
            self._dh_box_err.setData(x=np.zeros(0), y=np.zeros(0), height=np.zeros(0))
        else:
            from dynamix.core.microcanonical import box_counting_spectrum

            h, D, dD = box_counting_spectrum(self._h_map)
            self._dh_box_curve.setData(h, D)
            self._dh_box_err.setData(x=h, y=np.where(np.isfinite(D), D, 0.0),
                                     height=np.where(np.isfinite(dD), 2.0 * dD, 0.0))
        self._sync_dh_legend()

    def _update_cumulant_overlay(self) -> None:
        """tau(q), h(q) and D(h) from the cascade's cumulants (the workbook's reconstruction,
        c₀ = -tau(0) of the fit), drawn over the fitted ones while the box is on."""
        on = self.cumulant_check.isChecked() and self._cumulants is not None \
            and self._fit is not None
        if on:
            q = np.asarray(self._fit["q_list"], dtype=np.float64)
            at0 = np.flatnonzero(q == 0.0)
            on = at0.size > 0 and np.isfinite(np.asarray(self._fit["tau"])[at0[0]])
        if not on:
            for curve in self._cum_curves.values():
                curve.setData([], [])
            self._sync_dh_legend()
            return
        c0 = -float(np.asarray(self._fit["tau"])[at0[0]])
        qq = q[self._q_mask(q) & np.isfinite(q)]
        tau_c = spectra.cumulant_tau(self._cumulants, qq, c0=c0)
        h_c = np.gradient(tau_c, qq, edge_order=2) if qq.size >= 3 else np.full(qq.size, np.nan)
        self._cum_curves["tau"].setData(qq, tau_c)
        self._cum_curves["h"].setData(qq, h_c)
        self._cum_curves["D"].setData(h_c, qq * h_c - tau_c)
        self._sync_dh_legend()

    def _apply_source_visibility(self, name: str) -> None:
        items = self._source_items[name]
        shown = bool(items["visible"])
        for curve in items["curves"].values():
            curve.setVisible(shown)
        self._sync_dh_legend()
        source = next(s for s in self._sources if s["name"] == name)
        hd = self._source_table(source)
        if hd is None:
            return
        mask = self._q_mask(np.asarray(hd["q_list"], dtype=np.float64))
        for rows in items["rows"].values():
            for item, visible in zip(rows, mask):
                item.setVisible(shown and bool(visible)
                                and bool(np.isfinite(item.yData).any()))

    def _set_primary_visible(self) -> None:
        shown = self._primary_visible
        for curve in list(self._spectrum_curves.values()) + list(self._spectrum_errbars.values()):
            curve.setVisible(shown)
        for item in (self._cq_curve, self._cq_cum_curve, self._cq_ln_curve,
                     self._dh_hist_curve, self._dh_hist_err, self._dh_box_curve,
                     self._dh_box_err, *self._cum_curves.values()):
            item.setVisible(shown)
        self._update_family_visibility()
        self._sync_dh_legend()

    def _on_source_toggled(self, name: str, checked: bool) -> None:
        if name == self._source:
            self._primary_visible = bool(checked)
            self._set_primary_visible()
        elif name in self._source_items:
            self._source_items[name]["visible"] = bool(checked)
            self._apply_source_visibility(name)

    def _power_mean_view(self) -> bool:
        """A focus mode shows the Z family as power means, the space its lines meet in."""
        return self.fit_mode() != "naive"

    def _family_table(self, key: str) -> np.ndarray:
        return self._family_table_of(self._current_hd(), key)

    def _family_table_of(self, hd: dict, key: str) -> np.ndarray:
        """The rows a family panel draws for ``hd``: the table itself, or for the Z family in a
        focus mode its power means (log2 Z - log2 N_a)/q (no row at q = 0)."""
        table = np.asarray(hd[key], dtype=np.float64)
        if key != "tau_qa" or not self._power_mean_view():
            return table
        q = np.asarray(hd["q_list"], dtype=np.float64)
        N_a = np.asarray(hd["N_a"], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            log2N = np.where(N_a > 0, np.log2(np.where(N_a > 0, N_a, 1.0)), np.nan)
            out = (table - log2N[None, :]) / q[:, None]
        out[q == 0.0] = np.nan
        return out

    def _update_fit_lines(self) -> None:
        """The best-fit line over every family row: the per-q line over the window (naive,
        clipped to the data's own span as the figures do), or in a focus mode the family's
        lines drawn out to its focus, the focus marked and the field extent L ruled."""
        hd = self._current_hd()
        lo, hi = self._region.getRegion()
        mode = self.fit_mode()
        lines = spectra.partition_fit_lines(
            hd, float(lo), float(hi), fit_mode=mode,
            log2_L=self._log2_L if mode != "naive" else None)
        self._fit_lines = lines
        log2_s = np.asarray(hd["log2_scales"], dtype=np.float64)
        q = np.asarray(hd["q_list"], dtype=np.float64)
        for key, _label in FAMILY_TABLES:
            focus = lines["family_focus"][key]
            if key == "tau_qa" and self._power_mean_view():
                slope, intercept = np.full(q.size, np.nan), np.full(q.size, np.nan)
                nz = q != 0.0
                slope[nz], intercept[nz] = lines["power_mean"]["slope"], lines["power_mean"]["intercept"]
            else:
                slope, intercept = lines["lines"][key]["slope"], lines["lines"][key]["intercept"]
            x_end = focus[0] if focus is not None else float(log2_s.max())
            xs = np.linspace(float(log2_s.min()), x_end, 64)
            table = self._family_table(key)
            clip = focus is None and np.isfinite(table).any()
            if clip:
                y_lo, y_hi = float(np.nanmin(table)), float(np.nanmax(table))
                pad = 0.05 * (y_hi - y_lo)
            for i, item in enumerate(self._family_lines_by_table[key]):
                ys = slope[i] * xs + intercept[i]
                if clip:
                    ys = np.where((ys >= y_lo - pad) & (ys <= y_hi + pad), ys, np.nan)
                item.setData(xs, ys)
            marker = self._focus_markers[key]
            if focus is not None:
                marker.setData([focus[0]], [focus[1]])
            else:
                marker.setData([], [])
            marker.setVisible(focus is not None)
            show_L = mode != "naive" and self._log2_L is not None
            if show_L:
                self._L_lines[key].setPos(float(self._log2_L))
            self._L_lines[key].setVisible(show_L)

    def _set_bershadskii_visible(self, visible: bool) -> None:
        """The Bershadskii row (tau against ln q, the R² scan) joins or leaves the grid; the
        families keep half the height either way."""
        ci = self._glw.ci
        visible = bool(visible)
        if visible and self._pt_plot not in ci.items:
            ci.addItem(self._pt_plot, row=3, col=0, colspan=8)
            ci.addItem(self._pt_scan_plot, row=3, col=8, colspan=4)
        elif not visible and self._pt_plot in ci.items:
            ci.removeItem(self._pt_plot)
            ci.removeItem(self._pt_scan_plot)
        for col in range(12):                  # equal columns: thirds on top, quarters below
            ci.layout.setColumnStretchFactor(col, 1)
        ci.layout.setRowStretchFactor(0, 3)      # the families take what the squares leave
        ci.layout.setRowStretchFactor(1, 0)
        ci.layout.setRowStretchFactor(2, 0)      # the legend strip
        ci.layout.setRowStretchFactor(3, 1 if visible else 0)
        self._square_spectra()
        for widget in (self._pt_label, self._qstar_label, self._qstar_spin):
            widget.setVisible(visible)

    def _square_spectra(self) -> None:
        """The four spectra are squares: the row's height is one plot's width, a quarter of the
        grid's, so the panels keep one shape whatever the window's. Read once the plot widget
        has its new size (the dialog's resize comes first), hence the deferred call."""
        ci = self._glw.ci
        width = self._glw.viewport().width() if self._glw.viewport() is not None else 0
        if width <= 0:
            return
        side = max(120.0, (width - 5.0 * ci.layout.horizontalSpacing()) / 4.0)
        ci.layout.setRowFixedHeight(1, side)
        for col in range(12):
            ci.layout.setColumnFixedWidth(col, side / 3.0)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_glw"):
            self._square_timer.start()

    def _update_family_visibility(self) -> None:
        q = np.asarray(self._current_hd()["q_list"], dtype=np.float64)
        mask = self._q_mask(q)
        for key, items in self._family_items_by_table.items():
            for item, line, visible in zip(items, self._family_lines_by_table[key], mask):
                drawn = self._primary_visible and bool(visible) \
                    and bool(np.isfinite(item.yData).any())
                item.setVisible(drawn)
                line.setVisible(drawn)
        for name in self._source_items:
            self._apply_source_visibility(name)

    def fit_mode(self) -> str:
        """The fit mode as :func:`dynamix.core.spectra.fit_spectra` names it."""
        return {"naive": "naive", "fixed focus": "fixed_focus",
                "focus": "focus"}[self._fitmode_combo.currentText()]

    def _scale_axis_label(self) -> str:
        """``j`` for a dyadic table (a pyramid's levels), ``log₂ a`` for a ladder."""
        return "j" if self._current_hd().get("dyadic") else "log₂ a"

    def _physical_unit(self) -> str:
        """The unit the physical conversion reads at the middle of the scale range."""
        if self._physical is None:
            return ""
        log2_s = np.asarray(self._current_hd()["log2_scales"], dtype=np.float64)
        for v in (float(np.median(log2_s)), float(log2_s.min()), float(log2_s.max())):
            got = self._physical(v)
            if got is not None:
                return str(got[1])
        return ""

    def _on_xaxis_changed(self, text: str) -> None:
        physical = text.startswith("log₂ λ*") and self._physical is not None
        for plot in self._family_plots.values():
            if physical:
                plot.setAxisItems({"bottom": PhysicalLog2Axis(self._physical)})
                plot.setLabel("bottom", f"log₂ λ* ({self._physical_unit()})")
            else:
                plot.setAxisItems({"bottom": pg.AxisItem("bottom")})
                plot.setLabel("bottom", self._scale_axis_label())

    def refresh_physical_axes(self) -> None:
        """The bars' units changed: the physical axes re-read their conversion."""
        if self._physical is None:
            return
        for plot in self._family_plots.values():
            for name in ("top", "bottom"):
                axis = plot.getAxis(name)
                axis.picture = None
                axis.update()
            if isinstance(plot.getAxis("bottom"), PhysicalLog2Axis):
                plot.setLabel("bottom", f"log₂ λ* ({self._physical_unit()})")
        lo, hi = self._region.getRegion()
        self.range_label.setText(self._range_text(float(lo), float(hi)))

    def _on_fit_mode_changed(self, text: str) -> None:
        self._rebuild_family()                 # the Z family's rows change space with the mode
        self._refit()
        self.fitModeChanged.emit(text)

    def _on_all_chains_toggled(self, checked: bool) -> None:
        std, cmax = self._all_tables if checked else self._filtered_tables
        self._hd_by_choice = {"sup (cmax)": cmax, "no-sup (std)": std}
        self._rebuild_family()
        self._refit()

    def _range_text(self, lo: float, hi: float) -> str:
        from dynamix.core.wavelet_profiles import format_length

        text = f"fit window: log₂ a {lo:.2f} – {hi:.2f}"
        if self._physical is not None:
            ends = [self._physical(v) for v in (lo, hi)]
            if all(e is not None for e in ends):
                text += f"  ·  {format_length(*ends[0])} – {format_length(*ends[1])}"
        return text

    def _refit(self) -> None:
        lo, hi = self._region.getRegion()
        self.range_label.setText(self._range_text(float(lo), float(hi)))
        mode = self.fit_mode()
        self._fit = spectra.fit_spectra(
            self._current_hd(), float(lo), float(hi),
            eta=self._eta_spin.value(), frame=self._frame_combo.currentText(),
            fit_mode=mode, log2_L=self._log2_L if mode != "naive" else None)
        focus = self._fit.get("focus")
        if focus is None:
            self._focus_readout.setText("")
        elif focus.get("branch") == "unavailable":
            self._focus_readout.setText(f"focus: unavailable ({focus.get('reason', '')})")
        else:
            text = f"focus: {focus.get('branch', '')} · x̂₀ = {float(focus.get('x0', float('nan'))):.2f}"
            for name in ("h", "D"):
                fam = focus.get(name) or {}
                if fam.get("branch"):
                    text += f" · {name}: {fam['branch']}"
            if focus.get("note"):
                text += f" — {focus['note']}"
            self._focus_readout.setText(text)
        self._update_spectrum_plots()
        self._update_phase_transition()
        self._update_phase_tangent()
        self._update_specific_heat()
        self._update_fit_lines()
        self._refit_sources()
        self._update_dh_hist()
        self._update_cumulant_overlay()
        self._update_kinks()
        self._update_gaussian()
        self._sync_dh_legend()
        self.scaleWindowChanged.emit(float(lo), float(hi))
        self._publish()

    def _publish(self) -> None:
        fit = self.published_fit()
        self._last_published = fit["name"]
        self.fitChanged.emit(fit)

    def published_fit(self) -> dict:
        """The current fit in the schema every method's fitting tool publishes."""
        fit = self._fit
        q = np.asarray(fit["q_list"], dtype=np.float64)
        # One table (the leaders): the source and the tool; two: the source and the table chosen.
        name = (f"{self._source}, partition functions" if self._one_table
                else f"{self._source} · {self._table_combo.currentText()}")
        return {"name": name, "kind": vocabulary(self._kind)[0].lower(),
                "q": q, "tau": fit["tau"], "tau_err": fit["tau_err"], "h": fit["h"], "h_err": fit["h_err"],
                "D": fit["D"], "D_err": fit["D_err"], "hull": spectra.legendre_dh(q, fit["tau"]),
                "cq": spectra.specific_heat(q, fit["tau"]), "frame": fit.get("frame"), "eta": fit.get("eta")}

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

    def _update_dh_hist(self) -> None:
        """The histogram construction on the D(h) panel, shifted to this fit's D₀ = -tau(0)
        when the pin is on (eq. 21 with D₀ in place of d)."""
        if self._dh_hist is None:
            return
        h, D, dD = (np.asarray(a, dtype=np.float64) for a in self._dh_hist)
        shift = 0.0
        if self.pin_check.isChecked() and self._fit is not None:
            q = np.asarray(self._fit["q_list"], dtype=np.float64)
            at0 = np.flatnonzero(q == 0.0)
            if at0.size:
                tau0 = float(np.asarray(self._fit["tau"])[at0[0]])
                shift = (-tau0 - 2.0) if np.isfinite(tau0) else 0.0
        self._dh_hist_curve.setData(h, D + shift)
        self._dh_hist_err.setData(x=h, y=np.where(np.isfinite(D), D + shift, 0.0),
                                  height=np.where(np.isfinite(dD), 2.0 * dD, 0.0))
        self._sync_dh_legend()

    def _update_specific_heat(self) -> None:
        """c(q) off the CURRENT fit's tau(q), and -- with maxima moduli -- the cascade's
        cumulants under the fit window, brought into the fit's frame (the original frame takes
        the forward lift η back out of c₁ exactly as the fit takes η q out of τ)."""
        fit = self._fit
        if fit is None:
            return
        q = np.asarray(fit["q_list"], dtype=np.float64)
        mask = self._q_mask(q)
        heat = spectra.specific_heat(q, fit["tau"])
        self._cq_curve.setData(q[mask], heat[mask])
        if not (self.cumulant_check.isChecked() or self.cumulant_range_check.isChecked()):
            self._cumulants = None
            self._cq_cum_curve.setData([], [])
            self._cq_ln_curve.setData([], [])
            self._cq_label.setText("Cumulant fitting is opt-in; use Log-cumulants for independent ranges.")
            return
        hd = self._current_hd()
        own = self.cumulant_range_check.isChecked()
        lo, hi = (self._cum_regions[0] if own else self._region).getRegion()
        if self._extrema_mods is not None and self._scales_px is not None:
            c = spectra.cascade_cumulants(self._extrema_mods, self._scales_px, float(lo), float(hi))
        elif hd.get("lnT_moments") is not None:
            c = spectra.cumulants_from_moments(hd["lnT_moments"], hd["scales"], float(lo), float(hi))
        else:
            return
        if fit.get("frame") == "original":
            c["c1"] -= float(fit.get("eta", 0.0))
        self._cumulants = c
        shown = q[mask & np.isfinite(q)]
        grid = np.linspace(shown.min(), shown.max(), 200) if shown.size else q
        self._cq_cum_curve.setData(grid, spectra.cumulant_heat(c, grid))
        self._cq_ln_curve.setData(grid, -c["c2"] * grid ** 2)          # the log-normal's c(q) = −c₂ q², c₂ < 0
        self._cq_label.setText(
            f"c₁={c['c1']:.3f}  c₂={c['c2']:.3f}  c₃={c['c3']:.3f}  c₄={c['c4']:.3f}"
            f"  ·  R²min={c['r2']:.2f}"
            + (f"  ·  log₂ a {lo:.2f}–{hi:.2f}" if own else ""))

    def _on_cumulant_requested(self, _checked):
        self._update_specific_heat()
        self._update_cumulant_overlay()

    def _update_kinks(self) -> None:
        """The kink check (spec 15): a dashed line at each q* on tau(q) and the h interval the
        Legendre spectrum bridges there shaded on D(h); cleared when off or when tau is smooth."""
        from dynamix.core.nonconcave import kink_flags

        for item in self._kink_lines:
            self._spectrum_plots["tau"].removeItem(item)
        for item in self._kink_bands:
            self._spectrum_plots["D"].removeItem(item)
        self._kink_lines, self._kink_bands = [], []
        fit = self._fit
        if not self.kink_check.isChecked() or fit is None:
            self._kink_label.setText("")
            return
        flags = kink_flags(fit["q_list"], fit["tau"])
        warm = pg.mkColor(*Q_WARM)
        band_brush = pg.mkColor(*Q_WARM)
        band_brush.setAlpha(40)
        for k in flags:
            line = pg.InfiniteLine(pos=k["q"], angle=90,
                                   pen=pg.mkPen(warm, style=QtCore.Qt.DashLine))
            self._spectrum_plots["tau"].addItem(line)
            band = pg.LinearRegionItem(values=(k["h_lo"], k["h_hi"]), orientation="vertical",
                                       brush=pg.mkBrush(band_brush), pen=pg.mkPen(warm),
                                       movable=False)
            self._spectrum_plots["D"].addItem(band)
            self._kink_lines.append(line)
            self._kink_bands.append(band)
        self._kink_label.setText(
            "  ·  ".join(f"kink at q* = {k['q']:.2f}: h jumps {k['h_lo']:.2f} → {k['h_hi']:.2f}"
                         for k in flags) or "no kink")

    def _update_gaussian(self) -> None:
        """The Gaussian ensemble's D(h) points (spec 15) over the fit window, from the maxima
        moduli, with the integral scale L the field's extent (else four times the largest
        scale); the q grid widens with gamma so the saddles reach inside a dip."""
        if not hasattr(self, "_gaussian_points"):
            self._gaussian_points = pg.ScatterPlotItem(size=4, pen=None,
                                                       brush=pg.mkBrush(*Q_WARM, 160))
            self._spectrum_plots["D"].addItem(self._gaussian_points)
        on = (self.gaussian_check.isChecked() and self._fit is not None
              and self._extrema_mods is not None and self._scales_px is not None)
        self._gaussian_points.setVisible(on)
        if not on:
            return
        from dynamix.core.nonconcave import gaussian_ensemble

        lo, hi = self._region.getRegion()
        gammas = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        q = np.asarray(self._fit["q_list"], dtype=np.float64)
        grid = np.linspace(q.min() - 2.0 * gammas.max(), q.max() + 2.0 * gammas.max(), 81)
        length = 2.0 ** self._log2_L if self._log2_L is not None else None
        g = gaussian_ensemble(self._extrema_mods, self._scales_px, grid, gammas, float(lo),
                              float(hi), length_px=length)
        keep = np.isfinite(g["h"]) & np.isfinite(g["D"]) & (g["D"] >= 0.0)     # a dimension
        self._gaussian_points.setData(g["h"][keep], g["D"][keep])

    def _on_cumulant_range_toggled(self, on: bool) -> None:
        """Show the cumulants' own window (starting on the fit window) or hide it, then refit
        the cumulants."""
        if on:
            self._cum_syncing = True
            try:
                for region in self._cum_regions:
                    region.setRegion(self._region.getRegion())
            finally:
                self._cum_syncing = False
        for region in self._cum_regions:
            region.setVisible(bool(on))
        self._update_specific_heat()
        self._update_cumulant_overlay()

    def _on_cumulant_region_changed(self) -> None:
        """A drag of the cumulants' window on any panel mirrors onto the other two, then the
        cumulants refit; the spectrum fit stays."""
        if self._cum_syncing:
            return
        sender = self.sender()
        lo, hi = (sender if sender in self._cum_regions else self._cum_regions[0]).getRegion()
        self._cum_syncing = True
        try:
            for region in self._cum_regions:
                if region is not sender:
                    region.setRegion((lo, hi))
        finally:
            self._cum_syncing = False
        self._update_specific_heat()
        self._update_cumulant_overlay()

    def _update_phase_tangent(self) -> None:
        """The D(h) tangent at the phase transition, drawn across the whole view (redrawn as the
        view moves, so only the plot's margin cuts it); hidden while the negative branch stays
        concave."""
        fit = getattr(self, "_fit", None)
        if fit is None:
            return
        got = spectra.phase_transition_tangent(fit["q_list"], fit["tau"], fit["h"], fit["D"],
                                               q_pick=self._tangent_q)
        if self._tangent_q is None:                 # the line rests on the automatic q_c
            q = np.asarray(fit["q_list"], dtype=np.float64)
            self._place_q_pick(got["q_c"] if got is not None else float(np.nanmin(q)))
        for item in (self._tangent_line, self._tangent_marker, self._tangent_label):
            item.setVisible(got is not None)
        if got is None:
            return
        x = np.asarray(self._spectrum_plots["D"].getViewBox().viewRange()[0], dtype=np.float64)
        self._tangent_line.setData(x, got["intercept"] + got["q_c"] * x)
        self._tangent_marker.setData([0.0], [got["intercept"]])
        where = (f"q = {got['q_c']:g} (picked)" if got["picked"] else f"q_c = {got['q_c']:g}")
        self._tangent_label.setText(f"{got['intercept']:.3f} at h = 0 · tangent at {where}")
        self._tangent_label.setPos(0.0, got["intercept"])

    def _place_q_pick(self, q: float) -> None:
        self._placing_pick = True
        try:
            self._q_pick_line.setValue(float(q))
        finally:
            self._placing_pick = False

    def _on_q_pick_moved(self) -> None:
        """A drag of the c(q) line: the tangent follows the fitted q nearest to it."""
        if self._placing_pick or self._fit is None:
            return
        q = np.asarray(self._fit["q_list"], dtype=np.float64)
        if not np.isfinite(q).any():
            return
        self._tangent_q = float(q[np.nanargmin(np.abs(q - self._q_pick_line.value()))])
        self._update_phase_tangent()

    def _on_q_pick_finished(self) -> None:
        """The drag ends on the picked q itself (the fitted grid)."""
        if self._tangent_q is not None:
            self._place_q_pick(self._tangent_q)

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
        self._publish_range(float(lo), float(hi))

    def _on_scene_clicked(self, ev) -> None:
        """Click-to-set q* (cell 49's gesture pair): the tau-vs-ln-q panel maps the click's x
        back through exp; the scan panel takes it as q* directly."""
        for plot, is_log in ((self._pt_plot, True), (self._pt_scan_plot, False)):
            if plot.scene() is None:            # the Bershadskii row is off the grid
                continue
            vb = plot.getViewBox()
            if vb.sceneBoundingRect().contains(ev.scenePos()):
                x = float(vb.mapSceneToView(ev.scenePos()).x())
                q_star = float(np.exp(x)) if is_log else x
                lo, hi = self._qstar_spin.minimum(), self._qstar_spin.maximum()
                self._qstar_spin.setValue(min(max(q_star, lo), hi))
                return

    def _on_table_changed(self, _text: str) -> None:
        # One window, one published fit: the one under the previous table is retracted first.
        previous = getattr(self, "_last_published", None)
        if previous:
            self.fitChanged.emit({"name": previous, "valid": False})
        self._rebuild_family()
        self._refit()

    def _on_q_range_changed(self) -> None:
        # Display mask only -- the fit (and with it the phase-transition surface, which always
        # reads the full positive branch) is already over the full q_list (module docstring).
        self._update_family_visibility()
        self._update_spectrum_plots()
