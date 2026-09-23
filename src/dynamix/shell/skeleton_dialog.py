# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""SkeletonDialog: the WTMM skeleton log-log plot -- coefficient vs scale, colored by OLS
Hölder slope, with the "Normalize at finest scale" convergence fan and h-range chain selection
(EQSelect's own ``_WtmmSkeletonDialog``, EQ§4/§5a; the
notebook log-log conventions).

This is the user's most-requested missing display: EQSelect's answer to "no log-log plot of
coeff vs scale" (spec Mission). The shape below follows EQSelect's own Hölder-lines view as
closely as DynamiX's simpler, single-view scope allows -- there is no cascade-tree view here (the
spec's own non-goals list it as deferred); every chain this dialog is given is drawn (up to the
cap) as one leaf-to-root maxima-line polyline, colored by its own OLS slope.

**Figure construction.** One ``matplotlib.figure.Figure(constrained_layout=True)`` -- EQSelect's
own colorbar-ratchet lesson (without ``constrained_layout``, adding/removing the colorbar
shrinks the axes a little more every time it happens) -- holding two axes side by side (main
log-log plot : histogram, width ratio 3:1). The main axes carries ONE ``LineCollection`` for every
drawn chain (``alpha=0.2, linewidths=0.5``, CW§1's own v1-cell-57 numbers), colored by the SAME
per-chain array through ``coolwarm``, clipped to the finite values' 2nd-98th percentile (short chains produce outlier slopes that would otherwise wash out the color scale for everyone
else) and labelled with a colorbar "Hölder slope h". A SECOND, initially empty ``LineCollection``
sits on the same axes for the selection highlight (see "Highlight color" below). The histogram
axes shows the FULL (uncapped) finite-h distribution, ``bins=50, range=(-3, 2)`` (CW§1's own
hard-coded shape) -- independent of the draw cap, since a histogram costs nothing extra past a
few thousand chains and there is no reason to hide part of the real distribution just because not
every line fits on screen.

**Stats are computed exactly ONCE, at construction, cached on ``self._h``**
(``dynamix.core.chain_stats.stats_for(chains, "ols")`` -- the port, the SAME estimator
``ChainHolderFilter``'s live "kept N/M" reading already uses, so a value read off this plot always
agrees with the filter box's own number for the identical chain). :meth:`set_selection` only ever
rebuilds the highlight overlay's GEOMETRY; it never calls :func:`~dynamix.core.chain_stats.
stats_for` again -- a test asserts ``self._h`` keeps its object identity across the call.

**The draw cap is geometry-only.** ``MAX_DRAWN`` (2000, EQ§4's own number) limits how many
polylines are actually built into the main ``LineCollection`` -- simplest honest choice: the
FIRST ``MAX_DRAWN`` chains in ``chains``' own order, no re-sorting by size/persistence (this
dialog has no such secondary statistic to sort by, unlike EQSelect's own tree-based cap). Stats
(``self._h``), the h-range spins' defaults, and "Select h-range"'s own emitted indices all operate
over the FULL ``chains`` list regardless of the cap -- every index this dialog ever reports or
accepts is a real index into that list, the same index space ``Canvas`` picking and
``GroupPalette`` membership already use for these exact chains (``main_window.py``'s own call
site). The title notes the cap ("... [capped from N]") whenever it actually bit.

**Highlight color: black, not the theme's ``selection_accent``.** EQSelect's own Hölder-lines
highlight is a flat black overdraw, ``lw=1.8`` -- used here verbatim rather than importing
``dynamix.shell.theme`` for the app's shared selection accent. This is a deliberate, DOCUMENTED
choice (the design's "pick one, document"): pulling in ``theme.py`` would be this module's
ONLY ``dynamix.shell.*`` dependency, for a single line color, on a module that is otherwise exactly
as self-contained as ``topology_panel.py`` (``devloop.py``'s ``_DEPENDS_ON["skeleton_dialog"] =
()``) -- not worth the coupling for a highlight that already reads unambiguously as "selected" in
black-on-coolwarm. A future pass is free to switch to ``selection_accent`` if the two highlight
styles ever need to look visually related; nothing here would resist that, it just was not judged
worth the added dependency edge today.

**Normalize at finest scale.** Chain-dict index 0 is the finest scale (CW§1's own chain
dict contract). The checkbox shifts every drawn/highlighted polyline so its own index-0 point
sits at the plot origin -- a pure per-chain translation (``x -= x[0]; y -= y[0]``), so the SLOPE a
line displays is untouched; only where it starts moves. Axis labels switch to the Δ forms
(EQ§5a's own wording, log2 instead of EQSelect's log10 -- the project's scale doctrine keeps this
whole codebase in log2). Toggling never touches ``self._h`` or the colorbar's clim -- both are a
function of slope alone, and slope is exactly what a translation preserves.

**px_size (CW§1's physical-units offset).** ``px_size`` is frame units per pixel, or ``None`` for
a bare-pixel frame -- ``main_window.py`` derives this HONESTLY off the active field's own frame
(see that module's own ``_skeleton_px_size`` docstring) and hands in a plain float or ``None``;
this dialog never looks at a ``RasterField``/``LocalFrame`` itself. When given, every chain's x is
shifted by the CONSTANT ``log2(px_size)`` (``x = log2_scales + np.log2(px_um)``) -- a single
per-plot constant, so it changes nothing about "Normalize at finest scale" (the constant cancels
out of ``x - x[0]`` exactly like any other per-chain-invariant shift) and nothing about the
h-slope estimate (an x shift, not a scale). There is no unit STRING to label the axis with (the
constructor only ever receives a float), so the axis label stays the plain "log₂ scale" either
way -- see the module's own axis-label helper.

**Lazy matplotlib**, per ``right_panel.py``'s own precedent (its module docstring, ":70-73"):
imported only inside the methods that build a figure, never at module scope, and
``matplotlib.use(...)`` is never called anywhere -- importing ``FigureCanvasQTAgg`` IS the backend
selection; the harness's mandated ``QT_QPA_PLATFORM=offscreen`` needs nothing else (confirmed:
this module's own test suite builds a real ``FigureCanvasQTAgg`` under that exact platform).

**Lifecycle** is owned entirely by ``main_window.py``, not here: a fresh ``SkeletonDialog`` is
built on every "Skeleton plot…" click, from whatever chains are on screen AT THAT MOMENT -- no
caching of an old instance across opens (see that module's own ``_on_skeleton_button_clicked``
docstring for the full reasoning). This dialog itself has no opinion on that; it is simply
constructed, used, and eventually closed like any other ``QDialog``.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtWidgets

from dynamix.core import chain_stats

__all__ = ["SkeletonDialog", "MAX_DRAWN"]

#: EQ§4's own cap ("Cap 2000 drawn chains, noted in the title") -- geometry only, see module
#: docstring's "The draw cap is geometry-only" section.
MAX_DRAWN = 2000

#: CW§1's own histogram shape (v1-cell-57: ``bins=50, range=(-3, 2)``).
_HIST_BINS = 50
_HIST_RANGE = (-3.0, 2.0)

#: EQSelect's own Hölder-lines highlight -- see the module docstring's "Highlight color"
#: section for why this stays a plain matplotlib color name rather than the theme accent.
_HIGHLIGHT_COLOR = "black"
_HIGHLIGHT_LW = 1.8


def _chain_xy(chain: dict, px_log2: float, normalize: bool) -> tuple[np.ndarray, np.ndarray]:
    """One chain's (x, y) log-log points: ``x = log2_scales (+ px_log2)``, ``y = log2_mod``,
    truncated to the two arrays' common length (defensive -- real chains always pair these 1:1,
    same guard ``chain_stats.per_chain_stats`` already applies to the identical keys).

    ``normalize`` shifts both arrays so the FINEST-scale point (index 0, CW§1's own chain-dict
    contract) sits at the origin -- a pure translation, see the module docstring's "Normalize at
    finest scale" section. A no-op on an empty chain (nothing to shift by)."""
    log2_s = np.asarray(chain.get("log2_scales", ()), dtype=np.float64)
    log2_m = np.asarray(chain.get("log2_mod", ()), dtype=np.float64)
    k = min(log2_s.size, log2_m.size)
    x = log2_s[:k].astype(np.float64, copy=True)
    y = log2_m[:k].astype(np.float64, copy=True)
    if px_log2:
        x = x + px_log2
    if normalize and x.size:
        x = x - x[0]
        y = y - y[0]
    return x, y


class SkeletonDialog(QtWidgets.QDialog):
    """The log-log skeleton plot. See the module docstring for the full shape; this class only
    wires the pieces it describes.

    ``chains`` is the ACTIVE layer's own ``result["chains"]`` list -- every index this dialog
    ever emits (:attr:`selectionRequested`) or accepts (:meth:`set_selection`) names a position
    in THIS exact list. ``px_size`` is frame units per pixel, or ``None`` for a bare-pixel frame.
    """

    #: Chain indices (into the ``chains`` list this dialog was built with) whose OLS h falls in
    #: [h_lo, h_hi] -- emitted by a "Select h-range" click.
    selectionRequested = QtCore.Signal(list)

    def __init__(self, chains, px_size: float | None = None, parent=None):
        # Qt.Window (not the QDialog default Qt.Dialog): a real, independently resizable
        # top-level window with min/max/close chrome -- EQSelect's own "Standalone resizable
        # window", not a fixed-chrome modal-flavored dialog.
        super().__init__(parent, QtCore.Qt.Window)
        self.setModal(False)

        self._chains = list(chains)
        self._px_log2 = math.log2(px_size) if px_size else 0.0
        self._normalize = False
        self._selected: set[int] = set()

        # Cached ONCE -- see module docstring's "Stats are computed exactly ONCE" section.
        self._h = chain_stats.stats_for(self._chains, "ols")
        finite = self._h[np.isfinite(self._h)]
        self._h_lo = float(finite.min()) if finite.size else -1.0
        self._h_hi = float(finite.max()) if finite.size else 1.0

        n = len(self._chains)
        self._drawn_n = min(n, MAX_DRAWN)
        cap_note = f" [capped from {n}]" if n > MAX_DRAWN else ""
        self.setWindowTitle(f"WTMM skeleton — log₂ scale vs log₂|W|{cap_note}")

        self._build_ui()
        self._init_static_plot()
        self._update_axis_labels()

    # -- construction ---------------------------------------------------------------------------
    def _build_ui(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self._figure = Figure(constrained_layout=True)   # EQ§4's own colorbar-ratchet fix
        self._ax_main, self._ax_hist = self._figure.subplots(
            1, 2, gridspec_kw={"width_ratios": (3, 1)})
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)

        self._normalize_check = QtWidgets.QCheckBox("Normalize at finest scale")
        self._normalize_check.setToolTip(
            "Anchor every maxima line at its finest-scale coefficient so slopes fan out from a "
            "common origin without crossing.")
        self._normalize_check.toggled.connect(self._on_normalize_toggled)
        layout.addWidget(self._normalize_check)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("h ≥"))
        self._h_lo_spin = self._make_h_spin()
        controls.addWidget(self._h_lo_spin)
        controls.addWidget(QtWidgets.QLabel("h ≤"))
        self._h_hi_spin = self._make_h_spin()
        controls.addWidget(self._h_hi_spin)
        self._h_lo_spin.setValue(self._h_lo)
        self._h_hi_spin.setValue(self._h_hi)
        self._select_button = QtWidgets.QPushButton("Select h-range")
        self._select_button.setToolTip(
            "Select every chain whose OLS Hölder slope falls in [h≥, h≤] -- feeds the shared "
            "selection (Groups panel, canvas highlight).")
        self._select_button.clicked.connect(self._on_select_clicked)
        controls.addWidget(self._select_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.resize(900, 520)

    def _make_h_spin(self) -> QtWidgets.QDoubleSpinBox:
        """EQ§4's own spin shape: 3 decimals, range padded ±1.0 past the data's own finite
        min/max (``self._h_lo``/``self._h_hi``, computed at construction over ALL chains, not
        just the drawn cap)."""
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(self._h_lo - 1.0, self._h_hi + 1.0)
        spin.setSingleStep(max((self._h_hi - self._h_lo) / 100.0, 1e-6))
        return spin

    # -- static plot elements (built once; never rebuilt by normalize/selection) --------------
    def _init_static_plot(self) -> None:
        from matplotlib.collections import LineCollection

        segments = self._segments_for(range(self._drawn_n))
        colors = self._h[:self._drawn_n]
        finite = colors[np.isfinite(colors)]
        if finite.size:
            lo, hi = (float(v) for v in np.percentile(finite, [2, 98]))
            if lo == hi:                     # degenerate: every finite h identical -- set_clim
                lo, hi = lo - 0.5, hi + 0.5   # refuses an empty interval
        else:
            lo, hi = -1.0, 1.0

        self._line_collection = LineCollection(
            segments, array=colors, cmap="coolwarm", alpha=0.2, linewidths=0.5)
        self._line_collection.set_clim(lo, hi)
        self._ax_main.add_collection(self._line_collection)
        self._ax_main.autoscale()
        colorbar = self._figure.colorbar(self._line_collection, ax=self._ax_main)
        colorbar.set_label("Hölder slope h")

        # Selection highlight overlay -- see module docstring's "Highlight color" section.
        self._highlight_collection = LineCollection(
            [], color=_HIGHLIGHT_COLOR, linewidths=_HIGHLIGHT_LW)
        self._ax_main.add_collection(self._highlight_collection)

        all_finite = self._h[np.isfinite(self._h)]
        self._ax_hist.hist(all_finite, bins=_HIST_BINS, range=_HIST_RANGE)
        self._ax_hist.set_xlabel("h")
        self._ax_hist.set_ylabel("count")

    # -- geometry (rebuilt on normalize toggle / set_selection; never touches self._h) --------
    def _segments_for(self, indices) -> list:
        return [np.column_stack(_chain_xy(self._chains[i], self._px_log2, self._normalize))
                for i in indices]

    def _update_geometry(self) -> None:
        self._line_collection.set_segments(self._segments_for(range(self._drawn_n)))
        self._update_highlight()
        self._ax_main.relim()
        self._ax_main.autoscale_view()
        self._canvas.draw_idle()

    def _update_highlight(self) -> None:
        drawable = sorted(i for i in self._selected if i < self._drawn_n)
        self._highlight_collection.set_segments(self._segments_for(drawable))

    def _update_axis_labels(self) -> None:
        if self._normalize:
            self._ax_main.set_xlabel("Δ log₂ scale (from finest)")
            self._ax_main.set_ylabel("log₂|W| − log₂|W|(finest)")
        else:
            self._ax_main.set_xlabel("log₂ scale")
            self._ax_main.set_ylabel("log₂|W|")

    # -- gestures -------------------------------------------------------------------------------
    def _on_normalize_toggled(self, checked: bool) -> None:
        self._normalize = bool(checked)
        self._update_axis_labels()
        self._update_geometry()

    def _on_select_clicked(self) -> None:
        lo, hi = sorted((self._h_lo_spin.value(), self._h_hi_spin.value()))
        indices = [i for i, h in enumerate(self._h) if np.isfinite(h) and lo <= h <= hi]
        self.selectionRequested.emit(indices)

    # -- MainWindow -> dialog (the other half of the bidirectional sync) ----------------------
    def set_selection(self, indices) -> None:
        """Redraw the highlight overlay for ``indices`` (positions into THIS dialog's own
        ``chains`` list) WITHOUT touching :attr:`_h` -- the cached stats array keeps its object
        identity across every call (a test asserts this directly with ``is``). An index at or
        past :data:`MAX_DRAWN` is accepted (kept, in case a future redraw ever widens the cap)
        but never drawn -- there is no on-screen line for it to highlight yet."""
        self._selected = {int(i) for i in indices}
        self._update_highlight()
        self._canvas.draw_idle()
