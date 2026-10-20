# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""SkeletonDialog: the WTMM skeleton log-log plot -- coefficient vs scale, coloured by OLS
Hölder slope, with the "Normalize at finest scale" convergence fan and h-range chain selection
(EQSelect's own ``_WtmmSkeletonDialog``, EQ§4/§5a; the
notebook log-log conventions).

This display is EQSelect's answer to "no log-log plot of coeff vs scale" (spec Mission). The
shape below follows EQSelect's own Hölder-lines view as closely as DynamiX's simpler,
single-view scope allows -- there is no cascade-tree view here (the spec's own non-goals list it
as deferred); every chain this dialog is given is drawn (up to the cap) as one leaf-to-root
maxima-line polyline, coloured by its own OLS slope.

**Figure construction.** One ``pg.GraphicsLayoutWidget`` on the theme's ground holds, left to
right, the main log-log plot, a colour bar and the histogram (column stretch 3 : 0 : 1). The two
plots take the shared wheel gesture of every figure in the app (:mod:`dynamix.shell.plot_zoom`:
the wheel zooms x, Shift and the wheel zoom y) and pyqtgraph's own context menu exports them.
The drawn chains are bucketed into ``_COLOUR_BINS`` colour bins over a diverging blue-red map
whose centre reads against the ground (:func:`_colour_map`: the coolwarm family's white centre
on a dark ground, a grey centre on a light one), and each bin is ONE NaN-separated, antialiased
``PlotDataItem`` (``connect="finite"``), so two thousand chains cost two dozen items. A chain's bin is its OLS slope clipped to the 2nd-98th percentile of the drawn finite
values (short chains produce outlier slopes that would otherwise wash out the colour range for
everything else); the colour bar (``pg.ColorBarItem``, labelled "Hölder slope h") shows those
levels, and a chain without a finite slope draws in the muted ink. One more item on top carries
the selection highlight in the theme's selection accent, the role the canvas's own selected-chain
overlay wears. The histogram shows the FULL (uncapped) finite-h distribution as a filled step
curve, ``bins=50, range=(-3, 2)`` (CW§1's own hard-coded shape) -- independent of the draw cap,
since a histogram costs nothing extra past a few thousand chains and there is no reason to hide
part of the real distribution just because not every line fits on screen.

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

**Normalize at finest scale.** Chain-dict index 0 is the finest scale (CW§1's own chain
dict contract). The checkbox shifts every drawn/highlighted polyline so its own index-0 point
sits at the plot origin -- a pure per-chain translation (``x -= x[0]; y -= y[0]``), so the SLOPE a
line displays is untouched; only where it starts moves. Axis labels switch to the Δ forms
(EQ§5a's own wording, log2 instead of EQSelect's log10 -- the project's scale doctrine keeps this
whole codebase in log2). Toggling never touches ``self._h`` or the colour bar's levels -- both are a
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

**Lifecycle** is owned entirely by ``main_window.py``, not here: a fresh ``SkeletonDialog`` is
built on every "Skeleton plot…" click, from whatever chains are on screen AT THAT MOMENT -- no
caching of an old instance across opens (see that module's own ``_on_skeleton_button_clicked``
docstring for the full reasoning). This dialog itself has no opinion on that; it is simply
constructed, used, and eventually closed like any other ``QDialog``.
"""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core import chain_stats
from dynamix.shell.plot_zoom import WheelAxisViewBox
from dynamix.shell.theme import current, rgb
from dynamix.shell.window_fit import fit_to_screen

__all__ = ["SkeletonDialog", "MAX_DRAWN"]

#: EQ§4's own cap ("Cap 2000 drawn chains, noted in the title") -- geometry only, see module
#: docstring's "The draw cap is geometry-only" section.
MAX_DRAWN = 2000

#: CW§1's own histogram shape (v1-cell-57: ``bins=50, range=(-3, 2)``).
_HIST_BINS = 50
_HIST_RANGE = (-3.0, 2.0)

#: The colour bins the drawn chains are bucketed into, and the diverging maps they sample: the
#: coolwarm family (white centre) on a dark ground, a grey-centred blue-red on a light one.
_COLOUR_BINS = 24
_MAP_ON_DARK = "CET-D1"
_MAP_ON_LIGHT = "CET-D8"
_LINE_WIDTH = 1.0
_HIGHLIGHT_WIDTH = 2.0


def _colour_map(theme) -> str:
    """The diverging map whose centre reads against ``theme.ground``."""
    dark = sum(rgb(theme.ground)) < 3 * 128
    return _MAP_ON_DARK if dark else _MAP_ON_LIGHT


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
        super().__init__(parent, QtCore.Qt.Window)
        self.setModal(False)
        self._chains = list(chains)
        self._px_log2 = math.log2(px_size) if px_size else 0.0
        self._normalize = False
        self._selected: set[int] = set()
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

    def _build_ui(self) -> None:
        theme = current()
        layout = QtWidgets.QVBoxLayout(self)
        self._figure = pg.GraphicsLayoutWidget()
        self._figure.setBackground(theme.ground)
        self._main = self._figure.addPlot(row=0, col=0, viewBox=WheelAxisViewBox())
        self._colour_bar = pg.ColorBarItem(
            values=(-1.0, 1.0), width=14, colorMap=pg.colormap.get(_colour_map(theme)),
            label="Hölder slope h", interactive=False, pen=theme.ink_muted, colorMapMenu=False)
        for side in ("left", "right"):                 # the label's axis and the values' axis
            self._colour_bar.getAxis(side).setTextPen(theme.ink)
        self._figure.addItem(self._colour_bar, row=0, col=1)
        self._hist = self._figure.addPlot(row=0, col=2, viewBox=WheelAxisViewBox())
        self._figure.ci.layout.setColumnStretchFactor(0, 3)
        self._figure.ci.layout.setColumnStretchFactor(2, 1)
        for plot in (self._main, self._hist):
            plot.showGrid(x=True, y=True, alpha=0.15)
            for side in ("bottom", "left"):            # plain decimals, never an "(x0.001)" prefix
                axis = plot.getAxis(side)
                axis.enableAutoSIPrefix(False)
                axis.setTextPen(theme.ink)
                axis.setPen(theme.ink_muted)
        self._hist.setLabel("bottom", "h")
        self._hist.setLabel("left", "count")
        layout.addWidget(self._figure, 1)
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
        fit_to_screen(self, 960, 560)

    def _make_h_spin(self) -> QtWidgets.QDoubleSpinBox:
        """EQ§4's own spin shape: 3 decimals, range padded ±1.0 past the data's own finite
        min/max (``self._h_lo``/``self._h_hi``, computed at construction over ALL chains, not
        just the drawn cap)."""
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(self._h_lo - 1.0, self._h_hi + 1.0)
        spin.setSingleStep(max((self._h_hi - self._h_lo) / 100.0, 1e-6))
        return spin

    def _init_static_plot(self) -> None:
        theme = current()
        slopes = self._h[:self._drawn_n]
        finite = slopes[np.isfinite(slopes)]
        if finite.size:
            lo, hi = (float(v) for v in np.percentile(finite, [2, 98]))
            if lo == hi:                     # degenerate: every finite h identical
                lo, hi = lo - 0.5, hi + 0.5
        else:
            lo, hi = -1.0, 1.0
        self._colour_bar.setLevels((lo, hi))
        fraction = np.clip(np.nan_to_num((slopes - lo) / (hi - lo)), 0.0, 1.0)
        self._bin_of = np.where(np.isfinite(slopes),
                                np.rint(fraction * (_COLOUR_BINS - 1)), -1).astype(int)
        shades = pg.colormap.get(_colour_map(theme)).map(np.linspace(0.0, 1.0, _COLOUR_BINS),
                                                          mode="qcolor")
        self._bin_items = []
        for shade in shades:
            shade.setAlpha(220)
            item = pg.PlotDataItem(pen=pg.mkPen(shade, width=_LINE_WIDTH), connect="finite",
                                   antialias=True)
            self._main.addItem(item)
            self._bin_items.append(item)
        self._unfit_item = pg.PlotDataItem(pen=pg.mkPen(theme.ink_muted, width=_LINE_WIDTH),
                                           connect="finite", antialias=True)
        self._main.addItem(self._unfit_item)
        self._highlight_item = pg.PlotDataItem(
            pen=pg.mkPen(theme.selection_accent, width=_HIGHLIGHT_WIDTH), connect="finite",
            antialias=True)
        self._highlight_item.setZValue(10)
        self._main.addItem(self._highlight_item)
        self._update_geometry()
        all_finite = self._h[np.isfinite(self._h)]
        counts, edges = np.histogram(all_finite, bins=_HIST_BINS, range=_HIST_RANGE)
        fill = pg.mkColor(theme.ink)
        fill.setAlpha(70)
        self._hist_item = self._hist.plot(edges, counts, stepMode="center", fillLevel=0,
                                          brush=pg.mkBrush(fill), pen=pg.mkPen(theme.ink, width=1))

    def _polylines(self, indices) -> tuple[np.ndarray, np.ndarray]:
        """The chains at ``indices`` as one NaN-separated (x, y) pair, in the dialog's current
        px offset and normalize state."""
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        nan = np.array([np.nan])
        for i in indices:
            x, y = _chain_xy(self._chains[i], self._px_log2, self._normalize)
            xs += [x, nan]
            ys += [y, nan]
        if not xs:
            return np.empty(0), np.empty(0)
        return np.concatenate(xs[:-1]), np.concatenate(ys[:-1])

    def _update_geometry(self) -> None:
        drawn = np.arange(self._drawn_n)
        for b, item in enumerate(self._bin_items):
            item.setData(*self._polylines(drawn[self._bin_of == b]))
        self._unfit_item.setData(*self._polylines(drawn[self._bin_of < 0]))
        self._update_highlight()
        self._main.autoRange()

    def _update_highlight(self) -> None:
        drawable = sorted(i for i in self._selected if i < self._drawn_n)
        self._highlight_item.setData(*self._polylines(drawable))

    def _update_axis_labels(self) -> None:
        if self._normalize:
            self._main.setLabel("bottom", "Δ log₂ scale (from finest)")
            self._main.setLabel("left", "log₂|W| − log₂|W|(finest)")
        else:
            self._main.setLabel("bottom", "log₂ scale")
            self._main.setLabel("left", "log₂|W|")

    def _on_normalize_toggled(self, checked: bool) -> None:
        self._normalize = bool(checked)
        self._update_axis_labels()
        self._update_geometry()

    def _on_select_clicked(self) -> None:
        lo, hi = sorted((self._h_lo_spin.value(), self._h_hi_spin.value()))
        indices = [i for i, h in enumerate(self._h) if np.isfinite(h) and lo <= h <= hi]
        self.selectionRequested.emit(indices)

    def set_selection(self, indices) -> None:
        """Redraw the highlight overlay for ``indices`` (positions into THIS dialog's own
        ``chains`` list) WITHOUT touching :attr:`_h` -- the cached stats array keeps its object
        identity across every call (a test asserts this directly with ``is``). An index at or
        past :data:`MAX_DRAWN` is accepted (kept, in case a future redraw ever widens the cap)
        but never drawn -- there is no on-screen line for it to highlight yet."""
        self._selected = {int(i) for i in indices}
        self._update_highlight()
