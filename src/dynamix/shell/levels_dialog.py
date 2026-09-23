# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The two histogram dialogs: color slicing (display) vs h-band masking (analysis).

Split: the original single dialog blended two unrelated jobs into
one tag store, so dragging the reconstruction band silently rewrote the display ramp's
bounds. Now:

* :class:`LevelsDialog` -- "Slice…": the ENVI density-slice editor for the DISPLAYED raster's
  COLORING only. Class bounds + per-class colors (the picker's alpha channel makes a class
  transparent -- "none", which is how the Turiel set display lives here now: one opaque class,
  the rest alpha 0). Emits ``levelsApplied`` -> the ``ui.levels``/``ui.levels_colors`` tags.
  It never touches the chain.
* :class:`BandDialog` -- "Reconstruct…": the h-band MASK for ``band_recon``. One draggable
  region on the h histogram, live reconstruction preview, and the full-resolution commit.
  Emits ``bandPreviewRequested``/``reconstructRequested``; it never writes a display tag, so
  scrubbing the band can no longer move the color ramp.

Both share ``_HistogramDialog``: histogram + bin-width control + HORIZONTAL-ONLY mouse zoom
(the count axis autoscales; y-zoom on a histogram is never the wanted gesture).

pyqtgraph; chrome resolves through ``dynamix.shell.theme`` (the ``canvas.py`` precedent);
class colors are user data. Lifecycle: one per open, ``Qt.Window``, non-modal, live Apply.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core.stretch import parse_class_colors, parse_levels, resolve_breaks
from dynamix.shell.theme import RESTRAINED_DARK

__all__ = ["LevelsDialog", "BandDialog"]

#: Default class count when the slice dialog opens with no active spec.
_DEFAULT_CLASSES = 5


class _HistogramDialog(QtWidgets.QDialog):
    """Shared base: the value histogram, a bin-width spin, horizontal-only zoom."""

    def __init__(self, values, title: str, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setModal(False)
        self.setWindowTitle(title)
        v = np.asarray(values, dtype=np.float64).ravel()
        self._finite = v[np.isfinite(v)]
        if self._finite.size == 0:
            self._finite = np.zeros(1)
        self._lo = float(np.percentile(self._finite, 0.5))
        self._hi = float(np.percentile(self._finite, 99.5))
        if self._hi <= self._lo:
            self._hi = self._lo + 1.0

    def _build_histogram_ui(self, layout, controls) -> None:
        """Bin-width spin into ``controls``; the plot into ``layout``."""
        controls.addWidget(QtWidgets.QLabel("Bin width"))
        self._bin_spin = QtWidgets.QDoubleSpinBox()
        self._bin_spin.setDecimals(4)
        default_bw = (self._hi - self._lo) / 80.0
        self._bin_spin.setRange(default_bw / 50.0, (self._hi - self._lo))
        self._bin_spin.setSingleStep(default_bw / 4.0)
        self._bin_spin.setValue(default_bw)
        self._bin_spin.valueChanged.connect(lambda _v: self._redraw_histogram())
        controls.addWidget(self._bin_spin)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(RESTRAINED_DARK.ground)
        self._plot.setLabel("bottom", "value")
        self._plot.setLabel("left", "count")
        # Horizontal-only zoom: wheel/drag zoom moves x only; the count
        # axis re-autoscales to whatever is in view.
        vb = self._plot.getViewBox()
        vb.setMouseEnabled(x=True, y=False)
        vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        vb.setAutoVisible(y=True)
        self._hist_item = pg.PlotDataItem(stepMode="center", fillLevel=0,
                                          pen=pg.mkPen(RESTRAINED_DARK.ink_muted),
                                          brush=pg.mkBrush(RESTRAINED_DARK.raised))
        self._plot.addItem(self._hist_item)
        layout.addWidget(self._plot, 1)

    def _redraw_histogram(self) -> None:
        bw = max(self._bin_spin.value(), 1e-12)
        edges = np.arange(self._lo, self._hi + bw, bw)
        if edges.size < 3:
            edges = np.linspace(self._lo, self._hi, 3)
        counts, edges = np.histogram(self._finite, bins=edges)
        self._hist_item.setData(edges, counts)


class LevelsDialog(_HistogramDialog):
    """Color slicing for the DISPLAYED raster: class bounds + colors on its histogram.

    ``levels_text``/``colors_text`` seed from the layer's tags; ``colormap_name`` seeds
    missing class colors. A class picked with alpha 0 renders transparent ("none") -- the
    single-band set display is one opaque class between two transparent ones.
    """

    #: Apply/OK -- (breaks_text, colors_text) for the ui.levels/_colors tags.
    levelsApplied = QtCore.Signal(str, str, int)
    #: Coalesced live ticks (Live on): same payload as levelsApplied, but PREVIEW -- the
    #: canvas recolors immediately while bounds drag / colors change (manual feature
    #: delineation); no tag is filed until Apply/OK.
    levelsPreviewRequested = QtCore.Signal(str, str, int)

    def __init__(self, values, *, levels_text: str = "", colors_text: str = "",
                 colormap_name: str = "viridis", min_island: int = 0, parent=None):
        super().__init__(values, "Color slice — class bounds on the displayed histogram",
                         parent)
        self._seed_min_island = int(min_island)
        self._colormap_name = colormap_name
        self._lines: list = []
        self._regions: list = []
        self._colors: list = []                       # (r, g, b, a) per class
        self._live_pending = False
        self._syncing_regions = False
        self._live_timer = QtCore.QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(60)
        self._live_timer.timeout.connect(self._flush_live)
        # Lazy sieve tier -- the BandDialog contract: ticks never label; one sieved pass on
        # settle (~400 ms after the last gesture).
        self._settle_timer = QtCore.QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(400)
        self._settle_timer.timeout.connect(self._flush_settled)

        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        self._build_histogram_ui(layout, controls)
        self._add_button = QtWidgets.QPushButton("+ bound")
        self._add_button.clicked.connect(self._on_add_bound)
        controls.addWidget(self._add_button)
        self._remove_button = QtWidgets.QPushButton("− bound")
        self._remove_button.clicked.connect(self._on_remove_bound)
        controls.addWidget(self._remove_button)
        self._live_check = QtWidgets.QCheckBox("Live")
        self._live_check.setChecked(True)
        self._live_check.setToolTip("Recolor the view while dragging bounds / picking colors "
                                    "(preview; Apply files it into the layer)")
        controls.addWidget(self._live_check)
        controls.addWidget(QtWidgets.QLabel("Min island"))
        self._island_spin = QtWidgets.QSpinBox()
        self._island_spin.setRange(0, 100000)
        self._island_spin.setValue(getattr(self, "_seed_min_island", 0))
        self._island_spin.setToolTip("Suppress each class's islands below this many pixels "
                                     "(0 = off) — rendered transparent. Applied lazily after "
                                     "the drag settles, and on Apply.")
        self._island_spin.valueChanged.connect(lambda _v: self._settle_timer.start())
        controls.addWidget(self._island_spin)
        controls.addStretch(1)
        layout.insertLayout(0, controls)

        self._color_row = QtWidgets.QHBoxLayout()
        layout.addLayout(self._color_row)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Close)
        # The way BACK: the tag convention has always been levels = "" -> off, but no
        # gesture ever SENT empty. This files exactly that and closes -- the layer
        # returns to its continuous colormap untouched.
        clear_btn = buttons.addButton("Clear slice", QtWidgets.QDialogButtonBox.ResetRole)
        clear_btn.setToolTip("Remove the discrete slice — back to the continuous colormap")
        clear_btn.clicked.connect(
            lambda: (self.levelsApplied.emit("", "", 0), self.accept()))
        buttons.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self._emit)
        buttons.accepted.connect(lambda: (self._emit(), self.accept()))
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(760, 460)

        self._seed(levels_text, colors_text)
        self._redraw_histogram()
        self._rebuild_classes()

    # -- seed / bounds ------------------------------------------------------------------------
    def _seed(self, levels_text: str, colors_text: str) -> None:
        """Boundaries from the layer's tags (a count resolves through the SAME quantile rule
        the display uses); colors from the tag (RGBA -- "none" = alpha 0 survives the round
        trip), padded by colormap samples."""
        breaks = None
        try:
            spec = parse_levels(levels_text)
            if spec is not None:
                breaks = list(resolve_breaks(self._finite, spec))
        except (ValueError, TypeError):
            pass
        if not breaks:
            breaks = list(resolve_breaks(self._finite, _DEFAULT_CLASSES))
        try:
            self._colors = list(parse_class_colors(colors_text) or [])
        except ValueError:
            self._colors = []
        for x in breaks:
            self._add_line(float(x))

    def _add_line(self, x: float):
        line = pg.InfiniteLine(pos=x, angle=90, movable=True,
                               pen=pg.mkPen(RESTRAINED_DARK.ink, width=2.0),
                               hoverPen=pg.mkPen(RESTRAINED_DARK.amber, width=2.5))
        line.sigPositionChanged.connect(lambda _l: (self._update_regions(),
                                                     self._queue_live()))
        self._plot.addItem(line)
        self._lines.append(line)
        return line

    def bounds(self) -> list:
        return sorted(float(line.value()) for line in self._lines)

    # -- classes ------------------------------------------------------------------------------
    def _default_color(self, i: int, n: int) -> tuple:
        """Colormap sample at the class midpoint -- the seed a user then edits. RGBA."""
        try:
            import matplotlib
            rgba = matplotlib.colormaps[self._colormap_name]((i + 0.5) / max(n, 1))
            return tuple(int(round(c * 255)) for c in rgba[:3]) + (255,)
        except Exception:                                        # noqa: BLE001
            g = int(round(255 * (i + 0.5) / max(n, 1)))
            return (g, g, g, 255)

    def _rebuild_classes(self) -> None:
        """One tinted span + one color button per class; transparent classes tint nothing and
        label their button "none"."""
        n = len(self._lines) + 1
        while len(self._colors) < n:
            self._colors.append(self._default_color(len(self._colors), n))
        del self._colors[n:]
        for region in self._regions:
            self._plot.removeItem(region)
        self._regions = []
        for i in range(n):
            r, g, b, a = self._colors[i]
            # Movable as a UNIT: grabbing a class span drags its
            # bounding lines together; the span's own edge-lines are disarmed so edge drags
            # stay the InfiniteLines' job.
            region = pg.LinearRegionItem(orientation="vertical", movable=True,
                                         brush=pg.mkBrush(r, g, b, 70 if a else 0),
                                         pen=pg.mkPen(None))
            for edge_line in region.lines:
                edge_line.setMovable(False)
            region.setZValue(-10)
            region.sigRegionChanged.connect(lambda _r, k=i: self._on_region_dragged(k))
            self._plot.addItem(region)
            self._regions.append(region)
        self._update_regions()
        while self._color_row.count():
            item = self._color_row.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for i in range(n):
            r, g, b, a = self._colors[i]
            btn = QtWidgets.QPushButton("none" if a == 0 else f"class {i + 1}")
            if a:
                btn.setStyleSheet("background-color: #{:02x}{:02x}{:02x};".format(r, g, b))
            btn.setToolTip("Pick the class color — alpha 0 makes the class transparent "
                           "(the single-band set display)")
            btn.clicked.connect(lambda _c, k=i: self._on_pick_color(k))
            self._color_row.addWidget(btn)
        self._color_row.addStretch(1)

    def _update_regions(self) -> None:
        bounds = self.bounds()
        span = self._hi - self._lo
        edges = [self._lo - span] + bounds + [self._hi + span]
        prev = self._syncing_regions
        self._syncing_regions = True
        try:
            for i, region in enumerate(self._regions):
                if i + 1 < len(edges):
                    region.setRegion((edges[i], edges[i + 1]))
        finally:
            self._syncing_regions = prev

    def _on_region_dragged(self, i: int) -> None:
        """A class span was dragged as a unit: shift its REAL bounding lines by the drag
        delta (the outermost spans touch one real bound; interior spans move two). The moved
        lines' own signals re-normalize every span and queue the live preview."""
        if self._syncing_regions or i >= len(self._regions):
            return
        pairs = sorted((float(line.value()), line) for line in self._lines)
        span = self._hi - self._lo
        edges = [self._lo - span] + [v for v, _l in pairs] + [self._hi + span]
        new_lo, new_hi = self._regions[i].getRegion()
        # whole-unit drags move both edges by one delta; read it off whichever edge is real
        delta = (float(new_lo) - edges[i]) if i > 0 else (float(new_hi) - edges[i + 1])
        if abs(delta) < 1e-15:
            return
        prev = self._syncing_regions
        self._syncing_regions = True
        try:
            if i > 0:
                pairs[i - 1][1].setValue(pairs[i - 1][0] + delta)
            if i < len(pairs):
                pairs[i][1].setValue(pairs[i][0] + delta)
        finally:
            self._syncing_regions = prev
        self._update_regions()
        self._queue_live()

    # -- gestures -----------------------------------------------------------------------------
    def _on_add_bound(self) -> None:
        bounds = self.bounds()
        gaps = [self._lo] + bounds + [self._hi]
        widths = np.diff(gaps)
        i = int(np.argmax(widths))
        self._add_line(float(gaps[i] + widths[i] / 2.0))
        self._rebuild_classes()
        self._queue_live()

    def _on_remove_bound(self) -> None:
        if len(self._lines) <= 1:
            return
        line = self._lines.pop()
        self._plot.removeItem(line)
        self._rebuild_classes()
        self._queue_live()

    def _on_pick_color(self, i: int) -> None:
        from PySide6 import QtGui
        r, g, b, a = self._colors[i]
        picked = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(r, g, b, a), self, f"Class {i + 1} color",
            QtWidgets.QColorDialog.ShowAlphaChannel)
        if picked.isValid():
            self._colors[i] = (picked.red(), picked.green(), picked.blue(),
                               0 if picked.alpha() == 0 else 255)
            self._rebuild_classes()
            self._queue_live()
        # The modal color picker's close hands focus back to the MAIN window on macOS,
        # dropping this Qt.Window behind it -- reclaim our spot either way.
        self.raise_()
        self.activateWindow()

    def _texts(self) -> tuple:
        breaks_text = ", ".join(f"{b:g}" for b in self.bounds())
        colors_text = ",".join(
            "none" if c[3] == 0 else "#{:02x}{:02x}{:02x}".format(*c[:3])
            for c in self._colors)
        return breaks_text, colors_text

    def _queue_live(self) -> None:
        if not self._live_check.isChecked():
            return
        self._live_pending = True
        if not self._live_timer.isActive():
            self._live_timer.start()
        self._settle_timer.start()

    def _flush_live(self) -> None:
        if self._live_pending:
            self._live_pending = False
            self.levelsPreviewRequested.emit(*self._texts(), 0)   # ticks never sieve

    def _flush_settled(self) -> None:
        if self._live_check.isChecked() and self._island_spin.value() > 0:
            self.levelsPreviewRequested.emit(*self._texts(),
                                             int(self._island_spin.value()))

    def _emit(self) -> None:
        self.levelsApplied.emit(*self._texts(), int(self._island_spin.value()))


class BandDialog(_HistogramDialog):
    """The h-band MASK for reconstruction: one draggable region on the h histogram, live
    preview, full-resolution commit. Writes no display tag, ever."""

    #: Coalesced drag ticks (Live on) -- (h_lo, h_hi, mode) at most every ~50 ms. Mode
    #: "mask" (default) renders the BINARY singularity set -- the decision view: which pixels
    #: feed the inversion; "reconstruction" renders the decimated inverse.
    bandPreviewRequested = QtCore.Signal(float, float, str, int)
    #: "Reconstruct" -- (h_lo, h_hi): the full-resolution band_recon commit.
    reconstructRequested = QtCore.Signal(float, float, int)

    def __init__(self, h_values, *, h_lo: "float | None" = None,
                 h_hi: "float | None" = None, parent=None):
        super().__init__(h_values, "Reconstruct from h band — mask on the h histogram",
                         parent)
        self._preview_pending = None
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(50)
        self._preview_timer.timeout.connect(self._flush_preview)
        # LAZY sieve tier:
        # drag ticks NEVER label clusters; this timer restarts on every tick and fires one
        # sieved preview ~400 ms after the hand stops.
        self._settle_timer = QtCore.QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(400)
        self._settle_timer.timeout.connect(self._flush_settled)

        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        self._build_histogram_ui(layout, controls)
        self._live_check = QtWidgets.QCheckBox("Live")
        self._live_check.setChecked(True)
        self._live_check.setToolTip("Preview while dragging (decimated; Reconstruct commits "
                                    "at full resolution)")
        controls.addWidget(self._live_check)
        controls.addWidget(QtWidgets.QLabel("Preview"))
        self._preview_combo = QtWidgets.QComboBox()
        self._preview_combo.addItems(["mask", "reconstruction"])
        self._preview_combo.setToolTip("mask: the binary singularity set (which pixels feed "
                                       "the inversion — the decision view); reconstruction: "
                                       "the decimated inverse itself")
        self._preview_combo.currentTextChanged.connect(lambda _t: self._queue_preview())
        controls.addWidget(self._preview_combo)
        self._recon_button = QtWidgets.QPushButton("Reconstruct")
        self._recon_button.setToolTip("Commit the selected h band as a band_recon step on "
                                      "this layer's chain, at full resolution")
        self._recon_button.clicked.connect(self._on_reconstruct)
        controls.addWidget(self._recon_button)
        controls.addWidget(QtWidgets.QLabel("Min island"))
        self._island_spin = QtWidgets.QSpinBox()
        self._island_spin.setRange(0, 100000)
        self._island_spin.setValue(0)
        self._island_spin.setToolTip("Suppress mask islands below this many pixels (0 = off). "
                                     "Applied lazily — after the drag settles — and always on "
                                     "Reconstruct. Connectivity/max live on the band_recon "
                                     "device.")
        self._island_spin.valueChanged.connect(lambda _v: self._settle_timer.start())
        controls.addWidget(self._island_spin)
        controls.addStretch(1)
        layout.insertLayout(0, controls)

        lo = float(h_lo) if h_lo is not None else float(np.percentile(self._finite, 5))
        hi = float(h_hi) if h_hi is not None else float(np.percentile(self._finite, 25))
        band = pg.mkColor(RESTRAINED_DARK.amber)
        band.setAlpha(70)
        self._band_region = pg.LinearRegionItem(values=(lo, hi), orientation="vertical",
                                                brush=pg.mkBrush(band),
                                                pen=pg.mkPen(RESTRAINED_DARK.amber))
        self._band_region.sigRegionChanged.connect(lambda _r: self._queue_preview())
        self._plot.addItem(self._band_region)
        self._redraw_histogram()

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(760, 420)
        self._queue_preview()

    def band(self) -> tuple:
        return tuple(sorted(float(x) for x in self._band_region.getRegion()))

    def _queue_preview(self) -> None:
        """Coalesce drag ticks: pyqtgraph emits per mouse move; the preview costs ~20 ms.
        Ticks carry sieve 0 (never label mid-drag); the settle timer restarts, so one sieved
        pass follows the moment the drag stops."""
        if not self._live_check.isChecked():
            return
        self._preview_pending = self.band()
        if not self._preview_timer.isActive():
            self._preview_timer.start()
        self._settle_timer.start()

    def _flush_preview(self) -> None:
        if self._preview_pending is not None:
            lo, hi = self._preview_pending
            self._preview_pending = None
            self.bandPreviewRequested.emit(lo, hi, self._preview_combo.currentText(), 0)

    def _flush_settled(self) -> None:
        if self._live_check.isChecked() and self._island_spin.value() > 0:
            lo, hi = self.band()
            self.bandPreviewRequested.emit(lo, hi, self._preview_combo.currentText(),
                                           int(self._island_spin.value()))

    def _on_reconstruct(self) -> None:
        lo, hi = self.band()
        self.reconstructRequested.emit(lo, hi, int(self._island_spin.value()))
