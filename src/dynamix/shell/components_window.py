# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""ComponentsWindow: the grouping aids for a decomposition (``ssa2d``, ``tucker_HOOI_HOSVD``).

Golyandina & Usevich 2010 group the elementary components by looking at three things, all shown
here from what the decomposition already cached (nothing recomputes):

- **thumbnails** -- the eigenarrays (2D-SSA's ``L_r x L_c`` patterns: a harmonic shows up as a
  pair of shifted stripes, a trend as a smooth blob) or the component images themselves;
- **the shares** -- each component's eigenvalue (2D-SSA) or core-energy (tucker) share, log
  scale, so a slowly-decaying noise floor separates from the leading structure;
- **the w-correlation matrix** (2D-SSA only) -- separable components are nearly w-orthogonal,
  so the groups show up as bright blocks on the diagonal. Hovering reads one entry.

Picking thumbnails IS the group: a click selects one, ⌘-click toggles, shift-click takes a range
(the layer tree's own convention). Every change emits ``groupChanged`` with the device's Group
text ("1-3, 5", or "all" when every component is picked); main_window writes it into the
device's Group knob, so the canvas redraws from the cache. The knob and this window stay in step
both ways (``set_group``). Lifecycle is main_window's: the SkeletonDialog pattern, one window at
a time.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.devices.decompose import format_group, parse_group
from dynamix.shell.theme import RESTRAINED_DARK

__all__ = ["ComponentsWindow", "thumbnail"]

#: Thumbnail edge in screen pixels.
THUMB_PX = 72


def thumbnail(values, size: int = THUMB_PX) -> QtGui.QPixmap:
    """A grayscale thumbnail of one 2-D array, stretched min -> max (NaN black). Large arrays are
    strided down to <= 128 px first; small ones (eigenarrays) are scaled up without smoothing so
    each pattern cell stays a crisp square."""
    a = np.asarray(values, dtype=np.float64)
    step = max(1, int(np.ceil(max(a.shape) / 128)))
    a = a[::step, ::step]
    finite = np.isfinite(a)
    lo, hi = (float(a[finite].min()), float(a[finite].max())) if finite.any() else (0.0, 1.0)
    scaled = (a - lo) / (hi - lo) if hi > lo else np.full(a.shape, 0.5)
    g = np.ascontiguousarray(np.where(finite, np.clip(scaled * 255.0, 0, 255), 0)
                             .astype(np.uint8))
    img = QtGui.QImage(g.data, g.shape[1], g.shape[0], g.strides[0],
                       QtGui.QImage.Format_Grayscale8).copy()
    return QtGui.QPixmap.fromImage(img).scaled(
        size, size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)


class ComponentsWindow(QtWidgets.QDialog):
    """Thumbnails, shares and w-correlations of a decomposition; picking thumbnails sets the
    group."""

    #: The group as the device's Group knob reads it ("1-3, 5" or "all").
    groupChanged = QtCore.Signal(str)
    #: "Fork derivative…": fork what the decomposition row shows (main_window asks what to take).
    forkRequested = QtCore.Signal()

    def __init__(self, components, shares, *, eigenarrays=None, w_correlation=None,
                 group: str = "all", show: str = "recon", share_label: str = "share",
                 title: str = "Components", parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setModal(False)
        self.setWindowTitle(title)
        self._build_ui()
        self.set_decomposition(components, shares, eigenarrays=eigenarrays,
                               w_correlation=w_correlation, share_label=share_label)
        self.set_group(group)
        self.set_show(show)

    # -- construction ---------------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self._group_label = QtWidgets.QLabel()
        top.addWidget(self._group_label, 1)
        self._kind_combo = QtWidgets.QComboBox()
        self._kind_combo.addItems(["Eigenarrays", "Components"])
        self._kind_combo.setToolTip("Thumbnails of the eigenarrays (the window patterns) or of "
                                    "the component images")
        self._kind_combo.currentIndexChanged.connect(lambda _i: self._fill_thumbnails())
        top.addWidget(self._kind_combo)
        self._all_button = QtWidgets.QPushButton("All")
        self._all_button.setToolTip("Group = every component (the whole reconstruction)")
        self._all_button.clicked.connect(self._on_all_clicked)
        top.addWidget(self._all_button)
        self._fork_button = QtWidgets.QPushButton("Fork derivative…")
        self._fork_button.setToolTip("Write what the layer shows (this group, the residual, a "
                                     "component — or any of them as bands) as a dataset of its "
                                     "own")
        self._fork_button.clicked.connect(self.forkRequested)
        top.addWidget(self._fork_button)
        layout.addLayout(top)

        self._hint = QtWidgets.QLabel()
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f"color: {RESTRAINED_DARK.ink_muted};")
        layout.addWidget(self._hint)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._list = QtWidgets.QListWidget()
        self._list.setViewMode(QtWidgets.QListView.IconMode)
        self._list.setIconSize(QtCore.QSize(THUMB_PX, THUMB_PX))
        self._list.setResizeMode(QtWidgets.QListView.Adjust)
        self._list.setMovement(QtWidgets.QListView.Static)
        self._list.setSpacing(4)
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        split.addWidget(self._list)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(RESTRAINED_DARK.ground)
        self._wplot = self._glw.addPlot(row=0, col=0)
        self._wplot.setTitle("w-correlation")
        self._wplot.invertY(True)
        self._wplot.setAspectLocked(True)
        self._wplot.setLabel("bottom", "component")
        self._wimage = pg.ImageItem(axisOrder="row-major")
        self._wimage.setLookupTable(np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, 1))
        self._wplot.addItem(self._wimage)
        self._wmarks = pg.ScatterPlotItem(pxMode=False, symbol="s", size=1.0, brush=None,
                                          pen=pg.mkPen(RESTRAINED_DARK.amber, width=1.5))
        self._wplot.addItem(self._wmarks)
        self._wreading = QtWidgets.QLabel(" ")
        self._wreading.setStyleSheet(f"color: {RESTRAINED_DARK.ink_muted};")
        self._glw.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._splot = self._glw.addPlot(row=1, col=0)
        self._splot.setLogMode(y=True)
        self._splot.setLabel("bottom", "component")
        self._scurve = pg.PlotDataItem(pen=pg.mkPen(RESTRAINED_DARK.ink_muted), symbol="o",
                                       symbolSize=7)
        self._splot.addItem(self._scurve)
        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.addWidget(self._glw, 1)
        right_lay.addWidget(self._wreading)
        split.addWidget(right)
        split.setSizes([420, 360])
        layout.addWidget(split, 1)
        self.resize(820, 560)

    # -- data -----------------------------------------------------------------------------------
    @property
    def components(self):
        """The component stack this window shows (main_window's staleness check)."""
        return self._components

    def set_decomposition(self, components, shares, *, eigenarrays=None, w_correlation=None,
                          share_label: str = "share") -> None:
        """(Re)load a decomposition; the current group is kept where it still applies."""
        keep = self.group_text() if hasattr(self, "_components") else "all"
        self._components = components
        self._shares = np.asarray(shares, dtype=np.float64)
        self._eigenarrays = eigenarrays
        self._wcorr = None if w_correlation is None else np.asarray(w_correlation)
        self._n = len(components)
        self._kind_combo.setVisible(eigenarrays is not None)
        self._splot.setTitle(share_label)
        self._fill_thumbnails()
        has_w = self._wcorr is not None
        self._wplot.setVisible(has_w)
        self._wreading.setVisible(has_w)
        if has_w:
            self._wimage.setImage(self._wcorr, levels=(0.0, 1.0))
            self._wimage.setRect(QtCore.QRectF(0.5, 0.5, self._n, self._n))
        self._sx = np.arange(1, self._n + 1, dtype=np.float64)
        self._sy = np.maximum(self._shares[:self._n], 1e-16)
        self.set_group(keep)

    def _fill_thumbnails(self) -> None:
        picked = self._selected()
        use_eigen = self._eigenarrays is not None and self._kind_combo.currentIndex() == 0
        arrays = self._eigenarrays if use_eigen else self._components
        self._list.blockSignals(True)
        self._list.clear()
        for k in range(self._n):
            item = QtWidgets.QListWidgetItem(
                QtGui.QIcon(thumbnail(arrays[k])),
                f"C{k + 1} · {100.0 * float(self._shares[k]):.1f}%")
            item.setTextAlignment(QtCore.Qt.AlignHCenter)
            self._list.addItem(item)
        for k in picked:
            if k < self._n:
                self._list.item(k).setSelected(True)
        self._list.blockSignals(False)

    # -- the group ------------------------------------------------------------------------------
    def _selected(self) -> list:
        return sorted(self._list.row(item) for item in self._list.selectedItems())

    def group_text(self) -> str:
        idx = self._selected()
        return "all" if len(idx) == self._n else format_group(idx)

    def set_group(self, text: str) -> None:
        """Show ``text`` (the Group knob's value) as the picked thumbnails -- never emits."""
        idx = set(parse_group(text, self._n))
        self._list.blockSignals(True)
        for k in range(self._n):
            self._list.item(k).setSelected(k in idx)
        self._list.blockSignals(False)
        self._refresh_readouts()

    def set_show(self, show: str) -> None:
        """The device's Show, so the window can say when the group is not on screen."""
        self._hint.setText(
            "Show is 'component' — the group applies when Show is recon or residual"
            if show == "component" else
            "Click a thumbnail to pick it, ⌘-click to add or remove one, shift-click for a "
            "range. Recon = the sum of the group; residual = the data minus it.")

    def _on_all_clicked(self) -> None:
        self.set_group("all")
        self.groupChanged.emit("all")

    def _on_selection_changed(self) -> None:
        self._refresh_readouts()
        if self._selected():                  # an empty pick is not a group: keep the last one
            self.groupChanged.emit(self.group_text())

    def _refresh_readouts(self) -> None:
        idx = self._selected()
        if not idx:
            self._group_label.setText("Group: none picked (the knob keeps its last group)")
        else:
            pct = 100.0 * float(self._shares[idx].sum())
            text = self.group_text()
            name = f"all {self._n}" if text == "all" else f"[{text}]"
            self._group_label.setText(f"Group: {name} · {pct:.1f}%")
        amber = QtGui.QColor(RESTRAINED_DARK.amber)
        muted = QtGui.QColor(RESTRAINED_DARK.ink_muted)
        picked = set(idx)
        self._scurve.setData(self._sx, self._sy, symbolBrush=[
            pg.mkBrush(amber if k in picked else muted) for k in range(self._n)])
        diag = np.array([k + 1 for k in idx], dtype=np.float64)
        self._wmarks.setData(diag, diag)

    def _on_mouse_moved(self, pos) -> None:
        if self._wcorr is None or not self._wplot.sceneBoundingRect().contains(pos):
            return
        p = self._wplot.vb.mapSceneToView(pos)
        i, j = int(round(p.y())), int(round(p.x()))
        if 1 <= i <= self._n and 1 <= j <= self._n:
            self._wreading.setText(f"w(C{i}, C{j}) = {float(self._wcorr[i - 1, j - 1]):.3f}")
