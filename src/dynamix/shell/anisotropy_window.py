# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Anisotropy window: WTMMM angle statistics after Arnéodo, Decoster & Roux 2000 (§6,
Figs 24–26), in the pixel frame (the paper's A) or as bearings.

Three plots on the active result's WTMMM (xsmurf ``ssm``):

- P_a(A) at several scales over one flat reference: FLAT = isotropic; the shape's change
  across scales says whether anisotropy strengthens or weakens;
- the WTMMM of one scale in the gradient plane (T_ψ1, T_ψ2) -- as (east, north) components
  for bearings: radial symmetry = isotropy;
- the pdf of log2 M in the four axial sectors of the angle at that scale: shape invariance =
  M and A independent.

Bearings exist only for a georeferenced grid (:func:`dynamix.geo.mapping.field_north`): grid
north for a projected grid (plus true north through the meridian convergence), true north for
lon/lat. A swath or pixel grid offers the pixel frame alone, and says so.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from dynamix.core.anisotropy import (SECTOR_LABELS, angle_pdf, gradient_plane,
                                     sector_modulus_pdfs, wtmmm_by_scale)
from dynamix.core.orientation import Frame, from_arg, grid_to_true, to_azimuth

__all__ = ["AnisotropyWindow"]

_SECTOR_COLORS = ("#e15759", "#f2c14e", "#59a14f", "#4e79a7")
_SCALE_COLORS = ("#4e79a7", "#59a14f", "#f2c14e", "#e15759", "#b07aa1", "#9c755f")


class AnisotropyWindow(QtWidgets.QWidget):
    """A floating window; :meth:`set_result` feeds it the active result and its field."""

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Anisotropy — WTMMM angle statistics")
        self.resize(1180, 460)
        self._result = None
        self._field = None
        self._offset = (0.0, 0.0)
        self._wtmmm = None                  # (result, per-scale WTMMM)
        self._north = None                  # field_north(field): axis directions + frame
        outer = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Angle frame"))
        self.frame_combo = QtWidgets.QComboBox()
        bar.addWidget(self.frame_combo)
        bar.addWidget(QtWidgets.QLabel("Scale"))
        self.scale_spin = QtWidgets.QSpinBox()
        bar.addWidget(self.scale_spin)
        bar.addWidget(QtWidgets.QLabel("Bins"))
        self.bins_spin = QtWidgets.QSpinBox()
        self.bins_spin.setRange(8, 180)
        self.bins_spin.setValue(36)
        bar.addWidget(self.bins_spin)
        self.note = QtWidgets.QLabel("")
        bar.addWidget(self.note, 1)
        outer.addLayout(bar)
        plots = QtWidgets.QHBoxLayout()
        self.pdf_plot = pg.PlotWidget(title="P_a(A) across scales")
        self.pdf_plot.addLegend(offset=(-5, 5))
        self.plane_plot = pg.PlotWidget(title="WTMMM in the gradient plane")
        self.plane_plot.setAspectLocked(True)
        self.sector_plot = pg.PlotWidget(title="pdf of log2 M per angle sector")
        self.sector_plot.addLegend(offset=(-5, 5))
        for w in (self.pdf_plot, self.plane_plot, self.sector_plot):
            plots.addWidget(w, 1)
        outer.addLayout(plots, 1)
        outer.addWidget(QtWidgets.QLabel(
            "Arnéodo, Decoster & Roux 2000, §6: a FLAT P_a(A) means isotropic scaling; the "
            "gradient-plane cloud is round when isotropic; identical sector pdfs mean M and A "
            "are independent (the spectrum carries no direction)."))
        self.frame_combo.currentIndexChanged.connect(self.refresh)
        self.scale_spin.valueChanged.connect(self.refresh)
        self.bins_spin.valueChanged.connect(self.refresh)

    # -- data ---------------------------------------------------------------------------------
    def set_result(self, result, field, *, offset=(0.0, 0.0), scale_idx: int = 0) -> None:
        """The active result (needs per-scale ``extrema``), its field (for bearings) and the
        result's pixel offset inside that field (an ROI result's origin)."""
        self._result, self._field, self._offset = result, field, tuple(offset)
        n = len((result or {}).get("extrema") or [])
        self.scale_spin.blockSignals(True)
        self.scale_spin.setRange(0, max(n - 1, 0))
        self.scale_spin.setValue(min(max(int(scale_idx), 0), max(n - 1, 0)))
        self.scale_spin.blockSignals(False)
        self._rebuild_frames()
        self.refresh()

    def _rebuild_frames(self) -> None:
        from dynamix.geo.mapping import field_north

        current = self.frame_combo.currentData()
        north = None
        try:
            north = field_north(self._field) if self._field is not None else None
        except Exception:
            north = None
        self._north = north
        self.frame_combo.blockSignals(True)
        self.frame_combo.clear()
        self.frame_combo.addItem("pixel angle A", "pixel")
        if north is not None:
            if north[2] is Frame.GRID:
                self.frame_combo.addItem("azimuth — grid north", "grid")
            self.frame_combo.addItem("azimuth — true north", "true")
        idx = self.frame_combo.findData(current)
        self.frame_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.frame_combo.blockSignals(False)
        self.note.setText("" if north is not None else
                          "no georeference (a swath or pixel grid): pixel frame only")

    def _per_scale(self) -> list:
        if self._result is None:
            return []
        if self._wtmmm is None or self._wtmmm[0] is not self._result:
            self._wtmmm = (self._result, wtmmm_by_scale(self._result.get("extrema") or []))
        return self._wtmmm[1]

    def _orientation(self, w: dict):
        """This scale's WTMMM angles in the chosen frame."""
        from dynamix.geo.mapping import grid_north_bearing

        o = from_arg(w["arg"])
        kind = self.frame_combo.currentData() or "pixel"
        if kind == "pixel" or self._north is None:
            return o
        col_east, row_north, north = self._north
        az = to_azimuth(o, col_east=col_east, row_north=row_north, north=north)
        if kind == "true" and north is Frame.GRID:
            cols = np.asarray(w["x"], float) + self._offset[1]
            rows = np.asarray(w["y"], float) + self._offset[0]
            az = grid_to_true(az, grid_north_bearing(self._field, cols, rows))
        return az

    # -- drawing ------------------------------------------------------------------------------
    def refresh(self, *_a) -> None:
        for plot in (self.pdf_plot, self.plane_plot, self.sector_plot):
            plot.clear()
        per_scale = self._per_scale()
        if not per_scale:
            return
        scales = np.asarray(self._result.get("scales")
                            if self._result.get("scales") is not None
                            else np.arange(1, len(per_scale) + 1), dtype=float)
        bins = int(self.bins_spin.value())
        pick = np.unique(np.linspace(0, len(per_scale) - 1,
                                     min(len(per_scale), len(_SCALE_COLORS))).astype(int))
        for colour, si in zip(_SCALE_COLORS, pick):
            o = self._orientation(per_scale[si])
            if np.size(o.degrees) == 0:
                continue
            c, pdf = angle_pdf(o, bins=bins)
            label = f"a = {scales[si]:g} ({np.size(o.degrees)})"
            self.pdf_plot.plot(c, pdf, pen=pg.mkPen(colour, width=1.6), name=label)
        self.pdf_plot.addLine(y=1.0 / 360.0, pen=pg.mkPen("#888", style=QtCore.Qt.DashLine))
        pixel = (self.frame_combo.currentData() or "pixel") == "pixel"
        self.pdf_plot.setLabel("bottom", "A (deg, pixel frame)" if pixel else "azimuth (deg)")
        si = int(self.scale_spin.value())
        w = per_scale[si]
        o = self._orientation(w)
        mod = np.asarray(w["mod"], float)
        if pixel:
            t1, t2 = gradient_plane(mod, w["arg"])
            self.plane_plot.setLabel("bottom", "T_ψ1")
            self.plane_plot.setLabel("left", "T_ψ2")
        else:
            az = np.radians(np.asarray(o.degrees, float))
            t1, t2 = mod * np.sin(az), mod * np.cos(az)          # east, north components
            self.plane_plot.setLabel("bottom", "east")
            self.plane_plot.setLabel("left", "north")
        self.plane_plot.plot(t1, t2, pen=None, symbol="o", symbolSize=3,
                             symbolBrush=pg.mkBrush(78, 121, 167, 140), symbolPen=None)
        self.plane_plot.setTitle(f"WTMMM in the gradient plane, a = {scales[si]:g}")
        labels = SECTOR_LABELS["pixel" if pixel else "azimuth"]
        for (k, c, pdf, count), colour in zip(sector_modulus_pdfs(mod, o), _SECTOR_COLORS):
            self.sector_plot.plot(c, pdf, pen=pg.mkPen(colour, width=1.6),
                                  name=f"{labels[k]} ({count})")
        self.sector_plot.setLabel("bottom", "log2 M")
        self.sector_plot.setTitle(f"pdf of log2 M per angle sector, a = {scales[si]:g}")
