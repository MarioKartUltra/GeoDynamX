# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Map view: the layer stack drawn in one map CRS. A raster row in the map's CRS is placed
by an affine transform from its own axes; another CRS is reprojected for display (cached per
row and result). Each row paints with the Raster canvas's own paint code and its layer's
style. Display only -- analysis never reads what is drawn here."""
from __future__ import annotations

import dataclasses

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.geo.placement import field_crs, image_transform, reproject_for_display
from dynamix.shell.canvas import Canvas, lod_stride

MAX_DIM = 2048


@dataclasses.dataclass
class RowContent:
    kind: str
    field: object = None
    style: dict | None = None
    composite: dict | None = None
    result: dict | None = None
    scale_idx: int = 0
    grid_field: object = None
    reason: str | None = None


class _RowPainter:
    """One row's display state, painted with the Raster canvas's own methods."""

    _paint = Canvas._paint
    _class_rgba = Canvas._class_rgba
    _display_scalars = Canvas._display_scalars

    def __init__(self) -> None:
        self._colormap = "viridis"
        self._composite = None
        self._hillshade = (False, 315.0, 45.0, 1.0)
        self._stretch = ("linear", 2.0)
        self._levels, self._levels_colors, self._levels_sieve = "", "", 0

    def apply_style(self, style: dict, composite) -> None:
        self._colormap = style["colormap"]
        self._stretch = (style["stretch"], float(style["stretch_pct"]))
        self._hillshade = (bool(style["hillshade"]), float(style["sun_azimuth"]),
                           float(style["sun_altitude"]), float(style["z_factor"]))
        self._levels, self._levels_colors, self._levels_sieve = (
            style["levels"], style["levels_colors"], int(style["levels_sieve"]))
        self._composite = dict(composite) if composite else None


class MapView(QtWidgets.QWidget):
    rowClicked = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        self.view = self.plot.getViewBox()
        self.view.setAspectLocked(True)
        self.readout = QtWidgets.QLabel("")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.readout)
        self._crs = None
        self._frame = None              # the first frame, when the project has no CRS
        self._items: dict[str, pg.ImageItem] = {}
        self._vectors: dict[str, tuple] = {}         # extrema / chains rows
        self._painters: dict[str, _RowPainter] = {}
        self._warp_cache: dict = {}
        self._reasons: dict[str, str] = {}

    def set_map_crs(self, crs) -> None:
        self._crs = str(crs) if crs else None
        self._warp_cache.clear()

    def reasons(self) -> dict:
        return dict(self._reasons)

    def item_rect(self, row_id) -> QtCore.QRectF:
        item = self._items[row_id]
        return item.mapRectToView(item.boundingRect())

    def show_stack(self, drawn, content_of) -> dict:
        """Redraw every row of ``drawn`` (bottom first ``(row, visible, opacity)``) from
        ``content_of(row_id) -> RowContent``; returns ``{row_id: reason}`` for rows not drawn."""
        keep = {row.id for row, _v, _o in drawn}
        for rid in [r for r in self._items if r not in keep]:
            self.view.removeItem(self._items.pop(rid))
        for rid in [r for r in self._vectors if r not in keep]:
            for item in self._vectors.pop(rid):
                self.view.removeItem(item)
        self._reasons = {}
        self._frame = None
        for z, (row, visible, opacity) in enumerate(drawn):
            self.update_row(row.id, content_of(row.id), opacity=opacity if visible else 0.0, z=z)
        return self.reasons()

    def update_row(self, row_id, content, opacity: float = 1.0, z=None) -> None:
        self._reasons.pop(row_id, None)
        if opacity == 0.0:                   # hidden in the stack: not drawn, nothing to explain
            self._hide(row_id, None)
            return
        if content is None or content.reason:
            self._hide(row_id, content.reason if content is not None else "no content")
            return
        if content.kind in ("raster", "product", "channel"):
            self._draw_raster(row_id, content, opacity, z)

    def _hide(self, row_id, reason) -> None:
        if reason:
            self._reasons[row_id] = reason
        for item in (self._items.get(row_id), *self._vectors.get(row_id, ())):
            if item is not None:
                item.setVisible(False)

    def _placeable(self, field):
        """``(mode, reason)``: ``"affine"``, ``"warp"`` or ``None`` with the reason."""
        crs = field_crs(field)
        if self._crs is None:
            frame = getattr(field, "frame", None)
            if self._frame is None:
                self._frame = frame
                return "affine", None
            from dynamix.geo.placement import same_frame_family

            return ("affine", None) if same_frame_family(frame, self._frame) \
                else (None, "different frame")
        if crs is None:
            return None, "no CRS"
        return ("affine", None) if crs == self._crs else ("warp", None)

    def _draw_raster(self, row_id, content, opacity, z) -> None:
        field = content.field
        mode, reason = self._placeable(field)
        if mode is None:
            self._hide(row_id, reason)
            return
        item = self._items.get(row_id)
        if item is None:
            item = self._items[row_id] = pg.ImageItem()
            self.view.addItem(item)
        painter = self._painters.setdefault(row_id, _RowPainter())
        painter.apply_style(content.style, content.composite)
        values = np.asarray(getattr(field, "values", field), dtype=np.float64)
        s = lod_stride(values.shape[:2], MAX_DIM)
        if mode == "affine":
            painter._paint(item, field, s)
            x0, y0, sx, sy = image_transform(field, s)
        else:
            key = (row_id, id(values), self._crs)
            if key not in self._warp_cache:
                plane = values if values.ndim == 2 else values[..., 0]
                try:
                    self._warp_cache[key] = reproject_for_display(
                        plane[::s, ::s], field, self._crs, s=s, max_dim=MAX_DIM)
                except Exception as exc:                             # noqa: BLE001
                    self._hide(row_id, f"reprojection failed: {exc}")
                    return
            image, (x0, y0, sx, sy) = self._warp_cache[key]
            painter._paint(item, image, 1)
        item.setTransform(QtGui.QTransform(sx, 0.0, 0.0, sy, x0, y0))
        item.setOpacity(opacity)
        if z is not None:
            item.setZValue(z)
        item.setVisible(True)
