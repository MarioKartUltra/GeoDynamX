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
        self._chain_rows: dict = {}
        self._selected_row = None
        self._overlay_items: list = []
        self.extrema_style = {"colour": "modulus", "mode": "linear", "q": 1.0,
                              "subpixel": False, "ticks": True, "heads": False}
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self._on_clicked)
        self.scale_bar = pg.ScaleBar(size=1000, suffix="m")
        self.scale_bar.setParentItem(self.view)
        self.scale_bar.anchor((1, 1), (1, 1), offset=(-20, -20))

    def set_map_crs(self, crs) -> None:
        self._crs = str(crs) if crs else None
        self._warp_cache.clear()
        if self._crs is not None:
            from rasterio.crs import CRS

            geographic = CRS.from_user_input(self._crs).is_geographic
            size, suffix = (1.0, "°") if geographic else (1000.0, "m")
            self.scale_bar.size = size
            self.scale_bar.text.setText(pg.siFormat(size, suffix=suffix))
            self.scale_bar.updateBar()

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
        elif content.kind == "extrema":
            self._draw_extrema(row_id, content, opacity, z)
        elif content.kind == "chains":
            self._draw_chains(row_id, content, opacity, z)

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

    # -- extrema and chains -------------------------------------------------------------------
    def _grid(self, content):
        from dynamix.core.derivative import result_grid

        _frame, x_axis, y_axis = result_grid(content.result, content.grid_field)
        return x_axis, y_axis

    def _vector_items(self, row_id, n):
        items = self._vectors.get(row_id)
        if items is None:
            items = tuple(pg.ScatterPlotItem() if i == 0 else pg.PlotDataItem(connect="pairs")
                          for i in range(n))
            for it in items:
                self.view.addItem(it)
            self._vectors[row_id] = items
        return items

    def _to_map(self, field, x, y):
        """Grid coordinates of ``field`` -> map coordinates (reprojected points when the CRS
        differs from the map's)."""
        crs = field_crs(field)
        if self._crs is None or crs is None or crs == self._crs:
            return np.asarray(x), np.asarray(y)
        from rasterio.warp import transform

        tx, ty = transform(crs, self._crs, list(np.asarray(x)), list(np.asarray(y)))
        return np.asarray(tx), np.asarray(ty)

    def _draw_extrema(self, row_id, content, opacity, z) -> None:
        from dynamix.geo.extrema_draw import (block_positions, group_rgb, modulus_scalars,
                                              tick_ends, to_axes)

        result, s = content.result, content.scale_idx
        levels = result.get("extrema") or []
        if not levels:
            self._hide(row_id, "no extrema")
            return
        ext = levels[min(s, len(levels) - 1)]
        st = self.extrema_style
        cols, rows = block_positions(ext, subpixel=st["subpixel"])
        x_axis, y_axis = self._grid(content)
        bx, by = self._to_map(content.grid_field, *to_axes(cols, rows, x_axis, y_axis))
        mod = np.asarray(ext.get("mod", np.ones(len(cols))))
        if st["colour"] == "flat":
            brushes = pg.mkBrush(255, 255, 255)
        else:
            if st["colour"] == "group":
                rgb = group_rgb(result.get("chains"), s, cols, rows)
            else:
                rgb = np.full((len(cols), 3), -1)
            lut = pg.colormap.get("viridis").getLookupTable(nPts=256)
            k = (modulus_scalars(mod, st["mode"], st["q"]) * 255).astype(int)
            brushes = [pg.mkBrush(*(c if c[0] >= 0 else lut[i])) for c, i in zip(rgb, k)]
        pixel = abs(float(x_axis[1] - x_axis[0])) if len(x_axis) > 1 else 1.0
        blocks, ticks = self._vector_items(row_id, 2)
        blocks.setData(x=bx, y=by, symbol="s", size=pixel, pxMode=False, pen=None, brush=brushes)
        if st["ticks"] and "arg" in ext:
            c1, r1 = tick_ends(cols, rows, ext["arg"], mod, float(np.max(np.abs(mod))))
            ex, ey = self._to_map(content.grid_field, *to_axes(c1, r1, x_axis, y_axis))
            ticks.setData(np.column_stack([bx, ex]).ravel(), np.column_stack([by, ey]).ravel())
            ticks.setVisible(True)
        else:
            ticks.setVisible(False)
        for it in (blocks, ticks):
            it.setOpacity(opacity)
            if z is not None:
                it.setZValue(z)
        blocks.setVisible(True)

    def _trail_xy(self, content, chains):
        from dynamix.geo.extrema_draw import to_axes

        x_axis, y_axis = self._grid(content)
        xs, ys = [], []
        for chain in chains:
            cx, cy = to_axes(np.asarray(chain["x"], float), np.asarray(chain["y"], float),
                             x_axis, y_axis)
            xs.extend([*cx, np.nan])
            ys.extend([*cy, np.nan])
        return self._to_map(content.grid_field, np.asarray(xs), np.asarray(ys))

    def _draw_chains(self, row_id, content, opacity, z) -> None:
        """Chains as trails across scales; excluded chains dashed (the Raster tab's ghosts,
        gated by ``_show_ghosts``); a third item holds the picked chains (``set_highlight``)."""
        items = self._vectors.get(row_id)
        if items is None:
            items = (pg.PlotDataItem(connect="finite"),
                     pg.PlotDataItem(connect="finite",
                                     pen=pg.mkPen("#888888", style=QtCore.Qt.DashLine)),
                     pg.PlotDataItem(connect="finite", pen=pg.mkPen("#ffd400", width=2)))
            for it in items:
                self.view.addItem(it)
            self._vectors[row_id] = items
        trail, ghosts, picked = items
        result = content.result
        self._chain_rows[row_id] = (content, result.get("chains") or [])
        trail.setData(*self._trail_xy(content, result.get("chains") or []))
        excluded = result.get("chains_excluded") or []
        if excluded and result.get("_show_ghosts", True):
            ghosts.setData(*self._trail_xy(content, excluded))
            ghosts.setVisible(True)
        else:
            ghosts.setVisible(False)
        picked.setVisible(False)
        for it in items:
            it.setOpacity(opacity)
            if z is not None:
                it.setZValue(z)
        trail.setVisible(True)

    def set_highlight(self, row_id, chain_indices) -> None:
        """Draw the chains at ``chain_indices`` of a chains row in the highlight colour (the
        Raster tab's picked chains); ``None`` or empty clears it."""
        items = self._vectors.get(row_id)
        if items is None or len(items) < 3:
            return
        content, chains = self._chain_rows.get(row_id, (None, []))
        picked = items[2]
        if not chain_indices or content is None:
            picked.setVisible(False)
            return
        picked.setData(*self._trail_xy(content, [chains[i] for i in chain_indices
                                                 if 0 <= i < len(chains)]))
        picked.setVisible(True)

    # -- interaction --------------------------------------------------------------------------
    def set_selected_row(self, row_id) -> None:
        self._selected_row = row_id

    def _row_at(self, x, y):
        """The topmost visible raster row whose image covers map point ``(x, y)``."""
        hits = [(item.zValue(), rid) for rid, item in self._items.items()
                if item.isVisible() and self.item_rect(rid).normalized().contains(x, y)]
        return max(hits)[1] if hits else None

    def _readout_at(self, x, y) -> str:
        """Map coordinates, and the selected row's PAINTED value there (a display value: the
        LUT input of a plain raster; the Raster tab reads exact native values)."""
        text = f"{x:.10g}, {y:.10g}"
        item = self._items.get(self._selected_row)
        if item is not None and item.isVisible() and item.image is not None:
            p = item.mapFromView(QtCore.QPointF(x, y))
            i, j = int(np.floor(p.x())), int(np.floor(p.y()))
            img = item.image
            if img.ndim == 2 and 0 <= i < img.shape[0] and 0 <= j < img.shape[1]:
                text += f"  ·  {self._selected_row} = {img[i, j]:.6g}"
        return text

    def _on_mouse_moved(self, pos) -> None:
        p = self.view.mapSceneToView(pos)
        self.readout.setText(self._readout_at(p.x(), p.y()))

    def _on_clicked(self, event) -> None:
        p = self.view.mapSceneToView(event.scenePos())
        rid = self._row_at(p.x(), p.y())
        if rid is not None:
            self.rowClicked.emit(rid)

    # -- overlays outside the stack -----------------------------------------------------------
    def set_overlays(self, rois=(), references=(), points=()) -> None:
        """ROI outlines (``(field, (row, col, h, w), label)`` in the field's pixels), reference
        layers (``VectorLayer`` already in the map CRS) and point sets (``(x, y)`` in the map
        CRS), drawn above every stack row."""
        from dynamix.geo.extrema_draw import to_axes

        for item in self._overlay_items:
            self.view.removeItem(item)
        self._overlay_items = []
        top = 1e6
        for field, (r, c, h, w), _label in rois:
            cols = np.array([c, c + w, c + w, c, c]) - 0.5
            rows = np.array([r, r, r + h, r + h, r]) - 0.5
            x, y = self._to_map(field, *to_axes(cols, rows, field.x_axis, field.y_axis))
            item = pg.PlotDataItem(x, y, pen=pg.mkPen("#e8a33d", style=QtCore.Qt.DashLine))
            item.setZValue(top)
            self.view.addItem(item)
            self._overlay_items.append(item)
        for layer in references:
            for feature in layer.features:
                for part in feature.parts:
                    item = pg.PlotDataItem(part[:, 0], part[:, 1], pen=pg.mkPen("#cccccc"))
                    item.setZValue(top)
                    self.view.addItem(item)
                    self._overlay_items.append(item)
        for x, y in points:
            item = pg.ScatterPlotItem(x=x, y=y, size=4)
            item.setZValue(top)
            self.view.addItem(item)
            self._overlay_items.append(item)
