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
from dynamix.shell.canvas import (ARROW_COLOR, Canvas, _arrow_segments, hline_polylines,
                                  lod_stride, nice_round_scalebar)

MAX_DIM = 2048
#: Ground metres in one degree along the equator (WGS84 semi-major axis).
_M_PER_DEG = np.pi * 6378137.0 / 180.0


_BRUSHES: list = []


def _colour_index(scalars) -> np.ndarray:
    """[0, 1] modulus scalars as indices into a 256-entry table; a non-finite scalar (a zero
    modulus under a negative power) lands on an end of the table instead of outside it."""
    k = np.nan_to_num(np.asarray(scalars, dtype=np.float64) * 255, nan=0.0, posinf=255.0,
                      neginf=0.0)
    return np.clip(k.astype(int), 0, 255)


def _viridis_brushes() -> np.ndarray:
    """The 256 viridis brushes, built once, as an object array a per-point index picks from."""
    if not _BRUSHES:
        lut = pg.colormap.get("viridis").getLookupTable(nPts=256)
        brushes = np.empty(256, dtype=object)
        brushes[:] = [pg.mkBrush(*c) for c in lut]
        _BRUSHES.append(brushes)
    return _BRUSHES[0].copy()


class _Capture:
    """What ``Canvas._paint`` hands an ImageItem, kept to resample: the painted picture in
    pyqtgraph's (x, y[, channel]) orientation, its ``setImage`` keywords and its colormap."""

    def __init__(self) -> None:
        self.image, self.kw, self.cmap = None, {}, None

    def setImage(self, image, **kw) -> None:
        self.image, self.kw = image, kw

    def setColorMap(self, cmap) -> None:
        self.cmap = cmap


def _resample(image, index):
    """``image`` (pyqtgraph orientation, painted on the strided source grid) gathered through
    ``index`` (destination pixel -> flat source index, ``-1`` outside the source)."""
    src = image.transpose(1, 0, 2) if image.ndim == 3 else image.T
    flat = src.reshape(src.shape[0] * src.shape[1], -1)
    out = flat[np.where(index >= 0, index, 0)]
    outside = index < 0
    if out.dtype.kind == "f":
        out[outside] = np.nan
    elif out.shape[-1] == 4:
        out[outside, 3] = 0
    return out.transpose(1, 0, 2) if image.ndim == 3 else out[..., 0].T


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
        #: What each row last drew: a row whose content and style did not move is not redrawn
        #: (a live preview updates the selected layer's rows on every tick).
        self._drawn: dict = {}
        #: Ground metres per map unit: None for degrees (latitude-dependent), a number for a
        #: linear CRS or frame, 0.0 for pixel axes (no ground distance).
        self._unit_metres = 1.0
        self._selected_row = None
        self._fallback_row = None
        self._overlay_items: list = []
        #: ``blocks`` False draws an extrema row as the Raster tab does (H-line polylines, dots,
        #: arrows at the WTMMM); True draws xsmurf's blocks image, which the other keys style.
        self.extrema_style = {"blocks": False, "colour": "modulus", "mode": "linear", "q": 1.0,
                              "subpixel": False, "ticks": True, "heads": False}
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self._on_clicked)
        from dynamix.shell.wavelet_bar import DraggableScaleBar, WaveletBar

        self.scale_bar = DraggableScaleBar(size=1000, suffix="m")
        self.scale_bar.setParentItem(self.view)
        self.scale_bar.anchor((1, 1), (1, 1), offset=(-20, -20))
        self.view.sigRangeChanged.connect(self._update_scale_bar)
        #: The wavelet scale bar of the selected layer (set_wavelet_profile), above the
        #: distance bar.
        self.wavelet_bar = WaveletBar(offset=(-20.0, -60.0))
        self.wavelet_bar.setParentItem(self.view)

    def set_map_crs(self, crs) -> None:
        self._crs = str(crs) if crs else None
        self._warp_cache.clear()
        self._drawn.clear()
        if self._crs is not None:
            from rasterio.crs import CRS

            try:
                crs_obj = CRS.from_user_input(self._crs)
                self._unit_metres = (None if crs_obj.is_geographic
                                     else float(crs_obj.linear_units_factor[1] or 1.0))
            except Exception:                                        # noqa: BLE001
                self._unit_metres = 1.0       # a CRS PROJ cannot read: the bar reads map units
        self._update_scale_bar()

    def _frame_unit_metres(self, frame):
        """Ground metres per unit of a frame placed without a map CRS."""
        from dynamix.core.frames import GeographicFrame

        if isinstance(frame, GeographicFrame):
            return None
        units = str(getattr(frame, "units", "") or "").lower()
        if units in ("m", "metre", "metres", "meter", "meters"):
            return 1.0
        if units in ("km", "kilometre", "kilometres", "kilometer", "kilometers"):
            return 1000.0
        if units in ("ft", "foot", "feet"):
            return 0.3048
        if units in ("us-ft", "us_survey_foot", "ftus"):
            return 1200.0 / 3937.0
        return 0.0

    def _update_scale_bar(self, *_args) -> None:
        """A bar a round ground distance long, about a quarter of the view: degrees convert at
        the view centre's latitude (along a parallel); pixel axes read pixels."""
        (x0, x1), (y0, y1) = self.view.viewRange()
        width = x1 - x0
        if not width > 0:
            return
        per_unit = self._unit_metres
        if per_unit is None:
            lat = float(np.clip((y0 + y1) / 2.0, -89.0, 89.0))
            per_unit = _M_PER_DEG * np.cos(np.radians(lat))
        if per_unit and per_unit > 0:
            _label, metres = nice_round_scalebar(width * per_unit)
            size = metres / per_unit
            text = f"{metres / 1000:g} km" if metres >= 1000 else f"{metres:g} m"
        else:
            label, size = nice_round_scalebar(width)
            text = f"{label} px"
        self.scale_bar.size = size
        self.scale_bar.text.setText(text)
        self.scale_bar.updateBar()

    def set_wavelet_profile(self, profile, label: str, dx_data: float = 1.0) -> None:
        """The wavelet scale bar: ``profile`` of the selected layer's shown level with its
        label; ``dx_data`` the layer's map units per image pixel; ``None`` hides it."""
        self.wavelet_bar.set_profile(profile, label, dx_data)

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
            self._drawn.pop(rid, None)
            self._warp_cache.pop(rid, None)
            self._painters.pop(rid, None)
        for rid in [r for r in self._vectors if r not in keep]:
            for item in self._vectors.pop(rid):
                self.view.removeItem(item)
            self._drawn.pop(rid, None)
            self._chain_rows.pop(rid, None)
        self._reasons = {}
        self._frame = None
        for z, (row, visible, opacity) in enumerate(drawn):
            self.update_row(row.id, content_of(row.id), opacity=opacity if visible else 0.0, z=z)
        if self._crs is None:
            self._unit_metres = (self._frame_unit_metres(self._frame) if self._frame is not None
                                 else 1.0)
            self._update_scale_bar()
        return self.reasons()

    def _unchanged(self, row_id, signature, items) -> bool:
        """Whether ``row_id`` already draws ``signature`` (objects compared by identity, the
        rest by value); ``items`` then only take the new opacity and order."""
        previous = self._drawn.get(row_id)
        if previous is None or len(previous) != len(signature) or not items:
            return False
        return all((a == b) if isinstance(a, (str, int, float, type(None))) else (a is b)
                   for a, b in zip(previous, signature))

    def _settle(self, items, opacity, z) -> None:
        for it in items:
            it.setOpacity(opacity)
            if z is not None:
                it.setZValue(z)
            it.setVisible(True)

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
        elif content.kind == "outline":
            self._draw_outline(row_id, content, opacity, z)
        elif content.result.get("_roi_axes") is None and \
                getattr(content.grid_field, "x_axis", None) is None:
            self._hide(row_id, "no coordinates")          # a bare array places nowhere
        elif content.kind == "extrema":
            self._draw_extrema(row_id, content, opacity, z)
        elif content.kind == "chains":
            self._draw_chains(row_id, content, opacity, z)

    def set_row_opacity(self, row_id, opacity: float) -> None:
        """Fade a drawn row's items to ``opacity`` (a layer-stack slider tick)."""
        for item in (self._items.get(row_id), *self._vectors.get(row_id, ())):
            if item is not None:
                item.setOpacity(opacity)

    def _hide(self, row_id, reason) -> None:
        if reason:
            self._reasons[row_id] = reason
        for item in (self._items.get(row_id), *self._vectors.get(row_id, ())):
            if item is not None:
                item.setVisible(False)

    def _placeable(self, field):
        """``(mode, reason)``: ``"affine"``, ``"warp"`` or ``None`` with the reason."""
        if getattr(field, "x_axis", None) is None or getattr(field, "y_axis", None) is None:
            return None, "no coordinates"                  # a bare array places nowhere
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

    def _warp_map(self, raw, field, s):
        """Where each pixel of the Map's picture of ``field`` reads in its values decimated by
        ``s``: the flat source index per destination pixel (``-1`` outside), and the picture's
        transform. One reprojection per row and array; the painted picture is gathered
        through it, so composite, hillshade and levels draw as the affine path draws them."""
        ny, nx = np.shape(raw)[:2]
        shape = ((ny + s - 1) // s, (nx + s - 1) // s)
        index = np.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape)
        warped, xform = reproject_for_display(index, field, self._crs, s=s, max_dim=MAX_DIM)
        return np.where(np.isfinite(warped), warped, -1).astype(np.intp), xform

    def _draw_raster(self, row_id, content, opacity, z) -> None:
        field = content.field
        mode, reason = self._placeable(field)
        if mode is None:
            self._hide(row_id, reason)
            return
        signature = (field, repr(sorted((content.style or {}).items())),
                     repr(sorted((content.composite or {}).items())), self._crs)
        if self._unchanged(row_id, signature, [self._items.get(row_id)]):
            self._settle([self._items[row_id]], opacity, z)
            return
        item = self._items.get(row_id)
        if item is None:
            item = self._items[row_id] = pg.ImageItem()
            self.view.addItem(item)
        painter = self._painters.setdefault(row_id, _RowPainter())
        painter.apply_style(content.style, content.composite)
        raw = getattr(field, "values", field)
        s = lod_stride(np.shape(raw)[:2], MAX_DIM)
        if mode == "affine":
            painter._paint(item, field, s)
            x0, y0, sx, sy = image_transform(field, s)
        else:
            cached = self._warp_cache.get(row_id)
            if cached is None or cached[0] is not raw or cached[1] != self._crs:
                try:
                    cached = (raw, self._crs, *self._warp_map(raw, field, s))
                except Exception as exc:                             # noqa: BLE001
                    self._hide(row_id, f"reprojection failed: {exc}")
                    return
                self._warp_cache[row_id] = cached
            _raw, _crs, index, (x0, y0, sx, sy) = cached
            capture = _Capture()
            painter._paint(capture, field, s)      # on its own grid, as the affine path paints
            item.setImage(_resample(capture.image, index), **capture.kw)
            if capture.cmap is not None:
                item.setColorMap(capture.cmap)
        item.setTransform(QtGui.QTransform(sx, 0.0, 0.0, sy, x0, y0))
        self._drawn[row_id] = signature
        self._settle([item], opacity, z)

    # -- extrema and chains -------------------------------------------------------------------
    def _grid(self, content):
        from dynamix.core.derivative import result_grid

        _frame, x_axis, y_axis = result_grid(content.result, content.grid_field)
        return x_axis, y_axis

    def _extrema_items(self, row_id, image_mode):
        """An extrema row's (blocks, ticks): blocks an image or a scatter, rebuilt when the
        row changes between the two."""
        want = pg.ImageItem if image_mode else pg.ScatterPlotItem
        items = self._vectors.get(row_id)
        if items is not None and not isinstance(items[0], want):
            for it in items:
                self.view.removeItem(it)
            items = None
        if items is None:
            items = (want(), pg.PlotDataItem(connect="pairs"))
            for it in items:
                self.view.addItem(it)
            self._vectors[row_id] = items
        return items

    @staticmethod
    def _extrema_image(result, s, ext, cols, rows, mod, st, nx, ny) -> np.ndarray:
        """``(nx, ny, 4)`` RGBA, indexed (column, row): each maximum lights its own cell,
        coloured by modulus (viridis), by its chain's group, or white."""
        from dynamix.geo.extrema_draw import group_rgb, modulus_scalars

        n = len(cols)
        if st["colour"] == "flat":
            rgb = np.full((n, 3), 255, dtype=np.uint8)
        else:
            lut = pg.colormap.get("viridis").getLookupTable(nPts=256)[:, :3]
            k = _colour_index(modulus_scalars(mod, st["mode"], st["q"]))
            rgb = np.asarray(lut, dtype=np.uint8)[k]
            if st["colour"] == "group":
                own = group_rgb(result.get("chains"), s, cols, rows)
                grouped = own[:, 0] >= 0
                rgb[grouped] = own[grouped].astype(np.uint8)
        img = np.zeros((nx, ny, 4), dtype=np.uint8)
        ci = np.clip(np.rint(cols).astype(np.intp), 0, nx - 1)
        ri = np.clip(np.rint(rows).astype(np.intp), 0, ny - 1)
        img[ci, ri, :3] = rgb
        img[ci, ri, 3] = 255
        return img

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

    def _line_items(self, row_id):
        """An extrema row's (H-lines, isolated dots, arrows), rebuilt when the row changes
        from the blocks drawing."""
        items = self._vectors.get(row_id)
        if items is not None and not (len(items) == 3 and isinstance(items[0], pg.PlotDataItem)):
            for it in items:
                self.view.removeItem(it)
            items = None
        if items is None:
            items = (pg.PlotDataItem(connect="finite"), pg.ScatterPlotItem(pxMode=True),
                     pg.PlotDataItem(connect="pairs", pen=pg.mkPen(ARROW_COLOR, width=1.2)))
            for it in items:
                self.view.addItem(it)
            self._vectors[row_id] = items
        return items

    def _to_map_nan(self, field, x, y):
        """:meth:`_to_map` for NaN-separated polylines: the separators stay NaN."""
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.all():
            return self._to_map(field, x, y)
        mx, my = np.full(x.shape, np.nan), np.full(y.shape, np.nan)
        if ok.any():
            mx[ok], my[ok] = self._to_map(field, x[ok], y[ok])
        return mx, my

    def _draw_extrema(self, row_id, content, opacity, z) -> None:
        """An extrema row as the Raster tab draws it: every H-line a polyline in the layer's
        H-chain colour, isolated maxima as dots, and arrows along the argument at the WTMMM
        (or at every maximum, or none: the layer's own arrow mode)."""
        from dynamix.geo.extrema_draw import to_axes

        result, s = content.result, content.scale_idx
        levels = result.get("extrema") or []
        if not levels:
            self._hide(row_id, "no extrema")
            return
        if self.extrema_style.get("blocks"):
            self._draw_extrema_blocks(row_id, content, opacity, z)
            return
        ext = levels[min(s, len(levels) - 1)]
        style = content.style or {}
        mode = style.get("arrows", "wtmmm")
        signature = (ext, int(s), content.grid_field, result.get("_roi_axes"), mode,
                     repr(sorted(style.items())), self._crs)
        items = self._line_items(row_id)
        if self._unchanged(row_id, signature, list(items)):
            self._settle(items, opacity, z)
            return
        lines, dots, arrows = items
        x_axis, y_axis = self._grid(content)
        shape = tuple(result.get("_shape") or (len(y_axis), len(x_axis)))[:2]
        hx, hy = hline_polylines(ext, shape)
        lines.setData(*self._to_map_nan(content.grid_field, *to_axes(hx, hy, x_axis, y_axis)),
                      pen=pg.mkPen(style.get("color_hchain", "#ebebeb"),
                                   width=float(style.get("line_width", 1.0))))
        n = len(ext["x"])
        iso = np.asarray(ext.get("line_id", np.full(n, -1))) == -1
        cx = np.asarray(ext.get("x_sub", ext["x"]), dtype=np.float64)[iso]
        cy = np.asarray(ext.get("y_sub", ext["y"]), dtype=np.float64)[iso]
        dx, dy = self._to_map(content.grid_field, *to_axes(cx, cy, x_axis, y_axis))
        dots.setData(x=dx, y=dy, pen=None, size=float(style.get("point_size", 3.0)),
                     brush=pg.mkBrush(style.get("color_extrema", "#c8c8c8")))
        ax, ay = self._arrow_xy(result, ext, s, mode, x_axis, y_axis, content.grid_field)
        arrows.setData(ax, ay)
        self._settle(items, opacity, z)
        self._drawn[row_id] = signature

    def _arrow_xy(self, result, ext, s, mode, x_axis, y_axis, grid_field):
        """Map coordinates of the argument arrows at the WTMMM (``mode`` ``"wtmmm"``) or at
        every maximum (``"all"``): the Raster tab's own length law, half the scale (capped)
        at full strength down to 30 % for the weakest modulus."""
        from dynamix.core.anisotropy import wtmmm_by_scale
        from dynamix.geo.extrema_draw import to_axes

        if mode == "off":
            return [], []
        pts = wtmmm_by_scale([ext])[0] if mode == "wtmmm" else ext
        px = np.asarray(pts.get("x_sub", pts["x"]), dtype=np.float64)
        py = np.asarray(pts.get("y_sub", pts["y"]), dtype=np.float64)
        if px.size == 0:
            return [], []
        arg = np.asarray(pts.get("arg", np.zeros(px.size)), dtype=np.float64)
        mod = np.asarray(pts.get("mod", np.ones(px.size)), dtype=np.float64)
        scales = result.get("scales")
        a_idx = int(result.get("_scale_idx", s))
        a = (float(np.asarray(scales)[a_idx])
             if scales is not None and 0 <= a_idx < len(scales) else 4.0)
        ref = float(np.nanpercentile(mod, 95)) if mod.size else 1.0
        base = float(np.clip(0.5 * a, 1.5, 12.0))
        lengths = base * (0.3 + 0.7 * np.clip(mod / (ref if ref > 0 else 1.0), 0.0, 1.0))
        xs, ys = _arrow_segments(px, py, arg, lengths)
        return self._to_map(grid_field, *to_axes(xs, ys, x_axis, y_axis))

    def _draw_extrema_blocks(self, row_id, content, opacity, z) -> None:
        """The xsmurf drawing: every maximum its own block, an argument tick from each,
        coloured by modulus or chain group (``extrema_style["blocks"]``)."""
        from dynamix.geo.extrema_draw import (block_positions, group_rgb, modulus_scalars,
                                              tick_ends, to_axes)

        result, s = content.result, content.scale_idx
        levels = result.get("extrema") or []
        ext = levels[min(s, len(levels) - 1)]
        st = self.extrema_style
        # The extrema themselves, not the result dict around them: a lazy output landing hands
        # over a copy of the result with the same extrema.
        signature = (ext, int(s), content.grid_field, result.get("_roi_axes"),
                     result.get("chains") if st["colour"] == "group" else None,
                     repr(sorted(st.items())), self._crs)
        if self._unchanged(row_id, signature, list(self._vectors.get(row_id, ()))):
            blocks, ticks = self._vectors[row_id]
            self._settle([blocks], opacity, z)
            ticks.setOpacity(opacity)
            if z is not None:
                ticks.setZValue(z)
            ticks.setVisible(bool(st["ticks"] and "arg" in ext))
            return
        cols, rows = block_positions(ext, subpixel=st["subpixel"])
        x_axis, y_axis = self._grid(content)
        bx, by = self._to_map(content.grid_field, *to_axes(cols, rows, x_axis, y_axis))
        mod = np.asarray(ext.get("mod", np.ones(len(cols))))
        # On its own grid in the map's CRS the maxima are one image, a pixel per maximum, as
        # the Raster tab draws them: a single texture to pan. Subpixel placement, or a grid
        # the Map reprojects, keeps one square per maximum.
        image_mode = (not st["subpixel"] and len(x_axis) > 1 and len(y_axis) > 1
                      and (self._crs is None
                           or field_crs(content.grid_field) in (None, self._crs)))
        blocks, ticks = self._extrema_items(row_id, image_mode)
        if image_mode:
            blocks.setImage(self._extrema_image(result, s, ext, cols, rows, mod, st,
                                                len(x_axis), len(y_axis)), autoLevels=False)
            dx = (float(x_axis[-1]) - float(x_axis[0])) / (len(x_axis) - 1)
            dy = (float(y_axis[-1]) - float(y_axis[0])) / (len(y_axis) - 1)
            blocks.setTransform(QtGui.QTransform(dx, 0.0, 0.0, dy, float(x_axis[0]) - 0.5 * dx,
                                                 float(y_axis[0]) - 0.5 * dy))
        elif st["colour"] == "flat":
            brushes = pg.mkBrush(255, 255, 255)
        else:
            if st["colour"] == "group":
                rgb = group_rgb(result.get("chains"), s, cols, rows)
            else:
                rgb = np.full((len(cols), 3), -1)
            k = _colour_index(modulus_scalars(mod, st["mode"], st["q"]))
            brushes = _viridis_brushes()[k]            # one brush per colour, indexed per point
            grouped = np.flatnonzero(rgb[:, 0] >= 0)
            if grouped.size:
                own = {}
                for i in grouped:
                    key = tuple(int(v) for v in rgb[i])
                    brushes[i] = own.setdefault(key, pg.mkBrush(*key))
            brushes = list(brushes)
        if not image_mode:
            pixel = abs(float(x_axis[1] - x_axis[0])) if len(x_axis) > 1 else 1.0
            blocks.setData(x=bx, y=by, symbol="s", size=pixel, pxMode=False, pen=None,
                           brush=brushes)
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
        self._drawn[row_id] = signature

    def _draw_outline(self, row_id, content, opacity, z) -> None:
        """An ROI's outline row: its box (``content.result["roi"] = (row, col, h, w)`` in the
        pixels of ``content.field``'s axes) on the pixel edges, dashed."""
        from dynamix.geo.extrema_draw import to_axes

        r, c, h, w = content.result["roi"]
        field = content.field
        cols = np.array([c, c + w, c + w, c, c]) - 0.5
        rows = np.array([r, r, r + h, r + h, r]) - 0.5
        x, y = self._to_map(field, *to_axes(cols, rows, field.x_axis, field.y_axis))
        items = self._vectors.get(row_id)
        if items is None:
            items = (pg.PlotDataItem(pen=pg.mkPen("#e8a33d", style=QtCore.Qt.DashLine)),)
            self.view.addItem(items[0])
            self._vectors[row_id] = items
        items[0].setData(x, y)
        items[0].setOpacity(opacity)
        if z is not None:
            items[0].setZValue(z)
        items[0].setVisible(True)

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
        signature = (result.get("chains"), result.get("chains_excluded"), content.grid_field,
                     result.get("_roi_axes"), bool(result.get("_show_ghosts", True)), self._crs)
        if self._unchanged(row_id, signature, [trail]):
            self._settle([trail], opacity, z)
            for it in (ghosts, picked):
                it.setOpacity(opacity)
                if z is not None:
                    it.setZValue(z)
            ghosts.setVisible(bool(result.get("chains_excluded"))
                              and result.get("_show_ghosts", True))
            return
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
        self._drawn[row_id] = signature

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
    def set_selected_row(self, row_id, fallback=None) -> None:
        """The row the readout reads: ``row_id`` when it is drawn, else ``fallback`` (a
        layer's dataset row, for a layer with no product row)."""
        self._selected_row = row_id
        self._fallback_row = fallback

    def _row_at(self, x, y):
        """The topmost visible raster row whose image covers map point ``(x, y)``."""
        hits = [(item.zValue(), rid) for rid, item in self._items.items()
                if item.isVisible() and self.item_rect(rid).normalized().contains(x, y)]
        return max(hits)[1] if hits else None

    def _readout_at(self, x, y) -> str:
        """Map coordinates, and the selected row's PAINTED value there (a display value: the
        LUT input of a plain raster; the Raster tab reads exact native values)."""
        text = f"{x:.10g}, {y:.10g}"
        for rid in (self._selected_row, self._fallback_row):
            item = self._items.get(rid)
            if item is None or not item.isVisible() or item.image is None:
                continue
            p = item.mapFromView(QtCore.QPointF(x, y))
            i, j = int(np.floor(p.x())), int(np.floor(p.y()))
            img = item.image
            if img.ndim == 2 and 0 <= i < img.shape[0] and 0 <= j < img.shape[1]:
                text += f"  ·  {rid} = {img[i, j]:.6g}"
            break
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
            xs, ys = [], []                   # one item per layer, its parts NaN-separated
            for feature in layer.features:
                for part in feature.parts:
                    part = np.asarray(part, dtype=np.float64)
                    xs.extend([*part[:, 0], np.nan])
                    ys.extend([*part[:, 1], np.nan])
            if not xs:
                continue
            item = pg.PlotDataItem(np.asarray(xs[:-1]), np.asarray(ys[:-1]), connect="finite",
                                   pen=pg.mkPen("#cccccc"))
            item.setZValue(top)
            self.view.addItem(item)
            self._overlay_items.append(item)
        for x, y in points:
            item = pg.ScatterPlotItem(x=x, y=y, size=4)
            item.setZValue(top)
            self.view.addItem(item)
            self._overlay_items.append(item)
