# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The wavelet scale bar: the analysing wavelet's section at the level on show, drawn to scale
on the image (its width in data units), its height a fixed share of that width on screen above a
floor (so a zoom keeps the curve's proportions instead of stretching it flat), a dimension bracket exactly
λ* wide carrying the label with the physical length, green over a black stroke. Anchored to a
corner of the view in screen pixels, so it never clips, and draggable; the Map's distance bar
drags the same way. A right-click menu sets a bar's units, colour, outline and line width (its
style, remembered by the window). Display only."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.shell.canvas import ANALYZER_COLOR
from dynamix.shell.theme import RESTRAINED_DARK
from dynamix.shell.units import UNIT_CHOICES

CURVE_HEIGHT_PX = 28.0          # the curve's smallest peak-to-peak excursion on screen
CURVE_ASPECT = 0.25             # and, zoomed in, its excursion as a share of its width on screen
BRACKET_GAP_PX = 6.0            # between the curve's lowest point and the bracket
TICK_PX = 5.0                   # the bracket's end ticks
TAIL_FRACTION = 0.01            # the curve stops where |ψ| falls below this of its peak
CAP_LAMBDAS = 2.0               # and never runs past ±2λ*
_GREEN = QtGui.QColor(*ANALYZER_COLOR)
_BLACK = QtGui.QColor(0, 0, 0)

LINE_WIDTHS = {"thin": 0.8, "normal": 1.4, "bold": 2.4}
DEFAULT_WAVELET_STYLE = {"units": "auto", "color": _GREEN.name(), "outline": True,
                         "width": "normal"}
DEFAULT_SCALE_STYLE = {"units": "auto", "color": QtGui.QColor(RESTRAINED_DARK.ink).name(),
                       "outline": False, "width": "normal"}


def merged_style(style: dict, changes: dict) -> dict:
    """``style`` with the valid entries of ``changes`` applied (unknown keys and values are
    ignored, never stored)."""
    out = dict(style)
    changes = changes or {}
    if changes.get("units") in UNIT_CHOICES:
        out["units"] = changes["units"]
    if changes.get("width") in LINE_WIDTHS:
        out["width"] = changes["width"]
    if "outline" in changes:
        out["outline"] = bool(changes["outline"])
    colour = QtGui.QColor(str(changes.get("color", "")))
    if colour.isValid():
        out["color"] = colour.name()
    return out


def style_menu(style: dict, on_change, parent=None) -> QtWidgets.QMenu:
    """The bars' right-click menu: Units, Colour…, Outline, Line width; every pick calls
    ``on_change({key: value})``."""
    menu = QtWidgets.QMenu(parent)
    units = menu.addMenu("Units")
    group = QtGui.QActionGroup(units)
    for choice in UNIT_CHOICES:
        act = units.addAction(choice)
        act.setCheckable(True)
        act.setChecked(style["units"] == choice)
        group.addAction(act)
        act.triggered.connect(lambda _checked=False, c=choice: on_change({"units": c}))

    def pick_colour():
        colour = QtWidgets.QColorDialog.getColor(QtGui.QColor(style["color"]), parent,
                                                 "Bar colour")
        if colour.isValid():
            on_change({"color": colour.name()})

    menu.addAction("Colour…").triggered.connect(pick_colour)
    outline = menu.addAction("Outline")
    outline.setCheckable(True)
    outline.setChecked(bool(style["outline"]))
    outline.triggered.connect(lambda checked: on_change({"outline": bool(checked)}))
    widths = menu.addMenu("Line width")
    wgroup = QtGui.QActionGroup(widths)
    for name in LINE_WIDTHS:
        act = widths.addAction(name)
        act.setCheckable(True)
        act.setChecked(style["width"] == name)
        wgroup.addAction(act)
        act.triggered.connect(lambda _checked=False, n=name: on_change({"width": n}))
    return menu


class _Styled:
    """A bar with a style -- units, colour, outline, line width -- and a right-click menu. The
    host defines ``styleChanged`` (emitted on a menu pick, never on ``set_style``) and may
    override ``_apply_style``."""

    def _init_style(self, default: dict) -> None:
        self._style = dict(default)
        self._menu = None

    def style(self) -> dict:
        return dict(self._style)

    def colour(self) -> QtGui.QColor:
        return QtGui.QColor(self._style["color"])

    def set_style(self, changes: dict) -> None:
        """Apply ``changes`` (a partial style) without telling anyone."""
        self._style = merged_style(self._style, changes)
        self._apply_style()

    def _apply_style(self) -> None:
        self.update()

    def _change_style(self, changes: dict) -> None:
        self.set_style(changes)
        self.styleChanged.emit(self.style())

    def context_menu(self, parent=None) -> QtWidgets.QMenu:
        self._menu = style_menu(self._style, self._change_style, parent)
        return self._menu

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.context_menu().popup(ev.screenPos().toPoint())


class _Drag:
    """A corner-anchored item the pointer can drag. The remembered offset is the user's; when a
    zoom makes the item wider than the room it has, it is shifted into the view for that zoom
    only, so a zoom in and back out lands it where it started. The host defines ``moved``."""

    def _init_drag(self, corner=(1, 1), offset=(-20.0, -20.0), item_anchor=None) -> None:
        self._corner = (float(corner[0]), float(corner[1]))
        # The point of the item that sits on the corner: (0, 1) is the bottom of a rect
        # symmetric about x = 0, the bar's and the cascade's centre, so they grow symmetrically.
        self._item_anchor = (tuple(float(v) for v in item_anchor) if item_anchor is not None
                             else self._corner)
        self._offset = [float(offset[0]), float(offset[1])]
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton
                                     | QtCore.Qt.MouseButton.RightButton)

    def offset(self) -> tuple:
        return tuple(self._offset)

    def set_offset(self, offset) -> None:
        """Put the item where the user last left it (a remembered offset)."""
        self._offset = [float(offset[0]), float(offset[1])]
        self.place()

    def _anchor_at(self, offset) -> None:
        pg.GraphicsWidgetAnchor.anchor(self, self._item_anchor, self._corner, tuple(offset))

    def _overflow(self) -> tuple:
        """``(dx, dy)`` that brings the item a pixel inside its view; zeros when it is."""
        view = self.parentItem()
        if view is None:
            return 0.0, 0.0
        inside = view.boundingRect().adjusted(1.0, 1.0, -1.0, -1.0)
        mine = self.mapRectToParent(self.boundingRect())
        if mine.isEmpty() or inside.isEmpty():
            return 0.0, 0.0

        def shift(lo, hi, in_lo, in_hi):
            # An item larger than the view cannot keep both edges inside: it keeps its centre
            # inside instead (the edge rule would flip it between two positions).
            if hi - lo > in_hi - in_lo:
                centre = 0.5 * (lo + hi)
                return min(max(centre, in_lo), in_hi) - centre
            return max(0.0, in_lo - lo) - max(0.0, hi - in_hi)

        dx = shift(mine.left(), mine.right(), inside.left(), inside.right())
        dy = shift(mine.top(), mine.bottom(), inside.top(), inside.bottom())
        return dx, dy

    def place(self) -> None:
        if self.parentItem() is None:
            return
        self._anchor_at(self._offset)
        dx, dy = self._overflow()
        if dx or dy:
            self._anchor_at((self._offset[0] + dx, self._offset[1] + dy))

    def _clamp(self) -> None:
        self.place()

    def drag_by(self, dx: float, dy: float) -> None:
        self._offset[0] += float(dx)
        self._offset[1] += float(dy)
        if self.parentItem() is not None:         # a drag cannot leave the item off the view
            self._anchor_at(self._offset)
            ox, oy = self._overflow()
            self._offset[0] += ox
            self._offset[1] += oy
        self.place()
        self.moved.emit(tuple(self._offset))

    def mouseDragEvent(self, ev) -> None:
        ev.accept()
        d = ev.pos() - ev.lastPos()
        self.drag_by(d.x(), d.y())


class WaveletBar(_Drag, _Styled, pg.GraphicsWidgetAnchor, pg.GraphicsObject):
    moved = QtCore.Signal(object)
    styleChanged = QtCore.Signal(object)

    def __init__(self, corner=(1, 1), offset=(-110.0, -20.0)) -> None:
        pg.GraphicsObject.__init__(self)
        pg.GraphicsWidgetAnchor.__init__(self)
        self._init_drag(corner, offset, item_anchor=(0.0, 1.0))     # its centre on the corner
        self._init_style(DEFAULT_WAVELET_STYLE)
        self._profile = None
        self._dx_data = 1.0
        self.label = ""
        self._curve = QtGui.QPainterPath()
        self._text = QtGui.QPainterPath()
        self._bracket = QtCore.QLineF()
        self._rect = QtCore.QRectF()
        self._curve_x_px = np.empty(0)
        self._amplitude = CURVE_HEIGHT_PX
        self._font = QtGui.QFont()
        self._font.setBold(True)
        self._font.setPointSizeF(10.0)
        self.hide()

    # -- content ----------------------------------------------------------------------------
    def set_profile(self, profile, label: str, dx_data: float = 1.0) -> None:
        """``profile`` (a :class:`dynamix.core.wavelet_profiles.Profile`) with ``label`` and the
        view's data units per image pixel along x; ``None`` hides the bar."""
        self._profile, self.label, self._dx_data = profile, str(label), float(dx_data)
        if profile is None:
            self.hide()
            return
        self.setToolTip(self._tooltip())
        self._rebuild()
        self.show()
        self.place()

    def _tooltip(self) -> str:
        p = self._profile
        parts = [p.family]
        for key, text in (("a", "a"), ("sigma", "σ"), ("zero_crossing", "zero crossing")):
            if key in p.extra:
                parts.append(f"{text} {float(p.extra[key]):.3g} px")
        if np.isfinite(p.lambda_px):
            parts.append(f"λ* {p.lambda_px:.3g} px")
        if p.broad:
            parts.append("broad band")
        if p.off_ladder:
            parts.append("off the dyadic ladder")
        return " · ".join(parts)

    def changeParent(self) -> None:
        view = self.parentItem()
        if view is None:
            return
        view.sigRangeChanged.connect(self._rebuild)
        view.sigResized.connect(self._rebuild)      # a resize changes the pixels per data unit
        self._rebuild()
        self.place()

    def _px_per_data(self) -> float:
        view = self.parentItem()
        if view is None:
            return 1.0
        p0 = view.mapFromViewToItem(self, QtCore.QPointF(0.0, 0.0))
        p1 = view.mapFromViewToItem(self, QtCore.QPointF(1.0, 0.0))
        return abs(p1.x() - p0.x()) or 1.0

    def _rebuild(self, *_args) -> None:
        p = self._profile
        if p is None:
            return
        self.prepareGeometryChange()
        k = self._px_per_data() * self._dx_data             # screen px per image pixel
        xs, ys, self._curve_x_px, self._amplitude = _section(p, k)
        self._curve = QtGui.QPainterPath()
        if xs.size:
            self._curve.moveTo(xs[0], ys[0])
            for xi, yi in zip(xs[1:], ys[1:]):
                self._curve.lineTo(xi, yi)
        # The bracket hangs under the curve's lowest point (a Gaussian never dips below its base).
        y_bracket = (self._curve.boundingRect().bottom() if xs.size else 0.0) + BRACKET_GAP_PX
        half = 0.5 * p.lambda_px * k if np.isfinite(p.lambda_px) else 0.0
        self._bracket = QtCore.QLineF(-half, y_bracket, half, y_bracket)
        self._text = QtGui.QPainterPath()
        metrics = QtGui.QFontMetricsF(self._font)
        width = metrics.horizontalAdvance(self.label)
        self._text.addText(-0.5 * width, y_bracket + TICK_PX + metrics.ascent() + 2.0,
                           self._font, self.label)
        rect = self._curve.boundingRect().united(self._text.boundingRect())
        rect = rect.united(QtCore.QRectF(-half, y_bracket - TICK_PX, 2 * half, 2 * TICK_PX))
        self._rect = rect.adjusted(-3.0, -3.0, 3.0, 3.0)
        self.update()
        self.place()

    # -- readings for tests and tooltips ----------------------------------------------------
    def bracket_px(self) -> float:
        return float(self._bracket.length())

    def curve_x_px(self) -> np.ndarray:
        return self._curve_x_px

    def curve_width_px(self) -> float:
        return float(self._curve.boundingRect().width())

    def amplitude_px(self) -> float:
        return float(self._amplitude)

    # -- QGraphicsItem ----------------------------------------------------------------------
    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self._rect)

    def paint(self, painter, *_args) -> None:
        if self._profile is None:
            return
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        for pen in _stroke_pens(self._style):
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPath(self._curve)
            if self._bracket.length() > 0:
                painter.drawLine(self._bracket)
                for xe in (self._bracket.x1(), self._bracket.x2()):
                    painter.drawLine(QtCore.QLineF(xe, self._bracket.y1() - TICK_PX,
                                                   xe, self._bracket.y1() + TICK_PX))
        _draw_text(painter, self._text, self._style)


def _stroke_pens(style: dict) -> tuple:
    """The pens a bar's curve and bracket are stroked with: a black outline under the colour
    when the style asks for one."""
    width = LINE_WIDTHS[style["width"]]
    colour = QtGui.QPen(QtGui.QColor(style["color"]), width)
    if not style["outline"]:
        return (colour,)
    return (QtGui.QPen(_BLACK, width + 1.6), colour)


def _draw_text(painter, path, style: dict) -> None:
    if style["outline"]:
        painter.setPen(QtGui.QPen(_BLACK, 2.5))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(style["color"]))
    painter.drawPath(path)


class DraggableScaleBar(_Drag, _Styled, pg.ScaleBar):
    """pyqtgraph's distance bar, draggable; ``anchor(corner, corner, offset)`` places it and
    ``moved`` reports the offset the user left it at. Its style colours the bar and its text;
    its units choice is read by whoever writes the text."""

    moved = QtCore.Signal(object)
    styleChanged = QtCore.Signal(object)

    def __init__(self, size, width=5, brush=None, pen=None, suffix="m", corner=(1, 1),
                 offset=(-20.0, -20.0)) -> None:
        pg.ScaleBar.__init__(self, size, width=width, brush=brush, pen=pen, suffix=suffix,
                             offset=None)
        self._init_drag(corner, offset)
        self._init_style(DEFAULT_SCALE_STYLE)
        self._apply_style()

    def _apply_style(self) -> None:
        colour = self.colour()
        self.bar.setPen(pg.mkPen(colour, width=LINE_WIDTHS[self._style["width"]]))
        self.bar.setBrush(pg.mkBrush(colour))
        self.text.setColor(colour)
        self.update()

    def anchor(self, itemPos, parentPos, offset=(0, 0)) -> None:
        self._corner = (float(itemPos[0]), float(itemPos[1]))
        self._offset = [float(offset[0]), float(offset[1])]
        pg.GraphicsWidgetAnchor.anchor(self, itemPos, parentPos, offset)

    def setParentItem(self, p):
        ret = pg.GraphicsObject.setParentItem(self, p)
        if p is not None:
            pg.GraphicsWidgetAnchor.anchor(self, self._corner, self._corner,
                                           tuple(self._offset))
        return ret

    def boundingRect(self) -> QtCore.QRectF:
        rect = QtCore.QRectF(self.bar.rect())
        return rect.united(self.text.mapRectToParent(self.text.boundingRect()))


def _section(profile, k: float):
    """``(x_screen, y_screen, x_px, amplitude)`` of a profile's drawn section: |ψ| ≥ 1 % of its
    peak and |x| ≤ 2λ*, at ``k`` screen pixels per image pixel; the amplitude is CURVE_ASPECT of
    the section's width on screen, never below CURVE_HEIGHT_PX."""
    x, psi = np.asarray(profile.x_px, dtype=np.float64), np.asarray(profile.psi, dtype=np.float64)
    peak = float(np.max(np.abs(psi))) if psi.size else 0.0
    keep = np.abs(psi) >= TAIL_FRACTION * peak if peak > 0 else np.zeros(psi.shape, bool)
    if np.isfinite(profile.lambda_px):
        keep &= np.abs(x) <= CAP_LAMBDAS * profile.lambda_px
    xs = x[keep] * k
    width = float(xs.max() - xs.min()) if xs.size else 0.0
    amplitude = max(CURVE_HEIGHT_PX, CURVE_ASPECT * width)
    return xs, -0.5 * amplitude * psi[keep], x[keep], amplitude


class WaveletCascade(_Drag, _Styled, pg.GraphicsWidgetAnchor, pg.GraphicsObject):
    """A microcanonical tool's whole ladder of wavelets, nested on one centre over the raster:
    every scale's section to scale, the level on show at full strength and the rest faint,
    with a bracket and a label for the smallest and the largest. Draggable like the bar."""

    moved = QtCore.Signal(object)
    styleChanged = QtCore.Signal(object)

    def __init__(self, corner=(0, 1), offset=(180.0, -44.0)) -> None:
        pg.GraphicsObject.__init__(self)
        pg.GraphicsWidgetAnchor.__init__(self)
        self._init_drag(corner, offset, item_anchor=(0.0, 1.0))     # its centre on the corner
        self._init_style(DEFAULT_WAVELET_STYLE)
        self._profiles: list = []
        self.labels: tuple = ("", "")
        self.shown = 0
        self._paths: list = []
        self._widths: list = []
        self._amps: list = []
        self._brackets: list = []
        self._texts: list = []
        self._rect = QtCore.QRectF()
        self._font = QtGui.QFont()
        self._font.setBold(True)
        self._font.setPointSizeF(9.0)
        self.hide()

    def set_profiles(self, profiles, labels, shown: int = 0) -> None:
        """The ladder's profiles (smallest first), the labels of its two ends, and the index
        of the level on show; an empty list hides the cascade."""
        self._profiles = list(profiles)
        self.labels = (str(labels[0]), str(labels[1])) if len(labels) >= 2 else ("", "")
        self.shown = int(shown)
        if not self._profiles:
            self.hide()
            return
        self.setToolTip(" · ".join(p.family for p in self._profiles[:1])
                        + f" · {len(self._profiles)} scales")
        self._rebuild()
        self.show()
        self.place()

    def changeParent(self) -> None:
        view = self.parentItem()
        if view is None:
            return
        view.sigRangeChanged.connect(self._rebuild)
        view.sigResized.connect(self._rebuild)
        self._rebuild()
        self.place()

    def _px_per_data(self) -> float:
        view = self.parentItem()
        if view is None:
            return 1.0
        p0 = view.mapFromViewToItem(self, QtCore.QPointF(0.0, 0.0))
        p1 = view.mapFromViewToItem(self, QtCore.QPointF(1.0, 0.0))
        return abs(p1.x() - p0.x()) or 1.0

    def _rebuild(self, *_args) -> None:
        if not self._profiles:
            return
        self.prepareGeometryChange()
        k = self._px_per_data()
        self._paths, self._widths, self._amps = [], [], []
        rect = QtCore.QRectF()
        for p in self._profiles:
            xs, ys, _x_px, amp = _section(p, k)
            self._amps.append(amp)
            path = QtGui.QPainterPath()
            if xs.size:
                path.moveTo(xs[0], ys[0])
                for xi, yi in zip(xs[1:], ys[1:]):
                    path.lineTo(xi, yi)
            self._paths.append(path)
            self._widths.append(float(xs.max() - xs.min()) if xs.size else 0.0)
            rect = rect.united(path.boundingRect())
        metrics = QtGui.QFontMetricsF(self._font)
        self._brackets, self._texts = [], []
        y = max((p.boundingRect().bottom() for p in self._paths), default=0.0) + BRACKET_GAP_PX
        for p, label in zip((self._profiles[0], self._profiles[-1]), self.labels):
            half = 0.5 * p.lambda_px * k if np.isfinite(p.lambda_px) else 0.0
            self._brackets.append(QtCore.QLineF(-half, y, half, y))
            text = QtGui.QPainterPath()
            text.addText(-0.5 * metrics.horizontalAdvance(label),
                         y + TICK_PX + metrics.ascent() + 2.0, self._font, label)
            self._texts.append(text)
            rect = rect.united(text.boundingRect())
            rect = rect.united(QtCore.QRectF(-half, y - TICK_PX, 2 * half, 2 * TICK_PX))
            y += TICK_PX + metrics.height() + 10.0
        self._rect = rect.adjusted(-3.0, -3.0, 3.0, 3.0)
        self.update()
        self.place()

    def widths_px(self) -> list:
        return list(self._widths)

    def amplitudes_px(self) -> list:
        return list(self._amps)

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self._rect)

    def paint(self, painter, *_args) -> None:
        if not self._profiles:
            return
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        colour, width = self.colour(), LINE_WIDTHS[self._style["width"]]
        faint, shadow = QtGui.QColor(colour), QtGui.QColor(_BLACK)
        faint.setAlpha(190)
        shadow.setAlpha(110)
        faint_pens = [QtGui.QPen(faint, 0.8 * width)]
        if self._style["outline"]:
            faint_pens.insert(0, QtGui.QPen(shadow, 0.8 * width + 1.3))
        for i, path in enumerate(self._paths):
            if i == self.shown:
                continue
            for pen in faint_pens:
                pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawPath(path)
        if 0 <= self.shown < len(self._paths):
            for pen in _stroke_pens(self._style):
                pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawPath(self._paths[self.shown])
        for bracket in self._brackets:
            if bracket.length() <= 0:
                continue
            for pen in _stroke_pens(self._style):
                painter.setPen(pen)
                painter.drawLine(bracket)
                for xe in (bracket.x1(), bracket.x2()):
                    painter.drawLine(QtCore.QLineF(xe, bracket.y1() - TICK_PX,
                                                   xe, bracket.y1() + TICK_PX))
        for text in self._texts:
            _draw_text(painter, text, self._style)
