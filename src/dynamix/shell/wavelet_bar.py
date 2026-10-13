# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The wavelet scale bar: the analysing wavelet's section at the level on show, drawn to scale
on the image (its width in data units), a screen-constant height, a dimension bracket exactly
λ* wide carrying the label with the physical length, green over a black stroke. Anchored to a
corner of the view in screen pixels, so it never clips, and draggable; the Map's distance bar
drags the same way. Display only."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

from dynamix.shell.canvas import ANALYZER_COLOR

CURVE_HEIGHT_PX = 28.0          # the curve's peak-to-peak excursion on screen
BRACKET_GAP_PX = 6.0            # between the curve's lowest point and the bracket
TICK_PX = 5.0                   # the bracket's end ticks
TAIL_FRACTION = 0.01            # the curve stops where |ψ| falls below this of its peak
CAP_LAMBDAS = 2.0               # and never runs past ±2λ*
_GREEN = QtGui.QColor(*ANALYZER_COLOR)
_BLACK = QtGui.QColor(0, 0, 0)


class _Drag:
    """A corner-anchored item the pointer can drag: the anchor offset follows the drag, clamped
    so the item stays inside its view. The host defines ``moved``."""

    def _init_drag(self, corner=(1, 1), offset=(-20.0, -20.0)) -> None:
        self._corner = (float(corner[0]), float(corner[1]))
        self._offset = [float(offset[0]), float(offset[1])]
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)

    def offset(self) -> tuple:
        return tuple(self._offset)

    def set_offset(self, offset) -> None:
        """Put the item where the user last left it (a remembered offset)."""
        self._offset = [float(offset[0]), float(offset[1])]
        self.place()

    def place(self) -> None:
        if self.parentItem() is not None:
            self.anchor(itemPos=self._corner, parentPos=self._corner, offset=tuple(self._offset))
            self._clamp()

    def _clamp(self) -> None:
        view = self.parentItem()
        if view is None:
            return
        inside = view.boundingRect().adjusted(1.0, 1.0, -1.0, -1.0)   # a pixel in from the edge
        mine = self.mapRectToParent(self.boundingRect())
        if mine.isEmpty() or inside.isEmpty():
            return
        dx = max(0.0, inside.left() - mine.left()) - max(0.0, mine.right() - inside.right())
        dy = max(0.0, inside.top() - mine.top()) - max(0.0, mine.bottom() - inside.bottom())
        if dx or dy:
            self._offset[0] += dx
            self._offset[1] += dy
            self.anchor(itemPos=self._corner, parentPos=self._corner, offset=tuple(self._offset))

    def drag_by(self, dx: float, dy: float) -> None:
        self._offset[0] += float(dx)
        self._offset[1] += float(dy)
        self.place()
        self.moved.emit(tuple(self._offset))

    def mouseDragEvent(self, ev) -> None:
        ev.accept()
        d = ev.pos() - ev.lastPos()
        self.drag_by(d.x(), d.y())


class WaveletBar(_Drag, pg.GraphicsWidgetAnchor, pg.GraphicsObject):
    moved = QtCore.Signal(object)

    def __init__(self, corner=(1, 1), offset=(-20.0, -20.0)) -> None:
        pg.GraphicsObject.__init__(self)
        pg.GraphicsWidgetAnchor.__init__(self)
        self._init_drag(corner, offset)
        self._profile = None
        self._dx_data = 1.0
        self.label = ""
        self._curve = QtGui.QPainterPath()
        self._text = QtGui.QPainterPath()
        self._bracket = QtCore.QLineF()
        self._rect = QtCore.QRectF()
        self._curve_x_px = np.empty(0)
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
        x, psi = np.asarray(p.x_px, dtype=np.float64), np.asarray(p.psi, dtype=np.float64)
        peak = float(np.max(np.abs(psi))) if psi.size else 0.0
        keep = np.abs(psi) >= TAIL_FRACTION * peak if peak > 0 else np.zeros(psi.shape, bool)
        if np.isfinite(p.lambda_px):
            keep &= np.abs(x) <= CAP_LAMBDAS * p.lambda_px
        self._curve_x_px = x[keep]
        xs, ys = self._curve_x_px * k, -0.5 * CURVE_HEIGHT_PX * psi[keep]
        self._curve = QtGui.QPainterPath()
        if xs.size:
            self._curve.moveTo(xs[0], ys[0])
            for xi, yi in zip(xs[1:], ys[1:]):
                self._curve.lineTo(xi, yi)
        y_bracket = 0.5 * CURVE_HEIGHT_PX + BRACKET_GAP_PX
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

    # -- QGraphicsItem ----------------------------------------------------------------------
    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self._rect)

    def paint(self, painter, *_args) -> None:
        if self._profile is None:
            return
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        for pen in (QtGui.QPen(_BLACK, 3.0), QtGui.QPen(_GREEN, 1.4)):
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
        painter.setPen(QtGui.QPen(_BLACK, 2.5))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPath(self._text)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(_GREEN)
        painter.drawPath(self._text)


class DraggableScaleBar(_Drag, pg.ScaleBar):
    """pyqtgraph's distance bar, draggable; ``anchor(corner, corner, offset)`` places it and
    ``moved`` reports the offset the user left it at."""

    moved = QtCore.Signal(object)

    def __init__(self, size, width=5, brush=None, pen=None, suffix="m", corner=(1, 1),
                 offset=(-20.0, -20.0)) -> None:
        pg.ScaleBar.__init__(self, size, width=width, brush=brush, pen=pen, suffix=suffix,
                             offset=None)
        self._init_drag(corner, offset)

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
    """``(x_screen, y_screen, x_px)`` of a profile's drawn section: |ψ| ≥ 1 % of its peak and
    |x| ≤ 2λ*, at ``k`` screen pixels per image pixel."""
    x, psi = np.asarray(profile.x_px, dtype=np.float64), np.asarray(profile.psi, dtype=np.float64)
    peak = float(np.max(np.abs(psi))) if psi.size else 0.0
    keep = np.abs(psi) >= TAIL_FRACTION * peak if peak > 0 else np.zeros(psi.shape, bool)
    if np.isfinite(profile.lambda_px):
        keep &= np.abs(x) <= CAP_LAMBDAS * profile.lambda_px
    return x[keep] * k, -0.5 * CURVE_HEIGHT_PX * psi[keep], x[keep]


class WaveletCascade(_Drag, pg.GraphicsWidgetAnchor, pg.GraphicsObject):
    """A microcanonical tool's whole ladder of wavelets, nested on one centre over the raster:
    every scale's section to scale, the level on show at full strength and the rest faint,
    with a bracket and a label for the smallest and the largest. Draggable like the bar."""

    moved = QtCore.Signal(object)

    def __init__(self, corner=(0, 1), offset=(20.0, -44.0)) -> None:
        pg.GraphicsObject.__init__(self)
        pg.GraphicsWidgetAnchor.__init__(self)
        self._init_drag(corner, offset)
        self._profiles: list = []
        self.labels: tuple = ("", "")
        self.shown = 0
        self._paths: list = []
        self._widths: list = []
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
        self._paths, self._widths = [], []
        rect = QtCore.QRectF()
        for p in self._profiles:
            xs, ys, _x_px = _section(p, k)
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
        y = 0.5 * CURVE_HEIGHT_PX + BRACKET_GAP_PX
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

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self._rect)

    def paint(self, painter, *_args) -> None:
        if not self._profiles:
            return
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        faint, shadow = QtGui.QColor(_GREEN), QtGui.QColor(_BLACK)
        faint.setAlpha(190)
        shadow.setAlpha(110)
        for i, path in enumerate(self._paths):
            if i == self.shown:
                continue
            for pen in (QtGui.QPen(shadow, 2.4), QtGui.QPen(faint, 1.1)):
                pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawPath(path)
        if 0 <= self.shown < len(self._paths):
            for pen in (QtGui.QPen(_BLACK, 3.0), QtGui.QPen(_GREEN, 1.4)):
                pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawPath(self._paths[self.shown])
        for bracket in self._brackets:
            if bracket.length() <= 0:
                continue
            for pen in (QtGui.QPen(_BLACK, 3.0), QtGui.QPen(_GREEN, 1.2)):
                painter.setPen(pen)
                painter.drawLine(bracket)
                for xe in (bracket.x1(), bracket.x2()):
                    painter.drawLine(QtCore.QLineF(xe, bracket.y1() - TICK_PX,
                                                   xe, bracket.y1() + TICK_PX))
        for text in self._texts:
            painter.setPen(QtGui.QPen(_BLACK, 2.5))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPath(text)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(_GREEN)
            painter.drawPath(text)
