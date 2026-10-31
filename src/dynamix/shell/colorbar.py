# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The colour bar: the raster's ramp with its values, an overlay on the canvas.

The canvas hands the bar what it painted (:meth:`dynamix.shell.canvas.Canvas.colorbar_content`):
the strip's colours from its low end to its high end, the ticks as positions along it with their
text, and the title. The bar holds no data, so the picture and the bar beside it come from one
computation, and a figure rendered from the scene (``QWidget.grab``, pyqtgraph's Export…) carries
the bar as the screen shows it. It sits on one side of the view, centred along it, at the
distance scale bar's margin; tick labels face into the view and drop where they would overlap.
A right-click menu sets the side, the size, the text colour, the outline and the title. Display
only."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.shell.theme import current

SIDES = ("right", "left", "top", "bottom")
#: Per size: the strip's length as a share of the view along it, and its thickness in screen px.
SIZES = {"small": (0.3, 10.0), "medium": (0.45, 14.0), "large": (0.6, 18.0)}
#: ``color`` "" follows the theme's scale-bar role.
DEFAULT_COLORBAR_STYLE = {"side": "right", "size": "medium", "color": "", "outline": True}
MARGIN_PX = 12.0        # from the view's edge: the distance scale bar's margin
TICK_PX = 4.0           # a tick mark's length
GAP_PX = 3.0            # between a mark and its label, and between the labels and the title
_BLACK = QtGui.QColor(0, 0, 0)


def merged_colorbar_style(style: dict, changes: dict) -> dict:
    """``style`` with the valid entries of ``changes`` applied; anything else is ignored."""
    out = dict(style)
    changes = changes or {}
    if changes.get("side") in SIDES:
        out["side"] = changes["side"]
    if changes.get("size") in SIZES:
        out["size"] = changes["size"]
    if "outline" in changes:
        out["outline"] = bool(changes["outline"])
    if "color" in changes:
        colour = str(changes["color"] or "")
        if not colour:
            out["color"] = ""
        elif QtGui.QColor(colour).isValid():
            out["color"] = QtGui.QColor(colour).name()
    return out


class ColorBar(pg.GraphicsObject):
    #: A menu pick changed the style (never emitted by :meth:`set_style`).
    styleChanged = QtCore.Signal(object)
    #: The title typed in the menu's Title… ("" asks for the automatic one).
    labelEdited = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._style = dict(DEFAULT_COLORBAR_STYLE)
        self._colours = np.zeros((0, 4), dtype=np.uint8)
        self._ticks: list = []              # (position in [0, 1], text, priority)
        self._labelled: list = []           # the (position, text) whose text is drawn
        self.label = ""
        self._image = QtGui.QImage()
        self._strip = QtCore.QRectF()
        self._marks: list = []
        self._text = QtGui.QPainterPath()
        self._rect = QtCore.QRectF()
        self._view = None
        self._menu = None
        self._tick_font = QtGui.QFont()
        self._tick_font.setPointSizeF(9.0)
        self._title_font = QtGui.QFont()
        self._title_font.setBold(True)
        self._title_font.setPointSizeF(10.0)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.RightButton)
        self.hide()

    # -- content and style -------------------------------------------------------------------
    def set_content(self, colours, ticks, label: str) -> None:
        """The strip's RGBA bytes ``(n, 4)`` from its low end to its high end, the ticks
        ``(position, text, priority)`` (a higher priority keeps its label first), the title."""
        self._colours = np.ascontiguousarray(np.asarray(colours, dtype=np.uint8).reshape(-1, 4))
        self._ticks = [(float(p), str(t), int(r)) for p, t, r in ticks
                       if np.isfinite(p) and 0.0 <= float(p) <= 1.0]
        self.label = str(label)
        self._rebuild()

    def style(self) -> dict:
        return dict(self._style)

    def set_style(self, changes: dict) -> None:
        """Apply ``changes`` (a partial style) without telling anyone."""
        self._style = merged_colorbar_style(self._style, changes)
        self._rebuild()

    def _change_style(self, changes: dict) -> None:
        self.set_style(changes)
        self.styleChanged.emit(self.style())

    def colour(self) -> QtGui.QColor:
        return QtGui.QColor(self._style["color"] or current().scale_bar)

    # -- readings for tests --------------------------------------------------------------------
    def ticks(self) -> list:
        return [(p, t) for p, t, _r in self._ticks]

    def labelled(self) -> list:
        return list(self._labelled)

    def colours(self) -> np.ndarray:
        return self._colours.copy()

    def vertical(self) -> bool:
        return self._style["side"] in ("left", "right")

    def strip_scene_point(self, position: float) -> QtCore.QPointF:
        """The scene point on the strip's centre line at ``position`` (0 its low end)."""
        r = self._strip
        if self.vertical():
            local = QtCore.QPointF(r.center().x(), r.bottom() - position * r.height())
        else:
            local = QtCore.QPointF(r.left() + position * r.width(), r.center().y())
        return self.mapToScene(local)

    # -- layout ----------------------------------------------------------------------------------
    def changeParent(self) -> None:
        super().changeParent()
        view = self.parentItem()
        if view is not None and view is not self._view and hasattr(view, "sigResized"):
            view.sigResized.connect(self._rebuild)      # the length is a share of the view
            self._view = view
        self._rebuild()

    def _fit_labels(self, length: float, sizes: list) -> list:
        """Indices of the ticks whose labels fit, highest priority first, then in order: a label
        is kept when it clears every kept one along the strip."""
        kept = []
        for i in sorted(range(len(self._ticks)), key=lambda i: (-self._ticks[i][2], i)):
            at = self._ticks[i][0] * length
            if all(abs(at - self._ticks[j][0] * length) >= 0.5 * (sizes[i] + sizes[j]) + 2.0
                   for j in kept):
                kept.append(i)
        return sorted(kept)

    def _rebuild(self, *_args) -> None:
        view = self.parentItem()
        if view is None or not self._colours.size:
            return
        self.prepareGeometryChange()
        area = view.boundingRect()
        share, thick = SIZES[self._style["size"]]
        vertical = self.vertical()
        length = max(40.0, share * (area.height() if vertical else area.width()))
        tm, hm = QtGui.QFontMetricsF(self._tick_font), QtGui.QFontMetricsF(self._title_font)
        widths = [tm.horizontalAdvance(t) for _p, t, _r in self._ticks]
        keep = self._fit_labels(length, [tm.height()] * len(widths) if vertical else widths)
        self._labelled = [(self._ticks[i][0], self._ticks[i][1]) for i in keep]
        label_w = max((widths[i] for i in keep), default=0.0)
        title_w = hm.horizontalAdvance(self.label) if self.label else 0.0
        title_h = hm.height() + GAP_PX if self.label else 0.0
        text = QtGui.QPainterPath()
        marks = []
        side = self._style["side"]
        if vertical:
            width = max(thick + TICK_PX + GAP_PX + label_w, title_w)
            x0 = width - thick if side == "right" else 0.0
            y0 = title_h + 0.5 * tm.height()
            strip = QtCore.QRectF(x0, y0, thick, length)
            inner = -1.0 if side == "right" else 1.0
            edge = x0 if side == "right" else x0 + thick
            for p, t, _r in self._ticks:
                y = y0 + (1.0 - p) * length
                marks.append(QtCore.QLineF(edge, y, edge + inner * TICK_PX, y))
            for p, t in self._labelled:
                y = y0 + (1.0 - p) * length + 0.5 * (tm.ascent() - tm.descent())
                w = tm.horizontalAdvance(t)
                x = (edge - TICK_PX - GAP_PX - w) if side == "right" else edge + TICK_PX + GAP_PX
                text.addText(x, y, self._tick_font, t)
            if self.label:
                text.addText(width - title_w if side == "right" else 0.0, hm.ascent(),
                             self._title_font, self.label)
            height = y0 + length + 0.5 * tm.height()
        else:
            pad = 0.5 * label_w
            width = max(length + 2.0 * pad, title_w)
            x0 = 0.5 * (width - length)
            labels_h = tm.height() + GAP_PX
            if side == "bottom":                 # labels and title above the strip
                y0 = title_h + labels_h + TICK_PX
                mark_to, base = y0 - TICK_PX, title_h + tm.ascent()
                title_base = hm.ascent()
            else:                                # labels and title below the strip
                y0 = 0.0
                mark_to, base = y0 + thick + TICK_PX, y0 + thick + TICK_PX + GAP_PX + tm.ascent()
                title_base = y0 + thick + TICK_PX + labels_h + GAP_PX + hm.ascent()
            strip = QtCore.QRectF(x0, y0, length, thick)
            edge = y0 if side == "bottom" else y0 + thick
            for p, t, _r in self._ticks:
                x = x0 + p * length
                marks.append(QtCore.QLineF(x, edge, x, mark_to))
            for p, t in self._labelled:
                text.addText(x0 + p * length - 0.5 * tm.horizontalAdvance(t), base,
                             self._tick_font, t)
            if self.label:
                text.addText(0.5 * (width - title_w), title_base, self._title_font, self.label)
            height = (y0 + thick if side == "bottom"
                      else title_base + hm.descent() if self.label
                      else base + tm.descent())
        rgba = self._colours[::-1] if vertical else self._colours
        shape = (rgba.shape[0], 1) if vertical else (1, rgba.shape[0])
        buffer = np.ascontiguousarray(rgba)
        self._image = QtGui.QImage(buffer.data, shape[1], shape[0], 4 * shape[1],
                                   QtGui.QImage.Format.Format_RGBA8888).copy()
        self._strip, self._marks, self._text = strip, marks, text
        self._rect = QtCore.QRectF(0.0, 0.0, width, height).adjusted(-3.0, -3.0, 3.0, 3.0)
        self.update()
        self.place()

    def content_rect(self) -> QtCore.QRectF:
        """The strip, its labels and its title in the bar's own coordinates, without the margin
        the bounding rect keeps for the outline."""
        return self._rect.adjusted(3.0, 3.0, -3.0, -3.0)

    def place(self) -> None:
        """On the chosen side of the view, centred along it, ``MARGIN_PX`` in from its edge."""
        view = self.parentItem()
        if view is None:
            return
        area = view.boundingRect()
        rect = self.content_rect()
        side = self._style["side"]
        if side in ("left", "right"):
            x = area.right() - MARGIN_PX - rect.width() if side == "right" else area.left() + MARGIN_PX
            y = area.center().y() - self._strip.center().y()
        else:
            x = area.center().x() - self._strip.center().x()
            y = area.bottom() - MARGIN_PX - rect.height() if side == "bottom" else area.top() + MARGIN_PX
        self.setPos(x, y)

    # -- QGraphicsItem -----------------------------------------------------------------------
    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self._rect)

    def paint(self, painter, *_args) -> None:
        if self._image.isNull():
            return
        painter.drawImage(self._strip, self._image)          # nearest: blocks keep their edges
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        colour = self.colour()
        pens = [QtGui.QPen(colour, 1.0)]
        if self._style["outline"]:
            pens.insert(0, QtGui.QPen(_BLACK, 2.6))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        for pen in pens:
            painter.setPen(pen)
            painter.drawRect(self._strip)
            for mark in self._marks:
                painter.drawLine(mark)
        if self._style["outline"]:
            painter.setPen(QtGui.QPen(_BLACK, 2.5))
            painter.drawPath(self._text)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        painter.drawPath(self._text)

    # -- menu ------------------------------------------------------------------------------------
    def context_menu(self, parent=None) -> QtWidgets.QMenu:
        """Side, Size, Colour…, Outline and Title…: every pick restyles the bar."""
        menu = QtWidgets.QMenu(parent)
        for title, key, choices in (("Side", "side", SIDES), ("Size", "size", tuple(SIZES))):
            sub = menu.addMenu(title)
            group = QtGui.QActionGroup(sub)
            for choice in choices:
                act = sub.addAction(choice)
                act.setCheckable(True)
                act.setChecked(self._style[key] == choice)
                group.addAction(act)
                act.triggered.connect(lambda _c=False, k=key, c=choice: self._change_style({k: c}))

        def pick_colour():
            colour = QtWidgets.QColorDialog.getColor(self.colour(), parent, "Colour bar text")
            if colour.isValid():
                self._change_style({"color": colour.name()})

        menu.addAction("Colour…").triggered.connect(pick_colour)
        outline = menu.addAction("Outline")
        outline.setCheckable(True)
        outline.setChecked(bool(self._style["outline"]))
        outline.triggered.connect(lambda checked: self._change_style({"outline": bool(checked)}))

        def edit_title():
            text, ok = QtWidgets.QInputDialog.getText(
                parent, "Colour bar title", "Title (empty: the quantity and units shown)",
                text=self.label)
            if ok:
                self.labelEdited.emit(text.strip())

        menu.addAction("Title…").triggered.connect(edit_title)
        self._menu = menu
        return menu

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.context_menu().popup(ev.screenPos().toPoint())
