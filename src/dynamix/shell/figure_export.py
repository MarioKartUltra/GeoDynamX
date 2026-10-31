# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""File > Export Figure: a view rendered as the screen shows it, at a scale factor.

The Raster canvas is a ``QGraphicsView`` whose distance bar label and wavelet label are child
widgets; the Map view's bars are scene items. :func:`render_widget` renders a widget with its
children into an image ``scale`` times its size at that device pixel ratio, so text, lines and
the raster's pixels are drawn at the export's resolution. :func:`render_canvas` can instead set
the colour bar in a strip beside the image, so it covers no data."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.shell.colorbar import MARGIN_PX

SCALES = (1, 2, 3, 4)
#: Where the Raster canvas's colour bar goes in a figure: as on screen, or beside the image.
PLACEMENTS = ("over", "beside")


def _image(size: QtCore.QSize, scale: int) -> QtGui.QImage:
    image = QtGui.QImage(size * scale, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(scale)
    return image


def render_widget(widget, scale: int = 1) -> QtGui.QImage:
    """``widget`` and its child widgets as painted on screen, in an image ``scale`` times its
    size with device pixel ratio ``scale``."""
    image = _image(widget.size(), scale)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    # QWidget's render: pyqtgraph's GraphicsView.render draws the scene alone, stretched over the
    # whole device, without the child widgets.
    QtWidgets.QWidget.render(widget, painter, QtCore.QPoint(), QtGui.QRegion(),
                             QtWidgets.QWidget.RenderFlag.DrawWindowBackground
                             | QtWidgets.QWidget.RenderFlag.DrawChildren)
    painter.end()
    return image


def render_canvas(canvas, scale: int = 1, placement: str = "over") -> QtGui.QImage:
    """The Raster canvas as :func:`render_widget` renders it. With ``placement`` "beside" and
    the colour bar shown, the canvas is rendered without the bar and widened on the bar's side
    by a strip of its background, the bar's depth plus ``MARGIN_PX`` on each side, where the bar
    is drawn at its screen position along that side."""
    bar = canvas.color_bar
    if placement != "beside" or not bar.isVisible():
        return render_widget(canvas, scale)
    bar.hide()
    try:
        picture = render_widget(canvas, scale)
    finally:
        bar.show()
    inner = bar.content_rect()
    at = QtCore.QPointF(canvas.mapFromScene(bar.mapToScene(inner.topLeft())))
    side = bar.style()["side"]
    w, h = canvas.width(), canvas.height()
    if bar.vertical():
        strip = int(round(inner.width() + 2.0 * MARGIN_PX))
        size, offset = QtCore.QSize(w + strip, h), QtCore.QPoint(strip if side == "left" else 0, 0)
        origin = QtCore.QPointF(MARGIN_PX if side == "left" else w + MARGIN_PX, at.y())
    else:
        strip = int(round(inner.height() + 2.0 * MARGIN_PX))
        size, offset = QtCore.QSize(w, h + strip), QtCore.QPoint(0, strip if side == "top" else 0)
        origin = QtCore.QPointF(at.x(), MARGIN_PX if side == "top" else h + MARGIN_PX)
    image = _image(size, scale)
    image.fill(canvas.backgroundBrush().color())
    painter = QtGui.QPainter(image)
    painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Source)
    painter.drawImage(offset, picture)
    painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.translate(origin - inner.topLeft())
    bar.paint(painter)
    painter.end()
    return image


class FigureExportDialog(QtWidgets.QDialog):
    """The figure's scale factor and, while the Raster canvas shows its colour bar, the bar's
    placement."""

    def __init__(self, *, colorbar: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export figure")
        form = QtWidgets.QFormLayout(self)
        self.scale_combo = QtWidgets.QComboBox()
        for s in SCALES:
            self.scale_combo.addItem(f"{s}×", s)
        self.scale_combo.setCurrentIndex(SCALES.index(2))
        self.scale_combo.setToolTip("The image is this many times the view's size on screen")
        form.addRow("Scale", self.scale_combo)
        self.placement_combo = QtWidgets.QComboBox(self)
        for placement in PLACEMENTS:
            self.placement_combo.addItem(f"{placement} the image", placement)
        self.placement_combo.setToolTip(
            "Over the image: as the screen shows it. Beside the image: the figure is widened by "
            "the bar's strip, so the bar covers no data")
        self._colorbar = bool(colorbar)
        if colorbar:
            form.addRow("Colour bar", self.placement_combo)
        else:
            self.placement_combo.hide()
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def scale(self) -> int:
        return int(self.scale_combo.currentData())

    def placement(self) -> str:
        return str(self.placement_combo.currentData()) if self._colorbar else "over"
