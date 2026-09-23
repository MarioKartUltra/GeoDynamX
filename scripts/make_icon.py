# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Paint the GeoDynamix app icon: a two-color cartoon earth wearing a GeoDynamix sash.

Usage: python scripts/make_icon.py OUT.png   (writes a 1024x1024 PNG, transparent background)

Deliberately exactly two solid colors for the earth (ocean blue, land green) plus white text,
by design. Continents are cartoon blobs, not geography — the honest
version of "not a map projection". PySide6 is fine here: scripts/ is outside the package, so
the shell-only Qt rule (which the boundary tests scan src/ for) does not apply.
"""
from __future__ import annotations

import sys

from PySide6 import QtCore, QtGui

OCEAN = QtGui.QColor("#2E86D9")
LAND = QtGui.QColor("#3FAE5A")
TEXT = QtGui.QColor("#FFFFFF")
NAME = "GeoDynamix"


def paint(size: int = 1024) -> QtGui.QImage:
    img = QtGui.QImage(size, size, QtGui.QImage.Format_ARGB32_Premultiplied)
    img.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setPen(QtCore.Qt.NoPen)

    # The globe.
    centre, r = QtCore.QPointF(size / 2, size * 0.46), size * 0.40
    globe = QtGui.QPainterPath()
    globe.addEllipse(centre, r, r)
    p.setBrush(OCEAN)
    p.drawPath(globe)

    # Cartoon continents, clipped to the globe. Blobs, not geography.
    p.save()
    p.setClipPath(globe)
    p.setBrush(LAND)
    blobs = QtGui.QPainterPath()
    blobs.addEllipse(QtCore.QPointF(size * 0.34, size * 0.36), size * 0.13, size * 0.19)   # west
    blobs.addEllipse(QtCore.QPointF(size * 0.40, size * 0.58), size * 0.07, size * 0.11)   # its tail
    blobs.addEllipse(QtCore.QPointF(size * 0.63, size * 0.30), size * 0.17, size * 0.11)   # north-east
    blobs.addEllipse(QtCore.QPointF(size * 0.68, size * 0.47), size * 0.10, size * 0.13)   # east
    blobs.addEllipse(QtCore.QPointF(size * 0.76, size * 0.66), size * 0.06, size * 0.045)  # a small one
    p.drawPath(blobs)
    p.restore()

    # The sash, in the land green, wider than the globe so the name reads at dock size.
    band = QtCore.QRectF(size * 0.04, size * 0.62, size * 0.92, size * 0.20)
    p.setBrush(LAND)
    p.drawRoundedRect(band, size * 0.05, size * 0.05)

    font = QtGui.QFont("Helvetica Neue")
    font.setBold(True)
    px = int(size * 0.145)
    font.setPixelSize(px)
    while QtGui.QFontMetricsF(font).horizontalAdvance(NAME) > band.width() * 0.92 and px > 8:
        px -= 2                                   # shrink to fit the sash
        font.setPixelSize(px)
    p.setFont(font)
    p.setPen(TEXT)
    p.drawText(band, QtCore.Qt.AlignCenter, NAME)

    p.end()
    return img


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: make_icon.py OUT.png", file=sys.stderr)
        return 2
    QtGui.QGuiApplication([])
    ok = paint().save(argv[0], "PNG")
    if not ok:
        print(f"failed to write {argv[0]}", file=sys.stderr)
        return 1
    print(f"wrote {argv[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
