# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The layer panel's row buttons as small monochrome icons: an eye for Hide (struck through while
hidden), a padlock for Lock (open at rest, shut while locked), a snowflake for Freeze and a
magnifier for the Inspector. Each is an inline SVG stroked in the theme's toggle colours and
rendered once per state at the screen's pixel ratio; no icon file ships."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtSvg

from dynamix.shell.theme import current

_EYE = ('<path d="M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8s-2.4 4.5-6.5 4.5S1.5 8 1.5 8z"/>'
        '<circle cx="8" cy="8" r="2.2"/>')
_EYE_OFF = _EYE + '<path d="M2.5 13.5l11-11"/>'
_LOCK_OPEN = '<rect x="3" y="7.5" width="10" height="6.5" rx="1"/><path d="M5 7.5V5.2a3 3 0 0 1 5.6-1.5"/>'
_LOCK = '<rect x="3" y="7.5" width="10" height="6.5" rx="1"/><path d="M5 7.5V5.2a3 3 0 0 1 6 0v2.3"/>'
_SNOW = ('<path d="M8 1.5v13M2.4 4.75l11.2 6.5M13.6 4.75L2.4 11.25M6.4 3.1L8 4.7l1.6-1.6'
         'M6.4 12.9L8 11.3l1.6 1.6M2.6 6.9l2.2.6.6-2.2M13.4 9.1l-2.2-.6-.6 2.2"/>')
_GLASS = '<circle cx="7" cy="7" r="4.5"/><path d="M10.4 10.4L14 14"/>'

#: button name -> (glyph at rest, glyph while checked)
ICONS = {"hide": (_EYE, _EYE_OFF), "lock": (_LOCK_OPEN, _LOCK), "freeze": (_SNOW, _SNOW),
         "inspector": (_GLASS, _GLASS)}
SIZE = 14


def _pixmap(body: str, colour: str, size: int) -> QtGui.QPixmap:
    ratio = (QtGui.QGuiApplication.primaryScreen().devicePixelRatio()
             if QtGui.QGuiApplication.primaryScreen() is not None else 1.0)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none" stroke="{colour}" '
           f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode("utf-8")))
    pixmap = QtGui.QPixmap(int(size * ratio), int(size * ratio))
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def row_icon(name: str, size: int = SIZE) -> QtGui.QIcon:
    """The icon of row button ``name``: the rest glyph in the theme's set colour (on the rest
    background), the checked glyph in the rest colour (on the set background), the swap the
    ``rowToggle`` style rule makes."""
    theme = current()
    rest, checked = ICONS[name]
    icon = QtGui.QIcon()
    icon.addPixmap(_pixmap(rest, theme.toggle_set, size), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
    icon.addPixmap(_pixmap(checked, theme.toggle_rest, size), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On)
    return icon


def style_row_button(button, name: str) -> None:
    """Give ``button`` (a QAbstractButton) row icon ``name`` at the rows' compact size."""
    button.setIcon(row_icon(name))
    button.setIconSize(QtCore.QSize(SIZE, SIZE))
    button.setFixedSize(QtCore.QSize(22, 20))
