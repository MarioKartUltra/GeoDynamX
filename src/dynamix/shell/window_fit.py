# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A window opens inside the screen it opens on, and comes back where the user left it.

Every window used to ask for a hard-coded size; on a laptop some asked for more than the
screen has, and a layout whose minimum width grows (a label's text changing) let Qt widen a
window the user had just shrunk. :func:`fit_to_screen` clamps a requested size to a fraction
of the screen's available area, axis by axis; :func:`remember_geometry` restores the geometry
saved for a key and saves it again when the window closes (``Settings.window_geometry``).
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

__all__ = ["clamp_size", "fit_to_screen", "remember_geometry"]


def clamp_size(size: QtCore.QSize, available: QtCore.QRect, fraction: float = 0.92) -> QtCore.QSize:
    """``size`` where it fits inside ``fraction`` of ``available``, else that bound, per axis."""
    w = min(int(size.width()), int(available.width() * fraction))
    h = min(int(size.height()), int(available.height() * fraction))
    return QtCore.QSize(max(w, 1), max(h, 1))


def _screen_for(widget: QtWidgets.QWidget) -> QtGui.QScreen:
    screen = widget.screen()
    if screen is None and widget.parentWidget() is not None:
        screen = widget.parentWidget().screen()
    return screen or QtGui.QGuiApplication.primaryScreen()


def fit_to_screen(widget: QtWidgets.QWidget, width: int, height: int, fraction: float = 0.92) -> None:
    """Resize ``widget`` to ``width`` × ``height`` or to what its screen can show."""
    widget.resize(clamp_size(QtCore.QSize(width, height), _screen_for(widget).availableGeometry(),
                             fraction))


class _GeometryMemory(QtCore.QObject):
    """Saves the watched window's geometry under ``key`` when it closes, into the settings
    file that was current when the memory was installed (a close event can arrive after a
    test's environment is torn down; the file must not change under it)."""

    def __init__(self, widget: QtWidgets.QWidget, key: str, path):
        super().__init__(widget)
        self._key = key
        self._path = path
        widget.installEventFilter(self)

    def eventFilter(self, watched, event):                      # noqa: N802 (Qt's name)
        if event.type() == QtCore.QEvent.Type.Close and isinstance(watched, QtWidgets.QWidget):
            import json

            from dynamix.model.atomic_write import write_json_atomically

            g = watched.geometry()
            try:
                try:
                    data = json.loads(self._path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    data = {}
                memory = data.get("window_geometry") if isinstance(data.get("window_geometry"), dict) else {}
                memory[self._key] = [int(g.x()), int(g.y()), int(g.width()), int(g.height())]
                data["window_geometry"] = memory
                self._path.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomically(self._path, data)
            except OSError:
                pass
        return False


def remember_geometry(widget: QtWidgets.QWidget, key: str) -> None:
    """Restore the geometry saved under ``key`` when one exists and lands on a screen (clamped
    to that screen), and save the window's geometry under ``key`` when it closes."""
    from dynamix.shell.settings import load_settings, settings_path

    path = settings_path()
    saved = load_settings().window_geometry.get(key)
    if saved and len(saved) == 4:
        rect = QtCore.QRect(*saved)
        screen = QtGui.QGuiApplication.screenAt(rect.center())
        if screen is not None:
            size = clamp_size(rect.size(), screen.availableGeometry(), fraction=1.0)
            widget.setGeometry(QtCore.QRect(rect.topLeft(), size))
    widget._geometry_memory = _GeometryMemory(widget, key, path)
