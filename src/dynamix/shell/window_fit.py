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

__all__ = ["TOOL_WINDOW", "clamp_size", "fit_to_screen", "remember_geometry", "show_while_inactive"]


#: The window type of every tool window: a utility window owned by its opener, kept in front of
#: it (a plain ``Qt.Window`` falls behind the main window as soon as the main window is clicked).
TOOL_WINDOW = QtCore.Qt.WindowType.Tool


def show_while_inactive(widget: QtWidgets.QWidget) -> None:
    """Keep ``widget``, a tool window, on screen while the application is in the background:
    macOS hides tool windows then unless told otherwise."""
    widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)


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


class WrappingLayout(QtWidgets.QLayout):
    """A row of controls that wraps rather than imposing its combined width on the window: a
    tool window stays inside an 800-pixel screen whatever the platform's font metrics."""

    def __init__(self):
        super().__init__()
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(6)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QtCore.QRect(0, 0, width, 0), dry=True)

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        return size

    def sizeHint(self):
        return self.minimumSize()

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect)

    def _arrange(self, rect, dry=False):
        x, y, height = rect.x(), rect.y(), 0
        for item in self._items:
            if item.isEmpty():
                continue
            size = item.sizeHint()
            width = min(size.width(), rect.width())
            if x > rect.x() and x + width > rect.x() + rect.width():
                x, y, height = rect.x(), y + height + self.spacing(), 0
            item_height = (item.heightForWidth(width) if item.hasHeightForWidth()
                           else size.height())
            if not dry:
                item.setGeometry(QtCore.QRect(x, y, width, item_height))
            x += width + self.spacing()
            height = max(height, item_height)
        return y + height - rect.y()
