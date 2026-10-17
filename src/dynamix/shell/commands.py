# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The command table bound to QActions, the menu bar built from it, the Analysis menu from the device
registry. The window hands in one callable per command id; nothing here knows what a command does."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from dynamix.model.commands import COMMANDS, MENU_ORDER
from dynamix.model.device import DEVICES
from dynamix.model.families import FAMILY_ORDER, families_of, label_of


def bind_commands(window: QtWidgets.QMainWindow, handlers: dict) -> dict:
    """One QAction per command, owned by ``window``: id in ``data()``, the table's shortcuts, window
    context, checkable and grouped as declared. A command with no handler is bound disabled."""
    actions, groups = {}, {}
    for c in COMMANDS:
        action = QtGui.QAction(c.title, window)
        action.setData(c.id)
        if c.shortcuts:
            action.setShortcuts([QtGui.QKeySequence(k) for k in c.shortcuts])
        action.setShortcutContext(QtCore.Qt.WindowShortcut)
        action.setCheckable(c.checkable)
        if c.group:
            group = groups.get(c.group)
            if group is None:
                group = groups[c.group] = QtGui.QActionGroup(window)
                group.setExclusive(True)
            group.addAction(action)
        handler = handlers.get(c.id)
        if handler is None:
            action.setEnabled(False)
        elif c.checkable and not c.group:
            # An ungrouped toggle reports every change of state, set by hand or by code (the
            # auto-run setting is written whichever way it changes); a grouped choice fires on
            # the user's pick only.
            action.toggled.connect(lambda checked, h=handler: h(checked))
        elif c.checkable:
            action.triggered.connect(lambda checked=False, h=handler: h(checked))
        else:
            action.triggered.connect(lambda _checked=False, h=handler: h())
        window.addAction(action)                    # the shortcut lives on the window, menu or not
        actions[c.id] = action
    return actions


def watch_enabled(actions: dict, menus: dict, enabled: dict):
    """``enabled`` maps a command id to a predicate; the returned callable applies them all and runs
    before every menu opens (the palette calls it when it opens)."""
    def refresh() -> None:
        for cid, pred in enabled.items():
            actions[cid].setEnabled(bool(pred()))
    for menu in menus.values():
        menu.aboutToShow.connect(refresh)
    refresh()
    return refresh


def build_menus(window: QtWidgets.QMainWindow, actions: dict, *, extra=None) -> dict:
    """The menu bar in MENU_ORDER from the commands' categories; Analysis from the device registry
    by family (an entry per device, id ``device.<name>``, the palette's id for the same action,
    calling ``window.add_device``).
    ``extra`` maps a category to a callable that appends to that menu after the commands."""
    bar = window.menuBar()
    menus = {}
    for category in MENU_ORDER:
        menu = bar.addMenu(category)
        menus[category] = menu
        if category == "Analysis":
            _fill_analysis(window, menu)
            continue
        submenus: dict = {}
        for c in COMMANDS:
            if c.category != category:
                continue
            target = menu
            if c.submenu:
                target = submenus.get(c.submenu)
                if target is None:
                    target = submenus[c.submenu] = menu.addMenu(c.submenu)
            if c.separator_before:
                target.addSeparator()
            target.addAction(actions[c.id])
        if extra and category in extra:
            extra[category](menu)
    return menus


def _fill_analysis(window: QtWidgets.QMainWindow, menu: QtWidgets.QMenu) -> None:
    by_family: dict = {}
    for name, device in sorted(DEVICES.items()):
        for family in families_of(device):
            by_family.setdefault(family, []).append(device)
    for family in [*FAMILY_ORDER, *sorted(set(by_family) - set(FAMILY_ORDER))]:
        if family not in by_family:
            continue
        sub = menu.addMenu(family)
        for device in by_family[family]:
            action = QtGui.QAction(label_of(device), window)
            action.setData(f"device.{device.name}")
            action.triggered.connect(lambda _checked=False, n=device.name: window.add_device(n))
            sub.addAction(action)


class CommandPalette(QtWidgets.QFrame):
    """A popup over the window: a line edit and the enabled commands plus the registered devices,
    filtered as you type; Enter runs the current row, Esc closes and hands focus back. Entries are
    command ids and ``device.<name>``; a device entry takes the drop path at the end of the chain."""

    @property
    def _window(self) -> QtWidgets.QMainWindow:
        """The window: the palette's Qt parent, read back from there (an attribute of its own would
        be a reference cycle with the window that keeps the palette)."""
        return self.parentWidget()

    def __init__(self, window: QtWidgets.QMainWindow, actions: dict[str, QtGui.QAction]) -> None:
        super().__init__(window, QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint)
        self._actions = actions
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.line = QtWidgets.QLineEdit()
        self.line.setPlaceholderText("Command or device\u2026")
        self.line.textChanged.connect(self.filter)
        self.line.returnPressed.connect(self._run_current)
        self.line.installEventFilter(self)              # Up/Down/PageUp/PageDown move the list
        self.list = QtWidgets.QListWidget()
        self.list.itemActivated.connect(lambda item: self._run(item.data(QtCore.Qt.UserRole)))
        layout.addWidget(self.line)
        layout.addWidget(self.list)
        self.resize(520, 360)

    def entries(self) -> list:
        """``(entry id, text)`` for every enabled command and every registered device."""
        out = [(cid, f"{a.text()}    " + ", ".join(
                    s.toString(QtGui.QKeySequence.NativeText) for s in a.shortcuts()).rstrip())
               for cid, a in self._actions.items() if a.isEnabled() and cid != "view.palette"]
        for name, device in sorted(DEVICES.items()):
            out.append((f"device.{name}",
                        f"Add {label_of(device)}  ({name}, {', '.join(families_of(device))})"))
        return out

    def filter(self, text: str) -> list:
        """Rebuild the list with the entries whose id or text contains every word of ``text``;
        returns the entry ids shown, in order."""
        words = text.lower().split()
        self.list.clear()
        shown = []
        for entry_id, label in self.entries():
            hay = f"{entry_id} {label}".lower()
            if all(w in hay for w in words):
                item = QtWidgets.QListWidgetItem(label)
                item.setData(QtCore.Qt.UserRole, entry_id)
                self.list.addItem(item)
                shown.append(entry_id)
        if self.list.count():
            self.list.setCurrentRow(0)
        return shown

    def open(self) -> None:
        refresh = getattr(self._window, "refresh_commands", None)
        if refresh is not None:
            refresh()
        self.line.clear()
        self.filter("")
        self.move(self._window.mapToGlobal(
            QtCore.QPoint((self._window.width() - self.width()) // 2, 80)))
        self.show()
        self.line.setFocus()

    def _run_current(self) -> None:
        item = self.list.currentItem() or (self.list.item(0) if self.list.count() else None)
        if item is not None:
            self._run(item.data(QtCore.Qt.UserRole))

    def eventFilter(self, obj, event) -> bool:
        if obj is self.line and event.type() == QtCore.QEvent.KeyPress and event.key() in (
                QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown):
            QtWidgets.QApplication.sendEvent(self.list, event)
            return True
        return super().eventFilter(obj, event)

    def _run(self, entry_id: str) -> None:
        self.hide()
        if entry_id.startswith("device."):
            self._window.add_device(entry_id[len("device."):])
        else:
            self._actions[entry_id].trigger()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape:
            self.hide()
            self._window.activateWindow()
            return
        super().keyPressEvent(event)
