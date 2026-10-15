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
    window._command_groups = groups
    return actions


def build_menus(window: QtWidgets.QMainWindow, actions: dict, *, extra=None) -> dict:
    """The menu bar in MENU_ORDER from the commands' categories; Analysis from the device registry
    by family (an entry per device, id ``analysis.<device>``, calling ``window.add_device``).
    ``extra`` maps a category to a callable that appends to that menu after the commands."""
    bar = window.menuBar()
    menus = {}
    for category in MENU_ORDER:
        menu = bar.addMenu(category)
        menus[category] = menu
        if category == "Analysis":
            _fill_analysis(window, menu)
            continue
        for c in COMMANDS:
            if c.category == category:
                menu.addAction(actions[c.id])
        if extra and category in extra:
            extra[category](menu)
    return menus


def _fill_analysis(window, menu) -> None:
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
            action.setData(f"analysis.{device.name}")
            action.triggered.connect(lambda _checked=False, n=device.name: window.add_device(n))
            sub.addAction(action)
