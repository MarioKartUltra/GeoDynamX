# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Every menu action is a command; the shortcuts kept their keys; Space toggles the transport once."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from dynamix.model.commands import by_id
from dynamix.shell.main_window import MainWindow
from tests.test_shell_window import _FIELD, loaded, stub_devices, window  # noqa: F401


def _menu_actions(w):
    out = []
    for menu in w.menuBar().findChildren(QtWidgets.QMenu):
        out += [a for a in menu.actions() if not a.isSeparator() and a.menu() is None]
    return out


def test_every_menu_action_carries_a_command_id_and_the_bar_follows_the_order(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    titles = [a.text() for a in w.menuBar().actions()]
    assert titles == ["File", "Edit", "Layer", "Analysis", "View", "Racks", "Settings", "Help"]
    table = by_id()
    for action in _menu_actions(w):
        cid = action.data()
        assert cid and (cid in table or cid.startswith(("analysis.", "view.layouts.apply."))), action.text()


def test_shortcuts_keep_their_keys_and_window_context(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    for cid, key in {"chain.run": "Ctrl+Return", "transport.play": "Space", "view.selection.click": "C",
                     "view.selection.cycle": "V", "edit.cancel_transect": "Esc",
                     "edit.undo_rack_drop": "Ctrl+Shift+Z", "file.save_project": "Ctrl+S"}.items():
        action = w.commands[cid]
        assert key in [s.toString() for s in action.shortcuts()], cid
        assert action.shortcutContext() == QtCore.Qt.WindowShortcut, cid
    assert [s.toString() for s in w._zoom_action.shortcuts()] == ["Ctrl+0", "F"]


def test_settings_groups_are_exclusive_and_reflect_the_settings_file(qtbot, clean_registry):
    from dynamix.shell.settings import load_settings

    w = MainWindow()
    qtbot.addWidget(w)
    s = load_settings()
    engine = {k: w.commands[k] for k in ("settings.engine.auto", "settings.engine.mlx", "settings.engine.fftw")}
    assert [k for k, a in engine.items() if a.isChecked()] == [f"settings.engine.{s.compute_engine}"]
    prec = {k: w.commands[k] for k in ("settings.precision.32", "settings.precision.64")}
    assert [k for k, a in prec.items() if a.isChecked()] == [f"settings.precision.{s.compute_precision}"]


def test_the_analysis_menu_lists_devices_by_family(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    analysis = next(m for m in w.menuBar().findChildren(QtWidgets.QMenu) if m.title() == "Analysis")
    families = [a.menu().title() for a in analysis.actions() if a.menu() is not None]
    assert "Edge detection" in families
    ids = [a.data() for m in analysis.findChildren(QtWidgets.QMenu) for a in m.actions() if a.data()]
    assert "analysis.wtmm2d" in ids
