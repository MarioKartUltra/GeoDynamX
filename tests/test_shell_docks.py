# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The main window's panels are a fixed set of docks with today's arrangement as the default."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from dynamix.shell.docks import DOCK_NAMES
from dynamix.shell.main_window import MainWindow


def _areas(w):
    return {name: w.dockWidgetArea(w.docks[name]) for name in DOCK_NAMES}


def test_the_dock_set_is_exactly_the_four_named_docks(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    found = sorted(d.objectName() for d in w.findChildren(QtWidgets.QDockWidget))
    assert found == sorted(DOCK_NAMES)
    assert w.centralWidget().isAncestorOf(w._center_stack)


def test_the_default_layout_is_todays_arrangement(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    a = _areas(w)
    assert a["dock.browser"] == QtCore.Qt.LeftDockWidgetArea
    assert a["dock.layers"] == QtCore.Qt.LeftDockWidgetArea
    assert a["dock.inspector"] == QtCore.Qt.RightDockWidgetArea
    assert a["dock.chain"] == QtCore.Qt.BottomDockWidgetArea
    assert w.docks["dock.browser"].isAncestorOf(w.browser)
    assert w.docks["dock.layers"].isAncestorOf(w.layer_list)
    assert w.docks["dock.inspector"].isAncestorOf(w.layers_panel)   # the Layers stack stays in the inspector
    assert w.docks["dock.inspector"].isAncestorOf(w.right_panel)
    assert w.docks["dock.chain"].isAncestorOf(w._strip_scroll)   # the strips land here on first load
    chain = w.docks["dock.chain"]
    assert chain.allowedAreas() == QtCore.Qt.BottomDockWidgetArea
    assert not (chain.features() & QtWidgets.QDockWidget.DockWidgetFloatable)


def test_dock_state_round_trips(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, w.docks["dock.inspector"])
    moved = w.saveState()
    assert _areas(w)["dock.inspector"] == QtCore.Qt.LeftDockWidgetArea
    assert w.restoreState(w._default_dock_state)
    assert _areas(w)["dock.inspector"] == QtCore.Qt.RightDockWidgetArea
    assert w.restoreState(moved)
    assert _areas(w)["dock.inspector"] == QtCore.Qt.LeftDockWidgetArea


def test_old_splitter_sizes_seed_the_dock_widths(qtbot, tmp_path, monkeypatch, clean_registry):
    from dynamix.shell.settings import Settings, save_settings

    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    save_settings(Settings(splitter_sizes={"work": [180, 900, 220], "main": [700, 150]}))
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    assert w.docks["dock.browser"].width() == 180
    assert w.docks["dock.inspector"].width() == 220
    assert w.docks["dock.chain"].height() == 150


def test_reset_layout_restores_the_default_and_shows_hidden_docks(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, w.docks["dock.inspector"])
    w.docks["dock.layers"].close()
    assert not w.docks["dock.layers"].isVisible()
    w.reset_layout()
    assert _areas(w)["dock.inspector"] == QtCore.Qt.RightDockWidgetArea
    assert all(w.docks[n].isVisible() for n in DOCK_NAMES)


def test_named_layouts_save_restore_and_list_in_the_menu(qtbot, tmp_path, monkeypatch, clean_registry):
    from dynamix.shell.settings import load_settings

    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, w.docks["dock.inspector"])
    w.save_layout("inspector left")
    assert "inspector left" in load_settings().view_options["layouts"]
    assert "inspector left" in [a.text() for a in w._layouts_menu.actions()]
    w.reset_layout()
    assert _areas(w)["dock.inspector"] == QtCore.Qt.RightDockWidgetArea
    assert w.apply_layout("inspector left")
    assert _areas(w)["dock.inspector"] == QtCore.Qt.LeftDockWidgetArea
    assert not w.apply_layout("no such layout")


def test_a_layout_whose_state_no_longer_matches_falls_back_to_the_default(qtbot, tmp_path, monkeypatch,
                                                                            clean_registry):
    from dynamix.shell.settings import Settings, save_settings

    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    save_settings(Settings(view_options={"layouts": {"last": {"state": "00ff00ff"}}}))
    w = MainWindow()
    qtbot.addWidget(w)
    assert all(not w.docks[n].isHidden() for n in DOCK_NAMES)
    assert _areas(w)["dock.inspector"] == QtCore.Qt.RightDockWidgetArea


def test_the_view_menu_lists_a_toggle_per_dock_and_reset_layout(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    view = next(m for m in w.menuBar().findChildren(QtWidgets.QMenu) if m.title() == "View")
    texts = [a.text() for a in view.actions()]
    assert "Reset Layout" in texts
    for name in DOCK_NAMES:
        assert w.docks[name].toggleViewAction() in view.actions()


def test_a_floating_dock_restored_off_every_screen_docks_again(qtbot, clean_registry):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    dock = w.docks["dock.browser"]
    dock.setFloating(True)
    dock.move(100_000, 100_000)                       # a second monitor this machine does not have
    w._redock_offscreen()
    assert not dock.isFloating()
