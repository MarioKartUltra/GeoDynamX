# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The main window's panels are a fixed set of docks with today's arrangement as the default."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from dynamix.shell.docks import DOCK_NAMES
from dynamix.shell.main_window import MainWindow


def _areas(w):
    return {name: w.dockWidgetArea(w.docks[name]) for name in DOCK_NAMES}


def _font_report() -> str:
    """The font as it renders: the family asked for and the one found, its pixel size and a
    sample's width, beside the screen's dots per inch."""
    from PySide6 import QtGui

    app = QtWidgets.QApplication.instance()
    font, screen = app.font(), app.primaryScreen()
    info = QtGui.QFontInfo(font)
    return (f"style {app.style().name()!r} font {font.family()!r} {font.pointSizeF()}pt -> "
            f"{info.family()!r} {info.pixelSize()}px, 'cumulant spectrum' "
            f"{QtGui.QFontMetrics(font).horizontalAdvance('cumulant spectrum')}px; screen dpi "
            f"{screen.logicalDotsPerInch():g} logical, {screen.physicalDotsPerInch():g} physical, "
            f"ratio {screen.devicePixelRatio():g}")


def _layout_report(w):
    """The window's font and widths beside a dock-width assertion: a dock narrower than asked
    is a central widget whose minimum width took the rest."""
    docks = {n: (w.docks[n].width(), w.docks[n].minimumSizeHint().width()) for n in DOCK_NAMES}
    return (f"{_font_report()}; window {w.width()}, central {w.centralWidget().width()} "
            f"(min {w.centralWidget().minimumSizeHint().width()}), docks (width, min) {docks}")


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
    assert not w.docks["dock.layers"].isAncestorOf(w.layers_panel)
    first = w.right_panel._column.itemAt(0).widget()
    assert first is w.right_panel._sections["Layers"]             # and at the top of it
    assert w.docks["dock.layers"].windowTitle() == "Processing"    # the dock holds the processing tree
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
    assert w.docks["dock.browser"].width() == 180, _layout_report(w)
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
    save_settings(Settings(view_options={"dock_state": {"state": "00ff00ff"}}))
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
        action = w.commands[f"view.{name}"]
        assert action in view.actions() and action.isCheckable() and action.isChecked()
        w.docks[name].close()
        assert not action.isChecked()                              # follows the dock's visibility
        action.trigger()
        assert not w.docks[name].isHidden()                        # and drives it


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


def test_reset_layout_targets_the_constants_even_with_an_old_settings_file(qtbot, tmp_path, monkeypatch,
                                                                             clean_registry):
    from dynamix.shell.main_window import _PANEL_WIDTH
    from dynamix.shell.settings import Settings, save_settings

    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    save_settings(Settings(splitter_sizes={"work": [180, 900, 220], "main": [700, 150]}))
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    assert w.docks["dock.browser"].width() == 180, _layout_report(w)     # seeded this once
    w.reset_layout()
    qtbot.wait(20)
    assert w.docks["dock.browser"].width() == _PANEL_WIDTH         # the true default, not the seed
    w.close()                                                      # stores a dock state: the seed is spent
    w2 = MainWindow()
    qtbot.addWidget(w2)
    w2.resize(1400, 900)
    w2.show()
    qtbot.waitExposed(w2)
    assert w2.docks["dock.browser"].width() == _PANEL_WIDTH


def test_closing_stores_state_and_geometry_apart_from_named_layouts(qtbot, tmp_path, monkeypatch,
                                                                     clean_registry):
    from dynamix.shell.settings import load_settings

    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    w = MainWindow()
    qtbot.addWidget(w)
    avail = QtWidgets.QApplication.primaryScreen().availableGeometry()
    size = QtCore.QSize(min(1000, avail.width() - 40), min(640, avail.height() - 40))   # fits the screen
    w.resize(size)
    w.show()
    qtbot.waitExposed(w)
    w.save_layout("last")                                           # a user layout named "last"
    w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, w.docks["dock.inspector"])
    w.close()
    opts = load_settings().view_options
    assert set(opts["dock_state"]) == {"state", "geometry"}
    assert "last" in opts["layouts"] and opts["layouts"]["last"]["state"] != opts["dock_state"]["state"]
    w2 = MainWindow()
    qtbot.addWidget(w2)
    assert w2.dockWidgetArea(w2.docks["dock.inspector"]) == QtCore.Qt.LeftDockWidgetArea
    assert w2._geometry_restored and w2.height() == size.height()   # the width is pinned by the 800 px offscreen screen


def test_a_tabbed_dock_stays_shown_and_checked_when_its_tab_goes_behind(qtbot, clean_registry):
    """Qt emits visibilityChanged(False) for a dock whose tab goes behind another; that must never
    hide the dock or uncheck its View entry (the old toggleViewAction tracked hidden/shown only)."""
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    a, b = w.docks["dock.browser"], w.docks["dock.layers"]
    w.tabifyDockWidget(a, b)                      # b in front, a behind
    with qtbot.waitSignal(b.visibilityChanged, timeout=2000):
        a.raise_()                                # b goes behind: Qt reports it as a visibility change
    with qtbot.waitSignal(a.visibilityChanged, timeout=2000):
        b.raise_()
    assert not a.isHidden() and not b.isHidden()
    assert w.commands["view.dock.browser"].isChecked() and w.commands["view.dock.layers"].isChecked()
