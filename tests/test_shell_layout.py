# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""tests/test_shell_layout.py — the dock shell: four docks around the centre stack."""
import json

from PySide6 import QtCore, QtWidgets

from dynamix.shell.main_window import MainWindow


def _window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


# Every test here builds a bare ``MainWindow()``, which unconditionally calls
# ``register_builtin_devices()`` (main_window.py's own ``__init__``) -- the suite-wide
# ``_registry_leak_guard`` (tests/conftest.py) requires that be undone afterwards, so each test
# also takes ``clean_registry`` (unused directly; its fixture teardown restores ``DEVICES``),
# same pattern ``test_shell_window.py::test_auto_run_setting_restores_todays_behavior`` already
# uses for the same reason.


def test_zone_tree_is_four_docks_around_the_centre(qtbot, clean_registry):
    """Supersedes the two-splitter zone tree: browser and layers left, inspector right, the chain
    along the bottom, the centre stack as the central widget (shell/docks.py)."""
    w = _window(qtbot)
    assert set(w.docks) == {"dock.browser", "dock.layers", "dock.inspector", "dock.chain"}
    assert w.centralWidget().isAncestorOf(w._center_stack)
    assert w.dockWidgetArea(w.docks["dock.browser"]) == QtCore.Qt.LeftDockWidgetArea
    assert w.dockWidgetArea(w.docks["dock.layers"]) == QtCore.Qt.LeftDockWidgetArea
    assert w.dockWidgetArea(w.docks["dock.inspector"]) == QtCore.Qt.RightDockWidgetArea
    assert w.dockWidgetArea(w.docks["dock.chain"]) == QtCore.Qt.BottomDockWidgetArea


def test_no_fixed_geometry_on_the_docks(qtbot, clean_registry):
    w = _window(qtbot)
    assert w.docks["dock.browser"].widget().maximumWidth() > 220      # no setFixedWidth(220)
    assert w.docks["dock.chain"].widget().maximumHeight() > 196      # no setFixedHeight(196)


def test_every_dock_can_be_hidden_and_shown_again(qtbot, clean_registry):
    """Supersedes "every zone collapsible": a dock closes and its View-menu toggle brings it back."""
    w = _window(qtbot)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    for dock in w.docks.values():
        action = dock.toggleViewAction()
        assert action.isCheckable() and action.isChecked()
        dock.close()
        assert not dock.isVisible() and not action.isChecked()
        action.trigger()
        assert dock.isVisible()


def test_the_dock_layout_persists_across_windows(qtbot, tmp_path, monkeypatch, clean_registry):
    """Supersedes the splitter-size round trip: closing stores the arrangement, the next window
    restores it."""
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    w = _window(qtbot)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, w.docks["dock.inspector"])
    w.close()                                     # persistence hook fires here
    d = json.loads((tmp_path / "s.json").read_text())
    assert d["view_options"]["dock_state"]["state"]
    w2 = _window(qtbot)
    w2.resize(1400, 900)
    assert w2.dockWidgetArea(w2.docks["dock.inspector"]) == QtCore.Qt.LeftDockWidgetArea
