# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""tests/test_shell_layout.py — the splitter shell."""
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


def test_zone_tree_is_two_splitters(qtbot, clean_registry):
    w = _window(qtbot)
    assert isinstance(w._main_split, QtWidgets.QSplitter)
    assert w._main_split.orientation() == QtCore.Qt.Vertical
    assert isinstance(w._work_split, QtWidgets.QSplitter)
    assert w._work_split.orientation() == QtCore.Qt.Horizontal
    assert w._work_split.count() == 3            # left | center | right host
    assert w._main_split.count() == 2            # work | rack
    assert w._main_split.widget(0) is w._work_split


def test_no_fixed_geometry_left(qtbot, clean_registry):
    w = _window(qtbot)
    left = w._work_split.widget(0)
    assert left.maximumWidth() > 220              # the setFixedWidth(220) is gone
    rack = w._main_split.widget(1)
    assert rack.maximumHeight() > 196             # the setFixedHeight(196) is gone


def test_every_zone_collapsible(qtbot, clean_registry):
    w = _window(qtbot)
    for split in (w._main_split, w._work_split):
        for i in range(split.count()):
            assert split.isCollapsible(i)


def test_splitter_sizes_persist(qtbot, tmp_path, monkeypatch, clean_registry):
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "s.json"))
    w = _window(qtbot)
    w.resize(1400, 900)
    w._work_split.setSizes([180, 940, 280])
    w.close()                                     # persistence hook fires here
    d = json.loads((tmp_path / "s.json").read_text())
    assert d["splitter_sizes"]["work"] == w._work_split.sizes()
    w2 = _window(qtbot)
    w2.resize(1400, 900)
    assert w2._work_split.sizes() == w._work_split.sizes()
