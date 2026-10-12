# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The 3-D view is built ahead of the first Vector/Globe switch, once, after the window shows.
A real render surface cannot exist under the offscreen platform, so these tests pin the
scheduling and the guards; the build itself is checked on screen."""
from __future__ import annotations

import pytest


@pytest.fixture
def win(qtbot, clean_registry):
    from dynamix.shell.main_window import MainWindow

    w = MainWindow(steps=())
    qtbot.addWidget(w)
    return w


def test_showing_the_window_schedules_one_warm_up(win, monkeypatch):
    from PySide6 import QtCore

    scheduled = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot",
                        lambda ms, *a: scheduled.append(getattr(a[-1], "__name__", a[-1])))
    win.show()
    win.hide()
    win.show()
    assert scheduled.count("_warm_up_3d") == 1


def test_the_warm_up_builds_nothing_without_a_real_window_surface(win):
    win._warm_up_3d()
    assert win._arrangement is None


def test_the_tab_switch_still_builds_the_view_through_the_shared_path(win):
    win._set_center_view("vector")
    first = win._arrangement
    assert first is not None
    win._set_center_view("raster")
    win._ensure_arrangement()
    assert win._arrangement is first                    # built once, reused
