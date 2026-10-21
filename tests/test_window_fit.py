# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A window's opening size is clamped to the screen it opens on, and a window remembers where
the user left it (dynamix.shell.window_fit)."""
from __future__ import annotations

import os

import pytest
from PySide6 import QtCore, QtWidgets

from dynamix.shell.window_fit import clamp_size, fit_to_screen, remember_geometry


def test_clamp_size_keeps_a_fitting_size_and_shrinks_each_axis_that_does_not():
    screen = QtCore.QRect(0, 0, 1512, 900)                      # a 14-inch laptop's usable area
    assert clamp_size(QtCore.QSize(950, 700), screen) == QtCore.QSize(950, 700)
    assert clamp_size(QtCore.QSize(1150, 1000), screen) == QtCore.QSize(1150, 828)   # 0.92 × 900
    assert clamp_size(QtCore.QSize(1800, 500), screen) == QtCore.QSize(1391, 500)    # 0.92 × 1512
    assert clamp_size(QtCore.QSize(1150, 1000), screen, fraction=1.0) == QtCore.QSize(1150, 900)


def test_fit_to_screen_sizes_the_widget_within_its_screen(qtbot):
    w = QtWidgets.QWidget()
    qtbot.addWidget(w)
    fit_to_screen(w, 5000, 4000)
    w.show()
    avail = w.screen().availableGeometry()
    assert w.width() <= avail.width() and w.height() <= avail.height()
    fit_to_screen(w, 300, 200)
    assert (w.width(), w.height()) == (300, 200)


def test_remember_geometry_round_trips_through_the_settings_file(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(tmp_path / "settings.json"))
    from dynamix.shell.settings import load_settings

    first = QtWidgets.QWidget()
    qtbot.addWidget(first)
    remember_geometry(first, "fitter")                          # nothing saved yet: no change
    first.setGeometry(40, 60, 420, 330)
    first.show()
    qtbot.wait(10)
    first.close()
    saved = load_settings().window_geometry["fitter"]
    assert saved[2:] == [420, 330] and saved[:2] == [first.geometry().x(), first.geometry().y()]

    second = QtWidgets.QWidget()
    qtbot.addWidget(second)
    second.resize(900, 900)
    remember_geometry(second, "fitter")
    second.show()
    qtbot.wait(10)
    assert (second.width(), second.height()) == (420, 330)


def test_a_geometry_save_that_fails_keeps_the_previous_settings_file(qtbot, tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text('{\n  "theme": "light"\n}', encoding="utf-8")
    before, files = path.read_bytes(), sorted(tmp_path.iterdir())
    monkeypatch.setenv("DYNAMIX_SETTINGS_PATH", str(path))

    def fail(*_args):
        raise OSError(28, "No space left on device")

    w = QtWidgets.QWidget()
    qtbot.addWidget(w)
    remember_geometry(w, "fitter")
    w.show()
    qtbot.wait(10)
    with monkeypatch.context() as m:
        m.setattr(os, "replace", fail)
        w.close()
    assert path.read_bytes() == before
    assert sorted(tmp_path.iterdir()) == files


def test_the_floating_windows_open_narrower_than_a_laptop_screen(qtbot):
    # The anisotropy window's one-line caption used to set a 1280-px minimum width.
    from dynamix.shell.anisotropy_window import AnisotropyWindow

    w = AnisotropyWindow()
    qtbot.addWidget(w)
    w.show()
    assert w.minimumSizeHint().width() < 800
    avail = w.screen().availableGeometry()
    assert w.width() <= avail.width() and w.height() <= avail.height()
