# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The theme's stylesheet sits on Fusion everywhere but macOS, and the test application starts on
the same base style the launch gives the app."""
from __future__ import annotations

import sys

from PySide6 import QtWidgets

from dynamix.shell import theme


def test_the_base_style_is_fusion_off_macos():
    assert theme.BASE_STYLE == (None if sys.platform == "darwin" else "Fusion")


def test_applying_the_theme_sets_the_base_style(qapp, monkeypatch):
    before, sheet = qapp.style().name(), qapp.styleSheet()
    qapp.setStyle("Windows")
    monkeypatch.setattr(theme, "BASE_STYLE", "Fusion")
    try:
        theme.apply_theme(qapp, theme.RESTRAINED_DARK)
        qapp.setStyleSheet("")                 # a stylesheet's style reports no name of its own
        assert qapp.style().name() == "fusion"
        monkeypatch.setattr(theme, "BASE_STYLE", None)
        qapp.setStyle("Windows")
        theme.apply_theme(qapp, theme.RESTRAINED_DARK)
        qapp.setStyleSheet("")
        assert qapp.style().name() == "windows"          # macOS: the platform's own style stays
    finally:
        qapp.setStyle(before or "Fusion")
        qapp.setStyleSheet(sheet)


def test_the_test_application_starts_on_the_base_style(qapp):
    if theme.BASE_STYLE is not None and not qapp.styleSheet():
        assert qapp.style().name() == theme.BASE_STYLE.lower()
    assert QtWidgets.QApplication.instance() is qapp
