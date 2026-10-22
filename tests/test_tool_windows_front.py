# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tool windows stay in front of the main window and come back as they were left: each is a
``Qt.Tool`` owned by the main window, and closing then reopening a tool on the same result shows
the same window with its state, while a new landing of the layer builds it anew."""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest
from PySide6 import QtCore

from test_fitting_tools_shell import _leaders_result

SHELL = pathlib.Path(__file__).resolve().parents[1] / "src" / "dynamix" / "shell"


@pytest.fixture
def win(qtbot, clean_registry):
    from dynamix.shell.main_window import MainWindow

    window = MainWindow(steps=())
    qtbot.addWidget(window)
    return window


def test_no_shell_window_is_built_as_a_plain_top_level_window():
    """A plain ``Qt.Window`` can fall behind the main window; tool windows are built with
    ``TOOL_WINDOW`` (:mod:`dynamix.shell.window_fit`)."""
    plain = [f"{p.name}:{i + 1}" for p in sorted(SHELL.glob("*.py"))
             for i, line in enumerate(p.read_text(encoding="utf-8").splitlines())
             if re.search(r"super\(\)\.__init__\([^)]*QtCore\.Qt\.Window\)", line)]
    assert not plain, plain


def _result():
    result = _leaders_result()
    result["record"]["method"] = "wavelet_leaders"
    result["_landed"] = {"analysis_key": "k1", "filters": [], "level": None}
    return result


def test_a_tool_window_is_a_tool_owned_by_the_main_window(win, qtbot):
    win._active_result = _result()
    win._refresh_samples_buttons()
    win._on_ld_button_clicked()
    ld = win._ld_window
    qtbot.addWidget(ld)
    assert ld.windowFlags() & QtCore.Qt.WindowType.WindowType_Mask == QtCore.Qt.WindowType.Tool
    assert ld.parentWidget() is win


def test_a_closed_tool_reopens_as_it_was_left_until_the_result_changes(win, qtbot):
    win._active_result = _result()
    win._refresh_samples_buttons()
    win._on_generalized_button_clicked()
    first = win._generalized_window
    qtbot.addWidget(first)
    first.spins["q max"].setValue(2.5)
    first.close()
    win._on_generalized_button_clicked()
    assert win._generalized_window is first and first.isVisible()
    assert first.spins["q max"].value() == 2.5
    win._active_result = {**_result(), "_landed": {"analysis_key": "k1", "filters": [], "level": 3}}
    win._on_generalized_button_clicked()
    assert win._generalized_window is first                                  # a scale tick changes nothing it reads
    win._active_result = {**_result(), "_landed": {"analysis_key": "k2", "filters": [], "level": None}}
    win._on_generalized_button_clicked()
    fresh = win._generalized_window
    qtbot.addWidget(fresh)
    assert fresh is not first and fresh.spins["q max"].value() != 2.5
