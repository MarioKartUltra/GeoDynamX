# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The working folder: where every open, save and export dialog starts, and where a saved fork
is offered, set from Settings > Working Folder…."""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets

from dynamix.shell.settings import load_settings, start_path, update_settings
from tests.test_shell_window import stub_devices, window  # noqa: F401


def _record(monkeypatch, name, returns):
    seen = []

    def fake(_parent, _caption, start="", *_a, **_k):
        seen.append(start)
        return returns

    monkeypatch.setattr(QtWidgets.QFileDialog, name, fake)
    return seen


def test_start_path_is_inside_the_working_folder_only_while_it_exists(tmp_path):
    assert load_settings().working_folder is None
    assert start_path("a.npz") == "a.npz" and start_path() == ""
    update_settings(working_folder=str(tmp_path))
    assert start_path("a.npz") == str(tmp_path / "a.npz") and start_path() == str(tmp_path)
    update_settings(working_folder=str(tmp_path / "gone"))
    assert start_path("a.npz") == "a.npz"


def test_the_open_and_project_dialogs_start_in_the_working_folder(window, tmp_path, monkeypatch):
    update_settings(working_folder=str(tmp_path))
    opened = _record(monkeypatch, "getOpenFileNames", ([], ""))
    project = _record(monkeypatch, "getOpenFileName", ("", ""))
    saved = _record(monkeypatch, "getSaveFileName", ("", ""))
    window._on_open_clicked()
    window._on_open_project_clicked()
    window._on_save_project_as_clicked()
    assert opened == [str(tmp_path)] and project == [str(tmp_path)] and saved == [str(tmp_path)]


def test_a_saved_fork_is_offered_in_the_working_folder(window, tmp_path, monkeypatch):
    update_settings(working_folder=str(tmp_path))
    saved = _record(monkeypatch, "getSaveFileName", ("", ""))
    assert window._derivative_path("curvature filled", temporary=False) is None
    assert saved == [str(tmp_path / "curvature_filled.npz")]


def test_the_setting_is_chosen_from_the_settings_menu(window, tmp_path, monkeypatch):
    asked = _record(monkeypatch, "getExistingDirectory", str(tmp_path))
    window.commands["settings.working_folder"].trigger()
    assert load_settings().working_folder == str(tmp_path) and asked == [str(Path.home())]
    _record(monkeypatch, "getExistingDirectory", "")
    window.commands["settings.working_folder"].trigger()          # cancelled: the folder stays
    assert load_settings().working_folder == str(tmp_path)
