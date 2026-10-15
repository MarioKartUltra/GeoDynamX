# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The command table: stable dotted ids, one default shortcut each, the former shortcuts kept."""
from __future__ import annotations

import collections

from dynamix.model.commands import COMMANDS, MENU_ORDER, by_id


def test_ids_are_unique_dotted_and_categorised():
    ids = [c.id for c in COMMANDS]
    assert len(ids) == len(set(ids))
    for c in COMMANDS:
        assert "." in c.id and c.id == c.id.lower() and " " not in c.id, c.id
        assert c.category in MENU_ORDER, c.id
        assert c.title


def test_no_two_commands_share_a_default_shortcut():
    keys = collections.Counter(k for c in COMMANDS for k in c.shortcuts)
    assert not [k for k, n in keys.items() if n > 1], keys


def test_every_former_shortcut_is_present_with_the_same_key():
    former = {"chain.run": "Ctrl+Return", "transport.play": "Space", "view.selection.click": "C",
              "view.selection.cycle": "V", "edit.cancel_transect": "Esc", "edit.undo_transect": "Ctrl+Z",
              "edit.undo_rack_drop": "Ctrl+Shift+Z", "view.zoom_to_dataset": "Ctrl+0",
              "file.save_project": "Ctrl+S", "view.palette": "Ctrl+K"}
    table = by_id()
    for cid, key in former.items():
        assert key in table[cid].shortcuts, (cid, table[cid].shortcuts)
    assert "F" in table["view.zoom_to_dataset"].shortcuts


def test_headless_commands_are_the_settings_ones():
    assert {c.id for c in COMMANDS if c.headless} == {
        "settings.auto_run", "settings.engine.auto", "settings.engine.mlx", "settings.engine.fftw",
        "settings.precision.32", "settings.precision.64"}


def test_the_runner_lists_headless_commands_and_sets_a_setting(tmp_path):
    import json
    import os
    import subprocess
    import sys

    env = {**os.environ, "DYNAMIX_SETTINGS_PATH": str(tmp_path / "s.json"), "PYTHONNOUSERSITE": "1"}
    out = subprocess.run([sys.executable, "-m", "dynamix.run", "--list"],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "settings.engine.fftw" in out.stdout and "file.open_project" not in out.stdout
    out = subprocess.run([sys.executable, "-m", "dynamix.run", "settings.engine.fftw"],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert json.loads((tmp_path / "s.json").read_text())["compute_engine"] == "fftw"
    out = subprocess.run([sys.executable, "-m", "dynamix.run", "file.open_project"],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 2 and "needs a window" in out.stderr


def test_the_runner_imports_no_qt():
    import subprocess
    import sys

    code = "import sys; sys.modules['PySide6'] = None; import dynamix.run; print('ok')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**__import__("os").environ, "PYTHONNOUSERSITE": "1"})
    assert out.stdout.strip() == "ok", out.stderr
