# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Fast fork (Y): the active layer's shown result forked without the dialog, with the placement,
keep and contents the Fast Fork settings name."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore

from dynamix.core.rasterfield import RasterField
from dynamix.shell.fork_dialog import FastForkSettingsDialog, fast_fork_choice
from dynamix.shell.settings import FAST_FORK_DEFAULTS, load_settings, update_settings
from tests.test_field_preview import _add_noise, _stack
from tests.test_shell_window import stub_devices, window  # noqa: F401


class _NoDialog:
    def __init__(self, **_kw):
        raise AssertionError("a fast fork opens no dialog")


def test_the_choice_takes_the_shown_raster_or_every_band_of_a_produced_field():
    rasters = [("as shown", None), ("r2", None)]
    shown = fast_fork_choice(FAST_FORK_DEFAULTS, name="n", rasters=rasters, produced=False, vectors=(5, 2))
    assert shown == {"name": "n", "bands": [0], "vectors": False, "nest": False, "temporary": True}
    every = fast_fork_choice({**FAST_FORK_DEFAULTS, "rasters": "all"}, name="n", rasters=rasters,
                             produced=False, vectors=(0, 0))
    assert every["bands"] == [0, 1]
    produced = fast_fork_choice(FAST_FORK_DEFAULTS, name="n", rasters=rasters, produced=True, vectors=(0, 0))
    assert produced["bands"] == [0, 1]                    # a produced stack is shown whole
    lines_only = fast_fork_choice(FAST_FORK_DEFAULTS, name="n", rasters=[], produced=False, vectors=(5, 2))
    assert lines_only["bands"] == [] and lines_only["vectors"]


def test_y_is_the_fast_fork_shortcut_and_needs_a_raster(window):
    action = window.commands["layer.fast_fork"]
    assert [s.toString() for s in action.shortcuts()] == ["Y"]
    assert action.shortcutContext() == QtCore.Qt.WindowShortcut
    window.refresh_commands()
    assert not action.isEnabled()


def test_a_fast_fork_writes_what_is_shown_as_a_temporary_dataset_of_its_own(window, qtbot, monkeypatch):
    import dynamix.shell.main_window as mw

    win = window
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_stack(3), "mem:stack")
    _add_noise(win, qtbot)
    noised = np.asarray(win.canvas._field.values).copy()
    parent = win.layer.source_id
    monkeypatch.setattr(mw, "ForkDialog", _NoDialog)
    before = set(win.project.sources)
    win.commands["layer.fast_fork"].trigger()
    (new,) = set(win.project.sources) - before
    source = win.project.sources[new]
    assert source.temporary and source.nest_under is None
    assert source.label.startswith("mem:stack · ")             # the dialog's own default name
    np.testing.assert_array_equal(RasterField.from_file(source.path).values, noised)


def test_the_fast_fork_settings_choose_the_placement(window, qtbot, monkeypatch):
    import dynamix.shell.main_window as mw

    win = window
    update_settings(fast_fork={**FAST_FORK_DEFAULTS, "nest": True})
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win.load_field(_stack(1), "mem:stack")
    _add_noise(win, qtbot)
    parent = win.layer.source_id
    monkeypatch.setattr(mw, "ForkDialog", _NoDialog)
    before = set(win.project.sources)
    win.commands["layer.fast_fork"].trigger()
    (new,) = set(win.project.sources) - before
    assert win.project.sources[new].nest_under == parent


def test_the_settings_dialog_edits_and_returns_the_four_choices(qtbot):
    dialog = FastForkSettingsDialog(FAST_FORK_DEFAULTS)
    qtbot.addWidget(dialog)
    assert dialog.choices() == FAST_FORK_DEFAULTS
    dialog._nest.setChecked(True)
    dialog._file.setChecked(True)
    dialog._all_rasters.setChecked(True)
    dialog._vectors.setChecked(True)
    assert dialog.choices() == {"nest": True, "temporary": False, "rasters": "all", "vectors": True}


def test_the_fast_fork_settings_persist_and_a_bad_entry_falls_back(tmp_path, monkeypatch):
    assert load_settings().fast_fork == FAST_FORK_DEFAULTS
    update_settings(fast_fork={**FAST_FORK_DEFAULTS, "rasters": "all"})
    assert load_settings().fast_fork["rasters"] == "all"
    update_settings(fast_fork={"nest": "yes", "rasters": "some"})
    assert load_settings().fast_fork == FAST_FORK_DEFAULTS
