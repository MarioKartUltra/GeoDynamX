# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Save/open project through the window -- the File-menu wiring over model/projectfile.py.

The machinery (save_project/open_project) has its own suite in test_projectfile.py; these tests
cover the WINDOW's half: registries rebuilt, panel repopulated, fields reopened once per source,
missing sources skipped with a visible notice rather than refusing the project.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_shell_window import (STUB_CHAIN, _FIELD, stub_devices, window)  # noqa: F401


@pytest.fixture
def raster(tmp_path):
    p = tmp_path / "field.npz"
    np.savez(p, np.asarray(_FIELD, dtype=np.float64))
    return p


@pytest.fixture
def saved_project(qtbot, window, raster, tmp_path):
    with qtbot.waitSignal(window.resolved, timeout=10000):
        window.open_path(str(raster))
    written = window._save_project_to(tmp_path / "session.dynamix")
    return written


def test_save_writes_a_dynamix_file_and_tracks_the_path(saved_project, window):
    assert saved_project.is_file()
    assert window._project_path == saved_project
    assert "session" in window.windowTitle()


def test_open_project_restores_layers_chain_and_display(qtbot, stub_devices, saved_project):
    from dynamix.shell.main_window import MainWindow

    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=10000):
        win._open_project_path(saved_project)
    assert len(win.project.layers) == 1
    restored = win.project.layers[0]
    assert [ref.device for ref in restored.chain.steps] == [n for n, _ in STUB_CHAIN]
    assert win.layer is restored                      # first layer selected
    assert win.canvas._field is not None              # raster displayed
    assert win.layer_list.current_layer_id() == restored.layer_id


def test_open_project_with_missing_source_skips_and_notices(qtbot, stub_devices, saved_project,
                                                            raster):
    from dynamix.shell.main_window import MainWindow

    raster.rename(raster.with_name("moved-away.npz"))
    win = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win)
    win._open_project_path(saved_project)             # must not raise, must not dispatch
    assert win.project.layers                         # the project still knows the layer
    assert win.layer is None                          # nothing selectable was restored
    assert win.layer_list.current_layer_id() is None
    assert "missing" in win.strips.reading_label.text().lower()


# -- transects ----------------------------------------
#
# The WINDOW's half of the round trip (panel rows + the drawn line reaching the canvas), mirroring
# this file's own stated split with test_projectfile.py, which covers Project.transects/
# TransectRecord serialization directly (additive-key discipline, old-file tolerance).


def test_open_project_restores_the_transect_panel_and_line(qtbot, stub_devices, window, raster,
                                                            tmp_path):
    from dynamix.shell.main_window import MainWindow

    with qtbot.waitSignal(window.resolved, timeout=10000):
        window.open_path(str(raster))
    original = window._transect_panel.add_record((1.0, 2.0), (8.0, 2.0))
    written = window._save_project_to(tmp_path / "session.dynamix")

    win2 = MainWindow(steps=STUB_CHAIN)
    qtbot.addWidget(win2)
    with qtbot.waitSignal(win2.resolved, timeout=10000):
        win2._open_project_path(written)

    (restored,) = win2._transect_panel.records()
    assert restored.transect_id == original.transect_id
    assert restored.a == (1.0, 2.0) and restored.b == (8.0, 2.0)
    assert win2._transect_panel._list.count() == 1
    assert win2._transect_panel._list.item(0).text() == f"T{original.transect_id}"
    xs, ys = win2.canvas.transect_item.getData()
    assert xs is not None and xs.size > 0                   # the line reached the canvas
