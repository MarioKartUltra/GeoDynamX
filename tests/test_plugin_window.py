# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The window and the user's devices folder: a broken file beside a good one is a greyed row and
the app still starts; a plugin layer's step keys are the engine's own, so Run on a computed plugin
layer says so (the shell's walk of a chain carries the plugin's origin)."""
from __future__ import annotations

from dynamix.engine.resolve import resolve
from dynamix.model.device import DEVICES
from dynamix.shell.settings import devices_dir
from tests.test_browser import _top_level
from tests.test_plugins import BROKEN, GOOD
from tests.test_shell_roi_flow import _parent_field, parent_tif  # noqa: F401


def _folder(*files):
    folder = devices_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in files:
        (folder / name).write_text(text, encoding="utf-8")
    return folder


def test_the_app_starts_with_a_good_and_a_broken_device_file(qtbot, clean_registry):
    from dynamix.shell.main_window import MainWindow

    _folder(("double.py", GOOD), ("broken.py", BROKEN))
    win = MainWindow()
    qtbot.addWidget(win)
    errors = [r for r in win.plugin_results if r.error]
    assert len(errors) == 1 and "SyntaxError" in errors[0].error and "plug_double" in DEVICES
    labs = _top_level(win.browser, "Labs")
    assert [labs.child(i).text(1) for i in range(labs.childCount())] == ["plug_double"]
    dev = _top_level(win.browser, "Dev")
    broken = next(dev.child(i) for i in range(dev.childCount()) if dev.child(i).text(0) == "broken.py")
    assert "SyntaxError" in broken.toolTip(0) and dev.isExpanded()       # a failure opens Dev


def test_a_computed_plugin_layer_is_already_computed_on_run(qtbot, clean_registry, parent_tif):
    from dynamix.shell.main_window import MainWindow

    _folder(("double.py", GOOD))
    win = MainWindow(steps=(("plug_double", {"gain": 2.0}),))
    qtbot.addWidget(win)
    with qtbot.waitSignal(win.resolved, timeout=60000):
        win.load_field(_parent_field(parent_tif), parent_tif)
    keys = win._cache_keys_for(win.layer)
    assert keys and keys[-1] == resolve(win.layer, win.field, win.cache).analysis_key
    assert keys[-1] in win.cache
    win._run_transforms()
    assert "already computed" in win.statusBar().currentMessage()
