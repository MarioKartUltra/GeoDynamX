# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Each bundled theme on the main window, grabbed offscreen into docs/grabs/ for a look."""
from __future__ import annotations

import pathlib

from PySide6 import QtWidgets

from dynamix.shell.theme import RESTRAINED_DARK, apply_theme, bundled_themes, current, load_theme
from tests.test_shell_window import _FIELD, loaded, stub_devices, window  # noqa: F401

GRABS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "grabs"


def test_every_bundled_theme_grabs_the_window(qtbot, loaded):
    w = loaded
    w.resize(1100, 700)
    w.show()
    qtbot.waitExposed(w)
    app = QtWidgets.QApplication.instance()
    GRABS.mkdir(parents=True, exist_ok=True)
    try:
        for name in bundled_themes():
            apply_theme(app, load_theme(name))
            w.canvas.apply_theme_roles()            # what a Settings ▸ Theme pick does
            assert current().name == name
            for _ in range(3):
                app.processEvents()
            pix = w.grab()
            assert not pix.isNull() and pix.width() > 0
            assert pix.save(str(GRABS / f"theme_{name}.png"))
    finally:
        apply_theme(app, RESTRAINED_DARK)
