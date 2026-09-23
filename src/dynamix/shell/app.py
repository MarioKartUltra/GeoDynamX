# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The application entry point: build the QApplication, dress it, show the window.

    python -m dynamix.shell.app                      # the demo DEM
    python -m dynamix.shell.app field.npz            # any raster RasterField can read
    python -m dynamix.shell.app field.npz --render out.png   # headless frame, for verification

This is the ONLY place :func:`~dynamix.shell.theme.apply_theme` is called. A theme is an
application-wide stylesheet plus a default font; applying it anywhere else means two callers can
disagree about what the app looks like, and a widget-level stylesheet is exactly the literal-color
escape hatch the Theme Rule exists to close.

``--render`` mirrors demo.py's headless verification mode, with one addition it needs and demo.py
did not: the first resolve runs on the WORKER thread, so the render path pumps the event loop until
the window reports itself idle before grabbing the frame. Grabbing earlier would save a picture of
an empty canvas and call it a verified render.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from dynamix.shell.main_window import MainWindow
from dynamix.shell.settings import load_settings
from dynamix.shell.theme import RESTRAINED_DARK, apply_theme

#: How long ``--render`` waits for the first transform before grabbing anyway. Generous: this is a
#: WTMM run over whatever raster the caller passed, not a UI animation.
RENDER_TIMEOUT_S = 600.0

USAGE = "python -m dynamix.shell.app [field.npz] [--render out.png]"


def _default_raster() -> str:
    """The raster to open when the command line names none: ``Settings.open_on_launch`` if it
    points at an existing file, else the demo DEM, else the smallest test fixture."""
    wanted = load_settings().open_on_launch
    if wanted and Path(wanted).expanduser().is_file():
        return str(Path(wanted).expanduser())
    root = Path(__file__).resolve().parents[3]
    demo = root / "docs" / "demo" / "dem_crop.npz"
    return str(demo if demo.exists() else root / "tests" / "fixtures" / "kam_64.npz")


def main(argv=None) -> int:
    # A VTK/Cocoa segfault otherwise dies with no Python frame in the crash report (two of them
    # on 2026-08-25); faulthandler prints the Python stack to stderr first.
    import faulthandler
    try:
        faulthandler.enable()
    except (ValueError, OSError, AttributeError, io.UnsupportedOperation):
        pass        # a captured stderr (pytest) has no fileno; nothing to do
    argv = list(sys.argv[1:] if argv is None else argv)
    render_to = None
    if "--render" in argv:
        i = argv.index("--render")
        if i + 1 >= len(argv):
            print(f"usage: {USAGE}", file=sys.stderr)
            return 2
        render_to = argv[i + 1]
        del argv[i:i + 2]
    raster = argv[0] if argv else _default_raster()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_theme(app, RESTRAINED_DARK)
    win = MainWindow()
    win.resize(1400, 860)
    win.show()
    win.open_path(raster)

    if render_to:
        deadline = time.monotonic() + RENDER_TIMEOUT_S
        while win.is_computing and time.monotonic() < deadline:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        for _ in range(3):
            app.processEvents()
        win.grab().save(render_to)
        print(f"rendered -> {render_to}"
              f"{'' if not win.is_computing else '  (WARNING: still computing)'}")
        return 0

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
