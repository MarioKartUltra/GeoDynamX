# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Every module in the package must at least COMPILE -- no test subset may leave a syntax
error undetected.

The pyvista-backed arrangement modules are excluded from both offscreen gates and imported
lazily, so a syntax error there (an unterminated string literal in ``shell/arrangement/scene.py``,
say) leaves the gates green and kills the app at launch the moment the Vector view activates.
``compileall`` costs ~a second and catches that class of error -- it needs no Qt, no pyvista,
no display: syntax only.
"""
from __future__ import annotations

import compileall
import pathlib


def test_every_module_compiles():
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "dynamix"
    assert compileall.compile_dir(str(src), quiet=2, force=False), \
        "a module in src/dynamix does not compile -- run: python -m compileall src/dynamix"
