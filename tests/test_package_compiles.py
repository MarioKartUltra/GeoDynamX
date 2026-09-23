# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Every module in the package must at least COMPILE -- no test subset may leave a syntax
error undetected.

Exists because of 2026-09-16: a docstring edit left ``shell/arrangement/scene.py`` with an
unterminated string literal, both offscreen gates stayed green (the pyvista-backed arrangement
modules are excluded from them and imported lazily), and the app died at launch the moment the
Vector view activated. ``compileall`` costs ~a second and catches that class forever -- it
needs no Qt, no pyvista, no display: syntax only.
"""
from __future__ import annotations

import compileall
import pathlib


def test_every_module_compiles():
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "dynamix"
    assert compileall.compile_dir(str(src), quiet=2, force=False), \
        "a module in src/dynamix does not compile -- run: python -m compileall src/dynamix"
