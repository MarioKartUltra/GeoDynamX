# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Text read or written without an encoding uses the platform's codec: UTF-8 on macOS and Linux,
cp1252 on Windows, where a source with a Greek letter fails to read and a project or .gprv written
there reads back wrong elsewhere. Every text open(), read_text() and write_text() names one."""
from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Verbatim copies change only by a deliberate decision (tests/test_no_silent_drift.py); partition's
#: two text calls write and read json.dumps output, which is ASCII and reads back under any codec.
_VERBATIM = {"src/dynamix/core/partition.py"}

#: Calls that open text unless told otherwise.
_TEXT_CALLS = {"open", "read_text", "write_text"}


def _mode(call: ast.Call) -> "str | None":
    """The literal mode of an open() call, or None when it is not given or not a literal."""
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")
    pos = 0 if isinstance(call.func, ast.Attribute) and name == "open" else 1
    if len(call.args) > pos and isinstance(call.args[pos], ast.Constant):
        return call.args[pos].value
    return None


def _unnamed(path: pathlib.Path) -> list:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name not in _TEXT_CALLS or any(kw.arg == "encoding" for kw in node.keywords):
            continue
        if any(kw.arg is None for kw in node.keywords):        # **kwargs: can't tell
            continue
        if name == "open":
            # builtin open(path, mode) or Path.open(mode); other objects' .open (rasterio, zip,
            # Image) take no text mode and are skipped by their mode or receiver below
            if isinstance(f, ast.Attribute) and not (isinstance(f.value, ast.Call)
                                                     or isinstance(f.value, ast.Name)):
                continue
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and \
                    f.value.id in {"rasterio", "Image", "zipfile", "gzip", "io", "os", "tarfile",
                                   "webbrowser", "h5py", "zf", "zip_file", "z", "src", "ds"}:
                continue
            mode = _mode(node)
            if mode is not None and "b" in str(mode):
                continue
        out.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return out


def test_every_text_open_names_its_encoding():
    found = [hit for d in ("src", "tests") for p in sorted((ROOT / d).rglob("*.py"))
             if "_vendor" not in p.parts and str(p.relative_to(ROOT)) not in _VERBATIM
             for hit in _unnamed(p)]
    assert not found, "\n".join(found)
