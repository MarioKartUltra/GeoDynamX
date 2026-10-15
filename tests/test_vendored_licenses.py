# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Every vendored or derived tree carries its licence: a LICENSE file beside it or an entry in NOTICE;
every file whose SPDX header says GPL-2.0-only is one NOTICE names."""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "dynamix"
NOTICE = (REPO / "NOTICE").read_text()
DERIVED_ROOTS = (SRC / "_vendor", SRC / "core" / "mz_lastwave")


def _package_dirs(root):
    """``root`` itself when it holds .py files, plus every subdirectory that does; caches skipped."""
    out = []
    for d in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if d.name == "__pycache__":
            continue
        if any(f.suffix == ".py" for f in d.iterdir() if f.is_file()):
            out.append(d)
    return out


def _entry_for(d, notice):
    rel = d.relative_to(REPO).as_posix()
    has_file = any((d / n).exists() for n in ("LICENSE", "LICENSE.txt", "COPYING"))
    named = rel in notice or (rel + "/") in notice or d.name in notice
    return has_file or named


def test_every_derived_tree_has_a_licence_file_or_a_notice_entry():
    for root in DERIVED_ROOTS:
        for d in _package_dirs(root):
            assert _entry_for(d, NOTICE), f"{d.relative_to(REPO)}: no LICENSE beside it and no NOTICE entry"


def test_notice_lists_every_gpl_2_only_file():
    only = []
    for path in SRC.rglob("*.py"):
        head = path.read_text().splitlines()[:3]
        if any("SPDX-License-Identifier: GPL-2.0-only" in line for line in head):
            only.append(path.relative_to(REPO).as_posix())
    assert only, "the xsmurf ports carry GPL-2.0-only headers"
    missing = [p for p in only if p not in NOTICE]
    assert not missing, f"GPL-2.0-only files absent from NOTICE: {missing}"


def test_the_licence_check_fails_when_a_tree_loses_its_entry():
    struck = NOTICE.replace("mz_lastwave", "removed")
    assert struck != NOTICE
    assert not _entry_for(SRC / "core" / "mz_lastwave", struck)
