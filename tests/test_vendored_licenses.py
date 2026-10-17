# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Every vendored or derived tree carries its licence: a LICENSE file beside it or an entry in NOTICE;
every file whose SPDX header says GPL-2.0-only is one NOTICE names."""
from __future__ import annotations

import re
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "dynamix"
NOTICE = (REPO / "NOTICE").read_text(encoding="utf-8")
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
    """A LICENSE beside the directory, or its repo path written in NOTICE (a bare directory name
    is not an entry: "wtmm" appears in prose without naming a tree)."""
    rel = d.relative_to(REPO).as_posix()
    has_file = any((d / n).exists() for n in ("LICENSE", "LICENSE.txt", "COPYING"))
    return has_file or re.search(re.escape(rel) + r"(?!\w)", notice) is not None


def test_every_derived_tree_has_a_licence_file_or_a_notice_entry():
    for root in DERIVED_ROOTS:
        for d in _package_dirs(root):
            assert _entry_for(d, NOTICE), f"{d.relative_to(REPO)}: no LICENSE beside it and no NOTICE entry"


def test_notice_lists_every_gpl_2_only_file():
    only = []
    for path in SRC.rglob("*.py"):
        head = path.read_text(encoding="utf-8").splitlines()[:3]
        if any("SPDX-License-Identifier: GPL-2.0-only" in line for line in head):
            only.append(path.relative_to(REPO).as_posix())
    assert only, "the xsmurf ports carry GPL-2.0-only headers"
    missing = [p for p in only if p not in NOTICE]
    assert not missing, f"GPL-2.0-only files absent from NOTICE: {missing}"


def test_the_licence_check_fails_when_a_tree_loses_its_entry():
    for rel, word in (("core/mz_lastwave", "src/dynamix/core/mz_lastwave"),
                      ("_vendor/wtmm", "src/dynamix/_vendor/wtmm"),
                      ("_vendor/wtmm_ebsd", "src/dynamix/_vendor/wtmm_ebsd")):
        struck = NOTICE.replace(word, "removed")
        assert struck != NOTICE, word
        assert not _entry_for(SRC / rel, struck), rel
    assert _entry_for(SRC / "_vendor" / "wtmm", NOTICE.replace("wtmm_ebsd", "x"))   # its own path still names it


def test_a_longer_path_does_not_stand_in_for_a_shorter_one():
    """``src/dynamix/_vendor/wtmm`` is a prefix of ``src/dynamix/_vendor/wtmm_ebsd``: striking only the
    shorter path must fail the shorter tree while the longer one keeps its entry."""
    struck = re.sub(r"src/dynamix/_vendor/wtmm(?!\w)", "removed", NOTICE)
    assert struck != NOTICE
    assert "src/dynamix/_vendor/wtmm_ebsd" in struck
    assert not _entry_for(SRC / "_vendor" / "wtmm", struck)
    assert _entry_for(SRC / "_vendor" / "wtmm_ebsd", struck)


def test_the_bare_name_is_not_an_entry():
    assert not _entry_for(SRC / "_vendor" / "wtmm", "wtmm appears in prose, no path named")
