# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The third-party policy holds by test: THIRD_PARTY.md and the third_party/ trees are the same set, each
tree carries its LICENSE, and NOTICE names exactly the files whose header says GPL-2.0-only."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "dynamix"


def _table_rows(text: str) -> list[list[str]]:
    """The cells of every data row of the first Markdown table in ``text``."""
    rows = []
    for line in text.splitlines():
        if not line.startswith("|") or set(line.replace("|", "").strip()) <= {"-", " "}:
            continue
        rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows[1:]                      # the first row is the header


def test_third_party_md_lists_every_tree_and_each_has_a_license():
    page = ROOT / "THIRD_PARTY.md"
    assert page.is_file()
    rows = _table_rows(page.read_text(encoding="utf-8"))
    listed = {row[0] for row in rows}
    folder = ROOT / "third_party"
    trees = {p.name for p in folder.iterdir() if p.is_dir()} if folder.is_dir() else set()
    assert listed == trees, (sorted(listed), sorted(trees))
    for row in rows:
        name, url, commit, licence, path, _used_by = row
        assert (folder / name / "LICENSE").is_file(), name
        assert re.fullmatch(r"[0-9a-f]{7,40}", commit), (name, commit)
        assert url.startswith("https://") and path.startswith("third_party/"), name
        assert licence, name


def test_notice_and_spdx_headers_agree():
    """The xsmurf block of NOTICE names exactly the files whose header says GPL-2.0-only (the
    verbatim-copy modules NOTICE lists further down are GPL-2.0-or-later and carry no header)."""
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    block = notice.split("xsmurf -- ", 1)[1].split("\n\n", 1)[0]
    assert "GPL-2.0" in block
    named = {m.group(0) for m in re.finditer(r"src/dynamix/[\w/]+\.py", block)}
    only = {p.relative_to(ROOT).as_posix() for p in SRC.rglob("*.py")
            if "SPDX-License-Identifier: GPL-2.0-only" in p.read_text(encoding="utf-8")[:300]}
    assert only, "no GPL-2.0-only header anywhere: the xsmurf-derived files lost theirs"
    assert named == only, {"NOTICE only": sorted(named - only), "header only": sorted(only - named)}
