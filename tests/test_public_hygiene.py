# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The public tree carries no development-conversation residue.

Every tracked text file is scanned for the shapes that residue takes: dated attributions and
quotes ("user, 2026-...: ..."), references to development notes that are not part of this
repository (handoffs, specs, plans, review rounds, agent tooling), and machine-specific paths.
A hit fails with the file and line, so it is fixed where it was written.

Exempt, each for a stated reason:

- ``LICENSE`` -- the FSF text, verbatim.
- ``.gitignore`` -- it NAMES the local tooling files precisely so they are never committed.
- ``src/dynamix/_vendor/`` -- verbatim copies of the author's own packages.
- ``mzlib.py`` -- a byte-verified copy of research code; editing its comments would break the
  verbatim guard in ``tests/test_mzlib_port.py``.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

EXEMPT = {
    "LICENSE",
    ".gitignore",
    "tests/test_public_hygiene.py",
    "src/dynamix/core/mzlib.py",
}
EXEMPT_DIRS = ("src/dynamix/_vendor/",)

FORBIDDEN = {
    "dated attribution": r"\b[Uu]ser(?:'s)?(?: \w+)?,? ?\(?20\d\d-\d\d-\d\d",
    "attributed decision": r"\b[Uu]ser(?:'s)? (?:decision|correction|ask|request|rule|question|words|model"
                           r"|accepted|agreed|approved|insisted|confirmed)\b",
    "attributed quote": r"\b[Uu]ser\b[^\n\"“]{0,20}: ?[\"“]",
    "handoff note": r"\b[Hh]andoff\b|HANDOFF",
    "development docs": r"\.superpowers|\bdocs/(?:specs|plans|research|superpowers|audit)\b",
    "agent tooling": r"CLAUDE\.md|Co-Authored-By|Claude-Session|claude\.ai",
    "review round": r"\b[Rr]eview round\b|\b[Rr]e-review|final-review|\b[Ff]inding [IMC]?\d",
    "plan task": r"\bplan,? Task \d|\bTask \d+'s\b|\btask brief\b",
    "assistant memory": r"\bmemory:? [a-z]+(?:-[a-z]+){2,}\b|\[\[[a-z0-9]+(?:-[a-z0-9]+){2,}\]\]",
    "machine path": r"/Users/[a-z]|/home/[a-z]",
}

#: A space in a pattern matches any gap, including a line break inside a comment block, so a
#: phrase wrapped across two lines is still caught.
_GAP = r"(?:[ \t]+|[ \t]*\n[ \t]*(?:#:?[ \t]*)?)"


def _tracked_text_files():
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    for rel in filter(None, out.split("\0")):
        if rel in EXEMPT or rel.startswith(EXEMPT_DIRS):
            continue
        path = REPO / rel
        try:
            yield rel, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue                     # binary or deleted in the working tree


@pytest.mark.parametrize("label", sorted(FORBIDDEN))
def test_no_development_residue(label):
    pattern = re.compile(FORBIDDEN[label].replace(" ", _GAP))
    hits = [f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: {m.group(0)!r}"
            for rel, text in _tracked_text_files() for m in pattern.finditer(text)]
    assert not hits, f"{label} in the public tree:\n" + "\n".join(hits[:40])
