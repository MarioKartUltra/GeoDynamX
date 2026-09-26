# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The public tree carries no development-conversation residue.

Every tracked text file is scanned for the shapes that residue takes: dated attributions and
quotes ("user, 2026-...: ..."), references to development notes that are not part of this
repository (handoffs, specs, plans, review rounds, agent tooling), and machine-specific paths.
A hit fails with the file and line, so it is fixed where it was written.

Comments and docstrings are held to a stricter rule: they state the technical constraint, so a
full date or a "(user" attribution in one is history and fails wherever it appears. Dates that
are the data a comment describes are listed in ``ALLOWED_DATES`` with their reason.

Exempt, each for a stated reason:

- ``LICENSE`` -- the FSF text, verbatim.
- ``.gitignore`` -- it NAMES the local tooling files precisely so they are never committed.
- ``src/dynamix/_vendor/`` -- verbatim copies of the author's own packages.
- ``mzlib.py`` -- a byte-verified copy of research code; editing its comments would break the
  verbatim guard in ``tests/test_mzlib_port.py``.
"""
from __future__ import annotations

import ast
import io
import pathlib
import re
import subprocess
import tokenize

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


#: A full date, or a "(user" attribution, inside a comment or docstring. Comments are read as
#: tokens (Python, TOML, shell) and docstrings from the AST, so dates in fixtures and other data
#: strings are out of its reach. The app's own user ("(user cancelled)") is not an attribution.
HISTORY_IN_COMMENTS = re.compile(
    r"\b20\d\d-\d\d-\d\d\b"
    r"|\([Uu]ser(?:'s)?(?:\s*[:,]| (?:asked|chose|decided|wanted|said|saw|noted|reported|flagged"
    r"|insisted|requested|preferred|picked|call|choice|decision|request|mockups?)\b)")

#: Dates that are the data a comment describes: (file, date) -> reason.
ALLOWED_DATES = {
    ("src/dynamix/geo/footprints.py", "2015-11-28"): "an ASTER granule's acquisition date, the "
                                                     "label format's worked example",
    ("tests/test_footprint_browser.py", "2015-11-28"): "fixture granule acquisition date",
    ("tests/test_footprint_browser.py", "2018-10-12"): "fixture granule acquisition date",
}

_HASH_COMMENTED = (".toml", ".sh", ".cff")


def _comments_and_docstrings(rel, text):
    """Yield (line, text) for each comment and each docstring line of a tracked file."""
    if rel.endswith(".py"):
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
            tree = ast.parse(text)
        except (tokenize.TokenError, SyntaxError):
            return
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                yield tok.start[0], tok.string
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                for i, line in enumerate(node.value.value.splitlines()):
                    yield node.lineno + i, line
    elif rel.endswith(_HASH_COMMENTED):
        for n, line in enumerate(text.splitlines(), 1):
            m = re.search(r"(?:^|\s)(#.*)$", line)
            if m:
                yield n, m.group(1)


def test_no_history_in_comments():
    hits = [f"{rel}:{line}: {m.group(0)!r}"
            for rel, text in _tracked_text_files()
            for line, comment in _comments_and_docstrings(rel, text)
            for m in HISTORY_IN_COMMENTS.finditer(comment)
            if (rel, m.group(0)) not in ALLOWED_DATES]
    assert not hits, "history in a comment or docstring:\n" + "\n".join(hits[:40])
