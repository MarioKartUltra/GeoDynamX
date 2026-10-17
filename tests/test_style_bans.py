# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The anti-slop bans that a grep can see, reported for review and never failing on the tree
(spec 10 §3.2, bans 2, 6, 8, 10 and 11): bare or blanket excepts that swallow, emoji outside the
markers, dates or conversation markers in code, files over 1,500 lines, banner lines and check marks. Each
test prints its findings; ``-s`` shows them, and the CI job summary carries them."""
from __future__ import annotations

import os
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "dynamix"
VERBATIM = {"selection.py", "partition.py", "modality.py", "fftbackend.py", "mzlib.py"}
MARKERS = set("☺☹♪♫ヽ༼ຈل͜༽ﾉ")
#: The two files the spec exempts from the size ban; they may only shrink.
SIZE_EXEMPT = {"shell/main_window.py", "shell/arrangement/scene.py"}

_SWALLOW = re.compile(r"^\s*except(\s+Exception)?\s*:\s*(#.*)?$")
_PASS = re.compile(r"^\s*pass\s*(#.*)?$")
_DATE = re.compile(r"\b20[0-9]{2}-[01][0-9]-[0-3][0-9]\b")
_MARKER = re.compile(r"\(user:|\(reviewer:|\(agent:")
_BANNER = re.compile(r"^\s*#\s*[-─━=]{3,}")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def _files():
    for p in sorted(SRC.rglob("*.py")):
        rel = p.relative_to(SRC).as_posix()
        if rel.startswith("_vendor/") or p.name in VERBATIM:
            continue
        yield rel, p


def _report(title: str, findings: list[str]) -> None:
    print(f"\n{title}: {len(findings)} finding(s)")
    for f in findings:
        print("  " + f)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"## {title}: {len(findings)} finding(s)\n\n")
            fh.writelines(f"- `{f}`\n" for f in findings)


def test_ban_2_no_except_that_swallows():
    findings = []
    for rel, p in _files():
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if _SWALLOW.match(line) and i + 1 < len(lines) and _PASS.match(lines[i + 1]):
                findings.append(f"{rel}:{i + 1}: {line.strip()} / pass")
    _report("ban 2, except that swallows", findings)


def test_ban_6_no_emoji_outside_the_markers():
    findings = []
    for rel, p in _files():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for ch in _EMOJI.findall(line):
                if ch not in MARKERS:
                    findings.append(f"{rel}:{i}: {ch!r} in {line.strip()[:50]}")
                    break
    _report("ban 6, emoji outside the markers", findings)


def test_ban_8_no_dates_or_conversation_markers_in_code():
    """Dates, and the markers that attribute a line to someone in a conversation; agent names
    are the local hygiene test's job."""
    findings = []
    for rel, p in _files():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if _DATE.search(line) or _MARKER.search(line):
                findings.append(f"{rel}:{i}: {line.strip()[:60]}")
    _report("ban 8, dates or conversation markers in code", findings)


def test_ban_10_no_file_over_1500_lines():
    findings = []
    for rel, p in _files():
        n = len(p.read_text(encoding="utf-8").splitlines())
        if n > 1500 and rel not in SIZE_EXEMPT:
            findings.append(f"{rel}: {n} lines")
    _report("ban 10, files over 1,500 lines (main_window and scene exempt)", findings)


def test_ban_11_no_banner_lines_or_check_marks():
    findings = []
    for rel, p in _files():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if _BANNER.match(line) or "✓" in line or "✔" in line:
                findings.append(f"{rel}:{i}: {line.strip()[:60]}")
    _report("ban 11, banner lines and check marks", findings)


@pytest.mark.parametrize("rel", sorted(SIZE_EXEMPT))
def test_the_exempt_files_exist_so_the_exemption_is_not_stale(rel):
    assert (SRC / rel).is_file(), rel
