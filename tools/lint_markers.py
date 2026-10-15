# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
# ヽ༼ຈل͜ຈ༽ﾉ The comment-marker lint ヽ༼ຈل͜ຈ༽ﾉ
"""The code-style markers, checked and reported (never enforced): the file header after the SPDX
and copyright lines, sections of exactly ten faces, sub-blocks of one, wavelet sections framed by
seven note pairs, uncertainty as a sad face with TODO, and the banned banners and check marks.

    python tools/lint_markers.py [FILE ...]     the files given, or the Python files changed against main
    python tools/lint_markers.py --all          every Python file under src/ and tools/

Always exits 0. Findings go to stdout as ``path:line: RULE: excerpt`` and, under GitHub Actions, to the
job summary. Tests and the verbatim copies are never checked; the rules are the code-style spec's.
"""
from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACE, SAD, HEAD = "☺", "☹", "ヽ༼ຈل͜ຈ༽ﾉ"
NOTES = "♪♫"

#: Never checked: the tests and the verbatim copies, whose bytes are pinned by their own tests.
EXCLUDED = ("tests/", "src/dynamix/_vendor/", "src/dynamix/core/selection.py",
            "src/dynamix/core/partition.py", "src/dynamix/core/modality.py",
            "src/dynamix/core/fftbackend.py", "src/dynamix/core/mzlib.py")

RULES = {
    "H1": "no file header (# " + HEAD + " title " + HEAD + ") after the SPDX and copyright lines",
    "H2": "more than one file header",
    "H3": "the file header is not the first comment after the SPDX and copyright lines",
    "S1": "a section is exactly ten faces, a sub-block exactly one",
    "W1": "a wavelet section is seven note pairs, the title, seven note pairs, with no spaces",
    "W2": "notes outside a wavelet-section header",
    "T1": "a sad face is followed by ' TODO: '",
    "B1": "banner lines (dashes, box drawing) are banned",
    "B2": "check marks are banned",
}

_HEADER = re.compile(r"^# " + re.escape(HEAD) + r" .+ " + re.escape(HEAD) + r"$")
_FACES = re.compile(r"^\s*#\s*(" + FACE + r"+)(.*)$")
_NOTES_HEADER = re.compile(r"^\s*#\s*((?:" + NOTES + r")+)(.*?)((?:" + NOTES + r")+)\s*$")
_BANNER = re.compile(r"^\s*#\s*[-─━=]{3,}")
_BOX = re.compile("[─-╿]")
_CHECK = re.compile("[✓✔]")
_SPDX = re.compile(r"^# SPDX-License-Identifier:")
_COPYRIGHT = re.compile(r"^# Copyright")


@dataclasses.dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    excerpt: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.excerpt}"


def excluded(path: Path) -> bool:
    rel = path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else path.as_posix()
    return any(rel.startswith(e) if e.endswith("/") else rel == e for e in EXCLUDED)


def check_text(text: str, path: str = "<text>") -> list[Finding]:
    """Every finding in ``text`` (a Python file's contents)."""
    lines = text.splitlines()
    out: list[Finding] = []
    headers = [i for i, line in enumerate(lines, 1) if _HEADER.match(line)]
    first_comment = next((i for i, line in enumerate(lines, 1)
                          if line.startswith("#") and not _SPDX.match(line)
                          and not _COPYRIGHT.match(line)), None)
    if not headers:
        out.append(Finding(path, 1, "H1", lines[0][:60] if lines else ""))
    else:
        if len(headers) > 1:
            out.append(Finding(path, headers[1], "H2", lines[headers[1] - 1][:60]))
        if first_comment is not None and first_comment != headers[0]:
            out.append(Finding(path, headers[0], "H3", lines[headers[0] - 1][:60]))
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        if _HEADER.match(line):
            continue
        m = _NOTES_HEADER.match(line)
        if m:
            left, title, right = m.groups()
            if len(left) != 14 or len(right) != 14 or not title or title != title.strip():
                out.append(Finding(path, i, "W1", stripped[:60]))
            continue
        if NOTES[0] in line or NOTES[1] in line:
            out.append(Finding(path, i, "W2", stripped[:60]))
        m = _FACES.match(line)
        if m:
            faces, rest = m.groups()
            n = len(faces)
            if n not in (1, 10) or not rest.startswith(" ") or not rest.strip():
                out.append(Finding(path, i, "S1", stripped[:60]))
        if SAD in line and f"{SAD} TODO: " not in line:
            out.append(Finding(path, i, "T1", stripped[:60]))
        if _BANNER.match(line) or _BOX.search(line):
            out.append(Finding(path, i, "B1", stripped[:60]))
        if _CHECK.search(line):
            out.append(Finding(path, i, "B2", stripped[:60]))
    return out


def check_file(path: Path) -> list[Finding]:
    """The findings of one file; none for an excluded one."""
    path = Path(path)
    if excluded(path):
        return []
    rel = path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else str(path)
    return check_text(path.read_text(encoding="utf-8"), rel)


def changed_files() -> list[Path]:
    """The Python files changed against ``main`` (or every file when git cannot say)."""
    try:
        base = subprocess.run(["git", "merge-base", "main", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        names = subprocess.run(["git", "diff", "--name-only", base, "HEAD", "--"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.split()
        names += subprocess.run(["git", "diff", "--name-only", "--"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return all_files()
    return sorted({ROOT / n for n in names if n.endswith(".py") and (ROOT / n).is_file()})


def all_files() -> list[Path]:
    return sorted(p for folder in ("src", "tools") for p in (ROOT / folder).rglob("*.py"))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args == ["--all"]:
        files = all_files()
    elif args:
        files = [Path(a) for a in args]
    else:
        files = changed_files()
    findings = [f for path in files for f in check_file(path)]
    for f in findings:
        print(f)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"## Marker lint: {len(findings)} finding(s) in {len(files)} file(s)\n\n")
            for f in findings:
                fh.write(f"- `{f.path}:{f.line}` {f.rule} ({RULES[f.rule]}): `{f.excerpt}`\n")
    print(f"{len(findings)} finding(s) in {len(files)} file(s); exit 0 by design")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
