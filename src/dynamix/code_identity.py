# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The source identity loaded by this process, captured before scientific imports."""
from __future__ import annotations

import importlib.metadata
import subprocess
from pathlib import Path

DISTRIBUTION = "geodynamix-beta"

def _git_dirs(root: Path) -> tuple[Path, Path] | None:
    """``(gitdir, commondir)`` of the checkout at ``root`` (a worktree's ``.git`` is a file)."""
    dot = root / ".git"
    if dot.is_dir():
        return dot, dot
    if dot.is_file():
        text = dot.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            gitdir = (root / text.split(":", 1)[1].strip()).resolve()
            common = gitdir
            if (gitdir / "commondir").is_file():
                common = (gitdir / (gitdir / "commondir").read_text(encoding="utf-8").strip()).resolve()
            return gitdir, common
    return None


def _head_commit(gitdir: Path, common: Path) -> str | None:
    """The commit at HEAD, read from the files (no git process)."""
    head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head or None
    ref = head.split(":", 1)[1].strip()
    for base in (gitdir, common):
        if (base / ref).is_file():
            return (base / ref).read_text(encoding="utf-8").strip()
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    return None


def _git(root: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=30)


def code_identity(root=None) -> dict:
    """The code that runs: ``{"commit", "version", "patch", "checked"}``.

    ``root`` defaults to the checkout this module runs from. ``commit`` is read from the git files;
    ``None`` for an install without a checkout, where ``version`` (the distribution's) names the
    code. ``patch``: the bytes that ``git apply`` turns the commit into the source that ran
    (``src/``, untracked files included), or ``None`` when it matches the commit. ``checked`` is
    False when git could not be asked (not installed, timed out), so the record says the working
    tree was not checked."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        version = importlib.metadata.version(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        version = None
    dirs = _git_dirs(root)
    if dirs is None:
        return {"commit": None, "version": version, "patch": None, "checked": False}
    commit = _head_commit(*dirs)
    try:
        status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", "src")
        if status.returncode != 0:
            raise OSError(status.stderr.decode(errors="replace"))
        lines = status.stdout.decode().splitlines()
        if not lines:
            return {"commit": commit, "version": version, "patch": None, "checked": True}
        patch = _git(root, "diff", "--binary", "HEAD", "--", "src").stdout
        for line in lines:
            if line.startswith("?? "):
                path = line[3:].strip().strip('"')
                patch += _git(root, "diff", "--binary", "--no-index", "--", "/dev/null",
                              path).stdout
        return {"commit": commit, "version": version, "patch": patch or None, "checked": True}
    except (OSError, subprocess.SubprocessError):
        return {"commit": commit, "version": version, "patch": None, "checked": False}



STARTUP = code_identity()


def checked_identity():
    """Refuse to cite edited on-disk code as the code already loaded into memory."""
    now = code_identity()
    if now != STARTUP:
        raise ValueError("source code changed since process startup; restart before writing provenance")
    if not now["checked"] and (now["commit"] is not None or not now["version"]):
        raise ValueError("source code identity could not be checked; provenance was not written")
    return dict(STARTUP)
