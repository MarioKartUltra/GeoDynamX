# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""``python -m dynamix.run <command id> [key=value ...]``: the commands that need no window (the
settings ones), and ``--list`` to name them. Everything else answers that it needs a window.
``dynamix.shell.settings`` imports no Qt, and the test beside this module proves none loads."""
from __future__ import annotations

import sys

from dynamix.model.commands import by_id
from dynamix.shell.settings import update_settings

HEADLESS = {
    "settings.auto_run": lambda a: update_settings(auto_run_wtmm=a.get("on", "true").lower() == "true"),
    "settings.engine.auto": lambda a: update_settings(compute_engine="auto"),
    "settings.engine.mlx": lambda a: update_settings(compute_engine="mlx"),
    "settings.engine.fftw": lambda a: update_settings(compute_engine="fftw"),
    "settings.precision.32": lambda a: update_settings(compute_precision=32),
    "settings.precision.64": lambda a: update_settings(compute_precision=64),
}

USAGE = "python -m dynamix.run <command id> [key=value ...] | --list"


def _notes(settings) -> list[str]:
    """What the Settings menu says after the same change: the precision note for the engine chosen,
    an absent mlx, and that a running app applies the engine on restart."""
    from dynamix.core.fft_policy import precision_note

    notes = []
    note = precision_note(settings.compute_engine, int(settings.compute_precision))
    if note:
        notes.append(note)
    if settings.compute_engine == "mlx":
        try:
            import mlx.core  # noqa: F401
        except ImportError:
            notes.append("mlx is not installed here; FFTW3 will be used")
    notes.append("saved; a running GeoDynamix applies it on restart")
    return notes


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: {USAGE}", file=sys.stderr)
        return 2
    if argv[0] == "--list":
        for cid in sorted(HEADLESS):
            print(f"{cid}\t{by_id()[cid].title}")
        return 0
    cid, rest = argv[0], argv[1:]
    bare = [a for a in rest if "=" not in a]
    if bare:
        print(f"not key=value: {' '.join(bare)}", file=sys.stderr)
        return 2
    if cid not in by_id():
        print(f"unknown command {cid!r}", file=sys.stderr)
        return 2
    if cid not in HEADLESS:
        print(f"{cid} needs a window: run it from the app", file=sys.stderr)
        return 2
    settings = HEADLESS[cid](dict(a.split("=", 1) for a in rest))
    for note in _notes(settings):
        print(f"{cid}: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
