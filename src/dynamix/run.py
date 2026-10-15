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


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: {USAGE}", file=sys.stderr)
        return 2
    if argv[0] == "--list":
        for cid in sorted(HEADLESS):
            print(f"{cid}\t{by_id()[cid].title}")
        return 0
    cid, args = argv[0], dict(a.split("=", 1) for a in argv[1:] if "=" in a)
    if cid not in by_id():
        print(f"unknown command {cid!r}", file=sys.stderr)
        return 2
    if cid not in HEADLESS:
        print(f"{cid} needs a window: run it from the app", file=sys.stderr)
        return 2
    HEADLESS[cid](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
