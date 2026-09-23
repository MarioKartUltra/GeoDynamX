# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Drive the demo window through a set of states and render each to PNG.

This is the headless verification path: it proves the core gesture works -- that moving a filter
changes the picture and costs a fraction of a millisecond -- without anyone having to watch a
screen. Run it after touching the engine, the devices, or the shell.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=src python scripts/render_demo_states.py OUTDIR
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6 import QtWidgets

from dynamix.shell.demo import DemoWindow, _default_npz

#: (label, {device: {param: value}}) -- each applied on top of the demo's defaults.
STATES = [
    ("01_scale00_all", {"scale_select": {"scale_idx": 0}}),
    ("02_scale04_all", {"scale_select": {"scale_idx": 4}}),
    ("03_scale08_all", {"scale_select": {"scale_idx": 8}}),
    ("04_scale11_all", {"scale_select": {"scale_idx": 11}}),
    ("05_scale02_wedge040", {"scale_select": {"scale_idx": 2},
                             "orientation_wedge": {"centre": 40.0, "half_width": 15.0}}),
    ("06_scale02_wedge130", {"scale_select": {"scale_idx": 2},
                             "orientation_wedge": {"centre": 130.0, "half_width": 15.0}}),
    ("07_scale02_denoised", {"scale_select": {"scale_idx": 2},
                             "modulus_threshold": {"frac": 0.35}}),
]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    outdir = Path(argv[0] if argv else "/tmp/dxdemo")
    outdir.mkdir(parents=True, exist_ok=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = DemoWindow(_default_npz())
    win.resize(1280, 760)
    win.show()
    app.processEvents()

    print(f"{'state':22s} {'pts':>6s} {'ms':>8s}  cache")
    for label, overrides in STATES:
        win.reset_params()                       # each state composes from defaults, not the last
        t = time.perf_counter()
        for dev, params in overrides.items():
            win.set_params(dev, params)          # syncs widgets, so the screenshot is honest
        ms = (time.perf_counter() - t) * 1000
        for _ in range(2):
            app.processEvents()
        win.grab().save(str(outdir / f"{label}.png"))
        pts = win.readout.text().splitlines()[0].split(":")[-1].strip()
        cached = "hit" if "filters only" in win.readout.text() else "MISS"
        print(f"{label:22s} {pts:>6s} {ms:8.2f}  {cached}")
    print(f"\nwrote {len(STATES)} renders -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
