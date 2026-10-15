# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Chainers are scored on synthetic isotropic cusps -|r - r0|^h of known h: the longest chain
through the cusp gives h as the slope of log2 modulus against log2 scale. The transform runs with
no pseudo-fractional integration (``fracint_alpha = 0``): the default lift of order 1 multiplies
each modulus by a, so the slope would read h + 1, the integrated frame the spectrum fits undo
with their eta. The h and its error per chainer and per h are recorded in
``golden/chaining/benchmark.json`` on the recording machine and must reproduce there (backends
agree to float32 tolerance, so the numbers are per machine).

    PYTHONNOUSERSITE=1 python tests/test_chaining_benchmark.py --record
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

from dynamix.core.chaining import CHAINERS
from dynamix.core.wtmm_backend import run_wtmm2d

GOLDEN = Path(__file__).parent / "golden" / "chaining" / "benchmark.json"
H_VALUES = (0.3, 0.5, 0.7)
N = 128
WTMM = {"n_oct": 3, "n_voice": 4, "thresh": 1e-3, "fracint_alpha": 0.0}


def _machine() -> str:
    return f"{sys.platform}-{platform.machine()}"


def _cusp(h: float):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    y, x = np.mgrid[0:N, 0:N].astype(np.float64)
    r = np.hypot(x - (N / 2 - 0.5), y - (N / 2 - 0.5))
    values = -(r ** h)
    return RasterField(name=f"cusp{h}", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(N)), y_axis=np.arange(float(N)))


def _h_from(chains: list) -> float:
    """h from the longest chain (the cusp's own): the slope of log2 modulus against log2 scale."""
    longest = max(chains, key=lambda c: len(c["mod"]))
    slope, _ = np.polyfit(longest["log2_scales"], longest["log2_mod"], 1)
    return float(slope)


def _run() -> dict:
    out = {}
    for name in sorted(CHAINERS):
        out[name] = {}
        for h in H_VALUES:
            res = run_wtmm2d(_cusp(h), {**WTMM, "chainer": name})
            est = _h_from(res["chains"])
            out[name][str(h)] = {"h": est, "error": abs(est - h), "chains": len(res["chains"])}
    return out


def test_every_chainer_finds_the_cusp_and_the_recorded_scores_reproduce():
    golden = json.loads(GOLDEN.read_text())
    now = _run()
    for name, by_h in now.items():
        for h, score in by_h.items():
            assert score["chains"] > 0 and score["error"] < 0.05, (name, h, score)
    if golden["machine"] != _machine():
        pytest.skip(f"recorded on {golden['machine']}")
    assert sorted(now) == sorted(golden["scores"])
    for name, by_h in golden["scores"].items():
        for h, want in by_h.items():
            got = now[name][h]
            assert got["chains"] == want["chains"], (name, h)
            assert abs(got["h"] - want["h"]) < 1e-6, (name, h, got, want)


if __name__ == "__main__" and "--record" in sys.argv:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps({"machine": _machine(), "scores": _run()}, indent=1,
                                 sort_keys=True) + "\n")
    print(f"recorded {GOLDEN}")
    for name, by_h in json.loads(GOLDEN.read_text())["scores"].items():
        print(name, {h: round(s["error"], 3) for h, s in by_h.items()})
