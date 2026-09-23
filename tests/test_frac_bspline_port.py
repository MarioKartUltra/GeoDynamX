# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Verbatim-copy guard for dynamix.core.frac_bspline against its Creep origin.

The four evaluator bodies are copied byte-identical from
``Creep/wavelet/wtmm/wavelets.py`` (Parts A/A2, Unser-Blu fractional B-splines);
the ONLY permitted DynamiX difference is the wrapper block above them (lazy scipy shims for
``factorial``/``gamma_func`` -- the recorded import rewrite, the ``test_mzlib_port``
license). This file is that license's enforcement: function-body drift in EITHER repo fails
here, making divergence a decision instead of a discovery. Skipped when the Creep checkout
is absent, so DynamiX still tests standalone.

Functional gates below reproduce the Creep validation (``wtmm_cpu.ipynb`` cell 70): the
fractional forms REDUCE to the integer B-splines at integer orders, exactly.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from dynamix.core import frac_bspline as fb

_REPO = pathlib.Path(__file__).resolve().parent.parent
_CREEP = pathlib.Path("~/projects/Creep/wavelet/wtmm/wavelets.py").expanduser()

_COPIED = ("_bspline_centered", "_bspline_derivative",
           "_frac_bspline_centered", "_frac_bspline_derivative")


def _def_block(text: str, name: str) -> str:
    """The full ``def <name>(...)`` block, trailing blank lines trimmed -- located by the
    def line, so the guard survives unrelated edits elsewhere in either file."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if ln.startswith(f"def {name}(")), None)
    assert start is not None, f"no def {name} found"
    end = start + 1
    while end < len(lines) and (lines[end].startswith((" ", "\t")) or lines[end].strip() == ""):
        end += 1
    while lines[end - 1].strip() == "":
        end -= 1
    return "".join(lines[start:end])


@pytest.mark.skipif(not _CREEP.is_file(), reason=f"Creep checkout not present at {_CREEP}")
@pytest.mark.parametrize("name", _COPIED)
def test_copied_bodies_are_byte_identical_to_creep(name):
    ours = _def_block((_REPO / "src/dynamix/core/frac_bspline.py").read_text(), name)
    theirs = _def_block(_CREEP.read_text(), name)
    assert ours == theirs, (
        f"{name} has drifted from its Creep original (wtmm/wavelets.py). If the change is "
        f"deliberate, it must become a recorded edit here -- never a silent divergence."
    )


# ---------------------------------------------------------------- functional gates (cell 70)

def test_fractional_reduces_to_integer_bsplines_exactly():
    """Cell 70's own verification, reproduced: alpha=m integer orders are the classical
    B-splines to 0.00e+00 -- basis functions AND derivative wavelets."""
    u = np.linspace(-4, 4, 1001)
    for m in (2, 3, 4):
        np.testing.assert_array_equal(fb._frac_bspline_centered(u, float(m)),
                                      fb._bspline_centered(u, m))
    for m, n in ((3, 1), (4, 2), (5, 3)):
        np.testing.assert_array_equal(fb._frac_bspline_derivative(u, float(m), float(n)),
                                      fb._bspline_derivative(u, m, n))


def test_fractional_orders_interpolate_the_ladder():
    """A genuinely fractional order sits between its integer neighbours: the basis peak is
    monotone in alpha across the ladder. Mass note, measured: the copied truncation
    (``k_max = floor(alpha+1)+1``) is EXACT at integer orders (integral 1.000000) but leaves
    real mass out at fractional ones (0.915 at alpha=2.5, 1.015 at 3.5 -- the causal
    fractional spline's support is genuinely infinite). That is Creep's own validated
    behavior, copied; it is why the dyadic-cascade oracle chain uses the ANALYTIC
    |w||sinc(w/4)|^(alpha+1) form at fractional orders and Part A2 exactly at the integer
    anchors (where the two coincide)."""
    u = np.linspace(-6, 6, 4001)
    du = u[1] - u[0]
    peaks = []
    for a in (2.0, 2.5, 3.0, 3.5, 4.0):
        y = fb._frac_bspline_centered(u, a)
        assert abs(np.trapezoid(y, dx=du) - 1.0) < 0.1, a     # truncation-bounded, not exact
        peaks.append(y.max())
    assert all(p1 > p2 for p1, p2 in zip(peaks, peaks[1:])), peaks
    for m in (2.0, 3.0, 4.0):                                  # integer orders: mass exact
        y = fb._frac_bspline_centered(u, m)
        assert abs(np.trapezoid(y, dx=du) - 1.0) < 1e-6, m
