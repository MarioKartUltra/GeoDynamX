# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Analysis of a field end to end: ``dwt2d`` then ``extrema2`` on the working field.

The working field is periodic in both border modes (the field itself for ``"periodic"``, its 2N
mirror for ``"mirror"``), so detection always runs on the full working-field arrays with the
periodic rule, and only results are cropped to the field's own grid. Detection on cropped arrays
would differ in a border band two pixels wide.
"""
from __future__ import annotations

from dynamix.core.mz_lastwave.detect import extrema2
from dynamix.core.mz_lastwave.transform import dwt2d


def analyze(values, J, *, border="mirror", colocate_l1=False):
    """``(Transform, Extrep)`` of a 2-D field: the J-level transform and the extrema of its
    working field (``Transform.full_shape``), which is what :func:`e2recons` reconstructs from."""
    t = dwt2d(values, J, border=border)
    ex = extrema2(t.Wx_full, t.Wy_full, J, colocate_l1=colocate_l1, border="periodic")
    return t, ex


def primary_extrema(t, ex, l):
    """``(mask, mag, arg)`` of level ``l`` on the field's own grid: the working-field extrema of
    ``ex`` cropped to ``t.shape``, ``mag`` normalised as ``extrema2`` stores it."""
    return t._primary(ex.mask[l]), t._primary(ex.mag[l]), t._primary(ex.arg[l])
