# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""NMS-detection parity against the REAL xsmurf C: ``xsmurf_wrapper.wtmm2d`` wraps
``Extract_Gradient_Maxima_2D`` (edge/extrema.c) -> ``Remove_Gradient_NonMaxima_Slice_2D``
(edge/extrema_core.c), what xsmurf's ``wtmm2d`` Tcl command runs under imStudy's ``useNMaxSup 1``.
Both stacks receive the IDENTICAL float32 gradient images, so this pins detection alone: the
asymmetric tie rule, the bilinear sampling and the nearest-pixel shortcut near an axis.

The C sets a one-pixel frame to zero and rejects a pixel whose interpolation point falls outside
``[0, n - 1)``; the run here crops its padding back off instead, so the comparison is on pixels
two or more from the frame. Machine-specific by nature (needs the compiled XSmurfWrapper, the
``xsmurf`` environment with this tree's ``src`` first on the path); skips cleanly when absent.
"""
from __future__ import annotations

import numpy as np
import pytest

xw = pytest.importorskip("xsmurf_wrapper")

from conftest import fbm2d                                            # noqa: E402
from dynamix.core.wtmm_backend import (_cwt2d_numpy, _nms_extrema_scale,  # noqa: E402
                                       compute_scales2d)

N = 96


def _interior(xs, ys, frame=2):
    keep = (xs >= frame) & (xs < N - frame) & (ys >= frame) & (ys < N - frame)
    return set(zip(np.asarray(xs)[keep].tolist(), np.asarray(ys)[keep].tolist()))


@pytest.mark.parametrize("seed", [7, 11])
def test_the_asymmetric_rule_finds_xsmurfs_wtmm2d_maxima(seed):
    field = fbm2d(N, 0.6, seed=seed)
    scales = compute_scales2d(2, 2, 1.0)
    raw = _cwt2d_numpy(field, scales, derivs="first", verbose=False)
    for si, scale in enumerate(scales):
        dx = np.asarray(raw["dx"][si], dtype=np.float32)
        dy = np.asarray(raw["dy"][si], dtype=np.float32)
        theirs = xw.wtmm2d(xw.XImage.from_numpy(dx), xw.XImage.from_numpy(dy), float(scale),
                           thresh=0.0)[0].get_extrema_arrays()
        mod = np.hypot(dx.astype(np.float64), dy.astype(np.float64)).astype(np.float32)
        arg = np.arctan2(dy.astype(np.float64), dx.astype(np.float64))
        ours = _nms_extrema_scale(mod, arg, thresh=0.0, tie_rule="asymmetric")
        a = _interior(ours["x"], ours["y"])
        b = _interior(np.asarray(theirs["x"], int), np.asarray(theirs["y"], int))
        assert a == b, (f"scale {scale}: {len(a - b)} only ours, {len(b - a)} only theirs "
                        f"of {len(b)}")


@pytest.mark.parametrize("rule, matches", [("asymmetric", True), ("strict", False),
                                           ("keep_both", False)])
def test_on_an_exact_tie_xsmurf_keeps_the_pixel_ahead_of_the_gradient(rule, matches):
    """Smooth data never tie exactly, so the fBm above cannot tell the rules apart. Hand-made
    gradients can: dx is a ridge whose crest is two equal columns, dy is zero, so the gradient
    points along +x and the two crest pixels tie. The C keeps the one ahead (column 11)."""
    x = np.arange(32.0)
    profile = np.exp(-0.5 * ((x - 10.5) / 3.0) ** 2)
    profile[11] = profile[10]
    dx = np.tile(profile, (24, 1)).astype(np.float32)
    dy = np.zeros_like(dx)
    theirs = xw.wtmm2d(xw.XImage.from_numpy(dx), xw.XImage.from_numpy(dy), 4.0,
                       thresh=0.0)[0].get_extrema_arrays()
    b = set(zip(np.asarray(theirs["x"], int).tolist(), np.asarray(theirs["y"], int).tolist()))
    b = {(px, py) for px, py in b if 2 <= py < 22}
    assert {px for px, _ in b} == {11}
    ours = _nms_extrema_scale(np.abs(dx), np.zeros_like(dx), thresh=0.0, tie_rule=rule)
    a = {(px, py) for px, py in zip(ours["x"].tolist(), ours["y"].tolist()) if 2 <= py < 22}
    assert (a == b) is matches
