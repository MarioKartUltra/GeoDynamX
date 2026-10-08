# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The pre-fill of the regions and the correction of the known pixels, both through the
universal kernel (Turiel and del Pozo 2002, ``microcanonical.reconstruct_from_msc``).

The pre-fill puts the value prior (else the clear mean) into every hole and rebuilds the field
from its gradient with the hole boundary's gradients suppressed, so the first pyramid sees no
step (Pottier et al. 2008 Fig. 13). The correction rebuilds the smooth field consistent with
the known-pixel difference ``x - y`` away from the boundary ring and the loop subtracts it:
it reduces the known-pixel error by a finite factor without a step at the boundary (the
paper's Fig. 15); it is a smoothing step, not a projection, so every route ends by putting
``y`` back on the known pixels.
"""
from __future__ import annotations

import numpy as np

__all__ = ["boundary_ring", "correct", "prefill"]


def boundary_ring(mask) -> np.ndarray:
    """One pixel each side of every edge of ``mask``: its dilation minus its erosion."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.zeros(m.shape, dtype=bool)
    p = np.pad(m, 1, mode="constant", constant_values=False)
    dil = (p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])
    q = np.pad(m, 1, mode="constant", constant_values=True)
    ero = (q[1:-1, 1:-1] & q[:-2, 1:-1] & q[2:, 1:-1] & q[1:-1, :-2] & q[1:-1, 2:])
    return dil & ~ero


def _rebuild(signal, keep, pad):
    from dynamix.core.microcanonical import reconstruct_from_msc

    out = reconstruct_from_msc(np.asarray(signal, dtype=np.float64), np.asarray(keep, dtype=bool),
                               pad=pad)[0]
    return np.asarray(out, dtype=np.float64)


def prefill(y, holes, prior=None, *, pad=None) -> np.ndarray:
    """``y`` with its holes (``holes`` and every non-finite pixel) filled from ``prior`` where
    finite, else the clear mean, and the fill boundary smoothed; clear pixels off the boundary
    ring keep their values exactly. The rebuild from the gradient keeps the prior's shape and
    sets its level by the clear-pixel mean match, so a prior's constant offset does not
    survive: the caller fits the prior to the data first."""
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y)
    holes = np.asarray(holes, dtype=bool) | ~finite
    clear = ~holes
    mean = float(y[clear].mean()) if clear.any() else 0.0
    fill = np.full(y.shape, mean)
    if prior is not None:
        p = np.asarray(prior, dtype=np.float64)
        fill = np.where(np.isfinite(p), p, mean)
    x = np.where(holes, fill, y)
    ring = boundary_ring(holes)
    if not ring.any():
        return x
    smooth = _rebuild(x, ~ring, pad)
    smooth += x[clear].mean() - smooth[clear].mean()
    return np.where(holes | ring, smooth, x)


def correct(x, y, known, ring, *, pad=0) -> np.ndarray:
    """The smooth field consistent with ``x - y`` on the known pixels away from ``ring``,
    mean-matched on the known pixels: the loop subtracts it. ``pad=0`` on a working image
    that is already periodic (mirror-extended)."""
    known = np.asarray(known, dtype=bool)
    d = np.where(known, np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64), 0.0)
    smooth = _rebuild(d, known & ~np.asarray(ring, dtype=bool), pad)
    smooth += d[known].mean() - smooth[known].mean()
    return smooth
