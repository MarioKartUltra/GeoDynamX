# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Connected-component size filtering ("sieve", GDAL/ENVI's term) -- pure numpy/scipy.

Suppress islands of a boolean mask (or of each class of a class map) whose pixel count falls
outside ``[min_px, max_px]`` -- a fault-delineation cleanup: dithered measure-route h maps
delineate BOEM faults well but speckle with single-pixel "snow"; a size floor keeps the
coherent ribbons, and an optional ceiling can drop a giant background blob.

``connectivity`` 8 (default) counts diagonal neighbours -- the right choice for ribbon-like
structures, which a 4-connected sieve chops into fragments at every diagonal step.

Costs one ``scipy.ndimage.label`` + ``np.bincount`` per call (~5-15 ms at 512², ~100-200 ms at
2048²) -- callers in interactive paths run it LAZILY (on gesture settle, never per drag tick).
"""
from __future__ import annotations

import numpy as np

__all__ = ["sieve_mask", "sieve_classes"]

_STRUCTURES = {
    4: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    8: np.ones((3, 3), dtype=bool),
}


def sieve_mask(mask: np.ndarray, min_px: int = 0, max_px: int = 0, *,
               connectivity: int = 8) -> np.ndarray:
    """The mask with components outside ``[min_px, max_px]`` removed.

    ``min_px`` 0 disables the floor, ``max_px`` 0 disables the ceiling (so (0, 0) is the
    identity). Returns a NEW boolean array; the input is never mutated.
    """
    if connectivity not in _STRUCTURES:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")
    mask = np.asarray(mask, dtype=bool)
    if (min_px <= 1 and max_px <= 0) or not mask.any():
        return mask.copy()
    from scipy import ndimage

    labels, n = ndimage.label(mask, structure=_STRUCTURES[connectivity])
    if n == 0:
        return mask.copy()
    sizes = np.bincount(labels.ravel())
    keep = np.ones(n + 1, dtype=bool)
    keep[0] = False
    if min_px > 1:
        keep &= sizes >= min_px
    if max_px > 0:
        keep &= sizes <= max_px
    return keep[labels]


def sieve_classes(classes01: np.ndarray, n_classes: int, min_px: int = 0,
                  max_px: int = 0, *, connectivity: int = 8) -> np.ndarray:
    """Per-class sieve of a [0, 1] class field (``dynamix.core.stretch.classify``'s output):
    each class's islands are filtered INDEPENDENTLY; suppressed pixels become NaN (rendered
    transparent downstream). NaN input stays NaN."""
    v = np.asarray(classes01, dtype=np.float64)
    if min_px <= 1 and max_px <= 0:
        return v.copy()
    out = v.copy()
    for k in range(max(int(n_classes), 1)):
        level = k / max(n_classes - 1, 1)
        m = np.isfinite(v) & np.isclose(v, level)
        if not m.any():
            continue
        kept = sieve_mask(m, min_px, max_px, connectivity=connectivity)
        out[m & ~kept] = np.nan
    return out
