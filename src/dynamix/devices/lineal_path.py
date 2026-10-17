# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The lineal-path function of a two-phase map.

``L(r)`` is the probability that a segment of length ``r`` pixels (``r + 1`` pixels in a row),
dropped at a random position along a row or a column, lies entirely in phase 1 (``values >
threshold``); ``L(0)`` is the phase fraction and ``L`` never increases with ``r`` (Torquato,
Random Heterogeneous Materials, 2002, §2.4). Counted exactly from the run lengths of phase 1
along the rows and the columns, without periodic wrap: a run of ``n`` pixels holds ``n - r``
segments of length ``r``, over ``rows * (width - r)`` positions.
"""
from __future__ import annotations

import numpy as np

from dynamix.model.contracts import CURVE, FIELD
from dynamix.model.param import Param, ParamKind

_KERNELS = None


def _kernels():
    """The run-length histogram, compiled on first use (``cache=True``)."""
    global _KERNELS
    if _KERNELS is None:
        import numba

        @numba.njit(cache=True)
        def run_histogram(phase, hist):
            """``hist[n] += 1`` for every maximal run of ``n`` true pixels along the rows of
            ``phase``; ``hist`` has ``width + 1`` bins."""
            ny, nx = phase.shape
            for i in range(ny):
                n = 0
                for j in range(nx):
                    if phase[i, j]:
                        n += 1
                    else:
                        if n > 0:
                            hist[n] += 1
                        n = 0
                if n > 0:
                    hist[n] += 1

        _KERNELS = run_histogram
    return _KERNELS


def lineal_path(phase: np.ndarray, r_max: int, direction: str = "both") -> np.ndarray:
    """``L(r)`` for ``r = 0 .. r_max`` of the boolean map ``phase``, along the rows
    (``"horizontal"``), the columns (``"vertical"``) or the mean of both."""
    phase = np.ascontiguousarray(phase, dtype=np.bool_)
    kernel = _kernels()
    curves = []
    for name, arr in (("horizontal", phase), ("vertical", np.ascontiguousarray(phase.T))):
        if direction not in ("both", name):
            continue
        ny, nx = arr.shape
        hist = np.zeros(nx + 1, dtype=np.int64)
        kernel(arr, hist)
        n = np.arange(nx + 1, dtype=np.float64)
        r = np.arange(min(r_max, nx - 1) + 1, dtype=np.float64)
        segments = np.array([(hist * np.maximum(n - rr, 0.0)).sum() for rr in r])
        positions = ny * (nx - r)
        curve = np.full(r_max + 1, np.nan)
        curve[:len(r)] = segments / positions
        curves.append(curve)
    return np.nanmean(np.vstack(curves), axis=0) if len(curves) > 1 else curves[0]


class LinealPath:
    """``L(r)`` of the field thresholded into two phases, along the rows, the columns or both."""

    name = "lineal_path"
    label = "Lineal path"
    families = ("Microstructure",)
    role = "primary"                    # a drop on a layer forks a sibling over the same field
    consumes = ((FIELD,),)
    produces = frozenset({CURVE})
    capabilities = frozenset({"SelfReporting", "RegionAware"})
    params = (
        Param("threshold", ParamKind.FLOAT, default=0.0, min=-1e30, max=1e30,
              label="Phase 1 above"),
        Param("direction", ParamKind.CHOICE, default="both",
              choices=("both", "horizontal", "vertical"), label="Direction"),
        Param("r_max", ParamKind.INT, default=64, min=1, max=100000, soft_min=8, soft_max=512,
              label="Max r (px)"),
    )

    def roi_margin(self, params: dict) -> int:
        """A count of runs needs no halo: a segment either lies in the window or it does not."""
        return 0

    def compute(self, field, params: dict, *, progress=None) -> dict:
        values = np.asarray(field.values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("lineal_path: a single-band field")
        phase = values > float(params["threshold"])
        r_max = int(params["r_max"])
        curve = lineal_path(phase, r_max, params["direction"])
        return {"r": np.arange(r_max + 1, dtype=np.float64), "L": curve,
                "direction": params["direction"], "phase_fraction": float(phase.mean()),
                "_shape": tuple(values.shape)}

    def cache_key(self, params: dict):
        return (self.name, float(params["threshold"]), params["direction"], int(params["r_max"]))

    def reading(self, result: dict, params: dict) -> str:
        L = np.asarray(result.get("L", ()))
        if L.size == 0:
            return "no curve — run the transform"
        half = np.flatnonzero(L <= 0.5 * L[0])
        r_half = f"L halves at r = {int(half[0])} px" if half.size else "L stays above half of L(0)"
        return f"phase fraction {L[0]:.3f} · {r_half}"
