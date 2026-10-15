# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""dwt2d -- the periodized orthogonal dyadic pyramid as a Decomposition tool.

The field is mirror-extended once in each axis (``dynamix.core.dwt``), decomposed with an
orthogonal bank in periodization mode, and every band is reconstructed alone onto the native
grid as a per-level row: the approximation at level ``j`` (the low-pass at scale ``2**j``
px) and the horizontal, vertical and diagonal details at that level. The bands of every
level plus the coarsest approximation sum back to the field exactly. With cycle spinning on,
each row is the mean over ``n_spins`` random shifts: translation invariance at the price
of ``n_spins`` transforms. Non-finite pixels are filled with the finite mean for the
transform and restored on every row.

The foundation the wavelet-leader tools stand on; nothing here resamples (scales are
pixels, ``2**j``).
"""
from __future__ import annotations

import numpy as np

from dynamix.core.dwt import BANKS
from dynamix.model.contracts import COMPONENTS, FIELD
from dynamix.model.device import Output
from dynamix.model.param import Param, ParamKind

MAX_LEVELS = 8


class DWT2D:
    """The orthogonal dyadic pyramid, one row per band and level."""

    name = "dwt2d"
    capabilities = frozenset({"OutputProducer", "RegionAware", "Viewable"})
    label = "DWT"
    families = ("Decomposition",)
    consumes = ((FIELD,),)
    produces = (COMPONENTS,)
    params = (
        Param("wavelet", ParamKind.CHOICE, default="sym7", choices=BANKS, label="Wavelet"),
        Param("level", ParamKind.INT, default=0, min=0, max=10, soft_min=0, soft_max=8,
              label="Levels (0 = full depth)"),
        Param("cycle_spin", ParamKind.BOOL, default=False, label="Cycle spinning"),
        Param("n_spins", ParamKind.INT, default=8, min=1, max=64, soft_min=1, soft_max=16,
              label="Spins", active_when=("cycle_spin", (True,))),
        # "coarsest" names no row: the eager raster_out shows and every row can be hidden
        # (the shell's Show mechanism sets a hidden row back to the param's default).
        Param("show", ParamKind.CHOICE, default="coarsest",
              choices=("coarsest", "approximation", "horizontal", "vertical", "diagonal"),
              label="Show", view=True),
    )
    outputs = tuple(Output(name, "raster", lazy=True, label=name, per_level=True)
                    for name in ("approximation", "horizontal", "vertical", "diagonal"))

    def roi_margin(self, params: dict) -> int:
        """The coarsest atom a region's pixel sees: ``L 2**min(J, 4)`` px, at most 256."""
        from dynamix.core.dwt import filter_length

        J = int(params.get("level", 0)) or 3
        return min(filter_length(params.get("wavelet", "sym7")) * 2 ** min(J, 4), 256)

    def _levels(self, shape, params: dict) -> int:
        from dynamix.core.dwt import full_depth

        J = int(params.get("level", 0))
        return J if J > 0 else min(full_depth((2 * shape[0], 2 * shape[1])), MAX_LEVELS)

    @staticmethod
    def _values(field, device: str):
        from dynamix.devices.decompose import _field_values

        vals = _field_values(field, device)
        if vals.ndim != 2:
            raise ValueError(f"{device} takes a single-band field -- pick a band row of a stack")
        return vals

    @staticmethod
    def _finite(vals):
        finite = np.isfinite(vals)
        if finite.all():
            return vals, finite
        return np.where(finite, vals, vals[finite].mean() if finite.any() else 0.0), finite

    def _band(self, vals, name: str, J: int, j: int, band: str, spins: int) -> np.ndarray:
        from dynamix.core import dwt

        ny, nx = vals.shape
        shape = dwt.working_shape(vals.shape, J)
        x = dwt.mirror(vals, shape)
        rng = np.random.default_rng(0)
        out = np.zeros(shape)
        for _ in range(max(1, spins)):
            z, shift = dwt.cycle_shift(x, rng, J) if spins > 1 else (x, (0, 0))
            c = dwt.only_band(dwt.pyramid(z, name, J), j, band)
            out += dwt.cycle_unshift(dwt.inverse(c, name), shift)
        return dwt.quarter(out / max(1, spins), ny, nx)

    def wavelet_profile(self, params: dict, scale_px: float, level: int):
        """The bank's wavelet at the level whose scale is 2^level (the wavelet scale bar)."""
        from dynamix.core.wavelet_profiles import dwt_profile

        return dwt_profile(params["wavelet"], int(round(np.log2(float(scale_px)))))

    def compute(self, field, params: dict, *, progress=None) -> dict:
        from dynamix.devices.decompose import _result
        from dynamix.devices.ssa2d import _store

        vals = self._values(field, self.name)
        filled, finite = self._finite(vals)
        J = self._levels(vals.shape, params)
        if progress is not None:
            progress("dwt2d", 0.0)
        approx = self._band(filled, params["wavelet"], J, J, "approximation", 1)
        approx = np.where(finite, approx, np.nan)
        if progress is not None:
            progress("dwt2d", 1.0)
        res = _result(field, _store(approx), {"scales": 2.0 ** np.arange(1, J + 1),
                                              "_dwt_levels": J}, params)
        res["_shape"] = vals.shape
        return res

    def compute_output(self, name: str, values, result: dict, params: dict, *, fetch,
                       progress=None, cancel=None, level=None) -> dict:
        if name not in ("approximation", "horizontal", "vertical", "diagonal"):
            raise ValueError(f"{self.name}: no lazy output named {name!r}")
        vals = np.asarray(values, dtype=np.float64)
        filled, finite = self._finite(vals)
        J = int(result["_dwt_levels"])
        idx = int(np.clip(0 if level is None else int(level), 0, J - 1))
        spins = int(params.get("n_spins", 8)) if params.get("cycle_spin") else 1
        if progress is not None:
            progress(f"{name} level {idx + 1}/{J}", 0.0)
        raster = self._band(filled, params["wavelet"], J, idx + 1, name, spins)
        if progress is not None:
            progress(f"{name} level {idx + 1}/{J}", 1.0)
        return {"raster": np.where(finite, raster, np.nan).astype(np.float32),
                "diag": {"level": idx, "scale_px": 2.0 ** (idx + 1)}}

    def view(self, result: dict, params: dict) -> dict:
        return result

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k
        from dynamix.model.device import keyed_params

        return _k(self.name, source_id, keyed_params(self, params))
