# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""mz_edges — the Mallat-Zhong dyadic transform as a peer Transform device.

Emits the mz_edges product bundle (per-scale wrap-merged maxima chains, dyadic scales in px,
coarse policy) in the app's extrema/line_id schema. Cross-scale V-chains are Mallat-Hwang
SVII machinery (reserved script 11) -- "chains" is stamped empty deliberately.
"""
from __future__ import annotations

from dynamix.model.param import Param, ParamKind


class MZEdges:
    name = "mz_edges"
    params = (
        Param("n_levels", ParamKind.INT, default=4, min=1, max=10,
              soft_min=2, soft_max=6, label="Levels J"),
        Param("coarse", ParamKind.CHOICE, default="full",
              choices=("full", "thumbnail"), label="Coarse"),
        Param("dither", ParamKind.BOOL, default=False, label="Dither"),
        # 2026-09-19 split (§6.4): the M-Z tool's own wavelet menu -- the paper's dyadic
        # spline, or the Unser-Blu fractional order (SMOOTHNESS knob; one vanishing moment
        # always -- maxima-are-edges is the method). alpha binds only for frac_bspline
        # (the inert-knob precedent); 3.0 IS the paper's wavelet.
        Param("wavelet", ParamKind.CHOICE, default="mz_spline",
              choices=("mz_spline", "frac_bspline"), label="Wavelet"),
        # 2026-09-21: the wtmm2d subpixel/value refinement, on the dyadic maxima. POCS
        # input (mz_maxima) stays integer-supported regardless -- reconstruction is
        # pixel-exact by the papers' own numerics (the recorded split).
        Param("interpolate", ParamKind.BOOL, default=False, label="Interpolate"),
        Param("alpha", ParamKind.FLOAT, default=3.0, min=0.1, max=8.0,
              soft_min=2.0, soft_max=5.0, units="", label="α"),
    )

    def compute(self, field, params: dict, *, progress=None) -> dict:
        from dynamix.core import mz_edges

        bundle = mz_edges.analyze(
            field.values,
            params["n_levels"],
            coarse=params["coarse"],
            dither=params["dither"],
            wavelet=params["wavelet"],
            alpha=params["alpha"],
            interpolate=params["interpolate"],
            progress=progress,
        )
        bundle["chains"] = []
        # pre-split this stamped wavelet="mz_spline" by hand; the knob now carries it
        bundle["params"] = dict(params)
        bundle["_frame"] = field.frame
        bundle["_shape"] = field.values.shape
        return bundle

    def roi_margin(self, params: dict) -> int:
        """The transform's own measured impulse reach at the coarsest
        level (``mz_edges.mz_impulse_reach``) + 2 px for the bilinear NMS probes and the
        subpixel refinement -- so the mirror seam mz folds onto the WINDOW edge never reaches
        the ROI. Exact for the compact dyadic spline; the fractional order's algebraic tail
        is cut at 1e-3 of its |W| mass (oracle-pinned)."""
        from dynamix.core.mz_edges import mz_impulse_reach

        return mz_impulse_reach(int(params["n_levels"]), str(params["wavelet"]),
                                float(params["alpha"])) + 2

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)
