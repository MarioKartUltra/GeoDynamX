# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""mz_edges — the Mallat-Zhong dyadic transform as a peer Transform device.

Emits the mz_edges product bundle (per-scale wrap-merged maxima chains, dyadic scales in px,
coarse policy) in the app's extrema/line_id schema. Cross-scale V-chains are Mallat-Hwang
SVII machinery (reserved script 11) -- "chains" is stamped empty deliberately.

The edges are the analysis; everything else the tool shows is a LAZY output computed from the
cached analysis on request (``compute_output``, through ``engine.resolve.resolve_output``): the
full-resolution coarse channel S_J, its 2^J thumbnail, the reconstruction from the multiscale
edges with S_J pinned or with S = 0, and the residual (field minus reconstruction). The knobs
that only pick or shape those outputs (Show, Iterations, Mode, Coarse) are view-only, so the
analysis always runs with the full coarse channel and flipping one of them never re-runs it.
"""
from __future__ import annotations

import numpy as np

from dynamix.model.device import Output, keyed_params
from dynamix.model.param import Param, ParamKind


class MZEdges:
    name = "mz_edges"
    params = (
        Param("n_levels", ParamKind.INT, default=4, min=1, max=10,
              soft_min=2, soft_max=6, label="Levels J"),
        # Which coarse channel the reconstruction pins: S_J at full resolution, or the one
        # decoded from its 2^J thumbnail (which needs the grid divisible by 2^J).
        Param("coarse", ParamKind.CHOICE, default="full",
              choices=("full", "thumbnail"), label="Coarse", view=True),
        Param("dither", ParamKind.BOOL, default=False, label="Dither"),
        # The M-Z tool's own wavelet menu -- the paper's dyadic spline, or the Unser-Blu
        # fractional order (SMOOTHNESS knob; one vanishing moment always -- maxima-are-edges
        # is the method). alpha binds only for frac_bspline (the inert-knob precedent); 3.0 IS
        # the paper's wavelet.
        Param("wavelet", ParamKind.CHOICE, default="mz_spline",
              choices=("mz_spline", "frac_bspline"), label="Wavelet"),
        # The wtmm2d subpixel/value refinement, on the dyadic maxima. POCS input (mz_maxima)
        # stays integer-supported regardless -- reconstruction is pixel-exact by the papers'
        # own numerics.
        Param("interpolate", ParamKind.BOOL, default=False, label="Interpolate"),
        Param("alpha", ParamKind.FLOAT, default=3.0, min=0.1, max=8.0,
              soft_min=2.0, soft_max=5.0, units="", label="α"),
        # The raster output drawn in place of the field ("edges": the field itself, with the
        # maxima over it); each other choice names a lazy output below.
        Param("show", ParamKind.CHOICE, default="edges",
              choices=("edges", "coarse", "thumbnail", "recon", "recon_edges_only",
                       "residual"), label="Show", view=True),
        # POCS iterations of the reconstruction (fixed count, no early stop).
        Param("iterations", ParamKind.INT, default=10, min=1, max=500,
              soft_min=1, soft_max=50, label="Iterations", view=True),
        # How P_Gamma interpolates between the maxima: the vectorized separable operator, the
        # same one row by row through mzlib (the reference), or the maxima values alone.
        # The choices are ``core.mz_edges.RECON_MODES``, spelled here so the device module
        # stays free of the core import.
        Param("mode", ParamKind.CHOICE, default="separable",
              choices=("separable", "separable_mzlib", "set_points"), label="Mode",
              view=True),
    )

    outputs = (
        Output("edges", "vector", label="edges"),
        Output("coarse", "raster", lazy=True, label="coarse"),
        Output("thumbnail", "raster", lazy=True, label="thumbnail", grid="stride"),
        Output("recon", "raster", lazy=True, params=("iterations", "mode", "coarse"),
               label="recon (edges + coarse)"),
        Output("recon_edges_only", "raster", lazy=True, params=("iterations", "mode"),
               label="recon (edges only)"),
        Output("residual", "raster", lazy=True, params=("iterations", "mode", "coarse"),
               label="residual"),
    )

    def compute(self, field, params: dict, *, progress=None) -> dict:
        from dynamix.core import mz_edges

        # Always the full coarse channel: the thumbnail is derived from it on request, and the
        # Coarse knob only chooses what a reconstruction pins.
        bundle = mz_edges.analyze(
            field.values,
            params["n_levels"],
            coarse="full",
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

    def view(self, result: dict, params: dict) -> dict:
        """The view-only knobs pick or shape a lazy output, which the engine computes and caches
        under its own key; the analysis arrays are shown unchanged. The returned copy's
        ``params`` carry the current view-only knobs (the cached bundle keeps the ones it was
        first computed with), so a label read from ``params["show"]`` names what is shown."""
        current = {p.name: params[p.name] for p in self.params if p.view}
        return {**result, "params": {**result.get("params", {}), **current}}

    def compute_output(self, name: str, values, result: dict, params: dict, *, fetch,
                       progress=None, cancel=None) -> dict:
        """The lazy output ``name`` from the cached analysis ``result`` of ``values``:
        ``{"raster": float32 ndarray, "diag": dict}``, plus ``display_stride``/``full_dims``
        for the thumbnail, whose samples sit every 2^J pixels of the (ny, nx) field. The
        residual is ``values`` minus the recon ``fetch`` returns, so it reuses that one's work."""
        from dynamix.core import mz_edges

        values = np.asarray(values)
        if name == "coarse":
            return {"raster": mz_edges.coarse_image(values, result).astype(np.float32),
                    "diag": {}}
        if name == "thumbnail":
            return {"raster": mz_edges.coarse_thumbnail(values, result), "diag": {},
                    "display_stride": 2 ** len(result["mz_maxima"]),
                    "full_dims": tuple(values.shape)}
        if name in ("recon", "recon_edges_only"):
            coarse = params["coarse"] if name == "recon" else "none"
            img, diag = mz_edges.reconstruct(values, result, n_iter=params["iterations"],
                                             mode=params["mode"], coarse=coarse,
                                             progress=progress, cancel=cancel)
            return {"raster": img.astype(np.float32), "diag": diag}
        if name == "residual":
            recon = fetch("recon")
            return {"raster": (values - recon["raster"]).astype(np.float32),
                    "diag": recon["diag"]}
        raise ValueError(f"{self.name}: no lazy output named {name!r}")

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

        return _k(self.name, source_id, keyed_params(self, params))
