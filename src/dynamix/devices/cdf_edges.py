# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""cdf_edges — complex cross-diffusion filtering as the fourth peer analyzer.

(Named for the FAMILY, the scripts' own convention — 25_cdf_edges.py: the linear LCDF
and nonlinear NCDF variants both live behind the Variant knob.)

One LCDF/NCDF evolution (:mod:`dynamix.core.cdf` — lifted verbatim from the
research/reconstruction scripts, port-guarded), snapshotted at script 26's dyadic
schedule, an M-Z-convention edge pyramid out. The Ricker-CWT identity (script 28,
machine-pinned) is what makes this an ANALYZER and not just a smoother: the evolution's
Im channel IS a Mexican-hat scale space, so each snapshot yields honest per-scale edges
— ``"mz"`` mode takes grad-Re NMS maxima (the M-Z convention, directly comparable to
``mz_edges``), ``"marr"`` mode the Im zero crossings (the Marr-Hildreth convention).
The NONLINEAR variant (Perona-Malik-style: the Im channel — the edge detector itself —
modulates the diffusivity through ``k``) is the cross-diffusion coupling in its simplest laboratory form.

**Edge-mode provenance.** The CDF literature's OWN edge
concept is the Im/v channel and nothing more -- Barbeiro 2023 introduces w = (u, v) with
v in "the role of edge detector", and Gilboa's small-theta identity is why (Im ~ theta t
* smoothed Laplacian: the LoG/Marr lineage; the nonlinear variants steer their
diffusivity by that same channel). So ``"marr"`` is the authors' reading -- and even its
zero-crossing extraction is our operationalization of "edge detector". ``"mz"``
(grad-Re NMS) is OURS entirely: the symbol gate proves the
Re channel is a Gaussian-smoothing semigroup, and along-gradient modulus maxima of a
Gaussian-smoothed image at dyadic scales is the Mallat-Zhong DEFINITION -- so nothing is
attributed to the CDF authors; their evolution is used as an alternative GENERATOR of
the same scale space mz_edges builds with splines, which is what makes the two devices
comparable at matched scales. Neither mode has anything to do with the cone of
influence; boundary validity is the border-distrust machinery's business, as everywhere.

Output speaks the per-scale extrema schema (x/y/mod/arg/line_id — the mz_edges
precedent), so the display path, scale_select and the hline filters work unchanged;
``scales`` are the effective Gaussian sigmas in px (scales-in-pixels law), and the final
``filtered`` (Re) / ``edge_channel`` (Im) rasters ride the result for any consumer.
``chains`` is stamped empty deliberately — cross-scale linking is reserved, as
everywhere. Standalone transform: analyses the LAYER'S FIELD, refuses upstream results
with the shared pointed message.
"""
from __future__ import annotations

from dynamix.devices.holder_map import field_values_2d
from dynamix.model.device import Output
from dynamix.model.param import Param, ParamKind


def _line_ids(mask, ys, xs):
    import numpy as np

    if not ys.size:
        return np.zeros(0, dtype=np.int64)
    from scipy.ndimage import label

    labels, _n = label(mask, structure=np.ones((3, 3), dtype=bool))
    ids = labels[ys, xs].astype(np.int64) - 1
    _u, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return np.where(counts[inverse] == 1, -1, ids).astype(np.int64)


class CDFEdges:
    wants_cancel = True                  # the stop button reaches the snapshot loop
    name = "cdf_edges"
    params = (
        Param("n_levels", ParamKind.INT, default=4, min=1, max=8,
              soft_min=2, soft_max=5, label="Levels J"),
        Param("variant", ParamKind.CHOICE, default="linear",
              choices=("linear", "nonlinear"), label="Variant"),
        # The Perona-Malik edge threshold, in IMAGE units (the scripts' own semantics);
        # binds only under the nonlinear variant — the inert-knob precedent.
        Param("k_edge", ParamKind.FLOAT, default=1.0, min=0.001, max=1000.0,
              soft_min=0.2, soft_max=5.0, units="", label="k"),
        Param("theta", ParamKind.FLOAT, default=0.1047, min=0.001, max=0.5,
              soft_min=0.05, soft_max=0.2, units="rad", label="θ"),
        # Explicit-Euler stability for the 5-point Laplacian caps dt; 0.2 is the
        # scripts' own value.
        Param("dt", ParamKind.FLOAT, default=0.2, min=0.001, max=0.24,
              soft_min=0.1, soft_max=0.22, units="", label="dt"),
        Param("edge_mode", ParamKind.CHOICE, default="mz",
              choices=("mz", "marr"), label="Edges"),
        # xsmurf's follow takes four IMAGES -- detector="follow" feeds it this device's own
        # smoothed snapshots via FD derivative stacks (dynamix.core.xsmurf_follow.kapa_from_field)
        # and runs the exact ported detector; edge_mode/interpolate are inert under it
        # (follow's channels are native).
        Param("detector", ParamKind.CHOICE, default="nms",
              choices=("nms", "follow"), label="Detector"),
        # What the canvas SHOWS. "filtered" = the final Re (the denoised field
        # -- the cross-diffusion filter AS a filter), "edge_channel" = the final Im; either
        # rides the raster_out display contract (stretch/colormap/hillshade/export free-ride).
        # The extrema pyramid still computes and overlays either way (hide with H if unwanted).
        Param("show", ParamKind.CHOICE, default="edges",
              choices=("edges", "filtered", "edge_channel"), label="Show", view=True),
        # Subpixel refinement, both conventions -- "mz" mode gets the wtmm2d
        # parabola along the gradient (dynamix.core.subpixel); "marr" mode gets the
        # zero-crossing offset t = -Im/||grad Im|| along the gradient (first-order root of the
        # crossing), clamped to half a pixel. Positions land in x_sub/y_sub; integer support
        # untouched.
        Param("interpolate", ParamKind.BOOL, default=False, label="Interpolate"),
        Param("floor", ParamKind.FLOAT, default=0.05, min=0.0, max=0.5,
              soft_min=0.02, soft_max=0.2, units="", label="Floor"),
    )
    # The layer panel's output rows: the raster rows are a radio over ``show``, the edges row
    # hides the maxima drawing (``result["extrema"]`` stays for filters and tables).
    outputs = (Output("edges", "vector", label="edges"),
               Output("filtered", "raster", label="filtered"),
               Output("edge_channel", "raster", label="edge channel"))

    def compute(self, field, params: dict, *, progress=None, cancel=None) -> dict:
        import numpy as np

        from dynamix.core import cdf as lcdf

        vals = field_values_2d(field, self.name)
        theta, dt = float(params["theta"]), float(params["dt"])
        sigmas = [float(2.0 ** k) for k in range(int(params["n_levels"]))]
        iters = lcdf.sigma_iters(sigmas, theta=theta, dt=dt)
        if progress is not None:
            progress("cdf evolve", 0.0)
        if params["variant"] == "nonlinear":
            snaps, I, done = {}, vals, 0
            for s in sigmas:
                # Cooperative stop: the run_wtmm2d stage-boundary contract.
                if cancel is not None and cancel():
                    from dynamix.core.wtmm_backend import ComputeCancelled
                    raise ComputeCancelled()
                I = lcdf.ncdf(I, float(params["k_edge"]), iters[s] - done,
                              theta=theta, dt=dt)
                done = iters[s]
                snaps[iters[s]] = I
                if progress is not None:
                    progress("cdf evolve", done / max(iters.values()))
        else:
            # Chunked between snapshot counts -- the SAME op sequence as one call (the
            # port test pins pyramid equality), but with a progress tick per segment: the
            # single-call form was one 0.0 tick then 30+ s of silence at the 4096-px
            # window (measured; the mz silence disease). The guarded core is
            # untouched -- continuation is exact for the autonomous scheme.
            snaps, I, done = {}, vals, 0
            total = max(iters.values())
            for s_px in sigmas:
                if cancel is not None and cancel():
                    from dynamix.core.wtmm_backend import ComputeCancelled
                    raise ComputeCancelled()
                I, _ = lcdf.lcdf_evolve(I, iters[s_px] - done, theta=theta, dt=dt)
                done = iters[s_px]
                snaps[done] = I
                if progress is not None:
                    progress("cdf evolve", done / total)
        if progress is not None:
            progress("cdf edges", 0.8)
        extrema = []
        follow_det = params.get("detector") == "follow"
        runs_all, closed_all = [], []
        for s in sigmas:
            snap = snaps[iters[s]]
            if follow_det:
                # follow reads the Re channel (the Gaussian-smoothing semigroup the symbol
                # gate proves) whatever edge_mode says -- the knob binds only under nms.
                from dynamix.core.xsmurf_follow import follow_layer_from_snapshot
                layer, runs, closed = follow_layer_from_snapshot(snap.real,
                                                                float(params["floor"]))
                runs_all.append(runs)
                closed_all.append(closed)
                extrema.append(layer)
                continue
            if params["edge_mode"] == "marr":
                mask, coeff = lcdf.zero_crossings(snap.imag, float(params["floor"]))
                gy, gx = np.gradient(snap.imag)
            else:
                mask, coeff = lcdf.grad_nms(snap.real, float(params["floor"]))
                gy, gx = np.gradient(snap.real)
            ys, xs = np.where(mask)
            arg_pts = np.arctan2(gy, gx)[ys, xs]
            layer = {
                "x": xs.astype(np.int64), "y": ys.astype(np.int64),
                "mod": coeff[ys, xs],
                "arg": arg_pts,
                "line_id": _line_ids(mask, ys, xs),
            }
            if params["interpolate"] and xs.size:
                if params["edge_mode"] == "marr":
                    # First-order root of the Im crossing along the gradient direction:
                    # t = -Im(p) / ||grad Im(p)||, clamped to half a pixel.
                    a0 = snap.imag[ys, xs]
                    slope = coeff[ys, xs]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        t = np.where(slope > 0, -a0 / slope, 0.0)
                    t = np.clip(np.nan_to_num(t), -0.5, 0.5)
                    layer["x_sub"] = xs + t * np.cos(arg_pts)
                    layer["y_sub"] = ys + t * np.sin(arg_pts)
                else:
                    from dynamix.core.subpixel import refine_scale
                    x_sub, y_sub, mod_sub = refine_scale(coeff, arg_pts,
                                                         layer["x"], layer["y"])
                    layer["mod"] = mod_sub
                    layer["x_sub"] = x_sub
                    layer["y_sub"] = y_sub
            elif params["interpolate"]:
                layer["x_sub"] = xs.astype(np.float64)
                layer["y_sub"] = ys.astype(np.float64)
            extrema.append(layer)
        if progress is not None:
            progress("cdf edges", 1.0)
        out = {
            "extrema": extrema, "chains": [],
            "scales": np.asarray(sigmas, dtype=np.float64),
            "filtered": I.real.copy(), "edge_channel": I.imag.copy(),
            "params": dict(params),
            "_frame": getattr(field, "frame", None), "_shape": vals.shape,
        }
        if follow_det:
            out["_hline_runs"] = runs_all
            out["_hline_closed"] = closed_all
        return out

    def view(self, result: dict, params: dict) -> dict:
        """``show`` is view-only: every channel is computed and cached once, and the chosen one
        replaces the raw field on the canvas (the display contract) -- switching is a cache hit."""
        out = {k: v for k, v in result.items() if k != "raster_out"}
        out["params"] = {**result.get("params", {}), "show": params["show"]}
        if params["show"] != "edges":
            out["raster_out"] = result[params["show"]]
        return out

    def roi_margin(self, params: dict) -> int:
        """The evolution is an explicit 5-point stencil (edge-replicated),
        so information moves at most 1 px per iteration -- the coarsest level's iteration
        count reaches exactly as far as any ROI pixel can feel. +3 covers np.gradient, the
        NMS / follow neighbours and the subpixel probe. Exact (oracle-pinned)."""
        from dynamix.core import cdf as lcdf

        sigmas = [float(2.0 ** k) for k in range(int(params["n_levels"]))]
        iters = lcdf.sigma_iters(sigmas, theta=float(params["theta"]), dt=float(params["dt"]))
        return int(max(iters.values())) + 3

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k
        from dynamix.model.device import keyed_params

        return _k(self.name, source_id, keyed_params(self, params))
