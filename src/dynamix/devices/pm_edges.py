# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""pm_edges — Perona-Malik 1990 anisotropic diffusion as a peer analyzer.

One PM evolution (:mod:`dynamix.core.pm` — the paper's scheme (7)+(8)+(10): real
gradient-driven conduction ``g(|neighbor difference|)``, ``lam <= 1/4``, adiabatic
boundaries), snapshotted at the nominal-sigma dyadic schedule, per-snapshot grad-NMS
maxima out in the app's extrema schema. A peer of cdf_edges, never a mode of it: the
flow is real and its edge signal is the GRADIENT (Perona-Malik proper), where the cdf
family is complex with the Im channel as edge detector. What this device adds that no
sibling has is the paper's regime past the flux peak ``phi(s) = s*g(s)``: locally
backward diffusion that SHARPENS edges while the discrete max principle keeps the
evolution stable — so edges stay sharp and IN PLACE across the whole stack (immediate
localization, pinned as behavior in the tests) instead of blurring and drifting as in
any linear scale space.

``scales`` are NOMINAL effective Gaussian sigmas in px (the K -> inf heat-equation
limit of the schedule — the flow is nonlinear, so no true sigma exists; the ladder
exists to align snapshots with the other devices' dyadic sigma stacks). Edge extraction
is grad-NMS of the diffused image per snapshot (the paper's own recipe: threshold the
gradient of the diffused image; thinning is what NMS does). Output speaks the per-scale
extrema schema (x/y/mod/arg/line_id — the mz_edges precedent); ``chains`` is stamped
empty deliberately — cross-scale linking is reserved, as everywhere. ``filtered`` (the
final diffused raster) rides the result; ``show="filtered"`` puts it on the canvas via
the raster_out display contract. Standalone transform: analyses the LAYER'S FIELD,
refuses upstream results with the shared pointed message.
"""
from __future__ import annotations

from dynamix.devices.cdf_edges import _line_ids
from dynamix.devices.holder_map import field_values_2d
from dynamix.model.param import Param, ParamKind


class PMEdges:
    wants_cancel = True                  # the stop button reaches the snapshot loop
    name = "pm_edges"
    params = (
        Param("n_levels", ParamKind.INT, default=4, min=1, max=8,
              soft_min=2, soft_max=5, label="Levels J"),
        # The paper's K, in IMAGE units: the flux peak sits near K, so contrasts above it
        # sharpen, below it blur (their section IV-B reading).
        Param("k_edge", ParamKind.FLOAT, default=1.0, min=0.001, max=1000.0,
              soft_min=0.05, soft_max=5.0, units="", label="K"),
        # Both paper nonlinearities: "exp" privileges high-contrast edges, "frac" wide
        # regions -- their own characterization of the two scale-spaces.
        Param("g", ParamKind.CHOICE, default="exp",
              choices=("exp", "frac"), label="g"),
        # The paper's stability condition is 0 <= lambda <= 1/4 (their eq. (7)).
        Param("lam", ParamKind.FLOAT, default=0.2, min=0.001, max=0.25,
              soft_min=0.1, soft_max=0.25, units="", label="λ"),
        Param("show", ParamKind.CHOICE, default="edges",
              choices=("edges", "filtered"), label="Show", view=True),
        # xsmurf's follow takes four IMAGES -- detector="follow" feeds it this device's own
        # smoothed snapshots via FD derivative stacks (dynamix.core.xsmurf_follow.kapa_from_field)
        # and runs the exact ported detector; edge_mode/interpolate are inert under it (follow's
        # channels are native).
        Param("detector", ParamKind.CHOICE, default="nms",
              choices=("nms", "follow"), label="Detector"),
        # Subpixel refinement: the wtmm2d parabola along the gradient (the "mz"-mode
        # recipe of cdf_edges; PM has no second channel, so there is only one mode).
        Param("interpolate", ParamKind.BOOL, default=False, label="Interpolate"),
        Param("floor", ParamKind.FLOAT, default=0.05, min=0.0, max=0.5,
              soft_min=0.02, soft_max=0.2, units="", label="Floor"),
    )

    def compute(self, field, params: dict, *, progress=None, cancel=None) -> dict:
        import numpy as np

        from dynamix.core import cdf as _cdf
        from dynamix.core import pm as _pm

        vals = field_values_2d(field, self.name)
        k_edge, lam = float(params["k_edge"]), float(params["lam"])
        sigmas = [float(2.0 ** j) for j in range(int(params["n_levels"]))]
        iters = _pm.sigma_iters(sigmas, lam=lam)
        if progress is not None:
            progress("pm evolve", 0.0)
        # Chunked between snapshot counts -- the same op sequence as one call (the core
        # pins exact continuation for the autonomous scheme), with a progress tick per
        # segment (the cdf_edges precedent: the mz silence disease).
        snaps, I, done = {}, np.asarray(vals, dtype=np.float64), 0
        total = max(iters.values())
        for s in sigmas:
            # Cooperative stop: the same stage-boundary contract run_wtmm2d honors -- checked
            # before each evolution chunk.
            if cancel is not None and cancel():
                from dynamix.core.wtmm_backend import ComputeCancelled
                raise ComputeCancelled()
            I, _ = _pm.pm_evolve(I, k_edge, iters[s] - done, g=params["g"], lam=lam)
            done = iters[s]
            snaps[done] = I
            if progress is not None:
                progress("pm evolve", done / total)
        if progress is not None:
            progress("pm edges", 0.8)
        extrema = []
        follow = params.get("detector") == "follow"
        runs_all, closed_all = [], []
        for s in sigmas:
            snap = snaps[iters[s]]
            if follow:
                from dynamix.core.xsmurf_follow import follow_layer_from_snapshot
                layer, runs, closed = follow_layer_from_snapshot(snap, float(params["floor"]))
                runs_all.append(runs)
                closed_all.append(closed)
                extrema.append(layer)
                continue
            mask, coeff = _cdf.grad_nms(snap, float(params["floor"]))
            gy, gx = np.gradient(snap)
            ys, xs = np.where(mask)
            arg_pts = np.arctan2(gy, gx)[ys, xs]
            layer = {
                "x": xs.astype(np.int64), "y": ys.astype(np.int64),
                "mod": coeff[ys, xs],
                "arg": arg_pts,
                "line_id": _line_ids(mask, ys, xs),
            }
            if params["interpolate"] and xs.size:
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
            progress("pm edges", 1.0)
        out = {
            "extrema": extrema, "chains": [],
            "scales": np.asarray(sigmas, dtype=np.float64),
            "filtered": I.copy(),
            "params": dict(params),
            "_frame": getattr(field, "frame", None), "_shape": vals.shape,
        }
        if follow:
            out["_hline_runs"] = runs_all
            out["_hline_closed"] = closed_all
        return out

    def view(self, result: dict, params: dict) -> dict:
        """``show`` is view-only: the filtered field is always computed and cached; showing it
        is a cache hit."""
        out = {k: v for k, v in result.items() if k != "raster_out"}
        out["params"] = {**result.get("params", {}), "show": params["show"]}
        if params["show"] == "filtered":
            out["raster_out"] = result["filtered"]
        return out

    def roi_margin(self, params: dict) -> int:
        """Explicit Perona-Malik (edge-replicated neighbour differences)
        moves information at most 1 px per iteration; +3 for the gradient / NMS / follow /
        subpixel stencils. Exact (oracle-pinned)."""
        from dynamix.core import pm as _pm

        sigmas = [float(2.0 ** j) for j in range(int(params["n_levels"]))]
        iters = _pm.sigma_iters(sigmas, lam=float(params["lam"]))
        return int(max(iters.values())) + 3

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k
        from dynamix.model.device import keyed_params

        return _k(self.name, source_id, keyed_params(self, params))
