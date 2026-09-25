# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""ssa2d -- 2D singular-spectrum analysis (Golyandina & Usevich 2010) as a tool.

The rotation-invariant sibling of ``tucker_HOOI_HOSVD``'s 2-D delay embedding: the same
windows, but ONE SVD of the Hankel-block-Hankel matrix, so an eigenarray can be any
``L_r x L_c`` pattern (an oblique wave is one pair of components here, several separable pairs
there). Displays the
reconstruction, the residual or one elementary component. The Group knob is the paper's grouping
step: the reconstruction is the sum of the chosen components (those the w-correlations show
belong together), and the residual is the data minus that sum -- so removing just the first
component is Group "1" with Show = residual. All of that is view-only, computed once and cached
with the decomposition.

The solver (``dynamix.core.ssa2d``): Lanczos by default, or PROPACK or a randomized range
finder -- FFT products with the Hankel-block-Hankel matrix, so windows are limited only by the
knobs -- or the dense lag-covariance route (windows up to 1600 entries). ``keep = energy`` keeps
the fewest components reaching a cumulative share, and ``Show = cluster`` shows one group of the
automatic grouping (average-linkage clustering on 1 - |w-correlation|), both after Lopes et al.
2024's "pragmatic SSA".

Edges: a pixel near the edge of the analysed region is covered by fewer windows. The edge margin
(in windows) gives every pixel of a region of interest the full cover: the runner reads real data
around it (reflected only past the file's edge) and the result is cropped back. On the whole
field there is no real data beyond the edge; mirror extension (Lopes et al.'s signal extension)
is an option, off by default because in 2-D it adds rank (see ``edge_extension``).
"""
from __future__ import annotations

import math

from dynamix.model.param import Param, ParamKind

#: Window entries (L_r * L_c) above which the dense lag-covariance route gets slow (~4 s at
#: 40 x 40 on a 256^2 field); the FFT solvers are not limited by it.
MAX_WINDOW_ENTRIES = 1600

_SOLVER_CHOICES = ("lanczos", "propack", "randomized", "dense")


def _store(a):
    """A large output at the app's precision (float32 at 32 bit): it is kept per layer."""
    import numpy as np

    from dynamix.core.fft_policy import active

    return np.asarray(a, dtype=np.float32 if getattr(active(), "precision", 64) == 32
                      else np.float64)


class SSA2D:
    """2D-SSA: eigentriples of every window of the field, components, groups, w-correlations."""

    name = "ssa2d"
    params = (
        Param("rows_window", ParamKind.INT, default=16, min=2, max=512, soft_min=4,
              soft_max=40, label="Window: rows"),
        Param("cols_window", ParamKind.INT, default=16, min=2, max=512, soft_min=4,
              soft_max=40, label="Window: cols"),
        Param("n_components", ParamKind.INT, default=16, min=1, max=64, soft_min=1,
              soft_max=32, label="Components kept"),
        # "energy": n_components is the most to find; keep the fewest whose cumulative share
        # reaches the threshold.
        Param("keep", ParamKind.CHOICE, default="count", choices=("count", "energy"),
              label="Keep"),
        Param("energy_threshold", ParamKind.FLOAT, default=0.9, min=0.5, max=0.9999,
              soft_min=0.5, soft_max=0.99, label="Energy", active_when=("keep", ("energy",))),
        Param("solver", ParamKind.CHOICE, default="lanczos", choices=_SOLVER_CHOICES,
              label="Solver"),
        Param("power_iterations", ParamKind.INT, default=2, min=0, max=10, soft_min=0,
              soft_max=6, label="Power iterations", active_when=("solver", ("randomized",))),
        Param("margin_windows", ParamKind.FLOAT, default=1.0, min=0.0, max=4.0, soft_min=0.0,
              soft_max=2.0, label="Edge margin (windows)"),
        # Whole-field runs: mirror-extend by the edge margin (Lopes et al.'s signal extension),
        # or not. Off by default: in 2-D a reflected wave is a different wave (an oblique one
        # flips a frequency component), so the extension adds rank; on two test waves it raised
        # the edge error 3-18x, where the real-data margin of a region lowered it.
        Param("edge_extension", ParamKind.CHOICE, default="none", choices=("none", "mirror"),
              label="Whole-field edges"),
        Param("show", ParamKind.CHOICE, default="recon",
              choices=("recon", "residual", "component", "cluster"), label="Show", view=True),
        Param("component", ParamKind.INT, default=1, min=1, max=64, soft_min=1, soft_max=16,
              label="Component", view=True, active_when=("show", ("component",))),
        # Which components form the reconstruction ("1-3, 5"; "all" = every one kept); the
        # residual is the data minus that sum.
        Param("group", ParamKind.TEXT, default="all", label="Group", editable=True, view=True,
              active_when=("show", ("recon", "residual"))),
        # The automatic grouping: which cluster to show (largest share first), and the cut.
        Param("cluster", ParamKind.INT, default=1, min=1, max=64, soft_min=1, soft_max=8,
              label="Cluster", view=True, active_when=("show", ("cluster",))),
        Param("cluster_distance", ParamKind.FLOAT, default=0.5, min=0.05, max=0.95,
              soft_min=0.1, soft_max=0.9, label="Cluster distance", view=True,
              active_when=("show", ("cluster",))),
    )

    def check(self, params: dict) -> None:
        entries = int(params["rows_window"]) * int(params["cols_window"])
        if params.get("solver", "lanczos") == "dense" and entries > MAX_WINDOW_ENTRIES:
            raise ValueError(
                f"ssa2d: a {params['rows_window']} x {params['cols_window']} window has "
                f"{entries} entries; the dense solver takes up to {MAX_WINDOW_ENTRIES} (e.g. "
                "40 x 40) -- use the Lanczos, PROPACK or randomized solver")

    def roi_margin(self, params: dict) -> int:
        """The edge margin: ``margin_windows`` windows of ``max(L_r, L_c) - 1`` pixels -- at
        one window every pixel of the region is covered by the full set of windows."""
        span = max(int(params["rows_window"]), int(params["cols_window"])) - 1
        return int(math.ceil(float(params.get("margin_windows", 1.0)) * span))

    def _decompose(self, vals, params: dict, progress=None) -> dict:
        from dynamix.core.ssa2d import ssa2d

        if vals.ndim != 2:
            raise ValueError("ssa2d takes a single-band field -- use tucker_HOOI_HOSVD (band "
                             "mode) or pca for a multi-band stack")
        return ssa2d(vals, rows_window=int(params["rows_window"]),
                     cols_window=int(params["cols_window"]),
                     n_components=int(params["n_components"]),
                     solver=params.get("solver", "lanczos"),
                     power_iterations=int(params.get("power_iterations", 2)),
                     keep=params.get("keep", "count"),
                     energy_threshold=float(params.get("energy_threshold", 0.9)),
                     progress=progress)

    def _result(self, field, out: dict, r0: int, c0: int, h: int, w: int, params: dict) -> dict:
        """The decomposition's result cut to ``[r0, r0 + h) x [c0, c0 + w)``."""
        from dynamix.devices.decompose import _result

        sl = (slice(r0, r0 + h), slice(c0, c0 + w))
        recon = _store(out["recon"][sl])
        res = _result(field, recon, {
            "ssa_recon": recon, "ssa_residual": _store(out["residual"][sl]),
            "ssa_components": _store(out["components"][(slice(None),) + sl]),
            "ssa_eigen_share": out["eigen_share"],
            "ssa_eigenarrays": _store(out["eigenarrays"]),
            "ssa_factor_arrays": _store(out["factor_arrays"]),
            "ssa_w_correlation": out["w_correlation"],
            "ssa_solver_used": out["solver_used"], "ssa_energy_reached": out["energy_reached"],
        }, params)
        res["_shape"] = (h, w)
        return res

    def compute(self, field, params: dict, *, progress=None) -> dict:
        """The whole field, decomposed -- mirror-extended by the edge margin first when
        ``edge_extension`` is mirror, and cropped back."""
        import numpy as np

        from dynamix.devices.decompose import _field_values

        vals = _field_values(field, self.name)
        mirror = params.get("edge_extension", "none") == "mirror" and vals.ndim == 2
        m = self.roi_margin(params) if mirror else 0
        ext = np.pad(vals, m, mode="symmetric") if m else vals
        out = self._decompose(ext, params, progress)
        return self._result(field, out, m, m, vals.shape[0], vals.shape[1], params)

    def compute_roi(self, window_field, core, params: dict, *, info=None,
                    progress=None) -> dict:
        """A region: the runner's window (the ROI plus the edge margin of real data) decomposed
        as it is, then cut to the ROI."""
        from dynamix.devices.decompose import _field_values

        vals = _field_values(window_field, self.name)
        out = self._decompose(vals, params, progress)
        r0, c0, h, w = (int(v) for v in core)
        return self._result(window_field, out, r0, c0, h, w, params)

    def view(self, result: dict, params: dict) -> dict:
        """Show recon / residual (of the chosen group) / one component / one cluster -- a cache
        hit, never a re-decomposition. ``_view_note`` is what the layer row appends."""
        import numpy as np

        from dynamix.core.ssa2d import cluster_components
        from dynamix.devices.decompose import format_group, group_view

        comps = result["ssa_components"]
        share = np.asarray(result["ssa_eigen_share"])
        clusters, linkage = cluster_components(result["ssa_w_correlation"], share,
                                               float(params.get("cluster_distance", 0.5)))
        show = params["show"]
        if show == "component":
            k = min(max(int(params["component"]), 1), len(comps))
            shown, note = comps[k - 1], f"C{k} ({100.0 * float(share[k - 1]):.0f}%)"
        elif show == "cluster":
            c = min(max(int(params.get("cluster", 1)), 1), len(clusters))
            members = clusters[c - 1]
            shown = np.asarray(comps)[members].sum(axis=0)
            note = (f"cluster {c}/{len(clusters)}: {format_group(members)} "
                    f"({100.0 * float(share[members].sum()):.0f}%)")
        else:
            shown, note = group_view(result["ssa_recon"], result["ssa_residual"], comps, share,
                                     show, params["group"])
            if note is None:
                note = f"recon ({len(comps)} components)" if show == "recon" else "residual"
            elif note == "no valid group":
                note = (f"recon ({len(comps)} components; no valid group)" if show == "recon"
                        else "residual (no valid group)")
        if not result.get("ssa_energy_reached", True):
            note += " — energy threshold not reached"
        return {**result, "raster_out": shown, "_view_note": note,
                "_ssa_clusters": clusters, "_ssa_linkage": linkage,
                "params": {**result.get("params", {}), "show": show,
                           "component": params["component"], "group": params["group"],
                           "cluster": params.get("cluster", 1),
                           "cluster_distance": params.get("cluster_distance", 0.5)}}

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k
        from dynamix.model.device import keyed_params

        return _k(self.name, source_id, keyed_params(self, params))
