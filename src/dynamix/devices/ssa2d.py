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
"""
from __future__ import annotations

from dynamix.model.param import Param, ParamKind

#: Window entries (L_r * L_c) above which the lag-covariance route gets slow (~4 s at 40 x 40 on
#: a 256^2 field); larger windows want the Lanczos + FFT-matvec route, which is not built.
MAX_WINDOW_ENTRIES = 1600


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
        Param("show", ParamKind.CHOICE, default="recon",
              choices=("recon", "residual", "component"), label="Show", view=True),
        Param("component", ParamKind.INT, default=1, min=1, max=64, soft_min=1, soft_max=16,
              label="Component", view=True, active_when=("show", ("component",))),
        # Which components form the reconstruction ("1-3, 5"; "all" = every one kept); the
        # residual is the data minus that sum.
        Param("group", ParamKind.TEXT, default="all", label="Group", editable=True, view=True,
              active_when=("show", ("recon", "residual"))),
    )

    def check(self, params: dict) -> None:
        entries = int(params["rows_window"]) * int(params["cols_window"])
        if entries > MAX_WINDOW_ENTRIES:
            raise ValueError(
                f"ssa2d: a {params['rows_window']} x {params['cols_window']} window has "
                f"{entries} entries; keep it to {MAX_WINDOW_ENTRIES} (e.g. 40 x 40) -- larger "
                "windows need the Lanczos/FFT route, not built")

    def roi_margin(self, params: dict) -> int:
        """A decomposition is a statistic OF the analysed region -- margin pixels would mix
        outside data into it (the pca/tucker precedent)."""
        return 0

    def compute(self, field, params: dict, *, progress=None) -> dict:
        import numpy as np

        from dynamix.core.ssa2d import ssa2d
        from dynamix.devices.decompose import _field_values, _result

        vals = _field_values(field, self.name)
        if vals.ndim != 2:
            raise ValueError("ssa2d takes a single-band field -- use tucker_HOOI_HOSVD (band "
                             "mode) or pca for a multi-band stack")
        out = ssa2d(vals, rows_window=int(params["rows_window"]),
                    cols_window=int(params["cols_window"]),
                    n_components=int(params["n_components"]), progress=progress)
        recon = np.asarray(out["recon"], dtype=np.float64)
        return _result(field, recon, {
            "ssa_recon": recon, "ssa_residual": np.asarray(out["residual"], dtype=np.float64),
            "ssa_components": out["components"], "ssa_eigen_share": out["eigen_share"],
            "ssa_eigenarrays": out["eigenarrays"], "ssa_factor_arrays": out["factor_arrays"],
            "ssa_w_correlation": out["w_correlation"],
        }, params)

    def view(self, result: dict, params: dict) -> dict:
        """Show recon / residual (of the chosen group) / one component -- a cache hit, never a
        re-decomposition. ``_view_note`` is what the layer row appends."""
        from dynamix.devices.decompose import group_view

        comps = result["ssa_components"]
        share = result["ssa_eigen_share"]
        show = params["show"]
        if show == "component":
            k = min(max(int(params["component"]), 1), len(comps))
            shown, note = comps[k - 1], f"C{k} ({100.0 * float(share[k - 1]):.0f}%)"
        else:
            shown, note = group_view(result["ssa_recon"], result["ssa_residual"], comps, share,
                                     show, params["group"])
            if note is None:
                note = f"recon ({len(comps)} components)" if show == "recon" else "residual"
            elif note == "no valid group":
                note = (f"recon ({len(comps)} components; no valid group)" if show == "recon"
                        else "residual (no valid group)")
        return {**result, "raster_out": shown, "_view_note": note,
                "params": {**result.get("params", {}), "show": show,
                           "component": params["component"], "group": params["group"]}}

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k
        from dynamix.model.device import keyed_params

        return _k(self.name, source_id, keyed_params(self, params))
