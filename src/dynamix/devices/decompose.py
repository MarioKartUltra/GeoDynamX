# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Decomposition transforms: PCA and Tucker-HAVOK -- the common-tools additions (2026-09-21).

The originating code is the author's ``aster_him.ipynb``; the engines are the efficiency-mandated rebuilds in
:mod:`dynamix.core.pca` (covariance-trick PCA) and :mod:`dynamix.core.tucker_havok`
(stride-view Hankel + blockwise-Gram HOSVD -- the notebook's materialized-tensor crash class is
structurally impossible here).

Both are standalone Transforms heading their own chain (the ``field_values`` refusal contract),
displaying through ``raster_out`` (the band_recon precedent: stretch/colormap/hillshade/export
free-ride). PCA needs a MULTI-COMPONENT field ``(ny, nx, nc >= 2)`` -- today that arrives via
npz stacks (the GeoTIFF opener reads band 1 only; multiband/ASTER-HDF ingestion is its own
future step). Tucker-HAVOK takes scalar 2-D or multi-component fields alike.
"""
from __future__ import annotations

from dynamix.model.param import Param, ParamKind


def _field_values(field, device_name: str, *, min_comps: int = 1):
    """The layer field's values, 2-D or (ny, nx, nc) -- the ``field_values_2d`` refusals with
    the multi-component gate this module needs instead of the scalar-only one."""
    import numpy as np

    if isinstance(field, dict) or not hasattr(field, "values"):
        raise ValueError(
            f"{device_name} analyses the layer's own field -- place it FIRST in its chain "
            "(a Transform placed after another receives that transform's result)")
    vals = np.asarray(field.values, dtype=np.float64)
    if vals.ndim not in (2, 3):
        raise ValueError(f"{device_name} needs a 2-D or (ny, nx, nc) field; got {vals.shape}")
    nc = vals.shape[2] if vals.ndim == 3 else 1
    if nc < min_comps:
        # 2026-09-21: never say "components" here -- that is the KEEP knob's name (how many
        # PCs to retain), while THIS gate is about input BANDS. Same word, two meanings, one
        # dialog: the old message read as self-contradictory next to Components=6.
        raise ValueError(
            f"{device_name} runs across BANDS and needs a multi-band field "
            f"((ny, nx, nc), nc >= {min_comps}); this field has {nc}. The Components knob "
            f"picks how many PCs to KEEP, not how many bands exist -- open a multi-band "
            f"stack (npz/GeoTIFF (ny, nx, nc)), or use tucker on a single 2-D field")
    return vals


def _result(field, raster, extras: dict, params: dict) -> dict:
    import numpy as np

    vals = np.asarray(field.values)
    return {
        "raster_out": raster, "chains": [], "extrema": [],
        "params": dict(params), "_frame": getattr(field, "frame", None),
        "_shape": vals.shape[:2], **extras,
    }


class PCADevice:
    """Principal component analysis across a stack's bands; shows one component raster."""

    name = "pca"

    def roi_margin(self, params: dict) -> int:
        """A global decomposition is a statistic OF the analysed region
        -- margin pixels would mix outside data into it, so an ROI decomposes itself."""
        return 0

    params = (
        Param("n_components", ParamKind.INT, default=6, min=1, max=64,
              soft_min=1, soft_max=8, label="Components"),
        # Which component raster ``raster_out`` shows (1-based, clipped to what exists).
        Param("component", ParamKind.INT, default=1, min=1, max=64,
              soft_min=1, soft_max=6, label="Show PC"),
        # Correlation-PCA (per-band standardization) -- for incommensurate band units.
        Param("standardize", ParamKind.BOOL, default=False, label="Standardize"),
    )

    def compute(self, field, params: dict, *, progress=None) -> dict:
        import numpy as np

        from dynamix.core.pca import fit_pca

        vals = _field_values(field, self.name, min_comps=2)
        out = fit_pca(vals, int(params["n_components"]),
                      standardize=bool(params["standardize"]))
        k = out["images"].shape[0]
        idx = min(max(int(params["component"]), 1), k) - 1
        return _result(field, np.asarray(out["images"][idx], dtype=np.float64), {
            "pca_images": out["images"], "pca_components": out["components"],
            "explained_var_ratio": out["explained_var_ratio"],
        }, params)

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)


class TuckerHavok:
    """Delay-embedded Tucker (HOSVD) decomposition; shows the rank-truncated reconstruction
    or its residual."""

    name = "tucker_havok"

    def roi_margin(self, params: dict) -> int:
        """A global decomposition is a statistic OF the analysed region
        -- margin pixels would mix outside data into it, so an ROI decomposes itself."""
        return 0

    param_groups = {
        "EMBED": ("embed", "axis", "n_delays"),
        "RANKS": ("rank_delay", "rank_time", "rank_space", "rank_band"),
    }
    params = (
        # "delay" = the HAVOK Hankel embedding; "none" = Tucker of the
        # raw array's own modes (rows, cols[, band]) -- rank_delay/n_delays/axis are then
        # inert (the k_edge precedent; still keyed) and rank_time/rank_space read as the
        # rows/cols ranks their labels also name.
        Param("embed", ParamKind.CHOICE, default="delay", choices=("delay", "none"),
              label="Embedding"),
        Param("axis", ParamKind.CHOICE, default="rows", choices=("rows", "cols"),
              label="Tape axis"),
        Param("n_delays", ParamKind.INT, default=32, min=2, max=512,
              soft_min=8, soft_max=64, label="Delays"),
        # Rank 0 = "no truncation on this mode" (the mode's full size).
        Param("rank_delay", ParamKind.INT, default=4, min=0, max=512,
              soft_min=1, soft_max=16, label="Rank: delay"),
        Param("rank_time", ParamKind.INT, default=0, min=0, max=100000,
              soft_min=0, soft_max=64, label="Rank: time/rows"),
        Param("rank_space", ParamKind.INT, default=0, min=0, max=100000,
              soft_min=0, soft_max=64, label="Rank: space/cols"),
        Param("rank_band", ParamKind.INT, default=0, min=0, max=64,
              soft_min=0, soft_max=8, label="Rank: band"),
        Param("sweeps", ParamKind.INT, default=0, min=0, max=16,
              soft_min=0, soft_max=4, label="HOOI sweeps"),
        Param("show", ParamKind.CHOICE, default="recon", choices=("recon", "residual"),
              label="Show"),
    )

    def compute(self, field, params: dict, *, progress=None) -> dict:
        import numpy as np

        from dynamix.core.tucker_havok import tucker_havok, tucker_plain

        vals = _field_values(field, self.name)
        plain = params.get("embed", "delay") == "none"
        ranks = ([] if plain else [params["rank_delay"]]) \
            + [params["rank_time"], params["rank_space"]]
        if vals.ndim == 3:
            ranks.append(params["rank_band"])
        if progress is not None:
            progress("tucker decomposition", 0.1)
        if plain:
            out = tucker_plain(vals, ranks=ranks, sweeps=int(params["sweeps"]))
        else:
            out = tucker_havok(vals, n_delays=int(params["n_delays"]), ranks=ranks,
                               axis=params["axis"], sweeps=int(params["sweeps"]))
        if progress is not None:
            progress("tucker decomposition", 1.0)
        shown = out["recon"] if params["show"] == "recon" else out["residual"]
        if shown.ndim == 3:
            shown = shown[..., 0]           # multiband display: first band of the product
        return _result(field, np.asarray(shown, dtype=np.float64), {
            "tucker_core": out["core"], "tucker_factors": out["factors"],
            "tucker_energy": out["energy"], "shape_embedded": out["shape_embedded"],
        }, params)

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)
