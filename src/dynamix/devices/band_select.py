# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Band select: one band of a multiband stack as the working field, as a CHAIN STEP.

A multiband dataset (an imported sensor group, a multiband GeoTIFF) is ``(ny, nx, nc)``;
single-band tools want one 2-D plane. This field stage slices it in the chain, so the choice
composes, caches by its own params, and is visible in the rack -- never a hidden default.
The band's NAME (``provenance["bands"]``, stamped at import) labels the output; the slice is
a view of the same grid: axes, frame and CRS ride through untouched.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from dynamix.model.param import Param, ParamKind


class BandSelect:
    """One band of the stack, picked by its 1-based index."""

    name = "band_select"
    #: A FIELD STAGE (field -> field), pointwise: no margin.
    field_stage = True

    def roi_margin(self, params: dict) -> int:
        return 0

    params = (
        Param("band", ParamKind.INT, default=1, min=1, max=256, soft_min=1, soft_max=16,
              label="Band"),
        # A band ROW's own band, by NAME (provenance["bands"]): it wins over the index, so
        # removing another band from the dataset never shifts which band this row shows.
        Param("_band_id", ParamKind.TEXT, default="", label="Band id"),
    )

    def compute(self, field, params: dict, *, progress=None):
        if isinstance(field, dict):
            raise ValueError("band_select runs on the FIELD, before the analysing transform "
                             "— drag it to the front of the chain")
        values = np.asarray(field.values)
        names = (getattr(field, "provenance", {}) or {}).get("bands") or []
        band_id = str(params.get("_band_id") or "")
        if band_id and values.ndim == 2 and list(names) == [band_id]:
            prov = {**(field.provenance or {}), "band": band_id}   # the dataset IS this band
            return dataclasses.replace(field, name=f"{field.name}:{band_id}", provenance=prov)
        if values.ndim != 3:
            raise ValueError("band_select needs a multiband stack (ny, nx, nc); "
                             "this field is already single-band")
        nc = int(values.shape[-1])
        if band_id:
            if band_id not in names:
                raise ValueError(f"band {band_id} is no longer in this dataset")
            k = list(names).index(band_id) + 1
        else:
            k = int(params["band"])
        if not 1 <= k <= nc:
            raise ValueError(f"band {k} does not exist: the stack has {nc} band(s)")
        label = str(names[k - 1]) if k - 1 < len(names) else f"band {k}"
        prov = dict(getattr(field, "provenance", {}) or {})
        prov["band"] = label
        return dataclasses.replace(
            field, values=np.asarray(values[..., k - 1], dtype=np.float64),
            name=f"{field.name}:{label}", provenance=prov)

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k

        return _k(self.name, source_id, params)
