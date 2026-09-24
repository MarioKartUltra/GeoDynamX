# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""derived_vectors -- the vectors of a derivative dataset: the extrema and maxima lines a result
showed when it was forked, read back from the derivative's own file (:mod:`dynamix.core.derivative`).

A derivative dataset whose file carries vectors gets this device as its layer's first step; the
shell writes ``_path`` (hidden, like backproject's stamped params). It is part of the DATASET,
not an analysis: the drop rule treats a layer holding only it as raw (``vector_source``), so a
tool dropped on the derivative still spawns a child that runs on its raster, while filters
dropped on it (scale_select, the extrema and chain filters) narrow the frozen vectors like any
other result's.
"""
from __future__ import annotations

from dynamix.model.param import Param, ParamKind


class DerivedVectors:
    """Emit a derivative dataset's frozen extrema and maxima lines."""

    name = "derived_vectors"
    #: The dataset's own vectors, not an analyzer (``MainWindow._is_raw_raster_dataset``).
    vector_source = True
    params = (
        Param("_path", ParamKind.TEXT, default="", label="Derivative file"),
    )

    def compute(self, field, params: dict, *, progress=None) -> dict:
        import numpy as np

        from dynamix.core.derivative import read_vectors

        path = str(params.get("_path", ""))
        bundle = read_vectors(path) if path else None
        if bundle is None:
            bundle = {"extrema": [], "chains": [], "scales": np.zeros(0)}
        vals = np.asarray(field.values)
        return {**bundle, "params": dict(params), "_frame": getattr(field, "frame", None),
                "_shape": vals.shape[:2]}

    def cache_key(self, source_id: str, params: dict) -> str:
        from dynamix.engine.cache import cache_key as _k
        from dynamix.model.device import keyed_params

        return _k(self.name, source_id, keyed_params(self, params))
