# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The engine hands its stage store to a transform that declares ``uses_stage_cache``."""
from __future__ import annotations

import numpy as np

from dynamix.engine.cache import Cache


def test_resolve_hands_the_store_to_a_transform_that_declares_it(clean_registry):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.engine.resolve import resolve
    from dynamix.model.chain import Chain, DeviceRef
    from dynamix.model.device import register_device
    from dynamix.model.layer import Layer
    from dynamix.model.param import Param, ParamKind

    seen = []

    class _Staged:
        name = "staged"
        params = (Param("k", ParamKind.INT, default=1, min=1, max=9),)
        uses_stage_cache = True

        def compute(self, field, params, *, progress=None, stage_cache=None):
            seen.append(stage_cache)
            return {"extrema": [], "chains": []}

        def cache_key(self, source_id, params):
            return ""

    register_device(_Staged())
    field = RasterField(name="f", values=np.zeros((8, 8)), frame=LocalFrame(),
                        x_axis=np.arange(8.0), y_axis=np.arange(8.0))
    cache = Cache()
    layer = Layer(layer_id=1, name="L", source_id="s",
                  chain=Chain((DeviceRef("staged", {}),)).materialized())
    resolve(layer, field, cache)
    assert seen == [cache.stages]
