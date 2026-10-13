# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The partition tables follow the rack's chain filters: resolve stamps the transform's chain
count, so a filtered result can rebuild its tables over the survivors and say so."""
from __future__ import annotations

import numpy as np

from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache
from dynamix.engine.resolve import resolve
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.layer import Layer


def _field():
    rng = np.random.default_rng(5)
    vals = rng.standard_normal((64, 64)).cumsum(0).cumsum(1)
    return RasterField._from_bare_array(vals, "f", name="f")


def test_resolve_stamps_the_transforms_chain_count(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device

    register_builtin_devices()
    params = {**defaults_for(get_device("wtmm2d")), "n_oct": 2, "n_voice": 2}
    base = Chain((DeviceRef("wtmm2d", params),)).materialized()
    field, cache = _field(), Cache()
    full = resolve(Layer(layer_id=1, name="L", source_id="s", chain=base), field, cache).result
    n = len(full["chains"])
    assert n > 2 and full.get("_chains_total", n) == n       # untouched: no stamp needed
    long = Chain(base.steps + (DeviceRef("chain_length", {"min_len": 1000}),)).materialized()
    filtered = resolve(Layer(layer_id=2, name="F", source_id="s", chain=long), field, cache).result
    assert len(filtered["chains"]) < n and filtered["_chains_total"] == n
