# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Per-scale raster outputs. An output declared ``per_level`` has one cache entry per level of the
analysis's scale stack: ``output_key`` folds the level in and ``resolve_output`` hands it to
``compute_output``; every other output's key and call are unchanged. Each tool computes a level
with its own per-scale routine and the analysis's params, so the level-k raster shows what the
analysis detected at level k: with the NMS detector and Interpolate off, the NMS maxima of
wtmm2d's level-k modulus are its stored level-k extrema, and mz_edges' level-l modulus reads each
stored maximum's own ``mod``. The CDF / PM evolution runs once for every level. A band row of a
multiband stack works as any field does."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache, cache_key
from dynamix.engine.resolve import output_key, resolve, resolve_output
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import (Output, declared_outputs, defaults_for, get_device,
                                  keyed_params, register_device, validate_params)
from dynamix.model.layer import Layer
from dynamix.model.param import Param, ParamKind

FAST_WTMM = {"n_oct": 2, "n_voice": 2}


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _field(values, name="f") -> RasterField:
    ny, nx = values.shape[:2]
    return RasterField(name=name, values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(nx)), y_axis=np.arange(float(ny)))


def _smooth(ny=48, nx=56, seed=0):
    return np.random.default_rng(seed).standard_normal((ny, nx)).cumsum(0).cumsum(1)


def _layer(*steps, tags=None) -> Layer:
    chain = Chain(tuple(DeviceRef(n, dict(p)) for n, p in steps)).materialized()
    return Layer(layer_id=1, name="L", source_id="s", chain=chain, tags=dict(tags or {}))


def _out(device: str, name: str) -> Output:
    return next(o for o in declared_outputs(get_device(device)) if o.name == name)


# --------------------------------------------------------------------- the engine contract


class _Levels:
    """A fake analysis with a per-level output and a plain one."""

    name = "fake_levels"
    params = (Param("k", ParamKind.INT, default=1, min=1, max=9),)
    outputs = (Output("plain", "raster", lazy=True),
               Output("each", "raster", lazy=True, per_level=True))

    def __init__(self):
        self.calls: list = []

    def compute(self, field, params, *, progress=None):
        return {"extrema": [], "chains": [], "scales": np.array([1.0, 2.0, 4.0])}

    def cache_key(self, source_id, params):
        return ""

    def compute_output(self, name, values, result, params, *, fetch, progress=None,
                       cancel=None, **extra):
        self.calls.append((name, extra))
        if name == "plain":
            return {"raster": np.zeros((2, 2), np.float32), "diag": {}}
        return {"raster": np.full((2, 2), extra["level"], np.float32), "diag": {}}


@pytest.fixture
def fake(clean_registry):
    device = _Levels()
    register_device(device)
    return device


def test_output_is_not_per_level_by_default():
    assert Output("x", "raster").per_level is False


def test_a_plain_output_key_ignores_the_level():
    out = Output("plain", "raster", lazy=True, params=("n",))
    want = cache_key("dev/plain", "", {"n": 3}, upstream="A")
    assert output_key("dev", out, {"n": 3}, "A") == want
    assert output_key("dev", out, {"n": 3}, "A", level=5) == want


def test_a_per_level_output_key_folds_the_level_in():
    out = Output("each", "raster", lazy=True, per_level=True)
    keys = {output_key("dev", out, {}, "A", level=k) for k in range(3)}
    assert len(keys) == 3
    assert output_key("dev", out, {}, "A") == output_key("dev", out, {}, "A", level=0)
    assert output_key("dev", out, {}, "A", level=2) == cache_key(
        "dev/each", "", {"_level": 2}, upstream="A")


def test_resolve_output_hands_the_level_to_per_level_outputs_only(fake):
    field, cache = _field(_smooth()), Cache()
    layer = _layer(("fake_levels", {}))
    resolve_output(layer, field, cache, "plain", level=2)
    a = resolve_output(layer, field, cache, "each", level=2)
    b = resolve_output(layer, field, cache, "each", level=1)
    assert fake.calls == [("plain", {}), ("each", {"level": 2}), ("each", {"level": 1})]
    assert float(a["raster"][0, 0]) == 2.0 and float(b["raster"][0, 0]) == 1.0
    misses = cache.misses
    resolve_output(layer, field, cache, "each", level=2)            # its own cached entry
    resolve_output(layer, field, cache, "plain", level=0)           # one entry for every level
    assert cache.misses == misses and len(fake.calls) == 3


def test_no_level_is_level_zero(fake):
    field, cache = _field(_smooth()), Cache()
    layer = _layer(("fake_levels", {}))
    resolve_output(layer, field, cache, "each")
    assert fake.calls == [("each", {"level": 0})]
    resolve_output(layer, field, cache, "each", level=0)
    assert len(fake.calls) == 1
