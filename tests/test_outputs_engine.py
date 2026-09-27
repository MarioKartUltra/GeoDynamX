# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Lazily computed tool outputs. A device declares what it produces; ``resolve`` records the
analysis it ran; ``resolve_output`` computes a lazy output from that cached analysis under a key of
its own whose upstream is the analysis key -- so a view-only knob the output depends on re-keys the
output alone, and any change upstream of it (an analysis knob, the source) re-keys both."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.core.wtmm_backend import ComputeCancelled
from dynamix.engine.cache import Cache, cache_key
from dynamix.engine.resolve import output_key, resolve, resolve_output, source_identity
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import Output, declared_outputs, register_device
from dynamix.model.layer import Layer
from dynamix.model.param import Param, ParamKind


class _Fake:
    name = "fake_lazy"
    params = (Param("k", ParamKind.INT, default=1, min=1, max=9),
              Param("n", ParamKind.INT, default=2, min=1, max=9, view=True),
              Param("show", ParamKind.CHOICE, default="none",
                    choices=("none", "double", "plus"), view=True))
    outputs = (Output("edges", "vector"),
               Output("double", "raster", lazy=True, params=("n",)),
               Output("plus", "raster", lazy=True, params=("n",)))

    def __init__(self):
        self.calls: list[str] = []
        self.computes = 0

    def compute(self, field, params, *, progress=None):
        self.computes += 1
        return {"extrema": [], "chains": [], "params": dict(params), "k": params["k"]}

    def cache_key(self, source_id, params):
        return ""

    def view(self, result, params):
        return {**result, "k": -1}       # differs from the cached result compute_output reads

    def compute_output(self, name, values, result, params, *, fetch, progress=None, cancel=None):
        self.calls.append(name)
        if cancel is not None and cancel():
            raise ComputeCancelled("stopped")
        if name == "double":
            return {"raster": (values * params["n"] * result["k"]).astype(np.float32), "diag": {}}
        return {"raster": fetch("double")["raster"] + 1, "diag": {}}      # keyed under "double"


class _Shift:
    """A leading transform whose result (a shifted field) the analysis consumes."""

    name = "fake_shift"
    params = (Param("by", ParamKind.FLOAT, default=1.0, min=-9.0, max=9.0),)

    def compute(self, field, params, *, progress=None):
        return dataclasses.replace(field, values=field.values + params["by"])

    def cache_key(self, source_id, params):
        return ""


@pytest.fixture
def fake(clean_registry):
    device = _Fake()
    register_device(device)
    register_device(_Shift())
    return device


def _field() -> RasterField:
    rng = np.random.default_rng(7)
    v = rng.standard_normal((40, 48)).cumsum(0).cumsum(1)
    return RasterField(name="f", values=v, frame=LocalFrame(), x_axis=np.arange(48.0),
                       y_axis=np.arange(40.0))


def _layer(shift: bool = False, **params) -> Layer:
    steps = (DeviceRef("fake_lazy", dict(params)),)
    if shift:
        steps = (DeviceRef("fake_shift", {}),) + steps
    return Layer(layer_id=1, name="L", source_id="s", chain=Chain(steps).materialized())


def _out(name: str) -> Output:
    return next(o for o in _Fake.outputs if o.name == name)


def test_resolve_records_the_analysis_it_ran(fake):
    field = _field()
    r = resolve(_layer(n=3), field, Cache())
    assert r.analysis_device == "fake_lazy"
    assert r.analysis_params == {"k": 1, "n": 3, "show": "none"}     # validated, view knobs too
    assert r.analysis_key == cache_key("fake_lazy", source_identity(_layer()), {"k": 1})
    assert r.analysis_input is field
    assert r.result["k"] == -1                                       # the viewed result


def test_the_last_transform_is_the_analysis(fake):
    field, cache = _field(), Cache()
    r = resolve(_layer(shift=True), field, cache)
    sid = source_identity(_layer())
    shift_key = cache_key("fake_shift", sid, {"by": 1.0})
    assert r.analysis_device == "fake_lazy"
    assert r.analysis_key == cache_key("fake_lazy", sid, {"k": 1}, upstream=shift_key)
    assert r.analysis_input is cache.get(shift_key)                  # what the analysis consumed
    out = resolve_output(_layer(shift=True), field, cache, "double")
    np.testing.assert_array_equal(out["raster"], ((field.values + 1.0) * 2).astype(np.float32))


def test_an_output_is_computed_once_then_cached(fake):
    field, cache = _field(), Cache()
    a = resolve_output(_layer(), field, cache, "double")
    b = resolve_output(_layer(), field, cache, "double")
    assert fake.calls == ["double"]
    assert b is a
    np.testing.assert_array_equal(a["raster"], (field.values * 2 * 1).astype(np.float32))
    assert a["raster"].dtype == np.float32
    resolve_output(_layer(show="double"), field, cache, "double")    # a knob it does not read
    assert fake.calls == ["double"]


def test_an_output_knob_rekeys_the_output_alone(fake):
    field, cache = _field(), Cache()
    resolve_output(_layer(n=2), field, cache, "double")
    r2, r3 = resolve(_layer(n=2), field, cache), resolve(_layer(n=3), field, cache)
    assert r3.cache_misses == 0 and r3.analysis_key == r2.analysis_key
    assert (output_key("fake_lazy", _out("double"), r3.analysis_params, r3.analysis_key)
            != output_key("fake_lazy", _out("double"), r2.analysis_params, r2.analysis_key))
    misses = cache.misses
    out = resolve_output(_layer(n=3), field, cache, "double")
    assert cache.misses == misses + 1                                # the output alone
    assert fake.computes == 1                                        # the analysis ran once
    assert fake.calls == ["double", "double"]
    np.testing.assert_array_equal(out["raster"], (field.values * 3 * 1).astype(np.float32))


def test_an_analysis_knob_rekeys_the_output(fake):
    field, cache = _field(), Cache()
    one = resolve_output(_layer(k=1), field, cache, "double")
    two = resolve_output(_layer(k=2), field, cache, "double")
    assert fake.computes == 2 and fake.calls == ["double", "double"]
    np.testing.assert_array_equal(two["raster"], one["raster"] * 2)
    r1, r2 = resolve(_layer(k=1), field, cache), resolve(_layer(k=2), field, cache)
    assert r1.analysis_key != r2.analysis_key
    assert (output_key("fake_lazy", _out("double"), r1.analysis_params, r1.analysis_key)
            != output_key("fake_lazy", _out("double"), r2.analysis_params, r2.analysis_key))


def test_an_output_fetches_another_through_the_cache(fake):
    field, cache = _field(), Cache()
    plus = resolve_output(_layer(), field, cache, "plus")
    assert fake.calls == ["plus", "double"]
    double = resolve_output(_layer(), field, cache, "double")
    np.testing.assert_array_equal(plus["raster"], double["raster"] + 1)
    resolve_output(_layer(), field, cache, "plus")
    assert fake.calls == ["plus", "double"]                          # both were cached


def test_a_cancelled_output_caches_nothing(fake):
    field, cache = _field(), Cache()
    with pytest.raises(ComputeCancelled):
        resolve_output(_layer(), field, cache, "double", cancel=lambda: True)
    r = resolve(_layer(), field, cache)
    assert output_key("fake_lazy", _out("double"), r.analysis_params, r.analysis_key) not in cache
    assert r.analysis_key in cache                                   # the analysis stays
    out = resolve_output(_layer(), field, cache, "double")
    np.testing.assert_array_equal(out["raster"], (field.values * 2).astype(np.float32))


def test_an_roi_layer_is_refused(fake):
    field, cache = _field(), Cache()
    layer = _layer()
    layer.tags["roi.window"] = "0,0,8,8"
    with pytest.raises(ValueError, match="ROI"):
        resolve_output(layer, field, cache, "double")
    assert fake.calls == []
    r = resolve(layer, field, cache)
    assert r.analysis_device == "fake_lazy" and r.analysis_input is None
    assert r.analysis_key == cache_key("fake_lazy", source_identity(layer), {"k": 1})


@pytest.mark.parametrize("name", ["edges", "nope"])
def test_an_eager_or_undeclared_output_is_refused(fake, name):
    with pytest.raises(ValueError, match=name):
        resolve_output(_layer(), _field(), Cache(), name)
    assert fake.calls == []


def test_declared_outputs(fake):
    assert declared_outputs(object()) == ()
    assert declared_outputs(fake) == _Fake.outputs
    assert _out("double").kind == "raster" and _out("double").grid == "native"
