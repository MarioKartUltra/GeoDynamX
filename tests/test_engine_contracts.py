# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The engine skips a step that does not apply (``model.contracts.step_states``): it is never
computed or applied, a skipped transform neither advances ``upstream`` nor becomes the analysis,
the ROI region job takes applying steps only, and ``Renderable.step_states`` records every
step's state. A chain whose steps all apply resolves as before, multiband chains included."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.chain_product import materialize_selection
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache, cache_key
from dynamix.engine.resolve import (level_keeps, output_key, preview_resolve, resolve,
                                    resolve_output, selection_recipe, selection_steps)
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.contracts import EXTREMA, FIELD, STACK, VCHAINS
from dynamix.model.device import (apply_view, declared_outputs, get_device, keyed_params,
                                  register_device, validate_params)
from dynamix.model.families import label_of
from dynamix.model.layer import Layer

NY, NX = 64, 56
WTMM = {"n_oct": 2, "n_voice": 2}
MZ = {"n_levels": 3, "iterations": 2, "run_mode": "fixed"}


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _field(values=None) -> RasterField:
    if values is None:
        values = np.random.default_rng(0).standard_normal((NY, NX)).cumsum(0).cumsum(1)
    return RasterField(name="f", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(values.shape[1])),
                       y_axis=np.arange(float(values.shape[0])))


def _stack() -> RasterField:
    v = np.random.default_rng(0).standard_normal((NY, NX)).cumsum(0).cumsum(1)
    return _field(np.stack([v, v[::-1], v[:, ::-1]], axis=-1))


def _layer(*steps, window: str | None = None) -> Layer:
    chain = Chain(tuple(DeviceRef(n, dict(p)) for n, p in steps)).materialized()
    layer = Layer(layer_id=1, name="L", source_id="s", chain=chain)
    if window is not None:
        layer.tags["roi.window"] = window
    return layer


def _counts(result) -> list:
    return [len(e["x"]) for e in result["extrema"]]


class _StackOnly:
    """A field stage that takes a band stack only: on a single-band field it does not apply,
    so it must never compute there."""

    name = "stack_only_t"
    params = ()
    field_stage = True
    consumes = ((STACK,),)
    produces = (FIELD,)

    def roi_margin(self, params):
        return 0

    def compute(self, field, params, *, progress=None):
        raise AssertionError("a step that does not apply never computes")

    def cache_key(self, source_id, params):
        return f"stack_only:{source_id}"


class _Blur:
    """A first analyzer with a one-pixel halo; it declares no contracts, so it always applies."""

    name = "blur_t"
    params = ()

    def roi_margin(self, params):
        return 1

    def compute(self, field, params, *, progress=None):
        v = np.asarray(field.values)
        p = np.pad(v, 1, mode="edge")
        out = sum(p[1 + a:1 + a + v.shape[0], 1 + b:1 + b + v.shape[1]]
                  for a in (-1, 0, 1) for b in (-1, 0, 1)) / 9.0
        return {"raster_out": out, "extrema": [], "chains": [], "_shape": out.shape}

    def cache_key(self, source_id, params):
        return f"blur:{source_id}"


def test_a_transform_that_does_not_apply_never_runs_and_never_threads_upstream(clean_registry):
    register_device(_StackOnly())
    register_device(_Blur())
    field = _field()
    r = resolve(_layer(("stack_only_t", {}), ("blur_t", {})), field, Cache())
    assert [s.applies for s in r.step_states] == [False, True]
    assert r.step_states[0].reason
    assert r.transforms_run == ("blur_t",)
    assert r.analysis_device == "blur_t" and r.analysis_input is field
    assert r.analysis_key == cache_key("blur_t", "s", {}, upstream=None)


def test_the_roi_region_job_takes_applying_steps_only(clean_registry):
    register_device(_StackOnly())
    register_device(_Blur())
    field = _field()
    r = resolve(_layer(("stack_only_t", {}), ("blur_t", {}), window="10,12,8,9"), field, Cache())
    alone = resolve(_layer(("blur_t", {}), window="10,12,8,9"), field, Cache())
    assert r.transforms_run == ("blur_t",)
    assert r.result["raster_out"].shape == (8, 9)
    assert r.analysis_key == alone.analysis_key
    np.testing.assert_array_equal(r.result["raster_out"], alone.result["raster_out"])


def test_min_vchains_after_a_chainless_analysis_leaves_the_h_lines(builtins):
    field, cache = _field(), Cache()
    alone = resolve(_layer(("cdf_edges", {})), field, cache)
    r = resolve(_layer(("cdf_edges", {}), ("chain_topology", {}), ("min_vchains", {})),
                field, cache)
    assert [s.applies for s in r.step_states] == [True, False, False]
    current = f"({label_of(get_device('cdf_edges'))})"
    assert current in r.step_states[1].reason and current in r.step_states[2].reason
    assert r.transforms_run == ("cdf_edges",) and r.filters_run == ()
    assert r.analysis_device == "cdf_edges" and r.analysis_key == alone.analysis_key
    assert r.cache_misses == 0
    assert "topology" not in r.result
    assert _counts(r.result) == _counts(alone.result)


def test_chain_topology_after_mz_edges_is_skipped_and_mz_edges_stays_the_analysis(builtins):
    field, cache = _field(), Cache()
    layer = _layer(("mz_edges", MZ), ("chain_topology", {}))
    alone = _layer(("mz_edges", MZ))
    r = resolve(layer, field, cache)
    assert [s.applies for s in r.step_states] == [True, False]
    assert r.analysis_device == "mz_edges" and r.transforms_run == ("mz_edges",)
    assert r.analysis_key == resolve(alone, field, cache).analysis_key
    coarse = resolve_output(layer, field, cache, "coarse")
    assert "raster" in coarse
    assert resolve_output(alone, field, cache, "coarse") is coarse


def test_an_all_applying_chain_resolves_as_before(builtins):
    steps = (("wtmm2d", WTMM), ("scale_select", {"scale_idx": 1}),
             ("modulus_threshold", {"frac": 0.3}))
    field, cache = _field(), Cache()
    r = resolve(_layer(*steps), field, cache)
    assert len(r.step_states) == 3 and all(s.applies for s in r.step_states)
    assert r.transforms_run == ("wtmm2d",)
    assert r.filters_run == ("scale_select", "modulus_threshold")
    p = [validate_params(get_device(n), q) for n, q in steps]
    wtmm = get_device("wtmm2d")
    assert r.analysis_key == cache_key("wtmm2d", "s", keyed_params(wtmm, p[0]))
    want = apply_view(wtmm, cache.get(r.analysis_key), p[0])
    for (name, _q), params in zip(steps[1:], p[1:]):
        want = get_device(name).apply(want, params)
    want = materialize_selection(want)
    assert r.result["_scale_idx"] == want["_scale_idx"] == 1
    assert r.result["_scale_px"] == want["_scale_px"]
    assert len(r.result["extrema"]) == len(want["extrema"]) == 1
    for key, value in want["extrema"][0].items():
        np.testing.assert_array_equal(r.result["extrema"][0][key], value)


@pytest.mark.parametrize("steps, ran", [
    ((("pca", {}),), ("pca",)),
    ((("noise", {}), ("pca", {})), ("noise", "pca")),
    ((("noise", {}), ("tucker_HOOI_HOSVD", {})), ("noise", "tucker_HOOI_HOSVD")),
    ((("band_select", {"band": 2}), ("wtmm2d", WTMM)), ("band_select", "wtmm2d")),
    ((("band_select", {"band": 2}), ("noise", {}), ("mz_edges", MZ)),
     ("band_select", "noise", "mz_edges")),
])
def test_multiband_chains_on_a_stack_run_every_step(builtins, steps, ran):
    r = resolve(_layer(*steps), _stack(), Cache())
    assert all(s.applies for s in r.step_states)
    assert r.transforms_run == ran


def test_a_single_band_tool_on_a_stack_does_not_apply(builtins):
    r = resolve(_layer(("wtmm2d", WTMM)), _stack(), Cache())
    assert [s.applies for s in r.step_states] == [False]
    assert r.transforms_run == () and r.analysis_device is None
