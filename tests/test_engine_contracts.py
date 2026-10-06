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
                                    resolve_output, selection_recipe, selection_steps,
                                    shown_level)
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


class _DropAll:
    """A filter that needs V-chains and drops every extremum: after a chainless analysis it does
    not apply, so it must reach no selection."""

    name = "drop_all_t"
    params = ()
    consumes = ((EXTREMA, VCHAINS),)

    def apply(self, result, params):
        return {**result, "extrema": [{k: np.asarray(v)[:0] for k, v in e.items()}
                                      for e in result["extrema"]]}


def test_an_inactive_filter_does_not_reach_a_selecting_output(builtins):
    field, cache = _field(), Cache()
    plain = _layer(("mz_edges", MZ), ("hline_length", {"min_len": 4}))
    layer = _layer(("mz_edges", MZ), ("chain_length", {"min_len": 5}),
                   ("hline_length", {"min_len": 4}))
    assert [k for k, _ref in selection_steps(layer.chain.steps, FIELD)] == ["hline_length"]
    assert [k for k, _ref in selection_steps(layer.chain.steps)] == ["chain_length",
                                                                     "hline_length"]
    assert selection_recipe(layer, FIELD) == selection_recipe(plain, FIELD)
    r = resolve(layer, field, cache)
    out = next(o for o in declared_outputs(get_device("mz_edges")) if o.name == "recon_preview")
    resolve_output(layer, field, cache, "recon_preview")
    assert output_key("mz_edges", out, r.analysis_params, r.analysis_key,
                      selection_recipe(plain, FIELD)) in cache


def test_a_skipped_filter_keeps_its_repeat_number(builtins):
    layer = _layer(("mz_edges", MZ), ("hline_modulus", {"mode": "sup_all_scales"}),
                   ("hline_modulus", {"mode": "sup"}))
    assert [k for k, _ref in selection_steps(layer.chain.steps, FIELD)] == ["hline_modulus#2"]


def test_level_keeps_leaves_out_a_filter_that_does_not_apply(builtins):
    register_device(_DropAll())
    field, cache = _field(), Cache()
    layer = _layer(("mz_edges", MZ), ("drop_all_t", {}))
    r = resolve(layer, field, cache)
    raw = cache.get(r.analysis_key)
    dev = get_device("mz_edges")
    for level in range(1, len(raw["extrema"]) + 1):
        assert level_keeps(layer, raw, dev, r.analysis_params, level, FIELD).all()
        assert not level_keeps(layer, raw, dev, r.analysis_params, level).any()


def test_a_level_reduces_the_stack_before_the_filters_with_scale_selects_stamps(builtins):
    field, cache = _field(), Cache()
    threshold = ("modulus_threshold", {"frac": 0.3})
    by_level = resolve(_layer(("wtmm2d", WTMM), threshold), field, cache, level=2)
    by_step = resolve(_layer(("wtmm2d", WTMM), ("scale_select", {"scale_idx": 2}), threshold),
                      field, cache)
    assert by_level.analysis_key == by_step.analysis_key
    assert by_level.filters_run == ("modulus_threshold",)
    assert len(by_level.result["extrema"]) == 1 and by_level.result["_scale_idx"] == 2
    assert by_level.result["_scale_px"] == by_step.result["_scale_px"]
    assert by_level.result["_ext_base"] is by_step.result["_ext_base"]
    assert by_level.result["_ext_base_runs"] is by_step.result["_ext_base_runs"]
    for key, value in by_step.result["extrema"][0].items():
        np.testing.assert_array_equal(by_level.result["extrema"][0][key], value)
    assert len(by_level.result["chains"]) == len(cache.get(by_level.analysis_key)["chains"])


def test_a_level_reduces_a_chain_without_filters_and_spares_a_field(builtins):
    field, cache = _field(), Cache()
    assert len(resolve(_layer(("wtmm2d", WTMM)), field, cache).result["extrema"]) > 1
    r = resolve(_layer(("wtmm2d", WTMM)), field, cache, level=1)
    assert len(r.result["extrema"]) == 1 and r.result["_scale_idx"] == 1
    assert isinstance(resolve(_layer(("noise", {})), field, cache, level=1).result, RasterField)


def test_a_level_overrides_an_old_scale_select_step_and_none_keeps_it(builtins):
    field, cache = _field(), Cache()
    threshold = ("modulus_threshold", {"frac": 0.3})
    old = _layer(("wtmm2d", WTMM), ("scale_select", {"scale_idx": 3}), threshold)
    shown = resolve(old, field, cache, level=1)
    assert [s.applies for s in shown.step_states] == [True, False, True]
    assert shown.step_states[1].reason == "the scale is set by the Transport"
    assert shown.filters_run == ("modulus_threshold",)
    assert len(shown.result["extrema"]) == 1 and shown.result["_scale_idx"] == 1
    at_one = resolve(_layer(("wtmm2d", WTMM), ("scale_select", {"scale_idx": 1}), threshold),
                     field, cache)
    for key, value in at_one.result["extrema"][0].items():
        np.testing.assert_array_equal(shown.result["extrema"][0][key], value)
    today = resolve(old, field, cache)
    assert all(s.applies for s in today.step_states)
    assert today.filters_run == ("scale_select", "modulus_threshold")
    assert len(today.result["extrema"]) == 1 and today.result["_scale_idx"] == 3


def test_shown_level_reads_the_layer_tag():
    layer = Layer(layer_id=1, name="L", source_id="s")
    assert shown_level(layer) is None
    layer.tags["ui.scale_idx"] = "3"
    assert shown_level(layer) == 3
    layer.tags["ui.scale_idx"] = "three"
    assert shown_level(layer) is None


def test_the_preview_skips_a_leading_step_that_does_not_apply(builtins):
    field = _field()
    want = preview_resolve(_layer(("wtmm2d", WTMM)), field)
    got = preview_resolve(_layer(("pca", {}), ("wtmm2d", WTMM)), field)
    assert got is not None
    for key, value in want["extrema"][0].items():
        np.testing.assert_array_equal(got["extrema"][0][key], value)


def test_the_preview_takes_the_level_and_its_single_level_passes_whole(builtins):
    prev = preview_resolve(_layer(("wtmm2d", WTMM)), _field(), level=2)
    assert len(prev["extrema"]) == 1 and "_scale_idx" not in prev


def test_the_preview_with_a_level_skips_an_old_scale_select_step(builtins):
    old = _layer(("wtmm2d", WTMM), ("scale_select", {"scale_idx": 3}))
    assert preview_resolve(old, _field())["_scale_idx"] == 0
    assert "_scale_idx" not in preview_resolve(old, _field(), level=2)
