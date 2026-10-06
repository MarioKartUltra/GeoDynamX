# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The scale Transport, headless half: under a shown level ``scale_select`` never applies and the
engine reduces the stack before the filters; a saved ``scale_select`` seeds the shown level once;
the built-in racks carry no level step; multiband data and cache identity are untouched."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache
from dynamix.engine.resolve import resolve, shown_level, source_identity
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.contracts import FIELD, StepState, step_states
from dynamix.model.layer import Layer

NY, NX, J = 64, 56, 3
FAST = {"iterations": 2, "run_mode": "fixed"}
#: The reason ``step_states(..., level_view=True)`` gives a ``scale_select`` step.
SCALE_BY_TRANSPORT = "the scale is set by the Transport"


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _field(seed=0):
    values = np.random.default_rng(seed).standard_normal((NY, NX)).cumsum(0).cumsum(1)
    return RasterField(name="f", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(NX)), y_axis=np.arange(float(NY)))


def _layer(filters=()) -> Layer:
    steps = (DeviceRef("mz_edges", {"n_levels": J, **FAST}),)
    steps += tuple(DeviceRef(n, dict(p)) for n, p in filters)
    return Layer(layer_id=1, name="mz", source_id="s", chain=Chain(steps).materialized())


def test_level_view_makes_scale_select_inactive_and_leaves_every_other_step(builtins):
    steps = _layer((("scale_select", {"scale_idx": 1}), ("hline_length", {}))).chain.steps
    plain = step_states(steps, FIELD)
    viewed = step_states(steps, FIELD, level_view=True)
    assert plain[1].applies is True
    assert viewed[1] == StepState(False, SCALE_BY_TRANSPORT)
    assert [viewed[0], viewed[2]] == [plain[0], plain[2]]
    assert viewed[0].applies and viewed[2].applies


def test_level_view_changes_nothing_on_a_chain_without_scale_select(builtins):
    steps = _layer((("hline_length", {}),)).chain.steps
    assert step_states(steps, FIELD, level_view=True) == step_states(steps, FIELD)


def test_a_level_reduces_the_stack_before_the_filters_and_skips_scale_select(builtins):
    field, cache = _field(), Cache()
    layer = _layer((("scale_select", {"scale_idx": 0}), ("hline_length", {"min_len": 1})))
    r = resolve(layer, field, cache, level=2)
    raw = cache.get(r.analysis_key)
    assert r.result["_scale_idx"] == 2                       # never clipped back to 0
    assert len(r.result["extrema"]) == 1
    assert r.result["_scale_px"] == pytest.approx(float(np.asarray(raw["scales"])[2]))
    assert "scale_select" not in r.filters_run and "hline_length" in r.filters_run
    assert r.step_states[1] == StepState(False, SCALE_BY_TRANSPORT)
    assert r.step_states[2].applies
    # no level: the saved step applies as it always did
    old = resolve(layer, field, cache)
    assert old.cache_misses == 0
    assert old.step_states[1].applies and "scale_select" in old.filters_run
    assert old.result["_scale_idx"] == 0


def test_two_viewers_are_two_resolves_over_one_cached_analysis(builtins):
    field, cache = _field(), Cache()
    layer = _layer((("hline_length", {"min_len": 1}),))
    a = resolve(layer, field, cache, level=0)
    b = resolve(layer, field, cache, level=2)
    assert b.cache_misses == 0 and a.analysis_key == b.analysis_key
    assert (a.result["_scale_idx"], b.result["_scale_idx"]) == (0, 2)
    assert resolve(layer, field, cache, level=99).result["_scale_idx"] == J - 1   # clamped


def test_a_saved_scale_select_seeds_the_shown_level_once(builtins):
    from dynamix.engine.resolve import seed_shown_level

    layer = _layer((("scale_select", {"scale_idx": 2}),))
    chain = layer.chain
    assert shown_level(layer) is None
    assert seed_shown_level(layer) is True
    assert layer.tags["ui.scale_idx"] == "2" and shown_level(layer) == 2
    assert layer.chain is chain                              # the chain is not edited
    layer.tags["ui.scale_idx"] = "1"                         # the Transport moved since
    assert seed_shown_level(layer) is False
    assert layer.tags["ui.scale_idx"] == "1"
    plain = _layer()
    assert seed_shown_level(plain) is False and "ui.scale_idx" not in plain.tags


def test_a_level_leaves_stacks_band_rows_and_the_source_identity_alone(builtins):
    """Multiband data is untouched by the shown level: a decomposition of a stack has no
    extrema for a level to reduce, a band row scrubs its own analyzer's levels, and the level
    never joins the source identity a cache key is built on (``data.bands`` still does)."""
    values = np.random.default_rng(1).standard_normal((NY, NX, 3)).cumsum(0).cumsum(1)
    stack = RasterField(name="stack", values=values, frame=LocalFrame(units="px"),
                        x_axis=np.arange(float(NX)), y_axis=np.arange(float(NY)))
    pca = Layer(layer_id=2, name="pca", source_id="stack",
                chain=Chain((DeviceRef("pca", {"n_components": 3}),)).materialized(),
                tags={"data.bands": "1,2,3"})
    cache = Cache()
    whole = resolve(pca, stack, cache)
    at_level = resolve(pca, stack, cache, level=2)
    assert at_level.cache_misses == 0
    np.testing.assert_array_equal(at_level.result["raster_out"], whole.result["raster_out"])
    assert at_level.result["extrema"] == [] and "_scale_idx" not in at_level.result
    assert at_level.step_states == whole.step_states
    shown = dataclasses.replace(pca, tags={**pca.tags, "ui.scale_idx": "2"})
    assert source_identity(shown) == source_identity(pca) == "stack|bands:1,2,3"
    band = Layer(layer_id=3, name="band 2", source_id="stack",
                 chain=Chain((DeviceRef("band_select", {"band": 2}),
                              DeviceRef("mz_edges", {"n_levels": J, **FAST}),
                              DeviceRef("hline_length", {"min_len": 1}))).materialized())
    r = resolve(band, stack, cache, level=1)
    assert r.result["_scale_idx"] == 1 and len(r.result["extrema"]) == 1
    assert r.transforms_run == ("band_select", "mz_edges")


def test_the_built_in_racks_hold_no_scale_select_and_validate(builtins):
    from dynamix.model.presets import PRESETS

    for name, steps in PRESETS.items():
        assert "scale_select" not in [device for device, _ in steps], name
        Chain(tuple(DeviceRef(d, dict(p)) for d, p in steps)).materialized().validate()


def test_the_m_z_rack_shows_one_level_through_the_transport(builtins):
    from dynamix.model.presets import PRESETS

    steps = tuple(DeviceRef(d, {**p, **FAST} if d == "mz_edges" else dict(p))
                  for d, p in PRESETS["M–Z edges"])
    layer = Layer(layer_id=1, name="mz", source_id="s", chain=Chain(steps).materialized())
    assert [s.device for s in layer.chain.steps] == ["mz_edges", "hline_length"]
    r = resolve(layer, _field(), Cache(), level=1)
    assert r.result["_scale_idx"] == 1 and len(r.result["extrema"]) == 1
