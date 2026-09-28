# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The rack's filters reach the M–Z reconstruction: the engine runs them on each level with that
level's settings, hands the kept extrema to the selecting outputs and keys those outputs on the
filters; the display draws the constraints of the level on show; ``hline_holder`` selects by α."""
from __future__ import annotations

import json

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache
from dynamix.engine.resolve import (level_keeps, output_key, resolve, resolve_output,
                                    selection_recipe)
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import declared_outputs, get_device, validate_params
from dynamix.model.layer import Layer

NY, NX, J = 64, 56, 3
FAST = {"iterations": 2, "run_mode": "fixed"}


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _field(seed=0):
    values = np.random.default_rng(seed).standard_normal((NY, NX)).cumsum(0).cumsum(1)
    return RasterField(name="f", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(NX)), y_axis=np.arange(float(NY)))


def _layer(filters=(), **params) -> Layer:
    steps = (DeviceRef("mz_edges", {"n_levels": J, **FAST, **params}),)
    steps += tuple(DeviceRef(n, dict(p)) for n, p in filters)
    return Layer(layer_id=1, name="mz", source_id="s", chain=Chain(steps).materialized())


def _output(name):
    return next(o for o in declared_outputs(get_device("mz_edges")) if o.name == name)


def _levels(**states):
    return json.dumps({"levels": {k[1:]: {"state": v[0], "source": v[1]}
                                  for k, v in states.items()}})


def test_the_analysis_stores_an_alpha_per_maximum(builtins):
    field, cache = _field(), Cache()
    r = resolve(_layer(), field, cache)
    raw = cache.get(r.analysis_key)
    for e in raw["extrema"]:
        assert e["alpha"].shape == e["x"].shape


def test_level_keeps_is_the_filters_applied_to_that_level_alone(builtins):
    field, cache = _field(), Cache()
    layer = _layer((("scale_select", {"scale_idx": 0}), ("hline_length", {"min_len": 6})))
    r = resolve(layer, field, cache)
    raw = cache.get(r.analysis_key)
    dev = get_device("mz_edges")
    filt = get_device("hline_length")
    for l in range(1, J + 1):
        keep = level_keeps(layer, raw, dev, r.analysis_params, l)
        base = raw["extrema"][l - 1]
        kept = filt.apply({"extrema": [base]}, validate_params(filt, {"min_len": 6}))
        want = np.isin(base["y"] * NX + base["x"],
                       kept["extrema"][0]["y"] * NX + kept["extrema"][0]["x"])
        assert np.array_equal(keep, want) and 0 < keep.sum() < keep.size


def test_the_recon_key_moves_with_the_filters_and_the_coarse_key_does_not(builtins):
    field, cache = _field(), Cache()
    a = _layer((("hline_length", {"min_len": 2}),))
    b = _layer((("hline_length", {"min_len": 9}),))
    ra, rb = resolve(a, field, cache), resolve(b, field, cache)
    for name, differ in (("recon", True), ("recon_preview", True), ("residual", True),
                         ("coarse", False)):
        ka = output_key("mz_edges", _output(name), ra.analysis_params, ra.analysis_key,
                        selection_recipe(a))
        kb = output_key("mz_edges", _output(name), rb.analysis_params, rb.analysis_key,
                        selection_recipe(b))
        assert (ka != kb) is differ


@pytest.mark.parametrize("border", ["mirror", "periodic"])
def test_pass_all_filters_reconstruct_as_the_unfiltered_chain(builtins, border):
    field = _field()
    plain = resolve_output(_layer(border=border), field, Cache(), "recon")
    passing = _layer((("scale_select", {"scale_idx": 1}), ("hline_length", {"min_len": 1}),
                      ("modulus_threshold", {"frac": 0.0})), border=border)
    np.testing.assert_array_equal(resolve_output(passing, field, Cache(), "recon")["raster"],
                                  plain["raster"])


def test_removed_lines_change_the_recon_unless_every_level_reads_all(builtins):
    field = _field()
    plain = resolve_output(_layer(), field, Cache(), "recon")["raster"]
    cut = (("hline_length", {"min_len": 12}),)
    fewer = resolve_output(_layer(cut), field, Cache(), "recon")["raster"]
    assert not np.array_equal(fewer, plain)
    every = _levels(l1=("all", 2), l2=("all", 2), l3=("all", 2))
    np.testing.assert_array_equal(
        resolve_output(_layer(cut, recon_levels=every), field, Cache(), "recon")["raster"], plain)


def test_per_level_settings_apply_at_their_own_level(builtins):
    field, cache = _field(), Cache()
    text = json.dumps({"filters": {"2": {"hline_length": {"min_len": 40}}}})
    layer = _layer((("hline_length", {"min_len": 1}),), per_level=True, recon_levels=text)
    r = resolve(layer, field, cache)
    raw = cache.get(r.analysis_key)
    dev = get_device("mz_edges")
    assert level_keeps(layer, raw, dev, r.analysis_params, 1).all()
    assert not level_keeps(layer, raw, dev, r.analysis_params, 2).all()
    off = _layer((("hline_length", {"min_len": 1}),), per_level=False, recon_levels=text)
    assert level_keeps(off, raw, dev, resolve(off, field, cache).analysis_params, 2).all()


def test_the_display_draws_what_the_level_on_show_contributes(builtins):
    field, cache = _field(), Cache()
    rack = (("scale_select", {"scale_idx": 0}),)
    own = resolve(_layer(rack), field, cache).result
    raw = cache.get(resolve(_layer(rack), field, cache).analysis_key)
    assert np.array_equal(own["extrema"][0]["x"], raw["extrema"][0]["x"])
    assert own["_recon_reading"].startswith("level 1 (2 px): own")
    coder = resolve(_layer(rack, recon_levels=_levels(l1=("coder", 2))), field, cache).result
    assert np.array_equal(coder["extrema"][0]["x"], raw["extrema"][1]["x"])
    assert "coder from level 2" in coder["_recon_reading"]
    opened = resolve(_layer(rack, recon_levels=_levels(l1=("open", 2))), field, cache).result
    assert opened["extrema"][0]["x"].size == 0


def test_a_coder_and_an_open_level_each_change_the_recon(builtins):
    field = _field()
    plain = resolve_output(_layer(), field, Cache(), "recon")["raster"]
    for states in ({"l1": ("coder", 2)}, {"l1": ("open", 2)}, {"l1": ("predict", 2)},
                   {"l1": ("near", 2)}):
        img = resolve_output(_layer(recon_levels=_levels(**states)), field, Cache(),
                             "recon")["raster"]
        assert np.isfinite(img).all() and not np.array_equal(img, plain)


def test_hline_holder_keeps_lines_by_alpha(builtins):
    filt = get_device("hline_holder")
    layer = {"x": np.arange(4), "y": np.zeros(4, int), "mod": np.ones(4),
             "alpha": np.array([0.0, -1.0, np.nan, -2.0]), "line_id": np.full(4, -1)}
    out = filt.apply({"extrema": [layer]}, validate_params(filt, {"min_alpha": -1.5,
                                                                "max_alpha": 0.5}))
    assert out["extrema"][0]["x"].tolist() == [0, 1, 2]
    out = filt.apply({"extrema": [layer]}, validate_params(filt, {"min_alpha": -1.5,
                                                                "max_alpha": 0.5,
                                                                "keep_unfitted": False}))
    assert out["extrema"][0]["x"].tolist() == [0, 1]
    bare = {k: v for k, v in layer.items() if k != "alpha"}
    out = filt.apply({"extrema": [bare]}, validate_params(filt, {"max_alpha": 0.0}))
    assert out["extrema"][0]["x"].size == 4
