# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""View-only params: a knob that only picks WHICH computed output is shown is never part of a
cache key, so switching it is a cache hit -- the decomposition / filter is not run again."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache
from dynamix.engine.resolve import resolve
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import get_device, keyed_params
from dynamix.model.layer import Layer


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _field(bands: int = 0) -> RasterField:
    rng = np.random.default_rng(7)
    shape = (40, 48) if not bands else (40, 48, bands)
    v = rng.standard_normal(shape).cumsum(0).cumsum(1)
    return RasterField(name="f", values=v, frame=LocalFrame(), x_axis=np.arange(48.0),
                       y_axis=np.arange(40.0))


def _layer(device: str, params: dict) -> Layer:
    return Layer(layer_id=1, name="L", source_id="s",
                 chain=Chain((DeviceRef(device, dict(params)),)).materialized())


# (device, view param, first value, second value, field bands, fast params)
_CASES = [
    ("cdf_edges", "show", "edges", "filtered", 0, {"n_levels": 2}),
    ("cdf_edges", "show", "filtered", "edge_channel", 0, {"n_levels": 2}),
    ("pm_edges", "show", "edges", "filtered", 0, {"n_levels": 2}),
    ("tucker_havok", "show", "recon", "residual", 0, {"n_delays": 8, "rank_delay": 2}),
    ("pca", "component", 1, 2, 3, {"n_components": 3}),
]


@pytest.mark.parametrize("device,name,first,second,bands,fast", _CASES)
def test_switching_a_view_param_is_a_cache_hit_and_changes_what_is_shown(
        builtins, device, name, first, second, bands, fast):
    field, cache = _field(bands), Cache()
    a = resolve(_layer(device, {**fast, name: first}), field, cache)
    b = resolve(_layer(device, {**fast, name: second}), field, cache)
    assert a.cache_misses == 1
    assert b.cache_misses == 0 and b.cache_hits == 1          # nothing recomputed
    shown_a, shown_b = a.result.get("raster_out"), b.result.get("raster_out")
    assert (shown_a is None) != (shown_b is None) or not np.array_equal(shown_a, shown_b)
    assert b.result["params"][name] == second                 # the row label reads the view


def test_the_view_picks_the_output_it_names(builtins):
    field, cache = _field(), Cache()
    fast = {"n_levels": 2}
    edges = resolve(_layer("cdf_edges", {**fast, "show": "edges"}), field, cache).result
    filt = resolve(_layer("cdf_edges", {**fast, "show": "filtered"}), field, cache).result
    chan = resolve(_layer("cdf_edges", {**fast, "show": "edge_channel"}), field, cache).result
    assert edges.get("raster_out") is None                    # edges: the raw field stays up
    assert filt["raster_out"] is filt["filtered"]
    assert chan["raster_out"] is chan["edge_channel"]
    tuck = {"n_delays": 8, "rank_delay": 2}
    recon = resolve(_layer("tucker_havok", {**tuck, "show": "recon"}), field, cache).result
    resid = resolve(_layer("tucker_havok", {**tuck, "show": "residual"}), field, cache).result
    np.testing.assert_allclose(recon["raster_out"] + resid["raster_out"], field.values,
                               atol=1e-8)                     # recon + residual = the field


def test_keyed_params_drops_only_view_params(builtins):
    device = get_device("cdf_edges")
    params = {"n_levels": 2, "show": "filtered"}
    assert keyed_params(device, params) == {"n_levels": 2}


def test_every_show_like_param_is_view_only_and_its_device_has_a_view(builtins):
    """A display selector that is keyed recomputes the whole tool on every flip. Every built-in
    "show"/"component" param is view-only, and a device that declares a view param implements
    ``view(result, params)``."""
    from dynamix.model.device import DEVICES

    for name, device in DEVICES.items():
        view = [p.name for p in getattr(device, "params", ()) if p.view]
        for p in getattr(device, "params", ()):
            if p.name in ("show", "component"):
                assert p.view, f"{name}.{p.name} is a display selector but is keyed"
        if view:
            assert callable(getattr(device, "view", None)), f"{name} has view params, no view()"
