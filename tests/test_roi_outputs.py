# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Lazy outputs on ROI results: computed on the region runner's own processing window (the same
read, summed margin, reflected edges and field stages) by the tool's ``compute`` and
``compute_output`` there, then cut to the region -- the runner's own path for a tool without
``compute_roi``. The analysis it decorates is untouched."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.engine.cache import Cache
from dynamix.engine.resolve import resolve, resolve_output
from dynamix.model.chain import Chain, DeviceRef
from dynamix.model.device import defaults_for, get_device, validate_params
from dynamix.model.layer import Layer

ROI = (40, 44, 32, 32)
MZ = {"n_levels": 3}


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _field(ny=128, nx=128, seed=3):
    v = np.random.default_rng(seed).standard_normal((ny, nx)).cumsum(0).cumsum(1)
    return RasterField(name="f", values=v, frame=LocalFrame(), x_axis=np.arange(float(nx)),
                       y_axis=np.arange(float(ny)))


def _layer(device="mz_edges", roi=ROI, **params):
    p = {**defaults_for(get_device(device)), **(MZ if device == "mz_edges" else {}), **params}
    return Layer(layer_id=1, name="L", source_id="s",
                 chain=Chain((DeviceRef(device, p),)).materialized(),
                 tags={"roi.window": ",".join(str(v) for v in roi)})


def _window_reference(field, layer, name):
    """The output by hand: the runner's window, the tool's compute there, its compute_output
    (fetching other outputs the same way), cut to the region."""
    from dynamix.roi.runner import read_processing_window, roi_margin_of

    ref = layer.chain.steps[0]
    dev = get_device(ref.device)
    p = validate_params(dev, ref.params)
    m = roi_margin_of(dev, p)[0]
    r, c, h, w = ROI
    window, _info = read_processing_window(field, (r, c, h, w), m)
    raw = dev.compute(window, p)
    memo = {}

    def fetch(other):
        if other not in memo:
            memo[other] = dev.compute_output(other, np.asarray(window.values), raw, p,
                                             fetch=fetch)
        return memo[other]

    full = fetch(name)
    return full["raster"][m:m + h, m:m + w], window, m


@pytest.mark.parametrize("name", ["recon", "coarse"])
def test_an_roi_output_is_the_window_output_cut_to_the_region(builtins, name):
    field, layer = _field(), _layer()
    out = resolve_output(layer, field, Cache(), name, source_id="s")
    want, window, m = _window_reference(field, layer, name)
    np.testing.assert_array_equal(np.asarray(out["raster"]), want)
    r, c, h, w = ROI
    np.testing.assert_array_equal(out["roi_axes"][0], np.asarray(window.x_axis)[m:m + w])
    np.testing.assert_array_equal(out["roi_axes"][1], np.asarray(window.y_axis)[m:m + h])


def test_the_output_is_placed_where_the_roi_result_is(builtins):
    field, layer, cache = _field(), _layer(), Cache()
    result = resolve(layer, field, cache, source_id="s").result
    out = resolve_output(layer, field, cache, "recon", source_id="s")
    np.testing.assert_array_equal(out["roi_axes"][0], result["_roi_axes"][0])
    np.testing.assert_array_equal(out["roi_axes"][1], result["_roi_axes"][1])
    assert np.shape(out["raster"]) == tuple(result["_shape"])


def test_computing_an_output_leaves_the_roi_analysis_alone(builtins):
    field, layer, cache = _field(), _layer(), Cache()
    before = resolve(layer, field, cache, source_id="s")
    kept = cache.get(before.analysis_key)
    resolve_output(layer, field, cache, "recon", source_id="s")
    after = resolve(layer, field, cache, source_id="s")
    assert after.analysis_key == before.analysis_key
    assert cache.get(after.analysis_key) is kept


def test_two_regions_of_one_dataset_never_share_an_output(builtins):
    field, cache = _field(), Cache()
    a = resolve_output(_layer(roi=(40, 44, 32, 32)), field, cache, "coarse", source_id="s")
    b = resolve_output(_layer(roi=(60, 20, 32, 32)), field, cache, "coarse", source_id="s")
    assert not np.array_equal(np.asarray(a["raster"]), np.asarray(b["raster"]))


def test_a_tool_with_its_own_region_path_is_refused_with_the_reason(builtins):
    layer = _layer("wtmm2d")
    with pytest.raises(ValueError, match="wtmm2d.*own region path"):
        resolve_output(layer, _field(), Cache(), "modulus", source_id="s")


def test_an_output_on_its_own_coarser_grid_is_refused_on_a_region(builtins):
    with pytest.raises(ValueError, match="coarser grid"):
        resolve_output(_layer(), _field(), Cache(), "thumbnail", source_id="s")
