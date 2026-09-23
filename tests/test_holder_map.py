# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for the holder_map Transform (devices/holder_map.py) -- the per-pixel Hölder-exponent
raster device over dynamix.core.microcanonical's estimators."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import fbm2d
from dynamix.core import microcanonical as mc
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices.holder_map import HolderMap
from dynamix.model.device import defaults_for


def _field(n=64, H=0.5, seed=0):
    vals = fbm2d(n, H, seed=seed)
    return RasterField(name=f"f{n}", values=vals, frame=LocalFrame(),
                       x_axis=np.arange(n, dtype=np.float64),
                       y_axis=np.arange(n, dtype=np.float64))


def test_registered_as_a_builtin(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import DEVICES

    register_builtin_devices()
    assert "holder_map" in DEVICES


def test_compute_is_exactly_the_core_multiaffine_pipeline():
    """The device is a thin wrapper: its h_map equals the direct core call at the same
    (r_min, kappa, n_scales), and the result carries the display contract keys."""
    field = _field()
    params = defaults_for(HolderMap())
    res = HolderMap().compute(field, params)

    scales = np.geomspace(params["r_min"], params["r_min"] * params["kappa"],
                          params["n_scales"])
    T = mc.ricker_projections(np.asarray(field.values, dtype=np.float64), scales)
    h, r2 = mc.singularity_map_regression(T, scales, r2_min=0.0)
    np.testing.assert_array_equal(res["h_map"], h)
    np.testing.assert_array_equal(res["r2_map"], r2)
    assert res["h_map"].shape == field.values.shape
    assert res["chains"] == [] and res["extrema"] == []
    assert res["_shape"] == field.values.shape
    assert res["params"]["estimator"] == "multiaffine"


def test_measure_estimator_switches_route():
    field = _field()
    params = dict(defaults_for(HolderMap()), estimator="measure", r_min=2.0)
    res = HolderMap().compute(field, params)
    scales = np.geomspace(2.0, 2.0 * params["kappa"], params["n_scales"])
    T = mc.measure_projections(mc.gradient_measure(np.asarray(field.values, np.float64)),
                               scales)
    h, _ = mc.singularity_map_regression(T, scales, r2_min=0.0)
    np.testing.assert_array_equal(res["h_map"], h)


def test_refuses_an_upstream_result_dict_with_a_pointed_message():
    """A Transform placed after another receives that transform's RESULT (the chain_topology
    contract) -- holder_map analyses the field and says so instead of mis-analysing."""
    with pytest.raises(ValueError, match="FIRST in its chain"):
        HolderMap().compute({"chains": [], "extrema": []}, defaults_for(HolderMap()))


def test_refuses_non_scalar_fields():
    vals = np.zeros((16, 16, 3))
    field = RasterField(name="v", values=vals, frame=LocalFrame(),
                        x_axis=np.arange(16, dtype=np.float64),
                        y_axis=np.arange(16, dtype=np.float64))
    with pytest.raises(ValueError, match="scalar 2-D"):
        HolderMap().compute(field, defaults_for(HolderMap()))


def test_cache_key_tracks_every_param():
    dev = HolderMap()
    base = defaults_for(dev)
    k0 = dev.cache_key("src", base)
    assert dev.cache_key("src", base) == k0
    for change in ({"estimator": "measure"}, {"r_min": 2.0}, {"kappa": 12.0},
                   {"n_scales": 12}, {"wavelet": "lorentzian"}, {"q_tsallis": 2.0},
                   {"beta": 2.0}):
        assert dev.cache_key("src", dict(base, **change)) != k0, change
    assert dev.cache_key("other", base) != k0


def test_wavelet_param_reaches_both_estimators():
    """The envelope family rides params into whichever route is active."""
    field = _field()
    base = defaults_for(HolderMap())
    lor = HolderMap().compute(field, dict(base, wavelet="lorentzian", beta=2.0))
    scales = np.geomspace(base["r_min"], base["r_min"] * base["kappa"], base["n_scales"])
    T = mc.ricker_projections(np.asarray(field.values, np.float64), scales,
                              wavelet="lorentzian", beta=2.0, q_tsallis=base["q_tsallis"])
    h, _ = mc.singularity_map_regression(T, scales, r2_min=0.0)
    np.testing.assert_array_equal(lor["h_map"], h)
    gau = HolderMap().compute(field, base)
    assert not np.array_equal(lor["h_map"], gau["h_map"])
