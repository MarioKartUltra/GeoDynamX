# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""wavelet_skeleton device: the Tang-You/You-2006 medial-axis extractor as a peer primary
analyzer (the third representation next to wtmm2d and mz_edges — each method its own tool,
the 2026-09-19 design law). Emits the skeleton in the app's per-scale extrema schema so the
existing display path renders it unchanged (the mz_edges precedent)."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import wavelet_skeleton as wsk
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices.wavelet_skeleton import WaveletSkeleton
from dynamix.model.device import defaults_for


def _stripe_field(n=129, d=8):
    f = np.zeros((n, n))
    f[:, n // 2 - d // 2: n // 2 + d // 2] = 1.0
    return RasterField(name="stripe", values=f, frame=LocalFrame(),
                       x_axis=np.arange(n, dtype=np.float64),
                       y_axis=np.arange(n, dtype=np.float64))


def test_registered_as_a_builtin_transform(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import DEVICES, is_transform

    register_builtin_devices()
    assert "wavelet_skeleton" in DEVICES
    assert is_transform(DEVICES["wavelet_skeleton"])


def test_param_surface():
    params = {p.name: p for p in WaveletSkeleton.params}
    assert list(params) == ["s1", "s2", "t_frac", "edge_frac", "n_stages", "input",
                            "wavelet", "pair_dom"]
    assert params["s1"].default == 6.0 and params["s2"].default == 6.0
    assert params["n_stages"].default == 2
    assert params["s1"].units == "px"            # scales-in-pixels law


def test_compute_emits_the_extrema_schema_on_the_axis():
    field = _stripe_field()
    params = dict(defaults_for(WaveletSkeleton()), s1=8.0)
    res = WaveletSkeleton().compute(field, params)
    assert res["chains"] == []
    assert len(res["extrema"]) == 1
    e = res["extrema"][0]
    assert set(e) == {"x", "y", "mod", "arg", "line_id"}
    assert e["x"].dtype == np.int64 and e["line_id"].dtype == np.int64
    assert len(e["x"]) > 0
    assert np.all(np.abs(e["x"] - 64) <= 1)                     # the axis
    assert np.all(e["mod"] >= 0)
    np.testing.assert_array_equal(res["scales"], [8.0])
    assert res["_shape"] == field.values.shape
    # one connected axis: a single non-sentinel line id spanning it
    ids = e["line_id"][e["line_id"] >= 0]
    assert len(ids) > 0 and len(np.unique(ids)) == 1


def test_compute_is_exactly_the_core_pipeline():
    field = _stripe_field()
    params = dict(defaults_for(WaveletSkeleton()), s1=8.0)
    res = WaveletSkeleton().compute(field, params)
    core = wsk.skeletonize(np.asarray(field.values, np.float64), s1=8.0,
                           s2=params["s2"], t_frac=params["t_frac"],
                           edge_frac=params["edge_frac"],
                           n_stages=params["n_stages"])
    ys, xs = np.where(core["skeleton"])
    e = res["extrema"][0]
    np.testing.assert_array_equal(np.sort(e["y"] * 1000 + e["x"]),
                                  np.sort(ys * 1000 + xs))
    np.testing.assert_array_equal(res["skeleton_mask"], core["skeleton"])


def test_mod_arg_are_the_field_wt_values_at_the_skeleton():
    """mod/arg come from the STAGE-1 field transform (physically meaningful), never from
    the mask-stage moduli."""
    field = _stripe_field()
    params = dict(defaults_for(WaveletSkeleton()), s1=8.0)
    res = WaveletSkeleton().compute(field, params)
    core = wsk.skeletonize(np.asarray(field.values, np.float64), s1=8.0,
                           s2=params["s2"], t_frac=params["t_frac"],
                           edge_frac=params["edge_frac"],
                           n_stages=params["n_stages"])
    e = res["extrema"][0]
    np.testing.assert_array_equal(e["mod"], core["mod"][e["y"], e["x"]])
    np.testing.assert_array_equal(e["arg"], core["arg"][e["y"], e["x"]])


def test_refusals_are_the_shared_contract():
    dev = WaveletSkeleton()
    with pytest.raises(ValueError, match="FIRST in its chain"):
        dev.compute({"chains": [], "extrema": []}, defaults_for(dev))
    vals = np.zeros((16, 16, 3))
    field = RasterField(name="v", values=vals, frame=LocalFrame(),
                        x_axis=np.arange(16, dtype=np.float64),
                        y_axis=np.arange(16, dtype=np.float64))
    with pytest.raises(ValueError, match="scalar 2-D"):
        dev.compute(field, defaults_for(dev))


def test_cache_key_tracks_every_param():
    dev = WaveletSkeleton()
    base = defaults_for(dev)
    k0 = dev.cache_key("src", base)
    for change in ({"s1": 8.0}, {"s2": 4.0}, {"t_frac": 0.4},
                   {"edge_frac": 0.2}, {"n_stages": 3}):
        assert dev.cache_key("src", dict(base, **change)) != k0, change
    assert dev.cache_key("other", base) != k0


def test_gradient_input_skeletonizes_a_scarp():
    """A fault scarp is an EDGE -- a single step, no ribbon, invisible to the medial-axis
    detector on the raw signal (measured: 0 points). Under input="gradient" the diffused
    step's ||grad s|| is a ribbon whose axis IS the scarp trace (block-mountain DEM)."""
    from scipy.ndimage import gaussian_filter

    n = 129
    step = np.zeros((n, n))
    step[:, 64:] = 10.0
    scarp = gaussian_filter(step, 3.0)          # a degraded scarp: step * heat kernel
    field = RasterField(name="scarp", values=scarp, frame=LocalFrame(),
                        x_axis=np.arange(n, dtype=np.float64),
                        y_axis=np.arange(n, dtype=np.float64))
    dev = WaveletSkeleton()
    base = dict(defaults_for(dev), s1=8.0)
    raw = dev.compute(field, base)
    assert len(raw["extrema"][0]["x"]) == 0     # the step alone: no ribbon, honest empty
    grad = dev.compute(field, dict(base, input="gradient"))
    xs_ = grad["extrema"][0]["x"]
    assert len(xs_) > 50
    assert np.all(np.abs(xs_ - 63.5) <= 2)      # the trace, on the scarp midline
    assert dev.cache_key("s", base) != dev.cache_key("s", dict(base, input="gradient"))
