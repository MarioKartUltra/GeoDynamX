# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Chaining is pluggable: the benchmark chainer is today's chains2d, the WTMM device's ``chainer``
param names one, and at its default it leaves every existing cache key as it was."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.chaining import CHAINERS, DEFAULT, XsmurfBench, chainer, link
from dynamix.core.wtmm_backend import compute_scales2d, get_backend, run_wtmm2d
from dynamix.devices import register_builtin_devices
from dynamix.engine.cache import cache_key
from dynamix.model.device import get_device, keyed_params, validate_params
from dynamix.model.param import Param, ParamKind
from tests.conftest import fbm2d

def _field(values):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    return RasterField(name="f", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(values.shape[1])),
                       y_axis=np.arange(float(values.shape[0])))


#: ``cache_key("wtmm2d", "s", keyed_params(...))`` at the defaults before the ``chainer`` param
#: existed (main 9e222f7), and the same for the ROI device. The whole-raster key is the one its
#: boundary defaults gave it (spec 25: measured-reach padding and the Border cut, keyed at their
#: defaults so a result padded by the old 32 px is never served for them).
KEY_BEFORE = {"wtmm2d": "408e7f7288743cb072ae0a600a0af296",
              "wtmm2d_roi": "20cbab1b24b2d32580f53085bf1854a2"}


def test_the_benchmark_is_the_default_and_an_unknown_name_is_refused():
    assert DEFAULT == "xsmurf_bench" and isinstance(CHAINERS[DEFAULT], XsmurfBench)
    with pytest.raises(KeyError, match="xsmurf_bench"):
        chainer("nope")


def test_the_benchmark_chainer_links_as_chains2d_does():
    backend = get_backend("python")
    scales = compute_scales2d(2, 2)
    cwt = backend.cwt2d(fbm2d(64, 0.6, seed=3), scales)
    extrema = backend.extrema2d(cwt, scales, thresh=1e-3)
    params = {"similitude": 0.8, "box_ratio": 1.0, "dist2_max": 50.0, "min_chain_len": 2,
              "smooth": True}
    expected = backend.chains2d(extrema, scales, similitude=0.8, box_ratio=1.0, dist2_max=50.0,
                                min_len=2, smooth=True)
    got = link(extrema, scales, params, backend)
    assert len(got) == len(expected) > 0
    for a, b in zip(got, expected):
        assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["mod"], b["mod"])


def test_a_param_not_keyed_at_its_default_leaves_the_key_and_enters_it_when_set():
    class D:
        name = "d"
        params = (Param("gain", ParamKind.INT, default=1, min=0, max=9),
                  Param("way", ParamKind.CHOICE, default="a", choices=("a", "b"),
                        keyed_at_default=False))

    assert keyed_params(D(), {"gain": 1, "way": "a"}) == {"gain": 1}
    assert keyed_params(D(), {"gain": 1, "way": "b"}) == {"gain": 1, "way": "b"}


@pytest.mark.parametrize("name", sorted(KEY_BEFORE))
def test_the_wtmm_keys_at_the_defaults_are_what_they_were(clean_registry, name):
    register_builtin_devices()
    device = get_device(name)
    params = validate_params(device, {})
    assert params["chainer"] == DEFAULT
    assert cache_key(name, "s", keyed_params(device, params)) == KEY_BEFORE[name]
    other = cache_key(name, "s", keyed_params(device, {**params, "chainer": "other"}))
    assert other != KEY_BEFORE[name]


def test_run_wtmm2d_takes_the_chainer_and_refuses_an_unknown_one():
    field = _field(fbm2d(64, 0.6, seed=3))
    res = run_wtmm2d(field, {"n_oct": 2, "n_voice": 2, "chainer": DEFAULT})
    assert res["params"]["chainer"] == DEFAULT and len(res["chains"]) > 0
    with pytest.raises(KeyError, match="nope"):
        run_wtmm2d(field, {"n_oct": 2, "n_voice": 2, "chainer": "nope"})


@pytest.mark.parametrize("name", ["gradient_arg_horizontal", "vertical_experimental"])
def test_the_other_chainers_return_the_chain_schema_anchored_at_the_finest_scale(name):
    backend = get_backend("python")
    scales = compute_scales2d(2, 2)
    cwt = backend.cwt2d(fbm2d(64, 0.6, seed=3), scales)
    extrema = backend.extrema2d(cwt, scales, thresh=1e-3)
    params = {"similitude": 0.8, "box_ratio": 1.0, "dist2_max": 50.0, "min_chain_len": 2,
              "smooth": True, "chainer": name}
    chains = link(extrema, scales, params, backend)
    assert chains
    for c in chains:
        assert set(c) == {"x", "y", "mod", "log2_mod", "log2_scales"}
        k = len(c["mod"])
        assert 2 <= k <= len(scales) and len(c["x"]) == len(c["y"]) == len(c["log2_scales"]) == k
        assert c["x"].dtype == np.int64 and np.array_equal(c["log2_scales"], np.log2(scales[:k]))
    finest = {(int(x), int(y)) for x, y in zip(*[np.asarray(e) for e in (
        backend.single_maxima2d(extrema)[0]["x"], backend.single_maxima2d(extrema)[0]["y"])])}
    assert all((int(c["x"][0]), int(c["y"][0])) in finest for c in chains)


def test_the_unbanded_chainer_links_at_least_as_many_points_as_the_banded_one():
    backend = get_backend("python")
    scales = compute_scales2d(2, 2)
    cwt = backend.cwt2d(fbm2d(64, 0.6, seed=3), scales)
    extrema = backend.extrema2d(cwt, scales, thresh=1e-3)
    params = {"similitude": 0.8, "box_ratio": 1.0, "dist2_max": 50.0, "min_chain_len": 2,
              "smooth": True}
    banded = link(extrema, scales, {**params, "chainer": "gradient_arg_horizontal"}, backend)
    free = link(extrema, scales, {**params, "chainer": "vertical_experimental"}, backend)
    assert sum(len(c["mod"]) for c in free) >= sum(len(c["mod"]) for c in banded)
