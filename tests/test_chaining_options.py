# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Spec 14's chaining options on the benchmark chainer (xsmurf's vchain port).

Every option is off by default and then reproduces the benchmark exactly; each is a parameter of
wtmm2d and wtmm2d_roi that stays out of the key at its default; on, each changes the links the way
its rule says: the angle test, the scale-proportional search radius, the modulus-closer
tie-break and the border policy at the cone of influence.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.chaining import OPTION_DEFAULTS, link
from dynamix.core.wtmm_backend import (_WTMM2D_PARAM_DEFAULTS, compute_scales2d, get_backend,
                                       run_wtmm2d)
from dynamix.devices import register_builtin_devices
from dynamix.engine.cache import cache_key
from dynamix.model.device import get_device, keyed_params, validate_params
from tests.conftest import fbm2d


def _stack(n=160, seed=5):
    backend = get_backend("python")
    values = fbm2d(n, 0.5, seed=seed)
    scales = compute_scales2d(2, 4)
    cwt = backend.cwt2d(values, scales)
    return backend, backend.extrema2d(cwt, scales), scales, values.shape


def _params(**over):
    return {**_WTMM2D_PARAM_DEFAULTS, **OPTION_DEFAULTS, **over}


def _same(a, b) -> bool:
    return len(a) == len(b) and all(np.array_equal(x["x"], y["x"]) and np.array_equal(x["y"], y["y"])
                                    and np.allclose(x["mod"], y["mod"]) for x, y in zip(a, b))


def test_the_defaults_reproduce_the_benchmark_exactly():
    backend, ext, scales, shape = _stack()
    bare = backend.chains2d(ext, scales)
    assert _same(link(ext, scales, _params(), backend, shape=shape), bare)


def test_the_angle_test_only_removes_links_and_a_zero_threshold_removes_none():
    backend, ext, scales, shape = _stack()
    base = link(ext, scales, _params(), backend, shape=shape)
    loose = link(ext, scales, _params(angle_test=True, angle_threshold=0.0), backend, shape=shape)
    strict = link(ext, scales, _params(angle_test=True, angle_threshold=0.95), backend,
                  shape=shape)
    assert _same(loose, base)
    span = lambda cs: sum(len(c["x"]) for c in cs)                       # noqa: E731
    assert span(strict) < span(base)


def test_a_scale_proportional_radius_grows_with_its_constant():
    backend, ext, scales, shape = _stack()
    span = lambda cs: sum(len(c["x"]) for c in cs)                       # noqa: E731
    small = link(ext, scales, _params(radius_rule="scale_proportional", radius_c=0.2), backend,
                 shape=shape)
    large = link(ext, scales, _params(radius_rule="scale_proportional", radius_c=8.0), backend,
                 shape=shape)
    assert span(small) < span(large)


def test_the_modulus_closer_tie_break_is_deterministic_and_its_own_rule():
    backend, ext, scales, shape = _stack()
    a = link(ext, scales, _params(tie_break="modulus"), backend, shape=shape)
    b = link(ext, scales, _params(tie_break="modulus"), backend, shape=shape)
    assert _same(a, b)
    assert len(a) > 0


@pytest.mark.parametrize("policy", ["coi_trim", "coi_drop"])
def test_the_border_policy_keeps_every_chain_point_out_of_the_cone(policy):
    backend, ext, scales, shape = _stack()
    ny, nx = shape
    base = link(ext, scales, _params(), backend, shape=shape)
    out = link(ext, scales, _params(border_policy=policy, coi_k=2.0), backend, shape=shape)
    assert 0 < len(out) <= len(base)
    for c in out:
        a_px = 2.0 ** np.asarray(c["log2_scales"])               # normalized scales are pixels
        x, y = np.asarray(c["x"]), np.asarray(c["y"])
        edge = np.minimum.reduce([x, y, nx - 1 - x, ny - 1 - y])
        assert np.all(edge >= 2.0 * a_px)
    if policy == "coi_drop":
        kept = {(int(c["x"][0]), int(c["y"][0])) for c in out}
        assert kept <= {(int(c["x"][0]), int(c["y"][0])) for c in base}


def test_the_options_are_wtmm_params_out_of_the_key_at_their_defaults(clean_registry):
    register_builtin_devices()
    for name in ("wtmm2d", "wtmm2d_roi"):
        device = get_device(name)
        declared = {p.name: p for p in device.params}
        for key, default in OPTION_DEFAULTS.items():
            assert declared[key].default == default and not declared[key].keyed_at_default, key
            superseded = name == "wtmm2d" and key in ("border_policy", "coi_k")  # the Border cut
            assert declared[key].level == ("hidden" if superseded else "advanced"), key
        params = validate_params(device, {})
        assert not set(OPTION_DEFAULTS) & set(keyed_params(device, params))
        on = keyed_params(device, {**params, "angle_test": True})
        assert cache_key(name, "s", on) != cache_key(name, "s", keyed_params(device, params))


def test_the_run_takes_the_options_and_keys_its_chains_by_them():
    values = fbm2d(64, 0.5, seed=2)
    from tests.test_chaining import _field

    base = run_wtmm2d(_field(values), {"n_oct": 2, "n_voice": 4})
    trimmed = run_wtmm2d(_field(values), {"n_oct": 2, "n_voice": 4, "border_policy": "coi_drop"})
    assert len(trimmed["chains"]) < len(base["chains"])
    assert trimmed["params"]["border_policy"] == "coi_drop"
