# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The declarations tell the truth: after each analyzer, a filter applies exactly when running it
changes the result, and every contract an analyzer declares is in its result.

The filter settings each remove something after ``wtmm2d``, so a filter that changes nothing
elsewhere does so because of the analyzer it follows. A prerequisite step (``chain_topology``
before ``min_vchains``, ``group_paint`` before ``group_filter``) runs only where it applies, as
the engine runs a chain."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from conftest import fbm2d
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.model.chain import DeviceRef
from dynamix.model.contracts import (ALPHA, CONTRACT_KEYS, EXTREMA, FIELD, next_step_state,
                                     produces_of)
from dynamix.model.device import apply_view, get_device, is_transform, validate_params

N = 64
#: a_min 0.5 puts several wtmm2d scales under chain_classify's fit ceiling (min(shape) / 8 px)
#: on a 64-pixel field, so its Hölder fits are finite and the classifier has something to tag.
PRODUCERS = {
    "wtmm2d": {"n_oct": 2, "n_voice": 4, "a_min": 0.5},
    "mz_edges": {"n_levels": 3, "run_mode": "fixed", "iterations": 2},
    "wavelet_skeleton": {},
    "cdf_edges": {"n_levels": 3},
    "pm_edges": {"n_levels": 3},
}
#: Every registered filter but scale_select, which the Transport replaces: on a one-level
#: stack it has no other level to choose.
CONSUMERS = ("orientation_wedge", "modulus_threshold", "hline_length", "hline_modulus",
             "hline_holder", "chain_holder", "chain_modulus", "chain_length", "chain_classify",
             "min_vchains", "group_paint", "group_filter")
GROUP_SPEC = json.dumps({"g": {"signature": "", "chains": [0, 1, 2], "color": [255, 0, 0]}})


def _field() -> RasterField:
    vals = fbm2d(N, 0.5, seed=0)
    vals[:, N // 2:] += 3.0                                        # a step edge
    yy, xx = np.mgrid[0:N, 0:N]
    vals[(yy - 20) ** 2 + (xx - 16) ** 2 <= 6 ** 2] += 5.0         # a bright disc
    return RasterField(name="honesty", values=vals.astype(np.float64),
                       frame=LocalFrame(units="px"), x_axis=np.arange(float(N)),
                       y_axis=np.arange(float(N)))


@pytest.fixture(scope="module")
def results():
    """Each analyzer's result on the test field, computed once: the device classes run directly,
    so no registry is touched outside a test."""
    from dynamix.devices import BUILTIN_DEVICES

    by_name = {cls.name: cls for cls in BUILTIN_DEVICES}
    field = _field()
    out = {}
    for name, params in PRODUCERS.items():
        device = by_name[name]()
        p = validate_params(device, params)
        out[name] = apply_view(device, device.compute(field, p), p)
    return out


@pytest.fixture(scope="module")
def filters(results):
    """consumer -> (params, prerequisite steps); the chain cut-offs are the medians of
    wtmm2d's own chains, so each removes about half of them there."""
    from dynamix.core import chain_stats

    chains = results["wtmm2d"]["chains"]
    holder = chain_stats.stats_for(chains, "ols")
    holder = holder[np.isfinite(holder)]
    peak = chain_stats.max_log2_modulus_for(chains)
    peak = peak[np.isfinite(peak)]
    lengths = chain_stats.length_for(chains)
    return {
        "orientation_wedge": ({"centre": 0.0, "half_width": 30.0}, []),
        "modulus_threshold": ({"frac": 0.3}, []),
        "hline_length": ({"min_len": 5}, []),
        "hline_modulus": ({"frac": 0.3, "mode": "sup"}, []),
        "hline_holder": ({"min_alpha": -0.5, "max_alpha": 0.5, "keep_unfitted": False}, []),
        "chain_holder": ({"cutoff": round(float(np.median(holder)), 3), "estimator": "ols"},
                         []),
        "chain_modulus": ({"threshold": round(float(np.median(peak)), 3)}, []),
        "chain_length": ({"min_len": int(np.median(lengths)) + 1}, []),
        "chain_classify": ({"action": "exclude", "h_step_lo": -3.0, "h_step_hi": 3.0,
                            "straightness_min": 0.0, "fit_a_min": 0.01}, []),
        "min_vchains": ({"min_vchains": 2}, [("chain_topology", {})]),
        "group_paint": ({"spec_json": GROUP_SPEC}, []),
        "group_filter": ({"group": "g", "mode": "keep"},
                         [("group_paint", {"spec_json": GROUP_SPEC})]),
    }


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def _summary(result: dict) -> tuple:
    chains = result.get("chains") or []
    return (
        [int(np.size(level.get("x", ()))) for level in result.get("extrema") or []],
        len(chains),
        sum(1 for c in chains if c.get("tags")),
        len(result.get("chains_excluded") or []),
        int(result.get("_hchains_dropped") or 0),
    )


def _run(name: str, result: dict, params: dict) -> dict:
    device = get_device(name)
    p = validate_params(device, params)
    return device.compute(result, p) if is_transform(device) else device.apply(result, p)


@pytest.mark.parametrize("consumer", CONSUMERS)
@pytest.mark.parametrize("producer", list(PRODUCERS))
def test_a_filter_applies_exactly_where_it_changes_the_result(builtins, results, filters,
                                                              producer, consumer):
    params, pre = filters[consumer]
    steps = [DeviceRef(producer, PRODUCERS[producer])]
    result = copy.deepcopy(results[producer])
    for name, p in pre:
        if next_step_state(steps, FIELD, name, p).applies:
            result = _run(name, result, p)
        steps.append(DeviceRef(name, p))
    state = next_step_state(steps, FIELD, consumer, params)
    before = _summary(result)
    changed = _summary(_run(consumer, result, params)) != before
    assert state.applies == changed, state.reason


@pytest.mark.parametrize("producer", list(PRODUCERS))
def test_every_declared_contract_is_in_the_result(builtins, results, producer):
    device = get_device(producer)
    params = validate_params(device, PRODUCERS[producer])
    result = results[producer]
    produced = produces_of(device, params, frozenset({FIELD}))
    assert EXTREMA in produced, producer
    for contract in produced:
        for key in CONTRACT_KEYS[contract]:
            assert key in result, (producer, contract, key)
        if contract == ALPHA:
            assert all("alpha" in level for level in result["extrema"]), producer
