# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The moment orders of the partition functions are knobs on the WTMM tools: q from q_min to
q_max by q_step (the backend's −3…6 by ½ at the defaults), out of the cache key at their defaults
so a saved chain keeps its results; the tables carry the list that ran."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.core.wtmm_backend import DEFAULT_Q
from dynamix.devices.wtmm import q_list
from dynamix.model.device import defaults_for, get_device, keyed_params, validate_params


def test_the_q_list_follows_the_three_knobs_and_defaults_to_the_backends():
    assert np.array_equal(q_list({"q_min": -3.0, "q_max": 6.0, "q_step": 0.5}), DEFAULT_Q)
    assert np.array_equal(q_list({"q_min": -2.0, "q_max": 2.0, "q_step": 1.0}), [-2, -1, 0, 1, 2])
    assert np.array_equal(q_list({"q_min": 0.0, "q_max": 1.0, "q_step": 0.4}), [0.0, 0.4, 0.8])
    with pytest.raises(ValueError, match="q max"):
        q_list({"q_min": 2.0, "q_max": 2.0, "q_step": 0.5})


@pytest.mark.parametrize("name", ["wtmm2d", "wtmm2d_roi"])
def test_both_wtmm_tools_offer_the_knobs_in_a_moments_group_out_of_the_key_at_default(clean_registry, name):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    dev = get_device(name)
    assert dev.param_groups["MOMENTS"] == ("q_min", "q_max", "q_step")
    p = validate_params(dev, {})
    assert (p["q_min"], p["q_max"], p["q_step"]) == (-3.0, 6.0, 0.5)
    assert not {"q_min", "q_max", "q_step"} & set(keyed_params(dev, p))
    assert "q_step" in keyed_params(dev, validate_params(dev, {"q_step": 1.0}))   # the changed one alone


def test_the_partition_tables_carry_the_q_list_the_knobs_asked_for(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    dev = get_device("wtmm2d")
    rng = np.random.default_rng(3)
    from scipy.ndimage import gaussian_filter

    values = gaussian_filter(rng.standard_normal((64, 64)), 2.0)
    field = RasterField(name="f", values=values, frame=LocalFrame(units="px"),
                        x_axis=np.arange(64.0), y_axis=np.arange(64.0))
    res = dev.compute(field, {**defaults_for(dev), "n_oct": 2, "q_min": -1.0, "q_max": 2.0, "q_step": 1.0})
    assert np.array_equal(res["hd_std"]["q_list"], [-1, 0, 1, 2])
    assert np.array_equal(res["hd_cmax"]["q_list"], [-1, 0, 1, 2])
