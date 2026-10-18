# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The nms detector's tie rule. xsmurf's NMS path (``wtmm2d`` under imStudy's ``useNMaxSup 1``:
``Extract_Gradient_Maxima_2D`` -> ``Remove_Gradient_NonMaxima_Slice_2D``, edge/extrema_core.c)
rejects a pixel when its modulus is <= the neighbour along the gradient and when it is < the one
against it, so exactly one pixel of a tied pair survives; within ~1.8 deg of an axis
(|g|/norm > 0.9995) it reads the nearest pixel instead of interpolating. ``keep_both`` (>= on both
sides) was the detector before, and keeps a tied pair whole; ``strict`` keeps neither."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.wtmm_backend import _nms_extrema_scale

GOLDEN = "tests/golden/nms_keep_both.npz"


def _points(e) -> set:
    return set(zip(np.asarray(e["x"]).tolist(), np.asarray(e["y"]).tolist()))


def test_keep_both_is_the_detector_before_bit_for_bit():
    g = np.load(GOLDEN)
    for i in range(4):
        e = _nms_extrema_scale(g[f"s{i}_modin"], g[f"s{i}_argin"], thresh=1e-3,
                               tie_rule="keep_both")
        for k in ("x", "y", "mod", "arg", "line_id"):
            np.testing.assert_array_equal(e[k], g[f"s{i}_{k}"], err_msg=f"scale {i} {k}")


def _plateau():
    """A ridge along y whose crest is two equal columns (10 and 11), the gradient along +x."""
    x = np.arange(24.0)
    profile = np.exp(-0.5 * ((x - 10.5) / 3.0) ** 2)
    profile[11] = profile[10]                        # an exact tie, by construction
    mod = np.tile(profile, (20, 1))
    return mod, np.zeros_like(mod)


@pytest.mark.parametrize("rule, columns", [("asymmetric", {11}), ("keep_both", {10, 11}),
                                           ("strict", set())])
def test_a_two_pixel_plateau_keeps_one_column_both_or_neither(rule, columns):
    mod, arg = _plateau()
    e = _nms_extrema_scale(mod, arg, thresh=0.0, tie_rule=rule)
    assert set(np.asarray(e["x"]).tolist()) == columns


def test_near_an_axis_the_neighbour_is_the_nearest_pixel():
    """At 0.01 rad the gradient is within xsmurf's 0.9995 of the x axis: the forward neighbour is
    the pixel to the right, a tie, so the pixel goes; interpolated it would read 0.99 and stay."""
    mod = np.zeros((7, 7))
    mod[3, 3] = mod[3, 4] = 1.0
    arg = np.full_like(mod, 0.01)
    assert (3, 3) not in _points(_nms_extrema_scale(mod, arg, thresh=0.0, tie_rule="asymmetric"))
    assert (3, 3) in _points(_nms_extrema_scale(mod, arg, thresh=0.0, tie_rule="keep_both"))


def test_an_unknown_rule_is_refused():
    mod, arg = _plateau()
    with pytest.raises(ValueError, match="tie_rule"):
        _nms_extrema_scale(mod, arg, tie_rule="nope")


def test_the_wtmm_tools_take_it_keyed_and_the_run_records_it(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device, keyed_params
    from tests.conftest import make_field
    from tests.test_periodic_extension import _smooth

    register_builtin_devices()
    for name in ("wtmm2d", "wtmm2d_roi"):
        dev = get_device(name)
        p = next(q for q in dev.params if q.name == "tie_rule")
        assert p.default == "asymmetric" and p.active_when == ("detector", ("nms",))
        assert "tie_rule" in keyed_params(dev, defaults_for(dev))
    dev = get_device("wtmm2d")
    res = dev.compute(make_field(_smooth(96, seed=1)),
                      dict(defaults_for(dev), n_oct=1, n_voice=2, tie_rule="strict"))
    assert res["params"]["tie_rule"] == "strict"
