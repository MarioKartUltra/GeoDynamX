# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Live progress: the slow cores report from INSIDE their loops (per scale, per window pass, per
HOOI sweep, per stage), one fraction that only ever increases and ends at 1.0 -- so the strip's
"stage NN%" moves instead of sitting at one number until the result lands."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField


class _Rec:
    def __init__(self):
        self.calls = []

    def __call__(self, stage, frac):
        self.calls.append((stage, float(frac)))

    @property
    def fracs(self):
        return [f for _s, f in self.calls]


def _assert_live(rec, at_least):
    f = rec.fracs
    assert len(f) >= at_least, rec.calls
    assert all(b >= a - 1e-12 for a, b in zip(f, f[1:])), f     # never goes backwards
    assert all(0.0 <= x <= 1.0 + 1e-12 for x in f)
    assert f[-1] == pytest.approx(1.0)


def _img(ny=30, nx=34, seed=4):
    return np.random.default_rng(seed).standard_normal((ny, nx)).cumsum(0).cumsum(1)


def test_tucker_2d_reports_through_every_pass():
    from dynamix.core.tucker_havok import tucker_havok_2d

    rec = _Rec()
    tucker_havok_2d(_img(), n_delays=6, ranks=(3, 0, 0), sweeps=2, progress=rec)
    _assert_live(rec, at_least=20)


@pytest.mark.parametrize("which", ["tape", "plain"])
def test_tucker_1d_and_plain_report_per_mode_and_sweep(which):
    from dynamix.core.tucker_havok import tucker_havok, tucker_plain

    rec = _Rec()
    if which == "tape":
        tucker_havok(_img(), n_delays=6, ranks=(3, 4, 0), sweeps=2, progress=rec)
    else:
        tucker_plain(_img(), ranks=(4, 4), sweeps=2, progress=rec)
    _assert_live(rec, at_least=5)


@pytest.mark.parametrize("route", ["ricker", "measure"])
def test_holder_projections_report_per_scale(route):
    from dynamix.core import microcanonical as mc

    rec = _Rec()
    scales = np.geomspace(2, 12, 7)
    if route == "ricker":
        mc.ricker_projections(_img(), scales, progress=rec)
    else:
        mc.measure_projections(mc.gradient_measure(_img()), scales, progress=rec)
    assert rec.fracs == pytest.approx([(i + 1) / 7 for i in range(7)])


def test_the_skeleton_reports_per_stage():
    from dynamix.core.wavelet_skeleton import skeletonize

    rec = _Rec()
    skeletonize(_img(48, 52), n_stages=3, progress=rec)
    _assert_live(rec, at_least=3)


@pytest.mark.parametrize("device,params", [
    ("tucker_havok", {"n_delays": 6, "rank_delay": 3}),
    ("holder_multiaffine", {}),
    ("holder_measure", {}),
    ("holder_map", {}),
    ("wavelet_skeleton", {"n_stages": 3}),
])
def test_devices_report_live_not_one_number_then_done(clean_registry, device, params):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device

    register_builtin_devices()
    dev = get_device(device)
    field = RasterField(name="f", values=_img(), frame=LocalFrame(), x_axis=np.arange(34.0),
                        y_axis=np.arange(30.0))
    rec = _Rec()
    dev.compute(field, {**defaults_for(dev), **params}, progress=rec)
    _assert_live(rec, at_least=6)
