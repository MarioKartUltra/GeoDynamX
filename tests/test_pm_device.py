# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""pm_edges device: one Perona-Malik 1990 anisotropic-diffusion evolution (the REAL
gradient-driven flow -- dynamix.core.pm, the paper's scheme (7)+(8)+(10)), snapshotted at
the nominal-sigma dyadic schedule, per-snapshot grad-NMS maxima in the app's extrema
schema. A peer of cdf_edges, never a mode of it: different flow (real vs complex),
different edge signal (|grad I| vs the Im channel), and the paper's signature property --
immediate localization -- is pinned as behavior."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import cdf, pm
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices.pm_edges import PMEdges
from dynamix.model.device import defaults_for


def _step_field(n=96):
    f = np.zeros((n, n))
    f[:, n // 2:] = 1.0
    return RasterField(name="step", values=f, frame=LocalFrame(),
                       x_axis=np.arange(n, dtype=np.float64),
                       y_axis=np.arange(n, dtype=np.float64))


def test_registered_as_a_builtin_transform(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import DEVICES, is_transform

    register_builtin_devices()
    assert "pm_edges" in DEVICES
    assert is_transform(DEVICES["pm_edges"])


def test_param_surface():
    params = {p.name: p for p in PMEdges.params}
    assert list(params) == ["n_levels", "k_edge", "g", "lam", "show", "detector",
                            "interpolate", "floor"]
    assert params["g"].choices == ("exp", "frac")
    assert params["show"].choices == ("edges", "filtered")
    assert params["lam"].max <= 0.25                   # the paper's stability cap


def test_pyramid_finds_the_step_at_every_level():
    field = _step_field()
    res = PMEdges().compute(field, defaults_for(PMEdges()))
    assert res["chains"] == []
    np.testing.assert_array_equal(res["scales"], [1.0, 2.0, 4.0, 8.0])
    assert len(res["extrema"]) == 4
    for k, e in enumerate(res["extrema"]):
        assert set(e) == {"x", "y", "mod", "arg", "line_id"}
        assert len(e["x"]) > 0, k
    assert res["filtered"].shape == field.values.shape
    assert res["_shape"] == field.values.shape


def test_immediate_localization_the_edge_does_not_drift():
    """The paper's central claim vs linear scale space: with K below the edge contrast the
    edge stays sharp AND in place at every level -- no coarse-to-fine tracking needed. The
    cdf_edges pin tolerates drift growing with level (2 + k); this one must not."""
    field = _step_field()
    params = dict(defaults_for(PMEdges()), k_edge=0.2)
    res = PMEdges().compute(field, params)
    for k, e in enumerate(res["extrema"]):
        assert len(e["x"]) > 0, k
        assert np.all(np.abs(e["x"] - 47.5) <= 1.0), k


def test_the_pyramid_is_one_evolution_snapshotted():
    """The pyramid must equal one pm_evolve with sigma_iters snapshots, grad-NMS per
    snapshot -- the cdf_edges precedent, against the pm core directly."""
    field = _step_field()
    params = defaults_for(PMEdges())
    res = PMEdges().compute(field, params)
    iters = pm.sigma_iters((1.0, 2.0, 4.0, 8.0), lam=params["lam"])
    _I, snaps = pm.pm_evolve(np.asarray(field.values, dtype=np.float64),
                             params["k_edge"], max(iters.values()),
                             snapshots=set(iters.values()), g=params["g"],
                             lam=params["lam"])
    keep, mag = cdf.grad_nms(snaps[iters[2.0]], params["floor"])
    ys, xs = np.where(keep)
    e = res["extrema"][1]
    np.testing.assert_array_equal(np.sort(e["y"] * 1000 + e["x"]),
                                  np.sort(ys * 1000 + xs))
    np.testing.assert_array_equal(e["mod"], mag[e["y"], e["x"]])


def test_k_and_g_bind():
    field = _step_field()
    base = defaults_for(PMEdges())
    sharp = PMEdges().compute(field, dict(base, k_edge=0.2))
    blurry = PMEdges().compute(field, dict(base, k_edge=1000.0))
    assert not np.array_equal(sharp["filtered"], blurry["filtered"])
    frac = PMEdges().compute(field, dict(base, k_edge=0.2, g="frac"))
    assert not np.array_equal(sharp["filtered"], frac["filtered"])


def test_show_filtered_stamps_the_raster_out_display():
    field = _step_field()
    dev = PMEdges()
    params = dict(defaults_for(dev), n_levels=2)
    # ``show`` is view-only: one compute, the display pick is the device's view step.
    res = dev.compute(field, params)
    edges = dev.view(res, dict(params, show="edges"))
    assert "raster_out" not in edges
    filt = dev.view(res, dict(params, show="filtered"))
    np.testing.assert_array_equal(filt["raster_out"], filt["filtered"])
    assert any(e["x"].size for e in filt["extrema"])   # the pyramid still rides


def test_interpolate_adds_clamped_float_channels():
    field = _step_field()
    dev = PMEdges()
    base = dict(defaults_for(dev), n_levels=2, interpolate=True)
    res = dev.compute(field, base)
    for e in res["extrema"]:
        assert "x_sub" in e and e["x_sub"].shape == e["x"].shape
        if e["x"].size:
            t = ((e["x_sub"] - e["x"]) * np.cos(e["arg"])
                 + (e["y_sub"] - e["y"]) * np.sin(e["arg"]))
            assert np.all(np.abs(t) <= 0.5 + 1e-9)
    off = dev.compute(field, dict(base, interpolate=False))
    assert all("x_sub" not in e for e in off["extrema"])


def test_cache_key_tracks_every_param():
    dev = PMEdges()
    base = defaults_for(dev)
    k0 = dev.cache_key("src", base)
    for change in ({"n_levels": 3}, {"k_edge": 0.5}, {"g": "frac"}, {"lam": 0.1},
                   {"interpolate": True}, {"floor": 0.1}):
        assert dev.cache_key("src", dict(base, **change)) != k0, change
    # ``show`` is view-only: switching it is a cache hit, never a recompute.
    assert dev.cache_key("src", dict(base, show="filtered")) == k0


def test_refusals_are_the_shared_contract():
    dev = PMEdges()
    with pytest.raises(ValueError, match="FIRST in its chain"):
        dev.compute({"chains": [], "extrema": []}, defaults_for(dev))


def test_follow_detector_on_pm(qtbot=None):
    """detector='follow': xsmurf's own four-images design -- kapa/kapap from
    FD derivative stacks of the DIFFUSED snapshot, the exact ported detector on top."""
    field = _step_field()
    dev = PMEdges()
    params = {p.name: p for p in PMEdges.params}
    assert params["detector"].choices == ("nms", "follow")
    res = dev.compute(field, dict(defaults_for(dev), detector="follow"))
    e = res["extrema"][1]
    assert len(e["x"]) > 0
    assert np.all(np.abs(e["x"] - 47.5) <= 2)          # the step edge, still localized
    assert "x_sub" in e                                 # follow's native subpixel channel
    assert "_hline_runs" in res and "_hline_closed" in res
    k0 = dev.cache_key("src", defaults_for(dev))
    assert dev.cache_key("src", dict(defaults_for(dev), detector="follow")) != k0


def test_cancel_aborts_the_evolution():
    from dynamix.core.wtmm_backend import ComputeCancelled
    assert PMEdges.wants_cancel is True
    with pytest.raises(ComputeCancelled):
        PMEdges().compute(_step_field(), defaults_for(PMEdges()), cancel=lambda: True)
