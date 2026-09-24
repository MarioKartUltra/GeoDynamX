# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""cdf_edges device: one complex cross-diffusion evolution, a dyadic M-Z-convention edge
pyramid out (the fourth peer analyzer -- wtmm2d / mz_edges / wavelet_skeleton /
lcdf_edges). Linear (script 24's evolution) or nonlinear (Perona-Malik-style, the Im
channel modulating its own diffusivity); per-snapshot grad-Re NMS maxima ("mz" mode) or
Im zero crossings ("marr" mode) in the app's extrema schema."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import cdf as lcdf
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices.cdf_edges import CDFEdges
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
    assert "cdf_edges" in DEVICES
    assert is_transform(DEVICES["cdf_edges"])


def test_param_surface():
    params = {p.name: p for p in CDFEdges.params}
    assert list(params) == ["n_levels", "variant", "k_edge", "theta", "dt",
                            "edge_mode", "detector", "show", "interpolate", "floor"]
    assert params["variant"].choices == ("linear", "nonlinear")
    assert params["edge_mode"].choices == ("mz", "marr")
    assert params["dt"].max <= 0.24                    # explicit-Euler stability cap


def test_linear_pyramid_finds_the_step_at_every_level():
    field = _step_field()
    res = CDFEdges().compute(field, defaults_for(CDFEdges()))
    assert res["chains"] == []
    np.testing.assert_array_equal(res["scales"], [1.0, 2.0, 4.0, 8.0])
    assert len(res["extrema"]) == 4
    for k, e in enumerate(res["extrema"]):
        assert set(e) == {"x", "y", "mod", "arg", "line_id"}
        assert len(e["x"]) > 0, k
        assert np.all(np.abs(e["x"] - 47.5) <= 2 + k), k   # the step edge, per level
    assert res["filtered"].shape == field.values.shape      # final Re channel rides
    assert res["edge_channel"].shape == field.values.shape  # final Im channel rides
    assert res["_shape"] == field.values.shape


def test_the_pyramid_is_one_evolution_snapshotted():
    """The linear pyramid must equal script 26's recipe exactly: one lcdf_evolve with
    sigma_iters snapshots, grad-Re NMS per snapshot."""
    field = _step_field()
    params = defaults_for(CDFEdges())
    res = CDFEdges().compute(field, params)
    vals = np.asarray(field.values, dtype=np.float64)
    iters = lcdf.sigma_iters((1.0, 2.0, 4.0, 8.0), theta=params["theta"],
                             dt=params["dt"])
    _I, snaps = lcdf.lcdf_evolve(vals, max(iters.values()), snapshots=set(iters.values()),
                                 theta=params["theta"], dt=params["dt"])
    keep, mag = lcdf.grad_nms(snaps[iters[2.0]].real, params["floor"])
    ys, xs = np.where(keep)
    e = res["extrema"][1]
    np.testing.assert_array_equal(np.sort(e["y"] * 1000 + e["x"]),
                                  np.sort(ys * 1000 + xs))
    np.testing.assert_array_equal(e["mod"], mag[e["y"], e["x"]])


def test_marr_mode_reads_the_im_channel():
    field = _step_field()
    base = defaults_for(CDFEdges())
    mz = CDFEdges().compute(field, base)
    marr = CDFEdges().compute(field, dict(base, edge_mode="marr"))
    assert not np.array_equal(mz["extrema"][1]["x"], marr["extrema"][1]["x"])
    e = marr["extrema"][1]
    assert len(e["x"]) > 0
    assert np.all(np.abs(e["x"] - 47.5) <= 4)          # Marr zero crossings straddle the edge


def test_nonlinear_variant_differs_and_k_binds():
    field = _step_field()
    base = defaults_for(CDFEdges())
    lin = CDFEdges().compute(field, base)
    non = CDFEdges().compute(field, dict(base, variant="nonlinear", k_edge=0.2))
    assert not np.array_equal(lin["filtered"], non["filtered"])
    non2 = CDFEdges().compute(field, dict(base, variant="nonlinear", k_edge=5.0))
    assert not np.array_equal(non["filtered"], non2["filtered"])


def test_cache_key_tracks_every_param():
    dev = CDFEdges()
    base = defaults_for(dev)
    k0 = dev.cache_key("src", base)
    for change in ({"n_levels": 3}, {"variant": "nonlinear"}, {"k_edge": 0.5},
                   {"theta": 0.15}, {"dt": 0.1}, {"edge_mode": "marr"}, {"floor": 0.1}):
        assert dev.cache_key("src", dict(base, **change)) != k0, change


def test_refusals_are_the_shared_contract():
    dev = CDFEdges()
    with pytest.raises(ValueError, match="FIRST in its chain"):
        dev.compute({"chains": [], "extrema": []}, defaults_for(dev))


# ----------------------------------------------------- show + interpolate (2026-09-21)


def test_show_filtered_stamps_the_raster_out_display():
    field64 = _step_field()
    dev = CDFEdges()
    params = dict(defaults_for(dev), n_levels=2)
    # ``show`` is view-only: one compute, the display pick is the device's view step.
    res = dev.compute(field64, params)
    edges = dev.view(res, dict(params, show="edges"))
    assert "raster_out" not in edges
    filt = dev.view(res, dict(params, show="filtered"))
    np.testing.assert_array_equal(filt["raster_out"], filt["filtered"])
    imag = dev.view(res, dict(params, show="edge_channel"))
    np.testing.assert_array_equal(imag["raster_out"], imag["edge_channel"])
    # the extrema pyramid still rides either way
    assert any(e["x"].size for e in filt["extrema"])


def test_interpolate_adds_float_channels_in_both_edge_modes():
    field64 = _step_field()
    dev = CDFEdges()
    base = dict(defaults_for(dev), n_levels=2, interpolate=True)
    for mode in ("mz", "marr"):
        res = dev.compute(field64, dict(base, edge_mode=mode))
        for e in res["extrema"]:
            assert "x_sub" in e and e["x_sub"].shape == e["x"].shape
            if e["x"].size:
                t = ((e["x_sub"] - e["x"]) * np.cos(e["arg"])
                     + (e["y_sub"] - e["y"]) * np.sin(e["arg"]))
                assert np.all(np.abs(t) <= 0.5 + 1e-9)          # half-pixel clamp, both modes
        off = dev.compute(field64, dict(base, interpolate=False, edge_mode=mode))
        assert all("x_sub" not in e for e in off["extrema"])


def test_follow_detector_finds_a_closed_ring_on_a_disk():
    """detector='follow' on cdf_edges (2026-09-22): a disk's edge is a closed contour --
    the xsmurf search_lines walk must mark the big line CLOSED."""
    import numpy as np
    n = 96
    yy, xx = np.mgrid[0:n, 0:n]
    f = 1.0 / (1.0 + np.exp(-(np.hypot(yy - 47.7, xx - 48.2) - 27.3)))
    field = RasterField(name="disk", values=f, frame=LocalFrame(),
                        x_axis=np.arange(n, dtype=np.float64),
                        y_axis=np.arange(n, dtype=np.float64))
    dev = CDFEdges()
    res = dev.compute(field, dict(defaults_for(dev), detector="follow", n_levels=3))
    assert "_hline_closed" in res
    closed_any = False
    for runs, closed in zip(res["_hline_runs"], res["_hline_closed"]):
        for r, c in zip(runs, closed):
            if c and r.size > 40:
                closed_any = True
    assert closed_any


def test_cancel_aborts_the_evolution():
    from dynamix.core.wtmm_backend import ComputeCancelled
    assert CDFEdges.wants_cancel is True
    with pytest.raises(ComputeCancelled):
        CDFEdges().compute(_step_field(), defaults_for(CDFEdges()), cancel=lambda: True)
