# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The CWT leaders device (spec 30 §5): the scaling laws on fields with a known answer, the
stage-reuse law, the maxima measure, the ROI law and the Samples contract."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import fbm2d
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField


def _field(values, name="f"):
    values = np.asarray(values, dtype=np.float64)
    ny, nx = values.shape
    return RasterField(name=name, values=values, frame=LocalFrame(),
                       x_axis=np.arange(nx, dtype=np.float64), y_axis=np.arange(ny, dtype=np.float64))


def _params(dev, **over):
    from dynamix.model.device import defaults_for

    p = defaults_for(dev)
    p.update({"wavelet": "gaussian", "n_oct": 4, "a_min": 1.0, "edge": "reflect", "p": 0.0})
    p.update(over)
    return p


def _zeta(res, qs, set_name="cwt-leaders"):
    """The structure-function slopes ζ(q) over the device's own fit range: log of the mean of
    L^q over the valid pixels against log a."""
    s = res["samples"]["sets"][set_name]
    lo, hi = res["samples"]["fit_range"]
    scales = np.asarray(res["scales"])
    use = [j for j, a in enumerate(scales) if lo <= a <= hi]
    out = {}
    for q in qs:
        logs = [np.log(np.mean(s["values"][j][s["valid"][j]].astype(np.float64) ** q)) for j in use]
        out[q] = float(np.polyfit(np.log(scales[use]), logs, 1)[0])
    return out


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


@pytest.fixture
def dev(builtins):
    return builtins["cwt_leaders"]


@pytest.mark.parametrize("H", [0.3, 0.7])
def test_fbm_gives_linear_zeta_with_slope_h(dev, H):
    """ζ(q) = qH for fractional Brownian motion: the leaders' structure functions over the fit
    range, for q of both signs; finite size on 512² sets the tolerance."""
    res = dev.compute(_field(fbm2d(512, H, seed=3)), _params(dev))
    z = _zeta(res, (-1.0, 1.0, 2.0))
    for q, zeta in z.items():
        # the log of a mean is biased low by its variance, more at large |q| and at the coarse
        # end where the independent discs are few (spec 30 §2): the tolerance grows with |q|
        assert zeta == pytest.approx(q * H, abs=0.1 * max(1.0, abs(q))), (q, zeta, z)


def test_negative_moments_are_finite_on_noisy_fbm(dev):
    rng = np.random.default_rng(0)
    f = fbm2d(256, 0.5, seed=1) + 0.05 * rng.standard_normal((256, 256))
    res = dev.compute(_field(f), _params(dev, n_oct=3))
    s = res["samples"]["sets"]["cwt-leaders"]
    for q in (-10.0, -4.0, -1.0):
        for j, a in enumerate(res["scales"]):
            m = np.mean(s["values"][j][s["valid"][j]].astype(np.float64) ** q)
            assert np.isfinite(m) and m > 0, (q, a)


@pytest.mark.parametrize("h,operator,wavelet", [(0.3, "gradient", "gaussian"), (0.6, "gradient", "gaussian"),
                                                (0.3, "none", "mexican"), (0.6, "none", "mexican")])
def test_an_isotropic_cusp_returns_its_own_exponent_at_the_centre(dev, h, operator, wavelet):
    """|x − x₀|^h: the leader at x₀ scales as a^h, so the h-map's centre reads h. This also pins
    the stage's L¹ normalisation (an L² stage would read h + 1) for both operators."""
    n = 512                                   # four scales keep their 16 independent discs
    y, x = np.mgrid[:n, :n] - n / 2 + 0.5
    r = np.hypot(x, y)
    res = dev.compute(_field(r ** h), _params(dev, operator=operator, wavelet=wavelet))
    c = n // 2
    centre = np.nanmean(res["h_map"][c - 2:c + 2, c - 2:c + 2])
    assert centre == pytest.approx(h, abs=0.06), (centre, res["record"]["fit_range"])


def test_a_leaders_run_after_a_wtmm_run_reuses_the_cwt_stage(dev, builtins):
    wtmm = builtins["wtmm2d"]
    from dynamix.model.device import defaults_for

    f = _field(fbm2d(128, 0.5, seed=2), name="shared")
    cache: dict = {}
    wp = defaults_for(wtmm)
    wp.update({"wavelet": "gaussian", "n_oct": 4, "n_voice": 1, "a_min": 1.0, "edge": "reflect"})
    wtmm.compute(f, wp, stage_cache=cache)
    res = dev.compute(f, _params(dev), stage_cache=cache)
    assert res["record"]["cwt_cache_hits"] == ["cwt"]


def test_the_maxima_measure_covers_only_the_wtmms_maxima(dev):
    f = _field(fbm2d(128, 0.5, seed=4), name="maxima")
    cache: dict = {}
    every = dev.compute(f, _params(dev, n_oct=3), stage_cache=cache)
    only = dev.compute(f, _params(dev, n_oct=3, measure="wtmm"), stage_cache=cache)
    assert only["record"]["cwt_cache_hits"] == ["cwt"]            # the shared stage again
    smaller_somewhere = False
    for j in range(len(every["scales"])):
        n_every, n_only = every["_valid"][j].sum(), only["_valid"][j].sum()
        # a disc with no maximum inside has no leader; on a dense field most discs have one
        assert 0 < n_only <= n_every, (j, n_only, n_every)
        # a sup over a subset never exceeds the sup over the set
        both = every["_valid"][j] & only["_valid"][j]
        assert np.all(only["_leaders"][j][both] <= every["_leaders"][j][both] + 1e-6)
        smaller_somewhere |= bool(np.any(only["_leaders"][j][both] < every["_leaders"][j][both] - 1e-6))
    assert smaller_somewhere


def test_a_region_equals_the_whole_field_cut_to_it(dev):
    """The ROI law: the region's leaders are the whole field's inside the region, to the float32
    tolerance of the 1 % of the kernel's energy that lies beyond R99 (spec 30 §3.4)."""
    from dynamix.roi.runner import run_on_region

    f = _field(fbm2d(192, 0.5, seed=5), name="roi")
    params = _params(dev, n_oct=3)
    m = dev.roi_margin(params)
    assert m > 0
    r, c, h, w = 64, 70, 48, 40
    whole = dev.compute(f, params)
    got = run_on_region([(dev, params)], f, (r, c, h, w))
    assert got["_shape"] == (h, w) and got["_leaders"].shape[1:] == (h, w)
    for j in range(len(whole["scales"])):
        ref = whole["_leaders"][j, r:r + h, c:c + w]
        np.testing.assert_allclose(got["_leaders"][j], ref, rtol=2e-3, atol=1e-4)


def test_the_sample_sets_reach_the_windows_with_their_weights(dev):
    from dynamix.model.samples import has_sample_sets, sample_sets

    res = dev.compute(_field(fbm2d(128, 0.5, seed=6)), _params(dev, n_oct=3, p=2.0))
    assert has_sample_sets(res)
    sets = sample_sets(res)
    assert "CWT leaders" in sets and "CWT p-leaders" in sets
    cells = res["samples"]["cells"]
    for j, (mods, weight) in enumerate(zip(sets["CWT leaders"]["mods"], sets["CWT leaders"]["weights"])):
        assert mods.size == int(res["_valid"][j].sum()) and np.all(np.isfinite(mods))
        assert weight == pytest.approx(cells[j] / mods.size)
    assert res["samples"]["fit_range"] == res["record"]["fit_range"]


def test_the_lazy_outputs_and_the_reading(dev):
    res = dev.compute(_field(fbm2d(96, 0.5, seed=7)), _params(dev, n_oct=3, p=2.0))
    for name in ("leader", "p_leader"):
        out = dev.compute_output(name, None, res, _params(dev), fetch=None, level=1)
        assert out["raster"].shape == (96, 96) and out["diag"]["level"] == 1
        assert out["diag"]["radius_px"] == pytest.approx(res["_radii"][1])
    text = dev.reading(res, _params(dev))
    assert "gradient of gaussian" in text and "R " in text and "p = 2" in text
    with pytest.raises(ValueError, match="p_leader"):
        dev.compute_output("p_leader", None, dev.compute(_field(fbm2d(64, 0.5)), _params(dev, n_oct=2)),
                           _params(dev), fetch=None)


def test_sup_from_and_radius_k_do_not_move_zeta_beyond_its_error(dev):
    f = _field(fbm2d(512, 0.5, seed=8))
    base = _zeta(dev.compute(f, _params(dev)), (1.0, 2.0))
    cut = _zeta(dev.compute(f, _params(dev, sup_from=1)), (1.0, 2.0))
    wide = _zeta(dev.compute(f, _params(dev, radius_k=2.0)), (1.0, 2.0))
    for q in (1.0, 2.0):
        assert abs(cut[q] - base[q]) < 0.1 and abs(wide[q] - base[q]) < 0.1, (q, base, cut, wide)
