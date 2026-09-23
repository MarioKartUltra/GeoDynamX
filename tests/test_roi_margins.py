# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Every tool declares what it needs around an ROI.

The margin is the tool's own call, from its own settings; each one is measured -- an interior
ROI run with it must reproduce the same region of a whole-field run.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField

#: Transforms that never read a field's pixels: chain_topology consumes a WTMM RESULT,
#: backproject a point layer, stub_wavelet is a schema stub, and wtmm2d_roi reads its own
#: halos off the file (reads_source).
_NOT_FIELD_READERS = {"chain_topology", "backproject", "stub_wavelet", "wtmm2d_roi"}


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def test_every_builtin_field_reading_transform_declares_its_roi_margin(builtins):
    from dynamix.model.device import is_transform

    missing = sorted(name for name, dev in builtins.items()
                     if is_transform(dev) and name not in _NOT_FIELD_READERS
                     and not hasattr(dev, "roi_margin"))
    assert missing == [], f"no roi_margin declared: {missing}"


def test_noise_is_a_field_stage_with_no_margin_of_its_own(builtins):
    from dynamix.model.device import defaults_for

    noise = builtins["noise"]
    assert getattr(noise, "field_stage", False) is True
    assert noise.roi_margin(defaults_for(noise)) == 0


@pytest.mark.parametrize("name", ["pca", "tucker_havok"])
def test_global_statistics_tools_analyse_the_roi_itself(builtins, name):
    """PCA's covariance / the HOSVD are statistics OF the analysed region -- margin pixels
    would mix outside data into them, so they declare 0 (the ROI is what is decomposed)."""
    from dynamix.model.device import defaults_for

    dev = builtins[name]
    assert dev.roi_margin(defaults_for(dev)) == 0


# ------------------------------------------------------------ measured margins: the oracle

def _field(n=160, seed=11):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, n)).cumsum(0).cumsum(1)
    v[:, n // 2:] += 40.0 * v.std() / n                   # a step edge through the middle
    return RasterField(name="o", values=v, frame=LocalFrame(), x_axis=np.arange(float(n)),
                       y_axis=np.arange(float(n)))


ROI = (56, 56, 48, 48)


def _whole_and_roi(dev, params):
    from dynamix.roi.runner import crop_result_to_roi, run_on_region, split_lines

    f = _field()
    whole = dev.compute(f, params)
    r, c, h, w = ROI
    # the whole-field run cut to the ROI with the runner's own crop/relabel (offset = r == c)
    ref = crop_result_to_roi(whole, r, h, w, np.asarray(f.values).shape[:2])
    for lvl in ref.get("extrema") or []:
        if isinstance(lvl, dict) and "line_id" in lvl:
            lvl["line_id"] = split_lines(lvl)
    got = run_on_region([(dev, params)], f, ROI)
    return ref, got


def _assert_same_points(ref, got):
    assert len(ref["extrema"]) == len(got["extrema"])
    for a, b in zip(ref["extrema"], got["extrema"]):
        pa = set(zip(a["y"].tolist(), a["x"].tolist()))
        pb = set(zip(b["y"].tolist(), b["x"].tolist()))
        assert pa == pb, f"{len(pa ^ pb)} positions differ"


# ------------------------------------------------------------ D2: explicit diffusion tools

def _params(dev, **kw):
    from dynamix.model.device import defaults_for, validate_params

    p = defaults_for(dev)
    p.update(kw)
    return validate_params(dev, p)


@pytest.mark.parametrize("name,extra", [
    ("cdf_edges", {"variant": "linear", "detector": "nms"}),
    ("cdf_edges", {"variant": "nonlinear", "detector": "nms"}),
    ("cdf_edges", {"variant": "linear", "detector": "follow"}),
    ("pm_edges", {"detector": "nms"}),
    ("pm_edges", {"detector": "follow"}),
])
@pytest.mark.parametrize("n_levels", [2, 3])
def test_explicit_diffusion_tools_reproduce_the_whole_field_inside_the_roi(builtins, name,
                                                                           extra, n_levels):
    """An explicit stencil moves information at most 1 px per iteration, so a margin of the
    coarsest level's iteration count (+ the gradient/NMS/probe stencils) is EXACT. floor = 0:
    a relative floor is relative to the analysed window by nature (the wtmm2d_roi precedent)."""
    dev = builtins[name]
    params = _params(dev, n_levels=n_levels, floor=0.0, **extra)
    assert dev.roi_margin(params) < ROI[0]            # the oracle is an INTERIOR claim
    ref, got = _whole_and_roi(dev, params)
    _assert_same_points(ref, got)


# ------------------------------------------------------------ D3: Mallat-Zhong on the torus

@pytest.mark.parametrize("wavelet,alpha", [("mz_spline", 3.0), ("frac_bspline", 2.5)])
@pytest.mark.parametrize("n_levels", [2, 3])
def test_mz_edges_reproduces_the_whole_field_inside_the_roi(builtins, wavelet, alpha,
                                                            n_levels):
    """mz mirrors its input onto a 2N x 2N torus; the margin (the transform's own measured
    impulse reach at the coarsest level) keeps that mirror seam -- at the WINDOW edge -- out
    of the ROI."""
    from dynamix.roi.runner import crop_result_to_roi, run_on_region, split_lines

    dev = builtins["mz_edges"]
    params = _params(dev, n_levels=n_levels, wavelet=wavelet, alpha=alpha)
    m = dev.roi_margin(params)
    assert m > 0
    n = 2 * (m + 8) + 48
    f = _field(n=n)
    roi = (m + 8, m + 8, 48, 48)
    whole = dev.compute(f, params)
    ref = crop_result_to_roi(whole, roi[0], 48, 48, (n, n))
    for lvl in ref["extrema"]:
        lvl["line_id"] = split_lines(lvl)
    got = run_on_region([(dev, params)], f, roi)
    _assert_same_points(ref, got)


# ------------------------------------------------------------ D4: Hölder tools

def _holder_dh(dev, params):
    from dynamix.roi.runner import run_on_region

    m = dev.roi_margin(params)
    n = 2 * (m + 8) + 48
    f = _field(n=n)
    roi = (m + 8, m + 8, 48, 48)
    whole = dev.compute(f, params)["h_map"][roi[0]:roi[0] + 48, roi[1]:roi[1] + 48]
    got = run_on_region([(dev, params)], f, roi)["h_map"]
    assert np.array_equal(np.isnan(whole), np.isnan(got))
    return np.asarray(got, float) - np.asarray(whole, float)


def test_the_measure_route_reproduces_the_whole_field_hmap_inside_the_roi(builtins):
    """The measure route (positive Gaussian kernel, regression): the margin -- the largest
    kernel's measured 2-D reach -- makes an interior ROI's h(x) the whole-field h(x)."""
    dev = builtins["holder_measure"]
    dh = _holder_dh(dev, _params(dev, kappa=4.0, n_scales=4, wavelet="gaussian"))
    assert np.nanmax(np.abs(dh)) < 1e-3


@pytest.mark.parametrize("name,extra", [
    ("holder_multiaffine", {"wavelet": "g2"}),
    ("holder_multiaffine", {"wavelet": "g3"}),
    ("holder_map", {}),
])
def test_the_multiaffine_route_agrees_in_the_bulk_its_zero_mean_is_domain_coupled(builtins,
                                                                                  name,
                                                                                  extra):
    """KNOWN PROPERTY (2026-09-22, measured): the multiaffine kernels are made exactly
    zero-mean over the WHOLE analysed domain (ricker_projections), which DC-couples T to the
    domain's mean -- so h differs where |T| ~ 0 (zero crossings, where h is ill-conditioned
    anyway: the whole-field h spans -4..7.5 there). Bulk agreement measured: median |dh|
    0.005 (g2) / 0.0004 (g3); >0.05 on 5.3% / 0.7% of pixels. A local zero-mean would remove
    it but changes whole-field results too -- a separate decision."""
    dev = builtins[name]
    dh = np.abs(_holder_dh(dev, _params(dev, kappa=4.0, n_scales=4, **extra)))
    assert np.nanmedian(dh) < 0.01
    assert np.nanmean(dh > 0.05) < 0.08


def test_the_punctual_estimator_is_offset_by_its_domain_normalization(builtins):
    """KNOWN PROPERTY: punctual h = log T / log(r / sqrt(N)) -- Turiel's resolution scale is
    relative to the ANALYSED domain, so ROI + margin shifts h uniformly (measured ~0.017 here,
    spread < 0.01). Normalising by the dataset's full dims would remove it -- but also change
    today's windowed-open results: a separate decision."""
    dev = builtins["holder_measure"]
    dh = _holder_dh(dev, _params(dev, kappa=4.0, n_scales=4, wavelet="gaussian",
                                 estimator="punctual"))
    assert abs(np.nanmedian(dh)) < 0.05 and np.nanstd(dh) < 0.01


def test_a_heavy_tailed_holder_kernel_still_runs_on_an_roi_within_its_cap(builtins):
    """A Lorentzian with beta <= 1 has a non-integrable 2-D tail: its h(x) depends on the
    analysed domain by nature (Turiel 2008 app. A). It runs, with its capped margin."""
    from dynamix.roi.runner import run_on_region

    dev = builtins["holder_measure"]
    params = _params(dev, kappa=2.0, n_scales=3, r_min=1.0, wavelet="lorentzian", beta=1.0)
    m = dev.roi_margin(params)
    assert m <= 32 * 2 * 1.5                              # the cap, not the field
    res = run_on_region([(dev, params)], _field(n=96), (24, 24, 48, 48))
    assert res["h_map"].shape == (48, 48) and np.isfinite(res["h_map"]).any()


# ------------------------------------------------------------ D5: wavelet_skeleton

@pytest.mark.parametrize("kernel", ["tang_you", "gaussian"])
@pytest.mark.parametrize("n_stages", [1, 2, 3])
def test_wavelet_skeleton_reproduces_the_whole_field_skeleton_inside_the_roi(builtins,
                                                                             kernel,
                                                                             n_stages):
    """Reach adds up per stage: the s1 kernel + gradient + the directional minima search,
    then per refinement stage the s2 kernel + gradient + search + the locality dilation.
    edge_frac is relative to the ANALYSED peak (the relative-floor class), so the field
    carries one dominant bump inside the ROI -- the same peak in both runs."""
    from dynamix.roi.runner import crop_result_to_roi, run_on_region, split_lines

    dev = builtins["wavelet_skeleton"]
    params = _params(dev, s1=4.0, s2=4.0, n_stages=n_stages, wavelet=kernel)
    m = dev.roi_margin(params)
    n = 2 * (m + 8) + 48
    f = _field(n=n)
    yy, xx = np.mgrid[0:n, 0:n]
    c = n // 2
    bump = 1e4 * np.asarray(f.values).std() * np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / 18.0)
    f = RasterField(name="b", values=np.asarray(f.values) + bump, frame=f.frame,
                    x_axis=f.x_axis, y_axis=f.y_axis)
    roi = (m + 8, m + 8, 48, 48)
    whole = dev.compute(f, params)
    ref = crop_result_to_roi(whole, roi[0], 48, 48, (n, n))
    for lvl in ref["extrema"]:
        lvl["line_id"] = split_lines(lvl)
    got = run_on_region([(dev, params)], f, roi)
    _assert_same_points(ref, got)
