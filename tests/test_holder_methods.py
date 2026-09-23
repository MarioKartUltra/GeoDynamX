# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tests for the SPLIT microcanonical tools (devices/holder_methods.py) -- holder_measure and
holder_multiaffine, one device per method with only the wavelet family that method admits
(2026-09-19 design LAW). The conflated
holder_map stays registered for saved projects; these are its successors."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import fbm2d
from dynamix.core import microcanonical as mc
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.devices.holder_map import HolderMap
from dynamix.devices.holder_methods import HolderMeasure, HolderMultiaffine, method_arrays
from dynamix.model.device import defaults_for


def _field(n=64, H=0.5, seed=0):
    vals = fbm2d(n, H, seed=seed)
    return RasterField(name=f"f{n}", values=vals, frame=LocalFrame(),
                       x_axis=np.arange(n, dtype=np.float64),
                       y_axis=np.arange(n, dtype=np.float64))


def test_registered_as_builtins(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import DEVICES

    register_builtin_devices()
    assert "holder_measure" in DEVICES
    assert "holder_multiaffine" in DEVICES
    assert "holder_map" in DEVICES        # superseded, never deregistered (saved projects)


def test_wavelet_menus_are_method_true():
    """The smell being fixed: 'wavelet' meant a different kernel under each estimator. Each
    split tool lists ONLY the class its method admits, under the method's own names."""
    measure = {p.name: p for p in HolderMeasure.params}
    multi = {p.name: p for p in HolderMultiaffine.params}
    assert measure["wavelet"].choices == ("gaussian", "q_gaussian", "lorentzian",
                                          "frac_gaussian")
    assert multi["wavelet"].choices == ("g1", "g2", "g3", "q_mexican", "lorentzian_marr",
                                        "frac_gaussian")
    assert multi["wavelet"].default == "g2"     # today's default Ricker, method-true name
    assert measure["wavelet"].default == "gaussian"
    for params in (measure, multi):
        assert params["estimator"].choices == ("regression", "punctual")
        assert params["estimator"].default == "regression"
        assert "frac_n" in params


def test_multiaffine_regression_is_exactly_the_core_pipeline():
    """g2 default == the conflated device's multiaffine/gaussian default: the split changes
    the MENU, never the physics."""
    field = _field()
    params = defaults_for(HolderMultiaffine())
    res = HolderMultiaffine().compute(field, params)
    old = HolderMap().compute(field, defaults_for(HolderMap()))
    np.testing.assert_array_equal(res["h_map"], old["h_map"])
    np.testing.assert_array_equal(res["r2_map"], old["r2_map"])
    assert res["chains"] == [] and res["extrema"] == []
    assert res["_shape"] == field.values.shape


def test_measure_regression_is_exactly_the_core_pipeline():
    field = _field()
    params = dict(defaults_for(HolderMeasure()), r_min=2.0)
    res = HolderMeasure().compute(field, params)
    scales = np.geomspace(2.0, 2.0 * params["kappa"], int(params["n_scales"]))
    T = mc.measure_projections(mc.gradient_measure(np.asarray(field.values, np.float64)),
                               scales)
    h, r2 = mc.singularity_map_regression(T, scales, r2_min=0.0)
    np.testing.assert_array_equal(res["h_map"], h)
    np.testing.assert_array_equal(res["r2_map"], r2)


def test_punctual_estimator_is_ponts_single_finest_scale():
    """estimator="punctual" (Pont 2006 SS II.B): h is read from the FINEST scale alone,
    log-normalized by the ensemble mean -- the left-limb/MSC extractor with the sharpest
    localization. r2_map is all-NaN (no regression ran). The full scale grid is still
    projected: the no-support mask is a DATA-QUALITY property shared with the regression
    estimator (the cross-scale relative floor), never an estimator choice -- so ``scales``
    is the full grid and the ESTIMATE at supported pixels comes from scales[0] only."""
    field = _field()
    for dev, project in ((HolderMeasure(), "measure"), (HolderMultiaffine(), "multi")):
        params = dict(defaults_for(dev), estimator="punctual", r_min=2.0)
        res = dev.compute(field, params)
        vals = np.asarray(field.values, dtype=np.float64)
        scales = np.geomspace(2.0, 2.0 * params["kappa"], int(params["n_scales"]))
        if project == "measure":
            T = mc.measure_projections(mc.gradient_measure(vals), scales)
        else:
            T = mc.ricker_projections(vals, scales, wavelet="g2")
        h = mc.singularity_map_point(T[0], mc.relative_scale(2.0, vals.shape))
        h[mc.no_support_mask(T).reshape(vals.shape)] = np.nan
        np.testing.assert_array_equal(res["h_map"], h)
        assert np.all(np.isnan(res["r2_map"]))
        np.testing.assert_array_equal(res["scales"], scales)


def test_punctual_flat_patches_are_nan_on_both_methods():
    """The fabricated-exponent guard, device edition (adversarial review, 2026-09-19): a
    nodata flat has no honest exponent under EITHER estimator. The measure route's flat is
    dust (caught by the function-level floor); the MULTIAFFINE route's flat carries a
    uniform ~1e-3-of-mean response at badly-sampled fine scales (the discrete kernel's
    mean-subtraction DC-couples every pixel to the whole image), which only the cross-scale
    mask catches -- the same mask that NaNs it under regression, so the two estimators
    agree on WHERE an exponent exists and differ only in HOW it is estimated."""
    rng = np.random.default_rng(0)
    vals = rng.normal(0.0, 1.0, (128, 128)).cumsum(axis=1)   # textured half
    vals[:, :64] = 3.7                                        # exactly flat half
    field = RasterField(name="flat", values=vals, frame=LocalFrame(),
                        x_axis=np.arange(128, dtype=np.float64),
                        y_axis=np.arange(128, dtype=np.float64))
    for dev in (HolderMeasure(), HolderMultiaffine()):
        for estimator in ("punctual", "regression"):
            res = dev.compute(field, dict(defaults_for(dev), estimator=estimator))
            assert np.isnan(res["h_map"][:, :48]).all(), (dev.name, estimator)
            assert np.isfinite(res["h_map"][:, 80:]).any(), (dev.name, estimator)


def test_frac_n_reaches_both_routes():
    field = _field()
    for dev in (HolderMeasure(), HolderMultiaffine()):
        base = dict(defaults_for(dev), wavelet="frac_gaussian")
        a = dev.compute(field, base)
        b = dev.compute(field, dict(base, frac_n=1.2))
        assert not np.array_equal(a["h_map"], b["h_map"]), dev.name


def test_cache_key_tracks_every_param():
    for dev in (HolderMeasure(), HolderMultiaffine()):
        base = defaults_for(dev)
        k0 = dev.cache_key("src", base)
        assert dev.cache_key("src", base) == k0
        for change in ({"estimator": "punctual"}, {"r_min": 2.0}, {"kappa": 12.0},
                       {"n_scales": 12}, {"wavelet": "frac_gaussian"}, {"q_tsallis": 2.0},
                       {"beta": 2.0}, {"frac_n": 1.3}):
            assert dev.cache_key("src", dict(base, **change)) != k0, (dev.name, change)
        assert dev.cache_key("other", base) != k0
    assert HolderMeasure().cache_key("src", defaults_for(HolderMeasure())) != \
        HolderMultiaffine().cache_key("src", defaults_for(HolderMultiaffine()))


def test_refusals_are_the_shared_contract():
    for dev in (HolderMeasure(), HolderMultiaffine()):
        with pytest.raises(ValueError, match="FIRST in its chain"):
            dev.compute({"chains": [], "extrema": []}, defaults_for(dev))
        vals = np.zeros((16, 16, 3))
        field = RasterField(name="v", values=vals, frame=LocalFrame(),
                            x_axis=np.arange(16, dtype=np.float64),
                            y_axis=np.arange(16, dtype=np.float64))
        with pytest.raises(ValueError, match="scalar 2-D"):
            dev.compute(field, defaults_for(dev))


def test_method_arrays_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="method"):
        method_arrays(np.zeros((8, 8)), "canonical", defaults_for(HolderMeasure()))


def test_q_ranges_are_method_true():
    """The positive q-Gaussian is a kernel for every q in [-1, 3]; the q-Mexican hat (Borges
    et al. 2004's parameterization, in 2-D) exists only for q < 2 -- so each tool gets its own
    q range."""
    measure = {p.name: p for p in HolderMeasure.params}
    multi = {p.name: p for p in HolderMultiaffine.params}
    assert (measure["q_tsallis"].min, measure["q_tsallis"].max) == (-1.0, 3.0)
    assert (multi["q_tsallis"].min, multi["q_tsallis"].max) == (-1.0, 1.95)
    assert measure["q_tsallis"].default == multi["q_tsallis"].default == 1.5


@pytest.mark.parametrize("q", [-1.0, 0.0, 0.5])
def test_holder_measure_runs_below_q_one(q):
    field = _field()
    res = HolderMeasure().compute(field, dict(defaults_for(HolderMeasure()),
                                              wavelet="q_gaussian", q_tsallis=q))
    assert np.isfinite(res["h_map"]).any()
    assert not np.isinf(res["h_map"]).any()


def test_q_knob_warns_where_the_kernel_misbehaves():
    """The q knob's live line (``derived_reading``): silent inside each kernel's clean range
    (an EMPTY string, so the label exists and can light up later), a warning where it stops
    being well-behaved, silent when the chosen wavelet does not use q, None for other knobs."""
    from dynamix.devices.band_recon import BandReconMeasure, BandReconMultiaffine

    for meas, multi in ((HolderMeasure(), HolderMultiaffine()),
                        (BandReconMeasure(), BandReconMultiaffine())):
        pm = dict(defaults_for(meas), wavelet="q_gaussian")
        pa = dict(defaults_for(multi), wavelet="q_mexican")
        assert meas.derived_reading("q_tsallis", 1.5, None, pm) == ""
        assert meas.derived_reading("q_tsallis", -1.0, None, pm) == ""   # compact: still fine
        assert "⚠" in meas.derived_reading("q_tsallis", 2.5, None, pm)
        assert multi.derived_reading("q_tsallis", 0.0, None, pa) == ""
        assert "⚠" in multi.derived_reading("q_tsallis", -0.5, None, pa)
        assert multi.derived_reading("q_tsallis", -0.5, None, dict(pa, wavelet="g2")) == ""
        assert meas.derived_reading("r_min", 1.0, None, pm) is None
