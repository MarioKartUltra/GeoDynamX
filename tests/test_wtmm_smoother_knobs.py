# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The WTMM devices' smoother knob: the registry's names, their parameters, the readings, and
the same scaling exponent from every smoother (the smoothers design spec, §9)."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import fbm2d

from dynamix.core.rasterfield import RasterField
from dynamix.devices.wtmm import WTMM2D, smoother_params
from dynamix.model.device import defaults_for


def _param(dev, name):
    return next(p for p in dev.params if p.name == name)


def test_the_knob_offers_the_registry_with_its_parameters_active_beside_their_smoother():
    dev = WTMM2D()
    w = _param(dev, "wavelet")
    assert w.choices == ("mexican", "gaussian", "frac_bspline", "q_gaussian")
    assert w.choice_labels[2:] == ("fractional B-spline (ψ = ∂θ_α)", "q-Gaussian (ψ = ∂θ_q)")
    a, q = _param(dev, "alpha"), _param(dev, "q_tsallis")
    assert (a.default, a.min, a.max, a.soft_min) == (3.0, 1.0, 8.0, 2.0)
    assert a.active_when == ("wavelet", ("frac_bspline",))
    assert (q.default, q.min, q.max) == (1.5, 0.05, 1.95)
    assert q.active_when == ("wavelet", ("q_gaussian",))
    assert not a.view and not q.view                              # keyed: they change the transform
    assert "alpha" in dev.param_groups["PRE"] and "q_tsallis" in dev.param_groups["PRE"]
    d = defaults_for(dev)
    assert smoother_params(d) is None
    assert smoother_params(dict(d, wavelet="frac_bspline", alpha=2.5)) == {"alpha": 2.5}
    assert smoother_params(dict(d, wavelet="q_gaussian", q_tsallis=1.2)) == {"q": 1.2}
    from dynamix.devices.wtmm_roi import WTMM2DROI
    roi = WTMM2DROI()
    assert _param(roi, "wavelet").choices == w.choices
    assert _param(roi, "alpha").active_when == a.active_when
    assert _param(roi, "q_tsallis").active_when == q.active_when


def test_the_bar_and_the_reading_follow_the_knob():
    dev = WTMM2D()
    d = defaults_for(dev)
    p = dev.wavelet_profile(dict(d, wavelet="frac_bspline", alpha=3.0), 8.0, 0)
    assert "∂θ_α" in p.family and "α = 3" in p.family
    g = dev.wavelet_profile(dict(d, wavelet="gaussian"), 8.0, 0)
    assert "g1" in g.family
    field = RasterField._from_bare_array(np.zeros((8, 8)), "f", name="f")
    text = dev.derived_reading("a_min", 1.0, field, dict(d, wavelet="q_gaussian", q_tsallis=1.5))
    plain = dev.derived_reading("a_min", 1.0, field, dict(d, wavelet="gaussian"))
    assert "λ" in text and text != plain                          # q = 1.5 reads a longer λ*


def test_the_preview_and_the_level_outputs_take_the_knob():
    dev = WTMM2D()
    d = dict(defaults_for(dev), n_oct=1, n_voice=2, wavelet="frac_bspline")
    field = RasterField._from_bare_array(fbm2d(64, 0.6, seed=4), "f", name="f")
    pre = dev.preview(field, dict(d, alpha=3.0))
    assert len(pre["extrema"]) == 1
    res = dev.compute(field, dict(d, alpha=3.0))
    one = dev.compute_output("modulus", field.values, res, dict(d, alpha=3.0), fetch=None, level=1)
    two = dev.compute_output("modulus", field.values, res, dict(d, alpha=1.5), fetch=None, level=1)
    assert one["raster"].shape == (64, 64) and not np.allclose(one["raster"], two["raster"])


@pytest.mark.parametrize("wavelet,knob", [("frac_bspline", {"alpha": 3.0}),
                                          ("q_gaussian", {"q_tsallis": 1.5})])
def test_an_fbm_reads_the_same_h2_with_every_smoother(wavelet, knob):
    """Spec §9: on fBm H = 0.5 each smoother's h(2) lies within 0.1 of the Gaussian smoother's
    reading on the same field (the convention offset is the same for all)."""
    from dynamix.core import spectra

    dev = WTMM2D()
    field = RasterField._from_bare_array(fbm2d(128, 0.5, seed=3), "fbm", name="fbm")
    base = dict(defaults_for(dev), n_oct=3, n_voice=4)

    def h2(params):
        hd = dev.compute(field, params)["hd_std"]
        fit = spectra.fit_spectra(hd, float(hd["log2_scales"][0]), float(hd["log2_scales"][-1]))
        q = np.asarray(hd["q_list"], dtype=np.float64)
        return float(fit["h"][int(np.argmin(np.abs(q - 2.0)))])

    ref = h2(dict(base, wavelet="gaussian"))
    assert np.isfinite(ref)
    assert h2(dict(base, wavelet=wavelet, **knob)) == pytest.approx(ref, abs=0.1)
