# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The multiaffine tools' resolution floor on r1 (a warning, never a refusal).

Turiel 2008 (figure 2, section 4.2.1): a discretised wavelet must separate its positive and
negative parts, so its zero crossings set the minimum attainable resolution, and more crossings
cost resolution. Turiel 2009 puts the Mexican hat's at "several pixels". The tools read it as
every sign lobe at least 2 px wide at r1. The central disc counts by its diameter, so a wavelet
with one crossing at r keeps the floor r1 = 1 these tools always used. A compact support's edge
closes the last lobe although it is not a sign change: that is what raises the q-Mexican hat's
floor below q = 1.
"""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import microcanonical as mc
from dynamix.devices import holder_methods as hm

_BASE = {"estimator": "regression", "r_min": 1.0, "kappa": 8.0, "n_scales": 6, "beta": 1.0,
         "frac_n": 2.0, "q_tsallis": 1.5, "q_pairing": "fixed", "q_beta": 0.5}


def _floor(**kw):
    return hm.lobe_floor("multiaffine", {**_BASE, **kw})


@pytest.mark.parametrize("kw", [dict(wavelet="g1"), dict(wavelet="g2"),
                                dict(wavelet="lorentzian_marr", beta=3.0),
                                dict(wavelet="frac_gaussian", frac_n=1.5),
                                dict(wavelet="q_mexican", q_tsallis=1.5, q_pairing="q-paired"),
                                dict(wavelet="q_mexican", q_tsallis=0.9, q_pairing="q-paired")])
def test_one_crossing_at_r_keeps_the_floor_of_one_pixel(kw):
    assert _floor(**kw) == pytest.approx(1.0, rel=1e-6)


def test_more_crossings_raise_the_floor():
    assert _floor(wavelet="g3") == pytest.approx(2.0 / 1.517, rel=1e-3)            # ring 1.517 r
    assert _floor(wavelet="frac_gaussian", frac_n=5.0) == pytest.approx(2.0 / 1.372, rel=1e-3)


@pytest.mark.parametrize("q, pairing, beta, floor", [
    (0.5, "q-paired", 0.5, 2.0 / (np.sqrt(3.0) - 1.0)),                 # ring to the edge
    (0.1, "q-paired", 0.5, 2.0 / (np.sqrt(1.9 / 0.9) - 1.0)),
    (0.5, "fixed", 0.5, 2.0 * np.sqrt(1.5) / (np.sqrt(3.0) - 1.0)),     # fixed beta dilates it
    (0.5, "fixed", 1.0, 2.0 * np.sqrt(3.0) / (np.sqrt(3.0) - 1.0)),
    (1.5, "fixed", 0.5, np.sqrt(0.5)),                                  # crossing moved out
])
def test_the_q_mexican_floor_follows_q_and_the_width(q, pairing, beta, floor):
    assert _floor(wavelet="q_mexican", q_tsallis=q, q_pairing=pairing,
                  q_beta=beta) == pytest.approx(floor, rel=1e-9)


@pytest.mark.parametrize("q, beta", [(0.5, None), (0.2, 0.5), (0.7, 1.3), (1.4, 0.5)])
def test_the_q_mexican_closed_form_is_the_kernel_itself(q, beta):
    """The floor's crossing and edge, read off the kernel the projections sample."""
    rho = np.linspace(0.0, 12.0, 1_200_001)
    k = mc._marr_kernel(rho ** 2, 1.0, "q_gaussian", 1.0, mc._borges_q(q), 2.0, None,
                        stretch=mc._q_mexican_stretch(q, beta))
    nz = np.flatnonzero(k)
    bounds = list(rho[nz][np.flatnonzero(np.diff(np.sign(k[nz])))])
    if nz[-1] < rho.size - 1:
        bounds.append(rho[nz[-1]])                                      # the compact edge
    widths = [2.0 * bounds[0]] + list(np.diff(bounds))
    pairing = "q-paired" if beta is None else "fixed"
    assert _floor(wavelet="q_mexican", q_tsallis=q, q_pairing=pairing,
                  q_beta=beta or 0.5) == pytest.approx(2.0 / min(widths), rel=1e-4)


def test_the_warning_line_lights_below_the_floor_only():
    p = {**_BASE, "wavelet": "q_mexican", "q_tsallis": 0.5, "q_pairing": "q-paired"}
    assert hm.knob_warning("multiaffine", "r_min", 2.5, p).startswith("⚠")
    assert hm.knob_warning("multiaffine", "r_min", 3.0, p) == ""
    assert hm.knob_warning("multiaffine", "r_min", 1.0, {**p, "wavelet": "g2"}) == ""
    # empty, never None, where no floor applies: the label must exist to light up later
    assert hm.knob_warning("measure", "r_min", 0.2, {**p, "wavelet": "gaussian"}) == ""
    assert hm.knob_warning("multiaffine", "q_tsallis", -0.5, p) == \
        hm.q_warning("multiaffine", "q_tsallis", -0.5, p)               # q's own line is kept


# ------------------------------------------- the measure route: the compact q-Gaussian only

def _mfloor(**kw):
    return hm.lobe_floor("measure", {**_BASE, "wavelet": "q_gaussian", **kw})


@pytest.mark.parametrize("q, pairing, width, floor", [
    (0.0, "fixed", 0.5, np.sqrt(0.5)),        # support radius r / sqrt(w (1 - q)), 2 px across
    (-1.0, "fixed", 0.5, 1.0),
    (0.0, "fixed", 5.0, np.sqrt(5.0)),
    (0.0, "q-paired", 0.5, 0.5),             # paired width 1/(2(2 - q)) = 1/4 at q = 0
])
def test_the_compact_q_gaussian_needs_its_support_two_pixels_across(q, pairing, width, floor):
    assert _mfloor(q_tsallis=q, q_pairing=pairing, q_beta=width) == pytest.approx(floor)


@pytest.mark.parametrize("q, width", [(0.0, 0.5), (0.5, 2.0), (-0.5, 1.0)])
def test_the_measure_closed_form_is_the_kernel_itself(q, width):
    """The support radius, read off the kernel the projections sample (r = 20, so a pixel
    resolves it to ~5 %)."""
    n, r = 401, 20.0
    k = np.fft.fftshift(mc._radial_kernel((n, n), r, "q_gaussian", 1.0, q, 2.0, q_beta=width))
    y = np.arange(n) - n // 2
    edge = np.hypot(*np.meshgrid(y, y))[k > 0].max()
    assert _mfloor(q_tsallis=q, q_beta=width) == pytest.approx(2.0 / (2.0 * edge / r), rel=0.05)


@pytest.mark.parametrize("kw", [dict(wavelet="gaussian"), dict(wavelet="lorentzian"),
                                dict(wavelet="frac_gaussian"),
                                dict(wavelet="q_gaussian", q_tsallis=1.5)])
def test_other_positive_kernels_have_no_floor(kw):
    assert hm.lobe_floor("measure", {**_BASE, **kw}) is None
    assert hm.knob_warning("measure", "r_min", 0.1, {**_BASE, **kw}) == ""


@pytest.fixture
def builtins(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return clean_registry


def test_the_tools_show_it_on_r1(builtins):
    from dynamix.model.device import defaults_for, get_device

    for name in ("holder_multiaffine", "band_recon_multiaffine"):
        dev = get_device(name)
        p = {**defaults_for(dev), "wavelet": "g3"}
        assert dev.derived_reading("r_min", 1.0, None, p).startswith("⚠")
        assert dev.derived_reading("r_min", 1.4, None, p) == ""
    for name in ("holder_measure", "band_recon_measure"):
        dev = get_device(name)
        p = {**defaults_for(dev), "wavelet": "q_gaussian", "q_tsallis": 0.0, "q_beta": 5.0}
        assert dev.derived_reading("r_min", 1.0, None, p).startswith("⚠")
        assert dev.derived_reading("r_min", 2.5, None, p) == ""
        assert dev.derived_reading("r_min", 1.0, None, defaults_for(dev)) == ""
