# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The Wavelet ridge tool (spec 19 §3.1, Passot and Newell 1994 App. A) standing alone: the
ridge's scale-free rasters, the directional transform under the Scale slider as per-level lazy
outputs, the q-Morlet's section and λ* = 2πa/ω for the scale bar."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dynamix.core import analytic_cwt as acwt
from dynamix.core.frames import LocalFrame
from dynamix.core.rasterfield import RasterField
from dynamix.model.contracts import COMPONENTS, FIELD
from dynamix.model.device import defaults_for, get_device, keyed_params, validate_params


@pytest.fixture
def dev(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    return get_device("wavelet_ridge")


def _field(values):
    values = np.asarray(values, dtype=np.float64)
    return RasterField(name="f", values=values, frame=LocalFrame(units="px"),
                       x_axis=np.arange(float(values.shape[1])), y_axis=np.arange(float(values.shape[0])))


def _wave(n, p, q):
    return (2.0 * math.pi * p / n, 2.0 * math.pi * q / n)


def _stripes(n, k, amp=1.0, phase=0.0):
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    return amp * np.cos(k[0] * x + k[1] * y + phase)


def test_the_tool_sits_in_patterns_with_per_level_outputs_and_the_papers_defaults(dev):
    assert dev.label == "Wavelet ridge (Passot–Newell)" and dev.families == ("Patterns",)
    assert dev.consumes == ((FIELD,),) and COMPONENTS in dev.produces
    assert [(o.name, o.lazy, o.per_level) for o in dev.outputs] == [
        ("modulus", True, True), ("orientation", True, True), ("phase", True, True)]
    p = validate_params(dev, {})
    assert (p["k0"], p["omega"], p["anisotropy"], p["ridge_q"], p["q_pairing"], p["orientations"]) == (
        "spectral_peak", acwt.OMEGA, 1.0, 1.0, "fixed", 12)
    assert (p["n_oct"], p["n_voice"], p["a_min"]) == (3, 4, 2.0)
    assert "show" not in keyed_params(dev, p) and {"omega", "ridge_q", "n_oct"} <= set(keyed_params(dev, p))
    assert dev.roi_margin(p) == math.ceil(4.0 * 2.0 * 2 ** 3)


def test_stripes_give_their_amplitude_wavevector_and_direction_and_a_ladder(dev):
    n = 192
    k = _wave(n, 6, 5)
    p = validate_params(dev, {"periodic": True})
    res = dev.compute(_field(_stripes(n, k, amp=1.7, phase=0.3)), p)
    assert res["k0"] == pytest.approx(math.hypot(*k), rel=0.05)         # the spectral peak's bin
    assert res["converged"].all()
    assert np.allclose(res["ridge_amplitude"], 1.7, rtol=1e-5)
    assert np.allclose(res["ridge_wavenumber"], math.hypot(*k), rtol=1e-5)
    assert np.allclose(res["ridge_direction"], math.degrees(math.atan2(k[1], k[0])), atol=1e-3)
    assert np.allclose(res["ridge_scale"], acwt.OMEGA / math.hypot(*k), rtol=1e-5)
    assert res["raster_out"] is res["ridge_amplitude"] and res["_shape"] == (n, n)
    assert res["scales"].size == 3 * 4 + 1 and res["scales"][0] == 2.0
    rec = res["record"]
    assert rec["method"] == "wavelet_ridge" and rec["k0"]["method"] == "spectral_peak"
    assert rec["ladder"] == {"n_oct": 3, "n_voice": 4, "a_min": 2.0, "n_scales": 13, "orientations": 12}
    assert res["warnings"] == [] and "converged 100%" in dev.reading(res, p)
    assert dev.view(res, {"show": "ridge_direction"})["raster_out"] is res["ridge_direction"]
    assert dev.view(res, {"show": "modulus"})["raster_out"] is res["ridge_amplitude"]   # lazy: the engine's


def test_the_per_level_transform_peaks_at_the_matched_scale_and_orientation(dev):
    """At the ladder's level nearest a = ω/|k| the modulus over orientations reads up to A/2
    (the analytic ½), the winning orientation is the wave's within one step of the bank, and
    the phase there is the wave's; far from that scale the modulus is small."""
    n = 128
    k = _wave(n, 8, 0)                                  # λ = 16 px, a_match = ω/k = 6.1 px at ω = 3
    u = _stripes(n, k, amp=2.0, phase=0.4)
    p = validate_params(dev, {"periodic": True, "k0": "manual", "lambda0": 16.0})
    res = dev.compute(_field(u), p)
    scales = res["scales"]
    best = int(np.argmin(np.abs(scales - acwt.OMEGA / k[0])))
    out = dev.compute_output("modulus", u, res, p, fetch=None, level=best)
    assert out["diag"] == {"level": best, "scale_px": float(scales[best]),
                           "lambda_px": 2.0 * math.pi * float(scales[best]) / acwt.OMEGA}
    assert out["raster"].dtype == np.float32
    assert 0.6 < float(np.median(out["raster"])) <= 1.0 + 1e-6      # ≤ A/2, near it at the match
    theta = dev.compute_output("orientation", u, res, p, fetch=None, level=best)["raster"]
    assert np.all(np.minimum(theta % 180, 180 - theta % 180) <= 180 / 12 / 2 + 1e-6)
    phase = dev.compute_output("phase", u, res, p, fetch=None, level=best)["raster"]
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    assert np.allclose(np.angle(np.exp(1j * (phase - k[0] * x - 0.4))), 0.0, atol=0.05)
    far = dev.compute_output("modulus", u, res, p, fetch=None, level=scales.size - 1)["raster"]
    assert float(np.median(far)) < 0.1
    with pytest.raises(ValueError, match="no lazy output"):
        dev.compute_output("edges", u, res, p, fetch=None)


def test_the_profile_is_the_q_morlets_section_and_lambda_star_is_the_carriers_wavelength(dev):
    p = validate_params(dev, {"omega": 6.0})
    prof = dev.wavelet_profile(p, 10.0, 2)
    assert prof.lambda_px == pytest.approx(2.0 * math.pi * 10.0 / 6.0)
    assert prof.scale_px == 10.0 and np.max(np.abs(prof.psi)) == pytest.approx(1.0)
    assert prof.family.startswith("q-Morlet") and "ω = 6" in prof.family
    i0 = np.argmin(np.abs(prof.x_px))
    assert prof.psi[i0] == pytest.approx(1.0)                         # cos ω·0 under a unit envelope
    from dynamix.core.wavelet_profiles import lambda_px

    assert lambda_px("wavelet_ridge", p, 10.0, 2) == pytest.approx(prof.lambda_px)


def test_a_periodic_extension_stamp_counts_as_periodic(dev, monkeypatch):
    n = 96
    k = _wave(n, 6, 0)
    field = RasterField(name="f", values=_stripes(n, k), frame=LocalFrame(units="px"),
                        x_axis=np.arange(float(n)), y_axis=np.arange(float(n)),
                        provenance={"edges.filled": {"rows": n, "cols": n}})
    res = dev.compute(field, validate_params(dev, {"k0": "manual", "lambda0": 16.0}))
    assert res["record"]["reach_px"] == 0                              # no reflection: periodic
    seen = []

    def transformed(values, kappa, *, periodic=False, **wavelet):
        seen.append(periodic)
        return np.ones(values.shape, dtype=complex)

    monkeypatch.setattr(acwt, "transform", transformed)
    p = validate_params(dev, {"k0": "manual", "lambda0": 16.0})
    dev.compute_output("modulus", field.values, res, p, fetch=None, level=0)
    assert seen and all(seen)
