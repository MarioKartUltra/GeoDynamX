# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
import numpy as np
import pytest

from conftest import make_field
from dynamix.core import fft_policy
from dynamix.core.fractional import integrate
from dynamix.devices.fractional_integration import integrated_field
from dynamix.devices.wtmm import WTMM2D
from dynamix.model.device import defaults_for


@pytest.mark.parametrize("operator", ["bessel", "riesz"])
def test_fractional_operator_has_the_analytic_gain_preserves_dc_and_inverts(operator):
    fft_policy.configure("pyfftw", 64)
    x = 3+np.broadcast_to(np.cos(2*np.pi*np.arange(64)/8), (31, 64))
    s, length = .7, 10.
    z = (2*np.pi/8*length)**2
    gain = (1+z)**(-s/2) if operator == "bessel" else z**(-s/2)
    y = integrate(x, s, operator=operator, length_px=length, edge="periodic")
    np.testing.assert_allclose(y, 3+gain*(x-3), atol=1e-13)
    restored = integrate(y, -s, operator=operator, length_px=length, edge="periodic")
    np.testing.assert_allclose(restored, x, atol=1e-12)


def test_wtmm_input_integration_and_coefficient_lift_are_separate_and_cacheable():
    from dynamix.core.wtmm_backend import run_wtmm2d
    from dynamix.engine.cache import StageStore

    fft_policy.configure("pyfftw", 64)
    field = make_field(shape=(48, 48))
    dev = WTMM2D()
    p = {**defaults_for(dev), "n_oct": 2, "n_voice": 2, "integration_order": .4,
         "fracint_alpha": .3}
    store = StageStore()
    got = dev.compute(field, p, stage_cache=store)
    prepared = integrated_field(field, p, edge="mirror")
    # The backend has only the pseudo-lift; input integration has already
    # happened before it sees the field.
    fresh = run_wtmm2d(prepared, got["params"])
    np.testing.assert_allclose(got["hd_std"]["tau_qa"], fresh["hd_std"]["tau_qa"], equal_nan=True)
    assert got["_integration_order"] == .4 and got["params"]["fracint_alpha"] == .3
    again = dev.compute(field, {**p, "q_step": 1.}, stage_cache=store)
    assert "cwt" in again["cache_hits"]
    # A lazy modulus must show the integrated field, not the raw field.
    out = dev.compute_output("modulus", field.values, got, p, fetch=None, level=0)
    from dynamix.core.wtmm_levels import run_wtmm2d_level
    keys = ("n_oct", "n_voice", "a_min", "wavelet", "fracint_alpha", "detector", "pad", "pad_mode")
    expected = run_wtmm2d_level(prepared.values, {k: got["params"][k] for k in keys}, 0)
    np.testing.assert_allclose(out["raster"], expected["mod"], rtol=1e-6, atol=1e-7)


def test_input_integration_is_not_silently_applied_to_a_finite_roi_halo():
    dev = WTMM2D()
    p = {**defaults_for(dev), "integration_order": .5}
    with pytest.raises(ValueError, match="global"):
        dev.compute_roi(make_field(), (4, 4, 16, 16), p)
