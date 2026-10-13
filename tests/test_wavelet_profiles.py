# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Every wavelet's section and its λ* (the bandpass peak wavelength), one authority."""
from __future__ import annotations

import numpy as np
import pytest

from dynamix.core import wavelet_profiles as wp


def test_format_length_follows_the_notebooks_three_tiers():
    assert wp.format_length(480.0, "m") == "480 m"
    assert wp.format_length(52.04, "μm") == "52.0 μm"
    assert wp.format_length(3.456, "m") == "3.46 m"
    assert wp.format_length(11.3, "px") == "11.3 px"


def test_physical_goes_through_px_to_metres_or_stays_in_pixels():
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField
    from dynamix.shell.units import physical

    f = RasterField(name="f", values=np.zeros((4, 4)),
                    frame=LocalFrame(x0=0.0, y0=0.0, dx=10.0, dy=10.0, units="metre"),
                    x_axis=10.0 * np.arange(4), y_axis=10.0 * np.arange(4))
    value, unit = physical(11.3, f)
    assert (value, unit) == (pytest.approx(113.0), "metre")
    assert physical(11.3, np.zeros((4, 4))) == (pytest.approx(11.3), "px")


def test_wtmm_profile_is_psi_with_lambda_from_scale_units(clean_registry):
    from dynamix.core.scale_units import lambda_peak_px
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    res = {"scales": np.array([4.0, 8.0, 16.0]), "_scale_idx": 1}
    p = wp.profile_for("wtmm2d", {"wavelet": "gaussian"}, res, 0)
    assert p.scale_px == 8.0
    assert p.lambda_px == pytest.approx(lambda_peak_px(8.0, "gaussian", 1))
    assert p.lambda_px == pytest.approx(8.0 * np.sqrt(2.0))
    assert np.max(np.abs(p.psi)) == pytest.approx(1.0) and p.x_px[0] < 0 < p.x_px[-1]
    assert np.allclose(p.psi, -p.psi[::-1], atol=1e-9)                        # g1 is odd
    assert "g1" in p.family and not p.off_ladder and not p.broad
    m = wp.profile_for("wtmm2d", {"wavelet": "mexican"}, res, 0)
    assert m.lambda_px == pytest.approx(8.0 * np.sqrt(2.0 / 3.0))
    assert wp.lambda_px("wtmm2d", {"wavelet": "mexican"}, 8.0) == pytest.approx(m.lambda_px)


def test_profile_for_is_none_without_a_wavelet(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    assert wp.profile_for("ssa2d", {}, {"scales": np.array([2.0])}, 0) is None
    assert wp.profile_for("hline_length", {}, {}, 0) is None
    assert wp.profile_for("wtmm2d", {"wavelet": "gaussian"}, {"scales": np.array([2.0])}, 5) is None
