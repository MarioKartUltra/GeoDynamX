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


_MEASURE_DEFAULTS = {"beta": 1.0, "q_tsallis": 1.5, "frac_n": 2.0, "q_pairing": "fixed",
                     "q_beta": 0.5}


@pytest.mark.parametrize("wavelet,kw,c", [
    ("gaussian", {}, np.pi * np.sqrt(2)),
    ("lorentzian", {"beta": 2.0}, 4.047),
    ("lorentzian", {"beta": 1.5}, 6.283),
    ("q_gaussian", {"q_tsallis": 1.5}, 8.09),
    ("frac_gaussian", {"frac_n": 4.0}, 2.69)])
def test_measure_route_lambda_constants(wavelet, kw, c):
    p = wp.measure_profile(wavelet, 6.0, **{**_MEASURE_DEFAULTS, **kw})
    assert p.lambda_px / 6.0 == pytest.approx(c, rel=0.01)
    assert np.all(p.psi >= -1e-9) and p.psi[np.argmin(np.abs(p.x_px))] == pytest.approx(1.0)


@pytest.mark.parametrize("wavelet,kw,c", [
    ("g2", {}, np.pi), ("g1", {}, 3.535), ("g3", {}, 2.987),
    ("q_mexican", {"q_tsallis": 1.5, "q_pairing": "q-paired"}, 4.047),
    ("lorentzian_marr", {"beta": 2.5}, 3.636)])
def test_multiaffine_route_lambda_constants(wavelet, kw, c):
    p = wp.multiaffine_profile(wavelet, 6.0, **{**_MEASURE_DEFAULTS, **kw})
    assert p.lambda_px / 6.0 == pytest.approx(c, rel=0.01)


def test_q_mexican_fixed_and_paired_beta_differ_and_meet_at_q_one():
    paired = wp.multiaffine_profile("q_mexican", 6.0, **{**_MEASURE_DEFAULTS, "q_pairing": "q-paired"})
    fixed = wp.multiaffine_profile("q_mexican", 6.0, **_MEASURE_DEFAULTS)
    assert fixed.lambda_px != pytest.approx(paired.lambda_px, rel=0.02)
    assert not np.allclose(np.interp(fixed.x_px, paired.x_px, paired.psi), fixed.psi, atol=1e-3)
    one = wp.multiaffine_profile("q_mexican", 6.0, **{**_MEASURE_DEFAULTS, "q_tsallis": 1.0})
    assert one.lambda_px / 6.0 == pytest.approx(np.pi, rel=0.01)


def test_a_kernel_without_a_band_has_no_lambda_and_a_broad_one_is_flagged():
    none = wp.measure_profile("lorentzian", 6.0, **{**_MEASURE_DEFAULTS, "beta": 1.0})
    assert np.isnan(none.lambda_px) and none.psi.size
    broad = wp.measure_profile("lorentzian", 6.0, **{**_MEASURE_DEFAULTS, "beta": 1.2})
    assert broad.broad and broad.lambda_px / 6.0 == pytest.approx(11.29, rel=0.01)
    assert not wp.measure_profile("gaussian", 6.0, **_MEASURE_DEFAULTS).broad


def test_the_devices_route_their_own_params(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    res = {"scales": np.array([6.0])}
    hm = wp.profile_for("holder_measure", {"wavelet": "q_gaussian", **_MEASURE_DEFAULTS}, res, 0)
    assert hm.lambda_px / 6.0 == pytest.approx(8.09, rel=0.01)
    ma = wp.profile_for("holder_multiaffine", {"wavelet": "g2", **_MEASURE_DEFAULTS}, res, 0)
    assert ma.lambda_px / 6.0 == pytest.approx(np.pi, rel=0.01)
    hmap = wp.profile_for("holder_map", {"wavelet": "gaussian", "beta": 1.0, "q_tsallis": 1.5},
                          res, 0)
    assert hmap.lambda_px / 6.0 == pytest.approx(np.pi * np.sqrt(2), rel=0.01)
