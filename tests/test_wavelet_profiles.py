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
    hmap = wp.profile_for("holder_map", {"wavelet": "gaussian", "beta": 1.0, "q_tsallis": 1.5,
                                         "estimator": "measure"}, res, 0)
    assert hmap.lambda_px / 6.0 == pytest.approx(np.pi * np.sqrt(2), rel=0.01)


def test_mz_levels_converge_to_the_continuum_constant_and_fine_levels_are_off_ladder():
    cs = [wp.mz_profile(j).lambda_px / 2.0 ** j for j in range(1, 7)]
    assert cs[0] == pytest.approx(1.0, rel=0.02) and cs[5] == pytest.approx(1.8595, rel=0.01)
    assert wp.mz_profile(1).off_ladder and wp.mz_profile(2).off_ladder
    assert not wp.mz_profile(3).off_ladder


def test_cdf_profiles_are_the_ricker_or_the_gaussian_derivative_at_sigma():
    marr = wp.cdf_profile("marr", 4.0)
    assert marr.lambda_px == pytest.approx(np.pi * np.sqrt(2) * 4.0, rel=1e-3)
    assert marr.psi[np.argmin(np.abs(marr.x_px))] == pytest.approx(1.0)
    mz = wp.cdf_profile("mz", 4.0)
    assert mz.lambda_px == pytest.approx(2 * np.pi * 4.0, rel=1e-3)
    assert np.allclose(mz.psi, -mz.psi[::-1], atol=1e-9)


def test_dwt_bank_levels_scale_by_two_and_level_one_is_off_ladder():
    p2, p3 = wp.dwt_profile("sym7", 2), wp.dwt_profile("sym7", 3)
    assert p3.lambda_px == pytest.approx(2 * p2.lambda_px, rel=0.02)
    assert p3.lambda_px / 8.0 == pytest.approx(1.46, rel=0.02)
    assert wp.dwt_profile("sym7", 1).off_ladder and not p3.off_ladder


def test_tang_you_profile_is_the_gradient_of_theta():
    p = wp.tang_you_profile("tang_you", 6.0)
    assert np.allclose(p.psi, -p.psi[::-1], atol=1e-6) and np.isfinite(p.lambda_px)
    assert 2.0 < p.lambda_px / 6.0 < 8.0


def test_the_dyadic_and_skeleton_devices_route_their_levels(clean_registry):
    from dynamix.devices import register_builtin_devices

    register_builtin_devices()
    mz = wp.profile_for("mz_edges", {}, {"scales": np.array([2.0, 4.0, 8.0]), "_scale_idx": 2}, 0)
    assert mz.lambda_px == pytest.approx(wp.mz_profile(3).lambda_px)
    dwt = wp.profile_for("dwt2d", {"wavelet": "sym7"}, {"scales": np.array([2.0, 4.0, 8.0])}, 2)
    assert dwt.lambda_px == pytest.approx(wp.dwt_profile("sym7", 3).lambda_px)
    cdf = wp.profile_for("cdf_edges", {"edge_mode": "marr"}, {"scales": np.array([1.0, 2.0, 4.0])}, 2)
    assert cdf.lambda_px == pytest.approx(np.pi * np.sqrt(2) * 4.0, rel=1e-3)
    sk = wp.profile_for("wavelet_skeleton", {"wavelet": "tang_you"}, {"scales": np.array([6.0])}, 0)
    assert sk.lambda_px == pytest.approx(wp.tang_you_profile("tang_you", 6.0).lambda_px)


def test_a_plane_wave_peaks_where_lambda_reads_its_wavelength(clean_registry):
    """The acceptance of the spec (§9): the scale with the largest mean WTMM modulus has λ* = L."""
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device

    register_builtin_devices()
    L = 24.0
    y, x = np.mgrid[0:128, 0:128]
    from dynamix.core.rasterfield import RasterField

    wave = np.cos(2 * np.pi * x / L) + 0.05 * np.cos(2 * np.pi * y / 61.0)
    field = RasterField._from_bare_array(wave, "wave", name="wave")
    dev = get_device("wtmm2d")
    params = {**defaults_for(dev), "wavelet": "gaussian", "n_oct": 4, "n_voice": 8,
              "fracint_alpha": 0.0}
    res = dev.compute(field, params)
    strengths = [float(np.mean(np.asarray(ext["mod"]))) if len(ext["mod"]) else 0.0
                 for ext in res["extrema"]]
    a = float(np.asarray(res["scales"])[int(np.argmax(strengths))])
    assert wp.lambda_px("wtmm2d", params, a) == pytest.approx(L, rel=2 ** (1 / 8) - 1 + 0.05)


_MICRO = ("holder_measure", "holder_multiaffine", "holder_map", "band_recon_measure",
          "band_recon_multiaffine")


def test_microcanonical_devices_expose_their_scale_ladder(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device

    register_builtin_devices()
    for name in _MICRO:
        dev = get_device(name)
        params = {**defaults_for(dev), "r_min": 2.0, "kappa": 8.0, "n_scales": 4}
        np.testing.assert_allclose(dev.scale_ladder(params), np.geomspace(2.0, 16.0, 4))
    assert not hasattr(get_device("wtmm2d"), "scale_ladder")


def test_holder_map_profile_follows_its_estimator(clean_registry):
    from dynamix.devices import register_builtin_devices
    from dynamix.model.device import defaults_for, get_device

    register_builtin_devices()
    dev = get_device("holder_map")
    base = {**defaults_for(dev), "wavelet": "gaussian"}
    measure = dev.wavelet_profile({**base, "estimator": "measure"}, 6.0, 0)
    multi = dev.wavelet_profile({**base, "estimator": "multiaffine"}, 6.0, 0)
    assert measure.lambda_px / 6.0 == pytest.approx(np.pi * np.sqrt(2), rel=0.01)
    assert multi.lambda_px / 6.0 == pytest.approx(np.pi, rel=0.01)
    assert measure.psi.min() >= -1e-9 and multi.psi.min() < -0.05      # a kernel, a hat


def test_the_mz_section_is_the_continuum_quadratic_spline_at_the_levels_scale():
    """The bar draws the Mallat-Zhong wavelet itself -- psi = theta' with theta the cubic-spline
    smoother, support |x| <= 2^level, antisymmetric, positive left of centre like g1 -- sampled
    finely even at level 1, where the a trous filter has two taps; lambda* stays the level's."""
    for level in (1, 3):
        p = wp.mz_profile(level)
        s = 2.0 ** level
        assert p.x_px.size >= 65 and p.x_px.min() == pytest.approx(-s) and p.x_px.max() == pytest.approx(s)
        i = np.argmax(p.psi)
        assert p.x_px[i] < 0 and p.psi[i] == pytest.approx(1.0)
        np.testing.assert_allclose(p.psi, -p.psi[::-1], atol=1e-12)            # antisymmetric
        assert p.psi[np.argmin(np.abs(p.x_px))] == pytest.approx(0.0, abs=1e-12)
        assert p.psi[0] == pytest.approx(0.0, abs=1e-12)                        # compact support
    assert wp.mz_profile(1).lambda_px == pytest.approx(2.0, rel=0.02)           # the level's own
    assert wp.mz_profile(1).off_ladder and not wp.mz_profile(3).off_ladder


def test_the_measure_and_multiaffine_sections_are_sampled_finely_at_the_smallest_scales():
    """The bar draws the kernel's continuous section, sampled at r/16 at least (never at the
    pixel step, which at r = 1 px draws a kinked polyline): the Gaussian reads exp(-x^2/2) at
    half a pixel, the Ricker crosses zero at r."""
    p = wp.measure_profile("gaussian", 1.0)
    assert np.diff(p.x_px).max() <= 1.0 / 16.0 + 1e-12
    i = np.argmin(np.abs(p.x_px - 0.5))
    assert p.psi[i] == pytest.approx(np.exp(-0.125), abs=2e-3)
    big = wp.measure_profile("gaussian", 64.0)
    assert np.diff(big.x_px).max() <= 4.0 + 1e-12 and big.x_px.size <= 8193
    m = wp.multiaffine_profile("g2", 1.0)
    assert np.diff(m.x_px).max() <= 1.0 / 16.0 + 1e-12
    assert m.psi[np.argmin(np.abs(m.x_px - 1.0))] == pytest.approx(0.0, abs=5e-2)
    assert m.psi[np.argmin(np.abs(m.x_px))] == pytest.approx(1.0)


def test_the_dwt_and_tang_you_sections_are_continuous_at_the_finest_levels():
    """A bank's level-j atom is the cascade's limit function dilated by 2^j (PyWavelets'
    wavefun), not its handful of taps; the skeleton's gradient wavelet is the kernel's section
    sampled at s/16. Lambda* stays the level's own (the taps' spectral peak)."""
    p1 = wp.dwt_profile("sym7", 1)
    assert p1.x_px.size >= 256 and np.diff(p1.x_px).max() <= 2.0 / 16.0 + 1e-9
    assert p1.psi.max() == pytest.approx(1.0) and p1.psi.min() < -0.3     # a real wavelet shape
    assert p1.off_ladder and p1.lambda_px == pytest.approx(2.0, rel=0.05)  # level 1 at Nyquist
    p3 = wp.dwt_profile("sym7", 3)
    assert np.ptp(p3.x_px) == pytest.approx(np.ptp(p1.x_px) * 4.0, rel=1e-6)   # dilated by 2^j
    ty = wp.tang_you_profile("tang_you", 2.0)
    assert np.diff(ty.x_px).max() <= 2.0 / 16.0 + 1e-9
    np.testing.assert_allclose(ty.psi, -ty.psi[::-1], atol=1e-6)            # odd, like g1


def test_the_fractional_gaussians_band_constant_is_computed_once_per_order():
    """The measure route's fractional envelope needs a numerical Hankel transform for its
    lambda* constant, which depends on n alone: a ladder of six levels must not pay it six
    times (a knob tick redraws the whole cascade)."""
    import time

    wp.measure_profile("frac_gaussian", 1.0, frac_n=1.7)            # the one computation
    t0 = time.time()
    profiles = [wp.measure_profile("frac_gaussian", r, frac_n=1.7) for r in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)]
    assert time.time() - t0 < 1.0
    ratios = [p.lambda_px / p.scale_px for p in profiles]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)        # one constant, every level


def _mz_continuum_omega(alpha):
    """The peak of w |sinc(w/4)|^(alpha+1): tan(w/4) = (alpha+1) w / (4 alpha)."""
    lo, hi = 1e-3, 2.0 * np.pi - 1e-3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.tan(mid / 4.0) - (alpha + 1.0) * mid / (4.0 * alpha) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_the_mz_bar_draws_the_fractional_spline_at_its_order():
    """mz_edges' fractional B-spline: the bar's section is psi_alpha = theta_alpha' from the
    exact series (alpha = 1 the box pair, no ringing), alpha = 3 the classical path exactly,
    and lambda* the level's own from the fractional cascade's response, which at the coarse
    levels lands on the continuum constant 2 pi / w*, tan(w*/4) = (alpha+1) w*/(4 alpha)."""
    box = wp.mz_profile(1, alpha=1.0)
    inside = (np.abs(box.x_px) < 0.95) & (np.abs(box.x_px) > 0.05)   # the jump at 0 is 0
    assert np.allclose(np.abs(box.psi[inside]), 1.0, atol=1e-9)
    assert np.allclose(box.psi[np.abs(box.x_px) > 1.05], 0.0, atol=1e-9)
    assert box.extra["alpha"] == 1.0 and np.isfinite(box.lambda_px)
    same, ref = wp.mz_profile(2, alpha=3.0), wp.mz_profile(2)
    np.testing.assert_allclose(same.psi, ref.psi, atol=1e-9)
    assert same.lambda_px == pytest.approx(ref.lambda_px, rel=1e-3)
    assert ref.lambda_px / 4.0 == pytest.approx(2.0 * np.pi / _mz_continuum_omega(3.0), rel=0.1)
    for alpha in (1.5, 2.0, 4.0):
        p6 = wp.mz_profile(6, alpha=alpha)
        assert p6.lambda_px / 2.0 ** 6 == pytest.approx(2.0 * np.pi / _mz_continuum_omega(alpha), rel=0.02)
        assert not p6.off_ladder and wp.mz_profile(1, alpha=alpha).off_ladder
        assert "α" in p6.family


def test_the_wtmm_profile_reads_the_registry_for_the_new_smoothers():
    p = wp.wtmm_profile("q_gaussian", 8.0, {"q": 1.0})
    assert p.lambda_px == pytest.approx(8.0 * np.sqrt(2.0), rel=1e-3)
    assert np.max(np.abs(p.psi)) == pytest.approx(1.0) and np.allclose(p.psi, -p.psi[::-1], atol=1e-9)
    assert "∂θ_q" in p.family and "q-Gaussian q = 1" in p.family
    f = wp.wtmm_profile("frac_bspline", 8.0, {"alpha": 3.0})
    assert "∂θ_α" in f.family and "α = 3" in f.family and not f.off_ladder and not f.broad
    g = wp.wtmm_profile("gaussian", 8.0)
    assert "g1" in g.family                                                    # the pair as before


@pytest.mark.parametrize("name,params", [("frac_bspline", {"alpha": 3.0}), ("q_gaussian", {"q": 1.5})])
def test_a_plane_wave_peaks_where_the_registry_lambda_reads_its_wavelength(name, params):
    """The acceptance of the smoothers spec (§9): the scale with the largest mean modulus has
    λ* = L for the new smoothers too, through the backend and scale_units alone."""
    from dynamix.core.scale_units import lambda_peak_px
    from dynamix.core.wtmm_backend import PythonWTMMBackend, compute_scales2d

    L = 24.0
    y, x = np.mgrid[0:128, 0:128]
    wave = np.cos(2 * np.pi * x / L) + 0.05 * np.cos(2 * np.pi * y / 61.0)
    scales = compute_scales2d(4, 8)
    out = PythonWTMMBackend().cwt2d(wave, scales, wavelet=name, smoother_params=params,
                                    _engine="numpy")
    strengths = out["mod"].reshape(len(scales), -1).mean(axis=1)
    a = float(scales[int(np.argmax(strengths))])
    assert lambda_peak_px(a, name, 1, params=params) == pytest.approx(L, rel=2 ** (1 / 8) - 1 + 0.05)
