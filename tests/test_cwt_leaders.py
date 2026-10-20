# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""CWT leaders (spec 30): the energy radius of the analysing kernels, then the scaling laws."""
from __future__ import annotations

import numpy as np
import pytest

#: σ of the backend's Gaussian smoother per unit of its ``scale`` (``smoothers._SIGMA_PER_SCALE``):
#: θ̂ = exp(−(f·scale)²) with f in cycles per pixel gives σ = scale / (π√2) pixels.
SIGMA_PER_SCALE = 1.0 / (np.pi * np.sqrt(2.0))


def test_energy_radius_of_g1_and_g2_match_the_continuum_constants():
    """99 % of the energy of the gradient-of-Gaussian modulus lies within 2.58 σ, of the Ricker
    within 2.78 σ (the continuum values of spec 30 §2), measured from the backend's own Fourier
    kernels so the constant is in the stage's scale units."""
    from dynamix.core.cwt_leaders import energy_radius

    c_g1 = energy_radius("gaussian", "gradient", {}, energy=0.99)
    c_g2 = energy_radius("mexican", "none", {}, energy=0.99)
    assert c_g1 == pytest.approx(2.58 * SIGMA_PER_SCALE, abs=0.03 * SIGMA_PER_SCALE)
    assert c_g2 == pytest.approx(2.78 * SIGMA_PER_SCALE, abs=0.03 * SIGMA_PER_SCALE)


def test_energy_radius_grows_with_the_energy_fraction_and_is_cached():
    from dynamix.core import cwt_leaders

    lo = cwt_leaders.energy_radius("gaussian", "gradient", {}, energy=0.9)
    hi = cwt_leaders.energy_radius("gaussian", "gradient", {}, energy=0.999)
    assert lo < cwt_leaders.energy_radius("gaussian", "gradient", {}, energy=0.99) < hi
    # cached per (smoother, operator, the smoother's own knobs, energy); a foreign key is ignored
    assert ("gaussian", "gradient", (), 0.99) in cwt_leaders._RADII
    assert cwt_leaders.energy_radius("gaussian", "gradient", {"irrelevant": 1}, energy=0.99) == \
        cwt_leaders.energy_radius("gaussian", "gradient", {}, energy=0.99)


def test_q_gaussian_converges_to_the_gaussian_as_q_tends_to_one():
    from dynamix.core.cwt_leaders import energy_radius

    c = energy_radius("gaussian", "gradient", {}, energy=0.99)
    for q in (0.98, 1.02):
        assert energy_radius("q_gaussian", "gradient", {"q": q}, energy=0.99) == pytest.approx(c, rel=0.02)


def test_the_mass_kernel_of_the_gaussian_is_narrower_than_its_gradient():
    """θ itself (operator none) has no ring: 99 % of exp(−r²/2σ²)² lies within √(−2 ln 0.01) σ."""
    from dynamix.core.cwt_leaders import energy_radius

    c0 = energy_radius("gaussian", "none", {}, energy=0.99)
    assert c0 == pytest.approx(np.sqrt(-np.log(0.01)) * SIGMA_PER_SCALE, rel=0.02)
    assert c0 < energy_radius("gaussian", "gradient", {}, energy=0.99)


# ------------------------------------------------------------------- the stage (spec 30 §3.2)

def _field(values, name="f"):
    from dynamix.core.frames import LocalFrame
    from dynamix.core.rasterfield import RasterField

    values = np.asarray(values, dtype=np.float64)
    ny, nx = values.shape
    return RasterField(name=name, values=values, frame=LocalFrame(),
                       x_axis=np.arange(nx, dtype=np.float64), y_axis=np.arange(ny, dtype=np.float64))


LADDER = {"n_oct": 2, "n_voice": 1, "a_min": 2.0}


def test_the_mass_kernel_keeps_a_constant_and_the_ricker_removes_it():
    """``operator = none``: θ̂(0) = 1 for a smoothing θ, so a constant field is its own transform
    at every scale; the Mexican smoother has θ̂(0) = 0 (g2, two vanishing moments) and gives 0."""
    from dynamix.core.wtmm_backend import run_cwt_stage

    f = _field(np.full((40, 48), 3.0))
    mass = run_cwt_stage(f, dict(LADDER, wavelet="gaussian"), operator="none")
    assert mass["mod"].shape == (2, 40, 48) and mass["operator"] == "none"
    np.testing.assert_allclose(mass["mod"], 3.0, rtol=1e-5, atol=1e-5)
    ricker = run_cwt_stage(f, dict(LADDER, wavelet="mexican"), operator="none")
    np.testing.assert_allclose(ricker["mod"], 0.0, atol=1e-5)


def test_the_gradient_stage_is_the_wtmms_own_and_a_leaders_run_after_it_recomputes_nothing():
    from dynamix.core import fft_policy
    from dynamix.core.wtmm_backend import run_cwt_stage, run_wtmm2d

    rng = np.random.default_rng(5)
    f = _field(rng.standard_normal((48, 56)).cumsum(0).cumsum(1), name="shared")
    params = dict(LADDER, wavelet="gaussian")
    cache: dict = {}
    res = run_wtmm2d(f, params, stage_cache=cache)
    with fft_policy.observing() as calls:
        again = run_cwt_stage(f, params, operator="gradient", stage_cache=cache)
    assert "cwt" in again["cache_hits"] and not calls
    np.testing.assert_array_equal(again["scales"], res["scales"])
    # the scalar stage is a different entry: a miss, then a hit of its own
    first = run_cwt_stage(f, params, operator="none", stage_cache=cache)
    assert "cwt" not in first["cache_hits"]
    assert "cwt" in run_cwt_stage(f, params, operator="none", stage_cache=cache)["cache_hits"]


def test_the_lift_multiplies_each_scale_by_a_to_the_eta():
    from dynamix.core.wtmm_backend import run_cwt_stage

    rng = np.random.default_rng(6)
    f = _field(rng.standard_normal((32, 32)).cumsum(0))
    flat = run_cwt_stage(f, dict(LADDER, wavelet="gaussian"), operator="gradient")
    lifted = run_cwt_stage(f, dict(LADDER, wavelet="gaussian", fracint_alpha=0.5), operator="gradient")
    for i, a in enumerate(flat["scales"]):
        np.testing.assert_allclose(lifted["mod"][i], flat["mod"][i] * np.float32(a ** 0.5), rtol=1e-6)


def test_the_stage_refuses_an_unknown_operator():
    from dynamix.core.wtmm_backend import run_cwt_stage

    with pytest.raises(ValueError, match="operator"):
        run_cwt_stage(_field(np.zeros((8, 8))), dict(LADDER), operator="curl")
