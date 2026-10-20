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
    within 2.78 σ (Freddie Friday's continuum values, 2026-10-19), measured from the backend's own
    Fourier kernels so the constant is in the stage's scale units."""
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
