# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The disorder function and the Hessian measure on one field (spec 19 Phase A, §9's analytic cases):
the field form of Gunaratne, Hoffman and Kouri 1998 (PRE 57:5146, Eq. 2) and the Hessian measure of
Nathan and Gunaratne 2005 (PRE 71:035101, Eq. 1), with the derivative schemes of spec 19 §5 step 2."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dynamix.core import disorder
from dynamix.core.daf import daf_gain


def _stripes(n: int, cycles: int, angle: float = 0.0, phase: float = 0.0) -> np.ndarray:
    """cos(k·x + phase) on an n x n periodic grid, k = 2π·cycles/n rad/px along ``angle``."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    k = 2.0 * math.pi * cycles / n
    return np.cos(k * (x * math.cos(angle) + y * math.sin(angle)) + phase)


def _abs_cos_moment(beta: float) -> float:
    """⟨|cos φ|^β⟩ over a uniform phase = Γ((β+1)/2) / (√π Γ(β/2 + 1))."""
    return math.gamma((beta + 1) / 2) / (math.sqrt(math.pi) * math.gamma(beta / 2 + 1))


# ---------------------------------------------------------------- the transfer functions

def test_the_daf_derivative_scheme_is_the_daf_gain_at_pre57s_settings():
    k = np.linspace(0.0, math.pi, 64)
    assert np.allclose(disorder.transfer(k, "daf"), daf_gain(k, 4.0, 12), rtol=0, atol=1e-15)
    assert disorder.DAF_DEFAULTS == {"M": 12, "sigma_over_dx": 4.0}


def test_the_gaussian_scheme_is_a_gaussian_and_spectral_is_flat():
    k = np.linspace(0.0, math.pi, 64)
    assert np.allclose(disorder.transfer(k, "gaussian", sigma_d=1.0), np.exp(-0.5 * k ** 2))
    assert np.all(disorder.transfer(k, "spectral") == 1.0)


# ---------------------------------------------------------------- perfect and detuned stripes

@pytest.mark.parametrize("deriv", disorder.DERIVS)
def test_perfect_stripes_have_zero_disorder_under_every_scheme(deriv):
    """(Δ + k₀²)u = 0 for u = cos(k₀x): the two Helmholtz terms are filtered alike, so the discrete
    operator vanishes too (PRL 1995 Eq. 4). What is left is round-off, below the numerical zero, and
    a small moment order must not read it (a bare 1e-15 would give 0.06 at β = 0.1)."""
    n, cycles = 128, 6
    k0 = 2.0 * math.pi * cycles / n
    d = disorder.helmholtz_density(_stripes(n, cycles), k0, deriv=deriv, periodic=True)
    assert np.nanmax(d) < disorder.ZERO
    out = disorder.disorder_moments(d, disorder.DELTA_BETAS)
    assert np.all(out["mean"] == 0.0) and np.all(out["total"] == 0.0)


@pytest.mark.parametrize("deriv", ["spectral", "daf", "gaussian"])
def test_detuned_stripes_k0_3_k1_5_give_16_over_9_at_beta_1(deriv):
    """δ̄^field(1) = 2·|k₁² − k₀²|/(2k₀²) = 16/9: the PRL envelope value 8/9 times the field form's
    factor 2 at β = 1 (spec 19 §3.1, §9)."""
    n = 240
    k0 = 2.0 * math.pi * 3 / n
    d = disorder.helmholtz_density(_stripes(n, 5), k0, deriv=deriv, periodic=True)
    out = disorder.disorder_moments(d, [1.0])
    assert out["mean"][0] == pytest.approx(16 / 9, rel=1e-6)


def test_detuned_stripes_follow_the_closed_form_at_every_beta():
    """δ̄^field(β) = (2 − β)(16/9)^β ⟨|cos φ|^β⟩ / ⟨|cos φ|⟩^β (spec 19 §9): exact on the grid's own
    sampled phases, and within 4% of the continuum form ⟨|cos|^β⟩ = Γ((β+1)/2)/(√π Γ(β/2+1)),
    ⟨|cos|⟩ = 2/π (the cusp of |cos|^β at the zeros converges slowly at 48 samples per period:
    3% low at β = 0.1, 0.2% at β = 1.5)."""
    n = 240
    k0 = 2.0 * math.pi * 3 / n
    u = _stripes(n, 5)
    d = disorder.helmholtz_density(u, k0, deriv="spectral", periodic=True)
    betas = np.array(disorder.DELTA_BETAS)
    out = disorder.disorder_moments(d, betas)
    au = np.abs(u)
    au[au < disorder.ZERO] = 0.0                # cos(π/2) on the grid is 6e-17, a numerical zero
    sampled = [(2 - b) * (16 / 9) ** b * np.mean(au ** b) / np.mean(au) ** b for b in betas]
    assert np.allclose(out["mean"], sampled, rtol=1e-10)
    continuum = [(2 - b) * (16 / 9) ** b * _abs_cos_moment(b) * (math.pi / 2) ** b for b in betas]
    assert np.allclose(out["mean"], continuum, rtol=4e-2)


# ---------------------------------------------------------------- the variants and the β rules

def test_total_is_area_times_mean_and_the_rooted_mean_is_its_beta_root():
    n = 96
    k0 = 2.0 * math.pi * 3 / n
    d = disorder.helmholtz_density(_stripes(n, 4), k0, deriv="spectral", periodic=True)
    betas = np.array(disorder.DELTA_BETAS)
    out = disorder.disorder_moments(d, betas)
    assert np.allclose(out["total"], n * n * out["mean"], rtol=1e-12)
    assert np.allclose(out["rooted"], out["mean"] ** (1 / betas), rtol=1e-12)


@pytest.mark.parametrize("beta", [2.0, 2.5, 0.0])
def test_beta_at_or_above_2_and_beta_0_are_refused_for_the_disorder(beta):
    d = np.ones((8, 8))
    with pytest.raises(ValueError, match="PRL" if beta else "β = 0"):
        disorder.disorder_moments(d, [beta])


def test_negative_beta_floors_the_zeros_counts_them_and_warns():
    """Stripes with a perfect patch: D = 0 there, so β < 0 needs the floor (spec 19 §5 step 5)."""
    n = 130                                     # no sampled zero of the 7-cycle cosine is exact
    k0 = 2.0 * math.pi * 6 / n
    d = disorder.helmholtz_density(_stripes(n, 7), k0, deriv="spectral", periodic=True)
    d[:32, :32] = 0.0
    out = disorder.disorder_moments(d, [-0.5])
    assert np.isfinite(out["mean"]).all() and out["n_floor"] == 32 * 32 and "floor" in out["warning"]
    raised = disorder.disorder_moments(d, [-0.5], floor=10 * out["floor"])
    assert raised["mean"][0] != out["mean"][0]


def test_negative_beta_without_zeros_equals_the_direct_mean():
    rng = np.random.default_rng(0)
    d = rng.uniform(0.5, 1.5, (32, 32))
    out = disorder.disorder_moments(d, [-0.5])
    assert out["n_floor"] == 0 and out["warning"]
    assert out["mean"][0] == pytest.approx(2.5 * np.mean(d ** -0.5), rel=1e-12)


# ---------------------------------------------------------------- a Gaussian field: δ̄(1) = ε

def _gaussian_field(n: int, k0: float, width: float, seed: int) -> np.ndarray:
    """A periodic Gaussian random field whose spectrum is a ring of width ``width`` at ``k0``."""
    from dynamix.core import fft_policy

    rng = np.random.default_rng(seed)
    ky = 2.0 * math.pi * np.fft.fftfreq(n)[:, None]
    kx = 2.0 * math.pi * np.fft.fftfreq(n)[None, :]
    amp = np.exp(-0.5 * ((np.hypot(kx, ky) - k0) / width) ** 2)
    amp[0, 0] = 0.0
    spectrum = amp * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    return np.real(np.asarray(fft_policy.double().ifft2(spectrum)))


def _spectral_width(u: np.ndarray, k0: float) -> float:
    """ε² = ∫(k² − k₀²)² S d²k / (k₀⁴ ∫S d²k) from the realised power spectrum."""
    from dynamix.core import fft_policy

    n = u.shape[0]
    ky = 2.0 * math.pi * np.fft.fftfreq(n)[:, None]
    kx = 2.0 * math.pi * np.fft.fftfreq(n)[None, :]
    k2 = kx * kx + ky * ky
    power = np.abs(np.asarray(fft_policy.double().fft2(u))) ** 2
    return math.sqrt(float(np.sum((k2 - k0 * k0) ** 2 * power) / (k0 ** 4 * np.sum(power))))


def test_a_gaussian_field_reads_its_normalised_spectral_width_at_beta_1():
    """δ̄^field(1) = ε for a Gaussian field: (Δ + k₀²)u is Gaussian, and the two mean absolute
    values are σ√(2/π) each, so the ratio is σ_H / (k₀² σ_u) = ε (the disorder report §6.9)."""
    n = 1024
    k0 = 2.0 * math.pi * 60 / n
    u = _gaussian_field(n, k0, 0.15 * k0, seed=3)
    d = disorder.helmholtz_density(u, k0, deriv="spectral", periodic=True)
    out = disorder.disorder_moments(d, [1.0])
    assert out["mean"][0] == pytest.approx(_spectral_width(u, k0), rel=3e-2)


def test_a_gaussian_field_follows_the_closed_form_at_every_beta():
    """δ̄^field(β) = (2 − β) ε^β π^{β/2} Γ((β+1)/2) / √π for a Gaussian field."""
    n = 1024
    k0 = 2.0 * math.pi * 60 / n
    u = _gaussian_field(n, k0, 0.15 * k0, seed=5)
    eps = _spectral_width(u, k0)
    d = disorder.helmholtz_density(u, k0, deriv="spectral", periodic=True)
    betas = np.array(disorder.DELTA_BETAS)
    out = disorder.disorder_moments(d, betas)
    want = [(2 - b) * eps ** b * math.pi ** (b / 2) * math.gamma((b + 1) / 2) / math.sqrt(math.pi)
            for b in betas]
    assert np.allclose(out["mean"], want, rtol=5e-2)


# ---------------------------------------------------------------- the Hessian measure

def _interior(n: int, margin: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    m[margin:-margin, margin:-margin] = True
    return m


def test_a_plane_and_a_ruled_surface_have_zero_hessian_measure():
    """The DAF scheme is band-limited and flat to order M, so a plane reflected past its edges
    reads an exact zero inside the stencil's reach; the Gaussian at σ_d = 1 px is barely
    band-limited (e^{−π²/2} = 7e-3 at Nyquist), and its sinc tails leak ~1e-7 in."""
    n = 128
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    plane = 0.3 * x + 0.7 * y + 2.0
    inside = _interior(n, disorder.reach("daf") + 4)
    lam = disorder.hessian_density(plane, deriv="daf", mask=inside)
    assert np.nanmax(lam) < disorder.ZERO
    assert np.all(disorder.hessian_measure(lam, disorder.MU_BETAS)["mu"] == 0.0)
    assert np.nanmax(disorder.hessian_density(plane, deriv="gaussian", sigma_d=1.0, mask=inside)) < 1e-6
    ruled = np.sin(2 * math.pi * 4 * x / n) + 0.3 * np.cos(2 * math.pi * 8 * x / n)   # u = f(x)
    lam = disorder.hessian_density(ruled, deriv="spectral", periodic=True)
    assert np.nanmax(lam) < disorder.ZERO


def test_a_paraboloid_has_a_beta_independent_hessian_measure():
    """u = a x² + b y²: det H = 4ab everywhere, so μ(β) = (4|ab| / Var u)^{1/4} for every β
    (PRE 71:035101 Eq. 1), Var over the valid pixels; the DAF is exact on a quadratic."""
    n, a, b = 128, 0.5, -0.25
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    x -= n / 2
    y -= n / 2
    u = a * x * x + b * y * y
    mask = _interior(n, disorder.reach("daf") + 4)
    want = (4 * abs(a * b) / np.var(u[mask])) ** 0.25
    lam = disorder.hessian_density(u, deriv="daf", mask=mask)
    assert np.allclose(disorder.hessian_measure(lam, disorder.MU_BETAS, mask=mask)["mu"], want, rtol=1e-6)
    lam = disorder.hessian_density(u, deriv="gaussian", sigma_d=1.0, mask=mask)
    assert np.allclose(disorder.hessian_measure(lam, disorder.MU_BETAS, mask=mask)["mu"], want, rtol=1e-3)


def test_the_egg_crate_matches_its_closed_form():
    """u = cos x cos y: det H = k⁴ cos(x+y) cos(x−y) and Var u = 1/4, so
    μ(β) = √2 k ⟨|cos(x+y) cos(x−y)|^β⟩^{1/(4β)}: exact on the grid's sampled phases, within 5% of
    the continuum ⟨|cos|^β⟩² at 64 samples per period (the small-β cusps sample slowly)."""
    n, cycles = 192, 3
    k = 2.0 * math.pi * cycles / n
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    u = np.cos(k * x) * np.cos(k * y)
    lam = disorder.hessian_density(u, deriv="spectral", periodic=True)
    betas = np.array(disorder.MU_BETAS)
    out = disorder.hessian_measure(lam, betas)
    prod = np.abs(np.cos(k * (x + y)) * np.cos(k * (x - y)))
    prod[prod < disorder.ZERO] = 0.0
    sampled = [math.sqrt(2) * k * np.mean(prod ** b) ** (1 / (4 * b)) for b in betas]
    assert np.allclose(out["mu"], sampled, rtol=1e-9)
    continuum = [math.sqrt(2) * k * (_abs_cos_moment(b) ** 2) ** (1 / (4 * b)) for b in betas]
    settled = betas >= 0.5                       # below, the cusps' sampling error is raised to 1/(4β)
    assert np.allclose(out["mu"][settled], np.asarray(continuum)[settled], rtol=2e-2)


def test_the_hessian_measure_refuses_beta_0_and_floors_negative_orders():
    lam = np.abs(np.random.default_rng(1).standard_normal((32, 32)))
    with pytest.raises(ValueError, match="β = 0"):
        disorder.hessian_measure(lam, [0.0])
    lam[:8, :8] = 0.0
    out = disorder.hessian_measure(lam, [-0.5])
    assert np.isfinite(out["mu"]).all() and out["n_floor"] == 64 and "floor" in out["warning"]


def test_the_hessian_determinant_identity_holds_on_random_periodic_fields():
    """det H = ½[(Δu)² − ½ Δ|∇u|² + ∇u·∇Δu] (Hu 2005 Eq. 4, corrected; spec 19 §3.6)."""
    n = 128
    u = _gaussian_field(n, 2.0 * math.pi * 10 / n, 2.0 * math.pi * 4 / n, seed=11)
    kw = dict(deriv="spectral", periodic=True)
    ux, uy, uxx, uyy, uxy = disorder.filtered(
        u, [lambda kx, ky: 1j * kx, lambda kx, ky: 1j * ky, lambda kx, ky: -kx * kx,
            lambda kx, ky: -ky * ky, lambda kx, ky: -kx * ky], **kw)
    direct = uxx * uyy - uxy * uxy
    lap = uxx + uyy
    grad2 = ux * ux + uy * uy
    (lap_grad2,) = disorder.filtered(grad2, [lambda kx, ky: -(kx * kx + ky * ky)], **kw)
    lap_x, lap_y = disorder.filtered(lap, [lambda kx, ky: 1j * kx, lambda kx, ky: 1j * ky], **kw)
    identity = 0.5 * (lap * lap - 0.5 * lap_grad2 + ux * lap_x + uy * lap_y)
    assert np.allclose(direct, identity, rtol=0, atol=1e-8 * np.abs(direct).max())
    lam = disorder.hessian_density(u, **kw)
    assert np.allclose(lam, np.abs(direct) / np.var(u), rtol=1e-10)


# ---------------------------------------------------------------- invariance

def test_rolling_and_rotating_the_field_by_right_angles_leave_both_measures_unchanged():
    n = 128
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    k = 2.0 * math.pi / n
    u = (np.cos(k * (5 * x + 2 * y)) + 0.6 * np.cos(k * (3 * x - 6 * y) + 0.4)
         + 0.3 * np.cos(k * (7 * y) + 1.0))
    k0 = 2.0 * math.pi * 6 / n
    betas = np.array(disorder.DELTA_BETAS)

    def both(field):
        d = disorder.helmholtz_density(field, k0, deriv="spectral", periodic=True)
        lam = disorder.hessian_density(field, deriv="spectral", periodic=True)
        return (disorder.disorder_moments(d, betas)["mean"],
                disorder.hessian_measure(lam, disorder.MU_BETAS)["mu"])

    base = both(u)
    for other in (np.roll(np.roll(u, 17, axis=0), -40, axis=1), np.rot90(u), u.T, u[::-1, :]):
        got = both(other)
        assert np.allclose(got[0], base[0], rtol=1e-10) and np.allclose(got[1], base[1], rtol=1e-10)


# ---------------------------------------------------------------- k₀ and the spectral rule

def test_k0_by_the_spectral_peak_and_by_the_minimum_disorder_find_the_stripes_wavenumber():
    """On periodic stripes both estimators return the wavenumber; on stripes at an angle, which are
    not periodic on the grid, the reflect-padded Gaussian scheme gets within 2%."""
    n, cycles = 240, 5
    k1 = 2.0 * math.pi * cycles / n
    u = _stripes(n, cycles)
    assert disorder.k0_estimate(u, "spectral_peak", periodic=True) == pytest.approx(k1, abs=1e-12)
    assert disorder.k0_estimate(u, "min_disorder", deriv="spectral", periodic=True) == pytest.approx(k1, rel=1e-6)
    tilted = _stripes(n, cycles, angle=0.3)
    assert disorder.k0_estimate(tilted, "spectral_peak") == pytest.approx(k1, abs=2 * math.pi / n)
    assert disorder.k0_estimate(tilted, "min_disorder", deriv="gaussian") == pytest.approx(k1, rel=2e-2)
    assert disorder.K0_METHODS == ("min_disorder", "spectral_peak")


def test_spectral_derivatives_are_refused_on_a_non_periodic_crop():
    u = _stripes(64, 3)[:50, :40]
    with pytest.raises(ValueError, match="Periodic extension"):
        disorder.helmholtz_density(u, 0.3, deriv="spectral", periodic=False)
    assert np.isfinite(disorder.helmholtz_density(u, 0.3, deriv="daf", periodic=False)).all()


# ---------------------------------------------------------------- the harmonic filter (Hoffman et al. 2000)

def test_the_daf_filter_extends_the_field_keeps_the_band_and_leaves_the_block_clean():
    """Stripes with a second harmonic and a drift, cut non-periodically: the DAF extension makes
    the field periodic, the Goldman band [0.7, 1.3]·k_t drops the harmonic and the drift, and over
    the real block the disorder falls from the raw field's to near zero."""
    n, cycles = 160, 8
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    k1 = 2.0 * math.pi * cycles / n
    phase = k1 * (x * math.cos(0.25) + y * math.sin(0.25))
    u = np.cos(phase) + 0.3 * np.cos(2 * phase) + 0.4 * x / n
    v, _valid = disorder.detrend(u)
    k_t = disorder.k0_estimate(v, "spectral_peak")
    filtered, block, tuned = disorder.daf_filter(v, k_t)
    assert tuned.fallback is None and block == (n, n)
    assert filtered.shape[0] > n and filtered.shape[1] > n
    inside = np.zeros(filtered.shape, dtype=bool)
    inside[:n, :n] = True
    clean = disorder.disorder_moments(
        disorder.helmholtz_density(filtered, k1, deriv="daf", periodic=True, mask=inside), [1.0])
    raw = disorder.disorder_moments(disorder.helmholtz_density(v, k1, deriv="daf"), [1.0])
    # the floor is the periodization of a tilted, drifting crop (about 9% RMS against clean
    # stripes after the gap fill); the band mask itself leaves periodic stripes at exactly 0
    assert raw["mean"][0] > 0.3 and clean["mean"][0] < min(0.1, 0.1 * raw["mean"][0])
    same, block2, tuned2 = disorder.daf_filter(v, k_t, periodic=True)   # already periodic: no gap
    assert same.shape == v.shape and block2 == (n, n) and tuned2 is None
