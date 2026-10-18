# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The 2-D analytic q-Morlet as a Fourier multiplier (spec 19 §3.1): Passot and Newell 1994's
wavelet, A.14–A.16, in the code's unit-peak (L¹) convention, and its q-Gaussian envelope."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dynamix.core import analytic_cwt as acwt


def _stripes(shape, k, phase: float = 0.0) -> np.ndarray:
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    return np.cos(k[0] * x + k[1] * y + phase)


def _wave(n: int, p: int, q: int) -> tuple:
    """The grid wavevector 2π(p, q)/n rad/px: a cosine of it is periodic on an n × n grid."""
    return (2.0 * math.pi * p / n, 2.0 * math.pi * q / n)


def test_the_q1_wavelet_is_passot_newells_gaussian_a15_and_a16():
    """ψ̂(aR_θk) = e^{−½|aR_θk − ω|²} centred where aR_θk = ω (A.15); with the anisotropy the
    quadratic form R_ξ[α⁻¹]R_{−ξ} along the carrier and the prefactor 1/√(α₁α₂) (A.16)."""
    kx, ky = acwt.grid((64, 96))
    kappa = (0.31, -0.17)
    a = acwt.OMEGA / math.hypot(*kappa)
    # the wavelet frame: first axis along κ (the carrier ω sits there), second across
    e1 = np.array(kappa) / math.hypot(*kappa)
    e2 = np.array([-e1[1], e1[0]])
    q1 = a * (kx * e1[0] + ky * e1[1]) - acwt.OMEGA
    q2 = a * (kx * e2[0] + ky * e2[1])
    assert np.allclose(acwt.wavelet_hat(kx, ky, kappa), np.exp(-0.5 * (q1 ** 2 + q2 ** 2)),
                       rtol=0, atol=1e-13)
    alpha = (1.0, 3.0)
    expected = np.exp(-0.5 * (q1 ** 2 / alpha[0] + q2 ** 2 / alpha[1])) / math.sqrt(alpha[0] * alpha[1])
    assert np.allclose(acwt.wavelet_hat(kx, ky, kappa, alpha=alpha), expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("pq", [(8, 0), (6, 5), (-3, 7), (5, -6), (10, 0), (6, 0), (0, 7)])
def test_the_analytic_wavelet_reads_half_the_amplitude_and_the_phase_of_a_cosine(pq):
    """cos(k·x) = ½(e^{ik·x} + e^{−ik·x}): the wavelet centred on k keeps the first half alone,
    so |T| = ½ and arg T = k·x (A.6's "factor ½"; the opposite term is e^{−2|ω|²})."""
    n = 192
    k = _wave(n, *pq)
    t = acwt.transform(_stripes((n, n), k, 0.7), k, periodic=True)
    leak = math.exp(-2.0 * acwt.OMEGA ** 2)                 # the e^{−ik·x} half, 1.5e-8 at ω = 3
    assert np.allclose(np.abs(t), 0.5, atol=1e-9 + leak)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    assert np.allclose(np.angle(t * np.exp(-1j * (k[0] * x + k[1] * y + 0.7))), 0.0, atol=1e-9 + 2 * leak)


def test_a_misfit_node_reads_the_same_amplitude_once_corrected_by_its_own_wavevector():
    """At a node κ ≠ k the modulus is (A/2)·ψ̂(aR_θk): the local wavevector K = Im(∇T/T) is k
    exactly on a plane wave, and 2|T|/ψ̂(aR_θK) gives A back. This is what lets a ridge read the
    amplitude between the nodes of a bank."""
    n = 192
    k = _wave(n, 6, 5)
    turn = math.atan2(k[1], k[0]) + math.radians(10)
    kappa = (1.1 * math.hypot(*k) * math.cos(turn), 1.1 * math.hypot(*k) * math.sin(turn))
    u = _stripes((n, n), k)
    t, tx, ty = acwt.transform(u, kappa, periodic=True, derivatives=True)
    kx_loc, ky_loc = acwt.local_wavevector(t, tx, ty)
    assert np.allclose(kx_loc, k[0], atol=1e-9) and np.allclose(ky_loc, k[1], atol=1e-9)
    assert np.abs(t).mean() < 0.45                       # the misfit costs modulus ...
    amplitude = 2.0 * np.abs(t) / acwt.wavelet_hat(kx_loc, ky_loc, kappa)
    assert np.allclose(amplitude, 1.0, atol=1e-6)        # ... which the correction returns


@pytest.mark.parametrize("alpha", [(1.0, 1.0), (1.0, 2.0)])
def test_direct_quadrature_of_a14_is_two_pi_times_the_multiplier(alpha):
    """Passot–Newell's T is A.14's spatial integral with the unnormalised Gaussian e^{−|X|²/2}
    (mass 2π); the code's multiplier has unit peak, so T_PN = 2π·T_code, and |T_PN|/π is the
    amplitude where 2|T_code| is. Applying 1/π to the multiplier's output would understate A by
    2π. Checked at one pixel by the plain sum of A.14 (A.16's form under the anisotropy)."""
    n = 512
    amp = 1.7
    k = _wave(n, 24, 7)
    u = amp * _stripes((n, n), k, 0.2)
    t = acwt.transform(u, k, periodic=True, alpha=alpha)
    b = (n // 2, n // 2)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    e1 = np.array(k) / math.hypot(*k)
    a = acwt.OMEGA / math.hypot(*k)
    x1 = ((x - b[1]) * e1[0] + (y - b[0]) * e1[1]) / a
    x2 = (-(x - b[1]) * e1[1] + (y - b[0]) * e1[0]) / a
    psi_bar = np.exp(-0.5 * (alpha[0] * x1 ** 2 + alpha[1] * x2 ** 2)) * np.exp(-1j * acwt.OMEGA * x1)
    t_pn = np.sum(psi_bar * u) / a ** 2
    assert t_pn / t[b] == pytest.approx(2.0 * math.pi, rel=1e-8)
    leak = math.exp(-2.0 * acwt.OMEGA ** 2)                 # the e^{−ik·x} half, 1.5e-8 at ω = 3
    assert abs(t_pn) / math.pi * math.sqrt(alpha[0] * alpha[1]) == pytest.approx(amp, rel=1e-8 + 2 * leak)
    assert 2.0 * abs(t[b]) * math.sqrt(alpha[0] * alpha[1]) == pytest.approx(amp, rel=1e-8 + 2 * leak)


@pytest.mark.parametrize("q", [0.7, 1.0, 1.5])
def test_the_q_envelope_keeps_the_unit_peak_and_the_cosine_amplitude(q):
    """Θ̂_q is q_gaussian_ft over its mass, so ψ̂_q peaks at 1 for every q (L¹ exact), and on a
    unit cosine 2|T| is 1: the only loss is the negative-frequency leak Θ̂_q(2|ω|), exponentially
    small for q ≥ 1 and algebraic for q < 1 (spec 19 §3.1, "negative-frequency leak")."""
    n = 192
    k = _wave(n, 5, 6)
    kx, ky = acwt.grid((n, n))
    hat = acwt.wavelet_hat(kx, ky, k, q=q)
    assert acwt.envelope_hat(0.0, q) == pytest.approx(1.0)
    assert np.max(hat) == pytest.approx(1.0, abs=1e-6)         # the grid holds k itself
    t = acwt.transform(_stripes((n, n), k), k, periodic=True, q=q)
    leak = abs(float(acwt.envelope_hat(2.0 * acwt.OMEGA, q)))   # negative lobes below q = 1
    assert np.allclose(2.0 * np.abs(t), 1.0, atol=max(1e-9, 2.0 * leak))
    assert leak < 1e-3


def test_q_at_or_above_two_is_refused_in_two_dimensions():
    with pytest.raises(ValueError, match="q < 2"):
        acwt.wavelet_hat(*acwt.grid((8, 8)), (0.5, 0.0), q=2.0)


def test_the_zero_mean_term_removes_the_response_at_k_equals_zero():
    """Borges et al. 2004 eq. 23's correction, off by default (Passot–Newell never reconstruct):
    ψ̂_q(0) = 0 with it, and the peak drops by c_q² = Θ̂_q(|ω|)²."""
    kx, ky = acwt.grid((64, 64))
    kappa = (0.4, 0.3)
    hat = acwt.wavelet_hat(kx, ky, kappa, zero_mean=True)
    assert abs(hat[0, 0]) < 1e-15
    c = acwt.envelope_hat(acwt.OMEGA)
    peak = float(acwt.wavelet_hat(np.array(kappa[0]), np.array(kappa[1]), kappa, zero_mean=True))
    assert peak == pytest.approx(1 - c * c)
