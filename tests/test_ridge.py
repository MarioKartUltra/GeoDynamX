# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The wavelet ridge of a striped field (spec 19 §3.1, Passot and Newell 1994 Appendix A): at each
pixel the scale and orientation where the analytic q-Morlet's phase is stationary, carrying the
local wavevector K = ∇Ψ and the amplitude A = 2|T|·(curvature factor) (A.17–A.19), and the
phase-field form of the PRL's envelope operator built on them."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dynamix.core import analytic_cwt as acwt
from dynamix.core import disorder, ridge


def _wave(n: int, p: int, q: int) -> tuple:
    return (2.0 * math.pi * p / n, 2.0 * math.pi * q / n)


def _stripes(shape, k, phase: float = 0.0, amp: float = 1.0) -> np.ndarray:
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    return amp * np.cos(k[0] * x + k[1] * y + phase)


def _turned(k, angle: float, factor: float = 1.0) -> tuple:
    """``k`` turned by ``angle`` and scaled: a planform guess the bank is centred on."""
    mod, th = factor * math.hypot(*k), math.atan2(k[1], k[0]) + angle
    return (mod * math.cos(th), mod * math.sin(th))


@pytest.mark.parametrize("pq", [(8, 0), (6, 5), (-3, 7), (5, -6), (0, 9)])
@pytest.mark.parametrize("alpha", [(1.0, 1.0), (1.0, 2.0)])
def test_stripes_converge_to_their_wavevector_amplitude_and_phase_between_the_nodes(pq, alpha):
    """A cos(k·x + φ) with the bank centred 20° and 10% off k, so k sits between nodes in both
    scale and orientation: the fixed point lands on k, a_r = |ω|/|k|, A and Ψ are read exactly,
    in a few iterations, isotropic or not (A.15–A.18)."""
    n = 192
    k = _wave(n, *pq)
    u = _stripes((n, n), k, 0.3, 1.7)
    r = ridge.extract_ridge(u, _turned(k, math.radians(20), 1.1), alpha=alpha, periodic=True)
    assert r.converged.all() and r.iterations <= 4
    assert np.allclose(r.kx, k[0], atol=1e-6 * math.hypot(*k))
    assert np.allclose(r.ky, k[1], atol=1e-6 * math.hypot(*k))
    assert np.allclose(r.scale, acwt.OMEGA / math.hypot(*k), rtol=1e-6)
    assert np.allclose(r.amplitude, 1.7, rtol=1e-6)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    assert np.allclose(np.angle(np.exp(1j * (r.phase - k[0] * x - k[1] * y - 0.3))), 0.0, atol=1e-6)
    assert np.allclose(r.div_k, 0.0, atol=1e-6 * math.hypot(*k) ** 2)
    assert np.allclose(r.curvature, 1.0, rtol=1e-6)


def test_the_printed_a17_a18_hold_with_omega_along_the_diagonal():
    """Passot–Newell print the ridge for ω = (ω, ω): tan θ = (Ψ₁ − Ψ₂)/(Ψ₁ + Ψ₂) and
    a = ω√2/|∇Ψ| (A.17–A.18), i.e. θ = π/4 − ϑ and a = |ω|/|K| with |ω| = ω√2: the general form
    the code solves, θ_r = ξ − ϑ, read at ξ = 45°."""
    n = 192
    k = _wave(n, 6, 5)
    r = ridge.extract_ridge(_stripes((n, n), k), _turned(k, 0.2), periodic=True)
    theta = math.pi / 4 - np.arctan2(r.ky, r.kx)
    assert np.allclose(np.tan(theta), (r.kx - r.ky) / (r.kx + r.ky), rtol=1e-6)
    assert np.allclose(r.scale, (acwt.OMEGA / math.sqrt(2)) * math.sqrt(2) / np.hypot(r.kx, r.ky))


def test_a_slow_amplitude_modulation_is_read_to_a_part_in_a_thousand():
    """A(x) cos(kx) with A = 1 + 0.3 cos(2πx/L): on the ridge K = k exactly (the symmetric window
    keeps the phase) and |A| is the window's mean of A, off by ½a²A″ — below 1e-3 here. The
    phase-field density then follows its closed form ½√(4A′²k² + A″²)/(k₀²⟨A⟩) (spec 19 §3.1)."""
    ny, nx, cycles = 64, 1024, 100
    k = (2.0 * math.pi * cycles / nx, 0.0)
    x = np.arange(nx, dtype=np.float64)[None, :]
    amp = 1.0 + 0.3 * np.cos(2.0 * math.pi * x / nx)
    u = np.broadcast_to(amp * np.cos(k[0] * x), (ny, nx))
    r = ridge.extract_ridge(u, k, periodic=True)
    assert r.converged.all()
    assert np.allclose(r.kx, k[0], rtol=1e-6) and np.allclose(r.ky, 0.0, atol=1e-6 * k[0])
    assert np.allclose(r.amplitude, amp, atol=1e-3)
    d = disorder.phase_field_density(r.amplitude, r.kx, r.ky, r.div_k, k[0], deriv="spectral",
                                     periodic=True)
    w = 2.0 * math.pi / nx
    a1, a2 = -0.3 * w * np.sin(w * x), -0.3 * w * w * np.cos(w * x)
    closed = 0.5 * np.hypot(2.0 * a1 * k[0], a2) / (k[0] ** 2 * amp.mean())
    bias = (acwt.OMEGA / k[0]) ** 2 * w * w                 # the window's ½a²A″, relative, twice
    assert np.allclose(d, closed, atol=bias * closed.max())


def test_the_target_pattern_reads_unit_amplitude_and_the_prls_total_disorder():
    """u = cos(k₀r): the rings' curvature costs the raw modulus (1 + a⁴k₀²/r²)^{−1/4}, which A.19's
    factor returns, and the raw ∇Ψ reads k₀ low by ½∂_r atan(a²k₀/r), which the window's phase
    ½·arg det returns. So beyond the core |A| = 1 within 3%, K = ±k₀r̂ within 1%, and the
    phase-field density 1/(2k₀r) integrates to π(R − r_c)/k₀ over r_c = 2λ < r < R within 3%
    (PRL 1995 p. 3284, the core dropped as the PRL does); the envelope operator on A e^{i(Ψ − k₀·x)}
    agrees away from the ±k partition's fold. The crop is not periodic: the derivatives reflect."""
    n, lam = 192, 12.0
    k0 = 2.0 * math.pi / lam
    y, x = np.mgrid[0:n, 0:n].astype(np.float64) - (n - 1) / 2.0
    rr = np.hypot(x, y)
    r = ridge.extract_ridge(np.cos(k0 * rr), (k0, 0.0), bracket=(0.7, 1.4))
    r_in, r_out = 2.0 * lam, n / 2.0 - 2.5 * lam
    annulus = (rr > r_in) & (rr < r_out)
    assert r.converged[annulus].all()
    assert np.abs(r.amplitude[annulus] - 1.0).max() < 0.03
    a = acwt.OMEGA / k0
    assert np.abs(r.curvature[(rr > r_in) & (rr < r_in + 1)] - (1 + a ** 4 * (k0 / r_in) ** 2) ** 0.25).max() < 0.03
    radial = (r.kx * x + r.ky * y) / rr
    assert np.abs(np.abs(radial[annulus]) / k0 - 1.0).max() < 0.01
    # ⟨|A|⟩ normalises the density over the region measured: the reflected corners of this
    # non-periodic crop and the core ring read low amplitudes and would drag it down by 3%.
    d = disorder.phase_field_density(r.amplitude, r.kx, r.ky, r.div_k, k0, deriv="daf",
                                     mask=r.converged & annulus)
    assert np.nansum(d[annulus]) == pytest.approx(math.pi * (r_out - r_in) / k0, rel=0.03)
    # The operator route differentiates the demodulated envelope itself, so the pixels it masks
    # (the fold band, here also the annulus rims) are steps its derivatives ring on; compare
    # twelve pixels (three DAF widths) away from every rim.
    a_env = r.amplitude * np.exp(1j * (r.phase - k0 * (x + (n - 1) / 2.0)))
    d_op = disorder.envelope_density(a_env, (k0, 0.0), deriv="daf",
                                     mask=r.converged & annulus & ~r.boundary)
    inner = (rr > r_in + 12) & (rr < r_out - 12) & (np.abs(x) > 2 * lam + 12)
    assert np.nanmedian(np.abs(d_op[inner] / d[inner] - 1.0)) < 0.05


@pytest.mark.parametrize("q", [0.7, 1.5])
def test_the_q_morlet_ridge_reads_a_unit_cosine_for_q_either_side_of_one(q):
    """The amplitude is exact to 1e-3 for q on either side of 1 (the flat misfit is undone by the
    q-envelope itself). The wavevector carries the negative-frequency leak ε = Θ̂_q(2|ω|) as a
    ripple of 2ε: exponentially small for q ≥ 1, algebraic for the compact q < 1 envelope (spec 19
    §3.1, "negative-frequency leak per regime")."""
    n = 192
    k = _wave(n, 6, 5)
    r = ridge.extract_ridge(_stripes((n, n), k), _turned(k, 0.2, 1.05), q=q, periodic=True)
    assert r.converged.all()
    assert np.allclose(r.amplitude, 1.0, atol=1e-3)
    leak = abs(float(acwt.envelope_hat(2.0 * acwt.OMEGA, q)))   # negative lobes below q = 1
    assert np.allclose(np.hypot(r.kx, r.ky), math.hypot(*k), rtol=max(1e-4, 3.0 * leak))
    if q > 1:
        assert leak < 1e-4


def test_a_wave_outside_the_bank_is_not_converged_and_the_record_says_what_ran():
    n = 192
    k = _wave(n, 8, 0)
    r = ridge.extract_ridge(_stripes((n, n), k), (0.25 * k[0], 0.0), periodic=True)
    assert not r.converged.any()
    assert r.record["bank"] == {"bracket": [0.5, 2.0], "voices": 4, "orientations": 12, "nodes": 108}
    assert r.record["omega"] == acwt.OMEGA and r.record["q"] == 1.0 and r.record["q_beta"] == 0.5
    assert r.record["n_unconverged"] == n * n
