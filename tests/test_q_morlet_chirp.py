# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""A single oscillating singularity under the q-Morlet: the oracle for the complex-wavelet WTMM.

The one-sided chirp x₊^h sin(x^−β) has Hölder exponent h and oscillation exponent β at 0. Under a
Morlet-type wavelet (L¹ convention) the transform's modulus ridge sits where the chirp's instantaneous
frequency β x^−(β+1) meets the carrier k/a, so x_a = (βa/k)^(1/(1+β)), and along the ridge the
stationary-phase amplitude scales as a^((h − β/2)/(1+β)) while the wavelet's envelope is wide against
the chirp's stationary region (the chirp regime). The modulus reads both laws; the real and imaginary
parts oscillate with the carrier's phase along the ridge and read neither. Borges, Tsallis, Miranda &
Andrade 2004: the q-Morlet is the modulated q-Gaussian (eq. 23) with the carrier k_q of eq. 29 (the
envelope at one half after one period), here with the exact zero-mean correction."""
from __future__ import annotations

import numpy as np
import pytest


def _e_q(u, q):
    if q == 1.0:
        return np.exp(u)
    return np.clip(1.0 + (1.0 - q) * u, 0.0, None) ** (1.0 / (1.0 - q))


def _k_q(q, bw=0.5):
    if q == 1.0:
        return 2 * np.pi * np.sqrt(bw / np.log(2.0))
    return 2 * np.pi * np.sqrt((1 - q) * bw / (1 - 2 ** (q - 1)))


def _q_morlet(u, q, bw=0.5):
    env = _e_q(-bw * u * u, q)
    carrier = np.exp(1j * _k_q(q, bw) * u)
    return env * (carrier - (env * carrier).sum() / env.sum())


def _cwt(f, x, a, q):
    dx = x[1] - x[0]
    psi = _q_morlet(x / a, q) / a
    return np.fft.ifft(np.fft.fft(f) * np.fft.fft(np.fft.ifftshift(np.conj(psi)))) * dx


def _ridge_laws(h, beta, q, n=1 << 19, half=2.0, scales=np.geomspace(0.002, 0.2, 12)):
    x = (np.arange(n) - n // 2) * (2 * half / n)
    dx = x[1] - x[0]
    with np.errstate(all="ignore"):
        f = np.where(x > 0, np.abs(x) ** h * np.sin(np.abs(x) ** -beta), 0.0)
    f[~np.isfinite(f)] = 0.0
    x_min = 5 * (beta * dx) ** (1 / (1 + beta))                 # above the sampling limit
    sel = (x > x_min) & (x < 1.0)
    idx = np.flatnonzero(sel)
    xs, mods, res = [], [], []
    for a in scales:
        T = _cwt(f, x, a, q)
        i = idx[np.argmax(np.abs(T[sel]))]
        xs.append(x[i]); mods.append(abs(T[i])); res.append(abs(T[i].real))
    la = np.log(scales)
    pos = np.polyfit(la, np.log(xs), 1)
    amp = np.polyfit(la, np.log(mods), 1)
    scatter = lambda v: float(np.std(np.log(np.maximum(v, 1e-300))
                                     - np.polyval(np.polyfit(la, np.log(np.maximum(v, 1e-300)), 1), la)))
    return {"pos_slope": pos[0], "pos_prefactor": np.exp(pos[1]), "amp_slope": amp[0],
            "mod_scatter": scatter(mods), "re_scatter": scatter(res)}


@pytest.mark.parametrize("q", [1.0, 1.5, 0.7])
def test_the_q_morlet_modulus_reads_a_chirps_two_exponents_and_its_parts_do_not(q):
    h, beta = 0.6, 0.5
    r = _ridge_laws(h, beta, q)
    assert r["pos_slope"] == pytest.approx(1 / (1 + beta), abs=0.07)       # β from the ridge position
    assert r["amp_slope"] == pytest.approx((h - beta / 2) / (1 + beta), abs=0.03)   # then h from the amplitude
    assert h == pytest.approx(beta / 2 + (1 + beta) * r["amp_slope"], abs=0.05)
    assert r["mod_scatter"] < 0.03                                          # a clean power law
    assert r["re_scatter"] > 0.3                                            # the real part oscillates along the ridge


def test_the_q_synchronised_carrier_places_the_ridge_where_the_law_says():
    # x_a = (β a / k_q)^(1/(1+β)) at β = 1: the prefactor follows k_q(q) (eq. 29), exactly for the
    # compact q = 0.7 envelope (0.4223 against 0.4220) and within a few percent for q = 1; the
    # heavy-tailed q = 1.5 envelope's effective carrier sits above k_q (its spectrum is not symmetric
    # about the carrier). At β = 0.5 the compact envelope's ridge sits 39 % outside the law and the
    # heavy-tailed one 18 %, so the carrier law is pinned at β = 1 only; h and β still read (above).
    beta = 1.0
    for q, tol in ((0.7, 0.03), (1.0, 0.06), (1.5, 0.12)):
        r = _ridge_laws(0.3, beta, q)
        law = (beta / _k_q(q)) ** (1 / (1 + beta))
        assert r["pos_slope"] == pytest.approx(0.5, abs=0.02), q
        assert r["pos_prefactor"] == pytest.approx(law, rel=tol), q
