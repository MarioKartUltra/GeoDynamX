# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The 2-D analytic q-Morlet as a Fourier multiplier.

Passot and Newell 1994 (Physica D 74:301, Appendix A) read the slowly varying amplitude of a
striped field with a complex Gaussian wavelet ψ(X) = e^{−|X|²/2} e^{iω·X} rotated by θ and dilated
by a, in Fourier form T f(b, a, θ) = Σ_k f̂(k) e^{ik·b} e^{−|aR_θk − ω|²/2} (A.15): a Gaussian in k
centred where aR_θk = ω, so the wavelet sees one half of a real field's spectrum and its phase is
the field's. The anisotropic form (A.16) stretches the envelope along the carrier (α₁) and across
it (α₂), with the prefactor 1/√(α₁α₂). Here the Gaussian envelope is the radial q-Gaussian of
Borges, Tsallis, Miranda and Andrade 2004 (eq. 23) through its closed-form transform
(:func:`dynamix.core.q_fourier.q_gaussian_ft`), the Morlet at q = 1; β_q is its width (½ gives
Passot–Newell's e^{−|X|²/2}).

Conventions. A node is named by the wavevector it is centred on, κ = R_{−θ}ω/a, so a = |ω|/|κ| and
the wavelet frame has its first axis along κ. The multiplier has unit peak (unit mass in space,
the L¹ convention of the whole app), so on A cos(k·x) the node at κ = k returns |T| = A/2: the
analytic ½ of A.6. Passot–Newell's T is 2π times this (their Gaussian is unnormalised), which is
why their amplitude constant is |T|/π where ours is 2|T|.
"""
from __future__ import annotations

import math

import numpy as np

from dynamix.core import fft_policy
from dynamix.core.q_fourier import q_gaussian_ft

__all__ = ["OMEGA", "envelope_hat", "grid", "local_wavevector", "reach", "transform",
           "wavelet_hat"]

#: The carrier |ω| in wavelet units: Passot–Newell's 2-D runs (their 1-D wavelet takes "between 2
#: and 6"). The window is |ω|/k wide, so its phase curvature a²Ψ″ grows as ω²: on a target at 2λ,
#: ω = 6 reads the total disorder 3–7% high (3 at the bank's default, 7 at twice its nodes) where 3
#: reads it within 1%; the negative-frequency leak Θ̂_q(2|ω|) is 1.5e-8 at ω = 3 for q = 1.
OMEGA = 3.0


def grid(shape) -> tuple:
    """``(kx, ky)`` in rad/px for an FFT of ``shape``, broadcastable to it."""
    ny, nx = shape
    return (2.0 * math.pi * np.fft.fftfreq(nx)[None, :], 2.0 * math.pi * np.fft.fftfreq(ny)[:, None])


def envelope_hat(k, q: float = 1.0, q_beta: float = 0.5):
    """Θ̂_q(|k|): the transform of the radial q-Gaussian e_q(−β_q ρ²) over its mass, so the peak
    is 1; e^{−k²/2} at q = 1, β_q = ½. The mass is finite below q = 2 only (2-D)."""
    if q >= 2.0:
        raise ValueError(f"q = {q}: the 2-D q-Gaussian envelope needs q < 2 (its mass is infinite "
                         "from q = 2)")
    return q_gaussian_ft(k, q, q_beta) / q_gaussian_ft(0.0, q, q_beta)


def wavelet_hat(kx, ky, kappa, *, omega: float = OMEGA, alpha=(1.0, 1.0), q: float = 1.0,
                q_beta: float = 0.5, zero_mean: bool = False):
    """ψ̂_q(aR_θk) of the wavelet centred on the wavevector ``kappa``: Θ̂_q of the misfit k − κ,
    scaled by a = |ω|/|κ| and by 1/√α₁ along κ, 1/√α₂ across, over √(α₁α₂) (A.15; A.16 with the
    anisotropy). ``zero_mean`` subtracts c_q Θ̂_q(aR_θk), c_q = Θ̂_q(|ω|/√α₁), so that ψ̂_q(0) = 0
    (Borges et al. eq. 23); Passot–Newell never reconstruct and leave it out."""
    cx, cy = (float(c) for c in kappa)
    mod = math.hypot(cx, cy)
    a = float(omega) / mod
    e1 = (cx / mod, cy / mod)
    a1, a2 = (float(v) for v in alpha)

    def misfit(dx, dy):
        along = dx * e1[0] + dy * e1[1]
        across = dy * e1[0] - dx * e1[1]
        return a * np.hypot(along / math.sqrt(a1), across / math.sqrt(a2))

    kx, ky = np.asarray(kx, dtype=np.float64), np.asarray(ky, dtype=np.float64)
    hat = envelope_hat(misfit(kx - cx, ky - cy), q, q_beta)
    if zero_mean:
        c = envelope_hat(float(omega) / math.sqrt(a1), q, q_beta)
        hat = hat - c * envelope_hat(misfit(kx, ky), q, q_beta)
    return hat / math.sqrt(a1 * a2)


def reach(k_min: float, omega: float = OMEGA, alpha=(1.0, 1.0), q_beta: float = 0.5) -> int:
    """How far the widest wavelet of a bank down to |κ| = ``k_min`` reaches, in px: four envelope
    widths, a = |ω|/k_min stretched by the softer anisotropy axis."""
    return int(math.ceil(4.0 * float(omega) / float(k_min) /
                         math.sqrt(2.0 * float(q_beta) * min(alpha))))


def transform(u, kappa, *, periodic: bool = False, derivatives: bool = False, **wavelet):
    """T at the one node ``kappa`` over the field ``u``, on the FFT policy at 64 bits; with
    ``derivatives`` also ∂T/∂x and ∂T/∂y. A non-periodic field is reflected by the wavelet's
    reach first and cropped after."""
    u = np.asarray(u, dtype=np.float64)
    ny, nx = u.shape
    pad = 0 if periodic else reach(math.hypot(*kappa), wavelet.get("omega", OMEGA),
                                   wavelet.get("alpha", (1.0, 1.0)), wavelet.get("q_beta", .5))
    up = np.pad(u, pad, mode="reflect") if pad else u
    kx, ky = grid(up.shape)
    fft = fft_policy.double()
    spectrum = np.asarray(fft.fft2(up)) * wavelet_hat(kx, ky, kappa, **wavelet)
    crop = (slice(pad, pad + ny), slice(pad, pad + nx))
    t = np.asarray(fft.ifft2(spectrum))[crop]
    if not derivatives:
        return t
    return (t, np.asarray(fft.ifft2(1j * kx * spectrum))[crop],
            np.asarray(fft.ifft2(1j * ky * spectrum))[crop])


def local_wavevector(t, tx, ty) -> tuple:
    """K = ∇Ψ = Im(∇T/T), the local wavevector a node reads, with no phase unwrapping; NaN where
    T vanishes."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.imag(tx / t), np.imag(ty / t)
