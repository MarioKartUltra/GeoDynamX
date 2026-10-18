# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The wavelet ridge of a striped field: Passot and Newell 1994, Appendix A, in two dimensions.

At each pixel b the ridge is the scale and orientation where the analytic wavelet's phase Ψ = arg T
is stationary, which for a fixed-carrier wavelet is the fixed point aR_θ∇Ψ = ω (A.12): the wavelet
centred on the wavevector the field has there. Written on the wavevector κ = R_{−θ}ω/a the
iteration (A.13, A.17–A.18) is κ ← ∇Ψ(b; κ): the node centred on κ reads the local wavevector
K = Im(∇T/T), which becomes the next centre. On the ridge the modulus gives the amplitude.

Stationary phase under the Gaussian window N = R diag(α) Rᵀ − i a²H (H the phase Hessian, the first
wavelet axis along K) gives, at a node misfit by d = K − κ, T ≈ (A/2) e^{iφ} e^{−½a²dᵀN⁻¹d}/√det N.
Its modulus is A.19, written as the determinant it comes from: the amplitude is
2|T|·|det N|^{1/2}/√(α₁α₂), 1 on a plane wave and (1 + a⁴Ψ″²)^{1/4} per curved axis. Its argument
is A.8's ½·atan(a²Φ″): the field's phase is φ = Ψ + ½·arg det N, and the wavevector K = ∇φ — on a
target the raw ∇Ψ reads k₀ a percent low, which the operator |K|² − k₀² would turn into a spurious
disorder as large as the true one. The same expansion gives the fixed-point map a contraction
a⁴Ψ″²/(1 + a⁴Ψ″²), near 0.9 on tight rolls at ω = 6, so each round of two steps is extrapolated by
Aitken's Δ², in (log|κ|, ϑ) where the radial and the angular contraction separate.

T is computed on a log-polar bank of nodes around the planform wavevector k₀ — a bracket of |κ| at
``voices`` per octave, ``orientations`` over the half-plane about k₀'s direction — and read between
them by bilinear interpolation in (log|κ|, ϑ); a wave and its negative are one pattern, the node at
ϑ + π is the conjugate of the node at ϑ, and that is how the half-plane closes. Each node's modulus
is read as the amplitude it implies — the flat misfit undone by the q-envelope itself, the curvature
and its coupling to the misfit by the Gaussian window — so what is interpolated is already the
amplitude, exact on a plane wave at every node, which is what makes a coarse bank enough.

H is the Hessian of the ridge phase, whose gradient is the fixed point itself: H = ∇κ*, differenced
with each neighbour folded to the pixel's hemisphere, after κ* is smoothed over half the window
width in its tensor form κκᵀ (the correction to K is a second derivative of the ridge, and would
turn a ripple δ in κ* into 4ω²δ). A fixed node's own phase Hessian Im(T_ij/T − T_iT_j/T²) is not H:
the misfit grows as b moves along a curved roll, and it reads curvature flattened by 1/(1 + a⁴Ψ″²).

A pixel whose iteration leaves the bracket, fails to settle within ``max_iter`` steps, sits where T
vanishes, or has such a neighbour is not converged. The first pass stores the local wavevector of
every node (two float32 per pixel per node); the second recomputes only the nodes a pixel reads.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from dynamix.core import fft_policy
from dynamix.core.analytic_cwt import OMEGA, grid, local_wavevector, reach, wavelet_hat

__all__ = ["BRACKET", "ORIENTATIONS", "Ridge", "VOICES", "extract_ridge"]

#: The bank's bracket of |κ| in units of k₀ (the k₀ search bracket of :mod:`dynamix.core.disorder`).
BRACKET = (0.5, 2.0)
#: Scales per octave of the bank.
VOICES = 4
#: Orientations over the half-plane (Passot–Newell start θ from 12).
ORIENTATIONS = 12


@dataclasses.dataclass(frozen=True)
class Ridge:
    """The ridge, pixel by pixel: the amplitude A, the phase φ, the local wavevector K (folded
    into the half-plane about k₀), ∇·K, the ridge scale a_r = |ω|/|∇Ψ|, the curvature factor
    applied, the convergence mask, the pixels within one node of the ±k fold, the steps taken and
    the record."""
    amplitude: np.ndarray
    phase: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    div_k: np.ndarray
    scale: np.ndarray
    curvature: np.ndarray
    converged: np.ndarray
    boundary: np.ndarray
    iterations: int
    record: dict


def extract_ridge(u, k0_vec, *, omega: float = OMEGA, alpha=(1.0, 1.0), q: float = 1.0,
                  q_beta: float = 0.5, zero_mean: bool = False, bracket=BRACKET,
                  voices: int = VOICES, orientations: int = ORIENTATIONS, tol: float = 1e-4,
                  max_iter: int = 20, periodic: bool = False) -> Ridge:
    """The ridge of ``u`` about the planform wavevector ``k0_vec`` (its magnitude centres the
    bracket, its direction the half-plane). A non-periodic field is reflected by the widest
    wavelet's reach first."""
    u = np.asarray(u, dtype=np.float64)
    ny, nx = u.shape
    k0 = math.hypot(*k0_vec)
    th0 = math.atan2(k0_vec[1], k0_vec[0])
    lo, hi = (float(b) for b in bracket)
    n_s = int(round(voices * math.log2(hi / lo))) + 1
    n_o = int(orientations)
    step = math.pi / n_o
    mods = k0 * lo * 2.0 ** (np.arange(n_s) / voices)
    angs = th0 - math.pi / 2 + np.arange(n_o) * step
    a1, a2 = (float(v) for v in alpha)
    wavelet = dict(omega=omega, alpha=(a1, a2), q=q, q_beta=q_beta, zero_mean=zero_mean)

    pad = 0 if periodic else reach(k0 * lo, omega, alpha)
    up = np.pad(u, pad, mode="reflect") if pad else u
    kx, ky = grid(up.shape)
    fft = fft_policy.double()
    spectrum = np.asarray(fft.fft2(up))
    crop = (slice(pad, pad + ny), slice(pad, pad + nx))
    pixels = np.arange(ny * nx).reshape(ny, nx)

    def node(i, j):
        return (mods[i] * math.cos(angs[j]), mods[i] * math.sin(angs[j]))

    def transform(kappa, multipliers):
        s = spectrum * wavelet_hat(kx, ky, kappa, **wavelet)
        return [np.asarray(fft.ifft2(s * m)) for m in multipliers]

    # Pass 1: every node's local wavevector, and the strongest node per pixel to start from.
    bank = np.empty((n_s * n_o, 2, ny, nx), dtype=np.float32)
    best = np.full((ny, nx), -1.0)
    start = np.zeros((2, ny, nx))
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(n_s):
            for j in range(n_o):
                t, tx, ty = (f[crop] for f in transform(node(i, j), (1.0, 1j * kx, 1j * ky)))
                k_loc = local_wavevector(t, tx, ty)
                bank[i * n_o + j] = k_loc
                stronger = np.abs(t) > best
                best[stronger] = np.abs(t)[stronger]
                start[0][stronger], start[1][stronger] = k_loc[0][stronger], k_loc[1][stronger]
    flat = bank.transpose(1, 0, 2, 3).reshape(2, -1)     # (2, nodes·pixels)

    c0, s0 = math.cos(th0), math.sin(th0)

    def fold(vx, vy):
        flip = vx * c0 + vy * s0 < 0.0
        return np.where(flip, -vx, vx), np.where(flip, -vy, vy)

    def stencil(vx, vy):
        """The lower node (i0, j0) around κ, the fractions towards (i0 + 1, j0 + 1), and the
        pixels outside the bracket; j0 + 1 = n_o names node 0's conjugate."""
        mod = np.hypot(vx, vy)
        with np.errstate(invalid="ignore", divide="ignore"):
            li = np.log2(mod / (k0 * lo)) * voices
        outside = ~np.isfinite(li) | (li < -1e-9) | (li > n_s - 1 + 1e-9)
        li = np.clip(np.nan_to_num(li), 0.0, n_s - 1.0)
        i0 = np.minimum(li.astype(np.int64), max(n_s - 2, 0))
        dj = np.mod(np.arctan2(vy, vx) - angs[0], math.pi) / step
        j0 = np.minimum(dj.astype(np.int64), n_o - 1)
        return i0, j0, li - i0, dj - j0, outside

    def corners(i0, j0, fi, fj):
        return ((i0, j0, (1 - fi) * (1 - fj)), (i0 + 1, j0, fi * (1 - fj)),
                (i0, j0 + 1, (1 - fi) * fj), (i0 + 1, j0 + 1, fi * fj))

    def read(vx, vy):
        """The bank's local wavevector interpolated at κ, folded; NaN where a node read NaN."""
        i0, j0, fi, fj, outside = stencil(vx, vy)
        out = np.zeros((2, ny, nx))
        for i, j, w in corners(i0, j0, fi, fj):
            flip = j == n_o
            idx = (i * n_o + np.where(flip, 0, j)) * (ny * nx) + pixels
            sign = np.where(flip, -w, w)
            out[0] += sign * flat[0][idx]
            out[1] += sign * flat[1][idx]
        return (*fold(out[0], out[1]), outside)

    def aitken(x0, x1, x2):
        """Δ² extrapolation of a sequence contracting by a fixed factor; x2 where it stalls."""
        d1, d2 = x1 - x0, x2 - x1
        den = d2 - d1
        with np.errstate(invalid="ignore", divide="ignore"):
            acc = x2 - d2 * d2 / den
        return np.where(np.abs(den) > 1e-12 * (np.abs(x2) + 1.0), acc, x2)

    # The iteration runs in (log|κ|, ϑ) about k₀'s direction, where the map is near-diagonal — a
    # radial and an angular contraction, each a geometric sequence Aitken's Δ² extrapolates on its
    # own; in (κx, κy) the two rates mix and the extrapolation bounces.
    def polar(vx, vy):
        return np.log(np.hypot(vx, vy)), np.angle(np.exp(1j * (np.arctan2(vy, vx) - th0)))

    def cartesian(ell, ang):
        mod = np.exp(ell)
        return mod * np.cos(ang + th0), mod * np.sin(ang + th0)

    def advance(ell, ang):
        vx_, vy_, _ = read(*cartesian(ell, ang))
        return polar(vx_, vy_)

    ell, ang = polar(*fold(start[0], start[1]))
    converged = np.zeros((ny, nx), dtype=bool)
    steps = 0
    while steps < max_iter:
        l1, g1 = advance(ell, ang)
        l2, g2 = advance(l1, g1)
        steps += 2
        with np.errstate(invalid="ignore"):
            settled = np.hypot(l2 - l1, np.angle(np.exp(1j * (g2 - g1)))) <= tol
        acc_l, acc_g = aitken(ell, l1, l2), aitken(ang, g1, g2)
        finite = np.isfinite(acc_l) & np.isfinite(acc_g) & (np.abs(acc_g) <= math.pi / 2)
        ell, ang = np.where(finite, acc_l, l2), np.where(finite, acc_g, g2)
        converged = settled & np.isfinite(l2) & np.isfinite(g2)
        if converged.all():
            break
    vx, vy = cartesian(ell, ang)
    converged &= ~stencil(vx, vy)[4]

    # The wavevector field is smoothed over half the window width before anything is
    # differentiated from it: the phase correction below is a second derivative of the ridge,
    # which turns a ripple δ in κ* into 4ω²δ in K. The smoothing is of the tensor κκᵀ, which the
    # fold leaves alone, weighted by the converged mask; the magnitude is the smoothed trace (the
    # leading eigenvalue under-reads a turning director by ½(σ/r)²), the direction the
    # eigenvector, and the half-plane sign is put back.
    sigma_s = 0.5 * float(omega) / k0

    def smooth(f):
        pad_s = 0 if periodic else int(math.ceil(3.0 * sigma_s))
        fp = np.pad(f, pad_s, mode="reflect") if pad_s else f
        gx_, gy_ = grid(fp.shape)
        g = np.exp(-0.5 * sigma_s * sigma_s * (gx_ * gx_ + gy_ * gy_))
        out = np.real(np.asarray(fft.ifft2(np.asarray(fft.fft2(fp)) * g)))
        return out[pad_s:pad_s + ny, pad_s:pad_s + nx] if pad_s else out

    weight = smooth(converged.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        qxx, qxy, qyy = (smooth(np.where(converged, f, 0.0)) / weight
                         for f in (vx * vx, vx * vy, vy * vy))
        mod_s = np.sqrt(np.maximum(qxx + qyy, 0.0))
        ang_s = 0.5 * np.arctan2(2.0 * qxy, qxx - qyy)
    sx, sy = fold(mod_s * np.cos(ang_s), mod_s * np.sin(ang_s))
    keep = np.isfinite(sx) & np.isfinite(sy) & (weight > 1e-3)
    vx, vy = np.where(keep, sx, vx), np.where(keep, sy, vy)
    i0, j0, fi, fj, outside = stencil(vx, vy)
    converged &= ~outside

    # The phase Hessian is that of the ridge phase, whose gradient is the fixed point itself:
    # H = ∇κ*, by central differences with each neighbour folded to the pixel's hemisphere (a
    # fixed node's own Ψ_ij would read curvature flattened by 1/(1 + a⁴Ψ″²)). A pixel whose
    # neighbours are not converged reads them, so it is not converged either.
    def odd_gradient(f, axis):
        """∂f along ``axis`` of a field that changes sign with the fold, neighbours folded first."""
        fwd, bwd = np.roll(f, -1, axis), np.roll(f, 1, axis)
        s_fwd = np.sign(np.roll(vx, -1, axis) * vx + np.roll(vy, -1, axis) * vy)
        s_bwd = np.sign(np.roll(vx, 1, axis) * vx + np.roll(vy, 1, axis) * vy)
        g = 0.5 * (s_fwd * fwd - s_bwd * bwd)
        if not periodic:
            first, last = [slice(None)] * 2, [slice(None)] * 2
            first[axis], last[axis] = 0, -1
            g[tuple(first)] = (s_fwd * fwd)[tuple(first)] - f[tuple(first)]
            g[tuple(last)] = f[tuple(last)] - (s_bwd * bwd)[tuple(last)]
        return g

    around = np.pad(converged, 1, mode="wrap" if periodic else "edge")
    converged &= around[:-2, 1:-1] & around[2:, 1:-1] & around[1:-1, :-2] & around[1:-1, 2:]
    hxx, hyy = odd_gradient(vx, 1), odd_gradient(vy, 0)
    hxy = 0.5 * (odd_gradient(vx, 0) + odd_gradient(vy, 1))
    mod = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        c, s = vx / mod, vy / mod
        u2 = (float(omega) / mod) ** 2
    # The Gaussian window about the ridge, N = R diag(α) Rᵀ − i a²H in the lab frame (first wavelet
    # axis along K): T at a node misfit by d = κ* − κ_node is (A/2) e^{iφ} e^{−½ a² dᵀN⁻¹d} / √det N.
    n11, n22, n12 = a1 * c * c + a2 * s * s, a1 * s * s + a2 * c * c, (a1 - a2) * c * s
    det = (n11 - 1j * u2 * hxx) * (n22 - 1j * u2 * hyy) - (n12 - 1j * u2 * hxy) ** 2
    curvature = np.sqrt(np.abs(det)) / math.sqrt(a1 * a2)
    chi = 0.5 * np.angle(det)

    def quadratic(dx, dy):
        """a² dᵀN⁻¹d, and its flat-window value a²(d∥²/α₁ + d⊥²/α₂)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            q_full = u2 * ((n22 - 1j * u2 * hyy) * dx * dx - 2 * (n12 - 1j * u2 * hxy) * dx * dy
                           + (n11 - 1j * u2 * hxx) * dy * dy) / det
        along, across = dx * c + dy * s, dy * c - dx * s
        return q_full, u2 * (along * along / a1 + across * across / a2)

    # Pass 2: the ridge value read between its four nodes. Each node's modulus is read as the
    # amplitude it implies — the flat misfit undone by the q-envelope itself, the curvature and
    # its coupling to the misfit by the Gaussian window — and its phase shifted by the window's
    # ½ Im(a² dᵀN⁻¹d); the conjugate node (j = n_o) flips the phase.
    weights = {}
    for i, j, w in corners(i0, j0, fi, fj):
        flip = j == n_o
        key = i * n_o + np.where(flip, 0, j)
        for k in np.unique(key[w > 0]):
            here = (key == k) & (w > 0)
            direct, conj = weights.setdefault(int(k), (np.zeros((ny, nx)), np.zeros((ny, nx))))
            direct[here & ~flip] += w[here & ~flip]
            conj[here & flip] += w[here & flip]
    amplitude = np.zeros((ny, nx))
    phasor = np.zeros((ny, nx), dtype=np.complex128)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for key, (direct, conj) in weights.items():
            kappa = node(key // n_o, key % n_o)
            t = transform(kappa, (1.0,))[0][crop]
            both = direct + conj
            sign = np.where(conj > 0, -1.0, 1.0)             # the conjugate node reads −κ*
            q_full, q_flat = quadratic(sign * vx - kappa[0], sign * vy - kappa[1])
            amp = (2.0 * np.abs(t) / wavelet_hat(sign * vx, sign * vy, kappa, **wavelet)
                   * curvature * np.exp(0.5 * np.real(q_full - q_flat)))
            amplitude += np.where(both > 0, both * amp, 0.0)
            ph = t / np.abs(t) * np.exp(0.5j * sign * np.imag(q_full))
            phasor += direct * ph + conj * np.conj(ph)
    chi_x, chi_y = odd_gradient(chi, 1), odd_gradient(chi, 0)
    div_k = hxx + hyy + odd_gradient(chi_x, 1) + odd_gradient(chi_y, 0)

    with np.errstate(invalid="ignore"):
        off = np.abs(np.angle(np.exp(1j * (np.arctan2(vy, vx) - th0))))
    boundary = off > math.pi / 2 - step
    record = {
        "method": "wavelet_ridge", "wavelet": "q_morlet", "omega": float(omega), "xi_deg": 0.0,
        "alpha": [a1, a2], "q": float(q), "q_beta": float(q_beta), "zero_mean": bool(zero_mean),
        "k0_vector": [float(k0_vec[0]), float(k0_vec[1])],
        "bank": {"bracket": [lo, hi], "voices": int(voices), "orientations": n_o, "nodes": n_s * n_o},
        "tolerance": float(tol), "max_steps": int(max_iter), "steps": int(steps),
        "n_unconverged": int(np.count_nonzero(~converged)), "n_boundary": int(np.count_nonzero(boundary)),
        "stationary_phase": "Gaussian (A.8, A.19)" + ("" if q == 1.0 else "; exact at q = 1 only"),
        "wavevector_smoothing_px": sigma_s, "reach_px": pad,
    }
    return Ridge(amplitude=amplitude, phase=np.angle(phasor * np.exp(1j * chi)),
                 kx=vx + chi_x, ky=vy + chi_y, div_k=div_k, scale=float(omega) / mod,
                 curvature=curvature, converged=converged, boundary=boundary, iterations=int(steps),
                 record=record)
