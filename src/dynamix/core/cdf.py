# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Complex cross-diffusion filtering (CDF: linear LCDF + nonlinear NCDF) -- pure numpy, NO GUI.

The module is named for the FAMILY (the scripts' own convention -- 25_cdf_edges.py covers
both variants); the functions keep their variant-specific names.

Lifted from ``research/reconstruction`` scripts 24-27 (lcdf section; the corrected
verification is script 28). Function bodies are VERBATIM with one recorded edit class:
the scripts' module constants ``THETA``/``DT`` become the ``theta``/``dt`` parameters --
``tests/test_cdf_port.py`` reverses the recorded edits and byte-compares each body
against its origin script, and pins script 28's own gates functionally (periodic
explicit-Euler == the exact discrete symbol ``(1 + dt c lam(w))^n`` at machine precision;
the Ricker-CWT identity to single-digit percent; NCDF at huge k == LCDF exactly).

The physics (Gilboa small-theta complex diffusion; Barbeiro 2023 for the scale-space
reading): ``Re(I)`` evolves as the Gaussian-smoothed image, ``Im(I)/theta ~ t * LoG`` --
ONE evolution is a Mexican-hat scale space, so grad-Re NMS maxima are the M-Z-convention
edges and Im zero crossings are the Marr edges, per snapshot. ``sigma_iters`` is script
26's schedule: the iteration count that parks the evolution at effective Gaussian scale
sigma, ``n = sigma^2 / (2 cos theta) / dt``. NCDF is the Perona-Malik-style variant: the
Im channel (the edge detector) modulates its own diffusivity through ``k``.

Scales-in-pixels law: ``sigma`` below is the effective Gaussian standard deviation in px.
"""
from __future__ import annotations

import numpy as np

__all__ = ["lap", "lap_per", "lcdf_evolve", "lcdf_per", "ncdf", "grad_nms",
           "zero_crossings", "sigma_iters"]


def lap(I):
    Ip = np.pad(I, 1, mode="edge")
    return Ip[:-2, 1:-1] + Ip[2:, 1:-1] + Ip[1:-1, :-2] + Ip[1:-1, 2:] - 4 * I


def lap_per(I):
    return np.roll(I, 1, 0) + np.roll(I, -1, 0) + np.roll(I, 1, 1) + np.roll(I, -1, 1) - 4 * I


def lcdf_evolve(f, n_iter, snapshots=(), *, theta=np.pi / 30, dt=0.2):
    c = np.exp(1j * theta)
    I = f.astype(complex)
    out = {}
    for n in range(1, n_iter + 1):
        I = I + dt * c * lap(I)
        if n in snapshots:
            out[n] = I.copy()
    return I, out


def lcdf_per(f, n_iter, *, theta=np.pi / 30, dt=0.2):
    c = np.exp(1j * theta)
    I = f.astype(complex)
    for _ in range(n_iter):
        I = I + dt * c * lap_per(I)
    return I


def ncdf(f, k, n_iter, *, theta=np.pi / 30, dt=0.2):
    I = f.astype(complex)
    for _ in range(n_iter):
        d = np.exp(1j * theta) / (1.0 + (I.imag / (k * theta)) ** 2)
        Ip = np.pad(I, 1, mode="edge")
        dp = np.pad(d, 1, mode="edge")
        flux = np.zeros_like(I)
        for (dr, dc) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            In = Ip[1 + dr:I.shape[0] + 1 + dr, 1 + dc:I.shape[1] + 1 + dc]
            dn = dp[1 + dr:I.shape[0] + 1 + dr, 1 + dc:I.shape[1] + 1 + dc]
            flux += 0.5 * (d + dn) * (In - I)
        I = I + dt * flux
    return I


def grad_nms(R, floor=0.05):
    gy, gx = np.gradient(R)
    mag = np.hypot(gy, gx)
    ang = np.arctan2(gy, gx)
    d8 = np.round(ang / (np.pi / 4)).astype(int) % 8
    offs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    keep = np.zeros(R.shape, dtype=bool)
    for k8, (dr, dc) in enumerate(offs):
        m = d8 == k8
        fwd = np.roll(np.roll(mag, -dr, 0), -dc, 1)
        bwd = np.roll(np.roll(mag, dr, 0), dc, 1)
        keep |= m & (mag >= fwd) & (mag >= bwd)
    return keep & (mag > floor * mag.max()), mag


def zero_crossings(A, slope_frac=0.05):
    zc = np.zeros(A.shape, dtype=bool)
    for ax in (0, 1):
        zc |= (A * np.roll(A, -1, ax) < 0)
    gy, gx = np.gradient(A)
    slope = np.hypot(gy, gx)
    return zc & (slope > slope_frac * slope.max()), slope


def sigma_iters(sigmas, *, theta=np.pi / 30, dt=0.2) -> dict:
    """Script 26's schedule, as a function: iteration count per effective Gaussian scale,
    ``n = max(1, round(sigma^2 / (2 cos theta) / dt))`` -- the snapshot plan that turns one
    evolution into a dyadic scale stack."""
    return {float(s): max(1, int(round(float(s) ** 2 / (2 * np.cos(theta)) / dt)))
            for s in sigmas}
