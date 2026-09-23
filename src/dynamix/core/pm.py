# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Perona-Malik anisotropic diffusion -- the 1990 paper's own scheme, pure numpy, NO GUI.

Perona & Malik, *Scale-space and edge detection using anisotropic diffusion*, IEEE Trans.
PAMI 12(7):629-639 (1990): scheme (7)+(8)+(10) verbatim -- 4-nearest-neighbor arcs,
conduction coefficient ``g(|neighbor difference|)`` per arc (their eq. (10) form: the
brightness-conserving one their experiments use), update ``I += lam * sum(c * dI)`` with
``0 <= lam <= 1/4`` for stability, adiabatic boundaries via edge replication. Both paper
nonlinearities ship: ``g="exp"`` is ``exp(-(s/K)^2)`` (privileges high-contrast edges),
``g="frac"`` is ``1/(1+(s/K)^2)`` (privileges wide regions) -- their own reading of the
difference. This is the REAL-diffusivity, gradient-driven flow: distinct from
:mod:`dynamix.core.cdf`'s complex family, whose edge signal is the Im channel; PM's edge
enhancement (backward diffusion past the flux peak ``phi(s) = s*g(s)``, stabilized by the
discrete max principle) is exactly what that family does not do.

Scales-in-pixels law: PM is nonlinear, so no true Gaussian sigma exists. ``sigma_iters``
gives the NOMINAL schedule -- the iteration count that reaches effective Gaussian scale
sigma in the K -> inf (heat-equation) limit, ``n = sigma^2 / 2 / lam`` -- so snapshots
align with the other devices' dyadic sigma ladders at matched nominal scale.
"""
from __future__ import annotations

import numpy as np

__all__ = ["pm_evolve", "sigma_iters"]

_OFFS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def pm_evolve(f, k, n_iter, snapshots=(), *, g="exp", lam=0.2):
    if g == "exp":
        cond = lambda d: np.exp(-((d / k) ** 2))
    elif g == "frac":
        cond = lambda d: 1.0 / (1.0 + (d / k) ** 2)
    else:
        raise ValueError(f"unknown g {g!r}: expected 'exp' or 'frac'")
    I = np.asarray(f, dtype=np.float64).copy()
    out = {}
    for n in range(1, n_iter + 1):
        Ip = np.pad(I, 1, mode="edge")
        flux = np.zeros_like(I)
        for dr, dc in _OFFS:
            d = Ip[1 + dr:I.shape[0] + 1 + dr, 1 + dc:I.shape[1] + 1 + dc] - I
            flux += cond(d) * d
        I = I + lam * flux
        if n in snapshots:
            out[n] = I.copy()
    return I, out


def sigma_iters(sigmas, *, lam=0.2) -> dict:
    """Iteration count per NOMINAL effective Gaussian scale sigma (the K -> inf limit,
    where n steps of size lam reach sigma = sqrt(2 n lam)): ``n = max(1, round(sigma^2 /
    2 / lam))`` -- the snapshot plan that turns one evolution into a dyadic scale stack."""
    return {float(s): max(1, int(round(float(s) ** 2 / 2.0 / lam)))
            for s in sigmas}
