# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The reach of a scale: the radius holding all but ``eps`` of the analysing wavelet's |ψ| mass.

ψ is the x-component analyser as the transform samples it, IFFT(i s_x θ̂(s)) on an n × n grid. Where
θ̂ is not negligible at Nyquist, truncating it there makes the sampled kernel ring, and the ringing's
mass grows with the grid: the radius found on a 2048 grid is about twice the 1024 one. Where the
kernel is resolved the two agree. The reach exists when they agree within :data:`RATIO_LIMIT`; the
measured ratios sit close to 1 or close to 2. η (:mod:`dynamix.core.undersampling`) does not decide
this: it is 1e-6 to 1e-8 at scales whose mass radius still doubles with the grid.

Scales above :data:`S_REF` take the reference scale's radius times ``s / S_REF``: a resolved kernel
dilates with its scale, and measuring them directly would need grids past 4096.
"""
from __future__ import annotations

import dataclasses
import functools

import numpy as np

__all__ = ["GRID", "RATIO_LIMIT", "Reach", "S_REF", "measured_radius", "reach"]

#: The smaller of the two grids the radius is measured on; the larger is twice it.
GRID = 1024
#: Largest ratio of the larger grid's radius to the smaller one's that still counts as converged.
RATIO_LIMIT = 1.25
#: Scales above this (px) are scaled from it.
S_REF = 32.0
#: Ring width of the radial mass histogram, px.
_RING = 0.25


@dataclasses.dataclass(frozen=True)
class Reach:
    """``radius`` is None where the reach does not converge; ``measured`` is the larger grid's
    radius either way (None where the smoother refuses its parameters), ``ratio`` the larger
    grid's radius over the smaller's, ``grids`` the two grid sizes."""

    scale_px: float
    eps: float
    radius: "float | None"
    measured: "float | None"
    ratio: "float | None"
    grids: tuple
    converged: bool


@functools.lru_cache(maxsize=4)
def _rings(n: int):
    """The ring index of every pixel of an n × n IFFT output (origin at [0, 0]) and the count."""
    d = np.fft.fftfreq(n) * n
    idx = np.floor(np.hypot(d[:, None], d[None, :]) / _RING).astype(np.int32).ravel()
    return idx, int(idx.max()) + 1


@functools.lru_cache(maxsize=2048)
def measured_radius(name: str, items: tuple, scale_px: float, eps: float, n: int) -> float:
    """The radius (px, to a :data:`_RING`) outside which ``eps`` of the |ψ| mass lies, ψ the
    smoother's x-component analyser at ``scale_px`` sampled on an n × n grid; ``items`` are its
    params as sorted pairs."""
    from dynamix.core import fft_policy
    from dynamix.core.smoothers import smoother

    sm = smoother(name, dict(items) if items else None)
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.rfftfreq(n)[None, :]
    spec = 1j * (kx * scale_px) * sm.theta_hat(kx * scale_px, ky * scale_px)
    psi = np.asarray(fft_policy.double().irfft2(spec, s=(n, n)))
    idx, n_rings = _rings(n)
    mass = np.bincount(idx, weights=np.abs(psi).ravel(), minlength=n_rings)
    tail = np.cumsum(mass[::-1])[::-1] / mass.sum()
    return float((np.flatnonzero(tail > eps)[-1] + 1) * _RING)


def reach(name: str, params: "dict | None", scale_px: float, eps: float = 1e-3) -> Reach:
    """The reach of smoother ``name`` (with ``params``) at ``scale_px``, to ``eps`` of its mass."""
    s = float(scale_px)
    ref = round(min(s, S_REF), 9)
    items = tuple(sorted((params or {}).items()))
    grids = (GRID, 2 * GRID)
    try:
        r1 = measured_radius(str(name), items, ref, float(eps), grids[0])
        r2 = measured_radius(str(name), items, ref, float(eps), grids[1])
    except ValueError:                           # a q the smoother refuses: the run says so
        return Reach(s, float(eps), None, None, None, grids, False)
    ratio = r2 / r1
    k = s / ref
    ok = ratio <= RATIO_LIMIT
    return Reach(s, float(eps), r2 * k if ok else None, r2 * k, ratio, grids, ok)
