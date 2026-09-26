# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Anisotropy of a WTMM skeleton -- Arnéodo, Decoster & Roux 2000 (Eur. Phys. J. B 15:567,
§6 and Figs 24–26).

At each scale ``a`` the WTMMM (the maxima of the modulus along each maxima line: xsmurf's
``ssm``) carry a modulus M and an argument A. Their joint pdf P_a(M, A) says:

- P_a(A) FLAT -> isotropic scale invariance; non-uniform -> anisotropy, and its change across
  scales says whether the anisotropy strengthens or weakens (Fig. 24b);
- the WTMMM in the gradient plane (T_ψ1, T_ψ2) = (M cos A, M sin A): radial symmetry ->
  isotropy (Fig. 25);
- the pdf of M conditioned on A mod π in 4 sectors: shape invariance -> M and A independent,
  so the singularity spectrum carries no directional information (Fig. 26).

Pure numpy: angles come in as an :class:`~dynamix.core.orientation.Orientation`, so the same
statistics run in the pixel frame (the paper's A) or as bearings.
"""
from __future__ import annotations

import numpy as np

from dynamix.core.orientation import Orientation, axial_sector

__all__ = ["SECTOR_LABELS", "angle_pdf", "gradient_plane", "sector_modulus_pdfs",
           "wtmmm_by_scale"]

#: Fig. 26's sectors (orientation mod 180°, ±22.5°), and their compass reading as bearings.
SECTOR_LABELS = {"pixel": ("0°", "45°", "90°", "135°"),
                 "azimuth": ("N–S", "NE–SW", "E–W", "SE–NW")}


def wtmmm_by_scale(extrema: list) -> list:
    """The WTMMM at every scale (xsmurf ``ssm``, the exact port in the WTMM backend): one
    ``{"x", "y", "mod", "arg", "line_id"}`` dict per scale."""
    from dynamix.core.wtmm_backend import PythonWTMMBackend

    return PythonWTMMBackend().single_maxima2d(list(extrema))


def angle_pdf(o: Orientation, bins: int = 36) -> tuple:
    """``(centres_deg, pdf)`` of the angles over one full turn, normalised so the pdf
    integrates to 1 over DEGREES -- a flat (isotropic) pdf reads ``1/360``. PIXEL angles are
    binned over (−180, 180]; azimuths over [0, 360)."""
    d = np.asarray(o.degrees, dtype=np.float64).ravel()
    d = d[np.isfinite(d)]
    lo = -180.0 if o.frame.value == "pixel" else 0.0
    edges = np.linspace(lo, lo + 360.0, int(bins) + 1)
    counts, _ = np.histogram(np.mod(d - lo, 360.0) + lo, bins=edges)
    width = 360.0 / int(bins)
    pdf = counts / (max(d.size, 1) * width)
    return (edges[:-1] + edges[1:]) / 2.0, pdf


def gradient_plane(mod, arg_rad) -> tuple:
    """``(T_ψ1, T_ψ2) = (M cos A, M sin A)`` -- the wavelet gradient of each WTMMM (PIXEL A)."""
    m = np.asarray(mod, dtype=np.float64)
    a = np.asarray(arg_rad, dtype=np.float64)
    return m * np.cos(a), m * np.sin(a)


def sector_modulus_pdfs(mod, o: Orientation, *, n: int = 4, bins: int = 30,
                        log: bool = True) -> list:
    """``[(sector, centres, pdf, count)]``: the pdf of M (of log2 M when ``log``) within each
    axial sector of the angles, on SHARED bin edges so the curves compare directly. A sector
    holding nothing gives an all-zero pdf and ``count = 0``."""
    m = np.asarray(mod, dtype=np.float64).ravel()
    ok = np.isfinite(m) & (m > 0 if log else np.ones(m.shape, bool))
    val = np.log2(m[ok]) if log else m[ok]
    sec = axial_sector(Orientation(np.asarray(o.degrees).ravel()[ok], o.frame), n)
    if val.size == 0:
        return [(k, np.zeros(bins), np.zeros(bins), 0) for k in range(n)]
    edges = np.linspace(val.min(), val.max() if val.max() > val.min() else val.min() + 1.0,
                        int(bins) + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    out = []
    for k in range(int(n)):
        v = val[sec == k]
        counts, _ = np.histogram(v, bins=edges)
        pdf = counts / (max(v.size, 1) * (edges[1] - edges[0]))
        out.append((k, centres, pdf, int(v.size)))
    return out
