# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""η: the share of an analysing wavelet's spectral energy outside the pixel grid's band.

The grid carries the frequencies |k_x|, |k_y| ≤ ½ cycle/px. A wavelet whose band reaches past them is
not resolved at that scale, and the part outside folds back onto the lattice. η is the energy of the
x-component analyser, s_x θ̂(s) with s = k·a in cycles, outside that square over its total, on a k-grid
reaching eight times past Nyquist. It is one measure for every smoother in the registry, spec 20's
widths included, judged against spec 19's level, 1e-3 by default; the shell sets the level from the
user's settings (:func:`configure`).

The Turiel tools' kernels (:mod:`dynamix.core.microcanonical`) built in Fourier space get the same
measure of their own transform, |K̂|², as a labelled diagnostic (:func:`turiel_eta`); their warnings
come from Turiel's own sampling conditions instead.
"""
from __future__ import annotations

import functools

import numpy as np

__all__ = ["configure", "eta", "eta_warn", "turiel_eta"]

#: The warning level η is judged against; spec 19's 1e-3 unless the shell configures another.
_ETA_WARN = [1e-3]

#: Nodes per axis of the k-grid's quadrant. The quadrant reaches ±4 cycles/px (8× Nyquist) for
#: finest scales up to 15 px and shrinks with larger ones (the band sits ever further inside), so
#: the step in s stays below 0.08 everywhere a warning can fire.
_N = 801


def configure(eta_warn: float) -> None:
    """Set the level η is judged against (from the user's settings)."""
    value = float(eta_warn)
    if not value > 0.0:
        raise ValueError(f"the under-sampling warning level must be positive, got {eta_warn!r}")
    _ETA_WARN[0] = value


def eta_warn() -> float:
    """The level η is judged against."""
    return _ETA_WARN[0]


@functools.lru_cache(maxsize=512)
def _eta(name: str, items: tuple, scale_px: float) -> float:
    from dynamix.core.smoothers import smoother

    sm = smoother(name, dict(items) if items else None)
    k = np.linspace(0.0, max(1.0, min(4.0, 60.0 / scale_px)), _N)
    w = np.ones(_N)
    w[0] = 0.5                                   # the axis lines are shared by two quadrants
    kx, ky = np.meshgrid(k, k, indexing="ij")
    e = np.outer(w, w) * (kx * scale_px) ** 2 * np.abs(sm.theta_hat(kx * scale_px,
                                                                    ky * scale_px)) ** 2
    total = float(e.sum())
    if not total > 0.0:
        return 0.0
    return float(e[(kx > 0.5) | (ky > 0.5)].sum() / total)


def eta(name: str, params: "dict | None", scale_px: float) -> float:
    """η of the registry smoother ``name`` (with its ``params``) at ``scale_px``."""
    items = tuple(sorted((params or {}).items()))
    return _eta(str(name), items, round(float(scale_px), 9))


@functools.lru_cache(maxsize=256)
def turiel_eta(route: str, wavelet: str, r: float, *, beta: float = 1.0, q_tsallis: float = 1.5,
               frac_n: float = 2.0, q_beta: "float | None" = None) -> "float | None":
    """η of a Turiel kernel built in Fourier space at zero-crossing radius (or scale) ``r`` px:
    ``route`` "measure" or "marr", ``wavelet`` the tool's own name, resolved as
    ``microcanonical.ricker_projections`` resolves it; ``q_beta`` the width the device resolved.
    None where the Fourier build refuses the kernel."""
    from dynamix.core.microcanonical import (_MARR_ALIASES, _borges_q, _check_construction,
                                             _fourier_kernel, _frac_u0, _q_mexican_stretch)

    q, fn, stretch = float(q_tsallis), float(frac_n), 1.0
    if route == "marr":
        if wavelet == "q_mexican":
            stretch = _q_mexican_stretch(q, q_beta)
            wavelet, q = "q_gaussian", _borges_q(q)
        alias = _MARR_ALIASES.get(wavelet)
        if isinstance(alias, tuple):
            wavelet, fn = alias
        elif alias is not None:
            wavelet = alias
    try:
        _check_construction("fourier", route, wavelet, float(beta), q)
    except ValueError:
        return None
    u0 = _frac_u0(fn) if wavelet == "frac_gaussian" else None
    n, m = 512, 8                                # the same kernel on a grid m times finer
    k_hat = _fourier_kernel((n, n), float(r) * m, route, wavelet, float(beta), q, fn, u0=u0,
                            q_beta=q_beta, stretch=stretch)
    e = np.abs(np.nan_to_num(np.asarray(k_hat, dtype=np.float64))) ** 2
    i = np.minimum(np.arange(n), n - np.arange(n))
    inside = (i[:, None] <= n // (2 * m)) & (i[None, :] <= n // (2 * m))
    total = float(e.sum())
    return float(e[~inside].sum() / total) if total > 0.0 else 0.0
