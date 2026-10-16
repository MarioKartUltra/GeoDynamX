# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Nonconcave multifractal spectra (Touchette & Beck 2005, arXiv cond-mat/0507379 §V).

The Legendre (canonical) route from τ(q) to D(h) can only return a concave spectrum: across a
nonconcave dip it returns the concave envelope, a straight segment, and h(q) jumps over that
segment at one q*. Two app-wide options, off by default, for any method that supplies
per-scale coefficients (WTMM chain maxima, leaders):

- :func:`kink_flags` -- the jump in the finite-difference h(q) = dτ/dq, with the D(h) interval the
  envelope bridges there;
- :func:`gaussian_ensemble` -- the generalised (Gaussian) canonical ensemble by direct
  Boltzmann-weight fits, no Legendre transform: W_i ∝ |T_i|^q exp(γ (ln|T_i|)² / ln(a/L)), whose
  saddle f'(h) = q + 2γh is a minimum for 2γ > f''(h), so γ reaches points inside the dip.

Scales are in pixels; ``L`` (``length_px``) is the integral scale, so ``ln(a/L) < 0`` plays the
part of T&B's small ε and h_i ≈ ln|T_i| / ln(a/L). A coefficient's amplitude C biases h_i by
ln C / ln(a/L), more at coarse scales: fit over the same window as the standard spectrum.
"""
from __future__ import annotations

import numpy as np

__all__ = ["gaussian_ensemble", "kink_flags"]


def _slopes(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Least-squares slopes of ``Y[..., k]`` against ``x[k]``."""
    xc = x - x.mean()
    return np.tensordot(Y - Y.mean(axis=-1, keepdims=True), xc, axes=([-1], [0])) / np.sum(xc * xc)


def gaussian_ensemble(mods_per_scale, scales_px, q_list, gamma_list, log2_a_min: float,
                      log2_a_max: float, *, counts_per_scale=None,
                      length_px: float | None = None) -> dict:
    """h, D, ⟨h²⟩ and τ on the (q, γ) grid by direct Boltzmann-weight fits over the scales whose
    log2 a lies in ``[log2_a_min, log2_a_max]``.

    Per scale a, with T_i the positive finite coefficients (``counts_per_scale``, when given, is
    each one's multiplicity) and x = ln(a/L): W_i = |T_i|^q exp(γ (ln|T_i|)²/x) / Z. Then, as
    slopes against ln a: h = Σ W ln|T|, D = Σ W ln W, ⟨h²⟩ = Σ W (ln|T|)²/x, τ = ln Z. Per scale
    Σ W ln W = q Σ W ln|T| + γ Σ W (ln|T|)²/x − ln Z exactly, so ``residual`` = D − (q h + γ ⟨h²⟩
    − τ) is round-off. γ = 0 is the standard direct fit (Arneodo). Returns ``{"q", "gamma", "h",
    "D", "h2", "tau", "residual"}`` with ``(n_q, n_gamma)`` tables; NaN when fewer than two scales
    fall in the window."""
    q = np.asarray(q_list, dtype=np.float64)
    gam = np.asarray(gamma_list, dtype=np.float64)
    a = np.asarray(scales_px, dtype=np.float64)
    L = float(length_px) if length_px is not None else 4.0 * float(a.max())
    if np.any(a >= L):
        raise ValueError(f"every scale must lie below the integral scale L = {L}")
    keep = [k for k, s in enumerate(a) if log2_a_min <= np.log2(s) <= log2_a_max]
    shape = (q.size, gam.size)
    if len(keep) < 2:
        nan = np.full(shape, np.nan)
        return {"q": q, "gamma": gam, "h": nan, "D": nan.copy(), "h2": nan.copy(),
                "tau": nan.copy(), "residual": nan.copy()}
    S_h, S_D, S_h2, ln_Z = (np.empty(shape + (len(keep),)) for _ in range(4))
    for j, k in enumerate(keep):
        m = np.asarray(mods_per_scale[k], dtype=np.float64)
        c = (np.ones_like(m) if counts_per_scale is None
             else np.asarray(counts_per_scale[k], dtype=np.float64))
        ok = np.isfinite(m) & (m > 0) & (c > 0)
        ln_t, ln_c = np.log(m[ok]), np.log(c[ok])
        x = np.log(a[k] / L)
        lw = (ln_c[None, None, :] + q[:, None, None] * ln_t[None, None, :]
              + gam[None, :, None] * (ln_t * ln_t / x)[None, None, :])
        top = lw.max(axis=-1, keepdims=True)
        log_z = top[..., 0] + np.log(np.exp(lw - top).sum(axis=-1))
        w = np.exp(lw - log_z[..., None])
        S_h[..., j] = (w * ln_t).sum(axis=-1)
        S_D[..., j] = (w * (lw - log_z[..., None] - ln_c)).sum(axis=-1)
        S_h2[..., j] = (w * (ln_t * ln_t / x)).sum(axis=-1)
        ln_Z[..., j] = log_z
    la = np.log(a[keep])
    h, D, h2, tau = (_slopes(la, S) for S in (S_h, S_D, S_h2, ln_Z))
    residual = D - (q[:, None] * h + gam[None, :] * h2 - tau)
    return {"q": q, "gamma": gam, "h": h, "D": D, "h2": h2, "tau": tau, "residual": residual}


def kink_flags(q_list, tau, *, factor: float = 4.0, floor: float = 0.02) -> list:
    """The first-order transitions in τ(q): where the finite-difference h(q) = dτ/dq jumps by more
    than ``factor`` times its typical step and by more than ``floor``. Adjacent jumping cells are
    one kink. Each is ``{"q", "h_lo", "h_hi", "jump"}``: the q* of the jump and the h interval the
    Legendre spectrum bridges with a straight segment there (the affine D(h) piece)."""
    q = np.asarray(q_list, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    ok = np.isfinite(q) & np.isfinite(t)
    q, t = q[ok], t[ok]
    if q.size < 4:
        return []
    h = np.diff(t) / np.diff(q)                    # at the cell midpoints
    qm = 0.5 * (q[1:] + q[:-1])
    jumps = np.diff(h)
    typical = float(np.median(np.abs(jumps)))
    hot = np.abs(jumps) > max(factor * typical, floor)
    flags, i = [], 0
    while i < hot.size:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j + 1 < hot.size and hot[j + 1]:
            j += 1
        lo, hi = sorted((float(h[i]), float(h[j + 1])))
        flags.append({"q": float(0.5 * (qm[i] + qm[j + 1])), "h_lo": lo, "h_hi": hi,
                      "jump": hi - lo})
        i = j + 1
    return flags
