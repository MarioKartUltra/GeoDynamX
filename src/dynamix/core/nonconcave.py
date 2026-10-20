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
  Boltzmann-weight fits, no Legendre transform: W_i ∝ |T_i|^q exp(γ (ln|T_i| − δ ln(a/L))² / ln(a/L)),
  i.e. (a/L)^{q h_i + γ (h_i − δ)²}, whose saddle f'(h) = q + 2γ(h − δ) is a minimum for
  2γ > f''(h), so γ reaches points inside the dip. The shift δ is Leonarduzzi, Abry, Wendt,
  Jaffard and Touchette 2019 (26), g = −γ(h − δ)²: T&B's γ and theirs are the same number (the
  minus signs of 2^{−j(qφ − g)} cancel); a narrow q with a sweep of δ over the spectrum's
  support is their recipe (§III-C3), the wide q being unstable. Their (18)–(19) also take the
  amplitude out with c₁,₀, the intercept of the per-scale mean of ln|T| against ln(a/L):
  ``reference_amplitude="auto"``.

Scales are in pixels; ``L`` (``length_px``) is the integral scale, so ``ln(a/L) < 0`` plays the
part of T&B's small ε and h_i ≈ ln|T_i| / ln(a/L). A coefficient's amplitude C biases h_i by
ln C / ln(a/L), more at coarse scales: fit over the same window as the standard spectrum.
"""
from __future__ import annotations

import numpy as np

__all__ = ["gaussian_ensemble", "generalized_legendre", "kink_flags"]


def _slopes(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Least-squares slopes of ``Y[..., k]`` against ``x[k]``."""
    xc = x - x.mean()
    return np.tensordot(Y - Y.mean(axis=-1, keepdims=True), xc, axes=([-1], [0])) / np.sum(xc * xc)


def gaussian_ensemble(mods_per_scale, scales_px, q_list, gamma_list, log2_a_min: float,
                      log2_a_max: float, *, counts_per_scale=None,
                      length_px: float | None = None, reference_amplitude="auto",
                      delta_list=None, cancel=None) -> dict:
    """h, D, ⟨(h − δ)²⟩ and τ on the (q, γ) grid, or the (q, γ, δ) grid when ``delta_list`` is
    given, by direct Boltzmann-weight fits over the scales whose log2 a lies in
    ``[log2_a_min, log2_a_max]``.

    Per scale a, with T_i the positive finite coefficients (``counts_per_scale``, when given, is
    each one's multiplicity) and x = ln(a/L): W_i = |T_i|^q exp(γ (ln|T_i| − δx)²/x) / Z. Then,
    as slopes against ln a: h = Σ W ln|T|, D = Σ W ln W, ⟨(h−δ)²⟩ = Σ W (ln|T| − δx)²/x, τ = ln Z.
    Per scale Σ W ln W = q Σ W ln|T| + γ Σ W (ln|T| − δx)²/x − ln Z exactly, so ``residual`` =
    D − (q h + γ h2 − τ) is round-off. This identity does not validate the spectrum: weighted
    means can mix disjoint populations. Use :func:`generalized_legendre` for the generalized
    inversion. ``reference_amplitude`` divides the input before taking logarithms: a number in
    the measurement's units, or ``"auto"``, 2^{c₁,₀}, the intercept of the per-scale mean of
    ln|T| against x over the window's scales (Leonarduzzi et al. 2019 (18)); the one used is
    returned. Empty levels are omitted. γ = 0 is the standard direct fit (Arneodo). Returns
    ``{"q", "gamma", "delta", "h", "D", "h2", "tau", "residual", "reference_amplitude"}`` with
    ``(n_q, n_gamma)`` tables, ``(n_q, n_gamma, n_delta)`` when ``delta_list`` is given; NaN when
    fewer than two scales fall in the window."""
    q = np.asarray(q_list, dtype=np.float64)
    gam = np.asarray(gamma_list, dtype=np.float64)
    a = np.asarray(scales_px, dtype=np.float64)
    with_delta = delta_list is not None
    dl = np.asarray([0.0] if delta_list is None else delta_list, dtype=np.float64)
    if dl.ndim != 1 or dl.size == 0 or not np.isfinite(dl).all():
        raise ValueError("delta must be a nonempty finite grid")
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a) & (a > 0)) or np.unique(a).size != a.size:
        raise ValueError("scales must be distinct, finite and positive")
    if len(mods_per_scale) != a.size or (counts_per_scale is not None and len(counts_per_scale) != a.size):
        raise ValueError("one coefficient set and multiplicity set per scale is required")
    auto_reference = isinstance(reference_amplitude, str)
    if auto_reference and reference_amplitude != "auto":
        raise ValueError("reference amplitude must be a positive number or 'auto'")
    if not auto_reference:
        reference_amplitude = float(reference_amplitude)
        if reference_amplitude <= 0 or not np.isfinite(reference_amplitude):
            raise ValueError("reference amplitude must be finite and positive")
    if q.ndim != 1 or gam.ndim != 1 or q.size == 0 or gam.size == 0 or not np.isfinite(q).all() or not np.isfinite(gam).all():
        raise ValueError("q and gamma must be nonempty finite grids")
    L = float(length_px) if length_px is not None else 4.0 * float(a.max())
    if not np.isfinite(L) or np.any(a >= L):
        raise ValueError(f"every scale must lie below the integral scale L = {L}")
    keep = [k for k, s in enumerate(a) if log2_a_min <= np.log2(s) <= log2_a_max]
    samples = {}
    for k in keep:
        m = np.asarray(mods_per_scale[k], dtype=np.float64).ravel()
        c = np.ones_like(m) if counts_per_scale is None else np.asarray(counts_per_scale[k], dtype=np.float64).ravel()
        if c.shape != m.shape:
            raise ValueError("multiplicity shape must match the coefficients")
        ok = np.isfinite(m) & (m > 0) & np.isfinite(c) & (c > 0)
        if ok.any():
            samples[k] = np.log(m[ok]), np.log(c[ok])
    keep = [k for k in keep if k in samples]
    if auto_reference:
        # c₁,₀: the intercept of the multiplicity-weighted mean of ln|T| against ln(a/L).
        xs = [np.log(a[k] / L) for k in keep]
        means = [float((np.exp(ln_c) * ln_t).sum() / np.exp(ln_c).sum()) for ln_t, ln_c in (samples[k] for k in keep)]
        reference_amplitude = float(np.exp(np.polyfit(xs, means, 1)[1])) if len(keep) >= 2 else 1.0
    samples = {k: (ln_t - np.log(reference_amplitude), ln_c) for k, (ln_t, ln_c) in samples.items()}
    shape = (q.size, gam.size, dl.size)
    squeeze = (lambda t: t[..., 0]) if not with_delta else (lambda t: t)
    if len(keep) < 2:
        nan = np.full(shape, np.nan)
        return {"q": q, "gamma": gam, "delta": dl, "h": squeeze(nan), "D": squeeze(nan.copy()),
                "h2": squeeze(nan.copy()), "tau": squeeze(nan.copy()), "residual": squeeze(nan.copy()),
                "tau_r2": squeeze(nan.copy()), "n_scales": len(keep), "scales": a[keep],
                "log_partition": squeeze(np.full(shape + (len(keep),), np.nan)),
                "reference_amplitude": reference_amplitude}
    S_h, S_D, S_h2, ln_Z = (np.empty(shape + (len(keep),)) for _ in range(4))
    qq = np.repeat(q, gam.size * dl.size)
    gg = np.tile(np.repeat(gam, dl.size), q.size)
    dd = np.tile(dl, q.size * gam.size)
    for j, k in enumerate(keep):
        ln_t, ln_c = samples[k]
        x = np.log(a[k] / L)
        batch = max(1, 1_000_000 // ln_t.size)
        for start in range(0, qq.size, batch):
            if cancel is not None and cancel():
                raise InterruptedError("generalized ensemble cancelled")
            stop = min(start+batch, qq.size)
            shifted = ln_t[None, :] - dd[start:stop, None] * x          # ln|T| − δ x
            lw = ln_c + qq[start:stop, None]*ln_t + gg[start:stop, None]*(shifted*shifted/x)
            top = lw.max(axis=-1, keepdims=True)
            log_z = top[:, 0] + np.log(np.exp(lw-top).sum(axis=-1))
            w = np.exp(lw-log_z[:, None])
            S_h.reshape(-1, len(keep))[start:stop, j] = (w*ln_t).sum(axis=-1)
            S_D.reshape(-1, len(keep))[start:stop, j] = (w*(lw-log_z[:, None]-ln_c)).sum(axis=-1)
            S_h2.reshape(-1, len(keep))[start:stop, j] = (w*(shifted*shifted/x)).sum(axis=-1)
            ln_Z.reshape(-1, len(keep))[start:stop, j] = log_z
    la = np.log(a[keep])
    h, D, h2, tau = (_slopes(la, S) for S in (S_h, S_D, S_h2, ln_Z))
    residual = D - (q[:, None, None] * h + gam[None, :, None] * h2 - tau)
    predicted = ln_Z.mean(axis=-1, keepdims=True) + tau[..., None] * (la-la.mean())
    ss = ((ln_Z-ln_Z.mean(axis=-1, keepdims=True))**2).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1-((ln_Z-predicted)**2).sum(axis=-1)/ss
    return {"q": q, "gamma": gam, "delta": dl, "h": squeeze(h), "D": squeeze(D), "h2": squeeze(h2),
            "tau": squeeze(tau), "residual": squeeze(residual), "tau_r2": squeeze(r2),
            "log_partition": squeeze(ln_Z), "scales": a[keep], "n_scales": len(keep),
            "reference_amplitude": reference_amplitude}


def generalized_partition_table(mods_per_scale, scales_px, q_list, gamma: float, delta: float = 0.0, *,
                                counts_per_scale=None, length_px: float | None = None,
                                reference_amplitude="auto", chunk: int = 1 << 18) -> dict:
    """The generalized structure functions of Leonarduzzi et al. 2019 (20)–(21) in the partition
    schema the fitters read, one (γ, δ) at a time. Per scale a, with x = ln(a/L), the generalized
    weight of a coefficient is P_i = c_i |T_i|^q exp(γ (ln|T_i| − δx)²/x) (Touchette & Beck's
    ε^{qα + γα²} at δ = 0, c_i its multiplicity, the cells per sample); ``tau_qa`` = log2 Σ P,
    ``h_qa`` = Σ W ln|T| / ln 2, ``h2_qa`` = Σ W (ln|T| − δx)²/x / ln 2 and ``D_qa`` = Σ W log2(W/c),
    which is q h + γ h2 − τ per scale exactly, with W = P / Σ P; the slopes against log2 a are the
    ensemble's τ(q, γ), h and D (:func:`gaussian_ensemble`), and at γ = 0 the table is the
    ordinary one. ``reference_amplitude`` as in :func:`gaussian_ensemble` (``"auto"``: c₁,₀ over
    every scale given). The σ columns follow :func:`dynamix.core.microcanonical.partition_tables`
    with P in place of |T|^q. Every scale given is tabulated; the fitter chooses its window."""
    q = np.asarray(q_list, dtype=np.float64)
    a = np.asarray(scales_px, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a) & (a > 0)) or np.unique(a).size != a.size:
        raise ValueError("scales must be distinct, finite and positive")
    if len(mods_per_scale) != a.size or (counts_per_scale is not None and len(counts_per_scale) != a.size):
        raise ValueError("one coefficient set and multiplicity set per scale is required")
    L = float(length_px) if length_px is not None else 4.0 * float(a.max())
    if not np.isfinite(L) or np.any(a >= L):
        raise ValueError(f"every scale must lie below the integral scale L = {L}")
    gamma, delta = float(gamma), float(delta)
    samples = {}
    for k in range(a.size):
        m = np.asarray(mods_per_scale[k], dtype=np.float64).ravel()
        c = np.ones_like(m) if counts_per_scale is None else np.asarray(counts_per_scale[k], dtype=np.float64).ravel()
        if c.shape != m.shape:
            raise ValueError("multiplicity shape must match the coefficients")
        ok = np.isfinite(m) & (m > 0) & np.isfinite(c) & (c > 0)
        if ok.any():
            samples[k] = np.log(m[ok]), c[ok]
    if isinstance(reference_amplitude, str):
        if reference_amplitude != "auto":
            raise ValueError("reference amplitude must be a positive number or 'auto'")
        keep = sorted(samples)
        xs = [np.log(a[k] / L) for k in keep]
        means = [float((c * ln_t).sum() / c.sum()) for ln_t, c in (samples[k] for k in keep)]
        reference_amplitude = float(np.exp(np.polyfit(xs, means, 1)[1])) if len(keep) >= 2 else 1.0
    reference_amplitude = float(reference_amplitude)
    if reference_amplitude <= 0 or not np.isfinite(reference_amplitude):
        raise ValueError("reference amplitude must be finite and positive")
    ln2 = np.log(2.0)
    n_q, n_sc = q.size, a.size
    tau, h, h2, D = (np.full((n_q, n_sc), np.nan) for _ in range(4))
    s_tau, s_h, s_D = (np.full((n_q, n_sc), np.nan) for _ in range(3))
    N_a = np.zeros(n_sc)
    for k, (ln_t, c) in samples.items():
        ln_t = ln_t - np.log(reference_amplitude)
        x = np.log(a[k] / L)
        n = ln_t.size
        Z, ZL, ZL2, ZS, Z2 = (np.zeros(n_q) for _ in range(5))
        top = None
        for start in range(0, n, chunk):
            t = ln_t[start:start + chunk]
            lc = np.log(c[start:start + chunk])
            s2 = (t - delta * x) ** 2 / x
            lw = lc[None, :] + q[:, None] * t[None, :] + gamma * s2[None, :]
            if top is None:                      # one stabiliser per q over the whole scale
                top = lw.max(axis=1)
            w = np.exp(lw - top[:, None])
            Z += w.sum(axis=1)
            ZL += (w * t[None, :]).sum(axis=1)
            ZL2 += (w * t[None, :] ** 2).sum(axis=1)
            ZS += (w * s2[None, :]).sum(axis=1)
            Z2 += (w * w).sum(axis=1)
        log_z = top + np.log(Z)                   # ln Σ P
        n_eff = float(n)                          # the valid samples, as the ordinary table counts
        N_a[k] = n_eff
        tau[:, k] = log_z / ln2
        h[:, k] = ZL / Z / ln2
        h2[:, k] = ZS / Z / ln2
        D[:, k] = q * h[:, k] + gamma * h2[:, k] - tau[:, k]
        mean_q = Z / n
        var_pix = np.maximum(Z2 / n - mean_q ** 2, 0.0)
        s_sv = np.sqrt(var_pix / n_eff) / np.maximum(mean_q * ln2, 1e-300)
        s_tau[:, k] = np.sqrt(s_sv ** 2 + (1.0 / np.sqrt(n_eff) / ln2) ** 2)
        n_tilted = np.where(Z2 > 0, Z * Z / np.maximum(Z2, 1e-300), 1.0)
        var_logm = np.maximum(ZL2 / np.maximum(Z, 1e-300) - (ZL / np.maximum(Z, 1e-300)) ** 2, 0.0)
        s_h[:, k] = np.sqrt(var_logm / np.maximum(n_tilted, 1.0)) / ln2
        s_D[:, k] = np.sqrt(q ** 2 * s_h[:, k] ** 2 + s_tau[:, k] ** 2)
    return {"q_list": q, "scales": a, "log2_scales": np.log2(a), "N_a": N_a,
            "tau_qa": tau, "h_qa": h, "h2_qa": h2, "D_qa": D,
            "sigma_tau_qa": s_tau, "sigma_h_qa": s_h, "sigma_D_qa": s_D,
            "gamma": gamma, "delta": delta, "reference_amplitude": reference_amplitude}


def generalized_legendre(tau, q_list, gamma_list, h_grid, delta_list=None):
    """Finite-grid Touchette–Beck (28), with Leonarduzzi et al. 2019's shift: the infimum over
    (q, γ, δ) of q h + γ (h − δ)² − τ(q, γ, δ) (their Theorem 1: over a dense family of shifted
    and dilated g the infimum is the large-deviation spectrum). ``tau`` is ``(n_q, n_gamma)`` with
    no ``delta_list`` (δ = 0, the Gaussian ensemble as before) or ``(n_q, n_gamma, n_delta)``.

    This is an upper estimate on a bounded grid, not proof that a sampled h is
    occupied. ``boundary`` marks minima at q/gamma/delta limits (gamma=0 is allowed).
    Negative values are retained; clipping them would hide unsupported regions.
    """
    q, g, h = (np.asarray(v, dtype=float) for v in (q_list, gamma_list, h_grid))
    dl = np.asarray([0.0] if delta_list is None else delta_list, dtype=float)
    t = np.asarray(tau, dtype=float)
    if t.ndim == 2:
        t = t[:, :, None]
    if t.shape != (q.size, g.size, dl.size) or min(q.size, g.size, dl.size, h.size) == 0:
        raise ValueError("tau shape must match nonempty q/gamma/delta grids")
    if any(v.ndim != 1 or not np.isfinite(v).all() for v in (q, g, h, dl)):
        raise ValueError("q, gamma, delta and h grids must be finite vectors")
    D = np.full(h.size, np.nan)
    qi, gi, di = (np.zeros(h.size, int) for _ in range(3))
    for k, hk in enumerate(h):
        v = q[:, None, None]*hk + g[None, :, None]*(hk - dl[None, None, :])**2 - t
        if np.isfinite(v).any():
            i, j, m = np.unravel_index(np.nanargmin(v), v.shape)
            D[k], qi[k], gi[k], di[k] = v[i, j, m], i, j, m
    boundary = ((q[qi] == q.min()) | (q[qi] == q.max()) | (g[gi] == g.max()) |
                ((g[gi] == g.min()) & (g.min() != 0)))
    if dl.size > 1:
        boundary |= (dl[di] == dl.min()) | (dl[di] == dl.max())
    return {"h": h, "D": D, "q_star": q[qi], "gamma_star": g[gi], "delta_star": dl[di], "boundary": boundary}


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
