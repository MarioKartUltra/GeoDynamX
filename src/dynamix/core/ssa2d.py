# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""2D singular-spectrum analysis (Golyandina & Usevich 2010, "2D-extension of Singular Spectrum
Analysis: algorithm and elements of theory").

Every ``L_r x L_c`` window of the image is one column of the Hankel-block-Hankel matrix ``W``
(``L_r L_c x K_r K_c``, ``K = N - L + 1``); one SVD ``W = sum sqrt(l_i) U_i V_i^T`` gives the
eigentriples. Reshaped, ``U_i`` is an EIGENARRAY (an ``L_r x L_c`` pattern of any shape -- not
restricted to separable products, which is what makes 2D-SSA invariant to rotations of the
image axes) and ``V_i`` a FACTOR ARRAY (``K_r x K_c``). Each rank-1 term, projected back onto
Hankel-block-Hankel form (every pixel averaged over the windows covering it), is an elementary
reconstructed component; together they sum to the image.

Computed without forming ``W``: two blockwise passes over the windows --
``W W^T`` (the lag-covariance, ``(L_r L_c)^2``) accumulated from contiguous blocks of window
rows, its eigenbasis, then ``W^T U`` (the unnormalized factor arrays) in the same blocks; each
component is an overlap-add of its eigenarray over its factor array, divided by the window
counts. Pure BLAS, float64, no FFT. Cost ~ ``K_r K_c (L_r L_c)^2``: fine for windows up to a few
hundred entries; larger ones want the Lanczos + FFT-matvec route (``W v`` and ``W^T u`` are 2-D
correlations) -- not built.

Also returned: the eigenvalue shares ``l_i / sum l`` and the w-correlation matrix (Def 3.18) of
the kept components, the tool for grouping them (separable components show up as blocks).
"""
from __future__ import annotations

import numpy as np

__all__ = ["ssa2d", "window_counts"]


def window_counts(shape, rows_window: int, cols_window: int) -> np.ndarray:
    """How many ``L_r x L_c`` windows cover each pixel -- the ``w_x(i) w_y(j)`` weights of the
    paper's Def 3.15 (and the divisor of the Hankel-block-Hankel projection)."""
    ny, nx = shape
    kr, kc = ny - rows_window + 1, nx - cols_window + 1
    i = np.arange(ny)
    j = np.arange(nx)
    wr = np.minimum.reduce([i + 1, np.full(ny, rows_window), np.full(ny, kr), ny - i])
    wc = np.minimum.reduce([j + 1, np.full(nx, cols_window), np.full(nx, kc), nx - j])
    return np.outer(wr, wc).astype(np.float64)


def ssa2d(values, *, rows_window: int, cols_window: int, n_components: int = 16,
          progress=None, block: int = 32) -> dict:
    """2D-SSA of a 2-D field. Returns ``{"recon" (sum of the kept components), "residual",
    "components" (n, ny, nx), "eigen_share" (n,), "eigenarrays" (n, L_r, L_c),
    "factor_arrays" (n, K_r, K_c), "w_correlation" (n, n)}``; NaN pixels are filled with the
    finite mean for the decomposition and re-masked in every output."""
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"2D-SSA takes a 2-D field; got shape {a.shape}")
    finite = np.isfinite(a)
    if not finite.all():
        fill = a[finite].mean() if finite.any() else 0.0
        a = np.where(finite, a, fill)
    ny, nx = a.shape
    Lr, Lc = int(rows_window), int(cols_window)
    if not (1 <= Lr <= ny and 1 <= Lc <= nx) or not (1 < Lr * Lc < ny * nx):
        raise ValueError(f"window {Lr} x {Lc} does not fit a {ny} x {nx} field")
    Kr, Kc = ny - Lr + 1, nx - Lc + 1
    P = Lr * Lc
    view = np.lib.stride_tricks.sliding_window_view(a, (Lr, Lc))      # (Kr, Kc, Lr, Lc)

    def report(stage, frac):
        if progress is not None:
            progress(stage, frac)

    # Pass 1: the lag-covariance W W^T, from contiguous blocks of window rows.
    G = np.zeros((P, P))
    for k0 in range(0, Kr, block):
        B = np.ascontiguousarray(view[k0:k0 + block]).reshape(-1, P)
        G += B.T @ B
        report("2D-SSA lag covariance", 0.45 * min(k0 + block, Kr) / Kr)
    lam, U = np.linalg.eigh(G)
    order = np.argsort(lam)[::-1]
    lam = np.maximum(lam[order], 0.0)
    U = U[:, order]
    total = float(lam.sum())
    n = int(min(n_components, P))
    Un = U[:, :n]
    report("2D-SSA eigenarrays", 0.5)

    # Pass 2: W^T U -- the factor arrays times sqrt(lambda).
    A = np.empty((Kr * Kc, n))
    for k0 in range(0, Kr, block):
        B = np.ascontiguousarray(view[k0:k0 + block]).reshape(-1, P)
        A[k0 * Kc:k0 * Kc + B.shape[0]] = B @ Un
        report("2D-SSA factor arrays", 0.5 + 0.2 * min(k0 + block, Kr) / Kr)
    A = A.T.reshape(n, Kr, Kc)
    psi = Un.T.reshape(n, Lr, Lc)

    # Elementary reconstructed components: overlap-add of each eigenarray over its factor
    # array, divided by the window counts (the Hankel-block-Hankel projection).
    comps = np.zeros((n, ny, nx))
    for i in range(Lr):
        for j in range(Lc):
            comps[:, i:i + Kr, j:j + Kc] += psi[:, i, j][:, None, None] * A
        report("2D-SSA components", 0.7 + 0.3 * (i + 1) / Lr)
    w = window_counts(a.shape, Lr, Lc)
    comps /= w[None]

    # w-correlations (Def 3.18) between the kept components.
    gram = np.einsum("iyx,jyx,yx->ij", comps, comps, w)
    norms = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        wcorr = np.abs(gram) / np.outer(norms, norms)
    wcorr = np.nan_to_num(wcorr)
    np.fill_diagonal(wcorr, 1.0)

    safe = np.sqrt(np.where(lam[:n] > 0, lam[:n], np.inf))
    factor_arrays = A / safe[:, None, None]
    recon = comps.sum(axis=0)
    vals = np.where(finite, np.asarray(values, dtype=np.float64), np.nan)
    recon = np.where(finite, recon, np.nan)
    comps = np.where(finite[None], comps, np.nan)
    share = lam[:n] / total if total > 0 else np.zeros(n)
    return {"recon": recon, "residual": vals - recon, "components": comps,
            "eigen_share": share, "eigenarrays": psi, "factor_arrays": factor_arrays,
            "w_correlation": wcorr}
