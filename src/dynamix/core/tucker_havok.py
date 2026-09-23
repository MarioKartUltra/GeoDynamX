# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Tucker-HAVOK: delay-embedded Tucker (HOSVD) decomposition of a raster -- the ASTER port.

Ported in SPIRIT from the author's ASTER notebook (cells 29-48: a 4-D
time-delay Hankel tensor over a 2-D image + band axis, ``tensorly.tucker(init='svd')``, factor
recombination), rebuilt under the efficiency mandate -- the notebook crashed on a whole image because it MATERIALIZED the Hankel tensor. This implementation never does:

- **The delay embedding is a stride-tricks VIEW** (``sliding_window_view``): zero copy, however
  many delays.
- **HOSVD by mode-n Gram matrices**: each mode's factor basis is the eigenbasis of
  ``X_(n) X_(n)^T`` -- a matrix no bigger than that mode (delay ~ tens, bands ~ few, space ~
  width), accumulated BLOCKWISE along the embedded axis with contiguous BLAS GEMMs (one
  bounded copy per block, never the whole tensor). eigh of tiny matrices; Accelerate/AMX owns
  the GEMMs, which is why numba/mlx are not used on this path -- BLAS is the right tool, by
  measurement class not by omission. The core tensor comes from mode products applied
  largest-mode-first, blockwise for the first contraction.
- **Optional HOOI sweeps** (what ``tensorly.tucker(init='svd')`` iterates): each sweep
  re-derives one mode's basis from the OTHERS-projected tensor -- small once ranks applied.
  HOSVD alone is the standard quasi-optimal single pass and the interactive default.
- **Reconstruction back to a raster** by diagonal (Hankel-inverse) averaging over the delay
  axis -- the SSA convention, vectorized as ``n_delays`` shifted adds.

Axes convention: the chosen image axis plays HAVOK's "time"; the other axis is space; a
multi-component field adds the band mode. 2-D fields give a 3-mode tensor
``(delay, time', space)``; ``(ny, nx, nc)`` fields give 4 modes ``(delay, time', space, band)``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["delay_embed", "hosvd_tucker", "tucker_havok", "tucker_plain"]


def delay_embed(values, n_delays: int, *, axis: str = "rows"):
    """The Hankel VIEW: ``(n_delays, T', space[, band])`` over ``values`` -- zero copy.

    ``axis="rows"`` embeds down the rows (each column is a "tape"); ``"cols"`` embeds across.
    """
    a = np.asarray(values)
    if axis not in ("rows", "cols"):
        raise ValueError(f"axis must be 'rows' or 'cols', got {axis!r}")
    if axis == "cols":
        a = np.swapaxes(a, 0, 1)
    if a.shape[0] <= n_delays:
        raise ValueError(f"n_delays={n_delays} needs more than {a.shape[0]} samples along "
                         f"the embedded axis")
    view = np.lib.stride_tricks.sliding_window_view(a, n_delays, axis=0)
    # sliding_window_view puts the window axis LAST; HAVOK convention wants (delay, time', ...)
    return np.moveaxis(view, -1, 0)


def _mode_gram_blockwise(X, mode: int, block: int = 64) -> np.ndarray:
    """``X_(mode) X_(mode)^T`` accumulated blockwise along axis 1 (the embedded/time axis when
    ``mode != 1``) -- contiguous copies bounded to one block, float64 accumulation."""
    d = X.shape[mode]
    G = np.zeros((d, d))
    Xm = np.moveaxis(X, mode, 0)
    steps = range(0, Xm.shape[1], block) if Xm.ndim > 1 else [0]
    for start in steps:
        chunk = np.ascontiguousarray(
            Xm[:, start:start + block].reshape(d, -1), dtype=np.float64)
        G += chunk @ chunk.T
    return G


def hosvd_tucker(X, ranks, *, sweeps: int = 0, block: int = 64) -> dict:
    """Truncated HOSVD (+ optional HOOI sweeps) of the (possibly strided-view) tensor ``X``.

    ``ranks``: one int per mode (clipped to the mode's size). Returns ``{"core", "factors"
    (list of (d_n, r_n) with orthonormal columns), "energy" (per-mode explained-energy
    fractions of the kept eigenvalues)}``.
    """
    n_modes = X.ndim
    ranks = [int(min(r, X.shape[n])) for n, r in zip(range(n_modes), ranks)]
    factors, energy = [], []
    for n in range(n_modes):
        G = _mode_gram_blockwise(X, n, block=block)
        evals, evecs = np.linalg.eigh(G)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)
        U = evecs[:, order[: ranks[n]]]
        factors.append(U)
        total = float(evals.sum())
        energy.append(float(evals[: ranks[n]].sum()) / total if total > 0 else 1.0)

    def _project(T, skip=None):
        """T ×_n U_n^T over every mode (except ``skip``), LARGEST mode first so the tensor
        shrinks as early as possible."""
        order_ = sorted((n for n in range(n_modes) if n != skip),
                        key=lambda n: -T.shape[n])
        out = T
        for n in order_:
            out = np.tensordot(factors[n].T, out, axes=([1], [n]))
            out = np.moveaxis(out, 0, n)
        return out

    for _ in range(int(sweeps)):
        # HOOI: re-derive each mode's basis from the others-projected tensor (small).
        for n in range(n_modes):
            Y = _project(X, skip=n)
            Yn = np.moveaxis(Y, n, 0).reshape(Y.shape[n], -1)
            Gn = Yn @ Yn.T
            evals, evecs = np.linalg.eigh(Gn)
            factors[n] = evecs[:, np.argsort(evals)[::-1][: ranks[n]]]

    core = _project(X)
    return {"core": core, "factors": factors, "energy": energy}


def _hankel_average(H) -> np.ndarray:
    """Inverse of :func:`delay_embed` for a MATERIALIZED (small, reconstructed) Hankel stack:
    diagonal averaging (the SSA convention), vectorized as ``n_delays`` shifted adds."""
    n_delays, t_prime = H.shape[0], H.shape[1]
    n = t_prime + n_delays - 1
    out = np.zeros((n,) + H.shape[2:])
    counts = np.zeros(n)
    for d in range(n_delays):
        out[d:d + t_prime] += H[d]
        counts[d:d + t_prime] += 1.0
    return out / counts.reshape((-1,) + (1,) * (out.ndim - 1))


def tucker_plain(values, *, ranks=None, sweeps: int = 0) -> dict:
    """Tucker of the RAW dataset -- no delay embedding, no stacked copies. Modes are the array's own: ``(rows, cols[, band])``; for a 2-D field this is
    exactly a truncated SVD, done through the same blockwise-Gram machinery. Same NaN
    fill-and-remask contract as :func:`tucker_havok`; same return keys (``shape_embedded`` is
    the raw shape here)."""
    a = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.all():
        fill = a[finite].mean() if finite.any() else 0.0
        a = np.where(finite, a, fill)
    n_modes = a.ndim
    full = list(a.shape)
    r = list(ranks) if ranks is not None else []
    r = [int(r[i]) if i < len(r) and r[i] else full[i] for i in range(n_modes)]
    dec = hosvd_tucker(a, r, sweeps=sweeps)
    recon = dec["core"]
    for n, U in enumerate(dec["factors"]):
        recon = np.moveaxis(np.tensordot(U, recon, axes=([1], [n])), 0, n)
    recon = np.where(finite, recon, np.nan)
    vals = np.where(finite, np.asarray(values, dtype=np.float64), np.nan)
    return {"recon": recon, "residual": vals - recon, "core": dec["core"],
            "factors": dec["factors"], "energy": dec["energy"],
            "shape_embedded": tuple(a.shape)}


def tucker_havok(values, *, n_delays: int = 32, ranks=None, axis: str = "rows",
                 sweeps: int = 0) -> dict:
    """The full pipeline: delay-embed -> HOSVD Tucker -> rank-truncated reconstruction back on
    the image grid.

    ``ranks``: ``(r_delay, r_time, r_space[, r_band])`` -- missing/None entries default to the
    mode size (no truncation on that mode). Returns ``{"recon" (same shape as values),
    "residual", "core", "factors", "energy", "shape_embedded"}``.
    """
    a = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.all():
        # The embedding mixes neighbors along the tape; NaN would smear a full window. Fill
        # with the finite mean for the DECOMPOSITION and re-mask the outputs (the cwt2d
        # NaN-fill precedent).
        fill = a[finite].mean() if finite.any() else 0.0
        a = np.where(finite, a, fill)
    X = delay_embed(a, int(n_delays), axis=axis)
    n_modes = X.ndim
    full = list(X.shape)
    r = list(ranks) if ranks is not None else []
    r = [int(r[i]) if i < len(r) and r[i] else full[i] for i in range(n_modes)]
    dec = hosvd_tucker(X, r, sweeps=sweeps)
    core, factors = dec["core"], dec["factors"]
    # Reconstruct the (now small-rank) Hankel stack: core x_n U_n back out, then un-embed.
    H = core
    for n in range(n_modes):
        H = np.tensordot(factors[n], H, axes=([1], [n]))
        H = np.moveaxis(H, 0, n)
    tape = _hankel_average(H)
    recon = np.swapaxes(tape, 0, 1) if axis == "cols" else tape
    recon = np.where(finite, recon, np.nan)
    vals = np.where(finite, np.asarray(values, dtype=np.float64), np.nan)
    return {"recon": recon, "residual": vals - recon, "core": core,
            "factors": factors, "energy": dec["energy"], "shape_embedded": tuple(X.shape)}
