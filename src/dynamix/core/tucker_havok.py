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

__all__ = ["delay_embed", "hosvd_tucker", "tucker_havok", "tucker_havok_2d", "tucker_plain"]


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


def _scaled(progress, lo: float, hi: float):
    """``progress`` remapped from [0, 1] onto [lo, hi] -- so a phase reports its OWN 0..1 and
    the caller places it in the whole; ``None`` stays ``None``."""
    if progress is None:
        return None
    return lambda stage, frac: progress(stage, lo + (hi - lo) * float(frac))


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


def hosvd_tucker(X, ranks, *, sweeps: int = 0, block: int = 64, progress=None) -> dict:
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
        if progress is not None:
            progress("tucker bases", 0.7 * (n + 1) / n_modes)

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

    for sweep in range(int(sweeps)):
        # HOOI: re-derive each mode's basis from the others-projected tensor (small).
        for n in range(n_modes):
            Y = _project(X, skip=n)
            Yn = np.moveaxis(Y, n, 0).reshape(Y.shape[n], -1)
            Gn = Yn @ Yn.T
            evals, evecs = np.linalg.eigh(Gn)
            factors[n] = evecs[:, np.argsort(evals)[::-1][: ranks[n]]]
        if progress is not None:
            progress("tucker HOOI", 0.7 + 0.25 * (sweep + 1) / int(sweeps))

    core = _project(X)
    if progress is not None:
        progress("tucker core", 1.0)
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


#: At most this many components are kept per result (largest core energy first).
_MAX_COMPONENTS = 32


def _component_order(energy) -> tuple:
    """Indices of the (flattened) components by DESCENDING share of the core energy, capped at
    :data:`_MAX_COMPONENTS`, and those shares."""
    energy = np.asarray(energy, dtype=np.float64).ravel()
    total = float(energy.sum())
    frac = energy / total if total > 0 else np.zeros_like(energy)
    order = np.argsort(-frac, kind="stable")[:_MAX_COMPONENTS]
    return order, frac[order]


def _components_along_first_mode(core, factors, progress=None) -> tuple:
    """The reconstruction split along mode 0: component k keeps only index k of the core's
    first mode (a rank-1 term in that mode). Returns ``(order, shares, [tensor_k, ...])``."""
    energy = (core ** 2).reshape(core.shape[0], -1).sum(axis=1)
    order, shares = _component_order(energy)
    parts = []
    for k in order:
        H = np.tensordot(factors[0][:, [k]], core[[k]], axes=([1], [0]))
        for n in range(1, len(factors)):
            H = np.moveaxis(np.tensordot(factors[n], H, axes=([1], [n])), 0, n)
        parts.append(H)
        if progress is not None:
            progress("tucker components", len(parts) / len(order))
    return order, shares, parts


def tucker_plain(values, *, ranks=None, sweeps: int = 0, progress=None) -> dict:
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
    dec = hosvd_tucker(a, r, sweeps=sweeps, progress=_scaled(progress, 0.0, 0.8))
    recon = dec["core"]
    for n, U in enumerate(dec["factors"]):
        recon = np.moveaxis(np.tensordot(U, recon, axes=([1], [n])), 0, n)
    recon = np.where(finite, recon, np.nan)
    vals = np.where(finite, np.asarray(values, dtype=np.float64), np.nan)
    _order, shares, parts = _components_along_first_mode(
        dec["core"], dec["factors"], progress=_scaled(progress, 0.8, 1.0))
    components = np.stack([np.where(finite, c, np.nan) for c in parts])
    return {"recon": recon, "residual": vals - recon, "core": dec["core"],
            "factors": dec["factors"], "energy": dec["energy"],
            "shape_embedded": tuple(a.shape), "components": components,
            "component_energy": shares}


def tucker_havok(values, *, n_delays: int = 32, ranks=None, axis: str = "rows",
                 sweeps: int = 0, progress=None) -> dict:
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
    dec = hosvd_tucker(X, r, sweeps=sweeps, progress=_scaled(progress, 0.0, 0.7))
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
    if progress is not None:
        progress("tucker reconstruction", 0.75)
    _order, shares, parts = _components_along_first_mode(
        core, factors, progress=_scaled(progress, 0.75, 1.0))
    components = []
    for Hk in parts:
        ck = _hankel_average(Hk)
        ck = np.swapaxes(ck, 0, 1) if axis == "cols" else ck
        components.append(np.where(finite, ck, np.nan))
    return {"recon": recon, "residual": vals - recon, "core": core,
            "factors": factors, "energy": dec["energy"], "shape_embedded": tuple(X.shape),
            "components": np.stack(components), "component_energy": shares}


def _top_basis(G, r: int):
    """Leading-``r`` eigenbasis of the Gram ``G`` and the energy fraction it keeps."""
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    total = float(evals.sum())
    return evecs[:, order[:r]], (float(evals[:r].sum()) / total if total > 0 else 1.0)


def tucker_havok_2d(values, *, n_delays: int = 8, ranks=None, sweeps: int = 0,
                    progress=None) -> dict:
    """Symmetric 2-D delay embedding -> truncated HOSVD (+ HOOI sweeps) -> reconstruction on the
    image grid -- 2-D singular-spectrum analysis (Broomhead & King's delay embedding + SVD, in
    its multichannel/2-D form) done as a Tucker decomposition.

    BOTH axes are delayed: every ``L x L`` patch (``L = n_delays``) is one sample, giving the
    tensor ``X[dy, dx, y', x'(, band)] = values[y' + dy, x' + dx(, band)]`` -- no image axis
    plays "time" (on a 2-D field the 1-D tape's rows/cols choice biases the result).

    ``ranks``: ``(r_delay, r_rows, r_cols[, r_band])``; ``r_delay`` truncates BOTH delay modes
    (symmetry), 0/None = the mode's full size. The truncated reconstruction is ``X`` projected
    onto each mode's kept subspace, computed without ever building ``X`` (``L*L`` times the
    image): every quantity is a sum over shifted image WINDOWS --

    - HOSVD bases: delay Grams from contiguous row/column slabs; rows/cols/band Grams (only for
      modes that truncate) from ``L*L`` window GEMMs;
    - HOOI ``sweeps``: each truncated mode in turn (delay_y, delay_x, rows, cols, band -- the
      order :func:`hosvd_tucker` uses) re-derived from the data projected onto the OTHER modes'
      current bases (a mode left at full size projects as the identity);
    - ``C[a, b] = sum_{dy,dx} Uy[dy, a] Ux[dx, b] * window(dy, dx)`` then the rows/cols/band
      projectors; back through the delay bases, each pixel averaged over every patch covering it
      (the SSA convention, as :func:`_hankel_average` does in 1-D).

    Components: one per kept (delay_y, delay_x) pattern pair ``(a, b)``; ``combined_*`` pairs
    each with its orientation twin ``(b, a)`` (``r(r+1)/2`` groups) -- both top-32 by core energy.
    Returns ``{"recon", "residual", "core" (C), "factors" ([U_delay_y, U_delay_x, U_rows,
    U_cols(, U_band)], an untruncated mode is ``None``), "energy" (the HOSVD pass), "shape_embedded",
    "components", "component_energy", "component_pairs", "combined_components",
    "combined_energy", "combined_pairs"}``.
    """
    a = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.all():
        fill = a[finite].mean() if finite.any() else 0.0
        a = np.where(finite, a, fill)
    band = a.ndim == 3
    A = a if band else a[..., None]
    ny, nx, nc = A.shape
    L = int(n_delays)
    if ny <= L or nx <= L:
        raise ValueError(f"n_delays={L} needs more than {min(ny, nx)} pixels along each axis")
    my, mx = ny - L + 1, nx - L + 1
    r = list(ranks) if ranks is not None else []
    full = [L, my, mx, nc]
    r = [int(min(r[i], full[i])) if i < len(r) and r[i] else full[i] for i in range(4)]
    trunc_rows, trunc_cols, trunc_band = r[1] < my, r[2] < mx, r[3] < nc

    # Contiguous slabs: window(dy, dx) is a row slice of the column slab at dx, and a column
    # slice of the row slab at dy.
    col_slabs = [np.ascontiguousarray(A[:, d:d + mx]) for d in range(L)]
    row_slabs = [np.ascontiguousarray(A[d:d + my].transpose(1, 0, 2)) for d in range(L)]

    def window(dy, dx):
        return col_slabs[dx][dy:dy + my]                       # (my, mx, nc), contiguous

    def reduce_space(T, first):
        """Project axes ``first`` (rows), ``first+1`` (cols), ``first+2`` (band) onto their
        current bases where those modes truncate -- the 'others-projected' tensor for a Gram."""
        if trunc_rows:
            T = np.moveaxis(np.tensordot(U_rows.T, T, axes=([1], [first])), 0, first)
        if trunc_cols:
            T = np.moveaxis(np.tensordot(U_cols.T, T, axes=([1], [first + 1])), 0, first + 1)
        if trunc_band:
            T = np.moveaxis(np.tensordot(U_band.T, T, axes=([1], [first + 2])), 0, first + 2)
        return T

    def delay_core(Uy_, Ux_, report=None):
        C_ = np.zeros((Uy_.shape[1], Ux_.shape[1], my, mx, nc))
        for dy in range(L):
            for dx in range(L):
                C_ += np.einsum("a,b,ijk->abijk", Uy_[dy], Ux_[dx], window(dy, dx))
            if report is not None:
                report("tucker projection", (dy + 1) / L)
        return C_

    # -- HOSVD ---------------------------------------------------------------------------------
    Gy = np.zeros((L, L))
    Gx = np.zeros((L, L))
    for i in range(L):
        for j in range(i, L):
            gy = sum(np.vdot(S[i:i + my], S[j:j + my]) for S in col_slabs)
            gx = sum(np.vdot(S[i:i + mx], S[j:j + mx]) for S in row_slabs)
            Gy[i, j] = Gy[j, i] = gy
            Gx[i, j] = Gx[j, i] = gx
        if progress is not None:
            progress("tucker bases", 0.25 * (i + 1) / L)
    Uy, ey = _top_basis(Gy, r[0])
    Ux, ex = _top_basis(Gx, r[0])
    U_rows = U_cols = U_band = None
    e_rows = e_cols = e_band = 1.0
    if trunc_rows:
        G = np.zeros((my, my))
        for dy in range(L):
            for dx in range(L):
                W = window(dy, dx).reshape(my, -1)
                G += W @ W.T
        U_rows, e_rows = _top_basis(G, r[1])
    if trunc_cols:
        G = np.zeros((mx, mx))
        for dy in range(L):
            for dx in range(L):
                W = window(dy, dx).transpose(1, 0, 2).reshape(mx, -1)
                G += W @ W.T
        U_cols, e_cols = _top_basis(G, r[2])
    if trunc_band:
        G = np.zeros((nc, nc))
        for dy in range(L):
            for dx in range(L):
                W = window(dy, dx).reshape(-1, nc)
                G += W.T @ W
        U_band, e_band = _top_basis(G, r[3])

    # -- HOOI sweeps -----------------------------------------------------------------------------
    for sweep in range(int(sweeps)):
        # delay_y: the data projected onto delay_x (+ truncated space/band), Gram over dy.
        Z = np.zeros((L, Ux.shape[1], my, mx, nc))
        for dy in range(L):
            for dx in range(L):
                Z[dy] += np.einsum("b,ijk->bijk", Ux[dx], window(dy, dx))
        Zr = reduce_space(Z, 2).reshape(L, -1)
        Uy, _ = _top_basis(Zr @ Zr.T, r[0])
        # delay_x: the data projected onto the NEW delay_y (+ space/band), Gram over dx.
        Z = np.zeros((L, Uy.shape[1], my, mx, nc))
        for dy in range(L):
            for dx in range(L):
                Z[dx] += np.einsum("a,ijk->aijk", Uy[dy], window(dy, dx))
        Zr = reduce_space(Z, 2).reshape(L, -1)
        Ux, _ = _top_basis(Zr @ Zr.T, r[0])
        if trunc_rows or trunc_cols or trunc_band:
            C0 = delay_core(Uy, Ux)                           # (ra, rb, my, mx, nc)
            if trunc_rows:
                T = C0
                if trunc_cols:
                    T = np.moveaxis(np.tensordot(U_cols.T, T, axes=([1], [3])), 0, 3)
                if trunc_band:
                    T = np.moveaxis(np.tensordot(U_band.T, T, axes=([1], [4])), 0, 4)
                Tm = np.moveaxis(T, 2, 0).reshape(my, -1)
                U_rows, _ = _top_basis(Tm @ Tm.T, r[1])
            if trunc_cols:
                T = C0
                if trunc_rows:
                    T = np.moveaxis(np.tensordot(U_rows.T, T, axes=([1], [2])), 0, 2)
                if trunc_band:
                    T = np.moveaxis(np.tensordot(U_band.T, T, axes=([1], [4])), 0, 4)
                Tm = np.moveaxis(T, 3, 0).reshape(mx, -1)
                U_cols, _ = _top_basis(Tm @ Tm.T, r[2])
            if trunc_band:
                T = C0
                if trunc_rows:
                    T = np.moveaxis(np.tensordot(U_rows.T, T, axes=([1], [2])), 0, 2)
                if trunc_cols:
                    T = np.moveaxis(np.tensordot(U_cols.T, T, axes=([1], [3])), 0, 3)
                Tm = np.moveaxis(T, 4, 0).reshape(nc, -1)
                U_band, _ = _top_basis(Tm @ Tm.T, r[3])
        if progress is not None:
            progress("tucker HOOI", 0.3 + 0.2 * (sweep + 1) / int(sweeps))

    # -- projection + reconstruction -----------------------------------------------------------
    if progress is not None:
        progress("tucker projection", 0.5)
    C = delay_core(Uy, Ux, report=_scaled(progress, 0.5, 0.65))
    if U_rows is not None:
        C = np.einsum("ip,pq,abqjk->abijk", U_rows, U_rows.T, C)
    if U_cols is not None:
        C = np.einsum("jp,pq,abiqk->abijk", U_cols, U_cols.T, C)
    if U_band is not None:
        C = np.einsum("kp,pq,abijq->abijk", U_band, U_band.T, C)

    # Components: separate (a, b) and combined {(a, b), (b, a)}, top-energy first; every pair
    # either list needs is accumulated ONCE.
    E = (C ** 2).sum(axis=(2, 3, 4))                              # (ra, rb)
    ra = C.shape[0]
    order, shares = _component_order(E)
    sep_pairs = [divmod(int(i), C.shape[1]) for i in order]
    groups = [(i, j) for i in range(ra) for j in range(i, ra)]
    g_energy = np.array([E[i, j] + (E[j, i] if i != j else 0.0) for i, j in groups])
    g_order, g_shares = _component_order(g_energy)
    comb_groups = [groups[int(g)] for g in g_order]
    needed = list(dict.fromkeys(
        sep_pairs + [m for i, j in comb_groups for m in ((i, j), (j, i)) if i != j or m == (i, j)]))
    index = {pr: n for n, pr in enumerate(needed)}
    a_idx = np.array([pr[0] for pr in needed])
    b_idx = np.array([pr[1] for pr in needed])
    C_sel = C[a_idx, b_idx]                                     # (P, my, mx, nc)
    parts = np.zeros((len(needed), ny, nx, nc))
    out = np.zeros((ny, nx, nc))
    counts = np.zeros((ny, nx, 1))
    for dy in range(L):
        for dx in range(L):
            out[dy:dy + my, dx:dx + mx] += np.einsum("a,b,abijk->ijk", Uy[dy], Ux[dx], C)
            w = Uy[dy, a_idx] * Ux[dx, b_idx]
            parts[:, dy:dy + my, dx:dx + mx] += w[:, None, None, None] * C_sel
            counts[dy:dy + my, dx:dx + mx] += 1.0
        if progress is not None:
            progress("tucker reconstruction", 0.65 + 0.35 * (dy + 1) / L)
    recon = out / counts
    parts /= counts[None]
    comps = parts[[index[pr] for pr in sep_pairs]]
    comb = np.stack([parts[index[(i, j)]] + (parts[index[(j, i)]] if i != j else 0.0)
                     for i, j in comb_groups])
    if not band:
        recon, comps, comb, C = recon[..., 0], comps[..., 0], comb[..., 0], C[..., 0]
    recon = np.where(finite, recon, np.nan)
    comps = np.where(finite[None], comps, np.nan)
    comb = np.where(finite[None], comb, np.nan)
    vals = np.where(finite, np.asarray(values, dtype=np.float64), np.nan)
    factors = [Uy, Ux, U_rows, U_cols] + ([U_band] if band else [])
    energy = [ey, ex, e_rows, e_cols] + ([e_band] if band else [])
    shape = (L, L, my, mx) + ((nc,) if band else ())
    return {"recon": recon, "residual": vals - recon, "core": C, "factors": factors,
            "energy": energy, "shape_embedded": shape, "components": comps,
            "component_energy": shares, "component_pairs": np.array(sep_pairs),
            "combined_components": comb, "combined_energy": g_shares,
            "combined_pairs": np.array(comb_groups)}
