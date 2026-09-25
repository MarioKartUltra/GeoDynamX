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

Two routes, neither forming ``W``:

* ``solver="dense"`` -- two blockwise passes over the windows: ``W W^T`` (the lag-covariance,
  ``(L_r L_c)^2``) accumulated from contiguous blocks of window rows, its full eigenbasis, then
  ``W^T U`` (the unnormalized factor arrays) in the same blocks; each component is an
  overlap-add of its eigenarray over its factor array, divided by the window counts. Pure BLAS,
  float64. Cost ~ ``K_r K_c (L_r L_c)^2``: fine for windows up to a few hundred entries.
* ``"lanczos"`` / ``"propack"`` / ``"randomized"`` -- the fast route (Korobeynikov 2010 for
  Hankel matrices; Golyandina, Korobeynikov, Shlemov & Usevich 2015 for Hankel-block-Hankel):
  ``W^T u`` and ``W v`` are 2-D correlations of the image, done by FFT at the image's own size
  (their valid parts never wrap), and only the ``k`` leading eigentriples are found -- ARPACK's
  Lanczos on the implicit ``W W^T``, PROPACK's bidiagonalization of ``W``, or a randomized range
  finder whose power iterations are re-orthonormalized after every product (Halko, Martinsson &
  Tropp 2011; the randomized step of Lopes et al. 2024's "pragmatic SSA"). Components are FFT
  convolutions of eigenarray and factor array. The FFTs run at the app's precision
  (``fft_policy``), so these agree with ``dense`` to float32 tolerance.

The eigenvalue shares are ``l_i / trace(W W^T)``, exact on every route: the trace is the sum of
the squared pixels weighted by their window counts. Each eigenarray is signed so that its
largest-magnitude entry is positive (the components do not depend on the sign). ``keep="energy"``
keeps the fewest leading eigentriples whose cumulative share reaches ``energy_threshold``
(Lopes et al. 2024, section 2.3.3).

Also returned: the w-correlation matrix (Def 3.18) of the kept components, the tool for grouping
them (separable components show up as blocks).
"""
from __future__ import annotations

import numpy as np

__all__ = ["cluster_components", "ssa2d", "window_counts"]

#: The routes (see the module docstring).
SOLVERS = ("dense", "lanczos", "propack", "randomized")
#: Extra random directions of the randomized range finder (Halko et al. 2011's p).
_OVERSAMPLE = 10
#: FFT products are done this many arrays at a time, to bound memory on large fields.
_CHUNK = 8


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


def cluster_components(w_correlation, shares, distance: float = 0.5):
    """Group components by average-linkage hierarchical clustering on the distance
    ``1 - |w-correlation|``, cut at ``distance`` (Lopes et al. 2024 section 2.3.5, with 2D-SSA's
    own separability measure, the w-correlation of Def 3.18, for their plain correlation).
    Returns ``(clusters, linkage)``: clusters as lists of 0-based component indices, the largest
    total share first; ``linkage`` the scipy matrix (empty for a single component)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    R = np.abs(np.asarray(w_correlation, dtype=np.float64))
    m = R.shape[0]
    if m < 2:
        return [list(range(m))], np.zeros((0, 4))
    D = np.clip(1.0 - 0.5 * (R + R.T), 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    groups: dict = {}
    for i, label in enumerate(fcluster(Z, t=float(distance), criterion="distance")):
        groups.setdefault(int(label), []).append(i)
    share = np.asarray(shares, dtype=np.float64)
    return sorted(groups.values(), key=lambda g: (-share[g].sum(), g[0])), Z


def _fft_operators(a, Lr: int, Lc: int):
    """``(Wt, W)`` for the image ``a``: ``Wt`` maps a stack of ``L_r x L_c`` arrays to their dot
    products with every window (``K_r x K_c`` each), ``W`` maps a stack of ``K_r x K_c`` arrays to
    their window-weighted sums (``L_r x L_c`` each). Both are the correlation of the image with
    the zero-padded array, by FFT at the image's size; the parts kept never wrap."""
    from dynamix.core.fft_policy import active as _fft

    ny, nx = a.shape
    Kr, Kc = ny - Lr + 1, nx - Lc + 1
    F = _fft().fft2(a)

    def corr(stack, rows, cols):
        stack = np.asarray(stack, dtype=np.float64)
        out = np.empty((stack.shape[0], rows, cols))
        for c0 in range(0, stack.shape[0], _CHUNK):
            part = stack[c0:c0 + _CHUNK]
            pad = np.zeros((part.shape[0], ny, nx))
            pad[:, :part.shape[1], :part.shape[2]] = part
            full = np.real(_fft().ifft2(F[None] * np.conj(_fft().fft2(pad))))
            out[c0:c0 + part.shape[0]] = full[:, :rows, :cols]
        return out

    return (lambda u: corr(u, Kr, Kc)), (lambda v: corr(v, Lr, Lc))


def _fft_components(psi, A, shape):
    """Each eigenarray convolved with its factor array (the overlap-add of the dense route), by
    FFT: the full convolution is exactly the image's size, so nothing wraps."""
    from dynamix.core.fft_policy import active as _fft

    ny, nx = shape
    n, Lr, Lc = psi.shape
    _, Kr, Kc = A.shape
    out = np.empty((n, ny, nx))
    for c0 in range(0, n, _CHUNK):
        m = min(_CHUNK, n - c0)
        p = np.zeros((m, ny, nx))
        q = np.zeros((m, ny, nx))
        p[:, :Lr, :Lc] = psi[c0:c0 + m]
        q[:, :Kr, :Kc] = A[c0:c0 + m]
        out[c0:c0 + m] = np.real(_fft().ifft2(_fft().fft2(p) * _fft().fft2(q)))
    return out


def _lanczos(Wt, W, Lr, Lc, n, tol):
    """The ``n`` largest eigenpairs of the implicit ``W W^T`` by ARPACK (fixed start vector)."""
    from scipy.sparse.linalg import LinearOperator, eigsh

    P = Lr * Lc
    C = LinearOperator((P, P), dtype=np.float64,
                       matvec=lambda u: W(Wt(np.reshape(u, (1, Lr, Lc))))[0].ravel())
    lam, U = eigsh(C, k=n, which="LA", tol=tol,
                   v0=np.random.default_rng(0).standard_normal(P))
    order = np.argsort(lam)[::-1]
    return np.maximum(lam[order], 0.0), U[:, order]


def _propack(Wt, W, Lr, Lc, Kr, Kc, n):
    """The ``n`` leading singular triplets of ``W`` by PROPACK's Lanczos bidiagonalization, with
    FULL reorthogonalization (``delta=0``). Its default partial reorthogonalization estimates
    the loss of orthogonality for an exact operator; the FFT products at 32 bit are adjoint
    only to ~1e-7, so it misses the loss and returns ghost copies of the leading triplet. scipy's
    ``svds`` does not pass that option, hence the direct (private) PROPACK wrapper."""
    import scipy.sparse.linalg as sla

    try:
        import scipy.sparse.linalg._svdp as svdp
    except ImportError:
        raise ValueError("2D-SSA: this scipy has no PROPACK wrapper -- use the Lanczos "
                         "solver") from None
    op = sla.LinearOperator(
        (Lr * Lc, Kr * Kc), dtype=np.float64,
        matvec=lambda v: W(np.reshape(v, (1, Kr, Kc)))[0].ravel(),
        rmatvec=lambda u: Wt(np.reshape(u, (1, Lr, Lc)))[0].ravel())
    import inspect

    # a seeded Generator, under whichever keyword this scipy names it (rng / random_state)
    seed_kw = "rng" if "rng" in inspect.signature(svdp._svdp).parameters else "random_state"
    try:
        U, s, _vt = svdp._svdp(op, n, delta=0.0, **{seed_kw: np.random.default_rng(0)})[:3]
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"2D-SSA: PROPACK did not converge ({exc}) -- a nearly flat spectrum "
                         "(very noisy data); use the Lanczos solver") from None
    order = np.argsort(s)[::-1]
    return s[order] ** 2, U[:, order]


def _randomized(Wt, W, Lr, Lc, Kr, Kc, n, power_iterations):
    """A randomized range finder (fixed seed, oversampling ``_OVERSAMPLE``) with
    ``power_iterations`` passes of ``W W^T``, re-orthonormalized after every product -- without
    that, each pass multiplies the directions by ``s_i^2`` and the smaller ones are lost to
    round-off -- then the SVD of the small projected matrix ``Q^T W``."""
    P, K = Lr * Lc, Kr * Kc
    ell = min(n + _OVERSAMPLE, P, K)
    rng = np.random.default_rng(0)
    Q = np.linalg.qr(W(rng.standard_normal((ell, Kr, Kc))).reshape(ell, P).T)[0]
    for _ in range(int(power_iterations)):
        Z = np.linalg.qr(Wt(Q.T.reshape(ell, Lr, Lc)).reshape(ell, K).T)[0]
        Q = np.linalg.qr(W(Z.T.reshape(ell, Kr, Kc)).reshape(ell, P).T)[0]
    B = Wt(Q.T.reshape(ell, Lr, Lc)).reshape(ell, K)                  # Q^T W, row by row
    Ub, s, _vt = np.linalg.svd(B, full_matrices=False)
    return s[:n] ** 2, (Q @ Ub)[:, :n]


def ssa2d(values, *, rows_window: int, cols_window: int, n_components: int = 16,
          solver: str = "dense", power_iterations: int = 2, keep: str = "count",
          energy_threshold: float = 0.9, progress=None, block: int = 32) -> dict:
    """2D-SSA of a 2-D field. Returns ``{"recon" (sum of the kept components), "residual",
    "components" (n, ny, nx), "eigen_share" (n,), "eigenarrays" (n, L_r, L_c),
    "factor_arrays" (n, K_r, K_c), "w_correlation" (n, n), "solver_used", "energy_reached"}``;
    NaN pixels are filled with the finite mean for the decomposition and re-masked in every
    output.

    ``solver`` is one of :data:`SOLVERS`; an FFT solver hands over to ``dense`` when the kept
    count reaches the window size (a tiny window, where dense is exact and instant) and says so
    in ``solver_used``. ``keep="energy"`` treats ``n_components`` as the most to find and keeps
    the fewest whose cumulative share reaches ``energy_threshold``; ``energy_reached`` is False
    when ``n_components`` was not enough."""
    if solver not in SOLVERS:
        raise ValueError(f"unknown 2D-SSA solver {solver!r}; expected one of {SOLVERS}")
    if keep not in ("count", "energy"):
        raise ValueError(f"keep must be 'count' or 'energy', got {keep!r}")
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
    n = int(min(n_components, P))
    if solver != "dense" and n >= min(P, Kr * Kc) - 1:
        solver = "dense"
    w = window_counts(a.shape, Lr, Lc)

    def report(stage, frac):
        if progress is not None:
            progress(stage, frac)

    def kept(lam, total):
        if keep != "energy" or total <= 0:
            return n, True
        hit = np.flatnonzero(np.cumsum(lam[:n]) / total >= float(energy_threshold) - 1e-12)
        return (int(hit[0]) + 1, True) if hit.size else (n, False)

    if solver == "dense":
        view = np.lib.stride_tricks.sliding_window_view(a, (Lr, Lc))  # (Kr, Kc, Lr, Lc)

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
        m, reached = kept(lam, total)
        Un = U[:, :m]
        report("2D-SSA eigenarrays", 0.5)

        # Pass 2: W^T U -- the factor arrays times sqrt(lambda).
        A = np.empty((Kr * Kc, m))
        for k0 in range(0, Kr, block):
            B = np.ascontiguousarray(view[k0:k0 + block]).reshape(-1, P)
            A[k0 * Kc:k0 * Kc + B.shape[0]] = B @ Un
            report("2D-SSA factor arrays", 0.5 + 0.2 * min(k0 + block, Kr) / Kr)
        A = A.T.reshape(m, Kr, Kc)
        psi = Un.T.reshape(m, Lr, Lc)

        # Elementary reconstructed components: overlap-add of each eigenarray over its factor
        # array, divided by the window counts (the Hankel-block-Hankel projection).
        comps = np.zeros((m, ny, nx))
        for i in range(Lr):
            for j in range(Lc):
                comps[:, i:i + Kr, j:j + Kc] += psi[:, i, j][:, None, None] * A
            report("2D-SSA components", 0.7 + 0.3 * (i + 1) / Lr)
    else:
        from dynamix.core.fft_policy import active as _fft

        Wt, W = _fft_operators(a, Lr, Lc)
        total = float((w * a * a).sum())                     # trace(W W^T), exactly
        report("2D-SSA eigentriples", 0.05)
        if solver == "lanczos":
            tol = 1e-6 if getattr(_fft(), "precision", 64) == 32 else 1e-10
            lam, U = _lanczos(Wt, W, Lr, Lc, n, tol)
        elif solver == "propack":
            lam, U = _propack(Wt, W, Lr, Lc, Kr, Kc, n)
        else:
            lam, U = _randomized(Wt, W, Lr, Lc, Kr, Kc, n, power_iterations)
        m, reached = kept(lam, total)
        lam, psi = lam[:m], U[:, :m].T.reshape(m, Lr, Lc)
        report("2D-SSA factor arrays", 0.6)
        A = Wt(psi)
        report("2D-SSA components", 0.7)
        comps = _fft_components(psi, A, a.shape)
        report("2D-SSA components", 1.0)
    comps /= w[None]

    # Sign convention: each eigenarray's largest-magnitude entry positive (components unchanged).
    flat = psi.reshape(m, -1)
    sign = np.sign(flat[np.arange(m), np.argmax(np.abs(flat), axis=1)])
    sign[sign == 0] = 1.0
    psi = psi * sign[:, None, None]
    A = A * sign[:, None, None]

    # w-correlations (Def 3.18) between the kept components.
    gram = np.einsum("iyx,jyx,yx->ij", comps, comps, w)
    norms = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        wcorr = np.abs(gram) / np.outer(norms, norms)
    wcorr = np.nan_to_num(wcorr)
    np.fill_diagonal(wcorr, 1.0)

    safe = np.sqrt(np.where(lam[:m] > 0, lam[:m], np.inf))
    factor_arrays = A / safe[:, None, None]
    recon = comps.sum(axis=0)
    vals = np.where(finite, np.asarray(values, dtype=np.float64), np.nan)
    recon = np.where(finite, recon, np.nan)
    comps = np.where(finite[None], comps, np.nan)
    share = lam[:m] / total if total > 0 else np.zeros(m)
    return {"recon": recon, "residual": vals - recon, "components": comps,
            "eigen_share": share, "eigenarrays": psi, "factor_arrays": factor_arrays,
            "w_correlation": wcorr, "solver_used": solver, "energy_reached": bool(reached)}
