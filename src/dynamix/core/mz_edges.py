# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The M-Z peer transform's product bundle: dyadic maxima chains + coarse channel + preview.

"Discrete Approximation of Multiscale Edges" (Mallat-Zhong 1992) as a separate named
algorithm — a peer of the WTMM device, never the app's working representation. This module is the thin app-facing layer
over the verbatim oracle in ``mzlib``: it never imports Qt, never sees layers or chains-UI,
and its output speaks the app's per-scale extrema schema (x/y/mod/arg/line_id) so the shell
renders M-Z chains through the existing display path.

Normalization note (instrumentation doctrine §2): stored W1/W2 (and ``mod``) are the paper's
own dyadic-transform values (lambda-normalized per Table II), NOT the app measurement layer's
L1 |T| ~ a^h convention. Any exponent read off these values must cross the documented seam.

Torus doctrine: maxima are detected and chained on the mirrored
2Nx2N torus; components wrap-merge across the seam; positions shown to the app are the
primary-quadrant members of those components. Extrema index maps are never mirror-projected.

``extrema[k]["line_id"]`` is NOT simply ``_torus_components``'s dense torus-wide label
restricted to the primary quadrant (schema-equivalence correction): a
component with exactly one primary-quadrant member is an isolated extremum, remapped to
``-1`` -- the shell's sentinel for "not a line" (canvas.py, filters.py both special-case -1
as not belonging to any chain). A component with two or more primary-quadrant members keeps
its original torus-wide id (still 0-based, but no longer necessarily gapless once singletons
are pulled out of the visible range). ``mz_maxima`` and ``_torus_components``'s own raw
labeling are untouched by this rule -- it is applied only when building the app-facing
``extrema`` schema.

Scales convention (confirmed against ``mzlib.atrous2d_forward_full``'s own loop, not assumed):
the forward loop is ``for j in range(J): W1,W2 at this j; S = S * H((2**j)*w)``, i.e. loop
index ``j`` is 0-based and the pair appended on iteration ``j`` is the detail at dyadic scale
``2**(j+1)`` — same convention 1-D ``atrous_forward`` documents inline ("produces detail at
scale 2^{j+1}") and the same index ``lam(j+1)`` normalizes by. So ``mz_maxima[k]`` /
``extrema[k]`` (0-based k) live at scale ``2**(k+1)``, and ``bundle["scales"]`` is exactly
``2.0 ** np.arange(1, n_levels + 1)`` — monotone increasing, ``scales[k]`` indexing the same
``lam(k+1)`` the transform itself used. ``atrous2d_forward`` mirrors the (ny,nx) input to a
(2*ny, 2*nx) torus before transforming, so ``S`` and every ``(W1, W2)`` pair returned are
full-torus (2*ny, 2*nx) arrays — confirmed by reading the mirror step (two concatenations)
feeding straight into ``atrous2d_forward_full``, which never crops.

Coarse-channel policy: "full" (mapping mode, SV-B/SVIII) stores nothing and recomputes the
exact (2ny,2nx) torus coarse channel S from ``values`` at preview time. "thumbnail" (coding
mode, SIX) stores only the primary-quadrant coarse channel subsampled at stride 2**n_levels
(script 20's ``coding_coarse``: ``S[:ny:step, :nx:step]``, NOT a subsample of the whole torus,
which would keep three redundant mirrored copies) and restores it at preview time by exactly
script 20's decode move: mirror the thumbnail back onto its own small torus, then trig-
interpolate (Fourier zero-pad with a symmetric Nyquist split) up to the full (2ny,2nx) torus.
That decode is exact at the stored coarse samples and ring-free because the mirrored thumbnail
is smooth and periodic (script 20's own sample-exactness assertion is reproduced here).

Dither/coarse consistency: when ``analyze`` was called with
``dither=True``, its maxima constraints were extracted from the internally-dithered field,
never the caller's raw one. ``preview``'s "full" coarse recompute mirrors that exactly --
``_coarse_for`` re-derives the same dithered field (``_dithered`` is a fixed seed, so it is
bit-identical to what ``analyze`` used) before pinning S, so the coarse channel and the
maxima constraints it is reconstructed against always come from the same field. "thumbnail"
mode needs no such recompute: its stored ``coarse_thumb`` was already taken from the
dithered ``S`` inside ``analyze``, so it is consistent by construction.
"""
from __future__ import annotations

import functools

import numpy as np

# Every FFT TRANSFORM goes through the app-wide engine policy (2026-09-22: mlx or FFTW3
# at the configured 32/64-bit precision, never numpy's FFT by default).
from dynamix.core.fft_policy import active as _fft  # noqa: E402

from dynamix.core import mzlib

_DIVERGENCE_RATIO = 5.0  # resid growth marks inconsistency


def measure_lsb(values):
    u = np.unique(values[np.isfinite(values)])
    if u.size < 2:
        return None
    d = np.diff(u)
    d = d[d > 0]
    return float(d.min()) if d.size else None


def _dithered(values, lsb):
    rng = np.random.default_rng(0)  # fixed seed: analysis-time, deterministic, never written back
    return values + rng.uniform(-lsb / 2.0, lsb / 2.0, values.shape)


def _torus_components_unionfind(rows, cols, shape2):
    """Label wrap-merged 8-connected components of a maxima point set on the torus.

    THE REFERENCE IMPLEMENTATION, superseded 2026-09-19 by :func:`_torus_components`
    below (the O(points) python union-find loop was the mz_edges hot spot: measured 61 s
    of the 106 s total at the 4096-px overview size, 15.8M level-0 maxima). Kept intact
    as the equivalence oracle -- tests/test_mz_edges_core.py checks the fast labeller
    produces the same partitions.

    Direct modulo-indexed neighbor lookup (the earlier ``np.roll``
    version materialized a full ``shape2``-sized grid copy per offset -- ~1.6 GB transient
    at 4096x4096 -- for no reason, since only the ``rows``/``cols`` positions are ever read
    back out of it; indexing ``grid`` at ``(rows+dr) % ny2, (cols+dc) % nx2`` gets the same
    wrapped neighbor values with no grid-sized copies, O(points) instead) + union-find over
    pairs; numpy + O(points) python only. Returns int64 labels (dense, 0-based) aligned with
    ``rows``/``cols``.
    """
    n2y, n2x = shape2
    grid = np.full(shape2, -1, dtype=np.int64)
    grid[rows, cols] = np.arange(rows.size, dtype=np.int64)
    parent = np.arange(rows.size, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    here = grid[rows, cols]  # == np.arange(rows.size); loop-invariant, hoisted out
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        there = grid[(rows + dr) % n2y, (cols + dc) % n2x]
        for a, b in zip(here[there >= 0].tolist(), there[there >= 0].tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    roots = np.array([find(i) for i in range(rows.size)], dtype=np.int64)
    _, dense = np.unique(roots, return_inverse=True)
    return dense.astype(np.int64)


def _torus_components(rows, cols, shape2):
    """Fast wrap-merged 8-connected component labels -- same contract as the reference
    :func:`_torus_components_unionfind` (dense 0-based int64 labels aligned with
    ``rows``/``cols``, identical PARTITIONS; numbering may differ, which nothing
    downstream reads meaning into).

    ``scipy.ndimage.label`` does the grid labelling vectorized; the torus wrap then only
    needs the SEAM adjacencies merged -- at most ``3*(ny+nx)`` pairs instead of a python
    union-find over every point pair (the 2026-09-19 profiling fix: 61 s -> ~2 s at the
    overview size). Roots resolve vectorized by iterated parent-jumping."""
    from scipy import ndimage

    if rows.size == 0:
        return np.zeros(0, dtype=np.int64)
    grid = np.zeros(shape2, dtype=bool)
    grid[rows, cols] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3), dtype=bool))
    parent = np.arange(n + 1, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # 8-connected wrap seams: last row <-> first row and last col <-> first col, each at
    # the three column/row shifts; np.roll wraps the index, so the corner diagonals are
    # covered by the same shifts.
    seams = ((lab[-1, :], lab[0, :]), (lab[:, -1], lab[:, 0]))
    for a_edge, b_edge in seams:
        for shift in (-1, 0, 1):
            b = np.roll(b_edge, -shift)
            m = (a_edge > 0) & (b > 0)
            for a, bb in zip(a_edge[m].tolist(), b[m].tolist()):
                ra, rb = find(a), find(bb)
                if ra != rb:
                    parent[ra] = rb
    while True:
        p2 = parent[parent]
        if np.array_equal(p2, parent):
            break
        parent = p2
    _, dense = np.unique(parent[lab[rows, cols]], return_inverse=True)
    return dense.astype(np.int64)


def _mirror2d(a):
    """Same two-concatenation mirror ``atrous2d_forward`` and script 20's ``mirror2d_np`` use
    to periodize an array onto its own doubled torus."""
    m = np.concatenate([a, a[::-1, :]], axis=0)
    return np.concatenate([m, m[:, ::-1]], axis=1)


def _fourier_upsample_torus(t, factor):
    """Trig-interpolate a periodic (torus) image by an integer factor (script 20's
    ``fourier_upsample_torus``, generalized here to a non-square shape): zero-pad the
    spectrum with a symmetric Nyquist split so the result stays exactly real and reproduces
    the input at the coarse sample positions exactly."""
    ny, nx = t.shape
    ry, rx = ny // 2, nx // 2
    nfy, nfx = ny * factor, nx * factor
    Fs = np.fft.fftshift(_fft().fft2(t))
    T = np.zeros((ny + 1, nx + 1), dtype=complex)
    T[:ny, :nx] = Fs
    T[ny, :nx] = Fs[0, :]
    T[:ny, nx] = Fs[:, 0]
    T[ny, nx] = Fs[0, 0]
    T[0, :] *= 0.5
    T[ny, :] *= 0.5
    T[:, 0] *= 0.5
    T[:, nx] *= 0.5
    B = np.zeros((nfy, nfx), dtype=complex)
    cy, cx = nfy // 2, nfx // 2
    B[cy - ry: cy + ry + 1, cx - rx: cx + rx + 1] = T
    up = np.real(_fft().ifft2(np.fft.ifftshift(B))) * factor * factor
    # sample-exact to the FFT precision (2026-09-22: 32 or 64-bit is a user setting)
    tol = 1e-8 if getattr(_fft(), "precision", 64) == 64 else 1e-5 * max(1.0, np.abs(t).max())
    assert np.max(np.abs(up[::factor, ::factor] - t)) < tol, (
        "trig interpolation lost sample-exactness -- Nyquist split is wrong"
    )
    return up


# --- fractional-order forward (2026-09-19) -----------------------------------------------
#
# The M-Z scheme generalized to a REAL spline order (Unser-Blu, SIAM Review 2000), built
# entirely here so the verbatim-ported ``mzlib`` oracle stays untouched. The refinement
# filter is the SIGN-PRESERVING fractional power ``e^{iw/2} sgn(cos(w/2)) |cos(w/2)|^alpha``
# -- at alpha=3 that is byte-for-byte mzlib's ``Hf`` (cos^3), and the infinite cascade
# converges to ``psi_hat = i w |sinc(w/4)|^(alpha+1)``: the SYMMETRIC fractional-spline
# derivative wavelet, one vanishing moment kept (Gf unchanged -- maxima-are-edges IS the
# method; the fractional knob is SMOOTHNESS, never the moment order). The family is smooth
# in alpha and hits the classical wavelets at the odd-integer anchors (even |sinc| powers);
# even orders give Unser-Blu's symmetric generalization rather than the half-shifted
# classical spline. Oracle chain in tests/test_mz_frac.py: cascade <-> analytic form
# (tight, all orders) <-> the verbatim ``frac_bspline`` Part A2 copies (exact at integer
# anchors). Fractional POCS/preview is a follow-up: ``preview`` refuses fractional bundles.
#
# Lambda: Table II is computed for alpha=3 only, so the fractional path derives lambda_j
# NUMERICALLY from the table's own (rediscovered) definition -- the discrete dyadic
# step-edge modulus peak at level j over the continuum step response theta_alpha(0) --
# a recipe pinned by reproducing the paper's values at alpha=3 (measured unrounded:
# 1.5000, 1.1250, 1.0313, 1.0078, 1.0020 vs the printed 1.50, 1.12, 1.03, 1.01, 1.00; the
# frac path keeps the unrounded values, so alpha=3 differs from the mz_spline path only by
# that rounding, and only when use_lambda is on).

_LAM_FRAC_LEVELS = 12                      # covers the device's n_levels max with headroom
_THETA0_CACHE: dict = {}
_LAM_FRAC_CACHE: dict = {}


def _check_alpha(alpha) -> float:
    alpha = float(alpha)
    if not alpha > 0.0:
        raise ValueError(f"alpha must be > 0 (the spline order; 3 = the paper's wavelet), "
                         f"got {alpha!r}")
    return alpha


def _hf_frac(w, alpha: float):
    """Sign-preserving fractional refinement filter: ``cos^3 -> sgn(cos)|cos|^alpha``.
    The pure-modulus ``|cos|^alpha`` would NOT reduce to mzlib's cos^3 at alpha=3 (the
    cascade evaluates the filter at dilated arguments where cos < 0), and the naive
    ``cos**alpha`` is NaN there -- the sign-preserving power is the one fractionalization
    that is both well-defined and exact at the anchor."""
    c = np.cos(w / 2)
    return np.exp(1j * w / 2) * np.sign(c) * np.abs(c) ** alpha


def _theta0_frac(alpha: float) -> float:
    """``theta_alpha(0) = (1/2pi) int |sinc(w/4)|^(alpha+1) dw`` -- the continuum unit-step
    response peak (alpha=3: 4/3, mzlib's THETA0; alpha=1: 2). Computed by
    :func:`_theta0_integral` (finite part + analytic tail); a bare cutoff at w = 2000, used
    until 2026-09-24, dropped the slow tail: 45% low at alpha = 0.1, 2% at 0.5."""
    alpha = _check_alpha(alpha)
    t0 = _THETA0_CACHE.get(alpha)
    if t0 is None:
        t0 = _THETA0_CACHE[alpha] = _theta0_integral(alpha)
    return t0


def _theta0_integral(alpha: float, periods: int = 4000, per_period: int = 500) -> float:
    """``(1/pi) int_0^inf |sinc(w/4)|^s dw`` (``s = alpha + 1``): quadrature over whole periods
    of ``|sin(w/4)|`` (``4 pi`` each) up to ``W``, then the analytic tail -- beyond ``W`` the
    ``|sin|^s`` factor averages to ``m_s = Gamma((s+1)/2) / (sqrt(pi) Gamma(s/2 + 1))``, so
    ``int_W^inf ~ m_s 4^s W^(1-s) / (s - 1)``. The device Unser & Blu use for their
    B-spline autocorrelation sum (``fractsplineautocorr``: a finite part plus the asymptotic
    tail): the integrand decays only like ``w^-s``, so a cutoff alone loses ``~W^-alpha``."""
    from math import gamma, pi, sqrt

    s = float(alpha) + 1.0
    W = 4.0 * pi * periods
    w = np.linspace(0.0, W, periods * per_period + 1)
    finite = float(np.trapezoid(np.abs(np.sinc(w / 4 / np.pi)) ** s, w))
    mean = gamma((s + 1) / 2) / (sqrt(pi) * gamma(s / 2 + 1))
    return (finite + mean * 4.0 ** s * W ** (1.0 - s) / (s - 1.0)) / pi


def lam_frac(j: int, alpha: float) -> float:
    """``lambda_j`` for spline order alpha -- Table II's definition, computed: the discrete
    1-D dyadic cascade's step-edge modulus peak at level j (levels are 1-based, matching
    ``mzlib.lam``), over ``theta_alpha(0)``."""
    alpha = _check_alpha(alpha)
    table = _LAM_FRAC_CACHE.get(alpha)
    if table is None:
        n = 2 ** 16
        d = np.zeros(n)
        d[n // 2:] = 1.0
        w = 2 * np.pi * np.fft.fftfreq(n)
        Sh = _fft().fft(d)
        t0 = _theta0_frac(alpha)
        peaks = []
        for lvl in range(_LAM_FRAC_LEVELS):
            Wd = np.real(_fft().ifft(Sh * mzlib.Gf((2 ** lvl) * w)))
            peaks.append(float(np.abs(Wd).max()) / t0)
            Sh = Sh * _hf_frac((2 ** lvl) * w, alpha)
        table = _LAM_FRAC_CACHE[alpha] = peaks
    return table[j - 1]


def atrous2d_forward_frac(img, J: int, alpha: float, use_lambda: bool = True):
    """``mzlib.atrous2d_forward`` at spline order alpha: mirror onto the (2Ny,2Nx) torus,
    then the same loop with the sign-preserving fractional filter and the numeric lambda.
    Returns ``(S, [(W1, W2), ...])`` full-torus, exactly mzlib's contract."""
    alpha = _check_alpha(alpha)
    m = _mirror2d(np.asarray(img, dtype=np.float64))
    ny, nx = m.shape
    wy = 2 * np.pi * np.fft.fftfreq(ny)[:, None]
    wx = 2 * np.pi * np.fft.fftfreq(nx)[None, :]
    Sh = mzlib.FFT.fft2(m)
    out = []
    for j in range(J):
        f1 = mzlib.Gf((2 ** j) * wx) * np.ones_like(wy)
        f2 = np.ones_like(wx) * mzlib.Gf((2 ** j) * wy)
        W1 = np.real(mzlib.FFT.ifft2(Sh * f1))
        W2 = np.real(mzlib.FFT.ifft2(Sh * f2))
        if use_lambda:
            W1, W2 = W1 / lam_frac(j + 1, alpha), W2 / lam_frac(j + 1, alpha)
        out.append((W1, W2))
        Sh = Sh * _hf_frac((2 ** j) * wx, alpha) * _hf_frac((2 ** j) * wy, alpha)
    return np.real(mzlib.FFT.ifft2(Sh)), out


@functools.lru_cache(maxsize=64)
def mz_impulse_reach(n_levels: int, wavelet: str = "mz_spline", alpha: float = 3.0) -> int:
    """How far (px) any level's (W1, W2) response to a single-pixel impulse reaches -- the
    ROI runner's margin for this transform (spec 2026-09-22 §5), measured from the SAME
    forward the analysis runs (never a restated filter length). The radius holding all but
    ``tol`` of each level's |W| mass, maximised over levels; tol = 1e-9 for the compact dyadic
    spline (its whole support), 1e-3 for the fractional order's algebraic tail, capped at a
    quarter of the probe grid."""
    L = min(4096, max(64, 16 * 2 ** int(n_levels)))
    img = np.zeros((L, L))
    c = L // 2
    img[c, c] = 1.0
    if wavelet == "frac_bspline":
        _S, pairs = atrous2d_forward_frac(img, int(n_levels), float(alpha))
        tol = 1e-3
    else:
        _S, pairs = mzlib.atrous2d_forward(img, int(n_levels))
        tol = 1e-9
    yy, xx = np.mgrid[0:L, 0:L]
    rho = np.hypot(yy - c, xx - c).ravel()
    inside = rho <= L / 4.0              # clear of the mirror copies' own responses
    order = np.argsort(rho[inside])
    reach = 0.0
    for W1, W2 in pairs:
        mag = np.hypot(W1[:L, :L], W2[:L, :L]).ravel()[inside][order]
        cum = np.cumsum(mag)
        if cum[-1] <= 0:
            continue
        idx = int(np.searchsorted(cum / cum[-1], 1.0 - tol))
        reach = max(reach, float(rho[inside][order][min(idx, order.size - 1)]))
    return int(np.ceil(reach))


def analyze(values, n_levels, *, coarse="full", dither=False,
            wavelet="mz_spline", alpha=3.0, interpolate=False, progress=None):
    if coarse not in ("full", "thumbnail"):
        raise ValueError(f"coarse must be 'full' or 'thumbnail', got {coarse!r}")
    if wavelet not in ("mz_spline", "frac_bspline"):
        raise ValueError(f"unknown wavelet {wavelet!r}; expected 'mz_spline' or "
                         "'frac_bspline'")
    ny, nx = values.shape
    if 2 ** n_levels > min(ny, nx):
        raise ValueError(
            f"n_levels={n_levels} exceeds the grid: 2**{n_levels} > min{(ny, nx)}"
        )
    lsb = measure_lsb(values) if dither else None
    work = _dithered(np.asarray(values, dtype=np.float64), lsb) if lsb else np.asarray(
        values, dtype=np.float64
    )

    if progress is not None:
        progress("mz forward", 0.0)
    if wavelet == "frac_bspline":
        # The fractional-order forward (section above): same torus, same Gf, fractional
        # smoothness alpha, numeric lambda. mz_spline stays the untouched mzlib path.
        S, Wpairs = atrous2d_forward_frac(work, n_levels, alpha)
    else:
        S, Wpairs = mzlib.atrous2d_forward(work, n_levels)
    if progress is not None:
        progress("mz forward", 1.0)
    extrema, mz_maxima = [], []
    for _k, (W1, W2) in enumerate(Wpairs):
        if progress is not None:
            progress(f"mz maxima {_k + 1}/{n_levels}", _k / n_levels)
        rows, cols = mzlib.nms2d_dyadic(W1, W2)
        w1, w2 = W1[rows, cols], W2[rows, cols]
        labels = _torus_components(rows, cols, W1.shape)
        mz_maxima.append((rows, cols, w1, w2))
        primary = (rows < ny) & (cols < nx)
        # app-facing line_id: a component with exactly one primary-quadrant member is an
        # isolated extremum -> -1 (the shell's "not a line" sentinel); >=2 members keep the
        # component's torus-wide id (see the module docstring's schema-equivalence note).
        primary_labels = labels[primary]
        _, inverse, counts = np.unique(primary_labels, return_inverse=True, return_counts=True)
        line_id = np.where(counts[inverse] == 1, -1, primary_labels).astype(np.int64)
        layer = {
            "x": cols[primary].astype(np.int64),
            "y": rows[primary].astype(np.int64),
            "mod": np.hypot(w1[primary], w2[primary]),
            "arg": np.arctan2(w2[primary], w1[primary]),
            "line_id": line_id,
        }
        if interpolate:
            # 2026-09-21: the same parabolic refinement wtmm2d's knob applies, on the TORUS
            # modulus raster (probes crossing into mirror halves read correct data -- that is
            # what the torus is for). ``mz_maxima`` below keeps raw integer positions and raw
            # w1/w2: the POCS constraint support stays pixel-exact (the M-Z split, explicit).
            from dynamix.core.subpixel import refine_scale
            x_sub, y_sub, mod_sub = refine_scale(
                np.hypot(W1, W2), layer["arg"], layer["x"], layer["y"])
            layer["mod"] = mod_sub
            layer["x_sub"] = x_sub
            layer["y_sub"] = y_sub
        extrema.append(layer)

    if progress is not None:
        progress(f"mz maxima {n_levels}/{n_levels}", 1.0)
    thumb = None
    if coarse == "thumbnail":
        step = 2 ** n_levels
        if ny % step or nx % step:
            raise ValueError(
                f"coarse='thumbnail' requires the grid evenly divisible by "
                f"2**n_levels={step}; got shape {(ny, nx)}"
            )
        # primary-quadrant subsample only (script 20's coding_coarse) -- subsampling the
        # whole (2ny,2nx) torus would store three redundant mirrored copies of the same data.
        thumb = S[:ny:step, :nx:step].astype(np.float32)

    return {
        "extrema": extrema,
        "mz_maxima": mz_maxima,
        "scales": (2.0 ** np.arange(1, n_levels + 1)).astype(np.float64),
        "coarse_thumb": thumb,
        "coarse_policy": coarse,
        "lsb": lsb,
        "wavelet": wavelet,
        "alpha": float(alpha),
    }


def _coarse_for(values, bundle, n_levels):
    if bundle["coarse_policy"] == "full":
        v = np.asarray(values, dtype=np.float64)
        if bundle["lsb"] is not None:
            # analyze() built its maxima constraints from the dithered field, not the raw
            # one -- reproduce that exact field (fixed seed => bit-identical) so S and the
            # maxima it's pinned against are never a field/field mismatch (controller ruling,
            # mz-edges-port review; see the module docstring's dither/coarse-consistency note).
            v = _dithered(v, bundle["lsb"])
        S, _ = mzlib.atrous2d_forward(v, n_levels)
        return S
    # coding mode (script 20's decode): mirror the stored thumbnail onto its own small torus,
    # then trig-interpolate back up to the full (2ny,2nx) torus pocs2d expects.
    thumb = bundle["coarse_thumb"].astype(np.float64)
    step = 2 ** n_levels
    return _fourier_upsample_torus(_mirror2d(thumb), step)


def preview(values, bundle, *, n_iter=10, keep=None):
    """Budgeted POCS reconstruction of ``values`` from the bundle's maxima + coarse.

    ``keep[k]`` is a boolean mask over ``bundle["mz_maxima"][k]``'s full-torus arrays —
    never over ``extrema[k]``'s primary-quadrant points. ``diag["diverging"]`` marks the
    doctrine-12 inconsistency signature; every result is budget-labeled by ``n_iter``.
    """
    if bundle.get("wavelet", "mz_spline") != "mz_spline":
        # pocs2d's projections use the standard filter bank internally (Kf/Lf/Hf at
        # n_spline=3); reconstructing a fractional analysis with them would be a silent
        # filter mismatch. Fractional POCS is a follow-up -- refuse loudly instead.
        raise ValueError(
            f"preview requires the mz_spline wavelet; this bundle was analyzed with "
            f"{bundle['wavelet']!r} (fractional POCS is a follow-up)")
    n_levels = len(bundle["mz_maxima"])
    maxima = bundle["mz_maxima"]
    if keep is not None:
        maxima = [
            (r[k], c[k], w1[k], w2[k])
            for (r, c, w1, w2), k in zip(maxima, keep)
        ]
    S_true = _coarse_for(values, bundle, n_levels)
    img_hat, resid, _ = mzlib.pocs2d(
        maxima, S_true, values.shape, n_levels, n_iter=n_iter
    )
    diag = {
        "resid": resid,
        "n_iter": n_iter,
        "diverging": bool(resid[-1] > _DIVERGENCE_RATIO * min(resid)),
    }
    return img_hat, diag
