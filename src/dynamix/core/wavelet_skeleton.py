# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Wavelet skeletons -- the Tang-You constructed wavelet + modulus-minima thinning. Pure
numpy (scipy lazy, the sieve precedent), NO GUI.

A REIMPLEMENTATION against the corpus papers (the chain_stats/spectra license -- documented
reference semantics, never a copy of anything, so no drift guard applies):

* Tang & You 2003 (``tang_you_2003_ribbon_skeleton_wavelet``): the constructed wavelet --
  1-D odd psi with COMPACT SUPPORT [-1, 1] in piecewise log/sqrt closed form (eq 4-5), even
  primitive phi, isotropic 2-D smoothing ``theta = phi(sqrt(x^2+y^2))``, wavelet pair
  ``(psi1, psi2) = grad theta`` (eq 5), modulus ``|grad W_s f|`` (eq 7). Verified
  numerically: the three 1/x singularities cancel exactly at 0, each piece vanishes at the
  breakpoint where its root goes imaginary (machine-exact continuity), and the paper's OWN
  theorem holds on synthetic ribbons -- the two maxima contours sit ~s apart with positions
  EXACTLY equal across stroke widths 4/8/12 (the width-invariance neither the Gaussian nor
  the quadratic spline delivers).
* You, Chen, Fang & Tang 2006 (``you_etal_2006_thinning_modulus_minima``): the modulus
  MINIMA of the same transform are scale-independent and sit on the medial axis --
  Algorithm 1: first WT -> threshold low modulus -> initial skeleton; second WT on the
  initial-skeleton image -> modulus minima -> skeleton; repeat until thin.

Documented deviations from the papers (general-raster adaptations, each measured):

* **phi's integration constant.** The paper says ``phi(x) := int_0^x psi`` "compactly
  supported on [-1,1]" -- but the literal reading gives phi(1) = -0.383 and no compact
  support; the primitive that IS compact anchors at the support edge:
  ``phi(rho) = -int_rho^1 psi`` (phi(1) = 0, phi(0) = +0.383, a positive bump).
* **Normalization.** The printed closed form integrates to ``int int theta = 1/2``, not
  the paper's theta_hat(0,0) = 1; the discrete kernel is unit-sum normalized (which IS
  that condition, discretely) -- a global constant, positions untouched.
* **Stroke support (Algorithm 1 step 3).** "For every point of the underlying stroke"
  presumes character figure/ground; a general raster has none, and a bare global
  ``M < T`` floods the flat background into the initial skeleton (whose mask edges then
  seed ghost skeletons at stage 2 -- measured on the synthetic stripe). The minima
  detector itself supplies the generalization: a directional minimum is BY CONSTRUCTION
  flanked by larger moduli on both sides, so stage 1 is ``modulus_minima`` of the first
  WT, gated by two LOCAL, scale-free tests -- valley contrast (``M < t_frac * local
  flanking ridge``, the paper's threshold made relative) and ridge significance (the
  local ridge clears ``edge_quantile`` of the REAL-support moduli; quantiles over the
  raw field are dust-dominated and meaningless -- measured: q0.75 of a mostly-flat
  raster is 5e-16).
* **Minima detection.** "Find all modulus minimum points": implemented as 1-D local minima
  along any of the 4 principal grid directions with a relative support floor (the
  ``no_support_mask`` philosophy of this branch: FFT dust and exact flats have no honest
  extremum) -- at the axis the transform's own direction is degenerate (W ~ 0 there), so
  the fixed-direction stencil is the robust reading.

The scale ``s`` is the support RADIUS of theta_s in PIXELS (scales-in-pixels law): the
kernel dies exactly at rho = s, and the paper's theorem reads "maxima separation = s".
The method's own hypothesis (property 2) asks the scale to COVER the stroke width; the
ridge-PAIR gate relaxes it -- detection needs only that the flanking ridge tails within
``reach`` clear the significance floor, so a width-14 stripe at s=6 still yields the
exact axis (measured: cols 63-64, nothing else) through the edges' overlapping tails.
A genuinely unreachable interior (flat beyond every tail) stays honestly empty.
"""
from __future__ import annotations

import functools

import numpy as np

# Every FFT TRANSFORM goes through the app-wide engine policy (mlx or FFTW3 at the
# configured 32/64-bit precision, never numpy's FFT by default).
from dynamix.core.fft_policy import active as _fft  # noqa: E402

__all__ = ["psi_radial", "phi_radial", "theta_kernel", "gradient_wt", "modulus_minima",
           "skeletonize"]

_PHI_TABLE: "tuple | None" = None
_PHI_N = 20001


def psi_radial(x: np.ndarray) -> np.ndarray:
    """Tang-You 2003 eq 4-5 on x >= 0, literal: three closed-form pieces, each active only
    where its square root is real; zero from 1 on. (The full 1-D wavelet is the odd
    extension; the 2-D construction only ever evaluates rho >= 0.)"""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        m1 = (x > 0) & (x < 0.25)
        m2 = (x >= 0.25) & (x < 0.75)
        m3 = (x >= 0.75) & (x < 1.0)

        def p1(t):
            r = np.sqrt(1.0 - 16.0 * t ** 2)
            return -2.0 / np.pi * (-8.0 * t * np.log((1.0 + r) / (4.0 * t))
                                   + r / (2.0 * t))

        def p2(t):
            r = np.sqrt(9.0 - 16.0 * t ** 2)
            return -2.0 / np.pi * (8.0 * t * np.log((3.0 + r) / (4.0 * t))
                                   - 3.0 * r / (2.0 * t))

        def p3(t):
            r = np.sqrt(1.0 - t ** 2)
            return -2.0 / np.pi * (-4.0 * t * np.log((1.0 + r) / t) + 4.0 * r / t)

        out[m1] = p1(x[m1]) + p2(x[m1]) + p3(x[m1])
        out[m2] = p2(x[m2]) + p3(x[m2])
        out[m3] = p3(x[m3])
    return out


def _phi_samples():
    global _PHI_TABLE
    if _PHI_TABLE is None:
        xs = np.linspace(0.0, 1.0, _PHI_N)
        ps = psi_radial(xs)
        raw = np.concatenate([[0.0], np.cumsum((ps[:-1] + ps[1:]) * 0.5 * (xs[1] - xs[0]))])
        _PHI_TABLE = (xs, raw - raw[-1])       # anchored: phi(1) = 0 (the compact primitive)
    return _PHI_TABLE


def phi_radial(rho: np.ndarray) -> np.ndarray:
    """The compact-support primitive ``phi(rho) = -int_rho^1 psi`` (see the module
    docstring's integration-constant note), by dense cached table + linear interp; zero
    beyond the support."""
    xs, tab = _phi_samples()
    rho = np.asarray(rho, dtype=np.float64)
    return np.where(rho <= 1.0, np.interp(rho, xs, tab), 0.0)


#: The smoothing-kernel menu -- the papers' OWN comparison set (Tang-You 2003 benchmarks
#: the constructed kappa against "Gaussian function and quadratic spline"; the
#: width-invariance / separation-equals-scale theorems are proven for kappa ONLY, the
#: other two are the comparison baselines). "bspline2" evaluates through the VERBATIM
#: Creep copy (core/frac_bspline, drift-guarded); support radius = s exactly for both
#: compact kernels; the Gaussian (their critique: no compact support) uses sigma = s/2,
#: which in 2-D puts ~86% of its mass inside s (the Rayleigh tail e^-2 -- NOT the 1-D
#: 95%) -- the documented convention, and their non-compactness critique made visible.
THETA_KERNELS = ("tang_you", "gaussian", "bspline2")


def theta_kernel(shape: tuple, s: float, kernel: str = "tang_you") -> np.ndarray:
    """``theta_s`` wrap-centered on ``shape`` for FFT convolution: the chosen radial
    smoothing profile at support radius s px, unit DISCRETE sum (the paper's
    theta_hat(0,0) = 1, discretely)."""
    ny, nx = shape
    y = np.arange(ny, dtype=np.float64) - ny // 2
    x = np.arange(nx, dtype=np.float64) - nx // 2
    rho = np.hypot(y[:, None], x[None, :]) / float(s)
    if kernel == "tang_you":
        k = phi_radial(rho)
    elif kernel == "gaussian":
        k = np.exp(-2.0 * rho ** 2)                    # sigma = s/2 in rho units
    elif kernel == "bspline2":
        from dynamix.core.frac_bspline import _bspline_centered

        k = np.where(rho <= 1.0, _bspline_centered(1.5 * rho, 2), 0.0)
    else:
        raise ValueError(f"unknown kernel {kernel!r}; expected one of {THETA_KERNELS}")
    total = k.sum()
    if total <= 0:
        raise ValueError(f"theta kernel has no support at scale {s!r} on shape {shape}")
    return np.fft.ifftshift(k / total)


@functools.lru_cache(maxsize=64)
def theta_reach(s: float, kernel: str = "tang_you", tol: float = 1e-9) -> int:
    """How far (px) ``theta_kernel(s)`` reaches: the radius holding all but ``tol`` of its
    discrete mass, measured on the kernel itself (compact profiles give their exact support;
    the Gaussian its 1e-9 tail). The ROI runner's margin building block."""
    L = int(np.ceil(8.0 * float(s))) + 4
    n = 2 * L + 1
    k = np.fft.fftshift(theta_kernel((n, n), float(s), kernel))
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    rho = np.hypot(yy - c, xx - c).ravel()
    order = np.argsort(rho)
    cum = np.cumsum(np.abs(k).ravel()[order])
    idx = int(np.searchsorted(cum / cum[-1], 1.0 - tol))
    return int(np.ceil(rho[order][min(idx, rho.size - 1)]))


def gradient_wt(signal: np.ndarray, s: float, *, kernel: str = "tang_you",
                pad: "int | None" = None):
    """``(W1, W2) = s * grad(f * theta_s)`` -- eq 3's convention (the s factor keeps
    moduli scale-comparable), reflect-padded FFT convolution, central-difference gradient
    (the ``gradient_measure`` precedent)."""
    f = np.asarray(signal, dtype=np.float64)
    f = np.where(np.isfinite(f), f, f[np.isfinite(f)].mean() if np.isfinite(f).any() else 0.0)
    ny, nx = f.shape
    if pad is None:
        pad = int(np.ceil(s)) + 4
    fp = np.pad(f, pad, mode="reflect")
    sm = np.real(_fft().ifft2(_fft().fft2(fp)
                              * _fft().fft2(theta_kernel(fp.shape, s, kernel))))
    gy, gx = np.gradient(sm)
    W1 = float(s) * gx[pad:pad + ny, pad:pad + nx]
    W2 = float(s) * gy[pad:pad + ny, pad:pad + nx]
    return W1, W2


def _dir_ridge(M: np.ndarray, dy: int, dx: int, reach: int) -> np.ndarray:
    """Running max of M over offsets 1..reach along direction (dy, dx), edge-padded."""
    P = np.pad(M, reach, mode="edge")
    R = np.full_like(M, -np.inf)
    ny, nx = M.shape
    for k in range(1, reach + 1):
        R = np.maximum(R, P[reach + k * dy: reach + k * dy + ny,
                            reach + k * dx: reach + k * dx + nx])
    return R


def modulus_minima(W1: np.ndarray, W2: np.ndarray, *, reach: int = 6,
                   t_frac: float = 0.5, edge_frac: float = 0.1,
                   pair_dom: float = 0.5) -> np.ndarray:
    """Boolean mask of the medial modulus-minimum points: a 1-D local minimum of
    ``|grad W|`` along one of the 4 principal grid directions, flanked WITHIN ``reach`` px
    on BOTH sides of that direction by significant ridges.

    The ridge-PAIR test is Tang-You's own framework made operational (the medial axis is
    the symmetry point between a pair of contours): ``min(R+, R-)`` -- each side's running
    max over 1..reach -- must clear ``edge_frac`` of the PEAK modulus (significance;
    peak-relative because quantiles over a mostly-flat raster are background-dominated)
    and exceed the valley by 1/t_frac (contrast). One-sided gates fail structurally off the
    axis-aligned noiseless axis: the compact kernel's support/dust boundary at ~s+d/2 is a
    one-sided-ridge minimum (68% ghost pixels on a 45-degree stripe), slope-rasterization
    ripple makes shallow high-modulus dips (killed here by t_frac against their own
    flanks), and background noise valleys have flanks far below edge_frac * peak.
    ``reach`` should be the transform scale: the paper puts the contour pair at +-s/2.

    The 1-px flank must still RISE by more than the relative floor -- roundoff wobble
    along a translation-invariant ridge is noise, not structure -- and flats/dust have no
    honest extremum (the no-support philosophy of this branch)."""
    M = np.hypot(W1, W2)
    pos = M[np.isfinite(M) & (M > 0)]
    if pos.size == 0:
        return np.zeros(M.shape, dtype=bool)
    peak = float(pos.max())
    floor = max(1e-25, 1e-6 * peak)
    edge_ref = float(edge_frac) * peak
    reach = max(1, int(reach))
    dirs = ((0, 1), (1, 0), (1, 1), (1, -1))
    half = {}
    for dy, dx in dirs:
        half[(dy, dx)] = _dir_ridge(M, dy, dx, reach)
        half[(-dy, -dx)] = _dir_ridge(M, -dy, -dx, reach)
    # Pair DOMINANCE (the ring's inner-tail circle): a valley on a
    # curved slope can flank itself with its own chord (measured: angular ripple at
    # r ~ ridge - reach paired at 0.17 peak while the true ridge loomed at 0.98 seven px
    # away in another direction). The pair that flanks a REAL medial point is comparable
    # to the strongest ridge reachable from it at all; a point whose pair is under half
    # of that is on someone else's slope. Contrast-free and local, so a genuinely weak,
    # isolated ribbon (pair == its own ridges == R_all) is untouched.
    R_all = np.maximum.reduce(list(half.values()))
    Mp = np.pad(M, 1, mode="edge")
    out = np.zeros(M.shape, dtype=bool)
    for dy, dx in dirs:
        a = Mp[1 + dy: 1 + dy + M.shape[0], 1 + dx: 1 + dx + M.shape[1]]
        b = Mp[1 - dy: 1 - dy + M.shape[0], 1 - dx: 1 - dx + M.shape[1]]
        local_min = (M <= a) & (M <= b) & (np.maximum(a, b) > M + floor)
        if not local_min.any():
            continue
        ridge_pair = np.minimum(half[(dy, dx)], half[(-dy, -dx)])
        out |= (local_min & (ridge_pair >= edge_ref)
                & (M < float(t_frac) * ridge_pair)
                & (ridge_pair >= float(pair_dom) * R_all))
    return out


def skeletonize(signal: np.ndarray, *, s1: float = 6.0, s2: float = 6.0,
                t_frac: float = 0.5, edge_frac: float = 0.1,
                n_stages: int = 2, kernel: str = "tang_you",
                pair_dom: float = 0.5, progress=None) -> dict:
    """Algorithm 1 (You et al. 2006), general-raster edition -- see the module docstring's
    deviations. Stage 1: the ridge-pair-gated modulus minima of the first WT at ``s1``
    (:func:`modulus_minima` -- reach s1, valley contrast ``t_frac``, significance
    ``edge_frac`` of peak). Each further stage re-runs the WT at ``s2`` on the previous
    skeleton image and keeps its gated minima near the previous curve. ``n_stages`` counts
    WT passes.

    Support distrust (one policy, two causes): the raster border AND every non-finite
    region grow a band of one kernel support in which no skeleton is reported -- reflect
    padding and mean-filling both manufacture structure there (the wtmm valid_ranges /
    extrema2d NaN-dilation doctrine; without it a NaN disk on a structureless ramp
    fabricates a 149-px phantom ribbon).

    POLARITY-BLIND by construction (documented consequence of dropping the paper's
    figure/ground step): a dark corridor between two bright strokes -- a stroke GAP
    included -- is itself a ribbon and gets its own medial axis (measured: a 12-row gap in
    a bright stripe draws the gap corridor's perpendicular axis, not a bridge). For
    figure/ground data that is an artifact; for general rasters it is the honest reading.
    A polarity knob (converging-vs-diverging flank gradients) is the natural future
    refinement, reserved.

    Returns ``{"skeleton" (bool), "initial" (the stage-1 mask), "mod", "arg" (stage-1
    FIELD modulus/direction), "params"}``."""
    vals = np.asarray(signal, dtype=np.float64)
    params = {"s1": float(s1), "s2": float(s2), "t_frac": float(t_frac),
              "edge_frac": float(edge_frac), "n_stages": int(n_stages),
              "kernel": str(kernel), "pair_dom": float(pair_dom)}
    W1, W2 = gradient_wt(vals, s1, kernel=kernel)
    M1 = np.hypot(W1, W2)
    arg = np.arctan2(W2, W1)
    initial = modulus_minima(W1, W2, reach=int(np.ceil(s1)), t_frac=t_frac,
                             edge_frac=edge_frac, pair_dom=pair_dom)
    from scipy.ndimage import binary_dilation

    n_total = max(1, int(n_stages))
    if progress is not None:
        progress("wavelet skeleton", 1.0 / n_total)
    skel = initial
    for stage in range(1, int(n_stages)):
        if progress is not None and stage > 1:
            progress("wavelet skeleton", stage / n_total)
        if not skel.any():
            break
        V1, V2 = gradient_wt(skel.astype(np.float64), s2, kernel=kernel)
        # Refinement stays NEAR the previous curve: radius ~s2/2 bridges raggedness while
        # excluding theta's own sidelobe valleys at ~s2 (real contrast, killed by locality).
        skel = (modulus_minima(V1, V2, reach=int(np.ceil(s2)), t_frac=t_frac,
                               edge_frac=edge_frac, pair_dom=pair_dom)
                & binary_dilation(skel, iterations=max(1, int(np.ceil(s2 / 2.0)))))
    # Support distrust: border band + one kernel support around every non-finite pixel.
    border = int(np.ceil(max(s1, s2)))
    trust = np.ones(vals.shape, dtype=bool)
    if border > 0:
        trust[:border, :] = False
        trust[-border:, :] = False
        trust[:, :border] = False
        trust[:, -border:] = False
    nonfinite = ~np.isfinite(np.asarray(signal, dtype=np.float64))
    if nonfinite.any() and border > 0:
        trust &= ~binary_dilation(nonfinite, iterations=border)
    skel = skel & trust
    initial = initial & trust
    if progress is not None:
        progress("wavelet skeleton", 1.0)
    return {"skeleton": skel, "initial": initial, "mod": M1, "arg": arg, "params": params}
