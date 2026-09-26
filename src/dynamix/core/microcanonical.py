# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Microcanonical multifractal formalism (Turiel/Yahia/Perez-Vicente 2008) -- pure numpy, NO GUI.

The OTHER formalism: per-pixel singularity exponents h(x) from wavelet projections of the
gradient measure, D(h) from the histogram method (eq 21), the Most Singular Component, and
signal reconstruction from it. This is the pipeline the canonical (Arneodo WTMM) machinery in
``wtmm_backend``/``spectra`` deliberately is NOT (see ``spectra.fit_spectra``'s method guard):
canonical/Legendre can only ever return the CONCAVE HULL of D(h) (Jaffard 1994/1997; for
measures Brown-Michon-Peyriere 1992), while the histogram method here measures D(h) directly --
the two together are the diagnosis pair for nonconcavity (Touchette & Beck 2006: a kink in
tau(q) means nonconcave-or-affine D(h), undecidable from tau alone).

This is a fresh reimplementation of the prototype in
the author's ``microcanonical_wtmm.ipynb`` notebook, with the
prototype's known defects corrected rather than ported:

* **eq-21 resolution convention (THE bug).** Turiel 2008 SS4.4 is explicit: the resolution
  ``r0`` in ``D(h) = d - log(rho(h)/rho_max)/log(r0)`` is RELATIVE -- ``1/sqrt(nx*ny)`` for an
  image -- so ``log r0 < 0`` and ``D <= d``. The notebook passed r0 in PIXELS (> 1), flipping
  the sign of the correction and bending every D(h) it produced the wrong way.
  :func:`dh_histogram` takes ``r0_rel`` and refuses anything outside (0, 1).
* **Under-padding.** The notebook reflect-padded by a constant 32 px while projecting at scales
  up to 50 px; :func:`measure_projections` defaults the pad to twice the largest scale.
* **Only the regression estimator existed.** Pont et al. 2006 compares three estimators; the
  MULTISCALE local-singularity method (log-log slope over a range of scales -- our
  :func:`singularity_map_regression`) is their best performer, resolving both tails of D(h).
  :func:`singularity_map_point` adds Pont's "punctual" method: their explicit "extreme
  simplification" that reads the FINEST scale alone (log-normalized, prefactor cancelled by
  the ensemble mean). It is single-scale, so it resolves the LEFT tail (the MSM/most-singular
  points -- with the sharpest spatial localization, which is why MSC extraction uses it) but
  NOT the right tail (very-regular points, whose signature lives in the cross-scale decay a
  single snapshot cannot see, Pont SS II.B/V.B.2). The two estimators cross-validate --
  systematic disagreement is itself diagnostic.

Kernels here are POSITIVE unit-mass smoothing kernels, not admissible zero-mean wavelets --
deliberate and per the reference (a measure is projected, and log T must exist). The kernel
choice is a resolution trade-off (Turiel 2008 appendix A): Gaussian resolves every h;
Lorentzian ``L_beta`` truncates at ``h < 2*beta - d`` but localizes the MSC sharpest; the
q-Gaussian interpolates (the prototype's own experimental default).

Frame bookkeeping: :func:`fracint2d_fourier` is a TRUE Fourier integration ``F/|k|^alpha`` on
the FIELD -- the third lift mechanism, distinct from both per-scale ``a^power`` multiplies
(1D ``expo`` in-CWT; 2D ``fracint_alpha`` via ``wtmm_backend._apply_fracint2d``). It shifts
every h by +alpha. An h-map computed on the integrated field lives in the integrated frame:
shift back with ``h - alpha`` for reporting, and reconstruct the SAME field the map was
computed on -- :func:`reconstruct_from_msc` takes the signal explicitly so the caller cannot
mix frames silently.
"""
from __future__ import annotations

import functools

import numpy as np

# Every FFT TRANSFORM goes through the app-wide engine policy (mlx or FFTW3 at the
# configured 32/64-bit precision, never numpy's FFT by default).
from dynamix.core.fft_policy import active as _fft  # noqa: E402

__all__ = [
    "fracint2d_fourier", "gradient_measure", "measure_projections", "ricker_projections",
    "relative_scale", "no_support_mask", "singularity_map_regression",
    "singularity_map_point", "dh_histogram",
    "extract_msc", "band_mask", "reconstruction_by_bands", "reconstruct_from_msc",
    "BandReconstructor",
]

_WAVELETS = ("gaussian", "lorentzian", "q_gaussian", "frac_gaussian")

#: Where a kernel is built. "real" samples it on the pixel grid and normalises it there (Turiel's
#: construction: his figure-2 discretisation argument sets the minimum scale). "fourier"
#: evaluates the SAME continuous kernel's closed-form 2-D transform (:func:`_fourier_kernel`),
#: band-limited instead of sampled -- for comparing the two.
_CONSTRUCTIONS = ("real", "fourier")

#: The multiaffine route's own wavelet names (the wavelet family is a property of the METHOD,
#: so the split devices name kernels the method's way): g2 / q_mexican / lorentzian_marr are
#: method-true aliases of the original envelope names; g1 / g3 are the integer orders of the
#: fractional family below.
_MARR_ALIASES = {"g2": "gaussian", "lorentzian_marr": "lorentzian",
                 "g1": ("frac_gaussian", 1.0), "g3": ("frac_gaussian", 3.0)}


def _borges_q(q: float) -> float:
    """``q_mexican``'s q (Borges, Tsallis, Miranda & Andrade 2004, J. Phys. A 37 9125, eq. 15:
    the second derivative of the q-Gaussian raised to the power 2 - q) as ``q_gaussian``'s q.
    ``[e_q]^(2-q)`` IS a q'-Gaussian with ``q' = 1/(2 - q)``, so in 2-D (the Laplacian) the
    two are one family, relabelled -- ``(1-u)[1 - (1-q)u/(2-q)]_+^(q/(1-q))`` with the zero
    crossing at r. At q = 2 the 2-D Laplacian vanishes identically and above 2 the profile
    never changes sign: no 2-D wavelet (the paper's q < 3 is the 1-D bound)."""
    if q >= 2.0:
        raise ValueError(f"q_tsallis={q}: the 2-D q-Mexican hat needs q < 2 (at q = 2 it "
                         "vanishes identically; above 2 it never changes sign)")
    return 1.0 / (2.0 - q)


def paired_q_beta(q: float) -> float:
    """The q-paired width of the q-family in 2-D: ``beta(q) = 1/(2(2 - q))``.

    Borges et al. 2004's ``beta = 1/(3 - q)`` (end of their section 2) holds the escort
    (q-)variance of ``e_q^(-beta x^2)`` at one: in d dimensions that variance is
    ``1/(beta(d + 2 - d q))`` per component, so the pairing is ``1/(d + 2 - d q)`` --
    ``1/(3 - q)`` in 1-D, ``1/(2(2 - q))`` in 2-D. It also pins the q-Mexican hat's zero
    crossing (which sits at the escort RMS radius), equals the fixed default 1/2 at q = 1, and
    exists exactly where the 2-D q-Mexican hat is admissible (q < 2)."""
    if q >= 2.0:
        raise ValueError(f"q_tsallis={q}: the q-paired width needs q < 2 (the 2-D escort "
                         "variance is infinite from q = 2)")
    return 1.0 / (2.0 * (2.0 - q))


def _q_mexican_stretch(q: float, q_beta: "float | None") -> float:
    """The q-Mexican hat's radial stretch for width ``q_beta`` in the scale convention where r
    is the zero crossing at q = 1, beta = 1/2: ``u -> 2 beta (2 - q) u``. 1 (today's kernel,
    zero crossing at r for every q) when ``q_beta`` is None or the paired width."""
    return 1.0 if q_beta is None else 2.0 * float(q_beta) * (2.0 - float(q))


_FRAC_U0_CACHE: dict = {}


def _frac_u0(n: float) -> float:
    """First positive root of ``1F1((n+2)/2; 1; -u)`` -- the zero crossing of the order-n
    fractional Gaussian wavelet in the scaled variable ``u = rho^2/(2 sigma^2)``. Pins the
    zero-crossing-radius scale convention for EVERY order: ``sigma^2 = r^2/(2 u0(n))`` puts
    the kernel's sign change at radius r exactly as the Ricker convention does (n=2: the
    1F1 is ``(1-u)e^-u``, u0 = 1, sigma^2 = r^2/2 -- the existing code's constant). scipy
    is imported lazily (the sieve/transect precedent) so numpy stays the core's only hard
    dependency; only the fractional orders pay it."""
    u0 = _FRAC_U0_CACHE.get(float(n))
    if u0 is None:
        from scipy.optimize import brentq
        from scipy.special import hyp1f1
        a = (float(n) + 2.0) / 2.0
        lo, hi = 1e-6, 2e-6
        while hyp1f1(a, 1.0, -hi) > 0.0:
            lo, hi = hi, hi * 2.0
            if hi > 1e9:
                raise ValueError(f"no zero crossing for frac_n={n!r} -- the order is too "
                                 "close to 0 (the n->0 limit is the crossing-free Gaussian)")
        u0 = _FRAC_U0_CACHE[float(n)] = float(brentq(lambda u: hyp1f1(a, 1.0, -u), lo, hi))
    return u0


def fracint2d_fourier(field: np.ndarray, alpha: float, pad: "int | None" = None) -> np.ndarray:
    """True fractional integration in Fourier: ``F' = F / |k|^alpha``, DC preserved.

    Linear detrend + reflect pad before the FFT, retrend after -- the prototype's own
    edge-artifact treatment, kept. Shifts every singularity exponent by +alpha; ``alpha = 0``
    returns a copy. NOT the per-scale ``a^alpha`` lift (see the module docstring's frame note).
    """
    field = np.asarray(field)
    if alpha == 0:
        return field.copy()
    ny, nx = field.shape
    if pad is None:
        pad = max(ny, nx) // 2

    yy, xx = np.mgrid[:ny, :nx].astype(np.float64)
    A = np.column_stack([np.ones(ny * nx), xx.ravel(), yy.ravel()])
    finite = np.isfinite(field.ravel())
    if finite.any():
        coef, *_ = np.linalg.lstsq(A[finite], field.ravel()[finite].astype(np.float64),
                                   rcond=None)
    else:
        coef = np.zeros(3)
    trend = coef[0] + coef[1] * xx + coef[2] * yy
    residual = np.where(np.isfinite(field), field - trend, 0.0)

    padded = np.pad(residual, pad, mode="reflect")
    ky = np.fft.fftfreq(padded.shape[0]) * padded.shape[0]
    kx = np.fft.fftfreq(padded.shape[1]) * padded.shape[1]
    K = np.hypot(*np.meshgrid(kx, ky))
    K[0, 0] = 1.0
    F = _fft().fft2(padded)
    F_filt = F / (K ** alpha)
    F_filt[0, 0] = F[0, 0]
    out = np.real(_fft().ifft2(F_filt))[pad:pad + ny, pad:pad + nx]
    return out + trend


def gradient_measure(signal: np.ndarray) -> np.ndarray:
    """``||grad s||`` -- the multifractal measure density (Turiel 2008 eq 28), central
    differences. Apply :func:`fracint2d_fourier` BEFORE this when integrating; this function
    is deliberately just the measure."""
    gy, gx = np.gradient(np.asarray(signal, dtype=np.float64))
    return np.hypot(gx, gy)


def _radial_kernel(shape: tuple, r: float, wavelet: str, beta: float,
                   q_tsallis: float, frac_n: float = 2.0,
                   q_beta: "float | None" = None) -> np.ndarray:
    """Positive unit-mass radial kernel at scale ``r`` px, wrap-centered for FFT convolution.
    Unit DISCRETE mass, so the projection is a local average of the measure and
    ``T ~ r^h(x)`` directly (the 1/r^d normalization is absorbed, as in the prototype).
    ``q_beta`` is the q-Gaussian's width ``e_q^(-q_beta rho^2/r^2)``; None is 1/2 (sigma = r
    at q = 1)."""
    ny, nx = shape
    y = np.arange(ny, dtype=np.float64) - ny // 2
    x = np.arange(nx, dtype=np.float64) - nx // 2
    rho2 = (y[:, None] ** 2 + x[None, :] ** 2) / float(r) ** 2
    if wavelet == "gaussian":
        k = np.exp(-0.5 * rho2)
    elif wavelet == "lorentzian":
        k = (1.0 + rho2) ** (-beta)
    elif wavelet == "q_gaussian":
        b = 0.5 if q_beta is None else float(q_beta)
        if q_tsallis == 1.0:
            k = np.exp(-b * rho2)
        else:
            # Tsallis's [.]_+: below q = 1 the kernel is compactly supported, radius
            # r / sqrt(b (1-q)) -- q = 0 the Epanechnikov paraboloid, q = -1 a half-dome.
            base = np.clip(1.0 + b * (q_tsallis - 1.0) * rho2, 0.0, None)
            k = base ** (-1.0 / (q_tsallis - 1.0))
    elif wavelet == "frac_gaussian":
        # The measure route's fractional Gaussian fractionalizes the ENVELOPE, exp(-rho^n/2)
        # -- never a ||k||^n weighting (a kernel whose FT vanishes at k=0 has zero mean and
        # cannot be positive, and log T must exist). n=2 is byte-for-byte the gaussian branch;
        # n < 2 fattens the tail toward Lorentzian-like localization, n > 2 flattens the top
        # toward a disk.
        k = np.exp(-0.5 * rho2 ** (0.5 * float(frac_n)))
    else:
        raise ValueError(f"unknown wavelet {wavelet!r}; expected one of {_WAVELETS}")
    k /= k.sum()
    return np.fft.ifftshift(k)


#: FFT precision is a user setting (32 or 64-bit), so the no-support decision
#: must not depend on FFT round-off. A pixel has support at scale r when some pixel of the
#: measure (|grad s| > 0) lies within the kernel's REACH -- the radius holding all but
#: ``_SUPPORT_TOL`` of its 2-D |mass|, measured on the kernel itself (Turiel 2008's
#: mu(B_r(x)) > 0, with the ball the kernel actually integrates over). Unsupported pixels get
#: T = 0 EXACTLY, which :func:`no_support_mask` flags at any precision. The tolerance matches
#: the 64-bit relative floor, so a Gaussian's reach is its 6.4-sigma 1e-9 tail.
_SUPPORT_TOL = 1e-9
_SUPPORT_CAP = 32                       # heavy tails never converge: reach <= 32 r


@functools.lru_cache(maxsize=256)
def _support_reach(route: str, wavelet: str, r: float, beta: float, q_tsallis: float,
                   frac_n: float, q_beta: "float | None" = None) -> float:
    L = int(min(np.ceil(_SUPPORT_CAP * r) + 4, 1024))
    n = 2 * L + 1
    y = np.arange(n, dtype=np.float64) - L
    rho2 = y[:, None] ** 2 + y[None, :] ** 2
    if route == "measure":
        k = np.fft.fftshift(_radial_kernel((n, n), r, wavelet, beta, q_tsallis, frac_n,
                                           q_beta=q_beta))
    else:
        stretch = 1.0
        if wavelet == "q_mexican":
            stretch = _q_mexican_stretch(q_tsallis, q_beta)
            wavelet, q_tsallis = "q_gaussian", _borges_q(q_tsallis)
        alias = _MARR_ALIASES.get(wavelet)
        fn = frac_n
        if isinstance(alias, tuple):
            wavelet, fn = alias
        elif alias is not None:
            wavelet = alias
        u0 = _frac_u0(fn) if wavelet == "frac_gaussian" else None
        k = _marr_kernel(rho2, r, wavelet, beta, q_tsallis, fn, u0, stretch=stretch)
    mass = np.abs(k).ravel()
    rho = np.sqrt(rho2).ravel()
    order = np.argsort(rho)
    cum = np.cumsum(mass[order])
    idx = int(np.searchsorted(cum / cum[-1], 1.0 - _SUPPORT_TOL))
    return float(rho[order][min(idx, rho.size - 1)])


def _zero_unsupported(T, supported, scales, route, wavelet, beta, q_tsallis, frac_n,
                      q_beta=None):
    """Set T to exactly 0 wherever no supported pixel lies within the scale's reach."""
    from scipy.ndimage import distance_transform_edt

    if supported.all():
        return T
    if not supported.any():
        T[...] = 0.0
        return T
    d = distance_transform_edt(~supported)
    for i, r in enumerate(np.asarray(scales, dtype=np.float64)):
        reach = _support_reach(route, str(wavelet), float(r), float(beta), float(q_tsallis),
                               float(frac_n), None if q_beta is None else float(q_beta))
        T[i][d > reach] = 0.0
    return T


def measure_projections(measure: np.ndarray, scales, *, wavelet: str = "gaussian",
                        beta: float = 1.0, q_tsallis: float = 1.5, frac_n: float = 2.0,
                        pad: "int | None" = None, progress=None,
                        q_beta: "float | None" = None,
                        construction: str = "real") -> np.ndarray:
    """``T(x, r)`` -- the measure convolved with the positive kernel at each scale (px).

    Returns ``(n_scales, ny, nx)`` float64. ``pad`` defaults to twice the largest scale
    (reflect), correcting the prototype's constant 32. ``construction`` is where the kernel is
    built (:data:`_CONSTRUCTIONS`); "fourier" refuses the kernels
    :func:`_check_construction` names.
    """
    _check_construction(construction, "measure", wavelet, beta, q_tsallis)
    mu = np.asarray(measure, dtype=np.float64)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    scales = np.asarray(scales, dtype=np.float64)
    ny, nx = mu.shape
    if pad is None:
        pad = int(np.ceil(2.0 * scales.max()))
    mup = np.pad(mu, pad, mode="reflect")
    F = _fft().fft2(mup)
    T = np.empty((len(scales), ny, nx), dtype=np.float64)
    for i, r in enumerate(scales):
        if construction == "fourier":
            Kf = _fourier_kernel(mup.shape, r, "measure", wavelet, beta, q_tsallis, frac_n,
                                 q_beta=q_beta)
        else:
            Kf = _fft().fft2(_radial_kernel(mup.shape, r, wavelet, beta, q_tsallis, frac_n,
                                            q_beta=q_beta))
        conv = np.real(_fft().ifft2(F * Kf))
        T[i] = conv[pad:pad + ny, pad:pad + nx]
        if progress is not None:
            progress("holder projections", (i + 1) / len(scales))
    return _zero_unsupported(T, mu > 0, scales, "measure", wavelet, beta, q_tsallis, frac_n,
                             q_beta)


def _marr_kernel(rho2, r, wavelet, beta, q_tsallis, frac_n, u0, stretch=1.0):
    """The multiaffine route's zero-crossing-at-r kernel on the squared-radius grid ``rho2``
    (before the zero-mean / L1 normalisation) -- shared by :func:`ricker_projections` and
    :func:`_support_reach`, so the reach is measured on the SAME kernel the projections use.
    ``stretch`` scales the q-Gaussian branch's ``u`` (the q-Mexican hat's width,
    :func:`_q_mexican_stretch`); 1 is the unstretched kernel."""
    if wavelet == "gaussian":
        sigma2 = float(r) ** 2 / 2.0                 # zero crossing at rho = r
        k = (1.0 - rho2 / (2.0 * sigma2)) * np.exp(-rho2 / (2.0 * sigma2))
    elif wavelet == "q_gaussian":
        q = float(q_tsallis)
        sigma2 = float(r) ** 2 / 2.0                 # same crossing: the a*m = 1/2 identity
        u = rho2 / (2.0 * sigma2) * float(stretch)
        if q == 1.0:
            k = (1.0 - u) * np.exp(-u)
        else:
            # Below q = 1 the envelope is compact (support u < 1/(1-q)); its Laplacian is
            # bounded for q >= 1/2, integrable but unbounded at the edge for 0 < q < 1/2, and
            # degenerate at q <= 0 (the edge carries a ring point samples cannot see).
            if q <= 0.0:
                raise ValueError(f"q_tsallis={q}: the q-Gaussian's Laplacian needs q > 0")
            base = 1.0 + (q - 1.0) * u
            inside = base > 0.0
            k = (1.0 - u) * np.where(inside, np.where(inside, base, 1.0)
                                     ** (-1.0 / (q - 1.0) - 2.0), 0.0)
    elif wavelet == "lorentzian":
        sigma2 = float(r) ** 2 * float(beta)         # crossing at rho = sigma/sqrt(beta)
        u = rho2 / sigma2
        k = (1.0 - float(beta) * u) * (1.0 + u) ** (-float(beta) - 2.0)
    elif wavelet == "frac_gaussian":
        from scipy.special import hyp1f1
        sigma2 = float(r) ** 2 / (2.0 * u0)          # zero crossing at rho = r, every n
        k = hyp1f1((float(frac_n) + 2.0) / 2.0, 1.0, -rho2 / (2.0 * sigma2))
    else:
        raise ValueError(f"unknown wavelet {wavelet!r}; expected one of "
                         "('gaussian', 'q_gaussian', 'lorentzian', 'frac_gaussian', "
                         "'g1', 'g2', 'g3', 'q_mexican', 'lorentzian_marr')")
    return k


def _check_construction(construction: str, route: str, wavelet: str, beta: float,
                        q_tsallis: float) -> None:
    """Refuse an unknown construction, and a measure kernel that cannot be built in Fourier:
    one with no closed-form 2-D transform, or no finite mass to normalise by."""
    if construction not in _CONSTRUCTIONS:
        raise ValueError(f"unknown construction {construction!r}; expected one of "
                         f"{_CONSTRUCTIONS}")
    if construction != "fourier" or route != "measure":
        return
    if wavelet == "frac_gaussian":
        raise ValueError("frac_gaussian cannot be built in Fourier: its envelope exp(-rho^n/2) "
                         "has no closed-form 2-D transform -- build it in real space")
    if wavelet == "q_gaussian" and float(q_tsallis) >= 2.0:
        raise ValueError(f"a q-Gaussian at q = {q_tsallis} cannot be built in Fourier: from "
                         "q = 2 its 2-D mass is infinite, so there is no unit-mass kernel -- "
                         "build it in real space")
    if wavelet == "lorentzian" and float(beta) <= 1.0:
        raise ValueError(f"a Lorentzian with beta = {beta} cannot be built in Fourier: for "
                         "beta <= 1 its 2-D mass is infinite, so there is no unit-mass kernel "
                         "-- build it in real space")


def _fourier_kernel(shape: tuple, r: float, route: str, wavelet: str, beta: float,
                    q_tsallis: float, frac_n: float = 2.0, u0: "float | None" = None,
                    q_beta: "float | None" = None, stretch: float = 1.0) -> np.ndarray:
    """The kernel's DFT on an FFT grid of ``shape``, from the closed-form 2-D transform of the
    SAME continuous kernel :func:`_radial_kernel` / :func:`_marr_kernel` sample, at scale r as
    a pure L1 dilation: the unit-scale transform evaluated at ``kappa = r |k|``.

    * ``route="measure"``: divided by its value at k = 0, so the kernel has unit mass exactly
      (the real construction's unit discrete mass).
    * ``route="marr"`` (``wavelet`` already resolved from its alias, as in
      :func:`ricker_projections`): ``2 pi`` times the Hankel transform (the DFT convention),
      so the kernel is the L1-dilated continuous wavelet -- zero mean exactly, and its L1 norm
      the mother's constant. The real construction normalises each discrete kernel to L1 = 1
      instead; the ratio is scale-independent and cancels in every slope.

    The kernel is radial, so the transform is evaluated on one quadrant of frequencies and
    mirrored."""
    from dynamix.core.q_fourier import q_gaussian_ft

    ny, nx = shape
    iy = np.minimum(np.arange(ny), ny - np.arange(ny))
    ix = np.minimum(np.arange(nx), nx - np.arange(nx))
    ky = 2.0 * np.pi * np.arange(ny // 2 + 1) / ny
    kx = 2.0 * np.pi * np.arange(nx // 2 + 1) / nx
    kap = float(r) * np.hypot(ky[:, None], kx[None, :])
    q, b = float(q_tsallis), float(beta)
    if route == "measure":
        if wavelet == "gaussian":                     # e^(-rho^2/2)
            quad = np.exp(-0.5 * kap * kap)
        elif wavelet == "q_gaussian":                 # e_q^(-w rho^2), w = q_beta (1/2)
            w = 0.5 if q_beta is None else float(q_beta)
            quad = q_gaussian_ft(kap, q, w) * (2.0 * w * (2.0 - q))
        elif wavelet == "lorentzian":                 # (1 + rho^2)^-b: q = 1 + 1/b, width b
            ql = 1.0 + 1.0 / b
            quad = q_gaussian_ft(kap, ql, b) * (2.0 * b * (2.0 - ql))
        else:
            raise ValueError(f"{wavelet!r} cannot be built in Fourier on the measure route")
        return quad[np.ix_(iy, ix)]
    with np.errstate(invalid="ignore"):              # 0 * inf at k = 0 from q' >= 2
        if wavelet == "gaussian":                     # (1 - rho^2) e^-rho^2 = -lap(e^-rho^2)/4
            quad = kap * kap * np.exp(-0.25 * kap * kap) / 8.0
        elif wavelet == "q_gaussian":                 # -lap(e_q'^(-s rho^2)) / (4 s), s = stretch
            quad = kap * kap * q_gaussian_ft(kap, q, stretch) / (4.0 * stretch)
        elif wavelet == "lorentzian":                 # -lap((1 + rho^2/b)^-b) / 4
            quad = kap * kap * q_gaussian_ft(kap, 1.0 + 1.0 / b, 1.0) / 4.0
        elif wavelet == "frac_gaussian":              # 1F1(a; 1; -u0 rho^2), a = (n + 2)/2
            from scipy.special import gammaln

            a = (float(frac_n) + 2.0) / 2.0
            quad = 2.0 * kap ** float(frac_n) * np.exp(-kap * kap / (4.0 * u0)
                                                       - a * np.log(4.0 * u0) - gammaln(a))
        else:
            raise ValueError(f"unknown wavelet {wavelet!r}")
    quad = 2.0 * np.pi * quad
    quad[0, 0] = 0.0                                  # zero mean, exactly
    return quad[np.ix_(iy, ix)]


def ricker_projections(signal: np.ndarray, scales, *, wavelet: str = "gaussian",
                       beta: float = 1.0, q_tsallis: float = 1.5, frac_n: float = 2.0,
                       pad: "int | None" = None, progress=None,
                       q_beta: "float | None" = None,
                       construction: str = "real") -> np.ndarray:
    """``|T_psi s(x, r)|`` -- the MULTIAFFINE functional (Turiel 2008 SS4.2.1): the signal
    convolved with a zero-mean 2D 2nd-order ("Mexican-hat") wavelet, absolute value taken.
    The per-pixel log-log slope of the result is the multiaffine Holder exponent gamma(x)
    directly (h_measure = gamma - 1, eq 29-31).

    Three envelopes, all exact 2-D Laplacians of their positive counterparts and all
    parameterized by the SAME zero-crossing radius r (so the fig-2 minimum-resolution rule
    reads ``r_min = 1`` for every wavelet with one crossing; more crossings, or a compact
    support's edge, raise it -- ``holder_methods.lobe_floor``):

    * ``"gaussian"`` -- the Ricker/Marr LoG, ``(1 - rho^2/2 sigma^2) exp(-rho^2/2 sigma^2)``,
      crossing at ``rho = sqrt(2) sigma``.
    * ``"q_gaussian"`` -- the q-Mexican hat, Laplacian of the Tsallis q-Gaussian:
      ``(1 - rho^2/2 sigma^2) [1 + (q-1) rho^2/2 sigma^2]^(-1/(q-1) - 2)``. The bracket is
      q-INDEPENDENT (the ``a*m = 1/2`` cancellation), so the crossing sits at
      ``rho = sqrt(2) sigma`` for every q and ``q -> 1`` recovers the Ricker exactly. Tail
      ``rho^(-2/(q-1) - 2)`` -- the GROWING ``(1 - u)`` prefactor costs two envelope powers
      (measured slope -6.00 at q = 1.5) -- so the Appendix-A truncation is ``gamma >= 2/(q-1)``.
    * ``"lorentzian"`` -- the Lorentzian-Marr, Laplacian of ``(1 + rho^2/sigma^2)^(-beta)``:
      ``(1 - beta rho^2/sigma^2)(1 + rho^2/sigma^2)^(-beta-2)``, crossing at
      ``rho = sigma/sqrt(beta)`` (so ``sigma = r sqrt(beta)``). Tail ``rho^(-2 beta - 2)``,
      NOT the envelope's ``-2 beta - 4`` -- the growing ``(1 - beta u)`` prefactor costs two
      powers (measured slopes -4.00/-5.00/-7.00 at beta = 1/1.5/2.5) -- so truncation is
      ``gamma >= 2 beta``; the slow tail is the sharp-localization end of Turiel 2008
      SS4.2.2's trade-off. (Cross-check: the q = 1.5 == L2 family identity holds for these
      bounds, 4 == 4.)
    * ``"frac_gaussian"`` -- the fractional-order family: the isotropic
      inverse FT of ``||k||^n exp(-sigma^2 k^2 / 2)``, whose exact radial form is
      ``1F1((n+2)/2; 1; -rho^2/2 sigma^2)`` -- a CONTINUOUS (Riesz) differentiation order
      ``frac_n`` generalizing the derivative ladder: ``psi_hat ~ |k|^n`` at low frequency, so
      exact moments vanish only for the INTEGER orders below n (a fractional n is not "n
      vanishing moments"), and only ``gamma < n`` is measurable. ``sigma`` is calibrated PER ORDER so the first zero crossing
      sits at radius r (:func:`_frac_u0`; n=2 gives ``(1-u)e^-u`` -- the Ricker above,
      exactly, same ``sigma^2 = r^2/2``). Away from even integer n the real-space tail is
      algebraic, ``rho^-(n+2)`` (the Riesz fractional derivative is nonlocal), so the
      heavy-tail calibration compression applies. scipy evaluates the 1F1, lazily.
    * Method-true names (the split multiaffine device's own menu): ``"g1"``/``"g3"`` are
      frac_gaussian at n = 1/3, ``"g2"`` IS the Ricker (alias of ``"gaussian"``), and
      ``"lorentzian_marr"`` aliases ``"lorentzian"`` -- :data:`_MARR_ALIASES`, identical
      arrays. ``"q_mexican"`` is Borges et al. 2004's q-Mexican hat in 2-D (its OWN q,
      -1 < q < 2): the ``"q_gaussian"`` kernel at ``q' = 1/(2 - q)`` -- see :func:`_borges_q`.

    This is the estimator for SMOOTH/function-class fields (fBm, DEMs) where the gradient
    measure is mean-dominated (Pont 2006's ill-behaved case) -- the zero mean kills the
    constant background the measure route drowns in.

    ``scales`` are ZERO-CROSSING RADII in pixels: the kernel's sign change sits at radius r
    (sigma = r/sqrt(2)), so the paper's minimum-resolution rule (figure 2: the discretized
    wavelet must separate its positive and negative parts -- ~2 samples between the zero
    crossings) reads simply ``r_min = 1`` for a single crossing. The reference's typical fit
    range is ``r_1 = 1, kappa = r_2/r_1 = 10``. Kernels are exactly zero-mean on the discrete grid and
    L1-normalized, so prefactors are scale-consistent and slopes untouched.

    ``q_beta`` (``"q_mexican"`` only) is the Borges width: None -- or the paired
    :func:`paired_q_beta` -- keeps the zero crossing at r for every q;
    a fixed width keeps r the zero crossing at q = 1, beta = 1/2 and lets it drift with q
    (``r / sqrt(2 beta (2 - q))``).

    ``construction`` is where each wavelet is built (:data:`_CONSTRUCTIONS`): "real" samples
    it, makes it zero-mean on the grid and L1-normalises it (the default, above); "fourier"
    evaluates its closed-form transform (:func:`_fourier_kernel`) -- zero mean exactly, and T
    the real construction's times the mother's L1 norm wherever both are exact.
    """
    _check_construction(construction, "marr", wavelet, beta, q_tsallis)
    wavelet_arg, frac_n_arg, q_arg = wavelet, frac_n, q_tsallis
    stretch = 1.0
    if wavelet == "q_mexican":
        stretch = _q_mexican_stretch(q_tsallis, q_beta)
        wavelet, q_tsallis = "q_gaussian", _borges_q(q_tsallis)
    alias = _MARR_ALIASES.get(wavelet)
    if isinstance(alias, tuple):
        wavelet, frac_n = alias
    elif alias is not None:
        wavelet = alias
    u0 = _frac_u0(frac_n) if wavelet == "frac_gaussian" else None
    s = np.asarray(signal, dtype=np.float64)
    s = np.where(np.isfinite(s), s, s[np.isfinite(s)].mean())
    scales = np.asarray(scales, dtype=np.float64)
    ny, nx = s.shape
    if pad is None:
        pad = int(np.ceil(4.0 * scales.max()))
    sp = np.pad(s, pad, mode="reflect")
    F = _fft().fft2(sp)
    py, px = sp.shape
    y = np.arange(py, dtype=np.float64) - py // 2
    x = np.arange(px, dtype=np.float64) - px // 2
    rho2 = y[:, None] ** 2 + x[None, :] ** 2
    T = np.empty((len(scales), ny, nx), dtype=np.float64)
    for i, r in enumerate(scales):
        if construction == "fourier":
            Kf = _fourier_kernel(sp.shape, r, "marr", wavelet, beta, q_tsallis, frac_n, u0,
                                 stretch=stretch)
        else:
            k = _marr_kernel(rho2, r, wavelet, beta, q_tsallis, frac_n, u0, stretch=stretch)
            k -= k.mean()                                # exactly zero-mean on the grid
            k /= np.abs(k).sum()
            Kf = _fft().fft2(np.fft.ifftshift(k))
        conv = np.real(_fft().ifft2(F * Kf))
        T[i] = np.abs(conv[pad:pad + ny, pad:pad + nx])
        if progress is not None:
            progress("holder projections", (i + 1) / len(scales))
    gy, gx = np.gradient(s)
    return _zero_unsupported(T, np.hypot(gx, gy) > 0, scales, "marr", wavelet_arg, beta,
                             q_arg, frac_n_arg, q_beta)


def relative_scale(r_px: float, shape: tuple) -> float:
    """Pixel scale -> the paper's RELATIVE resolution: ``r_px / sqrt(nx*ny)`` (SS4.4)."""
    return float(r_px) / float(np.sqrt(shape[0] * shape[1]))


def no_support_mask(T: np.ndarray) -> np.ndarray:
    """Pixels with NO measure/signal support at some scale in the stack ``T`` -- flat
    boolean of length ``ny*nx``, True where an exponent would be fabricated, not measured.

    The log-floor guard: a pixel with no support at some scale (a flat/quantized/nodata-filled
    patch: ``||grad s|| = 0``, so the projection is ~0 until the kernel reaches real gradients)
    gets the log floor at that scale -- a -69 in the regression that manufactures slopes of
    +-hundreds. Those pixels have no honest exponent: NaN, not garbage. (Dither "fixes" this by
    giving flats random tiny gradients -- which is exactly the snow the sieve then has to
    clean.) Relative floor: exact zeros AND numerical dust (FFT leakage ~1e-16 of the signal)
    both count as "no support" -- an absolute cutoff misses the dust, whose log still swings
    tens of e-folds and fabricates slopes.

    Shared by BOTH estimators (the punctual estimator reads the one badly-sampled finest
    scale, where the multiaffine kernel's discrete mean-subtraction DC-couples a flat patch to
    the whole image at ~1e-3 of the mean -- a uniform fabricated h the single-scale dust floor
    cannot see; the CROSS-scale floor can, because well-sampled scales drop to dust on the same
    pixels). Support is a property of the DATA, so the estimators agree on WHERE an exponent
    exists and differ only in HOW it is estimated."""
    flat = np.asarray(T, dtype=np.float64).reshape(T.shape[0], -1)
    scale_mag = np.nanmean(np.where(flat > 0, flat, np.nan), axis=1, keepdims=True)
    scale_mag = np.where(np.isfinite(scale_mag) & (scale_mag > 0), scale_mag, 1.0)
    # The relative floor follows the FFT precision -- 32-bit round-off (~1e-7 of the peak)
    # would otherwise pass a 1e-9 floor and fabricate slopes. Structural no-support is decided
    # exactly upstream (_zero_unsupported); this catches round-off dust only.
    rel = 1e-9 if getattr(_fft(), "precision", 64) == 64 else 1e-6
    return (flat <= np.maximum(1e-25, rel * scale_mag)).any(axis=0)


def singularity_map_regression(T: np.ndarray, scales, *, fit_range=None,
                               r2_min: float = 0.8):
    """h(x) as the per-pixel OLS slope of ``log T`` vs ``log r`` (eq 16), vectorized.

    ``fit_range=(r_lo, r_hi)`` restricts the scales used (px, inclusive); needs >= 3.
    Returns ``(h_map, r2_map)`` float32 -- h is NaN where R^2 < r2_min (the prototype's gate).

    The gate is for junk pixels (zeros, borders), NOT a quality filter on intermittent
    measures: there, per-pixel log T vs log r is the random walk of the cascade weights, its
    wiggle is physics, and gating on R^2 is a selection bias (measured on the synthetic
    cascade: r2_min=0.8 shifted the retained median by +0.1). Pass ``r2_min=0`` when
    estimating spectra and judge dispersion downstream.
    """
    scales = np.asarray(scales, dtype=np.float64)
    n_scales, ny, nx = T.shape
    if fit_range is not None:
        mask = (scales >= fit_range[0]) & (scales <= fit_range[1])
    else:
        mask = np.ones(n_scales, dtype=bool)
    if mask.sum() < 3:
        raise ValueError(f"need >= 3 scales in fit range, got {int(mask.sum())}")
    log_r = np.log(scales[mask])
    Tm = T[mask]
    log_T = np.log(np.maximum(Tm, 1e-30)).reshape(mask.sum(), -1)
    floored = no_support_mask(Tm)

    A = np.column_stack([log_r, np.ones(len(log_r))])
    coeffs = np.linalg.inv(A.T @ A) @ (A.T @ log_T)
    resid = log_T - A @ coeffs
    ss_res = (resid ** 2).sum(axis=0)
    ss_tot = ((log_T - log_T.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-30)

    h_map = coeffs[0].reshape(ny, nx).astype(np.float32)
    r2_map = r2.reshape(ny, nx).astype(np.float32)
    h_map[r2_map < r2_min] = np.nan
    h_map[floored.reshape(ny, nx)] = np.nan
    return h_map, r2_map


def singularity_map_point(T0: np.ndarray, r0_rel: float) -> np.ndarray:
    """Finest-scale point estimate: ``h(x) = log(T(x,r0) / <T(r0)>) / log(r0_rel)``.

    The ensemble-mean normalization cancels the mean prefactor alpha(x) up to o(1/log r0)
    (Turiel 2008 SS4.4). This is Pont et al. 2006's "punctual" method (SS II.B) -- their
    explicit "extreme simplification" of the multiscale local-singularity regression: crude
    but PARAMETER-FREE and the sharpest LOCALIZATION (one finest scale, no cross-scale blur),
    so it is the natural MSM/left-tail extractor. Its known limit: it cannot resolve the RIGHT
    tail (very regular, large-h points -- SS V.B.2), for which use
    :func:`singularity_map_regression`.
    ``r0_rel`` is the RELATIVE resolution of the finest scale (:func:`relative_scale`);
    anything outside (0, 1) is refused -- the same convention bug guard as
    :func:`dh_histogram`. No-support projections give NaN: not just non-positive values but
    numerical dust below the RELATIVE floor of :func:`singularity_map_regression` -- FFT
    leakage keeps a nodata-flat pixel "positive" at ~1e-16 of the signal, and a plain ``> 0``
    gate would fabricate a uniform, plausible-looking h across the whole flat. Same floor,
    single scale.
    """
    if not 0.0 < r0_rel < 1.0:
        raise ValueError(f"r0_rel must be the RELATIVE resolution in (0, 1), got {r0_rel!r} "
                         "-- pass relative_scale(r_px, shape), never a pixel scale")
    T0 = np.asarray(T0, dtype=np.float64)
    pos = np.isfinite(T0) & (T0 > 0)
    if not pos.any():
        return np.full(T0.shape, np.nan, dtype=np.float32)
    scale_mag = T0[pos].mean()
    rel = 1e-9 if getattr(_fft(), "precision", 64) == 64 else 1e-6   # per precision
    pos &= T0 > np.maximum(1e-25, rel * scale_mag)
    if not pos.any():
        return np.full(T0.shape, np.nan, dtype=np.float32)
    mean = T0[pos].mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.log(T0 / mean) / np.log(r0_rel)
    return np.where(pos, h, np.nan).astype(np.float32)


def dh_histogram(h_map: np.ndarray, *, r0_rel: "float | None" = None, d: float = 2.0,
                 n_bins: int = 80, h_range=None):
    """D(h) by the histogram method -- eq 21, with the CORRECT resolution convention.

    ``D(h) = d - log(rho(h)/rho_max) / log(r0_rel)`` where ``r0_rel`` is RELATIVE (in (0,1);
    the paper's own ``1/sqrt(nx*ny)`` is the default, from ``h_map``'s size). ``log r0_rel``
    is then negative, so ``D <= d`` with equality at the mode -- passing a PIXEL scale (>= 1)
    flips the correction's sign and is refused loudly (the prototype notebook's bug).

    Returns ``(h_centers, D)``; bins with zero counts are NaN.
    """
    h_valid = h_map[np.isfinite(h_map)].ravel()
    if h_valid.size < 10:
        return np.array([]), np.array([])
    if r0_rel is None:
        r0_rel = 1.0 / np.sqrt(h_map.size)
    if not 0.0 < r0_rel < 1.0:
        raise ValueError(f"r0_rel must be the RELATIVE resolution in (0, 1), got {r0_rel!r} "
                         "-- the paper's convention is 1/sqrt(nx*ny) (Turiel 2008 SS4.4); "
                         "passing a pixel scale flips the sign of the eq-21 correction")
    if h_range is None:
        h_range = (np.percentile(h_valid, 1), np.percentile(h_valid, 99))
    counts, edges = np.histogram(h_valid, bins=n_bins, range=h_range, density=True)
    h_centers = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        D = d - np.log(counts / counts.max()) / np.log(r0_rel)
    D[counts == 0] = np.nan
    return h_centers, D


def band_mask(h_map: np.ndarray, h_lo: float, h_hi: float) -> np.ndarray:
    """The singularity set for an ARBITRARY h band, ``h_lo <= h < h_hi`` -- the MSC generalized:
the informative manifold need not be the MOST singular component, so
    reconstruction and set displays take any band. ``extract_msc`` is the ``(-inf, h_theta)``
    special case."""
    h = np.asarray(h_map)
    return np.isfinite(h) & (h >= h_lo) & (h < h_hi)


def reconstruction_by_bands(signal: np.ndarray, h_map: np.ndarray, edges, *,
                            cumulative: bool = True) -> list:
    """Reconstruction quality per h band (the Turiel 2009 fig-13 study, generalized).

    ``edges`` (ascending) define bands ``[e_i, e_{i+1})`` -- use ``-inf``/``+inf`` outer
    edges when full coverage matters (quantile edges can strand boundary pixels). ``cumulative=True`` adds bands from
    the most singular (lowest h) upward and reconstructs at each stage -- the progressive
    grid of the reference figures; ``False`` reconstructs each band ALONE, which is the
    which-band-carries-the-features question. Returns one dict per band:
    ``{h_lo, h_hi, density, psnr_db, rel_err}`` with density = the mask's fraction of finite
    h pixels. The signal must be the SAME field the h-map was computed on (frame note in the
    module docstring)."""
    h = np.asarray(h_map)
    edges = np.asarray(edges, dtype=np.float64)
    finite = np.isfinite(h)
    n_finite = max(int(finite.sum()), 1)
    acc = np.zeros(h.shape, dtype=bool)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = band_mask(h, float(lo), float(hi))
        acc = (acc | m) if cumulative else m
        _recon, psnr, rel_err = reconstruct_from_msc(signal, acc)
        rows.append({"h_lo": float(lo), "h_hi": float(hi),
                     "density": float(acc.sum()) / n_finite,
                     "psnr_db": float(psnr), "rel_err": float(rel_err)})
    return rows


class BandReconstructor:
    """Interactive band-reconstruction session: pay the setup ONCE (decimation, symmetric
    padding, gradients, frequency kernels), then :meth:`reconstruct` any h band in two forward
    rFFTs + one inverse on the decimated grid -- the live-histogram-drag engine (real-time filtering, no lag). The h-map is FROZEN during a drag, so only the
    mask changes per tick; everything mask-independent lives here.

    ``max_dim`` decimates (plain striding) so the larger dimension is <= it -- the same
    preview-then-refine posture as the finest-first WTMM preview; ``stride`` is exposed so the
    caller can build matching display axes. At ``max_dim`` >= the field and the same ``pad``,
    :meth:`reconstruct` matches :func:`reconstruct_from_msc` on the band mask exactly.
    """

    def __init__(self, signal: np.ndarray, h_map: np.ndarray, *,
                 max_dim: int = 512, pad: "int | None" = None):
        s = np.asarray(signal, dtype=np.float64)
        h = np.asarray(h_map, dtype=np.float64)
        if s.shape != h.shape:
            raise ValueError(f"signal {s.shape} and h_map {h.shape} must share a grid")
        self.stride = max(1, -(-max(s.shape) // int(max_dim)))
        s = s[::self.stride, ::self.stride]
        h = h[::self.stride, ::self.stride]
        ny, nx = s.shape
        if pad is None:
            pad = max(ny, nx) // 2
        self._pad, self._ny, self._nx = int(pad), ny, nx
        sp = np.pad(s, self._pad, mode="symmetric") if self._pad else s
        gy, gx = np.gradient(sp)
        self._gx, self._gy = gx, gy
        self._hp = np.pad(h, self._pad, mode="symmetric") if self._pad else h
        FY = np.fft.fftfreq(sp.shape[0])[:, None]
        FX = np.fft.rfftfreq(sp.shape[1])[None, :]
        F2 = FX ** 2 + FY ** 2
        F2[0, 0] = 1.0
        self._kx = -1j / (2.0 * np.pi) * FX / F2
        self._ky = -1j / (2.0 * np.pi) * FY / F2
        self._mean = float(s.mean())
        self._full_shape = sp.shape

    def reconstruct(self, h_lo: float, h_hi: float) -> np.ndarray:
        """The (decimated) reconstruction from ``h in [h_lo, h_hi)`` -- float32, mean-matched
        to the signal, shape ``(ny, nx)`` of the decimated grid."""
        m = np.isfinite(self._hp) & (self._hp >= h_lo) & (self._hp < h_hi)
        gx_hat = _fft().rfft2(np.where(m, self._gx, 0.0))
        gy_hat = _fft().rfft2(np.where(m, self._gy, 0.0))
        s_hat = self._kx * gx_hat + self._ky * gy_hat
        s_hat[0, 0] = 0.0
        r = _fft().irfft2(s_hat, s=self._full_shape)
        if self._pad:
            r = r[self._pad:self._pad + self._ny, self._pad:self._pad + self._nx]
        return (r - r.mean() + self._mean).astype(np.float32)


def extract_msc(h_map: np.ndarray, h_theta: float = 0.0):
    """Most Singular Component: ``{x : h(x) < h_theta}``. Returns ``(mask, density)`` where
    density is the MSC fraction of the finite-h pixels."""
    finite = np.isfinite(h_map)
    mask = finite & (h_map < h_theta)
    n_finite = int(finite.sum())
    density = float(mask.sum()) / n_finite if n_finite else 0.0
    return mask, density


def reconstruct_from_msc(signal: np.ndarray, msc_mask: np.ndarray, *,
                         pad: "int | None" = None):
    """Reconstruct from the essential gradient of ANY singularity set (Turiel 2008 eq 65;
    Turiel-del Pozo 2002; generalized to arbitrary h bands via :func:`band_mask`).

    ``s_hat = -i/(2 pi) * (f . g_hat) / ||f||^2`` over the gradient restricted to the mask --
    the universal kernel, in numpy's fftfreq (cycles/sample) convention. ``signal`` must be
    the SAME field the h-map was computed on (integrated, if fracint was applied) -- see the
    module docstring's frame note.

    **Border handling.** The Fourier kernel assumes periodicity; on any
    non-periodic field (every real image/DEM) the wrap-around gradient mismatch corrupts the
    inversion GLOBALLY -- measured 14.8 dB full-mask on the M-Z house vs 44 dB on (periodic)
    fBm; symmetric extension recovers 34.7 dB at half-size padding. Signal AND mask are
    symmetric-extended by ``pad`` (default: half the larger dimension), inverted on the
    extended domain, cropped back; metrics are on the crop. ``pad=0`` gives the raw
    periodic inversion (kept reachable for periodic synthetics).

    Returns ``(recon float32, psnr_db, rel_err)`` -- PSNR against the signal's own range.
    """
    signal = np.asarray(signal, dtype=np.float64)
    mask = np.asarray(msc_mask, dtype=bool)
    ny, nx = signal.shape
    if pad is None:
        pad = max(ny, nx) // 2
    if pad:
        s = np.pad(signal, pad, mode="symmetric")
        m = np.pad(mask, pad, mode="symmetric")
    else:
        s, m = signal, mask
    gy, gx = np.gradient(s)
    gx_hat = _fft().fft2(np.where(m, gx, 0.0))
    gy_hat = _fft().fft2(np.where(m, gy, 0.0))

    FX, FY = np.meshgrid(np.fft.fftfreq(s.shape[1]), np.fft.fftfreq(s.shape[0]))
    F2 = FX ** 2 + FY ** 2
    F2[0, 0] = 1.0
    s_hat = -1j / (2.0 * np.pi) * (FX * gx_hat + FY * gy_hat) / F2
    s_hat[0, 0] = 0.0
    recon = np.real(_fft().ifft2(s_hat))
    if pad:
        recon = recon[pad:pad + ny, pad:pad + nx]
    recon = recon - recon.mean() + signal.mean()

    span = float(signal.max() - signal.min())
    mse = float(np.mean((signal - recon) ** 2))
    psnr = 10.0 * np.log10(span ** 2 / max(mse, 1e-30))
    rel_err = float(np.sqrt(mse) / max(span, 1e-30))
    return recon.astype(np.float32), psnr, rel_err
