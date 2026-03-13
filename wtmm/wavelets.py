"""Wavelet definitions for CWT (Gaussian derivative family + compact-support wavelets)."""

import numpy as np
from math import comb
from scipy.special import factorial, gamma as gamma_func


# ---------------------------------------------------------------------------
# Gaussian derivative wavelets (faithful to LastWave wt1d_collection.c)
# ---------------------------------------------------------------------------
#
# Each wavelet has a stretch factor (gNfact) baked into the wavelet definition.
# No intrinsic normalization -- all normalization comes from expo parameter AFTER convolution.
# Support ranges determine filter truncation.

WAVELETS = {
    'g0': {
        'fact': 1.7,
        'func': lambda u: np.exp(-u**2 / 2),
        'x_min_factor': -np.sqrt(2 * np.log(10)) * 2.44949,  # -gfact*sqrt2log10*geps
        'x_max_factor':  np.sqrt(2 * np.log(10)) * 2.44949,
    },
    'g1': {
        'fact': 2.8,
        'func': lambda u: u * np.exp(-u**2 / 2),
        'x_min_factor': -6.0,
        'x_max_factor':  6.0,
    },
    'g2': {
        'fact': 3.0,
        'func': lambda u: (u**2 - 1) * np.exp(-u**2 / 2),
        'x_min_factor': -6.0,
        'x_max_factor':  6.0,
    },
    'g3': {
        'fact': 3.2,
        'func': lambda u: -u * (3 - u**2) * np.exp(-u**2 / 2),
        'x_min_factor': -6.3,
        'x_max_factor':  6.3,
    },
    'g4': {
        'fact': 3.2,
        'func': lambda u: (3 - 6*u**2 + u**4) * np.exp(-u**2 / 2),
        'x_min_factor': -6.4,
        'x_max_factor':  6.4,
    },
}


# ---------------------------------------------------------------------------
# Part A: B-spline derivative wavelets (bspline1 -- bspline4)
# ---------------------------------------------------------------------------

def _bspline_centered(u, m):
    """Centered cardinal B-spline of order m via Cox-de Boor recursion.

    Support: [-(m+1)/2, (m+1)/2].
    Uses the standard recursion B_m(t) with knots at integers 0..m+1,
    then shifts to center at 0.
    """
    # Shift to standard (uncentered) knots: t in [0, m+1]
    t = np.asarray(u, dtype=np.float64) + (m + 1) / 2.0
    result = np.zeros_like(t)

    # Explicit formula: B_m(t) = 1/m! * sum_{k=0}^{m+1} (-1)^k C(m+1,k) * max(t-k,0)^m
    for k in range(m + 2):
        sign = (-1) ** k
        coeff = comb(m + 1, k)
        shifted = np.maximum(t - k, 0.0)
        result = result + sign * coeff * shifted ** m
    result = result / factorial(m, exact=True)
    return result


def _bspline_derivative(u, m, n):
    """n-th derivative of centered B-spline of order m.

    Uses the finite-difference identity:
        B_m^(n)(t) = sum_{k=0}^{n} (-1)^k C(n,k) B_{m-n}(t + n/2 - k)
    """
    result = np.zeros_like(np.asarray(u, dtype=np.float64))
    for k in range(n + 1):
        sign = (-1) ** k
        coeff = comb(n, k)
        result = result + sign * coeff * _bspline_centered(u + n / 2.0 - k, m - n)
    return result


# Register bspline wavelets: m = n + 2 ensures C^1 continuity
# fact calibrated to match g2 physical extent: fact = 18.0 / half_support
_BSPLINE_SPECS = [
    # (name, n_vanishing_moments, m_order, half_support, fact)
    ('bspline1', 1, 3, 2.0, 9.0),
    ('bspline2', 2, 4, 2.5, 7.2),
    ('bspline3', 3, 5, 3.0, 6.0),
    ('bspline4', 4, 6, 3.5, 5.14),
]

for _name, _n, _m, _hs, _fact in _BSPLINE_SPECS:
    def _make_bspline_func(m, n):
        def func(u):
            return _bspline_derivative(u, m, n)
        return func

    WAVELETS[_name] = {
        'fact': _fact,
        'func': _make_bspline_func(_m, _n),
        'x_min_factor': -_hs,
        'x_max_factor':  _hs,
    }


# ---------------------------------------------------------------------------
# Part A2: Fractional B-spline wavelets (Unser & Blu, 2000)
# ---------------------------------------------------------------------------
#
# Extends B-splines to non-integer order alpha > -1/2.
# Uses Gamma functions instead of factorials.
# These are NEW functions — existing _bspline_centered / _bspline_derivative
# are NOT modified.

def _frac_bspline_centered(u, alpha):
    """Centered fractional B-spline of continuous order alpha.

    Extends _bspline_centered to real-valued alpha > -1/2.
    Support: [-(alpha+1)/2, (alpha+1)/2].

    Uses the explicit formula:
        beta_alpha(t) = 1/Gamma(alpha+1) * sum_{k=0}^{floor(t)} (-1)^k C_frac(alpha+1, k) * (t-k)^alpha

    where C_frac(alpha+1, k) = Gamma(alpha+2) / (Gamma(k+1) * Gamma(alpha+2-k))
    is the generalized binomial coefficient.

    Reference: Unser & Blu, "Fractional Splines and Wavelets", SIAM Review 2000.
    """
    t = np.asarray(u, dtype=np.float64) + (alpha + 1) / 2.0
    result = np.zeros_like(t)

    # Upper limit for the sum: we need t - k >= 0, so k <= t
    # But also the generalized binomial coeff C_frac(alpha+1, k) decays,
    # and the support is [0, alpha+1], so k goes up to floor(alpha+1)
    k_max = int(np.floor(alpha + 1)) + 1

    for k in range(k_max + 1):
        # Generalized binomial coefficient: C(alpha+1, k) using Gamma
        if k == 0:
            binom_k = 1.0
        else:
            binom_k = gamma_func(alpha + 2) / (gamma_func(k + 1) * gamma_func(alpha + 2 - k))

        sign = (-1.0) ** k
        shifted = np.maximum(t - k, 0.0)
        result = result + sign * binom_k * shifted ** alpha

    result = result / gamma_func(alpha + 1)
    return result


def _frac_bspline_derivative(u, alpha, n_alpha):
    """Fractional derivative of fractional B-spline.

    Computes the n_alpha-th order (possibly non-integer) derivative of
    the fractional B-spline of order alpha, using fractional finite differences:

        psi_alpha^(n)(t) = sum_{k=0}^{ceil(n)} (-1)^k C_frac(n, k) * beta_{alpha-n}(t + n/2 - k)

    For integer n_alpha, this reduces to _bspline_derivative.
    For non-integer n_alpha, uses generalized binomial coefficients.

    The resulting wavelet has n_alpha vanishing moments (fractional).

    Parameters
    ----------
    u : array_like
        Evaluation points (centered coordinates).
    alpha : float
        B-spline order (> -1/2 + n_alpha).
    n_alpha : float
        Derivative order (vanishing moments). Can be non-integer.
    """
    u = np.asarray(u, dtype=np.float64)
    result = np.zeros_like(u)

    # Number of terms: for fractional differences, we need enough terms
    # for convergence. The series C_frac(n, k) * (-1)^k decays for k > n.
    # Use ceil(n) + some extra terms for non-integer n.
    n_terms = int(np.ceil(n_alpha)) + 1

    # Remaining B-spline order
    alpha_rem = alpha - n_alpha

    for k in range(n_terms + 1):
        if k == 0:
            binom_k = 1.0
        else:
            binom_k = gamma_func(n_alpha + 1) / (gamma_func(k + 1) * gamma_func(n_alpha + 1 - k))

        sign = (-1.0) ** k
        result = result + sign * binom_k * _frac_bspline_centered(u + n_alpha / 2.0 - k, alpha_rem)

    return result


def register_frac_bspline(alpha, n_alpha=None, name=None):
    """Register a fractional B-spline wavelet in WAVELETS dict.

    Parameters
    ----------
    alpha : float
        B-spline order (e.g. 3.5). Must be > -0.5 + n_alpha.
    n_alpha : float, optional
        Derivative order (vanishing moments). Default: alpha - 2
        (ensures C^1 continuity like integer bsplines).
    name : str, optional
        Wavelet name. Default: f'fbspline{alpha}'.

    Returns
    -------
    name : str, the registered wavelet name.
    """
    if n_alpha is None:
        n_alpha = alpha - 2.0
    if alpha - n_alpha <= -0.5:
        raise ValueError(f"Need alpha - n_alpha > -0.5, got {alpha - n_alpha}")
    if name is None:
        name = f'fbspline{alpha}'

    half_support = (alpha + 1) / 2.0
    fact = 18.0 / half_support  # Match existing bspline calibration convention

    def _make_func(a, n):
        def func(u):
            return _frac_bspline_derivative(u, a, n)
        return func

    WAVELETS[name] = {
        'fact': fact,
        'func': _make_func(alpha, n_alpha),
        'x_min_factor': -half_support,
        'x_max_factor':  half_support,
    }
    return name


# ---------------------------------------------------------------------------
# Part A3: Tsallis q-Gaussian derivative wavelets (legacy)
# ---------------------------------------------------------------------------
#
# The q-Gaussian e_q(u) generalizes exp(-u^2/2):
#   q < 1:  [1 - (1-q)*u^2/2]_+^{1/(1-q)}  — compact support
#   q = 1:  exp(-u^2/2)                       — standard Gaussian
#   q > 1:  [1 + (q-1)*u^2/2]^{-1/(q-1)}     — power-law tails ~ |u|^{-2/(q-1)}
#
# The n-th derivative of e_q gives a wavelet with n vanishing moments
# and continuously tunable tail behavior.

def _q_gaussian(u, q_tsallis):
    """Tsallis q-Gaussian: e_q(u).

    Parameters
    ----------
    u : array_like
        Evaluation points.
    q_tsallis : float
        Tsallis parameter. q=1 gives standard Gaussian.
        q<1: compact support. 1<q<3: power-law tails.
    """
    u = np.asarray(u, dtype=np.float64)
    if abs(q_tsallis - 1.0) < 1e-12:
        return np.exp(-u**2 / 2)
    elif q_tsallis < 1.0:
        # Compact support: [1 - (1-q)*u^2/2]_+^{1/(1-q)}
        base = 1.0 - (1.0 - q_tsallis) * u**2 / 2.0
        return np.where(base > 0, base ** (1.0 / (1.0 - q_tsallis)), 0.0)
    else:
        # Heavy tails: [1 + (q-1)*u^2/2]^{-1/(q-1)}, valid for q < 3
        if q_tsallis >= 3.0:
            raise ValueError(f"q_tsallis must be < 3, got {q_tsallis}")
        base = 1.0 + (q_tsallis - 1.0) * u**2 / 2.0
        return base ** (-1.0 / (q_tsallis - 1.0))


def _q_gaussian_derivative(u, q_tsallis, n):
    """n-th derivative of the q-Gaussian, computed analytically.

    Uses the chain rule on e_q(u) = B(u)^p where:
      B(u) = 1 + s * u^2 / 2    (s = q-1 for q>1, s = -(1-q) for q<1)
      p = -1/(q-1)              (exponent)

    The derivatives are polynomials in u times B(u)^(p-n), which
    guarantees exact zero integral (admissibility).

    Parameters
    ----------
    u : array_like
        Evaluation points.
    q_tsallis : float
        Tsallis parameter.
    n : int
        Derivative order (1, 2, 3, or 4).
    """
    u = np.asarray(u, dtype=np.float64)

    if n == 0:
        return _q_gaussian(u, q_tsallis)

    # For q=1 (standard Gaussian), use Hermite polynomial forms matching g1-g4.
    # Convention: g_n(u) = He_n(u) * exp(-u^2/2) = (-1)^n * d^n/du^n [exp(-u^2/2)]
    if abs(q_tsallis - 1.0) < 1e-12:
        gauss = np.exp(-u**2 / 2)
        if n == 1:
            return u * gauss
        elif n == 2:
            return (u**2 - 1) * gauss
        elif n == 3:
            return u * (u**2 - 3) * gauss
        elif n == 4:
            return (u**4 - 6*u**2 + 3) * gauss
        else:
            raise ValueError(f"Derivative order n={n} not supported (max 4)")

    # General q != 1
    # e_q(u) = B^p  where B = 1 + s*u^2/2, s = (q-1), p = 1/(1-q)
    # We compute (-1)^n * d^n/du^n [B^p] to match the g_n sign convention.
    s = q_tsallis - 1.0   # positive for q>1, negative for q<1
    p = 1.0 / (1.0 - q_tsallis)  # negative for q>1, positive for q<1
    u2 = u**2
    B = 1.0 + s * u2 / 2.0

    if q_tsallis < 1.0:
        mask = B > 0
        # Regularize B near boundary to avoid singularities in B^{p-k}
        # when p < k. Use small epsilon so the wavelet is well-behaved.
        B_reg = np.where(mask, np.maximum(B, 1e-15), 1e-15)
    else:
        mask = np.ones_like(u, dtype=bool)
        B_reg = B

    sign = (-1.0) ** n  # match g_n convention

    if n == 1:
        result = sign * p * s * u * B_reg ** (p - 1)

    elif n == 2:
        bracket = B_reg + (p - 1) * s * u2
        result = sign * p * s * B_reg ** (p - 2) * bracket

    elif n == 3:
        bracket = 3 * B_reg + (p - 2) * s * u2
        result = sign * p * (p - 1) * s**2 * u * B_reg ** (p - 3) * bracket

    elif n == 4:
        A = B_reg + (p - 3) * s * u2
        C = 3 * B_reg + (p - 2) * s * u2
        inner = A * C + (2*p - 1) * s * u2 * B_reg
        result = sign * p * (p - 1) * s**2 * B_reg ** (p - 4) * inner
    else:
        raise ValueError(f"Derivative order n={n} not supported (max 4)")

    if q_tsallis < 1.0:
        result = np.where(mask, result, 0.0)

    return result


def register_q_gaussian(q_tsallis, n=2, name=None):
    """Register a q-Gaussian derivative wavelet in WAVELETS dict.

    Parameters
    ----------
    q_tsallis : float
        Tsallis parameter. q=1 gives standard Gaussian derivative (g_n).
        q<1: compact support. 1<q<3: power-law tails.
    n : int
        Derivative order (vanishing moments). Default 2 (like g2).
    name : str, optional
        Wavelet name. Default: f'qg{n}_q{q_tsallis}'.

    Returns
    -------
    name : str, the registered wavelet name.
    """
    if name is None:
        name = f'qg{n}_q{q_tsallis}'

    if abs(q_tsallis - 1.0) < 1e-12:
        # Standard Gaussian: use existing g_n support
        half_support = 6.0 + 0.1 * n
        fact = 3.0
    elif q_tsallis < 1.0:
        # Compact support: |u| < sqrt(2/(1-q))
        u_max = np.sqrt(2.0 / (1.0 - q_tsallis))
        half_support = u_max + 0.1  # small margin
        fact = 18.0 / half_support
    else:
        # Heavy tails: truncate where e_q drops below eps.
        # Use eps=1e-3 for practical filter widths (power-law tails
        # decay slowly, 1e-6 gives huge filters).
        eps = 1e-3
        u_trunc = np.sqrt(2.0 * (eps**(-(q_tsallis - 1.0)) - 1.0) / (q_tsallis - 1.0))
        half_support = min(u_trunc, 20.0)  # cap to prevent extreme filters
        fact = 18.0 / half_support

    def _make_func(qt, nn):
        def func(u):
            return _q_gaussian_derivative(u, qt, nn)
        return func

    WAVELETS[name] = {
        'fact': fact,
        'func': _make_func(q_tsallis, n),
        'x_min_factor': -half_support,
        'x_max_factor':  half_support,
    }
    return name


# ---------------------------------------------------------------------------
# Part A4: Borges 2004 q-Mexican hat (proper construction)
# ---------------------------------------------------------------------------
#
# From Borges, Tsallis, Miranda & Andrade (2004), J. Phys. A 37, 9125-9137.
# "Mother wavelet functions generalized through q-exponentials"
#
# The q-Mexican hat is NOT simply d^2/dx^2 of e_q(x), but rather:
#   ψ_q(x) = d^2/dx^2 [e_q^{-βx^2}]^{2-q}     (eq. 15)
#           = A_q [1 - (3-q)βx^2] [e_q^{-βx^2}]^q   (eq. 16)
#
# This construction raises the q-exponential to the power (2-q) BEFORE
# differentiating, which ensures proper admissibility for -1 < q < 3.
#
# At q=1: [e_1^{-βx^2}]^{2-1} = e^{-βx^2}, recovering the standard
#         Mexican hat ψ(x) = A(1-2βx^2)e^{-βx^2} with β=1/2.
#
# Normalization constants A_q (equations 17-18) ensure ||ψ_q||_2 = 1.

def _q_mexican_hat(u, q_tsallis, beta=0.5):
    """Borges 2004 q-Mexican hat wavelet (equation 16).

    ψ_q(x) = A_q [1 - (3-q)β x²] [e_q^{-βx²}]^q

    where [e_q^{-βx²}]^q = [1 + (1-q)βx²]^{q/(1-q)}  for q != 1.

    β and ``fact`` relationship
    ---------------------------
    The width parameter β and the CWT registration parameter ``fact`` both
    control the wavelet's effective spatial width.  Changing β is equivalent
    to rescaling the coordinate: ψ_q(x; β) ∝ ψ_q(x√(β/β₀); β₀).  In the
    CWT framework, dilation is handled by the scale *a* and the fixed
    stretch ``fact``, so β is **redundant** with ``fact``.

    The recommended default is **β = 1/2**, which:
    - Matches the standard Mexican hat at q = 1 (i.e. −g2 with fact = 3.0)
    - Gives A_q formulas (eqs. 17–18) and FT formulas (eqs. 20–21) in
      their simplest form
    - Ensures all q-Mexican hats at the same scale *a* probe comparable
      spatial frequencies (width variation with q is absorbed by ``fact``)

    Parameters
    ----------
    u : array_like
        Evaluation points.
    q_tsallis : float
        Tsallis parameter, -1 < q < 3. q=1 gives standard Mexican hat.
    beta : float
        Width parameter (default 0.5).  Use 0.5 for WTMM; other values
        are equivalent to rescaling ``fact`` and change the normalization.

    Returns
    -------
    result : ndarray
        Wavelet values at u.
    """
    u = np.asarray(u, dtype=np.float64)
    x2 = u**2

    if abs(q_tsallis - 1.0) < 1e-12:
        # Standard Mexican hat: A(1 - 2βx²)e^{-βx²}
        # With β=1/2: (1 - x²)e^{-x²/2} = -(x² - 1)e^{-x²/2}
        # Note: standard g2 = (u²-1)e^{-u²/2}, Mexican hat = -g2
        # We use the Borges convention: A_q * [1 - (3-q)βx²] * exp(-βx²)
        # At q=1: A_1 * (1 - 2βx²) * exp(-βx²)
        # With β=1/2 and A_1 = 2/(π^{1/4}√3):
        A = 2.0 / (np.pi**0.25 * np.sqrt(3.0))
        return A * (1.0 - 2.0 * beta * x2) * np.exp(-beta * x2)

    if q_tsallis <= -1.0 or q_tsallis >= 3.0:
        raise ValueError(f"q_tsallis must be in (-1, 3), got {q_tsallis}")

    # Compute [e_q^{-βx²}]^q
    # e_q^{-βx²} = [1 + (1-q)(-βx²)]^{1/(1-q)} = [1 - (1-q)βx²]^{1/(1-q)}
    # [e_q^{-βx²}]^q = [1 - (1-q)βx²]^{q/(1-q)}
    #
    # For q > 1: 1-q < 0, so base = 1 + (q-1)βx² > 0 always, exponent = q/(1-q) < 0
    # For q < 1: 1-q > 0, so base = 1 - (1-q)βx², has compact support at |x| = 1/√((1-q)β)
    base = 1.0 - (1.0 - q_tsallis) * beta * x2   # = 1 + (q-1)*β*x² for q>1

    polynomial = 1.0 - (3.0 - q_tsallis) * beta * x2

    # Normalization constant A_q
    A_q = _borges_normalization(q_tsallis, beta)

    if q_tsallis > 1.0:
        # base = 1 + (q-1)βx² > 0 always; exponent q/(1-q) is negative
        eq_power_q = base ** (q_tsallis / (1.0 - q_tsallis))
        return A_q * polynomial * eq_power_q
    else:
        # Compact support: base > 0 only for |x| < 1/√((1-q)β)
        mask = base > 0
        # Use np.maximum to avoid 0**negative (inf/nan) — np.where evaluates both branches
        safe_base = np.maximum(base, 1e-300)
        eq_power_q = np.where(mask, safe_base ** (q_tsallis / (1.0 - q_tsallis)), 0.0)
        return A_q * polynomial * eq_power_q


def _borges_normalization(q_tsallis, beta=0.5):
    """Normalization constant A_q from Borges 2004, equations 17-18.

    Ensures ||ψ_q||_2 = 1.

    Parameters
    ----------
    q_tsallis : float
        Tsallis parameter, -1 < q < 3.
    beta : float
        Width parameter.

    Returns
    -------
    A_q : float
    """
    if abs(q_tsallis - 1.0) < 1e-12:
        return 2.0 / (np.pi**0.25 * np.sqrt(3.0))

    q = q_tsallis
    if q > 1.0 and q < 3.0:
        # Eq. 17: A_q = (β^{1/4} / (π^{1/4}√3)) *
        #               [(q-1)^{5/2} Γ(2q/(q-1))]^{1/2} / [Γ(2q/(q-1) - 5/2)]^{1/2}
        arg1 = 2.0 * q / (q - 1.0)
        A_q = (beta**0.25 / (np.pi**0.25 * np.sqrt(3.0))) * \
              np.sqrt((q - 1.0)**2.5 * gamma_func(arg1) / gamma_func(arg1 - 2.5))
        return A_q
    elif q > -1.0 and q < 1.0:
        # Eq. 18: A_q = (β^{1/4} / (π^{1/4}√3)) * ((5-q)^{1/2}(3+q)^{1/2} / 2) *
        #               [(1-q)^{1/2} Γ(2q/(1-q) + 3/2)]^{1/2} / [Γ(2q/(1-q) + 1)]^{1/2}
        arg2 = 2.0 * q / (1.0 - q)
        A_q = (beta**0.25 / (np.pi**0.25 * np.sqrt(3.0))) * \
              (np.sqrt(5.0 - q) * np.sqrt(3.0 + q) / 2.0) * \
              np.sqrt((1.0 - q)**0.5 * gamma_func(arg2 + 1.5) / gamma_func(arg2 + 1.0))
        return A_q
    else:
        raise ValueError(f"q_tsallis must be in (-1, 3), got {q_tsallis}")


def register_q_mexican_hat(q_tsallis, beta=0.5, name=None):
    """Register a Borges 2004 q-Mexican hat wavelet in WAVELETS dict.

    This is the proper q-generalization of the Mexican hat from:
    Borges, Tsallis, Miranda & Andrade (2004), J. Phys. A 37, 9125.

    ψ_q(x) = A_q [1 - (3-q)βx²] [e_q^{-βx²}]^q

    Always has 2 vanishing moments (like the standard Mexican hat / g2).

    Registration computes ``fact`` so that the wavelet's support maps to
    ~18 physical pixels at scale a = 1, consistent with g2 and other
    wavelets.  At q = 1 with β = 0.5 this gives fact = 3.0, matching g2
    exactly.  For other q values, ``fact = 18.0 / half_support`` adapts to
    the wavelet's natural support width.

    β = 0.5 is recommended for WTMM analysis — it is the standard
    convention and ensures cross-q comparability.  Changing β is redundant
    with ``fact`` (see ``_q_mexican_hat`` docstring for details).

    Parameters
    ----------
    q_tsallis : float
        Tsallis parameter, -1 < q < 3. q=1 gives standard Mexican hat.
    beta : float
        Width parameter (default 0.5).  Use 0.5 for WTMM consistency.
    name : str, optional
        Wavelet name. Default: f'qmhat_q{q_tsallis}'.

    Returns
    -------
    name : str, the registered wavelet name.
    """
    if name is None:
        name = f'qmhat_q{q_tsallis}'

    if abs(q_tsallis - 1.0) < 1e-12:
        half_support = 6.0
        fact = 3.0
    elif q_tsallis < 1.0:
        # Compact support at |x| = 1/√((1-q)β)
        u_max = 1.0 / np.sqrt((1.0 - q_tsallis) * beta)
        half_support = u_max + 0.1
        fact = 18.0 / half_support
    else:
        # Power-law tails: [1+(q-1)βx²]^{q/(1-q)} ~ x^{2q/(1-q)}
        # Truncate where envelope drops below eps
        eps = 1e-3
        # base^{q/(1-q)} = eps  =>  base = eps^{(1-q)/q}
        # 1 + (q-1)βx² = eps^{(1-q)/q}
        # x² = (eps^{(1-q)/q} - 1) / ((q-1)*β)
        base_val = eps ** ((1.0 - q_tsallis) / q_tsallis)
        if base_val > 1.0:
            u_trunc = np.sqrt((base_val - 1.0) / ((q_tsallis - 1.0) * beta))
        else:
            u_trunc = 20.0
        half_support = min(u_trunc, 20.0)
        fact = 18.0 / half_support

    def _make_func(qt, b):
        def func(u):
            return _q_mexican_hat(u, qt, b)
        return func

    WAVELETS[name] = {
        'fact': fact,
        'func': _make_func(q_tsallis, beta),
        'x_min_factor': -half_support,
        'x_max_factor':  half_support,
    }
    return name


def _q_mexican_hat_fourier(omega, q_tsallis, beta=0.5):
    """Analytical Fourier transform of the q-Mexican hat wavelet.

    Uses Borges 2004 eqs. 20-21, derived from Gradshteyn-Ryzhik tables.
    Convention: F[f; y] = (1/√(2π)) ∫ e^{ixy} f(x) dx  (eq. 19 in paper).

    For 1 < q < 3 (eq. 20): uses modified Bessel function K_ν.
    For -1 < q < 1 (eq. 21): uses Bessel function J_{-ν}.
    At q = 1: recovers the Gaussian Mexican hat FT exactly.

    Parameters
    ----------
    omega : array_like
        Angular frequency values.
    q_tsallis : float
        Tsallis parameter, -1 < q < 3.
    beta : float
        Width parameter.

    Returns
    -------
    psi_hat : ndarray
        Fourier transform values (real-valued for this symmetric wavelet).
    """
    from scipy.special import kv, jv

    omega = np.asarray(omega, dtype=np.float64)
    abs_y = np.abs(omega)

    if abs(q_tsallis - 1.0) < 1e-12:
        # Standard Mexican hat: ψ_1 ∝ d²/dx²[e^{-βx²}]
        # F[e^{-βx²}; y] = (1/√(2β)) exp(-y²/(4β))  (paper's 1/√(2π) convention)
        # F[ψ_1; y] = (-iy)² * A_1/(2β(q-2)) * F[e^{-βx²}; y]
        #           = A_1/(2β) * y² * (1/√(2β)) exp(-y²/(4β))
        A1 = _borges_normalization(1.0, beta)
        return A1 / (2.0 * beta) * omega**2 * np.exp(-omega**2 / (4.0 * beta)) / np.sqrt(2.0 * beta)

    q = q_tsallis
    A_q = _borges_normalization(q, beta)
    result = np.zeros_like(omega)
    mask = abs_y > 1e-30
    y_s = np.where(mask, abs_y, 1.0)

    if q > 1.0 and q < 3.0:
        # Eq. 20 (corrected: Γ in denominator, from G-R cosine transform formula)
        # F[ψ_q; y] = A_q / ((2-q)β √(2(q-1)β) Γ((2-q)/(q-1)))
        #           × y² [|y|/(2√((q-1)β))]^ν K_ν(|y|/√((q-1)β))
        # where ν = (2-q)/(q-1) - 1/2
        #
        # At q=2 (Cauchy-Lorentz): exact closed-form FT
        if abs(q - 2.0) < 1e-8:
            # F[ψ_2; y] = A_q √π/(√2 β) |y| e^{-|y|/√β}
            # Derived from known FT of 1/(1+βx²) and x²/(1+βx²)²
            return A_q * np.sqrt(np.pi) / (np.sqrt(2.0) * beta) * \
                   abs_y * np.exp(-abs_y / np.sqrt(beta))
        s = q - 1.0
        nu = (2.0 - q) / s - 0.5
        sqrtqb = np.sqrt(s * beta)

        prefactor = A_q / ((2.0 - q) * beta * np.sqrt(2.0 * s * beta) *
                           gamma_func((2.0 - q) / s))

        arg_K = y_s / sqrtqb
        arg_pow = y_s / (2.0 * sqrtqb)

        bessel_part = arg_pow**nu * kv(nu, arg_K)
        result = np.where(mask, prefactor * omega**2 * bessel_part, 0.0)

    elif q > -1.0 and q < 1.0:
        # Eq. 21 (corrected: Γ in numerator for q < 1 branch)
        # F[ψ_q; y] = A_q Γ((2-q)/(1-q)+1) / (2(2-q)β √(2(1-q)β))
        #           × y² [2√((1-q)β)/y]^{-ν} J_{-ν}(y/√((1-q)β))
        # where ν = (2-q)/(q-1) - 1/2, so -ν > 0 for q < 1
        s = 1.0 - q  # > 0
        nu = (2.0 - q) / (q - 1.0) - 0.5  # negative for q < 1
        neg_nu = -nu
        sqrtqb = np.sqrt(s * beta)

        prefactor = A_q * gamma_func((2.0 - q) / s + 1.0) / \
                    (2.0 * (2.0 - q) * beta * np.sqrt(2.0 * s * beta))

        arg_J = y_s / sqrtqb
        arg_pow = 2.0 * sqrtqb / y_s

        bessel_part = arg_pow**(-nu) * jv(neg_nu, arg_J)
        result = np.where(mask, prefactor * omega**2 * bessel_part, 0.0)

    return result


# ---------------------------------------------------------------------------
# Part B: Cascade-reconstructed wavelets (db2--db4, coif1--coif3)
# ---------------------------------------------------------------------------

def _cascade_reconstruct(h, n_iter=12):
    """Cascade algorithm to reconstruct wavelet psi(t) from low-pass filter h.

    Uses iterative upsample-and-filter: start with delta, then repeatedly
    upsample by 2 (insert zeros) and convolve with sqrt(2)*h.

    Returns (t, psi) arrays where t is centered so that the center of |psi|^2
    is at 0.
    """
    h = np.asarray(h, dtype=np.float64)
    L = len(h)
    sqrt2 = np.sqrt(2)

    # High-pass (wavelet) filter via QMF relation
    g = np.array([(-1)**n * h[L - 1 - n] for n in range(L)])

    # Build phi via cascade: start from delta, iterate upsample + filter
    phi = np.array([1.0])
    for _ in range(n_iter):
        # Upsample by 2 (insert zeros between samples)
        up = np.zeros(2 * len(phi) - 1)
        up[::2] = phi
        # Convolve with sqrt(2) * h
        phi = np.convolve(up, sqrt2 * h)

    # Build psi: upsample phi once more and convolve with sqrt(2) * g
    up = np.zeros(2 * len(phi) - 1)
    up[::2] = phi
    psi = np.convolve(up, sqrt2 * g)

    # Time axis: after n_iter refinements of phi, grid spacing = 2^{-n_iter}
    # psi has one extra level of refinement
    dt = 1.0 / (2 ** (n_iter + 1))
    t = np.arange(len(psi)) * dt

    # Center at midpoint of support [0, L-1] so that x_min/max = ±(L-1)/2
    # Phase doesn't matter for WTMM (we use modulus).
    center = (L - 1) / 2.0
    t = t - center

    return t, psi


def _lazy_cascade_func(h, n_iter=12):
    """Return a closure that lazily builds a cubic interpolator for the wavelet.

    The cascade reconstruction (~0.6s) runs only on first call; subsequent
    calls use the cached interpolator.
    """
    from scipy.interpolate import interp1d

    cache = {}

    def func(u):
        if 'interp' not in cache:
            t, psi = _cascade_reconstruct(h, n_iter)
            cache['interp'] = interp1d(t, psi, kind='cubic',
                                       bounds_error=False, fill_value=0.0)
            cache['t_min'] = t[0]
            cache['t_max'] = t[-1]
        return cache['interp'](u)

    return func


# --- Filter coefficients (from PyWavelets / Daubechies 1992) ---

# Daubechies wavelets (analysis low-pass, dec_lo convention)
_DB2_H = np.array([
    -1.29409522551260369738e-01,
     2.24143868042013388875e-01,
     8.36516303737807942476e-01,
     4.82962913144534156107e-01,
])

_DB3_H = np.array([
     3.52262918857095333469e-02,
    -8.54412738820266581818e-02,
    -1.35011020010254584323e-01,
     4.59877502118491543470e-01,
     8.06891509311092547385e-01,
     3.32670552950082631938e-01,
])

_DB4_H = np.array([
    -1.05974017850690317016e-02,
     3.28830116668851965556e-02,
     3.08413818355607639854e-02,
    -1.87034811719093085891e-01,
    -2.79837694168598542788e-02,
     6.30880767929858921050e-01,
     7.14846570552915672181e-01,
     2.30377813308896506328e-01,
])

# Coiflet wavelets
_COIF1_H = np.array([
    -1.56557281357919929332e-02,
    -7.27326195125264501895e-02,
     3.84864846864857779174e-01,
     8.52572020211600389850e-01,
     3.37897662457481817722e-01,
    -7.27326195125264501895e-02,
])

_COIF2_H = np.array([
    -7.20549445520346975788e-04,
    -1.82320887091103230049e-03,
     5.61143481936883428002e-03,
     2.36801719468477701869e-02,
    -5.94344186464310919593e-02,
    -7.64885990782807612121e-02,
     4.17005184423239083635e-01,
     8.12723635449413506215e-01,
     3.86110066822762887373e-01,
    -6.73725547237255945054e-02,
    -4.14649367868717769192e-02,
     1.63873364632036409849e-02,
])

_COIF3_H = np.array([
    -3.45997731972727805847e-05,
    -7.09833025063790037977e-05,
     4.66216959820402881715e-04,
     1.11751877083063029875e-03,
    -2.57451768813679723533e-03,
    -9.00797613673062422257e-03,
     1.58805448636694518383e-02,
     3.45550275732977377197e-02,
    -8.23019271062998269972e-02,
    -7.17998216191548382925e-02,
     4.28483476377369998378e-01,
     7.93777222626087186619e-01,
     4.05176902409118244730e-01,
    -6.11233900029725524261e-02,
    -6.57719112814693640523e-02,
     2.34526961420771680455e-02,
     7.78259642567274631531e-03,
    -3.79351286438080192998e-03,
])

# Register cascade wavelets
# fact = 18.0 / half_support  (half_support = (L-1) / 2)
_CASCADE_SPECS = [
    # (name, filter_coeffs, support_width, fact)
    ('db2',   _DB2_H,   3.0,  12.0),
    ('db3',   _DB3_H,   5.0,  7.2),
    ('db4',   _DB4_H,   7.0,  5.14),
    ('coif1', _COIF1_H, 5.0,  7.2),
    ('coif2', _COIF2_H, 11.0, 3.27),
    ('coif3', _COIF3_H, 17.0, 2.12),
]

for _name, _h, _sw, _fact in _CASCADE_SPECS:
    _half = _sw / 2.0
    WAVELETS[_name] = {
        'fact': _fact,
        'func': _lazy_cascade_func(_h),
        'x_min_factor': -_half,
        'x_max_factor':  _half,
    }


# ---------------------------------------------------------------------------
# Public API (unchanged)
# ---------------------------------------------------------------------------

def wavelet_direct(x, a, wavelet_name='g2'):
    """Evaluate wavelet in direct space: psi(x / (a * gNfact))
    Matches LastWave's d_r_dN_gauss functions exactly.
    """
    w = WAVELETS[wavelet_name]
    u = x / (a * w['fact'])
    return w['func'](u)


def wavelet_support(a, wavelet_name='g2'):
    """Return (d_x_min, d_x_max) for the wavelet at scale a.
    These are the physical-space support bounds.
    """
    w = WAVELETS[wavelet_name]
    return a * w['x_min_factor'] * w['fact'], a * w['x_max_factor'] * w['fact']
