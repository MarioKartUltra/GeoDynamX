"""Dupont polynomial vs Venugopal cumulant fits of the partition function.

Two complementary methods for extracting low-order cumulants (c1, c2, c3)
from a multifractal partition function:

  Dupont polynomial fit
  ---------------------
  Fit a polynomial p(q) = -1 + c1*q + c2*q^2/2 + c3*q^3/6 + ... directly
  to the measured tau(q) curve.  Robust when tau(q) is well-sampled and
  the cumulant truncation is appropriate.  See Dupont et al.

  Venugopal cumulant fit
  ----------------------
  Fit each cumulant of ln|W(a)| against ln(a) directly:
        kappa_n[ln|W| | a]  ~  C_n * ln(a)  + const
  Slopes give c_n directly.  Independent of any tau-q fit.  See
  Venugopal et al. 2005.

  When both methods AGREE on (c1, c2, c3), the cascade is well-described
  by a low-order cumulant truncation (e.g. log-normal).  When they
  DISAGREE, the cascade is non-Gaussian and higher cumulants matter
  (or the chosen scale plateau is wrong).

Public API
----------
fit_dupont_polynomial(tau, q, *, scale_range=None, deg=3)
                                                  -> dict[c1, c2, c3, ...]
fit_venugopal_cumulants(holders_per_scale, log_scales,
                        *, scale_range=None, max_order=3)
                                                  -> dict[c1, c2, c3, ...]
plateau_r2(log_l, log_Z, q_grid, scale_range)
                                                  -> ndarray of per-q R^2
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _restrict_range(arr: np.ndarray, vals: np.ndarray,
                    rng: Optional[tuple[float, float]]):
    """Return (arr, vals) restricted to vals within rng (inclusive)."""
    if rng is None:
        return arr, vals
    lo, hi = float(rng[0]), float(rng[1])
    if lo > hi:
        lo, hi = hi, lo
    mask = (vals >= lo) & (vals <= hi)
    return arr[mask], vals[mask]


def fit_dupont_polynomial(
    tau: np.ndarray,
    q: np.ndarray,
    *,
    deg: int = 3,
) -> dict:
    """Fit tau(q) = c0 + c1*q + c2*q^2/2 + c3*q^3/6 + c4*q^4/24

    Uses ordinary least-squares polynomial regression on (q, tau).  The
    constant term `c0` is NOT forced -- it equals -D(0) where D(0) is
    the box-counting / capacity dimension of the support (e.g. -1 for a
    1D measure, -2 for a 2D measure).  Returned as the `c0` key.

    Parameters
    ----------
    tau : (n_q,) float    measured mass exponents
    q : (n_q,) float      generalized moments
    deg : int             max cumulant order (1 to 4).

    Returns
    -------
    dict with keys:
        'c0'             constant offset (= -D(0))
        'c1'..'c{deg}'   estimated cumulants
        'tau_fit'        polynomial fit values
        'residuals'      tau - tau_fit
        'r2'             goodness-of-fit
    """
    if deg < 1 or deg > 4:
        raise ValueError(f"deg must be 1..4; got {deg}")
    q = np.asarray(q, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    if q.shape != tau.shape:
        raise ValueError(f"q {q.shape} and tau {tau.shape} must match")

    # Design matrix: [1, q, q^2/2, q^3/6, q^4/24]
    factorial = [1, 1, 2, 6, 24]
    X = np.stack([np.ones_like(q)]
                  + [q ** k / factorial[k] for k in range(1, deg + 1)],
                  axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(X, tau, rcond=None)
    out = {'c0': float(coeffs[0])}
    for k in range(deg):
        out[f'c{k+1}'] = float(coeffs[k + 1])

    tau_fit = X @ coeffs
    out['tau_fit'] = tau_fit
    out['residuals'] = tau - tau_fit
    ss_res = float(np.sum(out['residuals'] ** 2))
    ss_tot = float(np.sum((tau - np.mean(tau)) ** 2))
    out['r2'] = 1.0 - ss_res / max(ss_tot, 1e-30)
    return out


def fit_venugopal_cumulants(
    holders_per_scale: dict | np.ndarray,
    log_scales: np.ndarray,
    *,
    scale_range: Optional[tuple[float, float]] = None,
    max_order: int = 3,
) -> dict:
    """Fit cumulants of ln|W(a)| vs ln(a) directly.

    Per Venugopal et al. 2005, when ln|W| is well-described by an
    n-th-order cumulant expansion in log scale, each cumulant scales:
        kappa_n[ln|W| | a]  ~  C_n * ln(a)  +  const
    The slope `C_n` (with the appropriate sign and factorial normalization)
    gives the cumulant `c_n` of the underlying multiplicative cascade.

    Parameters
    ----------
    holders_per_scale : dict[scale_idx -> (M,) array] OR (n_scales, M) array
        Per-scale ln|W(a)| samples (e.g., per-chain modulus at each scale,
        or per-pixel CWT modulus.)  A dict maps scale index to a variable-
        length array; an ndarray expects (n_scales, M_const) layout.
    log_scales : (n_scales,) float    ln(scale) per index.
    scale_range : tuple of (log_a_min, log_a_max), optional
        Restrict the regression to this scale window.
    max_order : int    1..4

    Returns
    -------
    dict with keys:
        'c1'..'c{max_order}'   estimated cumulants
        'cum_per_scale'        (max_order, n_scales) array of fitted cumulants
        'r2'                   float (worst per-cumulant R^2; close to 1 = good plateau)
    """
    if max_order < 1 or max_order > 4:
        raise ValueError(f"max_order must be 1..4; got {max_order}")

    # Build per-scale moments (raw moments first; convert to cumulants below)
    log_scales = np.asarray(log_scales, dtype=np.float64)
    n_scales = len(log_scales)
    moments = np.zeros((max_order, n_scales), dtype=np.float64)
    for si in range(n_scales):
        if isinstance(holders_per_scale, dict):
            samples = holders_per_scale.get(si)
        else:
            samples = np.asarray(holders_per_scale)[si]
        if samples is None or len(samples) == 0:
            moments[:, si] = np.nan
            continue
        x = np.asarray(samples, dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            moments[:, si] = np.nan
            continue
        moments[0, si] = np.mean(x)
        if max_order >= 2: moments[1, si] = np.var(x)
        if max_order >= 3:
            d = x - moments[0, si]
            moments[2, si] = np.mean(d ** 3)
        if max_order >= 4:
            d = x - moments[0, si]
            m4 = np.mean(d ** 4)
            moments[3, si] = m4 - 3 * moments[1, si] ** 2

    # Restrict to plateau
    valid = np.all(np.isfinite(moments), axis=0)
    if scale_range is not None:
        lo, hi = float(scale_range[0]), float(scale_range[1])
        if lo > hi: lo, hi = hi, lo
        valid &= (log_scales >= lo) & (log_scales <= hi)
    if valid.sum() < 2:
        out = {f'c{k+1}': float('nan') for k in range(max_order)}
        out['cum_per_scale'] = moments
        out['r2'] = float('nan')
        return out

    log_a_in = log_scales[valid]
    out = {}
    r2_min = 1.0
    for k in range(max_order):
        y = moments[k, valid]
        slope, intercept = np.polyfit(log_a_in, y, 1)
        # The "cumulant" is just the slope.  Sign convention:
        # ln|W| ~ k*log(a) + const, so kappa_n[ln|W|] ~ c_n * log(a) + ...
        # i.e. slope IS the cumulant directly.
        out[f'c{k+1}'] = float(slope)
        # R^2
        y_pred = slope * log_a_in + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        r2_min = min(r2_min, r2)
    out['cum_per_scale'] = moments
    out['r2'] = float(r2_min)
    return out


def plateau_r2(
    log_l: np.ndarray,
    log_Z: np.ndarray,
    q_grid: np.ndarray,
    scale_range: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """Per-q R^2 of a linear fit of log Z(q, L) vs log L within the plateau.

    R^2 close to 1 -> a clean log-log plateau; the chosen scale range is
    a good fit.  R^2 below ~0.95 flags a noisy or curved plateau where
    the tau(q) extraction is unreliable.

    Parameters
    ----------
    log_l : (n_scales,)
    log_Z : (n_q, n_scales)
    q_grid : (n_q,)
    scale_range : tuple of (lo, hi) on log_l, optional.

    Returns
    -------
    r2 : (n_q,)
    """
    log_l = np.asarray(log_l, dtype=np.float64)
    log_Z = np.asarray(log_Z, dtype=np.float64)
    valid = np.ones_like(log_l, dtype=bool)
    if scale_range is not None:
        lo, hi = float(scale_range[0]), float(scale_range[1])
        if lo > hi: lo, hi = hi, lo
        valid = (log_l >= lo) & (log_l <= hi)
    if valid.sum() < 2:
        return np.full(len(q_grid), np.nan, dtype=np.float64)
    x_in = log_l[valid]
    out = np.zeros(len(q_grid), dtype=np.float64)
    for qi in range(len(q_grid)):
        y = log_Z[qi, valid]
        slope, intercept = np.polyfit(x_in, y, 1)
        y_pred = slope * x_in + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        out[qi] = 1.0 - ss_res / max(ss_tot, 1e-30)
    return out
