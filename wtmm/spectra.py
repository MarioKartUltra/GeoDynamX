"""Multifractal spectrum computation and theoretical references."""

import numpy as np

from .partition import pf_get_T, pf_get_H, pf_get_D


def compute_spectra(pf, log2_a_min, log2_a_max, mode='extensive',
                    normalize=False, method='canonical'):
    """Compute tau(q), h(q), D(h) spectra via linear regression of slopes.

    Parameters
    ----------
    pf : dict from compute_partition_function
    log2_a_min, log2_a_max : float, scale range for regression (in log2)
    mode : str, 'extensive' (single signal) or 'intensive' (ensemble)
    normalize : bool, if True normalize Z(q,a) by a^{tau(1)} before
        computing D(q,a) slopes, following Arneodo 1995 Fig. 6.
        Two-pass: first computes tau(1) from T(1,a) slope, then
        detrends T(q,a) so that D(q,a) = q*H(q,a) - T_detrended(q,a).
    method : str
        'canonical' — h(q) and D(q) from Boltzmann-weighted thermal averages
            (slopes of H(q,a) and D(q,a)), tau(q) = q*h(q) - D(q).
            Preserves non-convex D(h) and phase transition structure.
        'legendre' — tau(q) from slopes of log2 Z(q,a), then
            h(q) = d tau/dq and D(h) = q*h - tau via Legendre transform.
            Gives convex hull of D(h), smooths over phase transitions.

    Returns
    -------
    spectra : dict with 'tau_q', 'h_q', 'D_q', 'q_list', 'tau_1', 'method'
    """
    log2_a = pf['log2_a']
    n_voice = pf['n_voice']
    a_min = pf['a_min']

    log2_a0 = np.log2(a_min)
    dx = 1.0 / n_voice
    idx_min = int((log2_a_min - log2_a0) / dx)
    idx_max = int((log2_a_max - log2_a0) / dx)

    idx_min = max(idx_min, 0)
    idx_max = min(idx_max, pf['index_max'])

    valid_idx = np.arange(idx_min, idx_max + 1)

    if len(valid_idx) < 2:
        raise ValueError('Not enough scales in range for regression')

    x = log2_a0 + dx * valid_idx
    q_array = pf['q_list']
    n_q = len(q_array)

    # --- First pass: compute tau(1) for normalization ---
    tau_1 = np.nan
    if normalize:
        q1_idx = np.where(np.isclose(q_array, 1.0))[0]
        if len(q1_idx) > 0:
            y_T1 = pf_get_T(pf, q1_idx[0], mode)[valid_idx]
            if np.all(np.isfinite(y_T1)):
                tau_1, _ = np.polyfit(x, y_T1, 1)
        if not np.isfinite(tau_1):
            raise ValueError('normalize=True but cannot compute tau(1) — '
                             'ensure q=1 is in q_list')

    # Detrend T(q,a) by tau(1)*log2(a) when normalizing
    # This is equivalent to Z_norm(q,a) = Z(q,a) / a^{tau(1)}
    tau_1_trend = tau_1 if (normalize and np.isfinite(tau_1)) else 0.0

    if method == 'canonical':
        h_q = np.full(n_q, np.nan)
        d_q = np.full(n_q, np.nan)

        for i in range(n_q):
            q = q_array[i]

            # H(q,a) — Boltzmann weights unaffected by normalization
            y_H = pf_get_H(pf, i, mode)[valid_idx]
            if np.all(np.isfinite(y_H)):
                slope_H, _ = np.polyfit(x, y_H, 1)
                h_q[i] = slope_H

            # D(q,a) = q*H(q,a) - T(q,a) + tau(1)*log2(a)
            y_D = pf_get_D(pf, i, q, mode)[valid_idx] + tau_1_trend * x
            if np.all(np.isfinite(y_D)):
                slope_D, _ = np.polyfit(x, y_D, 1)
                d_q[i] = slope_D

        tau_q = q_array * h_q - d_q

    elif method == 'legendre':
        tau_q = np.full(n_q, np.nan)

        for i in range(n_q):
            # T(q,a) detrended by tau(1)*log2(a)
            y_T = pf_get_T(pf, i, mode)[valid_idx] - tau_1_trend * x
            if np.all(np.isfinite(y_T)):
                slope_T, _ = np.polyfit(x, y_T, 1)
                tau_q[i] = slope_T

        h_q = np.gradient(tau_q, q_array)
        d_q = q_array * h_q - tau_q

    else:
        raise ValueError(f"method must be 'canonical' or 'legendre', got {method!r}")

    # Compute tau(1) from final spectra if not already set (normalize=False)
    if not normalize:
        q1_idx = np.where(np.isclose(q_array, 1.0))[0]
        if len(q1_idx) > 0:
            tau_1 = tau_q[q1_idx[0]]
        else:
            finite = np.isfinite(tau_q)
            if finite.sum() >= 2:
                tau_1 = np.interp(1.0, q_array[finite], tau_q[finite])

    return {
        'tau_q': tau_q,
        'h_q': h_q,
        'D_q': d_q,
        'q_list': q_array,
        'tau_1': tau_1,
        'method': method,
    }


def theoretical_devil_staircase(q_array, r, p, expo=-1.0):
    """Theoretical tau(q) and D(h) for WTMM on the devil's staircase.

    For a self-similar measure mu with weights p_i and contraction ratios r_i:
      tau_measure(q) solves: sum_i p_i^q * r_i^(-tau(q)) = 1

    The devil's staircase F = prim(mu) has h_F = h_mu + 1.
    The CWT with expo normalization gives: |coeff| = a^expo * |W[F]| ~ a^(expo + h_F)
    So the effective scaling exponent is: h_eff = h_mu + 1 + expo

    With expo=-1 (LastWave default):
      h_eff = h_mu + 1 + (-1) = h_mu
      tau_WTMM(q) = tau_measure(q)  (the integration and 1/a cancel out)

    With expo=0:
      h_eff = h_mu + 1
      tau_WTMM(q) = tau_measure(q) + q
    """
    r = np.array(r)
    p = np.array(p)

    from scipy.optimize import brentq

    tau_measure = np.zeros_like(q_array, dtype=float)
    for j, q in enumerate(q_array):
        def f(tau):
            return np.sum(p**q * r**(-tau)) - 1.0
        tau_measure[j] = brentq(f, -20, 20)

    # Account for integration (+1) and expo normalization
    # h_eff = h_measure + 1 + expo, so tau_eff = tau_measure + (1 + expo) * q
    tau_wtmm = tau_measure + (1.0 + expo) * q_array

    # D(h) via Legendre transform
    h_theory = np.gradient(tau_wtmm, q_array)
    D_theory = q_array * h_theory - tau_wtmm

    return tau_wtmm, h_theory, D_theory
