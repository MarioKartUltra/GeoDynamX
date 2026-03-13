"""Test signal generators for WTMM validation."""

import numpy as np


def ucantor(size, r, p, nFlip=0):
    """Generate non-uniform Cantor measure matching LastWave's UCantor().

    Parameters
    ----------
    size : int, signal length
    r : list of float, partition ratios (must sum to 1)
    p : list of float, probability weights
    nFlip : int, which partition to flip (0-based, -1 for none)

    Returns
    -------
    signal : numpy array of shape (size,)
    dx : float, spacing
    """
    dx = 1.0 / size
    signal = np.zeros(size)

    def _ucantor(left, right, proba, r, p, nFlip):
        if abs(right - left) <= dx:
            idx = int(left / dx)
            if idx >= size:
                idx = size - 1
            signal[idx] += proba
            return

        l = left
        for i in range(len(r)):
            if i == nFlip:
                _ucantor(l + (right - left) * r[i], l, proba * p[i], r, p, nFlip)
            else:
                _ucantor(l, l + (right - left) * r[i], proba * p[i], r, p, nFlip)
            l += (right - left) * r[i]

    _ucantor(0.0, 1.0, 1.0, r, p, nFlip)
    return signal, dx


def tau_equation(tau, q, r, p):
    """sum_i p_i^q * r_i^(-tau) - 1 = 0

    Used with scipy.optimize.brentq to solve for tau(q) analytically
    for a multinomial measure with parts (r_i, p_i).

    Parameters
    ----------
    tau : float, the unknown
    q : float, moment order
    r : array-like, partition ratios
    p : array-like, probability weights

    Returns
    -------
    float : residual (zero when tau is the solution)
    """
    r = np.asarray(r, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return np.sum(p**q * r**(-tau)) - 1.0
