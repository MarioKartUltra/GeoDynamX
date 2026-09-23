"""Wavelet-based statistical diagnostics: skewness, flatness, correlations.

These operate on the raw CWT coefficient matrix, not the WTMM skeleton.
"""

import numpy as np


def wavelet_skewness_flatness(cwt_matrix, causal_ranges=None):
    """Compute wavelet skewness and flatness (kurtosis) per scale.

    Skewness(a) = <T^3(t,a)> / <T^2(t,a)>^{3/2}
    Flatness(a) = <T^4(t,a)> / <T^2(t,a)>^2

    A Gaussian signal has skewness=0 and flatness=3 at all scales.
    Increasing flatness at small scales indicates intermittency.
    Non-zero skewness indicates asymmetric structures (e.g. ramp-cliffs).

    Parameters
    ----------
    cwt_matrix : 2D array, shape (n_scales, n_samples)
        Raw CWT coefficients (not absolute values).
    causal_ranges : list of (first, last) tuples, optional
        Per-scale valid sample ranges (from CWT border effects).
        Only valid samples are used in the moments.

    Returns
    -------
    dict with keys:
        'skewness' : 1D array (n_scales,)
        'flatness' : 1D array (n_scales,)
        'variance' : 1D array (n_scales,) — <T^2> per scale
    """
    n_scales, n_samples = cwt_matrix.shape
    skewness = np.full(n_scales, np.nan)
    flatness = np.full(n_scales, np.nan)
    variance = np.full(n_scales, np.nan)

    for i in range(n_scales):
        if causal_ranges is not None:
            fp, lp = causal_ranges[i]
            T = cwt_matrix[i, fp:lp + 1]
        else:
            T = cwt_matrix[i]

        if len(T) < 2:
            continue

        m2 = np.mean(T ** 2)
        if m2 == 0:
            continue

        variance[i] = m2
        m3 = np.mean(T ** 3)
        m4 = np.mean(T ** 4)

        skewness[i] = m3 / m2 ** 1.5
        flatness[i] = m4 / m2 ** 2

    return {
        'skewness': skewness,
        'flatness': flatness,
        'variance': variance,
    }


def magnitude_correlation_cross_scale(cwt_matrix, fine_scale_idx,
                                      causal_ranges=None):
    """Cross-scale correlation of wavelet modulus (Dupont et al. 2020 Fig. 12).

    Computes the Pearson correlation between |T(t, a_fine)| and |T(t, a)|
    for each scale a. Measures whether fine-scale fluctuations are driven
    by larger-scale structures.

    Parameters
    ----------
    cwt_matrix : 2D array, shape (n_scales, n_samples)
        Raw CWT coefficients.
    fine_scale_idx : int
        Index of the reference fine scale.
    causal_ranges : list of (first, last) tuples, optional
        Per-scale valid sample ranges.

    Returns
    -------
    corr : 1D array (n_scales,)
        Pearson correlation of |T(t, a_fine)| vs |T(t, a)| at each scale.
    """
    n_scales, n_samples = cwt_matrix.shape
    corr = np.full(n_scales, np.nan)

    abs_fine = np.abs(cwt_matrix[fine_scale_idx])

    for i in range(n_scales):
        abs_i = np.abs(cwt_matrix[i])

        # Determine valid range (intersection of both scales' causal ranges)
        if causal_ranges is not None:
            fp_f, lp_f = causal_ranges[fine_scale_idx]
            fp_i, lp_i = causal_ranges[i]
            fp = max(fp_f, fp_i)
            lp = min(lp_f, lp_i)
        else:
            fp, lp = 0, n_samples - 1

        if lp - fp < 2:
            continue

        x = abs_fine[fp:lp + 1]
        y = abs_i[fp:lp + 1]

        # Pearson correlation
        mx, my = np.mean(x), np.mean(y)
        dx, dy = x - mx, y - my
        denom = np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2))
        if denom > 0:
            corr[i] = np.sum(dx * dy) / denom

    return corr


# ---------------------------------------------------------------------------
# Arneodo et al. (1998) magnitude correlation pipeline
#
#   f[n] → W[j,n] → E=|W|² → S=boxavg(E, width∝a) → V=½ln(S+ε)
#        → Ṽ=V−⟨V⟩  → C[ℓ, j₁, j₂] = ⟨Ṽ[j₁,n]·Ṽ[j₂,n+ℓ]⟩
#
# The a⁻² factor from Eq. 4 is a constant per scale that cancels after
# centering, so we omit it.  The paper used a box window for χ.
# ---------------------------------------------------------------------------

def local_energy(cwt_matrix, scales, width_factor=1.0):
    """Local space-scale energy density S[j,n] (Arneodo Eq. 4, without a⁻²).

    S[j, n] = box_average( |W[j, ·]|², width = round(width_factor * a_j) )

    Parameters
    ----------
    cwt_matrix : 2D array (n_scales, n_samples)
    scales : 1D array (n_scales,)
    width_factor : float
        Proportionality between box width and scale.  1.0 means the
        averaging window equals the scale in samples.

    Returns
    -------
    S : 2D array (n_scales, n_samples)
    """
    from scipy.ndimage import uniform_filter1d

    n_scales, n_samples = cwt_matrix.shape
    S = np.empty_like(cwt_matrix, dtype=np.float64)
    for j, a in enumerate(scales):
        E = np.abs(cwt_matrix[j]) ** 2
        w = max(1, int(round(width_factor * a)))
        S[j] = uniform_filter1d(E, size=w, mode='reflect')
    return S


def magnitude_field(S, eps=1e-300):
    """Magnitude field V[j,n] = ½ ln S[j,n]  (Arneodo Eq. 5)."""
    return 0.5 * np.log(S + eps)


def center_by_scale(V):
    """Centered magnitude Ṽ[j,n] = V[j,n] − ⟨V[j,·]⟩."""
    return V - np.mean(V, axis=1, keepdims=True)


def magnitude_correlation(V_centered, idx1, idx2=None, max_lag=None):
    """Two-point magnitude correlation C(Δx, a₁, a₂) (Arneodo Eq. 6).

    C[ℓ] = ⟨Ṽ[j₁, n] · Ṽ[j₂, n+ℓ]⟩_n

    Parameters
    ----------
    V_centered : 2D array (n_scales, n_samples)
        Output of ``center_by_scale(magnitude_field(local_energy(...)))``.
    idx1 : int
        First scale index j₁.
    idx2 : int, optional
        Second scale index j₂.  None → same as idx1 (one-scale).
    max_lag : int, optional
        Maximum lag in samples.  Default: n_samples // 4.

    Returns
    -------
    dict with keys:
        'lags'  : 1D int array
        'C'     : 1D float array — raw covariance (plot this)
        'C_norm': 1D float array — C(Δx)/C(0) for convenience
    """
    if idx2 is None:
        idx2 = idx1

    v1 = V_centered[idx1]
    v2 = V_centered[idx2]
    n = len(v1)

    if max_lag is None:
        max_lag = n // 4

    lags = np.arange(0, max_lag + 1)
    C = np.empty(len(lags))

    for k, lag in enumerate(lags):
        if lag >= n:
            C[k] = np.nan
        else:
            C[k] = np.mean(v1[:n - lag] * v2[lag:n])

    C_norm = C / C[0] if C[0] != 0 and np.isfinite(C[0]) else C.copy()

    return {'lags': lags, 'C': C, 'C_norm': C_norm}


def space_scale_correlation(cwt_matrix, scales_or_idx, idx1=None,
                            idx2=None, max_lag=None, causal_ranges=None,
                            wavelet='g2', method=None, width_factor=1.0):
    """Convenience wrapper: magnitude correlation from raw CWT coefficients.

    Runs the full Arneodo pipeline (local_energy → magnitude_field →
    center_by_scale → magnitude_correlation) or the simpler Venugopal
    variant (ln|T| directly).

    Backward-compatible: old calls ``(cwt_matrix, scale_idx, ...)`` still
    work (detected when second arg is a scalar).

    Parameters
    ----------
    cwt_matrix : 2D array (n_scales, n_samples)
    scales_or_idx : 1D array of scales, or int (old-style scale_idx_1)
    idx1, idx2 : int — scale indices (a₁, a₂)
    max_lag : int
    causal_ranges : list of (first, last) tuples
    wavelet : str — unused (kept for API compat; box window is wavelet-agnostic)
    method : 'arneodo' | 'venugopal' | None (auto)
    width_factor : float — box width = width_factor * a  (default 1.0)

    Returns
    -------
    dict with 'lags', 'C', 'C_norm'
    """
    # --- backward compat -------------------------------------------------
    scales_or_idx = np.asarray(scales_or_idx)
    if scales_or_idx.ndim == 0:
        # Old call: (cwt_matrix, scale_idx_1, scale_idx_2, ...)
        scales = None
        _old_idx2 = idx1
        idx1 = int(scales_or_idx)
        idx2 = _old_idx2
    else:
        scales = scales_or_idx

    if idx2 is None:
        idx2 = idx1

    if method is None:
        method = 'arneodo' if scales is not None else 'venugopal'

    # --- valid sample range ----------------------------------------------
    n_scales, n_samples = cwt_matrix.shape
    if causal_ranges is not None:
        fp = max(causal_ranges[idx1][0], causal_ranges[idx2][0])
        lp = min(causal_ranges[idx1][1], causal_ranges[idx2][1])
    else:
        fp, lp = 0, n_samples - 1

    sub = cwt_matrix[:, fp:lp + 1]
    eps = 1e-300

    if method == 'arneodo':
        if scales is None:
            raise ValueError("method='arneodo' requires scales array")
        S = local_energy(sub, scales, width_factor=width_factor)
        V = magnitude_field(S, eps=eps)
    else:
        # Venugopal: V = ln|T|  (no box smoothing, no factor ½)
        V = np.log(np.abs(sub) + eps)

    Vt = center_by_scale(V)
    return magnitude_correlation(Vt, idx1, idx2, max_lag=max_lag)
