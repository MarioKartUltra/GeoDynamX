"""Partition function computation and D(h) spectrum extraction."""

import numpy as np


def _get_pf_compute_one_scale():
    """Lazy-compile the numba-jitted inner loop on first use."""
    from numba import njit

    @njit(cache=True)
    def _pf_compute_one_scale(absT_sorted, tSize, imin, q, scale_idx,
                               sTq_out, sTqLogT_out, logSTq_out, sTqLogT_sTq_out,
                               log2STq_out, sTqLogT_sTq2_out, logSTqSTqLogT_sTq_out):
        if q >= 0:
            tm = absT_sorted[imin]
        else:
            tm = absT_sorted[tSize - 1]

        sTq_val = 0.0
        sTqLogT_val = 0.0

        if q >= 0:
            for i in range(imin, tSize):
                tq = absT_sorted[i] ** q
                logt = np.log(absT_sorted[i] / tm)
                sTq_val += tq
                sTqLogT_val += tq * logt
        else:
            for i in range(tSize - 1, imin - 1, -1):
                tq = absT_sorted[i] ** q
                logt = np.log(absT_sorted[i] / tm)
                sTq_val += tq
                sTqLogT_val += tq * logt

        sTqLogT_val = sTqLogT_val + np.log(tm) * sTq_val

        sTq_out[scale_idx] = sTq_val
        sTqLogT_out[scale_idx] = sTqLogT_val

        if sTq_val != 0:
            N_nonzero = tSize - imin
            logSTq_val = np.log(sTq_val / N_nonzero)
            sTqLogT_sTq_val = sTqLogT_val / sTq_val

            logSTq_out[scale_idx] = logSTq_val
            sTqLogT_sTq_out[scale_idx] = sTqLogT_sTq_val
            log2STq_out[scale_idx] = logSTq_val * logSTq_val
            sTqLogT_sTq2_out[scale_idx] = sTqLogT_sTq_val * sTqLogT_sTq_val
            logSTqSTqLogT_sTq_out[scale_idx] = logSTq_val * sTqLogT_sTq_val

    return _pf_compute_one_scale


_pf_jit = None


def pf_compute_one_scale(absT_sorted, tSize, imin, q, scale_idx,
                          sTq_out, sTqLogT_out, logSTq_out, sTqLogT_sTq_out,
                          log2STq_out, sTqLogT_sTq2_out, logSTqSTqLogT_sTq_out):
    """Compute all 7 partition function arrays for one scale and one q value.
    Matches LastWave's PFComputeOneScaleFLOAT exactly.
    Lazy-compiles numba JIT on first call.
    """
    global _pf_jit
    if _pf_jit is None:
        _pf_jit = _get_pf_compute_one_scale()
    _pf_jit(absT_sorted, tSize, imin, q, scale_idx,
            sTq_out, sTqLogT_out, logSTq_out, sTqLogT_sTq_out,
            log2STq_out, sTqLogT_sTq2_out, logSTqSTqLogT_sTq_out)


def compute_partition_function(extrep_ord, n_oct, n_voice, a_min, q_list,
                               coarser_links=None, min_chain_voices=None,
                               track_lines=False):
    """Compute partition functions matching LastWave's PFComputeOneScaleFLOAT.

    Parameters
    ----------
    coarser_links : list of arrays, optional
        Chain links (from chain_all). Required if min_chain_voices or
        track_lines is set.
    min_chain_voices : int, optional
        If set, only include extrema belonging to chains spanning at least
        this many voices. Requires coarser_links.
    track_lines : bool, default False
        If True, also compute per-line Boltzmann weights, H(q,a), and D(q,a)
        via ``perline.compute_perline_partition``.  Requires *coarser_links*.
        The extra keys are added to the returned dict alongside the standard
        7 aggregate arrays (which are unchanged).

    Returns dict with all 7 per-scale arrays plus metadata.
    When *track_lines* is True, the dict also contains the per-line keys
    from ``compute_perline_partition`` (prefixed with nothing — they are
    merged directly).
    """
    n_scales = n_oct * n_voice

    # Optional chain-length filtering
    if min_chain_voices is not None and coarser_links is not None:
        keep_mask = [np.zeros(len(extrep_ord[s]), dtype=np.bool_) for s in range(n_scales)]
        for fi in range(len(extrep_ord[0])):
            cs, ci = 0, fi
            chain_indices = []
            while cs < n_scales and ci != -1:
                chain_indices.append((cs, ci))
                ci = coarser_links[cs][ci]
                cs += 1
            if len(chain_indices) >= min_chain_voices:
                for s, i in chain_indices:
                    keep_mask[s][i] = True
        # Build filtered ordinates
        filtered_ord = []
        n_kept = 0
        for s in range(n_scales):
            filtered_ord.append(extrep_ord[s][keep_mask[s]])
            n_kept += keep_mask[s].sum()
        n_total = sum(len(extrep_ord[s]) for s in range(n_scales))
        print(f'Chain filter: keeping {n_kept}/{n_total} extrema '
              f'(chains >= {min_chain_voices} voices)')
        extrep_ord = filtered_ord

    q_array = np.sort(np.array(q_list, dtype=np.float64))
    # Snap near-zero values to exact 0.0 (matches LastWave's PFQDIFFMAX tolerance)
    q_array[np.abs(q_array) < 1e-10] = 0.0
    n_q = len(q_array)

    # All 7 arrays per q value
    sTq = np.zeros((n_q, n_scales))
    sTqLogT = np.zeros((n_q, n_scales))
    logSTq = np.zeros((n_q, n_scales))
    sTqLogT_sTq = np.zeros((n_q, n_scales))
    log2STq = np.zeros((n_q, n_scales))
    sTqLogT_sTq2 = np.zeros((n_q, n_scales))
    logSTqSTqLogT_sTq = np.zeros((n_q, n_scales))

    n_ext = np.zeros(n_scales, dtype=np.int64)
    index_max = -1

    for scale_idx in range(n_scales):
        tSize = len(extrep_ord[scale_idx])
        n_ext[scale_idx] = tSize

        if tSize == 0:
            break

        absT = np.abs(extrep_ord[scale_idx]).copy()
        absT.sort()

        imin = 0
        while imin < tSize and absT[imin] == 0:
            imin += 1

        if imin < tSize:
            for nq in range(n_q):
                pf_compute_one_scale(absT, tSize, imin, q_array[nq], scale_idx,
                                     sTq[nq], sTqLogT[nq], logSTq[nq], sTqLogT_sTq[nq],
                                     log2STq[nq], sTqLogT_sTq2[nq], logSTqSTqLogT_sTq[nq])

        index_max = scale_idx

    factor = 2.0 ** (1.0 / n_voice)
    scale_values = np.array([a_min * factor**i for i in range(n_scales)])
    log2_a = np.log2(scale_values)

    result = {
        # Raw arrays (natural log)
        'sTq': sTq,
        'sTqLogT': sTqLogT,
        'logSTq': logSTq,
        'sTqLogT_sTq': sTqLogT_sTq,
        'log2STq': log2STq,
        'sTqLogT_sTq2': sTqLogT_sTq2,
        'logSTqSTqLogT_sTq': logSTqSTqLogT_sTq,
        # Metadata
        'q_list': q_array, 'scales': scale_values, 'log2_a': log2_a,
        'n_ext': n_ext, 'index_max': index_max,
        'n_voice': n_voice, 'n_oct': n_oct, 'a_min': a_min,
        'signal_number': 1,
    }

    if track_lines:
        if coarser_links is None:
            raise ValueError("track_lines=True requires coarser_links")
        from .perline import compute_perline_partition
        plpf = compute_perline_partition(
            extrep_ord, coarser_links, n_oct, n_voice, a_min, q_list,
        )
        # Merge per-line keys into the result dict
        for k, v in plpf.items():
            if k not in result:
                result[k] = v

    return result


def pf_get_T(pf, q_idx, mode='extensive'):
    """Access T(q,a) = log partition function.
    Extensive: log2(sTq);  Intensive: logSTq / ln(2) / signalNumber.
    Returns array over scales in log base 2."""
    LN2 = np.log(2.0)
    if mode == 'extensive':
        vals = pf['sTq'][q_idx]
        result = np.full_like(vals, np.nan)
        mask = vals > 0
        result[mask] = np.log(vals[mask]) / LN2
        return result
    else:
        return pf['logSTq'][q_idx] / LN2 / pf['signal_number']


def pf_get_H(pf, q_idx, mode='extensive'):
    """Access H(q,a) = weighted average of log|T|.
    Extensive: sTqLogT_sTq / ln(2);  Intensive: same / signalNumber.
    Returns array over scales in log base 2."""
    LN2 = np.log(2.0)
    if mode == 'extensive':
        return pf['sTqLogT_sTq'][q_idx] / LN2
    else:
        return pf['sTqLogT_sTq'][q_idx] / LN2 / pf['signal_number']


def pf_get_D(pf, q_idx, q_val, mode='extensive'):
    """Access D(q,a) = q * H(q,a) - T(q,a).
    Returns array over scales in log base 2."""
    return q_val * pf_get_H(pf, q_idx, mode) - pf_get_T(pf, q_idx, mode)


def pf_standard_addition(pf_accum, pf_new):
    """Add pf_new into pf_accum in-place. Matches LastWave's PFStandardAddition.

    All 7 arrays are summed element-wise; signalNumber incremented.
    """
    for key in ['sTq', 'sTqLogT', 'logSTq', 'sTqLogT_sTq',
                'log2STq', 'sTqLogT_sTq2', 'logSTqSTqLogT_sTq']:
        pf_accum[key] += pf_new[key]

    pf_accum['signal_number'] += pf_new['signal_number']
    pf_accum['index_max'] = min(pf_accum['index_max'], pf_new['index_max'])


def pf_get_var_T(pf, q_idx):
    """Variance of T(q,a) across ensemble. Matches PFAccessVarTQFLOAT.

    var[T] = (<log²Z> - <logZ>²/N) / (N-1)

    Requires signal_number >= 2. Returns in log₂ units.
    """
    LN2_SQ = np.log(2.0) ** 2
    N = pf['signal_number']
    if N < 2:
        return np.full(pf['sTq'].shape[1], np.nan)
    n = float(N)
    # log2STq and logSTq are in natural log; convert variance to log2 units
    var_nat = (pf['log2STq'][q_idx] - pf['logSTq'][q_idx]**2 / n) / (n - 1.0)
    return var_nat / LN2_SQ


def pf_get_var_H(pf, q_idx):
    """Variance of H(q,a) across ensemble. Matches PFAccessVarHQFLOAT.

    var[H] = (<(sTqLogT/sTq)²> - <sTqLogT/sTq>²/N) / (N-1)

    Requires signal_number >= 2. Returns in log₂ units.
    """
    LN2_SQ = np.log(2.0) ** 2
    N = pf['signal_number']
    if N < 2:
        return np.full(pf['sTq'].shape[1], np.nan)
    n = float(N)
    var_nat = (pf['sTqLogT_sTq2'][q_idx] -
               pf['sTqLogT_sTq'][q_idx]**2 / n) / (n - 1.0)
    return var_nat / LN2_SQ


def pf_get_var_D(pf, q_idx, q_val):
    """Variance of D(q,a) across ensemble. Matches PFAccessVarDQFLOAT.

    var[D] = (q²·<(H)²> - 2q·<logZ·H> + <log²Z> - <D>²/N) / (N-1)

    where D = q·H - logZ per realization.
    Requires signal_number >= 2. Returns in log₂ units.
    """
    LN2_SQ = np.log(2.0) ** 2
    N = pf['signal_number']
    if N < 2:
        return np.full(pf['sTq'].shape[1], np.nan)
    n = float(N)
    q = q_val
    # Mean of D across ensemble: <D> = (q·<H> - <logZ>) / N
    mean_D = (q * pf['sTqLogT_sTq'][q_idx] - pf['logSTq'][q_idx]) / n
    var_nat = (q**2 * pf['sTqLogT_sTq2'][q_idx]
               - 2 * q * pf['logSTqSTqLogT_sTq'][q_idx]
               + pf['log2STq'][q_idx]
               - mean_D * n * mean_D) / (n - 1.0)
    return var_nat / LN2_SQ


def pf_copy(pf):
    """Deep copy a partition function dict."""
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in pf.items()}
