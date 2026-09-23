"""Per-line partition functions and Boltzmann weight classification.

Computes the contribution of each individual maxima line to the partition
function Z(q, a), enabling identification of phase transitions and separation
of interwoven singularity sets (L_P vs L_N in Bacry et al. 1993 notation).

Theory
------
For a signal f(x) = s(x) + r(x) with singular part s and C^∞ part r:

  - Lines in L_P (perturbed singularity lines): |T_ψ| ~ a^h  (h < n_ψ)
  - Lines in L_N (smooth-part lines):          |T_ψ| ~ a^(n_ψ)

The partition function splits as Z_f = Z_s + Z_r, producing a phase
transition at q_crit where τ(q) = min{τ_s(q), q·n_ψ}.

The canonical method (Arneodo et al. 1995 Eqs. 28–30) avoids the
Legendre smoothing by computing:

  h(q) = Σ_ℓ w̃(q,ℓ) · ln|E_ℓ| / ln(a)     (Boltzmann-weighted energy)
  D(q) = Σ_ℓ w̃(q,ℓ) · ln w̃(q,ℓ) / ln(a)  (Boltzmann entropy)

where w̃(q,ℓ) = E_ℓ^q / Z(q,a) is the Boltzmann weight of line ℓ.

References
----------
- Bacry, Muzy & Arnéodo, J. Stat. Phys. 70, 635–674 (1993), §4.3
- Arnéodo, Bacry & Muzy, Physica A 213, 232–275 (1995), §3.4
- Muzy, Bacry & Arnéodo, Int. J. Bif. Chaos 4, 245–302 (1994), §6
"""

import numpy as np


def _build_line_energies(extrep_ord, coarser_links, n_scales):
    """Extract per-line supremum amplitudes at each scale.

    For each line ℓ starting at finest-scale index fi, computes:
        E_ℓ(s) = sup_{s' ≤ s} |T_ψ(x(s'), s')|

    This is the scale-adaptive modulus maximum along the line up to
    scale s, matching the chain_max logic but returning the full
    per-line, per-scale matrix.

    Parameters
    ----------
    extrep_ord : list of arrays
        Ordinate (wavelet coefficient) values per scale.
    coarser_links : list of int arrays
        Chain links from finer to coarser scales.
    n_scales : int
        Total number of scales.

    Returns
    -------
    line_energies : ndarray, shape (n_lines, n_scales)
        E_ℓ(s) = running supremum of |ordinate| along line ℓ at scale s.
        NaN where the line has terminated.
    line_finest_idx : ndarray, shape (n_lines,)
        Index into the finest scale for each line (line identity).
    line_lengths : ndarray, shape (n_lines,)
        Number of scales each line spans.
    """
    n_finest = len(extrep_ord[0])
    line_energies = np.full((n_finest, n_scales), np.nan)
    line_lengths = np.zeros(n_finest, dtype=np.int64)
    line_finest_idx = np.arange(n_finest, dtype=np.int64)

    for fi in range(n_finest):
        cs = 0
        ci = fi
        running_max = 0.0
        while cs < n_scales and ci != -1:
            val = abs(extrep_ord[cs][ci])
            running_max = max(running_max, val)
            line_energies[fi, cs] = running_max
            ci = coarser_links[cs][ci]
            cs += 1
        line_lengths[fi] = cs

    return line_energies, line_finest_idx, line_lengths


def compute_perline_partition(extrep_ord, coarser_links,
                              n_oct, n_voice, a_min, q_list,
                              min_line_voices=2):
    """Compute partition functions with per-line Boltzmann weights.

    Unlike compute_partition_function (which pools all extrema per scale),
    this function preserves the identity of each maxima line throughout
    the computation.

    Parameters
    ----------
    extrep_ord : list of arrays
        Ordinate values per scale (after chain_delete and optionally
        chain_max).
    coarser_links : list of int arrays
        Chain links from chain_all.
    n_oct, n_voice : int
        Octaves and voices per octave.
    a_min : float
        Minimum scale.
    q_list : array-like
        q values for moment computation.
    min_line_voices : int
        Minimum number of voices a line must span to be included.

    Returns
    -------
    result : dict with keys:
        'line_energies'  : (n_lines, n_scales) running sup|T| per line
        'line_lengths'   : (n_lines,) number of scales each line spans
        'line_finest_idx': (n_lines,) finest-scale index (line ID)
        'Z_q'            : (n_q, n_scales) total partition function
        'Z_line_q'       : (n_q, n_lines, n_scales) per-line |E|^q
        'w_line_q'       : (n_q, n_lines, n_scales) Boltzmann weights
        'h_line_q'       : (n_q, n_lines, n_scales) per-line h contrib
        'D_line_q'       : (n_q, n_lines, n_scales) per-line D contrib
        'H_q'            : (n_q, n_scales) Boltzmann-weighted <ln|E|>
        'D_q'            : (n_q, n_scales) Boltzmann entropy
        'q_list'         : sorted q array
        'scales'         : scale values
        'log2_a'         : log2 of scales
        'n_voice'        : voices per octave

    Notes
    -----
    H_q and D_q are stored in **natural log** units (base e).  The standard
    partition function dict from ``compute_partition_function`` also uses
    natural log internally.  Use ``perline_to_pf`` to convert to a standard
    PF dict (which the spectra/plot functions expect in log₂).
    """
    n_scales = n_oct * n_voice
    q_array = np.sort(np.asarray(q_list, dtype=np.float64))
    q_array[np.abs(q_array) < 1e-10] = 0.0
    n_q = len(q_array)

    # Build per-line energy matrix
    line_energies, line_finest_idx, line_lengths = \
        _build_line_energies(extrep_ord, coarser_links, n_scales)

    # Filter by minimum line length
    keep = line_lengths >= min_line_voices
    line_energies = line_energies[keep]
    line_finest_idx = line_finest_idx[keep]
    line_lengths = line_lengths[keep]
    n_lines = len(line_lengths)

    if n_lines == 0:
        raise ValueError(f"No lines with >= {min_line_voices} voices")

    # Scale array
    factor = 2.0 ** (1.0 / n_voice)
    scales = np.array([a_min * factor**i for i in range(n_scales)])
    log2_a = np.log2(scales)

    # Compute per-line |E|^q, total Z(q,a), and Boltzmann weights
    # Z_line_q[qi, li, si] = E_ℓ(s)^q  for line li at scale si
    Z_line_q = np.full((n_q, n_lines, n_scales), 0.0)
    Z_q = np.zeros((n_q, n_scales))
    w_line_q = np.full((n_q, n_lines, n_scales), np.nan)

    # Per-line contributions to H(q,a) and D(q,a)
    h_line_q = np.full((n_q, n_lines, n_scales), np.nan)
    D_line_q = np.full((n_q, n_lines, n_scales), np.nan)

    # Aggregate Boltzmann-weighted H and D
    H_q = np.full((n_q, n_scales), np.nan)
    D_q_agg = np.full((n_q, n_scales), np.nan)

    for qi, q in enumerate(q_array):
        for si in range(n_scales):
            # Gather energies of all lines alive at this scale
            energies = line_energies[:, si]  # (n_lines,)
            alive = np.isfinite(energies) & (energies > 0)

            if not np.any(alive):
                continue

            E = energies[alive]

            if q == 0.0:
                Eq = np.ones_like(E)
            else:
                Eq = E ** q

            Z_total = np.sum(Eq)
            if Z_total == 0:
                continue

            # Store per-line |E|^q
            Z_line_q[qi, alive, si] = Eq
            Z_q[qi, si] = Z_total

            # Boltzmann weights w = E^q / Z
            w = Eq / Z_total
            w_line_q[qi, alive, si] = w

            # Per-line h contribution: w * ln(E)
            ln_E = np.log(E)
            h_contrib = w * ln_E
            h_line_q[qi, alive, si] = h_contrib

            # Per-line D contribution: w * ln(w)
            # Handle w=0 carefully (0 * ln(0) = 0 by convention)
            with np.errstate(divide='ignore', invalid='ignore'):
                ln_w = np.where(w > 0, np.log(w), 0.0)
            D_contrib = w * ln_w
            D_line_q[qi, alive, si] = D_contrib

            # Aggregates
            H_q[qi, si] = np.sum(h_contrib)
            D_q_agg[qi, si] = np.sum(D_contrib)

    return {
        'line_energies': line_energies,
        'line_lengths': line_lengths,
        'line_finest_idx': line_finest_idx,
        'n_lines': n_lines,
        'Z_q': Z_q,
        'Z_line_q': Z_line_q,
        'w_line_q': w_line_q,
        'h_line_q': h_line_q,
        'D_line_q': D_line_q,
        'H_q': H_q,           # Σ w·ln(E)  — slope gives h(q)
        'D_q': D_q_agg,       # Σ w·ln(w)  — slope gives D(q)
        'q_list': q_array,
        'scales': scales,
        'log2_a': log2_a,
        'n_voice': n_voice,
    }


def classify_lines(plpf, q_classify=None, weight_threshold=0.01,
                    q_idx=None, method='multi_q',
                    log2_a_min=None, log2_a_max=None):
    """Classify lines into singularity-supporting (L_P) vs smooth (L_N).

    Three methods are available, in order of robustness:

    method='multi_q' (default, recommended)
        Uses the *consistency of Boltzmann weight across multiple q values*
        above q_crit.  For each line, computes its mean weight at each q in
        the range [q_classify, q_max].  An L_P line's weight stays large
        (or grows) as q increases because its energy scales slower than
        n_ψ.  An L_N line's weight decays toward zero as q grows.

        The discriminant is the *integral of log-weight over q* in the
        high-q regime.  This is thermodynamically natural: it measures
        the total free-energy contribution of a line across the
        "ordered phase" (q > q_crit), rather than its weight at any
        single temperature.  It is robust because:
          - It averages over many q values, not just one.
          - It doesn't require measuring per-line slopes (h_ℓ).
          - It is independent of scale-range choice (the weights
            already encode the scaling information).

    method='coarse_scale'
        Uses the Boltzmann weight at the coarsest scale where each line
        is alive, at a single large q.  The coarsest scale gives the
        best scaling discrimination because E_LP ~ a^h vs E_LN ~ a^(n_ψ)
        diverge most at large a.  Simpler but less robust than multi_q.

    method='mean_weight' (legacy)
        Uses the mean Boltzmann weight across all alive scales at a
        single q value.  The original approach — still works but is
        the least robust of the three.

    Parameters
    ----------
    plpf : dict
        Output of compute_perline_partition.
    q_classify : float, optional
        For multi_q: the lower bound of the q range (all q >= q_classify
        are used).  For other methods: the single q value.
        If None, defaults to 0.6 * q_max.
    weight_threshold : float
        Cumulative weight threshold for the L_P/L_N split.  Lines are
        sorted by their discriminant score (descending) and accumulated;
        those in the top (1 - threshold) fraction are L_P.
    q_idx : int, optional
        Explicit index into q_list.  Overrides q_classify for single-q
        methods.
    method : str
        'multi_q', 'coarse_scale', or 'mean_weight'.
    log2_a_min, log2_a_max : float, optional
        Scale range restriction.  If given, only scales in this range
        contribute to the weight computation.  Particularly useful for
        coarse_scale method to avoid edge effects.

    Returns
    -------
    classification : dict with keys:
        'is_LP'          : (n_lines,) bool, True if line ∈ L_P
        'is_LN'          : (n_lines,) bool, True if line ∈ L_N
        'LP_indices'     : array of line indices in L_P
        'LN_indices'     : array of line indices in L_N
        'line_scores'    : (n_lines,) discriminant score per line
        'q_classify'     : the q value (or lower bound) used
        'method'         : the method used
        'n_LP'           : number of L_P lines
        'n_LN'           : number of L_N lines
    """
    q_array = plpf['q_list']
    n_lines = plpf['n_lines']
    n_scales = len(plpf['scales'])
    log2_a = plpf['log2_a']

    # Default q_classify: 60% of the way through the positive q range
    if q_classify is None and q_idx is None:
        q_pos = q_array[q_array > 0]
        q_classify = 0.6 * q_pos.max() if len(q_pos) > 0 else q_array.max()

    if q_idx is None:
        q_idx = int(np.argmin(np.abs(q_array - q_classify)))
    actual_q = q_array[q_idx]

    # Optional scale range mask
    if log2_a_min is not None or log2_a_max is not None:
        s_lo = log2_a_min if log2_a_min is not None else log2_a[0]
        s_hi = log2_a_max if log2_a_max is not None else log2_a[-1]
        scale_mask = (log2_a >= s_lo) & (log2_a <= s_hi)
    else:
        scale_mask = np.ones(n_scales, dtype=bool)

    line_scores = np.zeros(n_lines)

    if method == 'multi_q':
        # Integrate log-weight over all q >= q_classify.
        # For L_P lines, log(w) stays roughly constant or increases.
        # For L_N lines, log(w) plummets as q grows.
        q_high_mask = np.arange(len(q_array)) >= q_idx
        q_high_idx = np.where(q_high_mask)[0]

        if len(q_high_idx) < 2:
            # Fall back to single q if not enough high-q values
            q_high_idx = np.array([q_idx])

        for li in range(n_lines):
            # For each q in the high-q range, get the mean weight
            # of this line across alive scales in the fitting range
            log_w_sum = 0.0
            n_q_contrib = 0
            for qi in q_high_idx:
                w_qi = plpf['w_line_q'][qi, li, :]  # (n_scales,)
                alive = np.isfinite(w_qi) & (w_qi > 0) & scale_mask
                if np.any(alive):
                    # Use geometric mean (= mean of log) for stability
                    log_w_sum += np.mean(np.log(w_qi[alive]))
                    n_q_contrib += 1
                else:
                    # Line is dead or has zero weight at this q — penalize
                    log_w_sum += -50.0  # effectively -∞
                    n_q_contrib += 1
            if n_q_contrib > 0:
                line_scores[li] = log_w_sum / n_q_contrib

    elif method == 'coarse_scale':
        # Use the Boltzmann weight at the coarsest alive scale
        w = plpf['w_line_q'][q_idx]  # (n_lines, n_scales)
        for li in range(n_lines):
            alive = np.isfinite(w[li]) & (w[li] > 0) & scale_mask
            if np.any(alive):
                # Coarsest alive scale index
                coarsest = np.where(alive)[0][-1]
                line_scores[li] = np.log(w[li, coarsest])
            else:
                line_scores[li] = -50.0

    elif method == 'mean_weight':
        # Legacy: mean weight at a single q across all alive scales
        w = plpf['w_line_q'][q_idx]
        for li in range(n_lines):
            alive = np.isfinite(w[li]) & scale_mask
            if np.any(alive):
                line_scores[li] = np.nanmean(w[li, alive])

    else:
        raise ValueError(f"method must be 'multi_q', 'coarse_scale', "
                         f"or 'mean_weight', got {method!r}")

    # Sort by score descending and accumulate for threshold split
    order = np.argsort(-line_scores)

    # For log-space scores (multi_q, coarse_scale), convert to
    # positive weights for cumulative thresholding
    if method in ('multi_q', 'coarse_scale'):
        # Shift scores so the maximum is 0, then exponentiate
        shifted = line_scores[order] - line_scores[order[0]]
        pos_weights = np.exp(shifted)
    else:
        pos_weights = line_scores[order]

    cumulative = np.cumsum(pos_weights)
    total = cumulative[-1] if len(cumulative) > 0 else 0.0

    if total > 0:
        cumulative_frac = cumulative / total
        cutoff_idx = np.searchsorted(cumulative_frac, 1.0 - weight_threshold)
        is_LP = np.zeros(n_lines, dtype=bool)
        is_LP[order[:cutoff_idx + 1]] = True
    else:
        is_LP = np.ones(n_lines, dtype=bool)

    is_LN = ~is_LP

    return {
        'is_LP': is_LP,
        'is_LN': is_LN,
        'LP_indices': np.where(is_LP)[0],
        'LN_indices': np.where(is_LN)[0],
        'line_scores': line_scores,
        'line_weights': np.exp(line_scores - line_scores.max()),  # compat
        'q_classify': actual_q,
        'method': method,
        'n_LP': int(is_LP.sum()),
        'n_LN': int(is_LN.sum()),
    }


def compute_subset_spectra(plpf, classification, log2_a_min, log2_a_max,
                           subset='LP'):
    """Compute τ(q), h(q), D(q) for a subset of lines (L_P or L_N).

    Reconstructs the partition function using only lines in the specified
    subset, then fits slopes in the scaling range.

    Parameters
    ----------
    plpf : dict
        Output of compute_perline_partition.
    classification : dict
        Output of classify_lines.
    log2_a_min, log2_a_max : float
        Scale range for regression (in log2 units).
    subset : str
        'LP' for singularity-supporting lines, 'LN' for smooth-part lines.

    Returns
    -------
    spectra : dict with keys:
        'tau_q', 'h_q', 'D_q' : arrays
        'q_list' : array
        'tau_err', 'h_err', 'D_err' : standard errors
        'subset' : str
        'n_lines_used' : int
    """
    from .spectra import _fit_slope

    if subset == 'LP':
        mask = classification['is_LP']
    elif subset == 'LN':
        mask = classification['is_LN']
    else:
        raise ValueError(f"subset must be 'LP' or 'LN', got {subset!r}")

    q_array = plpf['q_list']
    log2_a = plpf['log2_a']
    n_q = len(q_array)
    n_scales = len(log2_a)
    LN2 = np.log(2.0)

    # Scale index range for regression
    dx = 1.0 / plpf['n_voice']
    log2_a0 = log2_a[0]
    idx_min = max(0, int((log2_a_min - log2_a0) / dx))
    idx_max = min(n_scales - 1, int((log2_a_max - log2_a0) / dx))
    valid_idx = np.arange(idx_min, idx_max + 1)
    x = log2_a[valid_idx]

    if len(valid_idx) < 2:
        raise ValueError("Not enough scales in range for regression")

    # Rebuild Z, H, D from the line subset
    Z_subset = plpf['Z_line_q'][:, mask, :]  # (n_q, n_subset, n_scales)
    line_E = plpf['line_energies'][mask, :]    # (n_subset, n_scales)

    h_q = np.full(n_q, np.nan)
    d_q = np.full(n_q, np.nan)
    tau_q = np.full(n_q, np.nan)
    h_err = np.full(n_q, np.nan)
    d_err = np.full(n_q, np.nan)
    tau_err = np.full(n_q, np.nan)

    for qi in range(n_q):
        # T(q,a) = log2(Z_subset)
        Z_per_scale = np.sum(Z_subset[qi], axis=0)  # (n_scales,)
        T_q = np.full(n_scales, np.nan)
        pos = Z_per_scale > 0
        T_q[pos] = np.log(Z_per_scale[pos]) / LN2

        # H(q,a) and D(q,a) from Boltzmann weights within the subset
        H_q_a = np.full(n_scales, np.nan)
        D_q_a = np.full(n_scales, np.nan)

        for si in range(n_scales):
            Z_si = Z_per_scale[si]
            if Z_si <= 0:
                continue

            alive = np.isfinite(line_E[:, si]) & (line_E[:, si] > 0)
            if not np.any(alive):
                continue

            E = line_E[alive, si]
            Eq = Z_subset[qi, alive, si]
            w = Eq / Z_si

            ln_E = np.log(E) / LN2  # in log2
            H_q_a[si] = np.sum(w * ln_E)

            with np.errstate(divide='ignore', invalid='ignore'):
                ln_w = np.where(w > 0, np.log(w) / LN2, 0.0)
            D_q_a[si] = np.sum(w * ln_w)

        # Fit slopes over the regression range
        y_H = H_q_a[valid_idx]
        y_D = D_q_a[valid_idx]
        y_T = T_q[valid_idx]

        if np.all(np.isfinite(y_H)):
            slope, se, _ = _fit_slope(x, y_H)
            h_q[qi] = slope
            h_err[qi] = se

        if np.all(np.isfinite(y_D)):
            slope, se, _ = _fit_slope(x, y_D)
            d_q[qi] = slope
            d_err[qi] = se

        if np.all(np.isfinite(y_T)):
            slope, se, _ = _fit_slope(x, y_T)
            tau_q[qi] = slope
            tau_err[qi] = se

    return {
        'tau_q': tau_q,
        'h_q': h_q,
        'D_q': d_q,
        'q_list': q_array,
        'tau_err': tau_err,
        'h_err': h_err,
        'D_err': d_err,
        'subset': subset,
        'n_lines_used': int(mask.sum()),
        'method': 'canonical_perline',
    }


def detect_phase_transition_perline(plpf, log2_a_min, log2_a_max):
    """Detect phase transition in τ(q) from the per-line partition function.

    Computes τ(q) from the aggregate H and D, then looks for a change
    in slope of τ(q) vs q for positive q.  Returns the estimated q_crit
    and the two asymptotic slopes.

    Parameters
    ----------
    plpf : dict
        Output of compute_perline_partition.
    log2_a_min, log2_a_max : float
        Scale range for regression.

    Returns
    -------
    result : dict with keys:
        'tau_q'     : full τ(q) curve
        'h_q'       : full h(q) curve
        'D_q'       : full D(q) curve
        'q_crit'    : estimated phase transition q value (NaN if none)
        'slope_low' : τ slope for q < q_crit (the n_ψ regime)
        'slope_high': τ slope for q > q_crit (approaching h_min)
        'q_list'    : q values
    """
    from .spectra import _fit_slope

    q_array = plpf['q_list']
    log2_a = plpf['log2_a']
    n_q = len(q_array)
    LN2 = np.log(2.0)

    dx = 1.0 / plpf['n_voice']
    log2_a0 = log2_a[0]
    idx_min = max(0, int((log2_a_min - log2_a0) / dx))
    idx_max = min(len(log2_a) - 1, int((log2_a_max - log2_a0) / dx))
    valid_idx = np.arange(idx_min, idx_max + 1)
    x = log2_a[valid_idx]

    # Compute h(q) and D(q) from aggregate Boltzmann quantities
    h_q = np.full(n_q, np.nan)
    d_q = np.full(n_q, np.nan)

    for qi in range(n_q):
        y_H = plpf['H_q'][qi, valid_idx] / LN2  # convert to log2
        y_D = plpf['D_q'][qi, valid_idx] / LN2

        if np.all(np.isfinite(y_H)):
            slope, _, _ = _fit_slope(x, y_H)
            h_q[qi] = slope
        if np.all(np.isfinite(y_D)):
            slope, _, _ = _fit_slope(x, y_D)
            d_q[qi] = slope

    tau_q = q_array * h_q - d_q

    # Look for phase transition in positive q:
    # τ(q) should have different slopes before and after q_crit.
    # Use piecewise linear fit to find the breakpoint.
    pos_q = q_array > 0.5  # focus on positive q
    pos_finite = pos_q & np.isfinite(tau_q)

    q_crit = np.nan
    slope_low = np.nan
    slope_high = np.nan

    if np.sum(pos_finite) >= 6:
        q_pos = q_array[pos_finite]
        tau_pos = tau_q[pos_finite]

        best_rss = np.inf
        best_break = None

        # Try each possible breakpoint
        for bi in range(2, len(q_pos) - 2):
            q_lo, tau_lo = q_pos[:bi+1], tau_pos[:bi+1]
            q_hi, tau_hi = q_pos[bi:], tau_pos[bi:]

            if len(q_lo) < 2 or len(q_hi) < 2:
                continue

            c1 = np.polyfit(q_lo, tau_lo, 1)
            c2 = np.polyfit(q_hi, tau_hi, 1)
            rss = (np.sum((tau_lo - np.polyval(c1, q_lo))**2) +
                   np.sum((tau_hi - np.polyval(c2, q_hi))**2))

            if rss < best_rss:
                best_rss = rss
                best_break = bi
                slope_low = c1[0]
                slope_high = c2[0]

        if best_break is not None:
            q_crit = q_pos[best_break]

    return {
        'tau_q': tau_q,
        'h_q': h_q,
        'D_q': d_q,
        'q_crit': q_crit,
        'slope_low': slope_low,
        'slope_high': slope_high,
        'q_list': q_array,
    }


def plot_line_classification(plpf, classification, extrep_abs, scales,
                             coarser_links, n_scales, fig=None, axes=None):
    """Visualize the classification of maxima lines as L_P vs L_N.

    Plots the skeleton with L_P lines in red and L_N lines in blue,
    plus histograms of Boltzmann weights and line-length distributions.

    Parameters
    ----------
    plpf : dict
        Output of compute_perline_partition.
    classification : dict
        Output of classify_lines.
    extrep_abs : list of arrays
        Position arrays per scale (for plotting line positions).
    scales : array
        Scale values.
    coarser_links : list of int arrays
        Chain links.
    n_scales : int
        Total scales.

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    is_LP = classification['is_LP']
    is_LN = classification['is_LN']
    line_finest = plpf['line_finest_idx']
    weights = classification['line_weights']

    # Panel 1: Skeleton with classification coloring
    ax = axes[0]
    for li in range(plpf['n_lines']):
        fi = line_finest[li]
        positions = []
        scale_vals = []
        cs, ci = 0, fi
        while cs < n_scales and ci != -1:
            positions.append(extrep_abs[cs][ci])
            scale_vals.append(np.log2(scales[cs]))
            ci = coarser_links[cs][ci]
            cs += 1
        color = 'tab:red' if is_LP[li] else 'tab:blue'
        alpha = 0.7 if is_LP[li] else 0.3
        lw = 1.0 if is_LP[li] else 0.5
        ax.plot(positions, scale_vals, '-', color=color, alpha=alpha,
                linewidth=lw)

    ax.set_xlabel('Position')
    ax.set_ylabel('log₂(a)')
    ax.set_title(f'Skeleton: L_P ({classification["n_LP"]} red) '
                 f'vs L_N ({classification["n_LN"]} blue)')

    # Panel 2: Boltzmann weight distribution
    ax = axes[1]
    w_LP = weights[is_LP]
    w_LN = weights[is_LN]
    if len(w_LP) > 0:
        ax.hist(np.log10(w_LP + 1e-30), bins=50, alpha=0.7,
                color='tab:red', label=f'L_P ({len(w_LP)})')
    if len(w_LN) > 0:
        ax.hist(np.log10(w_LN + 1e-30), bins=50, alpha=0.7,
                color='tab:blue', label=f'L_N ({len(w_LN)})')
    ax.set_xlabel('log₁₀(Boltzmann weight)')
    ax.set_ylabel('Count')
    ax.set_title(f'Weight distribution (q={classification["q_classify"]:.1f})')
    ax.legend()

    # Panel 3: Line length distribution
    ax = axes[2]
    len_LP = plpf['line_lengths'][is_LP]
    len_LN = plpf['line_lengths'][is_LN]
    max_len = max(plpf['line_lengths'].max(), 1)
    bins = np.arange(0, max_len + 2) - 0.5
    if len(len_LP) > 0:
        ax.hist(len_LP, bins=bins, alpha=0.7, color='tab:red',
                label=f'L_P (med={np.median(len_LP):.0f})')
    if len(len_LN) > 0:
        ax.hist(len_LN, bins=bins, alpha=0.7, color='tab:blue',
                label=f'L_N (med={np.median(len_LN):.0f})')
    ax.set_xlabel('Line length (voices)')
    ax.set_ylabel('Count')
    ax.set_title('Line length distribution')
    ax.legend()

    plt.tight_layout()
    return fig, axes


def perline_to_pf(plpf, classification=None, subset=None):
    """Convert per-line partition results to a standard PF dict.

    Rebuilds the 7 aggregate arrays (sTq, logSTq, …) from a subset of
    lines so that the result can be passed directly to
    ``compute_spectra()`` or ``plot_partition_functions()``.

    Parameters
    ----------
    plpf : dict
        Output of ``compute_perline_partition``.
    classification : dict, optional
        Output of ``classify_lines``.  Required when *subset* is given.
    subset : {'LP', 'LN'}, optional
        If given, restrict to lines in L_P or L_N.  If None, use all lines.

    Returns
    -------
    pf : dict
        Standard partition-function dict with all 7 arrays, metadata,
        and ``signal_number=1``.  Arrays are in natural log (matching
        ``compute_partition_function`` convention).
    """
    if subset is not None:
        if classification is None:
            raise ValueError("classification required when subset is given")
        mask = classification['is_LP'] if subset == 'LP' else classification['is_LN']
    else:
        mask = np.ones(plpf['n_lines'], dtype=bool)

    q_array = plpf['q_list']
    n_q = len(q_array)
    n_scales = len(plpf['scales'])
    line_E = plpf['line_energies'][mask]       # (n_sub, n_scales)
    Z_line = plpf['Z_line_q'][:, mask, :]      # (n_q, n_sub, n_scales)

    sTq = np.zeros((n_q, n_scales))
    sTqLogT = np.zeros((n_q, n_scales))
    logSTq = np.zeros((n_q, n_scales))
    sTqLogT_sTq = np.zeros((n_q, n_scales))
    log2STq = np.zeros((n_q, n_scales))
    sTqLogT_sTq2 = np.zeros((n_q, n_scales))
    logSTqSTqLogT_sTq = np.zeros((n_q, n_scales))

    for qi in range(n_q):
        for si in range(n_scales):
            alive = np.isfinite(line_E[:, si]) & (line_E[:, si] > 0)
            if not np.any(alive):
                continue

            E = line_E[alive, si]
            Eq = Z_line[qi, alive, si]
            Zs = np.sum(Eq)
            if Zs <= 0:
                continue

            N = int(np.sum(alive))
            ln_E = np.log(E)
            Eq_lnE = np.sum(Eq * ln_E)

            sTq[qi, si] = Zs
            sTqLogT[qi, si] = Eq_lnE

            logZ = np.log(Zs / N)
            h_val = Eq_lnE / Zs

            logSTq[qi, si] = logZ
            sTqLogT_sTq[qi, si] = h_val
            log2STq[qi, si] = logZ * logZ
            sTqLogT_sTq2[qi, si] = h_val * h_val
            logSTqSTqLogT_sTq[qi, si] = logZ * h_val

    return {
        'sTq': sTq,
        'sTqLogT': sTqLogT,
        'logSTq': logSTq,
        'sTqLogT_sTq': sTqLogT_sTq,
        'log2STq': log2STq,
        'sTqLogT_sTq2': sTqLogT_sTq2,
        'logSTqSTqLogT_sTq': logSTqSTqLogT_sTq,
        'q_list': q_array,
        'scales': plpf['scales'],
        'log2_a': plpf['log2_a'],
        'n_ext': np.zeros(n_scales, dtype=np.int64),
        'index_max': n_scales - 1,
        'n_voice': plpf['n_voice'],
        'n_oct': int(n_scales / plpf['n_voice']),
        'a_min': plpf['scales'][0],
        'signal_number': 1,
    }


def plot_weight_skeleton(plpf, extrep_abs, scales, coarser_links, n_scales,
                          q_values=None, n_panels=5, cmap='magma',
                          log_scale=False, fig=None, axes=None):
    """Multi-panel skeleton colored by Boltzmann weight at different q values.

    Shows how the dominant maxima lines shift as q increases — at low q,
    L_N lines (smooth-part) carry most weight; at high q, L_P lines
    (singularity) dominate.  The transition is visible across panels.

    Parameters
    ----------
    plpf : dict
        Output of ``compute_perline_partition``.
    extrep_abs : list of arrays
        Position arrays per scale.
    scales : array
        Scale values.
    coarser_links : list of int arrays
        Chain links.
    n_scales : int
        Total number of scales.
    q_values : array-like, optional
        Specific q values to show.  If None, auto-pick ~n_panels values.
    n_panels : int
        Number of panels (used when q_values is None).
    cmap : str
        Matplotlib colormap for weight intensity.
    log_scale : bool
        If True, color by log10(weight) for better dynamic range.
    fig, axes : optional
        Existing figure/axes to draw into.

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib import cm

    q_array = plpf['q_list']

    if q_values is None:
        # Auto-pick spanning the q range
        idx = np.linspace(0, len(q_array) - 1, n_panels, dtype=int)
        q_values = q_array[idx]
    q_values = np.asarray(q_values)
    n_panels = len(q_values)

    if axes is None:
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5),
                                  sharey=True)
    if n_panels == 1:
        axes = [axes]

    colormap = cm.get_cmap(cmap)
    line_finest = plpf['line_finest_idx']
    n_lines = plpf['n_lines']

    for pi, q_val in enumerate(q_values):
        qi = int(np.argmin(np.abs(q_array - q_val)))
        ax = axes[pi]

        # Per-scale Boltzmann weights for this q
        w = plpf['w_line_q'][qi]  # (n_lines, n_scales)

        # Colormap normalization from all finite per-scale weights
        w_finite = w[np.isfinite(w)]
        w_pos = w_finite[w_finite > 0]

        if log_scale:
            if len(w_pos) > 0:
                lw_pos = np.log10(w_pos)
                vmin = np.percentile(lw_pos, 5)
                vmax = np.percentile(lw_pos, 95)
            else:
                vmin, vmax = -10, 0
        else:
            vmin = 0
            vmax = np.percentile(w_pos, 95) if len(w_pos) > 0 else 1

        norm = Normalize(vmin=vmin, vmax=vmax)

        for li in range(n_lines):
            fi = line_finest[li]
            positions = []
            scale_vals = []
            seg_weights = []
            cs, ci = 0, fi
            while cs < n_scales and ci != -1:
                positions.append(extrep_abs[cs][ci])
                scale_vals.append(np.log2(scales[cs]))
                w_here = w[li, cs]
                if np.isfinite(w_here) and w_here > 0:
                    seg_weights.append(np.log10(w_here) if log_scale else w_here)
                else:
                    seg_weights.append(vmin)
                ci = coarser_links[cs][ci]
                cs += 1
            # Color each segment by its scale's weight
            for j in range(len(positions) - 1):
                c = colormap(norm(seg_weights[j]))
                ax.plot(positions[j:j+2], scale_vals[j:j+2], '-',
                        color=c, linewidth=0.7, alpha=0.8)

        ax.set_xlabel('Position')
        if pi == 0:
            ax.set_ylabel('log₂(a)')
        ax.set_title(f'q = {q_array[qi]:.1f}')

    # Shared colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    label = 'log₁₀(weight)' if log_scale else 'Boltzmann weight'
    fig.colorbar(sm, ax=axes, label=label, shrink=0.8)
    fig.suptitle('Skeleton colored by Boltzmann weight', fontsize=13, y=1.02)
    plt.tight_layout()
    return fig, axes


def _modmin_boundaries(coeffs, extrep_idx, dx, x0):
    """Precompute modulus-minimum boundaries for each extremum at each scale.

    At each scale, the territory of extremum *i* extends from the position
    of the |T_ψ| minimum between extremum *i-1* and *i* to the minimum
    between *i* and *i+1*.  Edge extrema extend to the signal boundary.

    Returns
    -------
    bounds : list of (left, right) arrays, one pair per scale.
        ``left[i]`` and ``right[i]`` are the physical-coordinate boundaries
        of the *i*-th extremum at that scale.
    """
    n_scales = coeffs.shape[0]
    n_samp = coeffs.shape[1]
    bounds = []

    for s in range(n_scales):
        idx = extrep_idx[s]  # sample indices of extrema at this scale
        n_ext = len(idx)
        left = np.empty(n_ext)
        right = np.empty(n_ext)

        if n_ext == 0:
            bounds.append((left, right))
            continue

        absrow = np.abs(coeffs[s])

        for i in range(n_ext):
            # Left boundary
            if i == 0:
                left[i] = x0  # signal start
            else:
                seg = absrow[idx[i - 1]:idx[i] + 1]
                j_min = np.argmin(seg)
                left[i] = (idx[i - 1] + j_min) * dx + x0

            # Right boundary
            if i == n_ext - 1:
                right[i] = (n_samp - 1) * dx + x0  # signal end
            else:
                seg = absrow[idx[i]:idx[i + 1] + 1]
                j_min = np.argmin(seg)
                right[i] = (idx[i] + j_min) * dx + x0

        bounds.append((left, right))

    return bounds


def plot_weight_scalogram(plpf, extrep_abs, extrep_idx, coeffs, scales,
                          coarser_links, n_scales,
                          q_values=None, n_panels=5, cmap='magma',
                          log_scale=True, log_base=2, orient='horizontal',
                          classification=None,
                          dx=None, x0=0.0, units='', wavelet=None,
                          fig=None, axes=None):
    """Multi-panel filled scalogram colored by Boltzmann weight.

    Each extremum's region (bounded by the modulus minima of |T_ψ| on
    either side) is filled with a color representing the Boltzmann weight
    of that extremum's maxima line at the given q.  Every rectangle
    belonging to the same line shares a single color (geometric-mean
    weight across alive scales).

    Parameters
    ----------
    plpf : dict
        Output of ``compute_perline_partition``.
    extrep_abs : list of arrays
        Extremum positions per scale (physical coordinates).
    extrep_idx : list of arrays
        Extremum sample indices per scale (for modulus-minimum lookup).
    coeffs : ndarray, shape (n_scales, n_samples)
        Wavelet coefficients (used to find |T_ψ| minima between extrema).
    scales : array
        Scale values.
    coarser_links : list of int arrays
        Extrema chain links (finer→coarser).
    n_scales : int
        Total number of scales.
    q_values : array-like, optional
        Specific q values to show. If None, auto-pick ~n_panels values.
    n_panels : int
        Number of panels (used when q_values is None).
    cmap : str
        Matplotlib colormap.
    log_scale : bool
        If True, color by log(weight) for better dynamic range.
    log_base : int or float
        Base for log-weight colorbar: 2 (default, consistent with dyadic
        scales) or 10.  Only used when log_scale=True.
    orient : str
        'horizontal' (default): x = position, y = log₂(a).
        'vertical': x = log₂(a) (coarse left), y = position/depth (downward).
    classification : dict, optional
        Output of ``classify_lines``.  When provided, L_P line outlines
        are drawn in red and L_N in cyan over the filled scalogram.
    dx : float, optional
        Sampling interval (for physical coordinates and secondary axis).
    x0 : float
        Origin of physical coordinate axis (default 0).
    units : str
        Physical units label (e.g. 'ft').
    wavelet : str, optional
        Wavelet name for c_psi lookup (physical scale axis).
    fig, axes : optional
        Existing figure/axes.

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    from matplotlib import cm

    q_array = plpf['q_list']
    vertical = (orient == 'vertical')

    # Log conversion factor
    _log_fn = np.log2 if log_base == 2 else np.log10
    _log_label = 'log₂' if log_base == 2 else 'log₁₀'

    if q_values is None:
        idx = np.linspace(0, len(q_array) - 1, n_panels, dtype=int)
        q_values = q_array[idx]
    q_values = np.asarray(q_values)
    n_panels = len(q_values)

    if axes is None:
        if vertical:
            fig, axes = plt.subplots(1, n_panels,
                                      figsize=(4 * n_panels, 10),
                                      sharey=True)
        else:
            fig, axes = plt.subplots(1, n_panels,
                                      figsize=(4 * n_panels, 5),
                                      sharey=True)
    if n_panels == 1:
        axes = [axes]

    colormap = cm.get_cmap(cmap)
    line_finest = plpf['line_finest_idx']
    n_lines = plpf['n_lines']
    log2_scales = np.log2(scales)

    # Precompute modulus-minimum boundaries for every extremum at every scale
    _dx = dx if dx is not None else 1.0
    mod_bounds = _modmin_boundaries(coeffs, extrep_idx, _dx, x0)

    # Precompute scale row heights (half-distance to neighbors)
    row_height = np.zeros(n_scales)
    for s in range(n_scales):
        if s == 0:
            dh = log2_scales[1] - log2_scales[0] if n_scales > 1 else 0.1
        elif s == n_scales - 1:
            dh = log2_scales[s] - log2_scales[s - 1]
        else:
            dh = (log2_scales[s + 1] - log2_scales[s - 1]) / 2.0
        row_height[s] = dh

    for pi, q_val in enumerate(q_values):
        qi = int(np.argmin(np.abs(q_array - q_val)))
        ax = axes[pi]

        # Per-scale Boltzmann weights for this q
        w = plpf['w_line_q'][qi]  # (n_lines, n_scales)

        # Collapse to ONE weight per line: geometric mean across alive
        # scales, i.e. exp(mean(log(w))) for w > 0.
        w_line = np.full(n_lines, np.nan)
        for li in range(n_lines):
            row = w[li, :]
            mask = np.isfinite(row) & (row > 0)
            if np.any(mask):
                w_line[li] = np.exp(np.mean(np.log(row[mask])))

        # Colormap normalization from per-line weights
        wl_pos = w_line[np.isfinite(w_line) & (w_line > 0)]

        if log_scale:
            if len(wl_pos) > 0:
                lw_pos = _log_fn(wl_pos)
                vmin = np.percentile(lw_pos, 5)
                vmax = np.percentile(lw_pos, 95)
            else:
                vmin, vmax = (-30 if log_base == 2 else -10), 0
        else:
            vmin = 0
            vmax = np.percentile(wl_pos, 95) if len(wl_pos) > 0 else 1

        norm = Normalize(vmin=vmin, vmax=vmax)

        # Build rectangles — every rectangle in a line gets the SAME color
        # Boundaries come from modulus minima, not zero crossings
        rects = []
        colors = []

        for li in range(n_lines):
            wl = w_line[li]
            if not (np.isfinite(wl) and wl > 0):
                continue
            w_val = _log_fn(wl) if log_scale else wl

            fi = line_finest[li]
            cs, ci = 0, fi
            while cs < n_scales and ci != -1:
                bnd_left, bnd_right = mod_bounds[cs]

                if ci < len(bnd_left):
                    x_left = bnd_left[ci]
                    x_right = bnd_right[ci]

                    if x_right > x_left:
                        s_bot = log2_scales[cs] - row_height[cs] / 2.0
                        if vertical:
                            rect = Rectangle((s_bot, x_left),
                                              row_height[cs],
                                              x_right - x_left)
                        else:
                            rect = Rectangle((x_left, s_bot),
                                              x_right - x_left,
                                              row_height[cs])
                        rects.append(rect)
                        colors.append(w_val)

                ci = coarser_links[cs][ci]
                cs += 1

        if rects:
            pc = PatchCollection(rects, cmap=colormap, edgecolors='none')
            pc.set_array(np.array(colors))
            pc.set_clim(vmin, vmax)
            ax.add_collection(pc)

        pos_min = x0
        pos_max = (coeffs.shape[1] - 1) * _dx + x0
        s_lo = log2_scales[0] - row_height[0]
        s_hi = log2_scales[-1] + row_height[-1]

        if vertical:
            ax.set_xlim(s_hi, s_lo)           # coarse left, fine right
            ax.set_ylim(pos_max, pos_min)     # depth increasing downward
            ax.set_xlabel('log₂(a)')
            ax.set_xticks(range(int(np.ceil(s_lo)),
                                int(np.floor(s_hi)) + 1))
            if pi == 0:
                ax.set_ylabel(f'Depth ({units})' if units else 'Position')
        else:
            ax.set_xlim(pos_min, pos_max)
            ax.set_ylim(s_lo, s_hi)
            ax.set_xlabel('Position')
            if pi == 0:
                ax.set_ylabel('log₂(a)')

        # Classification overlay: draw L_P / L_N line outlines
        if classification is not None:
            is_LP = classification['is_LP']
            for li in range(n_lines):
                fi = line_finest[li]
                positions = []
                scale_vals = []
                cs2, ci2 = 0, fi
                while cs2 < n_scales and ci2 != -1:
                    positions.append(extrep_abs[cs2][ci2])
                    scale_vals.append(log2_scales[cs2])
                    ci2 = coarser_links[cs2][ci2]
                    cs2 += 1
                if len(positions) > 1:
                    color = 'tab:red' if is_LP[li] else 'tab:cyan'
                    alpha = 0.8 if is_LP[li] else 0.4
                    lw = 1.2 if is_LP[li] else 0.4
                    if vertical:
                        ax.plot(scale_vals, positions, '-', color=color,
                                alpha=alpha, linewidth=lw)
                    else:
                        ax.plot(positions, scale_vals, '-', color=color,
                                alpha=alpha, linewidth=lw)

        ax.set_title(f'q = {q_array[qi]:.1f}')

    # Physical scale secondary axis (vertical mode)
    if vertical and dx is not None and wavelet is not None:
        from .spectra import _resolve_c_psi, _add_physical_scale_secondary
        c = _resolve_c_psi(wavelet, None)
        for ax_ in axes:
            _add_physical_scale_secondary(ax_, dx, c, units, axis='x')

    # Shared colorbar — placed in its own axes so it doesn't resize panels
    plt.tight_layout()
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    label = f'{_log_label}(weight)' if log_scale else 'Boltzmann weight'
    # Add a thin axes to the right of the last panel
    bbox = axes[-1].get_position()
    cax = fig.add_axes([bbox.x1 + 0.015, bbox.y0, 0.012, bbox.height])
    fig.colorbar(sm, cax=cax, label=label)
    fig.suptitle('Thermodynamic scalogram', fontsize=13, y=1.02)
    return fig, axes


def plot_weight_evolution(plpf, classification, q_values=None,
                          fig=None, ax=None):
    """Plot how Boltzmann weights of L_P vs L_N evolve across q.

    Shows the total weight fraction captured by L_P lines as a function
    of q — this should show a sharp transition at q_crit.

    Parameters
    ----------
    plpf : dict
        Output of compute_perline_partition.
    classification : dict
        Output of classify_lines.
    q_values : array-like, optional
        Subset of q values to plot (default: all).

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    is_LP = classification['is_LP']
    q_array = plpf['q_list']
    n_q = len(q_array)

    # For each q, compute total Boltzmann weight of L_P vs L_N
    # averaged over scales
    LP_frac = np.full(n_q, np.nan)

    for qi in range(n_q):
        w = plpf['w_line_q'][qi]  # (n_lines, n_scales)
        # Average across scales where lines are alive
        w_LP_total = np.nansum(w[is_LP, :], axis=0)   # (n_scales,)
        w_all_total = np.nansum(w, axis=0)             # (n_scales,)
        valid = w_all_total > 0
        if np.any(valid):
            LP_frac[qi] = np.mean(w_LP_total[valid] / w_all_total[valid])

    ax.plot(q_array, LP_frac, 'ro-', markersize=4, label='L_P weight fraction')
    ax.axhline(0.5, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('q')
    ax.set_ylabel('Weight fraction in L_P')
    ax.set_title('Boltzmann weight evolution: L_P vs L_N')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark classification q
    q_class = classification['q_classify']
    ax.axvline(q_class, color='tab:red', ls=':', lw=1,
               label=f'q_classify={q_class:.1f}')

    return fig, ax
