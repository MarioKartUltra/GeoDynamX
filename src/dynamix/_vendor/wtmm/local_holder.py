"""Local Hölder exponent estimation via structure functions and active sets.

Implements the Nguyen et al. (2019) approach for computing pointwise
singularity exponents h̃(x) from wavelet coefficients, calibrated
against the global WTMM multifractal spectrum D(h).

The pipeline:

1. **Structure functions** S_p(a) = ⟨|T_ψ(x,a)|^p⟩_x
   Computed over the full spatial field (not just maxima).  These
   encode the same τ(q) scaling as the WTMM partition function but
   use all spatial points.

2. **Thresholds** from Boltzmann-weighted wavelet coefficients:
   For each moment order p, define a scale-dependent threshold
   T_p(a) = c_p · S_p(a)^{1/p}, where c_p is a constant calibrated
   so that the fractal dimension of the exceedance set matches D(h)
   from the global spectrum.

3. **Active sets** A_p(a):
   At each scale a, the set of positions x where
   |T_ψ(x,a)| ≥ T_p(a).

4. **Local exponents** h̃(x):
   For each position x, h̃(x) is the scaling exponent of the
   active-set membership function across scales.  Specifically,
   find the largest p (= finest singularity) such that x ∈ A_p(a)
   for a sufficient range of scales, and read off h̃ from the
   D(h) ↔ p mapping.

In 1D, the "active volumes" reduce to active sets on the line —
unions of intervals where the wavelet coefficient exceeds the
threshold.  The Hausdorff dimension of the active set A_p is
1 - (τ(p) - p·h(p)) in the multifractal formalism.

References
----------
- Nguyen, Laval, Keast & Bhatt, Phys. Rev. E 99, 053107 (2019)
- Bershadskii, Phys. Rev. E 57, 6029 (1998)
- Muzy, Bacry & Arnéodo, Int. J. Bif. Chaos 4, 245–302 (1994)
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1.  Structure functions from full CWT field
# ──────────────────────────────────────────────────────────────────────

def compute_structure_functions(coeffs, scales, p_list,
                                valid_ranges=None):
    """Compute wavelet-based structure functions S_p(a).

    S_p(a) = (1/N) Σ_x |T_ψ(x, a)|^p

    These are spatial averages over the full coefficient field,
    NOT restricted to maxima lines.

    Parameters
    ----------
    coeffs : ndarray, shape (n_scales, n_positions)
        CWT coefficient matrix from cwtd / cwtd_fftw / cwtd_mlx.
    scales : 1D array, shape (n_scales,)
        Scale values corresponding to rows of coeffs.
    p_list : array-like
        Moment orders p > 0.  Typically the same as q_list from
        the partition function, restricted to positive values.
    valid_ranges : list of (int, int), optional
        Per-scale (firstp, lastp) valid index ranges from the CWT.
        If given, only valid positions contribute to the average.
        This avoids border artefacts.

    Returns
    -------
    result : dict with keys:
        'S_p'     : ndarray (n_p, n_scales), structure function values
        'zeta_p'  : ndarray (n_p,), scaling exponents ζ(p) from
                    log-log regression of S_p vs a
        'p_list'  : 1D array of moment orders
        'scales'  : 1D array of scales
        'log_a'   : natural log of scales
    """
    p_array = np.asarray(p_list, dtype=np.float64)
    n_p = len(p_array)
    n_scales = coeffs.shape[0]

    S_p = np.zeros((n_p, n_scales))

    for si in range(n_scales):
        row = np.abs(coeffs[si])

        # Restrict to valid range if provided
        if valid_ranges is not None:
            fp, lp = valid_ranges[si]
            row = row[fp:lp + 1]

        N = len(row)
        if N == 0:
            S_p[:, si] = np.nan
            continue

        for pi, p in enumerate(p_array):
            S_p[pi, si] = np.mean(row ** p)

    # Fit scaling exponents ζ(p) via log-log regression
    log_a = np.log(scales)
    zeta_p = np.full(n_p, np.nan)

    for pi in range(n_p):
        y = np.log(S_p[pi])
        good = np.isfinite(y) & np.isfinite(log_a)
        if np.sum(good) >= 3:
            slope, _ = np.polyfit(log_a[good], y[good], 1)
            zeta_p[pi] = slope

    return {
        'S_p': S_p,
        'zeta_p': zeta_p,
        'p_list': p_array,
        'scales': scales,
        'log_a': log_a,
    }


def fit_zeta_in_range(sf_result, log2_a_min, log2_a_max):
    """Re-fit ζ(p) over a restricted scale range.

    Parameters
    ----------
    sf_result : dict
        Output of compute_structure_functions.
    log2_a_min, log2_a_max : float
        Scale range in log₂ units.

    Returns
    -------
    zeta_p : 1D array of fitted scaling exponents
    zeta_err : 1D array of standard errors
    """
    log2_a = np.log2(sf_result['scales'])
    mask = (log2_a >= log2_a_min) & (log2_a <= log2_a_max)
    x = np.log(sf_result['scales'][mask])

    n_p = len(sf_result['p_list'])
    zeta_p = np.full(n_p, np.nan)
    zeta_err = np.full(n_p, np.nan)

    for pi in range(n_p):
        y = np.log(sf_result['S_p'][pi, mask])
        good = np.isfinite(y) & np.isfinite(x)
        if np.sum(good) >= 3:
            coeffs, cov = np.polyfit(x[good], y[good], 1, cov=True)
            zeta_p[pi] = coeffs[0]
            zeta_err[pi] = np.sqrt(cov[0, 0])

    return zeta_p, zeta_err


# ──────────────────────────────────────────────────────────────────────
# 2.  Active sets and threshold calibration
# ──────────────────────────────────────────────────────────────────────

def compute_active_sets(coeffs, scales, sf_result, c_p,
                        valid_ranges=None):
    """Compute active sets A_p(a) for each moment order p and scale a.

    A position x is "active" at scale a for order p if:
        |T_ψ(x, a)| ≥ c_p · S_p(a)^{1/p}

    Parameters
    ----------
    coeffs : ndarray (n_scales, n_positions)
    scales : 1D array (n_scales,)
    sf_result : dict
        Output of compute_structure_functions.
    c_p : 1D array or float
        Threshold constant(s).  One per p value, or a single scalar
        applied to all.
    valid_ranges : list of (int, int), optional
        Per-scale valid index ranges.

    Returns
    -------
    active : dict with keys:
        'masks'      : ndarray (n_p, n_scales, n_positions), bool
                       True where |T_ψ| ≥ threshold
        'fractions'  : ndarray (n_p, n_scales), fraction of active points
        'thresholds' : ndarray (n_p, n_scales), the threshold T_p(a) used
        'c_p'        : 1D array, the constants used
    """
    S_p = sf_result['S_p']
    p_array = sf_result['p_list']
    n_p = len(p_array)
    n_scales, n_pos = coeffs.shape

    c_arr = np.atleast_1d(np.asarray(c_p, dtype=np.float64))
    if len(c_arr) == 1:
        c_arr = np.full(n_p, c_arr[0])

    masks = np.zeros((n_p, n_scales, n_pos), dtype=bool)
    fractions = np.zeros((n_p, n_scales))
    thresholds = np.zeros((n_p, n_scales))

    abs_coeffs = np.abs(coeffs)

    for pi, p in enumerate(p_array):
        for si in range(n_scales):
            # Threshold: c_p · S_p(a)^{1/p}
            T_pa = c_arr[pi] * (S_p[pi, si] ** (1.0 / p))
            thresholds[pi, si] = T_pa

            row = abs_coeffs[si]

            # Apply threshold
            active_mask = row >= T_pa

            # Zero out invalid positions
            if valid_ranges is not None:
                fp, lp = valid_ranges[si]
                active_mask[:fp] = False
                active_mask[lp + 1:] = False

            masks[pi, si] = active_mask

            # Fraction of valid positions that are active
            if valid_ranges is not None:
                fp, lp = valid_ranges[si]
                n_valid = lp - fp + 1
            else:
                n_valid = n_pos

            fractions[pi, si] = np.sum(active_mask) / max(n_valid, 1)

    return {
        'masks': masks,
        'fractions': fractions,
        'thresholds': thresholds,
        'c_p': c_arr,
    }


def calibrate_c_p(coeffs, scales, sf_result, D_h_target,
                   h_target, valid_ranges=None,
                   log2_a_min=None, log2_a_max=None,
                   c_range=(0.1, 5.0), n_grid=50):
    """Calibrate threshold constants c_p so active-set dimensions match D(h).

    For each moment order p, the active set A_p has fractal dimension
    d_A(p) measured by log-log regression of the fraction of active
    points vs scale.  In 1D:

        fraction(a) ~ a^{1 - d_A}

    We want d_A(p) = D(h(p)), where h(p) and D(h) come from the
    global WTMM spectrum.

    The mapping p → h is given by h(p) = dζ/dp (Legendre),
    or equivalently from the canonical h(q) at q=p.

    Parameters
    ----------
    coeffs : ndarray (n_scales, n_positions)
    scales : 1D array
    sf_result : dict from compute_structure_functions
    D_h_target : 1D array
        Target D(h) values, one per p in sf_result['p_list'].
        These come from the WTMM canonical spectrum evaluated
        at the h(q) values for q = p_list.
    h_target : 1D array
        Target h values, one per p (from WTMM h(q) at q=p).
    valid_ranges : list of (int, int), optional
    log2_a_min, log2_a_max : float, optional
        Scale range for dimension regression.
    c_range : tuple (float, float)
        Search range for c_p.
    n_grid : int
        Number of grid points for the 1D search.

    Returns
    -------
    c_p_opt : 1D array of optimal constants
    d_A_opt : 1D array of achieved active-set dimensions
    info : dict with 'c_grid', 'd_A_grid', 'cost_grid' for diagnostics
    """
    p_array = sf_result['p_list']
    n_p = len(p_array)
    log2_a = np.log2(scales)

    # Scale range mask
    if log2_a_min is None:
        log2_a_min = log2_a[0]
    if log2_a_max is None:
        log2_a_max = log2_a[-1]
    scale_mask = (log2_a >= log2_a_min) & (log2_a <= log2_a_max)
    log_a_fit = np.log(scales[scale_mask])

    c_grid = np.linspace(c_range[0], c_range[1], n_grid)
    c_p_opt = np.full(n_p, 1.0)
    d_A_opt = np.full(n_p, np.nan)
    cost_grid_all = np.zeros((n_p, n_grid))

    for pi, p in enumerate(p_array):
        target_dim = D_h_target[pi]
        if not np.isfinite(target_dim):
            continue

        # In 1D, fraction ~ a^{1 - d_A}, so
        # log(fraction) = (1 - d_A) · log(a) + const
        # d_A = 1 - slope
        target_slope = 1.0 - target_dim  # what we want from regression

        best_cost = np.inf
        best_c = 1.0
        best_d = np.nan

        for ci, c_val in enumerate(c_grid):
            # Compute thresholds and fractions for this c
            T_pa = c_val * (sf_result['S_p'][pi] ** (1.0 / p))
            abs_c = np.abs(coeffs)
            n_scales, n_pos = coeffs.shape

            fracs = np.zeros(n_scales)
            for si in range(n_scales):
                row = abs_c[si]
                active = row >= T_pa[si]
                if valid_ranges is not None:
                    fp, lp = valid_ranges[si]
                    active[:fp] = False
                    active[lp + 1:] = False
                    n_valid = lp - fp + 1
                else:
                    n_valid = n_pos
                fracs[si] = np.sum(active) / max(n_valid, 1)

            # Fit slope over the scaling range
            log_f = np.log(np.clip(fracs[scale_mask], 1e-30, None))
            good = np.isfinite(log_f) & (fracs[scale_mask] > 0)
            if np.sum(good) < 3:
                cost_grid_all[pi, ci] = np.inf
                continue

            slope, _ = np.polyfit(log_a_fit[good], log_f[good], 1)
            d_A = 1.0 - slope
            cost = (d_A - target_dim) ** 2
            cost_grid_all[pi, ci] = cost

            if cost < best_cost:
                best_cost = cost
                best_c = c_val
                best_d = d_A

        c_p_opt[pi] = best_c
        d_A_opt[pi] = best_d

    return c_p_opt, d_A_opt, {
        'c_grid': c_grid,
        'cost_grid': cost_grid_all,
    }


# ──────────────────────────────────────────────────────────────────────
# 3.  Local Hölder exponent h̃(x)
# ──────────────────────────────────────────────────────────────────────

def compute_local_holder(coeffs, scales, sf_result, c_p,
                         valid_ranges=None,
                         log2_a_min=None, log2_a_max=None,
                         method='finest_active'):
    """Compute local Hölder exponent h̃(x) at each spatial position.

    Two methods are available:

    method='finest_active' (default)
        For each position x, find the finest scale a_min(x) at which
        x is still active for any p.  The local exponent is estimated
        from the scaling of the wavelet coefficient at x:
            h̃(x) = lim_{a→0} ln|T_ψ(x,a)| / ln(a)
        approximated by regression over the scaling range.

    method='membership_scaling'
        For each position x and moment p, define the indicator
        I_p(x, a) = 1 if x ∈ A_p(a), 0 otherwise.  Compute the
        fraction of scales in [a_min, a_max] where x is active:
            F_p(x) = (1/N_a) Σ_a I_p(x, a)
        Then h̃(x) is the p-value at which F_p(x) drops below a
        threshold (say 0.5), mapped back to h through the
        Legendre transform h(p) = ζ'(p).

    Parameters
    ----------
    coeffs : ndarray (n_scales, n_positions)
    scales : 1D array
    sf_result : dict from compute_structure_functions
    c_p : 1D array or float
        Threshold constants (one per p, or scalar).
    valid_ranges : list of (int, int), optional
    log2_a_min, log2_a_max : float, optional
        Scale range for regression.
    method : str
        'finest_active' or 'membership_scaling'.

    Returns
    -------
    h_local : 1D array (n_positions,)
        Local Hölder exponent at each position.
        NaN at positions outside all valid ranges.
    info : dict with diagnostic arrays:
        'h_regression' : (n_positions,) pointwise log-log slopes
        'r_squared'    : (n_positions,) R² of pointwise regressions
        'n_scales_used': (n_positions,) number of scales in regression
    """
    n_scales, n_pos = coeffs.shape
    log2_a = np.log2(scales)
    log_a = np.log(scales)

    if log2_a_min is None:
        log2_a_min = log2_a[0]
    if log2_a_max is None:
        log2_a_max = log2_a[-1]
    scale_mask = (log2_a >= log2_a_min) & (log2_a <= log2_a_max)
    scale_idx = np.where(scale_mask)[0]

    abs_coeffs = np.abs(coeffs)

    h_local = np.full(n_pos, np.nan)
    r_squared = np.full(n_pos, np.nan)
    n_used = np.zeros(n_pos, dtype=int)

    if method == 'finest_active':
        # Pointwise log-log regression of |T_ψ(x,a)| vs a
        # over the scaling range, restricted to valid positions
        for xi in range(n_pos):
            # Check if this position is valid at enough scales
            vals = abs_coeffs[scale_idx, xi]

            # Apply valid_ranges: only use scales where xi is valid
            valid_at_x = np.ones(len(scale_idx), dtype=bool)
            if valid_ranges is not None:
                for j, si in enumerate(scale_idx):
                    fp, lp = valid_ranges[si]
                    if xi < fp or xi > lp:
                        valid_at_x[j] = False

            good = valid_at_x & (vals > 0)
            n_good = np.sum(good)
            n_used[xi] = n_good

            if n_good < 3:
                continue

            x_fit = log_a[scale_idx[good]]
            y_fit = np.log(vals[good])

            # Linear regression
            coeffs_fit = np.polyfit(x_fit, y_fit, 1)
            slope = coeffs_fit[0]
            h_local[xi] = slope

            # R²
            y_pred = np.polyval(coeffs_fit, x_fit)
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            if ss_tot > 0:
                r_squared[xi] = 1.0 - ss_res / ss_tot

    elif method == 'membership_scaling':
        # Active-set membership approach
        # First compute active sets for all p
        active_result = compute_active_sets(
            coeffs, scales, sf_result, c_p, valid_ranges)
        masks = active_result['masks']  # (n_p, n_scales, n_pos)
        p_array = sf_result['p_list']

        # For each position, find the critical p where membership
        # drops below 50% of scales in range
        for xi in range(n_pos):
            # Check validity
            if valid_ranges is not None:
                n_valid_scales = 0
                for si in scale_idx:
                    fp, lp = valid_ranges[si]
                    if fp <= xi <= lp:
                        n_valid_scales += 1
                if n_valid_scales < 3:
                    continue
            else:
                n_valid_scales = len(scale_idx)

            # Fraction of scales where x is active, for each p
            frac_p = np.zeros(len(p_array))
            for pi in range(len(p_array)):
                count = 0
                for si in scale_idx:
                    if valid_ranges is not None:
                        fp, lp = valid_ranges[si]
                        if xi < fp or xi > lp:
                            continue
                    if masks[pi, si, xi]:
                        count += 1
                frac_p[pi] = count / max(n_valid_scales, 1)

            # Find the largest p where frac_p > 0.5
            # This corresponds to the strongest singularity that
            # "captures" this point
            active_p = p_array[frac_p > 0.5]
            if len(active_p) > 0:
                p_crit = active_p[-1]  # largest p
                # Map p → h via ζ'(p) ≈ Δζ/Δp
                pi_crit = np.searchsorted(p_array, p_crit)
                zeta = sf_result['zeta_p']
                if pi_crit > 0 and pi_crit < len(zeta):
                    dp = p_array[pi_crit] - p_array[pi_crit - 1]
                    dz = zeta[pi_crit] - zeta[pi_crit - 1]
                    h_local[xi] = dz / dp if dp != 0 else np.nan
                elif pi_crit == 0 and len(zeta) > 1:
                    dp = p_array[1] - p_array[0]
                    dz = zeta[1] - zeta[0]
                    h_local[xi] = dz / dp if dp != 0 else np.nan

            n_used[xi] = n_valid_scales

    else:
        raise ValueError(f"method must be 'finest_active' or "
                         f"'membership_scaling', got {method!r}")

    return h_local, {
        'h_regression': h_local.copy(),
        'r_squared': r_squared,
        'n_scales_used': n_used,
    }


# ──────────────────────────────────────────────────────────────────────
# 4.  Consistency check: D(h̃) histogram vs global D(h)
# ──────────────────────────────────────────────────────────────────────

def holder_histogram(h_local, n_bins=50, h_range=None):
    """Compute the histogram of local Hölder exponents.

    The histogram approximates D(h) when properly normalized:
    the number of points with exponent in [h, h+dh] scales as
    N(h) ~ ε^{-D(h)} for resolution ε → 0.

    For a single resolution (our case), the histogram shape
    should qualitatively match the WTMM D(h) spectrum.

    Parameters
    ----------
    h_local : 1D array
        Local Hölder exponents from compute_local_holder.
    n_bins : int
        Number of histogram bins.
    h_range : tuple (float, float), optional
        Range of h values.  If None, auto-detected.

    Returns
    -------
    h_centers : 1D array, bin centers
    counts : 1D array, normalized histogram values
    """
    valid = np.isfinite(h_local)
    h_valid = h_local[valid]

    if len(h_valid) == 0:
        return np.array([]), np.array([])

    if h_range is None:
        h_range = (np.percentile(h_valid, 1), np.percentile(h_valid, 99))

    counts, edges = np.histogram(h_valid, bins=n_bins, range=h_range)
    h_centers = 0.5 * (edges[:-1] + edges[1:])

    # Normalize to peak = 1 (D(h_0) = d = 1 in 1D)
    if counts.max() > 0:
        counts = counts / counts.max()

    return h_centers, counts


# ──────────────────────────────────────────────────────────────────────
# 5.  Plotting utilities
# ──────────────────────────────────────────────────────────────────────

def plot_structure_functions(sf_result, log2_a_min=None, log2_a_max=None,
                             ax=None, title=None):
    """Plot S_p(a) vs a on log-log axes with fitted scaling lines.

    Parameters
    ----------
    sf_result : dict from compute_structure_functions
    log2_a_min, log2_a_max : float, optional
        Shaded scaling range.
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    log2_a = np.log2(sf_result['scales'])
    p_array = sf_result['p_list']
    S_p = sf_result['S_p']

    colors = plt.colormaps['viridis'](np.linspace(0.1, 0.9, len(p_array)))

    for pi, p in enumerate(p_array):
        ax.semilogy(log2_a, S_p[pi], '-o', color=colors[pi],
                    markersize=3, label=f'p={p:.1f}')

    if log2_a_min is not None and log2_a_max is not None:
        ax.axvspan(log2_a_min, log2_a_max, alpha=0.1, color='grey',
                   label='scaling range')

    ax.set_xlabel(r'$\log_2 a$')
    ax.set_ylabel(r'$S_p(a)$')
    ax.set_title(title or 'Structure functions')
    ax.legend(fontsize=7, ncol=2)

    return ax


def plot_zeta(sf_result, zeta_wtmm=None, ax=None, title=None):
    """Plot ζ(p) from structure functions, optionally overlaying WTMM τ(q).

    Parameters
    ----------
    sf_result : dict from compute_structure_functions
    zeta_wtmm : tuple (q_array, tau_q), optional
        WTMM-derived τ(q) for comparison.  In theory ζ(p) = τ(p)
        (same exponents, different estimators).
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    p_array = sf_result['p_list']
    zeta_p = sf_result['zeta_p']

    ax.plot(p_array, zeta_p, 'ko-', markersize=4, label=r'$\zeta(p)$ (struct. fn.)')

    if zeta_wtmm is not None:
        q_wt, tau_wt = zeta_wtmm
        ax.plot(q_wt, tau_wt, 'r--', linewidth=1.5, label=r'$\tau(q)$ (WTMM)')

    ax.set_xlabel(r'$p$')
    ax.set_ylabel(r'$\zeta(p)$')
    ax.set_title(title or r'$\zeta(p)$ vs $\tau(q)$ comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_local_holder(h_local, x_coords=None, ax=None,
                      cmap='RdYlBu_r', vmin=None, vmax=None,
                      title=None, xlabel=None, r_squared=None,
                      r2_threshold=0.7):
    """Plot the local Hölder exponent profile h̃(x) vs position.

    Parameters
    ----------
    h_local : 1D array (n_positions,)
    x_coords : 1D array, optional
        Physical coordinates (e.g., depths).  If None, uses indices.
    ax : matplotlib Axes, optional
    cmap : str
    vmin, vmax : float, optional
        Color limits.  If None, auto-detected from data.
    title : str, optional
    xlabel : str, optional
    r_squared : 1D array, optional
        R² of pointwise regressions.  If given, positions with
        R² < r2_threshold are shown as semi-transparent.
    r2_threshold : float
        Minimum R² to consider a local exponent reliable.

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))

    valid = np.isfinite(h_local)
    if x_coords is None:
        x_coords = np.arange(len(h_local), dtype=float)

    if vmin is None:
        vmin = np.nanpercentile(h_local[valid], 2) if valid.any() else 0
    if vmax is None:
        vmax = np.nanpercentile(h_local[valid], 98) if valid.any() else 1

    # Create colored line segments
    points = np.column_stack([x_coords, h_local])
    segments = np.array([points[i:i+2] for i in range(len(points) - 1)])

    # Color by h value (midpoint of each segment)
    h_mid = 0.5 * (h_local[:-1] + h_local[1:])

    # Filter out segments with NaN
    seg_valid = np.isfinite(h_mid)

    if r_squared is not None:
        # Alpha based on R²
        r2_mid = 0.5 * (r_squared[:-1] + r_squared[1:])
        alphas = np.where(r2_mid >= r2_threshold, 1.0, 0.2)
    else:
        alphas = np.ones(len(segments))

    lc = LineCollection(segments[seg_valid], cmap=cmap,
                        norm=plt.Normalize(vmin=vmin, vmax=vmax))
    lc.set_array(h_mid[seg_valid])
    lc.set_linewidth(1.5)

    # Apply alpha per segment
    colors = plt.colormaps[cmap](
        (h_mid[seg_valid] - vmin) / (vmax - vmin + 1e-30))
    colors[:, 3] = alphas[seg_valid]
    lc.set_colors(colors)

    ax.add_collection(lc)
    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_ylim(vmin - 0.1 * (vmax - vmin), vmax + 0.1 * (vmax - vmin))

    ax.set_xlabel(xlabel or 'Position')
    ax.set_ylabel(r'$\tilde{h}(x)$')
    ax.set_title(title or 'Local Hölder exponent')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=r'$\tilde{h}$', shrink=0.8)

    return ax


def plot_holder_vs_spectrum(h_local, h_q_wtmm, D_q_wtmm,
                            n_bins=50, ax=None, title=None):
    """Compare histogram of h̃(x) with the global D(h) spectrum.

    Parameters
    ----------
    h_local : 1D array
    h_q_wtmm : 1D array, h(q) from WTMM canonical method
    D_q_wtmm : 1D array, D(q) from WTMM canonical method
    n_bins : int
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    # Histogram of local exponents
    h_range = (min(np.nanmin(h_local), np.nanmin(h_q_wtmm)),
               max(np.nanmax(h_local), np.nanmax(h_q_wtmm)))
    h_centers, hist_vals = holder_histogram(h_local, n_bins, h_range)

    if len(h_centers) > 0:
        ax.fill_between(h_centers, 0, hist_vals, alpha=0.3,
                        color='steelblue', label=r'$\tilde{h}(x)$ histogram')
        ax.plot(h_centers, hist_vals, 'b-', linewidth=1)

    # Global D(h) spectrum
    valid = np.isfinite(h_q_wtmm) & np.isfinite(D_q_wtmm)
    ax.plot(h_q_wtmm[valid], D_q_wtmm[valid], 'ro-', markersize=4,
            linewidth=1.5, label='WTMM D(h)')

    ax.set_xlabel(r'$h$')
    ax.set_ylabel(r'$D(h)$ / normalized count')
    ax.set_title(title or r'Local $\tilde{h}$ vs global $D(h)$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
