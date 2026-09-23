"""Multifractal spectrum computation and theoretical references."""

import numpy as np

from .partition import pf_get_T, pf_get_H, pf_get_D


def _resolve_c_psi(wavelet=None, c_psi=None):
    """Get wavelet stretch factor from name or explicit value."""
    if c_psi is not None:
        return c_psi
    if wavelet is not None:
        from .wavelets import WAVELETS
        return WAVELETS[wavelet]['fact']
    return 1.0  # fallback: no wavelet correction


def _add_physical_sublabels(ax, dx, c_psi, units, axis='x'):
    """Add grey physical-unit text below (x) or left of (y) each tick."""
    scale = dx * c_psi
    if axis == 'x':
        ticks = ax.get_xticks()
        for t in ticks:
            phys = (2.0 ** t) * scale
            ax.annotate(f'{phys:.3g} {units}', xy=(t, 0),
                        xycoords=('data', 'axes fraction'),
                        xytext=(0, -22), textcoords='offset points',
                        ha='center', va='top', fontsize=7, color='grey',
                        annotation_clip=False)
    else:
        ticks = ax.get_yticks()
        for t in ticks:
            phys = (2.0 ** t) * scale
            ax.annotate(f'{phys:.3g} {units}', xy=(0, t),
                        xycoords=('axes fraction', 'data'),
                        xytext=(-50, 0), textcoords='offset points',
                        ha='right', va='center', fontsize=7, color='grey',
                        annotation_clip=False)


def _add_physical_scale_secondary(ax, dx, c_psi, units, axis='y'):
    """Add secondary axis on right (y) or top (x) showing physical units.

    Uses a twinned axis with ticks at the same octave positions as the primary
    axis, but labeled with the physical scale value at each octave.
    """
    scale = dx * c_psi
    if axis == 'y':
        sec = ax.twinx()
        # Match the primary y limits and tick at same octave integers
        y0, y1 = ax.get_ylim()
        sec.set_ylim(y0, y1)
        octave_ticks = list(range(int(np.ceil(min(y0, y1))),
                                  int(np.floor(max(y0, y1))) + 1))
        sec.set_yticks(octave_ticks)
        sec.set_yticklabels([f'{(2.0**t) * scale:.1f}' for t in octave_ticks])
        sec.set_ylabel(f'Scale ({units})', color='grey')
        sec.tick_params(axis='y', labelcolor='grey')
    else:
        sec = ax.twiny()
        x0, x1 = ax.get_xlim()
        sec.set_xlim(x0, x1)
        octave_ticks = list(range(int(np.ceil(min(x0, x1))),
                                  int(np.floor(max(x0, x1))) + 1))
        sec.set_xticks(octave_ticks)
        sec.set_xticklabels([f'{(2.0**t) * scale:.1f}' for t in octave_ticks])
        sec.set_xlabel(f'Scale ({units})', color='grey')
        sec.tick_params(axis='x', labelcolor='grey')
    return sec


def _fit_slope(x, y):
    """Fit a line y = slope*x + intercept, return (slope, stderr_slope, R2).

    Uses np.polyfit with covariance to get the standard error on the slope.
    """
    n = len(x)
    if n < 3:
        # With only 2 points the fit is perfect, no meaningful error
        slope, intercept = np.polyfit(x, y, 1)
        return slope, 0.0, 1.0

    # polyfit with cov=True returns coefficients and covariance matrix
    coeffs, cov = np.polyfit(x, y, 1, cov=True)
    slope = coeffs[0]
    stderr_slope = np.sqrt(cov[0, 0])

    # R^2
    y_pred = slope * x + coeffs[1]
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return slope, stderr_slope, r2


def compute_spectra(pf, log2_a_min, log2_a_max, mode='extensive',
                    method='canonical', L_ref=None):
    """Compute tau(q), h(q), D(h) spectra via linear regression of slopes.

    Parameters
    ----------
    pf : dict from compute_partition_function
    log2_a_min, log2_a_max : float, scale range for regression (in log2).
        If L_ref is set, these are in normalized units: log2(a/L_ref).
    mode : str, 'extensive' (single signal) or 'intensive' (ensemble)
    method : str
        'canonical' — h(q) and D(q) from Boltzmann-weighted thermal averages
            (slopes of H(q,a) and D(q,a)), tau(q) = q*h(q) - D(q).
            Preserves non-convex D(h) and phase transition structure.
            No normalization needed; q=1 is well-defined.
        'microcanonical' (or 'legendre') — tau(q) from slopes of log2 Z(q,a),
            then h(q) = d tau/dq and D(h) = q*h - tau via Legendre transform.
            Gives convex hull of D(h), smooths over phase transitions.
            Automatically normalizes by tau(1) so Renyi dimensions D_q =
            tau/(q-1) are smooth at q=1. True values are corrected back:
            tau_q = tau_norm + tau(1), D_q = D_norm - tau(1).
    L_ref : float, optional
        Reference length/time scale for normalizing the scale axis, e.g.
        the integral scale L_w in turbulence or a characteristic period in
        geophysics. When set, log2_a_min/max are interpreted as log2(a/L_ref)
        and the output includes 'log2_a_norm' for plotting. Slopes are
        unaffected (dividing a by a constant shifts log2(a) by a constant).

    Returns
    -------
    spectra : dict with keys:
        'tau_q'     : array, scaling exponents
        'h_q'       : array, Holder exponents
        'D_q'       : array, fractal dimensions
        'q_list'    : array, q values
        'tau_1'     : float, tau(1)
        'method'    : str
        'h_err'     : array, standard error on h(q) from regression
        'D_err'     : array, standard error on D(q) from regression
        'tau_err'   : array, standard error on tau(q) from regression
        'h_R2'      : array, R^2 of h(q,a) vs log2(a) fit
        'D_R2'      : array, R^2 of D(q,a) vs log2(a) fit
        'tau_R2'    : array, R^2 of T(q,a) vs log2(a) fit
        'log2_a_norm' : array, normalized scale axis (if L_ref is set)
        'L_ref'     : float or None
        For microcanonical only:
            'tau_q_norm' : array, normalized tau (tau_norm(1) = 0)
            'D_q_norm'   : array, normalized D (smooth Renyi dimensions)
    """
    log2_a = pf['log2_a']
    n_voice = pf['n_voice']
    a_min = pf['a_min']

    log2_a0 = np.log2(a_min)
    dx = 1.0 / n_voice

    # If L_ref is set, convert normalized bounds to absolute log2(a)
    log2_Lref = np.log2(L_ref) if L_ref is not None else 0.0
    abs_log2_min = log2_a_min + log2_Lref
    abs_log2_max = log2_a_max + log2_Lref

    idx_min = int(round((abs_log2_min - log2_a0) / dx))   # round, not truncate: log2(2^(k/nv)) can
    idx_max = int(round((abs_log2_max - log2_a0) / dx))   # land at k-eps -> int() would floor to k-1

    idx_min = max(idx_min, 0)
    idx_max = min(idx_max, pf['index_max'])

    valid_idx = np.arange(idx_min, idx_max + 1)

    if len(valid_idx) < 2:
        raise ValueError('Not enough scales in range for regression')

    x = log2_a0 + dx * valid_idx
    q_array = pf['q_list']
    n_q = len(q_array)

    # Accept 'legendre' as alias for 'microcanonical'
    if method == 'legendre':
        method = 'microcanonical'

    if method == 'canonical':
        h_q = np.full(n_q, np.nan)
        d_q = np.full(n_q, np.nan)
        h_err = np.full(n_q, np.nan)
        d_err = np.full(n_q, np.nan)
        h_r2 = np.full(n_q, np.nan)
        d_r2 = np.full(n_q, np.nan)
        tau_err = np.full(n_q, np.nan)
        tau_r2 = np.full(n_q, np.nan)

        for i in range(n_q):
            q = q_array[i]

            # H(q,a) — direct slope fit
            y_H = pf_get_H(pf, i, mode)[valid_idx]
            if np.all(np.isfinite(y_H)):
                slope_H, se_H, r2_H = _fit_slope(x, y_H)
                h_q[i] = slope_H
                h_err[i] = se_H
                h_r2[i] = r2_H

            # D(q,a) = q*H(q,a) - T(q,a) — direct slope fit
            y_D = pf_get_D(pf, i, q, mode)[valid_idx]
            if np.all(np.isfinite(y_D)):
                slope_D, se_D, r2_D = _fit_slope(x, y_D)
                d_q[i] = slope_D
                d_err[i] = se_D
                d_r2[i] = r2_D

            # T(q,a) — fit for tau consistency check and error
            y_T = pf_get_T(pf, i, mode)[valid_idx]
            if np.all(np.isfinite(y_T)):
                slope_T, se_T, r2_T = _fit_slope(x, y_T)
                tau_err[i] = se_T
                tau_r2[i] = r2_T

        tau_q = q_array * h_q - d_q

        # Get tau(1) from the spectra
        tau_1 = np.nan
        q1_idx = np.where(np.isclose(q_array, 1.0))[0]
        if len(q1_idx) > 0:
            tau_1 = tau_q[q1_idx[0]]
        else:
            finite = np.isfinite(tau_q)
            if finite.sum() >= 2:
                tau_1 = np.interp(1.0, q_array[finite], tau_q[finite])

        result = {
            'tau_q': tau_q,
            'h_q': h_q,
            'D_q': d_q,
            'q_list': q_array,
            'tau_1': tau_1,
            'method': 'canonical',
            'h_err': h_err,
            'D_err': d_err,
            'tau_err': tau_err,
            'h_R2': h_r2,
            'D_R2': d_r2,
            'tau_R2': tau_r2,
            'L_ref': L_ref,
        }
        if L_ref is not None:
            result['log2_a_norm'] = log2_a - log2_Lref
        return result

    elif method == 'microcanonical':
        # [WIP] Legendre method — under development
        # Needs: proper R² metric for shallow slopes, corrected vs uncorrected
        # tau normalization, and D(q) from T(q) validation.
        raise NotImplementedError(
            "Legendre/microcanonical method is under development. "
            "Use method='canonical' for now.")

    else:
        raise ValueError(f"method must be 'canonical' or 'microcanonical', "
                         f"got {method!r}")


def plot_spectra(spectra, label='', theory_q=None, theory_tau=None,
                 theory_h=None, theory_D=None, show_errors=True,
                 tangent_q=None, fig=None, axes=None):
    """Plot tau(q), h(q), and D(h) singularity spectrum.

    Parameters
    ----------
    spectra : dict from compute_spectra
    label : str, title label
    theory_q, theory_tau, theory_h, theory_D : arrays, optional
        Theoretical curves to overlay.
    show_errors : bool
        If True, show error bars from regression standard errors.
    tangent_q : float or list of float, optional
        Draw tangent line(s) to D(h) at the given q value(s).
        The tangent at q has slope q and passes through (h(q), D(q)).
        Useful for visualizing generalized dimensions (e.g. q=0, 1, 2).
    fig, axes : matplotlib figure and axes (length-3 array), optional
        If None, creates a new figure.

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_ in axes:
        for spine in ax_.spines.values():
            spine.set_linewidth(0.5)

    q = spectra['q_list']
    tau_q = spectra['tau_q']
    h_q = spectra['h_q']
    D_q = spectra['D_q']

    has_theory = theory_q is not None and theory_tau is not None
    is_monofractal = (has_theory and theory_h is not None
                      and len(theory_h) == 1)

    tau_err = spectra.get('tau_err')
    h_err = spectra.get('h_err')
    D_err = spectra.get('D_err')

    # 1. tau(q)
    ax = axes[0]
    if has_theory:
        ax.plot(theory_q, theory_tau, 'k-', label='Theory', linewidth=1)
    if show_errors and tau_err is not None and np.any(np.isfinite(tau_err)):
        ax.errorbar(q, tau_q, yerr=tau_err, fmt='bo', markersize=6,
                    capsize=2, label='WTMM')
    else:
        ax.plot(q, tau_q, 'bo', markersize=6, label='WTMM')
    ax.set_xlabel('q')
    ax.set_ylabel('\u03c4(q)')
    ax.set_title(f'\u03c4(q) \u2014 {label}' if label else '\u03c4(q)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. h(q)
    ax = axes[1]
    if has_theory and not is_monofractal:
        ax.plot(theory_q, theory_h, 'k-', label='Theory', linewidth=1)
    elif is_monofractal:
        ax.axhline(theory_h[0], color='k', ls='--', lw=1,
                   label=f'Theory h*={theory_h[0]:.3f}')
    if show_errors and h_err is not None and np.any(np.isfinite(h_err)):
        ax.errorbar(q, h_q, yerr=h_err, fmt='ro', markersize=6,
                    capsize=2, label='WTMM')
    else:
        ax.plot(q, h_q, 'ro', markersize=6, label='WTMM')
    ax.set_xlabel('q')
    ax.set_ylabel('h(q)')
    ax.set_title(f'h(q) \u2014 {label}' if label else 'h(q)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if has_theory and not is_monofractal:
        h_pad = 0.15 * (theory_h.max() - theory_h.min())
        ax.set_ylim(theory_h.min() - h_pad, theory_h.max() + h_pad)

    # 3. D(h) singularity spectrum
    ax = axes[2]
    if has_theory and not is_monofractal:
        mask = theory_D > -0.5
        ax.plot(theory_h[mask], theory_D[mask], 'k-', label='Theory',
                linewidth=1)
    elif is_monofractal:
        ax.plot(theory_h[0], theory_D[0], 'k*', markersize=15,
                label=f'Theory (h*={theory_h[0]:.3f})')
    if show_errors and D_err is not None and np.any(np.isfinite(D_err)):
        ax.errorbar(h_q, D_q, xerr=h_err, yerr=D_err, fmt='ro',
                    markersize=6, capsize=2, linestyle='none', label='WTMM')
    else:
        ax.plot(h_q, D_q, 'ro', markersize=6, linestyle='none', label='WTMM')
    ax.set_xlabel('h')
    ax.set_ylabel('D(h)')
    ax.set_title(f'D(h) \u2014 {label}' if label else 'D(h)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    # Ensure origin (0,0) is visible
    xlim = list(ax.get_xlim())
    ylim = list(ax.get_ylim())
    xlim[0] = min(xlim[0], -0.05)
    ylim[0] = min(ylim[0], -0.05)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)

    # Tangent lines to D(h) at specified q values
    if tangent_q is not None:
        if np.isscalar(tangent_q):
            tangent_q = [tangent_q]
        tangent_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#e377c2']
        for ti, tq in enumerate(tangent_q):
            # Interpolate h(q) and D(q) at the requested q
            q_arr = spectra['q_list']
            h_at_q = np.interp(tq, q_arr, h_q)
            D_at_q = np.interp(tq, q_arr, D_q)
            # Tangent: D = D(q) + q * (h - h(q))  =>  D = q*h + (D(q) - q*h(q))
            # which equals D = q*h - tau(q)
            xlim_cur = ax.get_xlim()
            h_line = np.linspace(xlim_cur[0], xlim_cur[1], 100)
            D_line = D_at_q + tq * (h_line - h_at_q)
            tc = tangent_colors[ti % len(tangent_colors)]
            ax.plot(h_line, D_line, '--', color=tc, lw=1,
                    label=f'q={tq:g}: h={h_at_q:.3f}, D={D_at_q:.3f}')
            ax.plot(h_at_q, D_at_q, 's', color=tc, markersize=8, zorder=5)
        ax.legend(fontsize=7)

    plt.tight_layout()
    return fig, axes


def print_spectra_comparison(spectra, label='', theory_q=None, theory_tau=None,
                             theory_h=None, theory_D=None):
    """Print comparison table of WTMM vs theory."""
    has_theory = theory_q is not None and theory_tau is not None
    is_monofractal = (has_theory and theory_h is not None
                      and len(theory_h) == 1)

    if has_theory and not is_monofractal:
        tau_theory_q = np.interp(spectra['q_list'], theory_q, theory_tau)
        h_theory_q = np.interp(spectra['q_list'], theory_q, theory_h)
        print(f'\n{label}: theory vs WTMM comparison')
        print(f'h range: [{h_theory_q.min():.3f}, {h_theory_q.max():.3f}]')
        print(f'\n{"q":>5s}\t {"tau_WTMM":>8s}\t {"tau_theory":>8s}\t '
              f'{"diff":>7s}\t\t {"h_WTMM":>8s}\t {"h_theory":>8s}')
        for i, q_val in enumerate(spectra['q_list']):
            print(f'{q_val:5.1f}\t {spectra["tau_q"][i]:8.4f}\t '
                  f'{tau_theory_q[i]:8.4f}\t '
                  f'{spectra["tau_q"][i] - tau_theory_q[i]:+.4f}\t '
                  f'{spectra["h_q"][i]:8.4f}\t {h_theory_q[i]:8.4f}')
    elif is_monofractal:
        print(f'\n{label}: monofractal h* = {theory_h[0]:.4f}')
        print(f'WTMM h(q) range: [{spectra["h_q"].min():.4f}, '
              f'{spectra["h_q"].max():.4f}]')
        print(f'WTMM h(q) mean:  {spectra["h_q"].mean():.4f}')
    else:
        print(f'\n{label}: no theoretical spectrum available')
        print(f'WTMM h(q) range: [{spectra["h_q"].min():.4f}, '
              f'{spectra["h_q"].max():.4f}]')


def plot_partition_functions(pf, log2_a_min, log2_a_max, mode='extensive',
                             theory_tau=None, theory_h=None, theory_D=None,
                             theory_q=None, normalize=False,
                             legend='sparse',
                             dx=None, units='', wavelet=None, c_psi=None,
                             fig=None, axes=None):
    """Plot T(q,a), H(q,a), D(q,a) partition functions vs log2(a).

    Parameters
    ----------
    pf : dict from compute_partition_function
    log2_a_min, log2_a_max : float
        Regression range (shown as dashed vertical lines).
    mode : str
        'extensive' or 'intensive'.
    theory_tau, theory_h, theory_D : arrays, optional
        Theoretical slopes for each q in pf['q_list']. Interpolated from
        theory_q if lengths don't match.
    theory_q : array, optional
        q values corresponding to theory arrays (for interpolation).
    normalize : bool
        If True, shift each curve so it passes through zero at the midpoint
        of the regression range. Theoretical lines become zero-intercept
        lines with known slope, enabling direct visual comparison.
    legend : str
        'all' — label every q value in the legend.
        'sparse' (default) — only label q_min, q=0 (or nearest), and q_max.
        'none' — no legend, just a colorbar.
    fig, axes : matplotlib figure and axes (length-3 array), optional

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_ in axes:
        for spine in ax_.spines.values():
            spine.set_linewidth(0.5)

    log2_a = pf['log2_a']
    valid = slice(0, pf['index_max'] + 1)
    q_list = pf['q_list']
    n_q = len(q_list)
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_q))

    # Determine which q indices get legend labels
    if legend == 'sparse' and n_q > 3:
        i_zero = int(np.argmin(np.abs(q_list)))
        label_set = {0, i_zero, n_q - 1}
    elif legend == 'none':
        label_set = set()
    else:
        label_set = set(range(n_q))

    def _qlabel(i, q):
        if i not in label_set:
            return '_nolegend_'
        # Clean up float artifacts (e.g. 3e-15 → 0)
        q_disp = round(q, 6)
        return f'q={q_disp:g}'

    # Reference scale for normalization (midpoint of regression range)
    log2_a_ref = 0.5 * (log2_a_min + log2_a_max)

    # Interpolate theory to match pf's q_list if needed
    has_theory = theory_tau is not None
    if has_theory and theory_q is not None:
        theory_tau = np.asarray(theory_tau)
        # Monofractal: theory_h/D may be single-element (Dirac delta),
        # not a function of q. Only interp if lengths match theory_q.
        if len(theory_tau) == len(theory_q):
            tau_th = np.interp(q_list, theory_q, theory_tau)
        else:
            tau_th = None
        if theory_h is not None and len(theory_h) == len(theory_q):
            h_th = np.interp(q_list, theory_q, theory_h)
        else:
            h_th = None
        if theory_D is not None and len(theory_D) == len(theory_q):
            D_th = np.interp(q_list, theory_q, theory_D)
        else:
            D_th = None
    elif has_theory:
        tau_th = np.asarray(theory_tau)
        h_th = np.asarray(theory_h) if theory_h is not None else None
        D_th = np.asarray(theory_D) if theory_D is not None else None
    else:
        tau_th = h_th = D_th = None

    # Monofractal: if theory_h/D are single-element, expand to constant arrays
    is_monofractal = (theory_h is not None and len(np.atleast_1d(theory_h)) == 1)
    if is_monofractal:
        h_star = float(np.atleast_1d(theory_h)[0])
        D_f = float(np.atleast_1d(theory_D)[0]) if theory_D is not None else 1.0
        # h(q) = h* for all q, D(q) = D_f for all q
        h_th = np.full(n_q, h_star)
        D_th = np.full(n_q, D_f)
        # tau(q) = q*h* - D_f
        if tau_th is None:
            tau_th = q_list * h_star - D_f

    x = log2_a[valid]

    # Compute reference values at a_ref for normalization
    # (interpolate each curve at log2_a_ref)
    def _ref_val(y_arr):
        """Interpolate y at log2_a_ref."""
        return np.interp(log2_a_ref, x, y_arr)

    # T(q,a)
    ax = axes[0]
    for i, q in enumerate(q_list):
        y = pf_get_T(pf, i, mode)[valid]
        offset = _ref_val(y) if normalize else 0.0
        ax.plot(x, y - offset, 'o-', color=colors[i], markersize=3,
                label=_qlabel(i, q))
        if tau_th is not None:
            y_th = tau_th[i] * (x - log2_a_ref) if normalize else \
                   tau_th[i] * (x - log2_a_ref) + _ref_val(y)
            ax.plot(x, y_th, '--', color=colors[i], lw=1, alpha=0.6)
    ax.set_xlabel('log\u2082(a)')
    ylabel = 'T(q,a) \u2013 T(q,a\u2080)' if normalize else 'T(q,a) = log\u2082 Z(q,a)'
    ax.set_ylabel(ylabel)
    ax.set_title('T(q,a)' + (' [normalized]' if normalize else ''))
    ax.legend(fontsize=7, ncol=2)
    ax.axvline(log2_a_min, color='gray', ls='--', lw=0.5)
    ax.axvline(log2_a_max, color='gray', ls='--', lw=0.5)

    # H(q,a)
    ax = axes[1]
    for i, q in enumerate(q_list):
        y = pf_get_H(pf, i, mode)[valid]
        offset = _ref_val(y) if normalize else 0.0
        ax.plot(x, y - offset, 'o-', color=colors[i], markersize=3,
                label=_qlabel(i, q))
        if h_th is not None:
            y_th = h_th[i] * (x - log2_a_ref) if normalize else \
                   h_th[i] * (x - log2_a_ref) + _ref_val(y)
            ax.plot(x, y_th, '--', color=colors[i], lw=1, alpha=0.6)
    ax.set_xlabel('log\u2082(a)')
    ylabel = 'H(q,a) \u2013 H(q,a\u2080)' if normalize else 'H(q,a)'
    ax.set_ylabel(ylabel)
    ax.set_title('H(q,a)' + (' [normalized]' if normalize else ''))
    ax.legend(fontsize=7, ncol=2)
    ax.axvline(log2_a_min, color='gray', ls='--', lw=0.5)
    ax.axvline(log2_a_max, color='gray', ls='--', lw=0.5)

    # D(q,a)
    ax = axes[2]
    for i, q in enumerate(q_list):
        y = pf_get_D(pf, i, q, mode)[valid]
        offset = _ref_val(y) if normalize else 0.0
        ax.plot(x, y - offset, 'o-', color=colors[i], markersize=3,
                label=_qlabel(i, q))
        if D_th is not None:
            y_th = D_th[i] * (x - log2_a_ref) if normalize else \
                   D_th[i] * (x - log2_a_ref) + _ref_val(y)
            ax.plot(x, y_th, '--', color=colors[i], lw=1, alpha=0.6)
    ax.set_xlabel('log\u2082(a)')
    ylabel = 'D(q,a) \u2013 D(q,a\u2080)' if normalize else 'D(q,a)'
    ax.set_ylabel(ylabel)
    ax.set_title('D(q,a) = q\u00b7H \u2013 T' + (' [normalized]' if normalize else ''))
    ax.legend(fontsize=7, ncol=2)
    ax.axvline(log2_a_min, color='gray', ls='--', lw=0.5)
    ax.axvline(log2_a_max, color='gray', ls='--', lw=0.5)

    # Physical unit sub-labels on x-axis of all three panels
    if dx is not None:
        c = _resolve_c_psi(wavelet, c_psi)
        for ax_ in axes:
            _add_physical_sublabels(ax_, dx, c, units, axis='x')

    plt.tight_layout()
    return fig, axes


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

    # Filter out zero-probability parts (avoid 0^q = inf for q < 0)
    mask = p > 0
    p_nz = p[mask]
    r_nz = r[mask]

    tau_measure = np.zeros_like(q_array, dtype=float)
    for j, q in enumerate(q_array):
        def f(tau, _q=q):
            return np.sum(p_nz**_q * r_nz**(-tau)) - 1.0
        tau_measure[j] = brentq(f, -20, 20)

    # Account for integration (+1) and expo normalization
    # h_eff = h_measure + 1 + expo, so tau_eff = tau_measure + (1 + expo) * q
    tau_wtmm = tau_measure + (1.0 + expo) * q_array

    # D(h) via Legendre transform
    h_theory = np.gradient(tau_wtmm, q_array)
    D_theory = q_array * h_theory - tau_wtmm

    return tau_wtmm, h_theory, D_theory
