"""Generalized entropy diagnostics for WTMM partition functions (1D scalar).

Companion module to ``partition.py`` and ``spectra.py``: takes the same
``pf`` dict and exposes Rényi entropy R_q(a), Tsallis entropy T_q(a),
generalized Rényi dimensions D_q (slope of R_q vs log_2 a), the
information dimension D_1 (slope of Shannon entropy), the Hanel–Thurner
(c, d) universality classifier, and a Lesche-stability diagnostic for
shuffle tests.

Why this is essentially free
----------------------------
With the natural normalization p_i(a) = |T_i(a)| / Σ_j |T_j(a)| of the
WTMM measure on extrema, Σ_i p_i(a)^q = Z(q,a) / Z(1,a)^q.  In bits,

    R_q(a) = (1/(1-q)) [T(q,a) − q · T(1,a)]
    T_q(a) = (1 − Σ p_i^q) / (q − 1) = (1 − 2^{(1−q) R_q(a)}) / (q − 1)

so every quantity is a closed-form function of the existing
partition-function output ``pf['sTq']`` (via ``pf_get_T``).  A linear
regression of R_q(a) vs log_2(a) over a chosen scale window yields the
generalized Rényi dimension

    D_q = − d R_q / d log_2 a = (τ(q) − q · τ(1)) / (q − 1).

For q = 1 the (1−q) denominator is degenerate; Shannon entropy is
computed directly from the per-extremum amplitudes via
``shannon_entropy_per_scale``.

Pipeline rigour
---------------
Because every quantity is a function of ``pf``, **all** rigorous
upstream choices propagate automatically:

* **chainmax** — call ``chain_max_wrapper(extrep_ord, coarser_links,
  n_voice=…, a_min=…, expo=…)`` before building the pf and the running
  scale-adaptive supremum is baked in.
* **smooth chains** — call ``chain_delete_all`` first to remove
  unchained / sign-flipping extrema.
* **min_chain_voices** — pass it to ``compute_partition_function`` to
  drop short chains (the same kwarg is accepted there).
* **min wavelet scale** — set ``log2_a_min`` for the slope fit (or use
  ``interactive_entropy`` to drag it).

Hanel–Thurner is a separate axis: it does NOT operate on WTMM
scale-resolved data.  See the docstring of
``hanel_thurner_exponents`` for what to feed it.

References
----------
- Amigó, Balogh & Hernández, Entropy 20, 813 (2018)
  -- generalized entropies review.
- Hanel & Thurner, EPL 93, 20006 (2011) -- (c, d) classification.
- Lesche, J. Stat. Phys. 27, 419 (1982) -- entropy stability axiom.
- Beck & Schögl, "Thermodynamics of Chaotic Systems" (1993),
  ch. 11 for D_q from p_i ~ a^h.
"""

import numpy as np

from .partition import pf_get_T

LN2 = np.log(2.0)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _resolve_q1_index(q_list, atol=1e-6):
    """Find index of q = 1 in q_list. Required for natural normalization.

    The Rényi/Tsallis layer needs τ(1) as the normalization reference so
    that p_i = |T_i| / M_1 is a probability measure.  Without q = 1 in
    the grid we cannot compute the centered entropy — fail loudly rather
    than silently using the unnormalized Z(q, a).
    """
    q_arr = np.asarray(q_list)
    hits = np.where(np.abs(q_arr - 1.0) < atol)[0]
    if len(hits) == 0:
        raise ValueError(
            "q=1 must be present in q_list for entropy normalization. "
            "Add 1.0 to your q grid (the τ(1) Rényi-dimension reference)."
        )
    return int(hits[0])


def _fit_slope(x, y):
    """Linear regression with stderr and R²; lifted from spectra.py."""
    n = len(x)
    if n < 3:
        slope, intercept = np.polyfit(x, y, 1)
        return slope, 0.0, 1.0
    coeffs, cov = np.polyfit(x, y, 1, cov=True)
    slope = coeffs[0]
    se = np.sqrt(cov[0, 0])
    pred = slope * x + coeffs[1]
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, se, r2


# ---------------------------------------------------------------------------
# Per-scale entropy curves
# ---------------------------------------------------------------------------

def renyi_curves(pf, mode='extensive'):
    """Per-scale R_q(a) for every q in ``pf['q_list']``.

    R_q(a) = (T(q,a) − q · T(1,a)) / (1 − q)   (in bits)

    Returns
    -------
    R_qa : ndarray, shape (n_q, n_scales)
        NaN for q within 1e-6 of 1.0 (use ``shannon_entropy_per_scale``).
    log2_a : ndarray, shape (n_scales,)
        Same as ``pf['log2_a']``.
    """
    q_list = np.asarray(pf['q_list'])
    q1_idx = _resolve_q1_index(q_list)
    T1 = pf_get_T(pf, q1_idx, mode)
    n_q = len(q_list)
    n_scales = pf['sTq'].shape[1]
    R = np.full((n_q, n_scales), np.nan)
    for i in range(n_q):
        q = q_list[i]
        if abs(q - 1.0) < 1e-6:
            continue
        Tq = pf_get_T(pf, i, mode)
        R[i] = (Tq - q * T1) / (1.0 - q)
    return R, pf['log2_a']


def tsallis_curves(pf, mode='extensive'):
    """Per-scale Tsallis T_q(a).

    T_q = (1 − Σ p_i^q) / (q − 1) = (1 − 2^{(1-q) R_q}) / (q − 1).
    """
    R, log2_a = renyi_curves(pf, mode)
    q_list = np.asarray(pf['q_list'])
    n_q = len(q_list)
    T = np.full_like(R, np.nan)
    for i in range(n_q):
        q = q_list[i]
        if abs(q - 1.0) < 1e-6:
            continue
        sum_pq = 2.0 ** ((1.0 - q) * R[i])
        T[i] = (1.0 - sum_pq) / (q - 1.0)
    return T, log2_a


def shannon_entropy_per_scale(extrep_ord):
    """Shannon entropy S_1(a) of the natural WTMM measure (bits per scale).

    p_i = |T_i| / Σ_j |T_j| with zeros excluded (matches pf convention).
    Cannot be derived from ``pf`` summary moments alone -- needs the raw
    per-extremum amplitudes.

    Parameters
    ----------
    extrep_ord : list of arrays
        Per-scale ordinate (wavelet coefficient) values.  Same object you
        pass to ``compute_partition_function``; pass it AFTER any
        chain_max / chain_delete_all transforms so chainmax/smoothing is
        inherited.
    """
    n_scales = len(extrep_ord)
    S1 = np.full(n_scales, np.nan)
    for s in range(n_scales):
        absT = np.abs(extrep_ord[s])
        absT = absT[absT > 0]
        if len(absT) == 0:
            continue
        p = absT / absT.sum()
        S1[s] = -np.sum(p * np.log2(p))
    return S1


# ---------------------------------------------------------------------------
# Slope fit -> generalized dimensions
# ---------------------------------------------------------------------------

def compute_entropy_spectra(pf, log2_a_min, log2_a_max, mode='extensive',
                             L_ref=None, extrep_ord=None):
    """Slope fit of R_q(a) vs log_2(a) → generalized Rényi dimensions.

    Mirrors ``compute_spectra`` in spectra.py.  All rigorous pipeline
    choices baked into the pf are inherited.

    Parameters
    ----------
    pf : dict
        From ``compute_partition_function``.  Must contain q = 1.
    log2_a_min, log2_a_max : float
        Regression window in log₂.  If ``L_ref`` is set these are
        normalized as log₂(a / L_ref).
    mode : {'extensive', 'intensive'}
        Same as in ``pf_get_T``.
    L_ref : float, optional
        Reference scale (e.g. integral scale, characteristic period).
    extrep_ord : list of arrays, optional
        If given, also computes Shannon S_1(a) and the information
        dimension D_1 (slope of S_1 vs log₂ a).

    Returns
    -------
    dict with keys
        'q_list'       : (n_q,)
        'R_q'          : R_q at the midpoint scale (level, in bits)
        'T_q'          : Tsallis at midpoint
        'D_q'          : Rényi dimension = − slope of R_q vs log₂ a
        'D_q_err'      : standard error on D_q
        'D_q_R2'       : R² of the R_q vs log₂ a fit
        'log2_a_mid'   : midpoint of regression window (absolute log₂)
        'L_ref'        : reflected back
        'method'       : 'renyi'
        # Only if extrep_ord is given:
        'S_1_curve'    : full Shannon curve over scales
        'S_1_mid'      : Shannon at midpoint
        'D_1', 'D_1_err', 'D_1_R2' : information dimension fit
    """
    log2_a = pf['log2_a']
    n_voice = pf['n_voice']
    a_min = pf['a_min']
    log2_a0 = np.log2(a_min)
    dx = 1.0 / n_voice
    log2_Lref = np.log2(L_ref) if L_ref is not None else 0.0
    abs_log2_min = log2_a_min + log2_Lref
    abs_log2_max = log2_a_max + log2_Lref
    idx_min = max(0, int((abs_log2_min - log2_a0) / dx))
    idx_max = min(int((abs_log2_max - log2_a0) / dx), pf['index_max'])
    valid_idx = np.arange(idx_min, idx_max + 1)
    if len(valid_idx) < 2:
        raise ValueError('Not enough scales in fit range')

    x = log2_a0 + dx * valid_idx
    log2_a_mid = 0.5 * (abs_log2_min + abs_log2_max)

    R_qa, _ = renyi_curves(pf, mode)
    T_qa, _ = tsallis_curves(pf, mode)
    q_list = np.asarray(pf['q_list'])
    n_q = len(q_list)

    R_q_mid = np.full(n_q, np.nan)
    T_q_mid = np.full(n_q, np.nan)
    D_q = np.full(n_q, np.nan)
    D_q_err = np.full(n_q, np.nan)
    D_q_R2 = np.full(n_q, np.nan)

    for i in range(n_q):
        y = R_qa[i, valid_idx]
        if not np.all(np.isfinite(y)):
            continue
        slope, se, r2 = _fit_slope(x, y)
        D_q[i] = -slope
        D_q_err[i] = se
        D_q_R2[i] = r2
        R_q_mid[i] = float(np.interp(log2_a_mid, x, y))
        yT = T_qa[i, valid_idx]
        if np.all(np.isfinite(yT)):
            T_q_mid[i] = float(np.interp(log2_a_mid, x, yT))

    result = {
        'q_list': q_list,
        'R_q': R_q_mid,
        'T_q': T_q_mid,
        'D_q': D_q,
        'D_q_err': D_q_err,
        'D_q_R2': D_q_R2,
        'log2_a_mid': log2_a_mid,
        'L_ref': L_ref,
        'method': 'renyi',
    }

    if extrep_ord is not None:
        S1 = shannon_entropy_per_scale(extrep_ord)
        result['S_1_curve'] = S1
        y = S1[valid_idx]
        if np.all(np.isfinite(y)):
            slope, se, r2 = _fit_slope(x, y)
            result['D_1'] = -slope
            result['D_1_err'] = se
            result['D_1_R2'] = r2
            result['S_1_mid'] = float(np.interp(log2_a_mid, x, y))

    return result


# ---------------------------------------------------------------------------
# Hanel-Thurner (c, d) classifier  --  separate axis
# ---------------------------------------------------------------------------

def hanel_thurner_exponents(W, S, log_base=np.e):
    """Empirical (c, d) fit of S(W) ≈ A · W^c · (log W)^d.

    What this is for
    ----------------
    Hanel-Thurner classifies a *non-extensive* system by how its
    accessible-state entropy scales with system size W.  The (c, d)
    exponents form a 2D universality bin.  This is **orthogonal** to
    WTMM scale resolution — feed it size-vs-entropy of the underlying
    counting set, not pf['sTq'].

    Typical inputs for an EBSD pipeline:

      * Avalanche-size distribution from connected components of
        thresholded ρ_GND.  Compute Shannon (or Rényi) entropy of the
        size histogram for nested subwindows of side W and feed (W, S).
      * Number of distinct visited microstates as a function of
        cumulative observation window length.

    The ordinary Boltzmann-Gibbs case has (c, d) = (1, 0); Tsallis-style
    nonextensive systems have c < 1; stretched-exponential systems have
    (c, d) = (1, d).  See Amigó et al. 2018 §3 for the universality
    table.

    Method
    ------
    Multilinear regression of  log S = log A + c · log W + d · log log W.
    Returns the fit + R² and the residual-vs-W curve so you can
    diagnose departures from the asymptotic regime.

    Parameters
    ----------
    W, S : array_like
        Same length.  Only points with W > 1 and S > 0 contribute.
    log_base : float
        Base of S; only affects A, not (c, d).

    Returns
    -------
    dict with keys 'c', 'd', 'A', 'R2', 'n_points', 'residuals',
    'W_used', 'S_used'.
    """
    W = np.asarray(W, dtype=float)
    S = np.asarray(S, dtype=float)
    if W.shape != S.shape:
        raise ValueError("W and S must have the same shape")
    valid = (W > 1) & (S > 0) & np.isfinite(S) & np.isfinite(W)
    if valid.sum() < 4:
        raise ValueError(
            f"Need at least 4 valid (W>1, S>0) points; got {valid.sum()}")
    Wv, Sv = W[valid], S[valid]
    lW = np.log(Wv)
    lS = np.log(Sv) if log_base == np.e else np.log(Sv) / np.log(log_base)
    llW = np.log(lW)
    A_mat = np.column_stack([np.ones_like(lW), lW, llW])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, lS, rcond=None)
    a, c, d = coeffs
    pred = A_mat @ coeffs
    ss_res = np.sum((lS - pred) ** 2)
    ss_tot = np.sum((lS - lS.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        'c': float(c), 'd': float(d), 'A': float(np.exp(a)),
        'R2': float(r2), 'n_points': int(valid.sum()),
        'residuals': lS - pred,
        'W_used': Wv, 'S_used': Sv,
    }


# ---------------------------------------------------------------------------
# Lesche stability  --  shuffle-test sanity
# ---------------------------------------------------------------------------

def lesche_distance(pf_a, pf_b, mode='extensive'):
    """Per-q, per-scale |R_q^a − R_q^b| between two partition functions.

    Lesche stability is "small change in PDF ⇒ small change in entropy".
    A genuine multifractal field, when shuffled in a way that destroys
    *only* correlations (Kantelhardt route 3), should still yield
    similar R_q(a); a large Lesche distance indicates the spectrum was
    riding on the broad single-point PDF (route 4) rather than on
    long-range correlations.

    Returns
    -------
    delta_R : ndarray, shape (n_q, n_scales)
        Absolute difference in R_q(a) between pf_a and pf_b.
    """
    R_a, _ = renyi_curves(pf_a, mode)
    R_b, _ = renyi_curves(pf_b, mode)
    if R_a.shape != R_b.shape:
        raise ValueError(
            f"pfs must share q_list and scale grid; "
            f"got shapes {R_a.shape} vs {R_b.shape}")
    return np.abs(R_a - R_b)


# ---------------------------------------------------------------------------
# Plotting (non-interactive)
# ---------------------------------------------------------------------------

def plot_entropy_curves(pf, log2_a_min=None, log2_a_max=None, mode='extensive',
                         legend='sparse', fig=None, axes=None):
    """Plot R_q(a) and T_q(a) vs log₂(a). Mirrors ``plot_partition_functions``.

    Vertical guides at ``log2_a_min/max`` mark the regression window if
    given.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax_ in axes:
        for spine in ax_.spines.values():
            spine.set_linewidth(0.5)

    R, log2_a = renyi_curves(pf, mode)
    T, _ = tsallis_curves(pf, mode)
    q_list = np.asarray(pf['q_list'])
    valid = slice(0, pf['index_max'] + 1)
    x = log2_a[valid]
    n_q = len(q_list)
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_q))

    if legend == 'sparse' and n_q > 3:
        i_zero = int(np.argmin(np.abs(q_list)))
        label_set = {0, i_zero, n_q - 1}
    elif legend == 'none':
        label_set = set()
    else:
        label_set = set(range(n_q))

    def _qlab(i, q):
        if i not in label_set:
            return '_nolegend_'
        return f'q={round(q, 6):g}'

    ax = axes[0]
    for i, q in enumerate(q_list):
        ax.plot(x, R[i, valid], 'o-', color=colors[i], markersize=3,
                label=_qlab(i, q))
    ax.set_xlabel('log₂(a)')
    ax.set_ylabel('R_q(a)  [bits]')
    ax.set_title('Rényi entropy R_q(a)')
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for i, q in enumerate(q_list):
        ax.plot(x, T[i, valid], 'o-', color=colors[i], markersize=3,
                label=_qlab(i, q))
    ax.set_xlabel('log₂(a)')
    ax.set_ylabel('T_q(a)')
    ax.set_title('Tsallis entropy T_q(a)')
    ax.legend(fontsize=7, ncol=2)

    if log2_a_min is not None and log2_a_max is not None:
        for ax_ in axes:
            ax_.axvline(log2_a_min, color='gray', ls='--', lw=0.5)
            ax_.axvline(log2_a_max, color='gray', ls='--', lw=0.5)

    plt.tight_layout()
    return fig, axes


def plot_entropy_spectra(spectra, label='', show_errors=True,
                          fig=None, axes=None):
    """Plot R_q vs q, T_q vs q, D_q vs q from ``compute_entropy_spectra``."""
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_ in axes:
        for spine in ax_.spines.values():
            spine.set_linewidth(0.5)

    q = np.asarray(spectra['q_list'])
    R_q = spectra['R_q']
    T_q = spectra['T_q']
    D_q = spectra['D_q']
    D_err = spectra.get('D_q_err')

    ax = axes[0]
    ax.plot(q, R_q, 'bo-', markersize=5)
    ax.set_xlabel('q')
    ax.set_ylabel('R_q  [bits]  at midpoint scale')
    ax.set_title(f'R_q -- {label}' if label else 'R_q')
    ax.axhline(0, color='grey', lw=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(q, T_q, 'go-', markersize=5)
    ax.set_xlabel('q')
    ax.set_ylabel('T_q  at midpoint scale')
    ax.set_title(f'T_q -- {label}' if label else 'T_q')
    ax.axhline(0, color='grey', lw=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if show_errors and D_err is not None and np.any(np.isfinite(D_err)):
        ax.errorbar(q, D_q, yerr=D_err, fmt='ro', capsize=2, markersize=5)
    else:
        ax.plot(q, D_q, 'ro', markersize=5)
    ax.set_xlabel('q')
    ax.set_ylabel('D_q  (Rényi dimension)')
    ax.set_title(f'D_q -- {label}' if label else 'D_q')
    ax.grid(True, alpha=0.3)
    if 'D_1' in spectra and np.isfinite(spectra['D_1']):
        ax.axhline(spectra['D_1'], color='k', ls='--', lw=1,
                   label=f"D_1 (info) = {spectra['D_1']:.3f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Interactive scale-window picker
# ---------------------------------------------------------------------------

def interactive_entropy(pf, mode='extensive', L_ref=None, extrep_ord=None,
                         dx=None, units='', wavelet=None, c_psi=None):
    """Click-to-set scale-range fitter for entropy spectra.

    Mirrors ``interactive_spectra`` in interactive.py.

    Layout (2x3):
        [R_q(a) curves]  [T_q(a) curves]  [R²(q)]
        [D_q vs q]       [R_q vs q]       [summary]

    Left-click on R_q / T_q panels = set min scale (snaps to voices);
    right-click = set max.

    Requires the ipympl backend for live updates: run
    ``%matplotlib widget`` in the notebook before calling.

    All upstream chainmax / smooth-chains / min_chain_voices choices are
    inherited via ``pf``.  Pass ``extrep_ord`` to additionally show the
    Shannon S_1(a) curve and report the information dimension D_1.
    """
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if 'ipympl' not in backend and 'widget' not in backend and 'nbagg' not in backend:
        print("WARNING: interactive dragging requires the ipympl backend.\n"
              "Run  %matplotlib widget  in a notebook cell first.\n"
              "Falling back to a static plot.")

    import matplotlib.pyplot as plt

    log2_a = pf['log2_a']
    n_voice = pf['n_voice']
    idx_max = pf['index_max']
    valid = slice(0, idx_max + 1)
    x_all = log2_a[valid]
    q_list = np.asarray(pf['q_list'])
    n_q = len(q_list)

    dxa = 1.0 / n_voice
    voice_grid = x_all

    def _snap_to_voice(xv):
        i = int(np.argmin(np.abs(voice_grid - xv)))
        return voice_grid[i]

    MIN_GAP_VOICES = 2
    x_lo, x_hi = x_all[0], x_all[-1]
    span = x_hi - x_lo
    init_min = _snap_to_voice(x_lo + 0.2 * span)
    init_max = _snap_to_voice(x_hi - 0.2 * span)

    state = {
        'log2_a_min': init_min,
        'log2_a_max': init_max,
        'spectra': None,
    }

    R_qa, _ = renyi_curves(pf, mode)
    T_qa, _ = tsallis_curves(pf, mode)
    if extrep_ord is not None:
        S1_curve = shannon_entropy_per_scale(extrep_ord)
    else:
        S1_curve = None

    colors_q = plt.cm.coolwarm(np.linspace(0, 1, n_q))
    if n_q > 3:
        i_zero = int(np.argmin(np.abs(q_list)))
        label_set = {0, i_zero, n_q - 1}
    else:
        label_set = set(range(n_q))

    def _qlab(i):
        if i not in label_set:
            return '_nolegend_'
        return f'q={round(q_list[i], 6):g}'

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.subplots_adjust(hspace=0.32, wspace=0.30)
    ax_R, ax_T, ax_R2 = axes[0]
    ax_Dq, ax_Rq, ax_txt = axes[1]

    # R_q(a) panel
    R_static_lines = []
    for i in range(n_q):
        ln, = ax_R.plot(x_all, R_qa[i, valid], 'o-', color=colors_q[i],
                         markersize=2, label=_qlab(i))
        R_static_lines.append(ln)
    ax_R.set_xlabel('log₂(a)')
    ax_R.set_ylabel('R_q(a) [bits]')
    ax_R.set_title('Rényi R_q(a) -- click to set range')
    ax_R.legend(fontsize=6, ncol=1)
    vline_min_R = ax_R.axvline(init_min, color='#e74c3c', ls='--', lw=2)
    vline_max_R = ax_R.axvline(init_max, color='#2980b9', ls='--', lw=2)
    R_fit_lines = []
    for i in range(n_q):
        ln, = ax_R.plot([], [], '-', color=colors_q[i], lw=1.5, alpha=0.7)
        R_fit_lines.append(ln)

    # Shannon overlay (if available)
    if S1_curve is not None:
        ax_R.plot(x_all, S1_curve[valid], 'k--', lw=1.2, alpha=0.8,
                  label='S_1 (Shannon)')
        ax_R.legend(fontsize=6, ncol=1)

    # T_q(a) panel
    for i in range(n_q):
        ax_T.plot(x_all, T_qa[i, valid], 'o-', color=colors_q[i],
                   markersize=2, label=_qlab(i))
    ax_T.set_xlabel('log₂(a)')
    ax_T.set_ylabel('T_q(a)')
    ax_T.set_title('Tsallis T_q(a)')
    ax_T.legend(fontsize=6, ncol=1)
    vline_min_T = ax_T.axvline(init_min, color='#e74c3c', ls='--', lw=1, alpha=0.5)
    vline_max_T = ax_T.axvline(init_max, color='#2980b9', ls='--', lw=1, alpha=0.5)

    # R^2 panel
    ax_R2.set_xlabel('q')
    ax_R2.set_ylabel('R² of R_q(a) fit')
    ax_R2.set_ylim(-0.05, 1.05)
    ax_R2.grid(True, alpha=0.3)
    r2_line, = ax_R2.plot([], [], 'mo-', markersize=4, label='R²[D_q]')
    if S1_curve is not None:
        r2_S1_marker, = ax_R2.plot([], [], 'k*', markersize=10,
                                    label='R²[D_1]')
    ax_R2.set_title('Goodness-of-fit')
    ax_R2.legend(fontsize=7)

    # D_q vs q
    ax_Dq.set_xlabel('q')
    ax_Dq.set_ylabel('D_q  (Rényi dimension)')
    ax_Dq.grid(True, alpha=0.3)
    Dq_line, = ax_Dq.plot([], [], 'ro-', markersize=5)
    D1_hline = ax_Dq.axhline(np.nan, color='k', ls='--', lw=1, alpha=0.0)

    # R_q vs q (level at midpoint)
    ax_Rq.set_xlabel('q')
    ax_Rq.set_ylabel('R_q at midpoint  [bits]')
    ax_Rq.grid(True, alpha=0.3)
    Rq_line, = ax_Rq.plot([], [], 'bo-', markersize=5, label='R_q')
    Tq_line, = ax_Rq.plot([], [], 'g^-', markersize=4, alpha=0.7,
                          label='T_q')
    ax_Rq.legend(fontsize=8)
    ax_Rq.axhline(0, color='grey', lw=0.5)

    # summary
    ax_txt.axis('off')
    summary_text = ax_txt.text(0.04, 0.98, '', transform=ax_txt.transAxes,
                               fontsize=10, verticalalignment='top',
                               fontfamily='monospace')

    def _update():
        a_min_v = state['log2_a_min']
        a_max_v = state['log2_a_max']
        try:
            sp = compute_entropy_spectra(pf, a_min_v, a_max_v, mode=mode,
                                          L_ref=L_ref, extrep_ord=extrep_ord)
        except (ValueError, np.linalg.LinAlgError):
            return
        state['spectra'] = sp

        log2_a0 = np.log2(pf['a_min'])
        idx_lo = max(0, int((a_min_v - log2_a0) / dxa))
        idx_hi = min(idx_max, int((a_max_v - log2_a0) / dxa))
        fit_x = x_all[idx_lo:idx_hi + 1]

        for i in range(n_q):
            if len(fit_x) >= 2:
                y = R_qa[i, idx_lo:idx_hi + 1]
                if np.all(np.isfinite(y)):
                    c = np.polyfit(fit_x, y, 1)
                    R_fit_lines[i].set_data(fit_x, np.polyval(c, fit_x))
                else:
                    R_fit_lines[i].set_data([], [])

        q = sp['q_list']
        D_q = sp['D_q']
        R_q = sp['R_q']
        T_q = sp['T_q']
        Dq_line.set_data(q[np.isfinite(D_q)], D_q[np.isfinite(D_q)])
        Rq_line.set_data(q[np.isfinite(R_q)], R_q[np.isfinite(R_q)])
        Tq_line.set_data(q[np.isfinite(T_q)], T_q[np.isfinite(T_q)])
        ax_Dq.relim(); ax_Dq.autoscale_view(); ax_Dq.set_title('D_q vs q')
        ax_Rq.relim(); ax_Rq.autoscale_view()
        ax_Rq.set_title('R_q, T_q at midpoint')

        if 'D_1' in sp and np.isfinite(sp['D_1']):
            D1_hline.set_ydata([sp['D_1'], sp['D_1']])
            D1_hline.set_alpha(1.0)
            D1_hline.set_label(f"D_1 = {sp['D_1']:.3f}")
            ax_Dq.legend(fontsize=8)

        r2 = sp.get('D_q_R2')
        if r2 is not None:
            r2_line.set_data(q, r2)
            ax_R2.set_xlim(q.min() - 0.5, q.max() + 0.5)
            ax_R2.set_title('R² of R_q(a) fit')
        if S1_curve is not None and 'D_1_R2' in sp:
            r2_S1_marker.set_data([1.0], [sp['D_1_R2']])

        # summary
        h_finite = D_q[np.isfinite(D_q)]
        if len(h_finite) > 0:
            D_min, D_max = h_finite.min(), h_finite.max()
        else:
            D_min = D_max = np.nan
        D0 = D2 = np.nan
        q0_idx = np.where(np.isclose(q, 0.0))[0]
        q2_idx = np.where(np.isclose(q, 2.0))[0]
        if len(q0_idx) > 0:
            D0 = D_q[q0_idx[0]]
        if len(q2_idx) > 0:
            D2 = D_q[q2_idx[0]]
        D1_str = f"{sp['D_1']:.4f}" if 'D_1' in sp else 'n/a'
        mean_r2 = np.nanmean(r2) if r2 is not None else np.nan

        txt = (
            f"Range:    [{a_min_v:.2f}, {a_max_v:.2f}]\n"
            f"Scales:   {idx_hi - idx_lo + 1}\n"
            f"────────────\n"
            f"D_0 (capacity):  {D0:.4f}\n"
            f"D_1 (info):      {D1_str}\n"
            f"D_2 (corr):      {D2:.4f}\n"
            f"────────────\n"
            f"D_q range: [{D_min:.4f}, {D_max:.4f}]\n"
            f"ΔD_q:      {D_max - D_min:.4f}\n"
            f"⟨R²⟩:       {mean_r2:.4f}\n"
            f"────────────\n"
            f"Left-click  = set min\n"
            f"Right-click = set max\n"
            f"(snaps to voices)"
        )
        summary_text.set_text(txt)
        fig.canvas.draw_idle()

    def _set_min(xv):
        xv = _snap_to_voice(xv)
        gap = MIN_GAP_VOICES * dxa
        xv = min(xv, state['log2_a_max'] - gap)
        xv = max(xv, voice_grid[0])
        state['log2_a_min'] = xv
        vline_min_R.set_xdata([xv, xv])
        vline_min_T.set_xdata([xv, xv])

    def _set_max(xv):
        xv = _snap_to_voice(xv)
        gap = MIN_GAP_VOICES * dxa
        xv = max(xv, state['log2_a_min'] + gap)
        xv = min(xv, voice_grid[-1])
        state['log2_a_max'] = xv
        vline_max_R.set_xdata([xv, xv])
        vline_max_T.set_xdata([xv, xv])

    def _on_click(event):
        if event.inaxes not in (ax_R, ax_T) or event.xdata is None:
            return
        if event.button == 1:
            _set_min(event.xdata)
        elif event.button == 3:
            _set_max(event.xdata)
        else:
            return
        _update()

    fig.canvas.mpl_connect('button_press_event', _on_click)

    _update()

    if dx is not None:
        from .spectra import _resolve_c_psi, _add_physical_sublabels
        c = _resolve_c_psi(wavelet, c_psi)
        for ax_ in (ax_R, ax_T):
            _add_physical_sublabels(ax_, dx, c, units, axis='x')

    fig.suptitle("Left-click = set min  |  Right-click = set max  |  "
                 "(snaps to voices)",
                 fontsize=10, y=0.99)

    return fig, state
