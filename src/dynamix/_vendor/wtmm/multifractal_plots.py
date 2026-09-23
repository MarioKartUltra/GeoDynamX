"""Publication-quality multifractal plots, lifted/generalised from v1.

These functions take pre-computed data and return Figure / Axes objects.
Widget glue lives in the notebook; this module is pure-data-to-figure.

Style helpers
-------------
QUARTZ_SLIP_PALETTE     curated colour map for slip-system D(h) curves
                         (matches v1's centered-D(h) viz).
QUARTZ_SLIP_LABELS      LaTeX-formatted slip-system labels.
get_pub_fonts()         FontProperties dict (Arial-with-fallback) for
                         titles / legend / axis labels / ticks.
specific_heat(tau, q)   C(q) = -d^2 tau / d q^2  (Schmitt 1996).

Plotting
--------
fit_cumulants(holders, *, max_order=4)
publication_spectra_2x2(spectra, *, include_cumulant_fits=True, ...)
multi_mode_overlay(spectra_by_mode, *, palette=None, ...)
centered_dh_plot(spectra_by_mode, *, palette=None, ...)
partition_loglog_panel(Z, ls, q_grid, *, ...)
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# =========================================================================
# Style helpers (lifted from v1 centered-D(h) cell so v2 figures match)
# =========================================================================

# Curated quartz slip-system palette (high-contrast, colourblind-friendly).
QUARTZ_SLIP_PALETTE: dict[str, str] = {
    # basal
    'basal_a':            '#d62728',   # red
    'basal_ac':           '#8c564b',   # brown
    # prism
    'prism_a':            '#2ca02c',   # green
    'prism_c':            '#1f77b4',   # blue
    'prism_ac':           '#17becf',   # teal
    # rhomb (positive / negative / dauphine-merged)
    'rhomb_pos_a':        '#ff7f0e',   # orange
    'rhomb_neg_a':        '#bcbd22',   # olive
    'rhomb_a':            '#ff7f0e',   # orange (dauphine-merged)
    'rhomb_pos_ac':       '#9467bd',   # purple
    'rhomb_neg_ac':       '#e377c2',   # pink
    'rhomb_ac':           '#9467bd',   # purple (dauphine-merged)
    # steep dipyramid (hint: gray family)
    'steep_dipyr_a':      '#7f7f7f',
    'steep_dipyr_ac':     '#9e9e9e',
    # generic / total
    'gnd_total':          'black',
    'alpha_sigma_max':    'black',
    'alpha_sigma_min':    '#444444',
}


QUARTZ_SLIP_LABELS: dict[str, str] = {
    'basal_a':         r'$(0001)\langle a \rangle$',
    'basal_ac':        r'$(0001)\langle c\!+\!a \rangle$',
    'prism_a':         r'$\{m\}\langle a \rangle$',
    'prism_c':         r'$\{m\}[c]$',
    'prism_ac':        r'$\{m\}\langle c\!+\!a \rangle$',
    'rhomb_pos_a':     r'$+r\!:\!\{10\bar{1}1\}\langle a \rangle$',
    'rhomb_neg_a':     r'$-r\!:\!\{01\bar{1}1\}\langle a \rangle$',
    'rhomb_a':         r'$r\!:\!\{r\}\langle a \rangle$',
    'rhomb_pos_ac':    r'$+r\!:\!\{10\bar{1}1\}\langle c\!+\!a \rangle$',
    'rhomb_neg_ac':    r'$-r\!:\!\{01\bar{1}1\}\langle c\!+\!a \rangle$',
    'rhomb_ac':        r'$r\!:\!\{r\}\langle c\!+\!a \rangle$',
}


_PUB_FONT_FAMILY = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']


def get_pub_fonts():
    """Return a dict of FontProperties for publication-quality figures.

    Keys: 'title', 'legend', 'axlabel', 'tick'.  Falls back gracefully if
    Arial isn't installed.
    """
    from matplotlib import font_manager as _fm
    return {
        'title':   _fm.FontProperties(family=_PUB_FONT_FAMILY, size=12, weight='bold'),
        'legend':  _fm.FontProperties(family=_PUB_FONT_FAMILY, size=9,  style='italic'),
        'axlabel': _fm.FontProperties(family=_PUB_FONT_FAMILY, size=10),
        'tick':    _fm.FontProperties(family=_PUB_FONT_FAMILY, size=10),
    }


def specific_heat(tau: np.ndarray, q: np.ndarray) -> np.ndarray:
    """C(q) = -d^2 tau / d q^2 (Schmitt 1996 cascade specific heat).

    Positive bumps in C(q) flag a freezing transition (multi-modal cascade);
    monofractal cascades give C(q) ~ 0 everywhere.
    """
    q = np.asarray(q, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    if q.size < 3:
        return np.full_like(q, np.nan)
    d_tau_dq = np.gradient(tau, q)
    d2_tau_dq2 = np.gradient(d_tau_dq, q)
    return -d2_tau_dq2


def fit_cumulants(holders: np.ndarray, *, max_order: int = 4) -> dict:
    """Fit the leading cumulants c_1, c_2, ... of the Holder distribution.

    Multifractal cumulant expansion: tau(q) ~ -1 + sum_n (c_n / n!) * q^n
    The first cumulants are estimated from the moments of `holders`:
        c_1 = mean(h)              (= alpha_0, peak of D(h))
        c_2 = -var(h)              (controls D(h) width; -1 / log_2(scale))
        c_3 = third central moment (asymmetry)
        c_4 = fourth central moment - 3*var^2 (excess kurtosis)

    Parameters
    ----------
    holders : array_like
        Per-chain Holder exponents (NaN-safe).
    max_order : int
        Highest cumulant to compute (capped at 4; higher orders are
        statistically unreliable on real data).

    Returns
    -------
    dict
        Keys: 'c1', 'c2', ... up to max_order, plus 'n' (sample size).
    """
    h = np.asarray(holders, dtype=np.float64)
    h = h[np.isfinite(h)]
    n = h.size
    out = {'n': int(n)}
    if n == 0:
        for k in range(1, max_order + 1):
            out[f'c{k}'] = float('nan')
        return out
    out['c1'] = float(np.mean(h))
    if max_order >= 2:
        out['c2'] = -float(np.var(h))
    if max_order >= 3:
        d = h - out['c1']
        out['c3'] = float(np.mean(d ** 3))
    if max_order >= 4:
        d = h - out['c1']
        m4 = float(np.mean(d ** 4))
        var = float(np.var(h))
        out['c4'] = m4 - 3 * var * var
    return out


def _legendre_dh(q: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """h(q) = d tau / d q  (numerical), D(h) = q*h - tau (Legendre)."""
    h = np.gradient(tau, q)
    Dh = q * h - tau
    return h, Dh


def partition_loglog_panel(
    Z: np.ndarray,
    ls: np.ndarray,
    q_grid: np.ndarray,
    *,
    title: Optional[str] = None,
):
    """Log-log partition plot: log Z vs log L for each q in q_grid.

    Parameters
    ----------
    Z : (n_q, n_scales) float
    ls : (n_scales,)  scales (in pixels or um -- caller decides)
    q_grid : (n_q,)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=110)
    log_l = np.log(np.asarray(ls, dtype=np.float64))
    cmap = plt.cm.viridis
    n_q = len(q_grid)
    for qi, q in enumerate(q_grid):
        c = cmap(qi / max(n_q - 1, 1))
        ax.plot(log_l, Z[qi], 'o-', color=c, ms=4, label=f'q={q:+.2f}')
    ax.set(xlabel=r'$\log L$ (box size)',
            ylabel=r'$\log Z(q, L)$',
            title=title or 'Partition function')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _convention_labels(convention: str) -> dict:
    """Resolve axis-label dict for one of: 'wtmm' | 'physics'.

    'wtmm'    -> h, D(h), h(q)              (WTMM literature standard)
    'physics' -> alpha, f(alpha), alpha(q)  (Halsey/Procaccia)
    """
    if convention == 'wtmm':
        return {'x_h':       r'$h$',
                'y_dh':      r'$D(h)$',
                'y_hq':      r'$h(q)$',
                'title_dh':  r'$D(h)$ — singularity spectrum',
                'title_hq':  r'$h(q) = d\tau/dq$',
                'title_overlay': r'Multi-mode $D(h)$ overlay',
                'centered_x': r'$h - h_{peak}$ (centered, x = layout)',
                'centered_y': r'$D(h)$',
                'centered_title': r'Centered $D(h)$ — group layout'}
    if convention == 'physics':
        return {'x_h':       r'$\alpha$',
                'y_dh':      r'$f(\alpha)$',
                'y_hq':      r'$\alpha(q)$',
                'title_dh':  r'$f(\alpha)$ — singularity spectrum',
                'title_hq':  r'$\alpha(q) = d\tau/dq$',
                'title_overlay': r'Multi-mode $f(\alpha)$ overlay',
                'centered_x': r'$\alpha - \alpha_{peak}$ (centered, x = layout)',
                'centered_y': r'$f(\alpha)$',
                'centered_title': r'Centered $f(\alpha)$ — group layout'}
    raise ValueError(f"convention must be 'wtmm' or 'physics', got {convention!r}")


def publication_spectra_2x2(
    spectra: dict,
    *,
    convention: str = 'wtmm',
    include_cumulant_fits: bool = True,
    title: Optional[str] = None,
    cumulant_max_order: int = 3,
):
    """2x2 publication panel: tau(q), D(h), h(q), C(q).

    The cumulant-fit overlay (smooth curve through tau(q) and a Gaussian
    D(h) using c_1, c_2) is OPTIONAL via `include_cumulant_fits`.

    Parameters
    ----------
    spectra : dict with at least:
        'q'    : (n_q,) float
        'tau'  : (n_q,) float
        'h'    : (n_q,) float    (optional; recomputed if missing)
        'D_h'  : (n_q,) float    (optional)
        'C_q'  : (n_q,) float    (optional cumulant function; if absent,
                                  the C(q) panel falls back to the
                                  generalised dimension D(q) = tau(q)/(q-1).)
        'holders' : (N,) float   (per-chain Holders; needed for cumulant
                                  fits if include_cumulant_fits=True)
    include_cumulant_fits : bool
        If True, overlay c1, c2 (and optionally c3, c4) Gaussian/cubic
        approximations on the D(h) and tau(q) panels.  If False, plot raw
        spectra only.
    title : str, optional
    cumulant_max_order : int
        Up to which cumulant to fit (1-4).  Default 3 (mean, variance,
        skew).

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    q = np.asarray(spectra['q'], dtype=np.float64)
    tau = np.asarray(spectra['tau'], dtype=np.float64)
    if 'h' in spectra and 'D_h' in spectra:
        h_arr = np.asarray(spectra['h'], dtype=np.float64)
        D_h   = np.asarray(spectra['D_h'], dtype=np.float64)
    else:
        h_arr, D_h = _legendre_dh(q, tau)
    if 'C_q' in spectra:
        C_q = np.asarray(spectra['C_q'], dtype=np.float64)
        c_label = r'$C(q) = -d^2\tau/dq^2$'
    else:
        # Compute C(q) ourselves rather than fall back to D(q) -- the
        # specific heat is the canonical 4th panel in multifractal
        # publication figures.
        C_q = specific_heat(tau, q)
        c_label = r'$C(q) = -d^2\tau/dq^2$'

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=110)
    if title:
        fig.suptitle(title, fontsize=12)
    a_tau, a_dh, a_hq, a_cq = axes.flat

    # tau(q)
    a_tau.plot(q, tau, 'o-', color='black', ms=4, label='measured')
    a_tau.set(xlabel='q', ylabel=r'$\tau(q)$', title=r'$\tau(q)$ — mass exponent')
    a_tau.axhline(0, color='gray', lw=0.5, alpha=0.5)
    a_tau.axvline(1.0, color='gray', lw=0.5, alpha=0.5, ls=':')
    a_tau.grid(alpha=0.3)

    # D(h) vs h (WTMM convention) or f(alpha) vs alpha (physics)
    lbl = _convention_labels(convention)
    a_dh.plot(h_arr, D_h, 'o-', color='black', ms=4, label='measured')
    a_dh.set(xlabel=lbl['x_h'], ylabel=lbl['y_dh'], title=lbl['title_dh'])
    a_dh.grid(alpha=0.3)

    # h(q) or alpha(q)
    a_hq.plot(q, h_arr, 'o-', color='black', ms=4)
    a_hq.set(xlabel='q', ylabel=lbl['y_hq'], title=lbl['title_hq'])
    a_hq.axvline(1.0, color='gray', lw=0.5, alpha=0.5, ls=':')
    a_hq.grid(alpha=0.3)

    # C(q) — specific heat (Schmitt 1996)
    a_cq.plot(q, C_q, 'o-', color='black', ms=4)
    a_cq.axhline(0, color='gray', lw=0.5, alpha=0.5)
    a_cq.set(xlabel='q', ylabel=c_label,
              title=r'$C(q)$ — specific heat (cascade phase-transition)')
    a_cq.grid(alpha=0.3)

    # ---- optional cumulant overlay ----
    if include_cumulant_fits and 'holders' in spectra:
        cum = fit_cumulants(np.asarray(spectra['holders']),
                             max_order=cumulant_max_order)
        if np.isfinite(cum['c1']):
            c1 = cum['c1']
            c2 = cum.get('c2', float('nan'))
            c3 = cum.get('c3', float('nan'))
            # tau(q) cumulant expansion: tau(q) ~ -1 + c1*q + c2*q^2/2 + c3*q^3/6
            tau_fit = -1.0 + c1 * q
            if np.isfinite(c2):
                tau_fit = tau_fit + 0.5 * c2 * q * q
            if cumulant_max_order >= 3 and np.isfinite(c3):
                tau_fit = tau_fit + (1.0 / 6.0) * c3 * q ** 3
            a_tau.plot(q, tau_fit, '--', color='red', lw=1.5,
                        label=f'cumulant fit (c$_1$={c1:.2f}, c$_2$={c2:.2f})'
                              if np.isfinite(c2) else f'cumulant fit (c$_1$={c1:.2f})')
            a_tau.legend(fontsize=8)
            # D(h) Gaussian approximation:  D(h) = 1 - (h - c1)^2 / (-2 c2)
            # only valid when c2 < 0 (so denom > 0).
            if np.isfinite(c2) and c2 < 0:
                h_fit = np.linspace(h_arr.min(), h_arr.max(), 200)
                D_fit = 1.0 - (h_fit - c1) ** 2 / (-2.0 * c2)
                a_dh.plot(h_fit, D_fit, '--', color='red', lw=1.5,
                            label='Gaussian (cumulant)')
                a_dh.axvline(c1, color='gray', lw=0.5, alpha=0.5)
                a_dh.legend(fontsize=8)
            # Cumulant C(q) ~= -c2 (constant) for the leading-order
            # cumulant truncation; overlay as a horizontal red line.
            if np.isfinite(c2):
                a_cq.axhline(-c2, color='red', lw=1.2, ls='--', alpha=0.7,
                              label=f'leading cum: C ~ {-c2:+.3f}')
                a_cq.legend(fontsize=8)
            # h(q) cumulant: h(q) = c1 + c2*q + c3*q^2/2
            h_fit_q = np.full_like(q, c1, dtype=np.float64)
            if np.isfinite(c2):
                h_fit_q = h_fit_q + c2 * q
            if cumulant_max_order >= 3 and np.isfinite(c3):
                h_fit_q = h_fit_q + 0.5 * c3 * q ** 2
            a_hq.plot(q, h_fit_q, '--', color='red', lw=1.5)

    fig.tight_layout()
    return fig


def _color_for_mode(mode: str, palette: Optional[dict], i: int):
    """Pick the curated palette entry if known, else fall back to tab10."""
    import matplotlib.pyplot as plt
    if palette is not None and mode in palette:
        return palette[mode]
    # Strip _shuf / _shufall / _pruned / _h_pruned suffix and try again
    for suf in ('_shufall', '_shuf', '_h_pruned', '_pruned'):
        if mode.endswith(suf):
            base = mode[:-len(suf)]
            if palette is not None and base in palette:
                return palette[base]
            break
    return plt.cm.tab10(i % 10)


def multi_mode_overlay(
    spectra_by_mode: dict,
    *,
    palette: Optional[dict] = None,
    use_pub_fonts: bool = True,
    title: Optional[str] = None,
    legend_loc: str = 'best',
    convention: str = 'wtmm',
    use_alpha_label: Optional[bool] = None,   # deprecated alias
):
    """Overlay D(h) curves from multiple modes on one panel.

    Parameters
    ----------
    spectra_by_mode : dict[str, dict]
        Each value must have 'h' and 'D_h' keys (or 'alpha' / 'f').
    palette : dict[mode_name -> color], optional
    use_pub_fonts : bool
    convention : 'wtmm' | 'physics'
        WTMM default uses h / D(h).  Pass 'physics' for alpha / f(alpha).
    use_alpha_label : bool, optional
        Deprecated alias for `convention='physics'`.  If supplied,
        overrides `convention`.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    if use_alpha_label is not None:
        convention = 'physics' if use_alpha_label else 'wtmm'
    lbl = _convention_labels(convention)
    fonts = get_pub_fonts() if use_pub_fonts else None

    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    for i, (mode, sp) in enumerate(spectra_by_mode.items()):
        h  = np.asarray(sp.get('alpha', sp.get('h')))
        Dh = np.asarray(sp.get('f',     sp.get('D_h')))
        ls = ':' if any(s in mode for s in ('_shuf', '_shufall')) else '-'
        col = _color_for_mode(mode, palette, i)
        label = QUARTZ_SLIP_LABELS.get(mode.replace('gnd_', ''), mode)
        ax.plot(h, Dh, marker='o', ms=3, lw=1.4, ls=ls,
                color=col, label=label)
    if fonts:
        ax.set_xlabel(lbl['x_h'], fontproperties=fonts['axlabel'])
        ax.set_ylabel(lbl['y_dh'], fontproperties=fonts['axlabel'])
        ax.set_title(title or lbl['title_overlay'],
                      fontproperties=fonts['title'])
        ax.legend(prop=fonts['legend'], loc=legend_loc, ncol=1,
                    frameon=False, handlelength=2.4)
    else:
        ax.set(xlabel=lbl['x_h'], ylabel=lbl['y_dh'],
                title=title or lbl['title_overlay'])
        ax.legend(fontsize=8, loc=legend_loc, ncol=1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def centered_dh_plot(
    spectra_by_mode: dict,
    *,
    group_assignments: Optional[dict] = None,
    group_centers: Optional[dict] = None,
    palette: Optional[dict] = None,
    use_pub_fonts: bool = True,
    convention: str = 'wtmm',
    use_alpha_label: Optional[bool] = None,    # deprecated alias
    title: Optional[str] = None,
):
    """Centered f(α): each mode's spectrum centred so peaks stack at a
    group centre.  Same x-axis-as-layout convention as v1.

    Parameters
    ----------
    spectra_by_mode : dict[mode_name -> {'alpha'|'h', 'f'|'D_h'}]
    group_assignments : dict[mode_name -> group_label]
    group_centers : dict[group_label -> float]
    palette : dict, optional      curated colour map (e.g. QUARTZ_SLIP_PALETTE)
    use_pub_fonts : bool          Arial fonts
    use_alpha_label : bool        physics convention (α, f(α))

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    if use_alpha_label is not None:
        convention = 'physics' if use_alpha_label else 'wtmm'
    lbl = _convention_labels(convention)
    fonts = get_pub_fonts() if use_pub_fonts else None

    if group_assignments is None:
        group_assignments = {m: 'main' for m in spectra_by_mode}
    groups = list(dict.fromkeys(group_assignments.values()))
    if group_centers is None:
        group_centers = {g: float(i) for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
    for i, (mode, sp) in enumerate(spectra_by_mode.items()):
        h  = np.asarray(sp.get('alpha', sp.get('h')))
        Dh = np.asarray(sp.get('f',     sp.get('D_h')))
        if h.size == 0: continue
        peak_idx = int(np.argmax(Dh))
        h_peak = h[peak_idx]
        group = group_assignments.get(mode, 'main')
        x_offset = group_centers.get(group, 0.0)
        ls = '-' if all(s not in group for s in ('shuf',)) else ':'
        col = _color_for_mode(mode, palette, i)
        label = QUARTZ_SLIP_LABELS.get(mode.replace('gnd_', ''), mode)
        ax.plot(h - h_peak + x_offset, Dh, marker='o', ms=3, lw=1.4,
                ls=ls, color=col, label=label)
    ttl = title or lbl['centered_title']
    if fonts:
        ax.set_xlabel(lbl['centered_x'], fontproperties=fonts['axlabel'])
        ax.set_ylabel(lbl['centered_y'], fontproperties=fonts['axlabel'])
        ax.set_title(ttl, fontproperties=fonts['title'])
        ax.legend(prop=fonts['legend'], ncol=1, loc='upper left',
                  bbox_to_anchor=(1.02, 1.0), frameon=False,
                  handlelength=2.4, handletextpad=0.6, labelspacing=0.4)
    else:
        ax.set(xlabel=lbl['centered_x'], ylabel=lbl['centered_y'], title=ttl)
        ax.legend(fontsize=8, ncol=1, loc='upper left',
                  bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    return fig
