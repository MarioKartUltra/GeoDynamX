"""Interactive scale range selector for WTMM multifractal spectra.

Requires the ipympl backend: run ``%matplotlib widget`` in the notebook
before importing this module.
"""

import numpy as np


def interactive_spectra(pf, mode='extensive', L_ref=None,
                        dx=None, units='', wavelet=None, c_psi=None,
                        dh_xlim=(-0.2, 2.6), dh_ylim=(-0.5, 1.5)):
    """Launch an interactive 2x4 figure for exploring scale-range sensitivity.

    Click on the T(q,a), H(q,a), or D(q,a) panels to set the regression
    range; the spectra panels update immediately.  Positions snap to voice
    grid.

    Parameters
    ----------
    pf : dict
        Partition function from ``compute_partition_function``.
    mode : str
        ``'extensive'`` or ``'intensive'``.
    L_ref : float or None
        Reference length scale (passed to ``compute_spectra``).

    Returns
    -------
    fig : matplotlib Figure
    state : dict
        Live references: ``state['log2_a_min']``, ``state['log2_a_max']``,
        ``state['spectra']`` (most recent result).
    """
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if 'ipympl' not in backend and 'widget' not in backend and 'nbagg' not in backend:
        print("WARNING: interactive dragging requires the ipympl backend.\n"
              "Run  %matplotlib widget  in a notebook cell first.\n"
              "Falling back to a static plot.")

    import matplotlib.pyplot as plt

    from .partition import pf_get_T, pf_get_H, pf_get_D
    from .spectra import compute_spectra

    # ── static data ──────────────────────────────────────────────────
    log2_a = pf['log2_a']
    n_voice = pf['n_voice']
    idx_max = pf['index_max']
    valid = slice(0, idx_max + 1)
    x_all = log2_a[valid]
    q_list = pf['q_list']
    n_q = len(q_list)

    # Voice grid — the discrete log₂(a) values where scales exist
    dx = 1.0 / n_voice
    voice_grid = x_all

    def _snap_to_voice(x):
        """Snap x to the nearest voice grid point."""
        idx = np.argmin(np.abs(voice_grid - x))
        return voice_grid[idx]

    MIN_GAP_VOICES = 2

    # Initial range: middle 60 % of available scales, snapped
    x_lo, x_hi = x_all[0], x_all[-1]
    span = x_hi - x_lo
    init_min = _snap_to_voice(x_lo + 0.2 * span)
    init_max = _snap_to_voice(x_hi - 0.2 * span)

    state = {
        'log2_a_min': init_min,
        'log2_a_max': init_max,
        'spectra': None,
    }

    # ── pre-compute static partition-function curves ─────────────────
    colors_q = plt.cm.coolwarm(np.linspace(0, 1, n_q))

    # Sparse legend: only label q_min, q≈0, q_max
    if n_q > 3:
        i_zero = int(np.argmin(np.abs(q_list)))
        label_set = {0, i_zero, n_q - 1}
    else:
        label_set = set(range(n_q))

    def _qlabel(i):
        if i not in label_set:
            return '_nolegend_'
        q_disp = round(q_list[i], 6)
        return f'q={q_disp:g}'

    T_curves = [pf_get_T(pf, i, mode)[valid] for i in range(n_q)]
    H_curves = [pf_get_H(pf, i, mode)[valid] for i in range(n_q)]
    D_curves = [pf_get_D(pf, i, q_list[i], mode)[valid] for i in range(n_q)]

    # ── create figure (2×4) ──────────────────────────────────────────
    #  T(q,a)          | H(q,a)          | D(q,a)         | R²
    #  h(q) & D(q) vs q| D(h) vs h       | summary text   | (empty)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.subplots_adjust(hspace=0.35, wspace=0.30)

    ax_T   = axes[0, 0]
    ax_H   = axes[0, 1]
    ax_Da  = axes[0, 2]   # D(q,a) vs log2(a)
    ax_R2  = axes[0, 3]
    ax_hDq = axes[1, 0]   # h(q) & D(q) vs q
    ax_Dh  = axes[1, 1]   # D(h) vs h
    ax_txt = axes[1, 2]   # summary
    axes[1, 3].axis('off') # unused

    # ── T(q,a) panel ─────────────────────────────────────────────────
    for i in range(n_q):
        ax_T.plot(x_all, T_curves[i], 'o-', color=colors_q[i], markersize=2,
                  label=_qlabel(i))
    ax_T.set_xlabel('log₂(a)')
    ax_T.set_ylabel('T(q,a) = log₂ Z(q,a)')
    ax_T.set_title('T(q,a) — click to set range')
    ax_T.legend(fontsize=6, ncol=1)

    vline_min_T = ax_T.axvline(init_min, color='#e74c3c', ls='--', lw=2)
    vline_max_T = ax_T.axvline(init_max, color='#2980b9', ls='--', lw=2)

    T_fit_lines = []
    for i in range(n_q):
        ln, = ax_T.plot([], [], '-', color=colors_q[i], lw=1.5, alpha=0.7)
        T_fit_lines.append(ln)

    # ── H(q,a) panel ─────────────────────────────────────────────────
    for i in range(n_q):
        ax_H.plot(x_all, H_curves[i], 'o-', color=colors_q[i], markersize=2,
                  label=_qlabel(i))
    ax_H.set_xlabel('log₂(a)')
    ax_H.set_ylabel('H(q,a)')
    ax_H.set_title('H(q,a)')
    ax_H.legend(fontsize=6, ncol=1)

    vline_min_H = ax_H.axvline(init_min, color='#e74c3c', ls='--', lw=1, alpha=0.5)
    vline_max_H = ax_H.axvline(init_max, color='#2980b9', ls='--', lw=1, alpha=0.5)

    H_fit_lines = []
    for i in range(n_q):
        ln, = ax_H.plot([], [], '-', color=colors_q[i], lw=1.5, alpha=0.7)
        H_fit_lines.append(ln)

    # ── D(q,a) panel ───────────────────────────────────────────────
    for i in range(n_q):
        ax_Da.plot(x_all, D_curves[i], 'o-', color=colors_q[i], markersize=2,
                   label=_qlabel(i))
    ax_Da.set_xlabel('log₂(a)')
    ax_Da.set_ylabel('D(q,a) = q·H − T')
    ax_Da.set_title('D(q,a)')
    ax_Da.legend(fontsize=6, ncol=1)

    vline_min_Da = ax_Da.axvline(init_min, color='#e74c3c', ls='--', lw=1, alpha=0.5)
    vline_max_Da = ax_Da.axvline(init_max, color='#2980b9', ls='--', lw=1, alpha=0.5)

    D_fit_lines = []
    for i in range(n_q):
        ln, = ax_Da.plot([], [], '-', color=colors_q[i], lw=1.5, alpha=0.7)
        D_fit_lines.append(ln)

    # ── R² panel ─────────────────────────────────────────────────────
    ax_R2.set_xlabel('q')
    ax_R2.set_ylabel('R²')
    ax_R2.set_ylim(-0.05, 1.05)
    ax_R2.grid(True, alpha=0.3)
    # Three lines; visibility toggled per method
    r2_tau_line, = ax_R2.plot([], [], 'bs-', markersize=4, label='τ(q) [T fit]')
    r2_h_line,   = ax_R2.plot([], [], 'ro-', markersize=4, label='h(q) [H fit]')
    r2_D_line,   = ax_R2.plot([], [], 'g^-', markersize=4, label='D(q) [D fit]')

    # ── h(q) & D(q) vs q panel ───────────────────────────────────────
    ax_hDq.set_xlabel('q')
    ax_hDq.grid(True, alpha=0.3)
    hq_line, = ax_hDq.plot([], [], 'ro-', markersize=5, label='h(q)')
    Dq_line, = ax_hDq.plot([], [], 'g^-', markersize=5, label='D(q)')
    tauq_line, = ax_hDq.plot([], [], 'bs--', markersize=4, alpha=0.6,
                              label='τ(q)=qh−D')
    ax_hDq.legend(fontsize=7, loc='best')

    # ── D(h) vs h panel ──────────────────────────────────────────────
    ax_Dh.set_xlabel('h')
    ax_Dh.set_ylabel('D(h)')
    ax_Dh.grid(True, alpha=0.3)
    ax_Dh.set_aspect('equal')
    ax_Dh.axhline(0, color='grey', lw=0.5)
    ax_Dh.axvline(0, color='grey', lw=0.5)
    Dh_line, = ax_Dh.plot([], [], 'ro-', markersize=5)
    tangent_line, = ax_Dh.plot([], [], 'k--', lw=1, alpha=0.6,
                                label='q=1 tangent')

    # ── summary text panel ───────────────────────────────────────────
    ax_txt.axis('off')
    summary_text = ax_txt.text(0.05, 0.95, '', transform=ax_txt.transAxes,
                               fontsize=10, verticalalignment='top',
                               fontfamily='monospace')

    # ── update function ──────────────────────────────────────────────
    def _update():
        a_min_val = state['log2_a_min']
        a_max_val = state['log2_a_max']

        try:
            sp = compute_spectra(pf, a_min_val, a_max_val,
                                 mode=mode, method='canonical', L_ref=L_ref)
        except (ValueError, np.linalg.LinAlgError):
            return
        state['spectra'] = sp

        q = sp['q_list']
        tau_q = sp['tau_q']
        h_q = sp['h_q']
        D_q = sp['D_q']

        # ── regression lines on T and H panels ──────────────────
        log2_a0 = np.log2(pf['a_min'])
        dxa = 1.0 / pf['n_voice']
        idx_lo = max(0, int(round((a_min_val - log2_a0) / dxa)))   # round, not truncate (FP: log2 of a
        idx_hi = min(idx_max, int(round((a_max_val - log2_a0) / dxa)))  # geometric scale lands at k-eps)
        fit_x = x_all[idx_lo:idx_hi + 1]

        for i in range(n_q):
            if len(fit_x) >= 2:
                y_T = T_curves[i][idx_lo:idx_hi + 1]
                if np.all(np.isfinite(y_T)):
                    c = np.polyfit(fit_x, y_T, 1)
                    T_fit_lines[i].set_data(fit_x, np.polyval(c, fit_x))
                else:
                    T_fit_lines[i].set_data([], [])

                y_H = H_curves[i][idx_lo:idx_hi + 1]
                if np.all(np.isfinite(y_H)):
                    c = np.polyfit(fit_x, y_H, 1)
                    H_fit_lines[i].set_data(fit_x, np.polyval(c, fit_x))
                else:
                    H_fit_lines[i].set_data([], [])

                y_D = D_curves[i][idx_lo:idx_hi + 1]
                if np.all(np.isfinite(y_D)):
                    c = np.polyfit(fit_x, y_D, 1)
                    D_fit_lines[i].set_data(fit_x, np.polyval(c, fit_x))
                else:
                    D_fit_lines[i].set_data([], [])

        # ── h(q) & D(q) vs q ────────────────────────────────────
        finite = np.isfinite(h_q)
        hq_line.set_data(q[finite], h_q[finite])
        finite_D = np.isfinite(D_q)
        Dq_line.set_data(q[finite_D], D_q[finite_D])
        finite_tau = np.isfinite(tau_q)
        tauq_line.set_data(q[finite_tau], tau_q[finite_tau])
        ax_hDq.relim()
        ax_hDq.autoscale_view()
        ax_hDq.set_title('h(q), D(q), τ(q)')

        # ── D(h) vs h ───────────────────────────────────────────
        finite_Dh = np.isfinite(h_q) & np.isfinite(D_q)
        Dh_line.set_data(h_q[finite_Dh], D_q[finite_Dh])
        ax_Dh.set_xlim(*dh_xlim)
        ax_Dh.set_ylim(*dh_ylim)
        tau_1 = sp.get('tau_1', np.nan)
        if np.isfinite(tau_1):
            h_tan = np.array([-0.2, 2.6])
            tangent_line.set_data(h_tan, h_tan - tau_1)
            tangent_line.set_label(f'q=1: D=h{-tau_1:+.3f}')
        else:
            tangent_line.set_data([], [])
        ax_Dh.legend(fontsize=7, loc='upper right')
        ax_Dh.set_title('D(h)')

        # ── R² ─────────────────────────────────────────────────────
        r2_tau = sp.get('tau_R2', np.full(len(q), np.nan))
        r2_h = sp.get('h_R2', np.full(len(q), np.nan))
        r2_D = sp.get('D_R2', np.full(len(q), np.nan))

        r2_tau_line.set_data(q, r2_tau)
        r2_h_line.set_data(q, r2_h)
        r2_D_line.set_data(q, r2_D)
        ax_R2.set_title('R²')
        ax_R2.legend(fontsize=6)
        ax_R2.set_xlim(q.min() - 0.5, q.max() + 0.5)

        # ── summary text ─────────────────────────────────────────
        h_range = (np.nanmin(h_q), np.nanmax(h_q))

        h1 = h2 = D0 = np.nan
        q1_idx = np.where(np.isclose(q, 1.0))[0]
        q2_idx = np.where(np.isclose(q, 2.0))[0]
        q0_idx = np.where(np.isclose(q, 0.0))[0]
        if len(q1_idx) > 0:
            h1 = h_q[q1_idx[0]]
        if len(q2_idx) > 0:
            h2 = h_q[q2_idx[0]]
        if len(q0_idx) > 0:
            D0 = D_q[q0_idx[0]]

        fin_r2_h = r2_h[np.isfinite(r2_h)]
        mean_r2_h = np.nanmean(fin_r2_h) if len(fin_r2_h) > 0 else np.nan
        fin_r2_D = r2_D[np.isfinite(r2_D)]
        mean_r2_D = np.nanmean(fin_r2_D) if len(fin_r2_D) > 0 else np.nan

        txt = (
            f"Range:   [{a_min_val:.2f}, {a_max_val:.2f}]\n"
            f"Scales:  {idx_hi - idx_lo + 1}\n"
            f"─────────────────────\n"
            f"τ(1):    {tau_1:.4f}\n"
            f"h(1):    {h1:.4f}\n"
            f"h(2):    {h2:.4f}\n"
            f"D(q=0):  {D0:.4f}\n"
            f"─────────────────────\n"
            f"h range: [{h_range[0]:.4f}, {h_range[1]:.4f}]\n"
            f"Δh:      {h_range[1] - h_range[0]:.4f}\n"
            f"⟨R²⟩h:   {mean_r2_h:.4f}\n"
            f"⟨R²⟩D:   {mean_r2_D:.4f}\n"
            f"─────────────────────\n"
            f"Left-click  = set min\n"
            f"Right-click = set max\n"
            f"(snaps to voices)"
        )
        summary_text.set_text(txt)

        fig.canvas.draw_idle()

    # ── click logic: left = min, right = max, snap to voices ─────────
    def _set_min(x):
        x = _snap_to_voice(x)
        min_gap = MIN_GAP_VOICES * dx
        x = min(x, state['log2_a_max'] - min_gap)
        x = max(x, voice_grid[0])
        state['log2_a_min'] = x
        vline_min_T.set_xdata([x, x])
        vline_min_H.set_xdata([x, x])
        vline_min_Da.set_xdata([x, x])

    def _set_max(x):
        x = _snap_to_voice(x)
        min_gap = MIN_GAP_VOICES * dx
        x = max(x, state['log2_a_min'] + min_gap)
        x = min(x, voice_grid[-1])
        state['log2_a_max'] = x
        vline_max_T.set_xdata([x, x])
        vline_max_H.set_xdata([x, x])
        vline_max_Da.set_xdata([x, x])

    def _on_click(event):
        if event.inaxes not in (ax_T, ax_H, ax_Da) or event.xdata is None:
            return
        if event.button == 1:       # left click → set min
            _set_min(event.xdata)
        elif event.button == 3:     # right click → set max
            _set_max(event.xdata)
        else:
            return
        _update()

    fig.canvas.mpl_connect('button_press_event', _on_click)

    # ── initial draw ─────────────────────────────────────────────────
    _update()

    # Physical unit sub-labels on x-axis of T, H, D panels
    if dx is not None:
        from .spectra import _resolve_c_psi, _add_physical_sublabels
        c = _resolve_c_psi(wavelet, c_psi)
        for _ax in (ax_T, ax_H, ax_Da):
            _add_physical_sublabels(_ax, dx, c, units, axis='x')

    fig.suptitle("Left-click = set min  |  Right-click = set max  |  "
                 "(snaps to voices)",
                 fontsize=10, y=0.99)

    return fig, state
