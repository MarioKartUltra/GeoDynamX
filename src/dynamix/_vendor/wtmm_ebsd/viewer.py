"""WTMMM viewer precompute — chain skeleton + per-scale extrema bundles + scalebar.

Also provides the v1 green-aesthetic ``add_wavelet_scalebar`` overlay for any
2D wavelet-domain figure (modulus snapshot, extrema map, …).

Faithful port of v1's `_precompute_mode` (cell 83 of pergrain_pantleon)
with the following bug fixes and efficiency improvements:

  1.  Partition-function arrays come from
      :func:`wtmm_ebsd.partition.build_hd_from_chains` directly on the
      chain modulus values, NOT from the round-trip through
      ``ext.get_extrema_arrays()`` + ``chain_lookup`` that mis-counted
      most chains in v1 (the "1627 surviving points" phenomenon).
  2.  The per-scale ``on_vc`` rebuild in v1 used the get_extrema_arrays /
      get_single_maxima_arrays mismatch implicitly; here we mark
      ``on_vc`` from a direct chain-position set, which is guaranteed
      to find every chain extremum.
  3.  ``mods_along`` in the chain-segment splitter no longer uses a
      Python list comprehension; it indexes a dict built once per scale.
  4.  Chainmax lookup is a dict per scale; values for mid-chain are
      explicit (no implicit fallthrough to the raw modulus).
  5.  Profile-panel data (``running_max_matrix``) computed once per chain
      with ``np.maximum.accumulate``.

Memory: bundles are returned as plain dicts; the caller (notebook) is
responsible for caching across SVD modes.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from .partition import build_hd_from_chains, DEFAULT_Q_LIST


# ---------------------------------------------------------------------------
# Wavelet-scalebar overlay (v1 green aesthetic, ported here so any cell
# showing a wavelet-domain figure can call it without redefining)
# ---------------------------------------------------------------------------

def _wavelet_shape(wavelet: str = 'gaussian',
                    n_pts: int = 301) -> tuple[np.ndarray, np.ndarray]:
    """Unit-scale analyzing-wavelet kernel for the scalebar inset.

    'gaussian'  -> first derivative of Gaussian (g1)  -- v1 default.
    'mexican'   -> third derivative of Gaussian (g3)  -- v1 'mexican'.
    """
    x = np.linspace(-4, 4, n_pts)
    if wavelet == 'mexican':
        psi = x * (x ** 2 - 3) * np.exp(-x ** 2 / 2)
    else:
        psi = -x * np.exp(-x ** 2 / 2)
    psi /= np.max(np.abs(psi))
    return x, psi


def add_wavelet_scalebar(ax,
                          scale: float,
                          *,
                          px_to_unit: float = 1.0,
                          unit_str: str = 'px',
                          wavelet: str = 'gaussian',
                          loc: str = 'lower right',
                          pad_frac: float = 0.04,
                          fontsize: int = 7,
                          color: str = '#39FF14',
                          clip_on: bool = False,
                          unclamp: bool = False) -> None:
    """Overlay a wavelet kernel + scale-length label on a 2D axes (v1 style).

    The wavelet trace is rendered in axes-fractional coords so it scales
    with the figure; physical width corresponds to ``scale`` pixels of the
    parent image.  Bright-green (``#39FF14``) is the v1 default and reads
    well over IPF / band-contrast / dark backgrounds with a black stroke.

    Should ONLY be applied to wavelet-domain figures: per-scale modulus
    snapshots, extrema overlays, chain rendering at a chosen scale, etc.
    Putting it on a non-wavelet field (KAM, CPO θ_attractor, IPF map) is
    misleading because the scale has no relationship to the displayed data.
    """
    import matplotlib.patheffects as path_effects

    x_unit, psi = _wavelet_shape(wavelet)
    bar_px = scale
    label_val = bar_px * px_to_unit
    imgs = ax.get_images()
    img_nx = int(imgs[0].get_array().shape[1]) if imgs else 512
    img_ny = int(imgs[0].get_array().shape[0]) if imgs else 512
    margin = pad_frac
    if unclamp:
        # No clamp — bar grows to its physical scale even past the panel edge.
        # Combine with clip_on=True to truncate cleanly at the axes border.
        bar_frac = bar_px / img_nx
    else:
        bar_frac = min(bar_px / img_nx, 0.40)
        max_bar = (1 - 2 * margin) / 4.0
        bar_frac = min(bar_frac, max_bar)
    bcx = ((1 - margin - 2 * bar_frac) if 'right' in loc
            else (margin + 2 * bar_frac))
    wamp_frac = max(min(bar_px / img_ny * 0.5, 0.06), 0.015)
    label_y = margin
    wcy = label_y + 0.04 + wamp_frac
    fxc = [path_effects.withStroke(linewidth=2.5, foreground='black')]
    fxt = [path_effects.withStroke(linewidth=2.0, foreground='black')]

    wx = np.array([bcx + xv / 2.0 * bar_frac for xv in x_unit])
    wy = np.array([wcy + p * wamp_frac for p in psi])
    cl = (wx >= margin) & (wx <= 1 - margin)
    if cl.any():
        ax.plot(wx[cl], wy[cl], color=color, lw=1.4,
                transform=ax.transAxes, zorder=11, clip_on=clip_on,
                path_effects=fxc)
    if label_val >= 100:
        lbl = f'{label_val:.0f} {unit_str}'
    elif label_val >= 10:
        lbl = f'{label_val:.1f} {unit_str}'
    else:
        lbl = f'{label_val:.2f} {unit_str}'
    ax.text(bcx, label_y, lbl, ha='center', va='bottom',
            fontsize=fontsize, color=color, fontweight='bold',
            transform=ax.transAxes, zorder=13, clip_on=clip_on,
            path_effects=fxt)


def add_map_scalebar(ax,
                      *,
                      px_um: float = 1.0,
                      target_frac: float = 0.20,
                      loc: str = 'lower right',
                      pad_frac: float = 0.04,
                      color: str = 'white',
                      edge: str = 'black',
                      label_unit: str = r'$\mu$m',
                      fontsize: int = 8,
                      bar_height_frac: float = 0.012,
                      length_um: Optional[float] = None) -> None:
    """Overlay a generic spatial scale bar on a 2D map axes.

    Designed for any pixel-coordinate map (KAM, IPF, grain ID, GROD, ...).
    Picks a "nice" round length (1/2/5 * 10^k) close to ``target_frac`` of
    the displayed image width and renders a horizontal bar in axes-fractional
    coordinates so it survives ``aspect='equal'``, ``xticks=[]``, and
    ``axis('off')``.

    For wavelet-domain figures showing a single scale, prefer
    :func:`add_wavelet_scalebar` instead.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    px_um : float
        Pixel size in µm (or whatever physical unit ``label_unit`` names).
    target_frac : float
        Aim length ~= this fraction of image width before snapping to a
        round number.  Ignored when ``length_um`` is supplied.
    length_um : float, optional
        Force this physical length instead of auto-snapping.
    loc : {'lower right', 'lower left', 'upper right', 'upper left'}
    color, edge : str
        Bar fill colour and stroke colour.  White-on-black reads on most
        backgrounds (IPF, hot, viridis, gray, ...).
    """
    import matplotlib.patheffects as path_effects
    from matplotlib.patches import Rectangle

    imgs = ax.get_images()
    if not imgs:
        return
    img_nx = int(imgs[0].get_array().shape[1])
    img_um = max(img_nx * float(px_um), 1e-9)

    if length_um is None or length_um <= 0:
        target_um = target_frac * img_um
        # Snap to 1 / 2 / 5 * 10^k.
        exp = np.floor(np.log10(target_um))
        base = 10.0 ** exp
        for mult in (1.0, 2.0, 5.0, 10.0):
            length_um = mult * base
            if length_um >= target_um:
                break

    bar_frac = float(length_um) / img_um
    bar_frac = min(bar_frac, 1.0 - 2.0 * pad_frac)

    margin = pad_frac
    if 'right' in loc:
        x0 = 1.0 - margin - bar_frac
    else:
        x0 = margin
    if 'upper' in loc:
        y_bar = 1.0 - margin - bar_height_frac
        y_lbl = y_bar - 0.01
        va = 'top'
    else:
        y_bar = margin + 0.025
        y_lbl = y_bar + bar_height_frac + 0.005
        va = 'bottom'

    bar_fx = [path_effects.withStroke(linewidth=2.0, foreground=edge)]
    txt_fx = [path_effects.withStroke(linewidth=1.6, foreground=edge)]

    ax.add_patch(Rectangle((x0, y_bar), bar_frac, bar_height_frac,
                            transform=ax.transAxes, facecolor=color,
                            edgecolor=edge, linewidth=0.6,
                            zorder=12, clip_on=False))
    if length_um >= 100:
        lbl = f'{length_um:.0f} {label_unit}'
    elif length_um >= 10:
        lbl = f'{length_um:.0f} {label_unit}'
    elif length_um >= 1:
        lbl = f'{length_um:.1f} {label_unit}'
    else:
        lbl = f'{length_um:.2f} {label_unit}'
    ax.text(x0 + bar_frac / 2.0, y_lbl, lbl,
            ha='center', va=va, fontsize=fontsize,
            color=color, fontweight='bold',
            transform=ax.transAxes, zorder=13, clip_on=False,
            path_effects=txt_fx)


def precompute_viewer_bundle(
    svd_mode_dict: dict,
    scales: np.ndarray,
    *,
    q_list: Optional[np.ndarray] = None,
    min_chain_n: int = 3,
) -> dict:
    """Build all data structures the WTMMM viewer needs for one mode.

    Parameters
    ----------
    svd_mode_dict : dict
        One value of ``svd_wtmm`` from
        :func:`wtmm_ebsd.alpha_jacobian_twtmm` /
        :func:`wtmm_ebsd.scalar_field_wtmm`.  Must contain keys
        ``'chains'``, ``'holders'``, ``'mod'``, ``'ext_images'``.
    scales : (n_sc,) float
        Wavelet scales (pixel units).
    q_list : (n_q,) float, optional
        q grid for the partition function.  Defaults to
        :data:`wtmm_ebsd.partition.DEFAULT_Q_LIST`.
    min_chain_n : int
        Minimum chain length (in scale steps) to include in the
        viewer's skeleton structures.  Chains shorter than this are
        kept for the chain-segment renderer but excluded from the
        per-chain matrices used by the profile / D(h) panels.

    Returns
    -------
    dict with keys::

        # Chain skeleton
        skel_chains       : list[dict]      length n_ch
        n_ch              : int             # chains with n >= min_chain_n
        chain_n           : (n_ch,) int     length per chain
        chain_anchor      : (n_ch, 2) int   (x0, y0) per chain
        mod_matrix        : (n_ch, n_sc) float64, NaN-padded
        log2_mod_matrix   : (n_ch, n_sc) float64
        running_max_matrix     : (n_ch, n_sc) float64
        log2_running_max_matrix: (n_ch, n_sc) float64

        # Per-scale rendered data
        wt_x_by_scale, wt_y_by_scale       : list of (n_pts,) float64
        wt_logmod_by_scale, wt_arg_by_scale: list of (n_pts,) float64
        wtmmm_to_chain_by_scale            : list of (n_pts,) int
        chain_segments_by_scale            : list of list of segments
        maxima_coords_cache                : list[None|list[(x,y)]]
        chainmax_by_scale                  : list[dict[(x,y), float]]

        # Partition-function dicts (η=0 / integrated frame; caller
        # applies frame correction via fit_hq_Dq_weighted).
        hd_std            : dict from build_hd_from_chains (standard)
        hd_cmax           : dict from build_hd_from_chains (chainmax)
        q_list            : (n_q,) the q grid used

    Notes
    -----
    The returned dict is self-contained — it does not reference any
    notebook globals.
    """
    chains = svd_mode_dict['chains']
    holders = svd_mode_dict.get('holders', [])
    ext_images = svd_mode_dict['ext_images']

    # Self-defending scale handling: if the mode dict carries its own scales
    # (set by the TWTMM driver at compute time), prefer those — that way modes
    # computed with different scale grids stay consistent with their own
    # ext_images even if the caller passes a stale global `scales`. Always
    # bound `nps` to the shorter of (scales, ext_images) so we never index
    # past the end of either.
    if 'scales' in svd_mode_dict and svd_mode_dict['scales'] is not None:
        scales_arr = np.asarray(svd_mode_dict['scales'], dtype=np.float64)
    elif scales is not None:
        scales_arr = np.asarray(scales, dtype=np.float64)
    else:
        scales_arr = np.arange(1, len(ext_images) + 1, dtype=np.float64)
    nps = min(len(scales_arr), len(ext_images))
    if nps < len(scales_arr):
        scales_arr = scales_arr[:nps]
    if q_list is None:
        q_list = DEFAULT_Q_LIST
    q_list = np.asarray(q_list, dtype=np.float64)

    # ------------------------------------------------------------------
    # Skeleton: filter chains with n >= min_chain_n; build per-chain matrices
    # ------------------------------------------------------------------
    skel_chains: list[dict] = []
    for ch in chains:
        n = len(ch['mod'])
        if n < min_chain_n:
            continue
        mods = np.maximum(np.abs(np.asarray(ch['mod'])).astype(np.float64), 1e-30)
        skel_chains.append({
            'log2_s': np.log2(scales_arr[:n]),
            'log2_m': np.log2(mods),
            'mod': mods,
            'n': n,
            'x0': int(ch['x'][0]),
            'y0': int(ch['y'][0]),
            'x': np.asarray(ch['x'], dtype=np.int32),
            'y': np.asarray(ch['y'], dtype=np.int32),
        })
    n_ch = len(skel_chains)
    chain_n = np.array([sc['n'] for sc in skel_chains], dtype=np.int32)
    chain_anchor = np.array(
        [(sc['x0'], sc['y0']) for sc in skel_chains], dtype=np.int32
    ).reshape(n_ch, 2) if n_ch else np.zeros((0, 2), dtype=np.int32)

    # Padded chain modulus matrix (n_ch, n_sc), NaN where chain is short
    mod_matrix = np.full((n_ch, nps), np.nan, dtype=np.float64)
    for ci, sc in enumerate(skel_chains):
        mod_matrix[ci, :sc['n']] = sc['mod']
    mod_safe = np.where(np.isnan(mod_matrix), np.nan,
                         np.maximum(mod_matrix, 1e-30))
    log2_mod_matrix = np.log2(mod_safe)

    # Running max along each chain (cmax convention)
    running_max_matrix = np.full_like(mod_safe, np.nan)
    for ci in range(n_ch):
        nv = chain_n[ci]
        if nv > 0:
            running_max_matrix[ci, :nv] = np.maximum.accumulate(mod_safe[ci, :nv])
    log2_running_max_matrix = np.log2(running_max_matrix)

    # Per-scale chainmax dict: maps (x, y) at scale si -> running max along
    # the chain that visits (x, y) at si.
    chainmax_by_scale: list[dict[tuple[int, int], float]] = [
        {} for _ in range(nps)
    ]
    for ci, sc in enumerate(skel_chains):
        rm = 0.0
        for si in range(sc['n']):
            m = float(sc['mod'][si])
            if m > rm:
                rm = m
            chainmax_by_scale[si][(int(sc['x'][si]), int(sc['y'][si]))] = rm

    # Map (x, y, si) -> chain index ci, used to mark on_vc and
    # wtmmm_to_chain_by_scale below.  Also union across all chains
    # (incl. those filtered out by min_chain_n) so the visualiser still
    # marks short chains as "on a vertical chain" for context.
    chain_pos_to_ci_by_scale: list[dict[tuple[int, int], int]] = [
        {} for _ in range(nps)
    ]
    for ci, sc in enumerate(skel_chains):
        for si in range(sc['n']):
            chain_pos_to_ci_by_scale[si][
                (int(sc['x'][si]), int(sc['y'][si]))
            ] = ci

    # Position set including ALL chains (even those < min_chain_n) so the
    # spatial-chain renderer marks them too.
    chain_pos_set_by_scale: list[set[tuple[int, int]]] = [
        set() for _ in range(nps)
    ]
    for ch in chains:
        for si in range(min(len(ch['mod']), nps)):
            chain_pos_set_by_scale[si].add(
                (int(ch['x'][si]), int(ch['y'][si]))
            )

    # ------------------------------------------------------------------
    # Per-scale extrema buckets + chain segments (rendered map data)
    # ------------------------------------------------------------------
    wt_x_by_scale: list[np.ndarray] = []
    wt_y_by_scale: list[np.ndarray] = []
    wt_logmod_by_scale: list[np.ndarray] = []
    wt_arg_by_scale: list[np.ndarray] = []
    wtmmm_to_chain_by_scale: list[np.ndarray] = []
    chain_segments_by_scale: list[list[tuple]] = []

    for si in range(nps):
        ext = ext_images[si]
        edata = ext.get_extrema_arrays()
        n_extr = int(ext.extr_nb)
        if n_extr == 0:
            wt_x_by_scale.append(np.array([]))
            wt_y_by_scale.append(np.array([]))
            wt_logmod_by_scale.append(np.array([]))
            wt_arg_by_scale.append(np.array([]))
            wtmmm_to_chain_by_scale.append(np.array([], dtype=np.int64))
            chain_segments_by_scale.append([])
            continue

        x_int = edata['x'].astype(np.int64)
        y_int = edata['y'].astype(np.int64)
        pos1d = edata['pos'].astype(np.int64)
        mod_e = np.abs(edata['mod']).astype(np.float64)
        arg_e = edata['arg'].astype(np.float64)
        lx_ext = int(ext.lx)

        # Vectorised on_vc: mark extrema whose (x, y) is in the chain
        # position set at this scale.
        chain_set = chain_pos_set_by_scale[si]
        on_vc = np.zeros(n_extr, dtype=bool)
        for k in range(n_extr):
            if (int(x_int[k]), int(y_int[k])) in chain_set:
                on_vc[k] = True

        wt_x_by_scale.append(x_int[on_vc].astype(np.float64))
        wt_y_by_scale.append(y_int[on_vc].astype(np.float64))
        mods_vc = np.maximum(mod_e[on_vc], 1e-30)
        wt_logmod_by_scale.append(np.log(mods_vc))
        wt_arg_by_scale.append(arg_e[on_vc])

        # wt_to_chain: map each surviving extremum to its chain ci (or -1)
        ci_lookup = chain_pos_to_ci_by_scale[si]
        on_idx = np.where(on_vc)[0]
        wt_to_chain = np.full(len(on_idx), -1, dtype=np.int64)
        for j, k in enumerate(on_idx):
            wt_to_chain[j] = ci_lookup.get(
                (int(x_int[k]), int(y_int[k])), -1
            )
        wtmmm_to_chain_by_scale.append(wt_to_chain)

        # ---- chain segments from ext.get_lines() ----
        # Build dense lookup: 1d-pos -> mod, 1d-pos -> wt_idx (in
        # wt_x_by_scale[si]), 1d-pos -> on_vc flag.
        pos_to_mod = dict(zip(pos1d.tolist(), mod_e.tolist()))
        on_vc_pos = set(int(p) for p in pos1d[on_vc])
        pos_to_wtmmm = dict(zip(pos1d[on_vc].tolist(),
                                 np.arange(int(on_vc.sum())).tolist()))

        scale_segments: list[tuple] = []
        for line in ext.get_lines():
            positions = np.asarray(line['extrema_pos'], dtype=np.int64)
            if positions.size < 2:
                continue
            xs = (positions % lx_ext).astype(np.float64)
            ys = (positions // lx_ext).astype(np.float64)
            # Vectorised mods_along via dense dict lookup
            mods_along = np.fromiter(
                (pos_to_mod.get(int(p), 0.0) for p in positions),
                dtype=np.float64, count=positions.size,
            )

            # Indices of WTMMM points along this line
            line_wtmmm = [j for j, p in enumerate(positions)
                           if int(p) in on_vc_pos]
            if not line_wtmmm:
                coords = list(zip(xs.tolist(), ys.tolist()))
                scale_segments.append((coords, None, len(coords)))
                continue
            line_wt_indices = [pos_to_wtmmm.get(int(positions[j]), -1)
                                for j in line_wtmmm]

            # Split between adjacent WTMMM at the LOCAL MIN of |T|
            split_indices = [0]
            for wi in range(len(line_wtmmm) - 1):
                a, b = line_wtmmm[wi], line_wtmmm[wi + 1]
                if b - a > 1:
                    sub = mods_along[a + 1:b]
                    split_indices.append(a + 1 + int(np.argmin(sub)))
                else:
                    split_indices.append(b)
            split_indices.append(positions.size)
            for seg_i in range(len(split_indices) - 1):
                start = split_indices[seg_i]
                end = min(split_indices[seg_i + 1] + 1, positions.size)
                if end - start < 2:
                    continue
                seg_coords = list(zip(xs[start:end].tolist(),
                                       ys[start:end].tolist()))
                if len(seg_coords) < 2:
                    continue
                wt_idx = line_wt_indices[min(seg_i, len(line_wt_indices) - 1)]
                scale_segments.append((seg_coords, wt_idx, len(seg_coords)))
        chain_segments_by_scale.append(scale_segments)

    # ------------------------------------------------------------------
    # Maxima-line coords per chain (used by the "Maxima lines" overlay)
    # ------------------------------------------------------------------
    maxima_coords_cache: list = []
    for ci in range(n_ch):
        sc = skel_chains[ci]
        if sc['n'] < 2:
            maxima_coords_cache.append(None)
            continue
        coords = [(int(sc['x'][si]), int(sc['y'][si]))
                   for si in range(min(sc['n'], nps))]
        maxima_coords_cache.append(coords)

    # ------------------------------------------------------------------
    # Partition function from chains (correct sums via partition module)
    # ------------------------------------------------------------------
    hd_std, hd_cmax = build_hd_from_chains(chains, scales_arr, q_list=q_list)

    return {
        # skeleton
        'skel_chains':              skel_chains,
        'n_ch':                     n_ch,
        'chain_n':                  chain_n,
        'chain_anchor':             chain_anchor,
        'mod_matrix':               mod_matrix,
        'log2_mod_matrix':          log2_mod_matrix,
        'running_max_matrix':       running_max_matrix,
        'log2_running_max_matrix':  log2_running_max_matrix,
        # per-scale
        'wt_x_by_scale':            wt_x_by_scale,
        'wt_y_by_scale':            wt_y_by_scale,
        'wt_logmod_by_scale':       wt_logmod_by_scale,
        'wt_arg_by_scale':          wt_arg_by_scale,
        'wtmmm_to_chain_by_scale':  wtmmm_to_chain_by_scale,
        'chain_segments_by_scale':  chain_segments_by_scale,
        'maxima_coords_cache':      maxima_coords_cache,
        'chainmax_by_scale':        chainmax_by_scale,
        # partition (correct, via wtmm_ebsd.partition)
        'hd_std':   hd_std,
        'hd_cmax':  hd_cmax,
        'q_list':   q_list,
        # original chain refs (for downstream cells)
        'chains':       chains,
        'holders':      holders,
        'ext_images':   ext_images,
    }


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def viewer_bundle_diagnostics(bundle: dict) -> dict:
    """Quick counts that help spot the v1 'surviving points' bug.

    Returns
    -------
    dict with::
        n_ch:            int        # chains in skel
        n_total_chains:  int        # raw chain count from svd_wtmm
        sum_chain_lengths: int      # Σ over chains of len(ch['mod'])
        sum_wt_pts:      int        # Σ over scales of len(wt_x_by_scale[si])
        sum_segments:    int        # Σ over scales of #chain segments
        partition_n_pts: int        # hd_std['N_a'].sum()  (chain visits used)
    """
    n_total = len(bundle['chains'])
    sum_lengths = sum(len(ch['mod']) for ch in bundle['chains'])
    sum_wt = sum(int(len(a)) for a in bundle['wt_x_by_scale'])
    sum_segs = sum(len(s) for s in bundle['chain_segments_by_scale'])
    pf_n_pts = int(bundle['hd_std']['N_a'].sum())
    return {
        'n_ch':              bundle['n_ch'],
        'n_total_chains':    n_total,
        'sum_chain_lengths': sum_lengths,
        'sum_wt_pts':        sum_wt,
        'sum_segments':      sum_segs,
        'partition_n_pts':   pf_n_pts,
    }
