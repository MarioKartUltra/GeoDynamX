"""Tensor wavelet-transform-modulus-maxima (TWTMM) driver.

Encapsulates v1 cell 39: takes a sample-frame demeaned orientation field
(combined_log_sample, shape (ny, nx, 3)), runs CWT to get kappa = grad g
at every wavelet scale, assembles the Pantleon Nye-tensor alpha components
per scale, computes both the kappa-Jacobian SVD (legacy 3x2) and the
alpha-Jacobian SVD (Pantleon 6x2), runs scalar WTMM on each requested SVD
modulus, extracts chains, and returns a single svd_wtmm dict.

Public API
----------
alpha_jacobian_twtmm(combined_log_sample, scales, *, **opts) -> svd_wtmm
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _tensor_svd_3x2(k11, k12, k21, k22, k31, k32):
    """Per-pixel SVD-based modulus + argument of the 3x2 Jacobian of g.

    sigma_max  = leading singular value
    sigma_min  = trailing singular value
    arg        = orientation of the leading right singular vector (rad)

    All inputs are (ny, nx) float arrays.  Returns three (ny, nx) arrays.
    """
    # J^T J is 2x2 symmetric: A C, B C; eigenvalues = sigma^2.
    # A = sum_i k_{i1}^2;  B = sum_i k_{i1} k_{i2};  C = sum_i k_{i2}^2
    A = k11 * k11 + k21 * k21 + k31 * k31
    C = k12 * k12 + k22 * k22 + k32 * k32
    B = k11 * k12 + k21 * k22 + k31 * k32
    tr = A + C
    df = A - C
    disc = np.sqrt(df * df + 4.0 * B * B)
    lam_max = 0.5 * (tr + disc)
    lam_min = np.maximum(0.5 * (tr - disc), 0.0)
    sigma_max = np.sqrt(lam_max)
    sigma_min = np.sqrt(lam_min)
    arg = 0.5 * np.arctan2(2.0 * B, df)
    return sigma_max.astype(np.float32), sigma_min.astype(np.float32), arg.astype(np.float32)


def _alpha_jacobian_2x2_eig(A_sum, B_sum, C_sum):
    """2x2 eigenvalue closed form for the alpha-Jacobian inner product J^T J,
    where A_sum, B_sum, C_sum are the diagonal/off-diagonal contributions
    summed across the 6 alpha components."""
    tr = A_sum + C_sum
    df = A_sum - C_sum
    disc = np.sqrt(df * df + 4.0 * B_sum * B_sum)
    lam_max = 0.5 * (tr + disc)
    lam_min = np.maximum(0.5 * (tr - disc), 0.0)
    sigma_max = np.sqrt(lam_max).astype(np.float32)
    sigma_min = np.sqrt(lam_min).astype(np.float32)
    arg = (0.5 * np.arctan2(2.0 * B_sum, df)).astype(np.float32)
    return sigma_max, sigma_min, arg


def alpha_jacobian_twtmm(
    combined_log_sample: np.ndarray,
    scales: np.ndarray,
    *,
    cwt_2d_f32,                      # injected callable (CWT operator)
    xsm,                             # injected xsmurf_wrapper module
    chain_and_extract,               # injected chain_and_extract callable
    grain_id: Optional[np.ndarray] = None,
    px_um: float = 1.0,
    fracint_alpha: float = 1.0,
    pad: int = 32,
    wavelet: str = 'gaussian',
    wavelet_hessian: str = 'gaussian',
    use_wavelet_hessian: bool = True,
    chain_method: str = 'greedy',
    similitude: float = 0.8,
    border_pct: float = 1.0,
    dist_frac: float = 0.1,
    smooth_chain: bool = True,
    thresh: float = 1e-3,
    a_min: float = 1.0,
    n_oct: int = 4,
    n_vox: int = 11,
    svd_modes: tuple[str, ...] = (
        'alpha_sigma_max', 'alpha_sigma_min',
        'sigma_max', 'sigma_min', 'modL', 'modT',
    ),
    verbose: bool = True,
) -> dict:
    """Run the alpha-Jacobian TWTMM pipeline end-to-end.

    Parameters (high level)
    -----------------------
    combined_log_sample : (ny, nx, 3) float
        SAMPLE-frame per-grain demeaned orientation field.  Built by
        wtmm_ebsd.demean.compute_log_orientation_per_grain.
    scales : (n_sc,) float
        Wavelet scales in pixel units.
    cwt_2d_f32 : callable
        CWT operator with signature
            cwt_2d_f32(field, scales, pad, derivs, wavelet, verbose)
            returning a dict with keys 'dx', 'dy' (and 'dxx','dxy','dyy' if
            derivs='all').
    xsm, chain_and_extract : injected
        xsmurf_wrapper module + the anchor_chain.chain_and_extract function.
    grain_id : (ny, nx) int, optional
        For per-grain chain bucketing.  If None, all chains go to grain 0.
    fracint_alpha : float
        Fractional integration exponent applied to derivatives at each
        scale.  v1 default = 1.
    use_wavelet_hessian : bool
        True -> exact alpha-Jacobian via 2nd-derivative CWT (`derivs='all'`).
        False -> pixel-level np.gradient on the alpha fields.
    svd_modes : tuple of str
        Which SVD-modulus modes to run scalar WTMM on.  Names must be in:
        {alpha_sigma_max, alpha_sigma_min, sigma_max, sigma_min, modL, modT}.
    verbose : bool

    Returns
    -------
    svd_wtmm : dict
        For each mode in `svd_modes`:
            svd_wtmm[mode] = {
                'chains':          list of chain dicts,
                'holders':         list of {'h': ...},
                'chains_by_grain': dict[int, list[chain]],
                'mod':             (n_sc, ny, nx) float32   modulus,
                'ext_images':      list of length n_sc      raw ext images,
            }
    """
    if combined_log_sample.ndim != 3 or combined_log_sample.shape[-1] != 3:
        raise ValueError(
            f"combined_log_sample must be (ny, nx, 3); got {combined_log_sample.shape}"
        )
    work_field = combined_log_sample
    n_sc = len(scales)
    ny, nx = work_field.shape[:2]

    if verbose:
        print('CWT (3 rotation components) -> kappa_ij(a)...')
    derivs = [cwt_2d_f32(work_field[..., c], scales, pad=pad, derivs='first',
                          wavelet=wavelet, verbose=False) for c in range(3)]
    if use_wavelet_hessian:
        if verbose:
            print(f'CWT (3 components) -> wavelet Hessian (derivs=all, '
                  f'{wavelet_hessian})...')
        derivs_hess = [cwt_2d_f32(work_field[..., c], scales, pad=pad,
                                   derivs='all', wavelet=wavelet_hessian,
                                   verbose=False)
                       for c in range(3)]
    else:
        derivs_hess = None

    if fracint_alpha != 0:
        if verbose:
            print(f'Pseudo-fractional integration eta={fracint_alpha}...')
        for c in range(3):
            for si, scale in enumerate(scales):
                f = np.float32(scale ** fracint_alpha)
                derivs[c]['dx'][si] *= f
                derivs[c]['dy'][si] *= f
                if use_wavelet_hessian:
                    derivs_hess[c]['dxx'][si] *= f
                    derivs_hess[c]['dxy'][si] *= f
                    derivs_hess[c]['dyy'][si] *= f

    # Allocate svd field arrays
    svd_fields = {
        'sigma_max':       np.zeros((n_sc, ny, nx), dtype=np.float32),
        'sigma_min':       np.zeros((n_sc, ny, nx), dtype=np.float32),
        'modL':            np.zeros((n_sc, ny, nx), dtype=np.float32),
        'modT':            np.zeros((n_sc, ny, nx), dtype=np.float32),
        'arg':             np.zeros((n_sc, ny, nx), dtype=np.float32),
        'alpha_sigma_max': np.zeros((n_sc, ny, nx), dtype=np.float32),
        'alpha_sigma_min': np.zeros((n_sc, ny, nx), dtype=np.float32),
        'alpha_arg':       np.zeros((n_sc, ny, nx), dtype=np.float32),
    }

    if verbose:
        print('Building alpha fields + 6x2 Jacobian SVD per scale...')
    for si in range(n_sc):
        # kappa_ij = d g_i / d x_j
        k11 = derivs[0]['dx'][si]; k12 = derivs[0]['dy'][si]
        k21 = derivs[1]['dx'][si]; k22 = derivs[1]['dy'][si]
        k31 = derivs[2]['dx'][si]; k32 = derivs[2]['dy'][si]

        # Pantleon alpha components (zero elastic strain, Eqs. 11 & 13)
        a12_v = k21
        a13_v = k31
        a21_v = k12
        a23_v = k32
        a33_v = -(k11 + k22)
        adf_v = k11 - k22

        # Legacy 3x2 SVD diagnostics (kappa Jacobian)
        smax, smin, karg = _tensor_svd_3x2(k11, k12, k21, k22, k31, k32)
        svd_fields['sigma_max'][si] = smax
        svd_fields['sigma_min'][si] = smin
        svd_fields['arg'][si]       = karg
        svd_fields['modL'][si] = np.sqrt(
            k11 ** 2 + k22 ** 2 + 0.25 * (k21 + k12) ** 2 + k31 ** 2 + k32 ** 2
        ).astype(np.float32)
        svd_fields['modT'][si] = (0.5 * np.abs(k21 - k12)).astype(np.float32)

        # 6x2 alpha-Jacobian SVD via 2x2 eigendecomposition of J^T J
        A_sum = np.zeros_like(a12_v); B_sum = np.zeros_like(a12_v)
        C_sum = np.zeros_like(a12_v)
        if use_wavelet_hessian:
            g1_xx = derivs_hess[0]['dxx'][si]; g1_xy = derivs_hess[0]['dxy'][si]; g1_yy = derivs_hess[0]['dyy'][si]
            g2_xx = derivs_hess[1]['dxx'][si]; g2_xy = derivs_hess[1]['dxy'][si]; g2_yy = derivs_hess[1]['dyy'][si]
            g3_xx = derivs_hess[2]['dxx'][si]; g3_xy = derivs_hess[2]['dxy'][si]; g3_yy = derivs_hess[2]['dyy'][si]
            da_x_list = [g2_xx, g3_xx, g1_xy, g3_xy, -(g1_xx + g2_xy), (g1_xx - g2_xy)]
            da_y_list = [g2_xy, g3_xy, g1_yy, g3_yy, -(g1_xy + g2_yy), (g1_xy - g2_yy)]
        else:
            da_x_list = []
            da_y_list = []
            for arr in (a12_v, a13_v, a21_v, a23_v, a33_v, adf_v):
                da_x_list.append(np.gradient(arr, axis=1).astype(np.float32))
                da_y_list.append(np.gradient(arr, axis=0).astype(np.float32))
        for da_x, da_y in zip(da_x_list, da_y_list):
            A_sum += (da_x * da_x).astype(np.float32)
            B_sum += (da_x * da_y).astype(np.float32)
            C_sum += (da_y * da_y).astype(np.float32)

        a_smax, a_smin, a_arg = _alpha_jacobian_2x2_eig(A_sum, B_sum, C_sum)
        svd_fields['alpha_sigma_max'][si] = a_smax
        svd_fields['alpha_sigma_min'][si] = a_smin
        svd_fields['alpha_arg'][si]       = a_arg

    # Map mode -> argument field
    arg_for_mode = {
        'sigma_max':       'arg',
        'sigma_min':       'arg',
        'modL':            'arg',
        'modT':            'arg',
        'alpha_sigma_max': 'alpha_arg',
        'alpha_sigma_min': 'alpha_arg',
    }
    # Auto-fallback: if a mode has no native arg field, fall back to 'arg'
    # (the κ-Jacobian principal direction).  Any extension modes will use this.

    svd_wtmm: dict = {}
    for mode in svd_modes:
        if mode not in svd_fields:
            if verbose:
                print(f'  skip mode {mode!r} (not built)')
            continue
        if mode in ('arg', 'alpha_arg'):
            continue
        if verbose:
            print(f'  WTMM on {mode}...')
        mod = svd_fields[mode]
        arg = svd_fields[arg_for_mode[mode]]
        ext_images = []
        for si, scale in enumerate(scales):
            dx_r = (mod[si] * np.cos(arg[si])).astype(np.float32)
            dy_r = (mod[si] * np.sin(arg[si])).astype(np.float32)
            ext, _, _ = xsm.wtmm2d(
                xsm.XImage.from_numpy(dx_r),
                xsm.XImage.from_numpy(dy_r),
                scale, thresh=thresh)
            ext_images.append(ext)

        chains = chain_and_extract(
            ext_images, scales, method=chain_method,
            a_min=a_min, n_oct=n_oct, n_vox=n_vox,
            similitude=similitude, smooth=smooth_chain,
            border_percent=border_pct, dist_frac=dist_frac)

        holders = xsm.compute_holder_exponents(chains)
        chains_by_grain: dict[int, list] = {}
        for ch in chains:
            x0, y0 = int(ch['x'][0]), int(ch['y'][0])
            if grain_id is not None and 0 <= y0 < ny and 0 <= x0 < nx:
                gid0 = int(grain_id[y0, x0])
            else:
                gid0 = 0
            chains_by_grain.setdefault(gid0, []).append(ch)

        svd_wtmm[mode] = {
            'chains': chains,
            'holders': holders,
            'chains_by_grain': chains_by_grain,
            'mod': mod,
            'ext_images': ext_images,
            'scales': np.asarray(scales, dtype=np.float64).copy(),
        }
        if verbose and holders:
            h_vals = [h['h'] for h in holders]
            if h_vals:
                print(f'    {len(chains)} chains, h median = {np.median(h_vals):.3f}')

    return svd_wtmm


def scalar_field_wtmm(
    field: np.ndarray,
    scales: np.ndarray,
    *,
    cwt_2d_f32,
    xsm,
    chain_and_extract,
    grain_id: Optional[np.ndarray] = None,
    fracint_alpha: float = 1.0,
    pad: int = 32,
    wavelet: str = 'gaussian',
    chain_method: str = 'greedy',
    similitude: float = 0.8,
    border_pct: float = 1.0,
    dist_frac: float = 0.1,
    smooth_chain: bool = True,
    thresh: float = 1e-3,
    a_min: float = 1.0,
    n_oct: int = 4,
    n_vox: int = 11,
    verbose: bool = True,
) -> dict:
    """Run scalar 2D WTMM on a single (ny, nx) field.

    Standard 2D WTMM: CWT first derivatives -> gradient modulus + arg ->
    NMS extrema -> chain extraction -> Hölder slopes.  Returns the same
    schema as :func:`alpha_jacobian_twtmm` so downstream cells (partition
    function, viewer, spectra) can drive scalar fields with no API change.

    Parameters
    ----------
    field : (ny, nx) float
        The scalar field to analyse (e.g. a slip-system GND density map,
        the total GND, dist_to_gb, theta_attractor, KAM, etc.).
    scales : (n_sc,) float
        Wavelet scales in pixel units.
    cwt_2d_f32, xsm, chain_and_extract : injected
        See :func:`alpha_jacobian_twtmm`.
    grain_id : (ny, nx) int, optional
        For per-grain chain bucketing.
    fracint_alpha : float
        Pseudo-fractional integration order η.  Multiplies dx, dy by
        a^η at each scale (Wendt et al. 2009).
    Other args are 1:1 with :func:`alpha_jacobian_twtmm`.

    Returns
    -------
    dict with one key, 'scalar', mapping to the standard mode dict::
        {
            'chains':          list of chain dicts,
            'holders':         list of {'h': ...},
            'chains_by_grain': dict[int, list[chain]],
            'mod':             (n_sc, ny, nx) float32  gradient modulus,
            'ext_images':      list of XExtImage,
        }

    Notes
    -----
    For multi-field workflows, call this once per field and merge the
    'scalar' dict under your own key in the global ``svd_wtmm`` registry,
    e.g. ``svd_wtmm['gnd_total'] = scalar_field_wtmm(rho_tot, ...)['scalar']``.
    """
    if field.ndim != 2:
        raise ValueError(f"field must be 2D (ny, nx); got shape {field.shape}")
    field = np.ascontiguousarray(field, dtype=np.float32)
    n_sc = len(scales)
    ny, nx = field.shape

    if verbose:
        print(f'  Scalar WTMM on field shape {field.shape}, {n_sc} scales')

    derivs = cwt_2d_f32(field, scales, pad=pad, derivs='first',
                         wavelet=wavelet, verbose=False)

    if fracint_alpha != 0:
        for si, scale in enumerate(scales):
            f = np.float32(scale ** fracint_alpha)
            derivs['dx'][si] *= f
            derivs['dy'][si] *= f

    mod = np.zeros((n_sc, ny, nx), dtype=np.float32)
    ext_images = []
    for si, scale in enumerate(scales):
        dx = derivs['dx'][si].astype(np.float32)
        dy = derivs['dy'][si].astype(np.float32)
        mod[si] = np.sqrt(dx * dx + dy * dy)
        ext, _, _ = xsm.wtmm2d(
            xsm.XImage.from_numpy(dx),
            xsm.XImage.from_numpy(dy),
            scale, thresh=thresh)
        ext_images.append(ext)

    chains = chain_and_extract(
        ext_images, scales, method=chain_method,
        a_min=a_min, n_oct=n_oct, n_vox=n_vox,
        similitude=similitude, smooth=smooth_chain,
        border_percent=border_pct, dist_frac=dist_frac)

    holders = xsm.compute_holder_exponents(chains)

    chains_by_grain: dict[int, list] = {}
    for ch in chains:
        x0, y0 = int(ch['x'][0]), int(ch['y'][0])
        if grain_id is not None and 0 <= y0 < ny and 0 <= x0 < nx:
            gid0 = int(grain_id[y0, x0])
        else:
            gid0 = 0
        chains_by_grain.setdefault(gid0, []).append(ch)

    if verbose and holders:
        h_vals = [h['h'] for h in holders if 'h' in h]
        if h_vals:
            print(f'    {len(chains)} chains, h median = {np.median(h_vals):.3f}')

    return {
        'scalar': {
            'chains':          chains,
            'holders':         holders,
            'chains_by_grain': chains_by_grain,
            'mod':             mod,
            'ext_images':      ext_images,
        }
    }
