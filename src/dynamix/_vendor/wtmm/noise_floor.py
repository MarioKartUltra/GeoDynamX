"""GND-density noise-floor estimation following Wallis et al. 2019 (JGR Solid Earth 124, 6337-6358).

Two estimators:

1. `simple_noise_floor` — Eq. 13 of Wallis 2019:
       rho_min = theta / (b * d)
   One scalar per slip-system family; ignores the rotation by which the slip system
   appears in the section. Use for quick magnitude-of-effect comparisons.

2. `noise_floor_per_grain` — Eq. 14 of Wallis 2019:
   Plug a synthetic orientation gradient phi = theta/d into the same NNLS/LP
   solver used to invert the real data, with each grain's reference orientation
   fed as the rotation. Produces per-grain, per-family noise floors that vary
   with crystal orientation by 1-3 orders of magnitude (e.g., ~10^12 m^-2 for
   a (010)[100] edge in an olivine grain with [100] in-plane vs ~10^14 m^-2
   for the same family in a grain with [100] near surface-normal; Wallis 2019
   Fig. 8).

Reference angular-precision values (theta in radians):
    HR-EBSD                     ~3e-4   rad   (Wilkinson et al. 2006a)
    Conventional Hough-EBSD     ~8.7e-3 rad   (= 0.5 deg; Humphreys 1999)

The same module also provides masking helpers so you can threshold a measured
gnd_per_family map by k * noise_floor before running the WTMM partition
function -- a Kantelhardt-style control test for whether the multifractal
spectrum survives a noise-aware floor.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Default theta values (rad). Override via the `theta` argument.
# ---------------------------------------------------------------------------
THETA_HR_EBSD       = 3e-4    # Wilkinson 2006a / Wallis 2016
THETA_CONV_EBSD_DEG = 0.5     # Humphreys 1999
THETA_CONV_EBSD     = float(np.deg2rad(THETA_CONV_EBSD_DEG))   # ~8.7e-3 rad


# ---------------------------------------------------------------------------
# Eq. 13 -- single scalar per family
# ---------------------------------------------------------------------------

def simple_noise_floor(burgers_m, step_m, theta=THETA_CONV_EBSD):
    """rho_min = theta / (b * d).  All inputs SI units (m, m, rad).
    Returns rho_min in m^-2."""
    return float(theta) / (float(burgers_m) * float(step_m))


def simple_noise_floor_table(burgers_dict, step_um, theta=THETA_CONV_EBSD):
    """Apply Eq. 13 to a dict of {family_name: |b|_m} pairs.

    Parameters
    ----------
    burgers_dict : dict
        e.g. ``BURGERS_MAGNITUDE_M['Quartz']`` from wtmm.slip_systems.
    step_um : float
        EBSD step size in micrometres.
    theta : float
        Angular precision in rad.  Default = conventional Hough-EBSD ~0.5 deg.

    Returns
    -------
    dict
        ``{family_name: rho_min_m^-2}``.
    """
    step_m = float(step_um) * 1e-6
    return {fam: simple_noise_floor(b_mag, step_m, theta=theta)
            for fam, b_mag in burgers_dict.items()}


# ---------------------------------------------------------------------------
# Eq. 14 -- per-grain, orientation-aware
# ---------------------------------------------------------------------------

def _quat_to_rotmat(q):
    """Hamilton-convention unit quaternion (w, x, y, z) -> 3x3 rotation matrix
    that maps crystal-frame vectors to sample-frame vectors (matching orix
    convention used elsewhere in this codebase)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _synthetic_alpha_sample(theta, step_m, mode='uniform', n_mc=64, rng=None):
    """Build a synthetic Nye tensor alpha in the SAMPLE frame whose components
    are at the noise-floor magnitude phi = theta/step.

    Parameters
    ----------
    mode : {'uniform', 'mc'}
        'uniform' -- all 6 measurable components = +phi (deterministic; Wallis 2019).
        'mc'      -- iid Gaussian draws with std phi; n_mc replicates returned.

    Returns
    -------
    ndarray of shape (3, 3) for 'uniform' or (n_mc, 3, 3) for 'mc'.
    """
    phi = float(theta) / float(step_m)
    if mode == 'uniform':
        a = np.zeros((3, 3), dtype=np.float64)
        # Six measurable components per Pantleon 2008 / Wallis Eq. 6:
        # alpha_12, alpha_13, alpha_21, alpha_23, alpha_33, (alpha_11 - alpha_22)
        # Filling all entries to phi gives a worst-case floor; sign convention is
        # irrelevant for a non-negative solver (NNLS/LP) since the absolute
        # magnitude is what's read by the geometry matrix.
        a[0, 1] = phi
        a[0, 2] = phi
        a[1, 0] = phi
        a[1, 2] = phi
        a[2, 2] = phi
        a[0, 0] =  0.5 * phi
        a[1, 1] = -0.5 * phi   # so alpha_11 - alpha_22 = phi
        return a
    elif mode == 'mc':
        if rng is None:
            rng = np.random.default_rng(0)
        out = np.zeros((n_mc, 3, 3), dtype=np.float64)
        # Same 6 components, each drawn iid from N(0, phi)
        idx = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 2)]
        for k in range(n_mc):
            for (i, j) in idx:
                out[k, i, j] = rng.normal(0.0, phi)
            d11 = rng.normal(0.0, phi / np.sqrt(2.0))
            d22 = -d11
            out[k, 0, 0] = d11
            out[k, 1, 1] = d22
        return out
    else:
        raise ValueError(f'Unknown mode {mode!r}; expected uniform or mc')


def noise_floor_per_grain(grain_info, ebsd_data, slip_db, phase_name_lookup,
                           burgers_magnitude_m,
                           step_um=None,
                           theta=THETA_CONV_EBSD,
                           solver='lp',
                           line_energy='character',
                           poisson_ratio=0.08,
                           mode='uniform',
                           n_mc=64,
                           unit_m_inv2=True,
                           rng=None,
                           progress=False):
    """Per-grain, per-family GND-density noise floor (Wallis 2019 Eq. 14).

    For each grain in ``grain_info``, builds a synthetic Nye tensor at the
    noise-floor magnitude phi = theta/step in the SAMPLE frame, rotates it to
    the crystal frame using the grain's reference orientation, and runs the
    same NNLS/LP slip-system solver used in the production pipeline. Output
    is a dict of per-(grain, family) noise-floor scalars in m^-2 (or rad/pixel
    if ``unit_m_inv2=False``).

    Parameters
    ----------
    grain_info : dict
        Per-grain reference orientations (from cell 12 of pergrain notebook):
        ``{grain_id: {'phase_id': int, 'q_ref': (4,) quaternion, ...}}``.
    ebsd_data : dict
        EBSD dataset; only used to look up ``step_um`` if not provided.
    slip_db : dict
        Slip-system database; e.g. ``wtmm.slip_systems.SLIP_SYSTEMS_DB``.
    phase_name_lookup : callable
        ``phase_id -> phase_name`` (must match keys in slip_db).
    burgers_magnitude_m : dict
        e.g. ``wtmm.slip_systems.BURGERS_MAGNITUDE_M``.
    step_um : float or None
        EBSD step size (um). If None, taken from ebsd_data['meta']['step_um']
        or ``ebsd_data.get('step_um')``.
    theta : float
        Angular precision (rad). Default conventional Hough-EBSD value.
    solver, line_energy, poisson_ratio : passthrough to
        ``wtmm.slip_systems.resolve_gnd_per_pixel``.
    mode : {'uniform', 'mc'}
        'uniform' -- single deterministic synthetic alpha (Wallis convention).
        'mc'      -- average over ``n_mc`` Gaussian noise realizations to get
                     a smoother, less worst-case-biased floor.
    unit_m_inv2 : bool
        If True (default), convert raw rad/pixel output to m^-2 using
        ``rho_si = rho_raw / (px_size_m * |b|_m)``.

    Returns
    -------
    dict
        ``{(grain_id, family_name): rho_floor_m^-2}`` (or rad/pixel).
    """
    from .slip_systems import (build_geometry_matrix, line_energy_weights,
                                _solve_one_pixel)

    if step_um is None:
        meta = ebsd_data.get('meta', {}) if isinstance(ebsd_data, dict) else {}
        step_um = meta.get('step_um') or ebsd_data.get('step_um')
        if step_um is None:
            raise ValueError("step_um not provided and not present in ebsd_data['meta']")
    step_m = float(step_um) * 1e-6
    phi = float(theta) / step_m

    if rng is None:
        rng = np.random.default_rng(0)

    # Build synthetic alpha in sample frame (3x3 or N_mc x 3 x 3)
    alpha_sample = _synthetic_alpha_sample(theta, step_m, mode=mode,
                                            n_mc=n_mc, rng=rng)

    out = {}
    for gid, info in grain_info.items():
        pid   = int(info.get('phase_id', 0))
        q_ref = np.asarray(info.get('q_ref'), dtype=np.float64)
        if q_ref is None or q_ref.shape != (4,):
            continue
        pname = phase_name_lookup(pid) if callable(phase_name_lookup) else \
                phase_name_lookup.get(pid)
        if pname is None or pname not in slip_db:
            continue

        systems = slip_db[pname]
        n_sys   = len(systems)
        A   = build_geometry_matrix(systems)
        u_w = line_energy_weights(systems, mode=line_energy,
                                   poisson_ratio=poisson_ratio)
        R   = _quat_to_rotmat(q_ref)

        # Rotate alpha to crystal frame: a_c = R^T a_s R
        if alpha_sample.ndim == 2:
            a_c = R.T @ alpha_sample @ R
            alpha_vecs = a_c[:2, :].reshape(1, 6)   # 6-vector solver input
        else:
            a_c = np.einsum('ji,njk,kl->nil', R, alpha_sample, R)
            alpha_vecs = a_c[:, :2, :].reshape(-1, 6)

        # Solve and aggregate
        rho_accum = np.zeros(n_sys, dtype=np.float64)
        for k in range(alpha_vecs.shape[0]):
            rho = _solve_one_pixel(A, alpha_vecs[k], solver, u_w, {})
            rho_accum += np.abs(rho)        # abs: noise is unsigned in magnitude
        rho_avg = rho_accum / max(alpha_vecs.shape[0], 1)

        # Aggregate by family + optional m^-2 conversion
        phase_bmag = burgers_magnitude_m.get(pname, {}) if unit_m_inv2 else None
        fam_total = {}
        for s, ss in enumerate(systems):
            fam = ss.get('family', ss['name'])
            v = float(rho_avg[s])
            if unit_m_inv2 and phase_bmag is not None:
                b_mag = phase_bmag.get(fam)
                if b_mag is not None:
                    v = v / (step_m * b_mag)
            fam_total[fam] = fam_total.get(fam, 0.0) + v
        for fam, v in fam_total.items():
            out[(gid, fam)] = v
        if progress and (gid % 50 == 0):
            print(f'  grain {gid}: noise floor for {len(fam_total)} families, '
                  f'phi = {phi:.3e} rad/m')

    return out


# ---------------------------------------------------------------------------
# Masking + SNR helpers
# ---------------------------------------------------------------------------

def mask_below_floor(gnd_per_family, noise_floor_per_grain, grain_id,
                      k=1.0, fill=0.0):
    """Zero out (or NaN out) pixels where the measured GND density is below
    k * noise_floor for that grain and family.

    Parameters
    ----------
    gnd_per_family : dict
        ``{(phase_id, family_name): (ny, nx) array}`` -- the production output.
    noise_floor_per_grain : dict
        ``{(grain_id, family_name): float}`` from ``noise_floor_per_grain``.
    grain_id : ndarray (ny, nx)
        Per-pixel grain ID map.
    k : float
        Multiplier on noise floor; pixels with rho < k * floor are masked.
        k=1 -> Wallis convention; k=2 or 3 -> more conservative.
    fill : float
        Value to assign masked pixels (0.0, np.nan, etc.).

    Returns
    -------
    dict
        Same keys as input ``gnd_per_family``, with masked maps.
    """
    out = {}
    for (pid, fam), rho_map in gnd_per_family.items():
        masked = rho_map.copy()
        # Build per-pixel floor map by looking up each grain's floor
        floor_map = np.zeros_like(rho_map, dtype=np.float64)
        for gid in np.unique(grain_id):
            if gid == 0: continue
            floor = noise_floor_per_grain.get((int(gid), fam))
            if floor is None: continue
            floor_map[grain_id == gid] = floor
        below = (masked < k * floor_map) & (floor_map > 0)
        masked = np.where(below, fill, masked)
        out[(pid, fam)] = masked.astype(rho_map.dtype)
    return out


def snr_map(gnd_per_family, noise_floor_per_grain, grain_id, eps=1e-30):
    """Per-family signal-to-noise map: rho / floor, with floor inferred per
    grain. Pixels in grains with no floor estimate get NaN.

    Returns
    -------
    dict
        ``{(phase_id, family_name): (ny, nx) SNR map}``.
    """
    out = {}
    for (pid, fam), rho_map in gnd_per_family.items():
        floor_map = np.full_like(rho_map, np.nan, dtype=np.float64)
        for gid in np.unique(grain_id):
            if gid == 0: continue
            floor = noise_floor_per_grain.get((int(gid), fam))
            if floor is None: continue
            floor_map[grain_id == gid] = floor
        snr = rho_map / np.where(np.isfinite(floor_map), floor_map + eps, np.nan)
        out[(pid, fam)] = snr.astype(np.float32)
    return out


__all__ = [
    'THETA_HR_EBSD',
    'THETA_CONV_EBSD',
    'THETA_CONV_EBSD_DEG',
    'simple_noise_floor',
    'simple_noise_floor_table',
    'noise_floor_per_grain',
    'mask_below_floor',
    'snr_map',
]
