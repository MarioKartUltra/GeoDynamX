"""Niezgoda-style 2-point spatial correlations + ensemble PCA + RVE checks.

Implements the FFT-based formula

    f^{np}_r = (1/S) Σ_s m^n_s · m^p_{s+r}                  (periodic)
            = (1/S_r) Σ_s m^n_s · m^p_{s+r}                  (non-periodic)

where m^n_s ∈ [0, 1] is the local-state probability density at spatial bin
s for state n (binary: 0/1; one-hot of categorical: stack of binaries).
This is the same algorithm as pymks and Tony Fast's SpatialStatisticsFFT,
both following Niezgoda et al. Acta Materialia 56:5285 (2008).

Public API
----------
two_point(m_n, m_p)                — 2D auto/cross 2-pt correlation
two_point_radial(f_r)              — azimuthal average
discretise_field(field, n_bins)    — continuous → categorical
one_hot_stack(cls, n_bins)         — int class map → bool stack
correlation_matrix(masks)          — N×N pairwise 2-pt
coherence_length(f_r, phi)         — RVE / scan-size adequacy check
pca_on_2pt(f_r_stack, n_pc)        — PCA on flattened 2-pt for ensemble
phase_specific_correlation(...)    — convenience wrapper for phase × scalar
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ============================================================
# Core 2-point statistics
# ============================================================

def two_point(
    m_n: np.ndarray,
    m_p: np.ndarray | None = None,
    *,
    periodic: bool = False,
    centred: bool = True,
) -> np.ndarray:
    """2-point auto- or cross-correlation of two indicator/probability fields.

    Parameters
    ----------
    m_n : (ny, nx) float or bool
        Local-state probability density for state n.  Boolean is fine.
    m_p : same shape, optional
        Local-state for state p; defaults to m_n (autocorrelation).
    periodic : bool
        If False, normalise by the per-lag valid-placement count S_r.
    centred : bool
        If True, fftshift so r=0 is at the centre of the output.

    Returns
    -------
    f_r : (ny, nx) float64
        f^{np}_r for every lag r.
    """
    a = np.asarray(m_n, dtype=np.float64)
    b = a if m_p is None else np.asarray(m_p, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f'shapes differ: {a.shape} vs {b.shape}')

    A = np.fft.fftn(a)
    B = np.fft.fftn(b)
    f = np.fft.ifftn(np.conj(A) * B).real

    if periodic:
        f = f / a.size
    else:
        # S_r = (S1 - |r1|)(S2 - |r2|) — non-periodic normalisation.
        # Build the per-axis valid-count vectors via fftshift trick.
        ny, nx = a.shape
        ry = np.minimum(np.arange(ny), ny - np.arange(ny))
        rx = np.minimum(np.arange(nx), nx - np.arange(nx))
        S_y = (ny - ry).astype(np.float64)
        S_x = (nx - rx).astype(np.float64)
        denom = np.outer(S_y, S_x)
        f = f / np.maximum(denom, 1.0)

    if centred:
        f = np.fft.fftshift(f)
    return f


def two_point_radial(
    f_r: np.ndarray,
    *,
    n_bins: int = 60,
    r_max_frac: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally average a centred 2D 2-point map.

    Parameters
    ----------
    f_r : (ny, nx) float, with r=0 at the centre (i.e. fftshifted).
    n_bins : int
    r_max_frac : float
        Maximum radius as fraction of min(ny, nx) / 2.

    Returns
    -------
    r_centre_px : (n_bins,) bin centres in pixels (multiply by step for µm)
    f_radial    : (n_bins,) radial average of f_r in each bin
    """
    f = np.asarray(f_r, dtype=np.float64)
    ny, nx = f.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.indices((ny, nx))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = r_max_frac * min(ny, nx)
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_idx = np.digitize(r.ravel(), bin_edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    sums   = np.bincount(bin_idx[valid], weights=f.ravel()[valid], minlength=n_bins)
    counts = np.bincount(bin_idx[valid], minlength=n_bins)
    f_rad = sums / np.maximum(counts, 1)
    r_centre = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return r_centre, f_rad


# ============================================================
# Discretisation
# ============================================================

def discretise_field(
    field: np.ndarray,
    *,
    n_bins: int = 3,
    edges: np.ndarray | None = None,
    quantile_edges: bool = True,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin a continuous scalar into ``n_bins`` ordinal classes.

    Parameters
    ----------
    field : (ny, nx) float
    n_bins : int
    edges : (n_bins + 1,) float, optional
        Explicit bin edges; if None, computed from ``quantile_edges``.
    quantile_edges : bool
        If True (default), edges at equal-count quantiles → balanced classes.
        If False, linspace(field.min(), field.max(), n_bins+1).
    valid_mask : (ny, nx) bool, optional
        Pixels considered for edge estimation AND output (False → cls = -1).

    Returns
    -------
    cls : (ny, nx) int8 — class index ∈ {0, ..., n_bins-1}; -1 outside valid_mask
    edges_used : (n_bins + 1,) float
    """
    f = np.asarray(field, dtype=np.float64)
    if valid_mask is None:
        valid_mask = np.isfinite(f)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    sample = f[valid_mask]
    if sample.size == 0:
        return np.full(f.shape, -1, dtype=np.int8), np.zeros(n_bins + 1)
    if edges is None:
        if quantile_edges:
            edges = np.quantile(sample, np.linspace(0, 1, n_bins + 1))
        else:
            edges = np.linspace(sample.min(), sample.max(), n_bins + 1)
    edges = np.asarray(edges, dtype=np.float64)
    # np.digitize is exclusive on the right end; clip for safety
    cls = np.full(f.shape, -1, dtype=np.int8)
    cls_valid = np.digitize(sample, edges[1:-1])
    cls_valid = np.clip(cls_valid, 0, n_bins - 1).astype(np.int8)
    cls[valid_mask] = cls_valid
    return cls, edges


def one_hot_stack(cls: np.ndarray, n_bins: int) -> np.ndarray:
    """(ny, nx) int class map → (n_bins, ny, nx) boolean one-hot stack.

    Pixels with class == -1 (excluded) are False in every channel.
    """
    out = np.zeros((n_bins,) + cls.shape, dtype=bool)
    for k in range(n_bins):
        out[k] = (cls == k)
    return out


# ============================================================
# Pairwise correlation matrix
# ============================================================

def correlation_matrix(
    masks: dict[str, np.ndarray],
    *,
    periodic: bool = False,
    radial_bins: int = 60,
) -> dict[tuple[str, str], dict]:
    """Compute auto + cross 2-pt + radial average for every pair (n, p) of masks.

    Returns
    -------
    out[(n, p)] = {
        'f_r':     (ny, nx) 2D correlation map
        'r_px':    (n_bins,) radial bin centres in pixels
        'f_rad':   (n_bins,) radial average
    }
    """
    out: dict[tuple[str, str], dict] = {}
    names = list(masks.keys())
    for i, n in enumerate(names):
        for j in range(i, len(names)):
            p = names[j]
            f_r = two_point(masks[n], masks[p], periodic=periodic)
            r_px, f_rad = two_point_radial(f_r, n_bins=radial_bins)
            out[(n, p)] = {'f_r': f_r, 'r_px': r_px, 'f_rad': f_rad}
            if i != j:
                out[(p, n)] = out[(n, p)]
    return out


# ============================================================
# RVE / coherence length
# ============================================================

def coherence_length(
    f_r: np.ndarray,
    phi: float,
    *,
    threshold_frac: float = 0.05,
    n_bins: int = 60,
    px_um: float = 1.0,
) -> tuple[float, float]:
    """Estimate the coherence length of a 2-point correlation.

    Per Adams-Kalidindi-Fullwood (2012), the *coherence length* C is the
    smallest r at which f_r(r) → φ² (i.e., the autocorrelation has decayed
    to the 1-point statistic squared).  For r ≥ C the field looks
    statistically uncorrelated; if your scan is smaller than 2C, the
    2-point statistics are sample-size-biased.

    Parameters
    ----------
    f_r : (ny, nx) — autocorrelation map (centred)
    phi : float    — volume fraction (1-point statistic) of the same field
    threshold_frac : float
        Relative tolerance |f_r - φ²| / φ² below which we say "decayed".
    n_bins : int
    px_um  : float
        Pixel size in µm.

    Returns
    -------
    C_um : float     — coherence length in µm; +inf if not reached on this scan
    fraction_reached : float
        f_r at the largest sampled r, divided by φ² — diagnostic for whether
        the scan is anywhere near the coherence length.
    """
    r_px, f_rad = two_point_radial(f_r, n_bins=n_bins)
    target = phi * phi
    if target <= 0:
        return float('inf'), 0.0
    rel_err = np.abs(f_rad - target) / target
    below = np.where(rel_err < threshold_frac)[0]
    fraction_reached = float(f_rad[-1] / target) if target > 0 else 0.0
    if not below.size:
        return float('inf'), fraction_reached
    return float(r_px[below[0]] * px_um), fraction_reached


# ============================================================
# PCA across an ensemble of microstructures
# ============================================================

def pca_on_2pt(
    f_r_stack: np.ndarray,
    n_components: int = 2,
    *,
    centre: bool = True,
) -> dict:
    """PCA on a stack of flattened 2-point correlations.

    Useful when you have multiple scans (or sub-regions of one scan) and
    want a low-dimensional similarity space — sub-regions / scans that
    look similar end up close in PC space (Adams-Kalidindi-Fullwood 2012).

    Parameters
    ----------
    f_r_stack : (J, ny, nx) — J samples of the same observable
    n_components : int      — number of principal components to return
    centre : bool           — subtract the mean f_r before SVD (recommended)

    Returns
    -------
    dict with keys:
        scores       — (J, n_components)
        components   — (n_components, ny, nx) eigenmaps
        explained    — (n_components,) explained-variance ratio
        mean_f_r     — (ny, nx) the mean (only if centre=True; else zeros)
    """
    X = np.asarray(f_r_stack, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f'expected (J, ny, nx); got shape {X.shape}')
    J, ny, nx = X.shape
    flat = X.reshape(J, -1)
    if centre:
        mean_flat = flat.mean(axis=0)
        flat_c = flat - mean_flat[None, :]
    else:
        mean_flat = np.zeros(flat.shape[1])
        flat_c = flat
    # Thin SVD: U S Vt where U is (J, J), S is (J,), Vt is (J, ny*nx)
    U, S, Vt = np.linalg.svd(flat_c, full_matrices=False)
    var = (S ** 2) / max(J - 1, 1)
    var_ratio = var / max(var.sum(), 1e-30)
    k = min(n_components, len(S))
    scores     = U[:, :k] * S[:k]
    components = Vt[:k].reshape(k, ny, nx)
    return {
        'scores':       scores,
        'components':   components,
        'explained':    var_ratio[:k],
        'mean_f_r':     mean_flat.reshape(ny, nx),
        'singular':     S[:k],
    }


# ============================================================
# Tile a 2D field into sub-regions (Latypov-Kalidindi 2017 §4.5 SVE sets)
# ============================================================

def tile_field(
    field: np.ndarray,
    *,
    tile_px: int = 128,
    stride: Optional[int] = None,
    drop_partial: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Divide a 2D scalar/boolean field into uniform tiles.

    Used to generate a sub-region SVE ensemble from one large EBSD scan,
    so PCA and other ensemble-based methods (Latypov-Kalidindi 2017) can
    be applied without needing many physically-distinct scans.

    Parameters
    ----------
    field : (ny, nx) or (ny, nx, ...) — anything with leading 2 spatial axes
    tile_px : int       — size of each tile in pixels (square)
    stride : int, optional — step between tiles; default = tile_px // 2
                              (50% overlap, doubling the effective ensemble)
    drop_partial : bool  — if True, drop tiles that don't fully fit

    Returns
    -------
    tiles : (n_tiles, tile_px, tile_px[, ...]) — stacked tiles
    coords : list of (y0, x0) — origin of each tile in pixel coordinates
    """
    if stride is None:
        stride = max(1, tile_px // 2)
    ny, nx = field.shape[:2]
    tiles = []
    coords = []
    for y0 in range(0, ny - (tile_px - 1 if drop_partial else 0), stride):
        if drop_partial and y0 + tile_px > ny:
            break
        for x0 in range(0, nx - (tile_px - 1 if drop_partial else 0), stride):
            if drop_partial and x0 + tile_px > nx:
                break
            tile = field[y0:y0 + tile_px, x0:x0 + tile_px]
            if tile.shape[:2] == (tile_px, tile_px):
                tiles.append(tile)
                coords.append((y0, x0))
    return (np.stack(tiles, axis=0) if tiles
            else np.zeros((0, tile_px, tile_px) + field.shape[2:],
                           dtype=field.dtype)), coords


# ============================================================
# Convenience: phase × scalar-class
# ============================================================

def phase_specific_correlation(
    phase_mask: np.ndarray,
    scalar_class_mask: np.ndarray,
    *,
    periodic: bool = False,
    radial_bins: int = 60,
) -> dict:
    """2-pt cross-correlation between a phase mask and a scalar-class mask.

    Useful for asking "does scalar class C of feature X cluster within
    phase P?" — e.g. phase=Calcite × KAM_high, phase=Quartz × dauphine, etc.

    Returns
    -------
    dict with f_r, r_px, f_rad, plus phi_phase, phi_class, phi_phi (= product
    of one-point stats, the asymptotic value of f_r at the coherence length).
    """
    a = np.asarray(phase_mask).astype(np.float64)
    b = np.asarray(scalar_class_mask).astype(np.float64)
    f_r = two_point(a, b, periodic=periodic)
    r_px, f_rad = two_point_radial(f_r, n_bins=radial_bins)
    phi_a = float(a.mean()); phi_b = float(b.mean())
    return {
        'f_r':       f_r,
        'r_px':      r_px,
        'f_rad':     f_rad,
        'phi_phase': phi_a,
        'phi_class': phi_b,
        'phi_phi':   phi_a * phi_b,
    }
