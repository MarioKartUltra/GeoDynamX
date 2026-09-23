"""Pattern-quality damage proxies derived from EBSD raw fields.

These fields are CHEAP scalar diagnostics that complement the orientation-
based damage measures (KAM, GND density). They use only the raw EBSD
quality metrics (band contrast, mean angular deviation) and require no
symmetry handling, no Karcher mean, no NNLS.

Provenance: lifted from `ebsd_wtmm.ipynb` cell 24 (BC gradient) and
generalised. Used in v1 only as visualisation; promoted here to a
first-class auxiliary observable that can be fed to scalar WTMM, used as
a chain-pruning mask, or correlated with KAM (cell 105 of the per-grain
notebook).

Public API
----------
bc_gradient_magnitude(bc) -> ndarray
bc_damage_mask(bc, *, percentile=95) -> ndarray (bool)
confidence_weighted_damage(bc, mad, *, bc_low_pct=10, mad_high_deg=1.5) -> ndarray
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def bc_gradient_magnitude(bc: np.ndarray) -> np.ndarray:
    """|nabla BC| via central differences -- damage / GB proxy.

    Low band contrast usually means a poorly-resolved Kikuchi pattern,
    which correlates with high lattice distortion (deformation, damage,
    or GB proximity).  The spatial gradient of BC highlights TRANSITIONS:
    steep change in BC = grain boundary or step in damage state.

    Parameters
    ----------
    bc : (ny, nx) float | int
        Raw band contrast field from the EBSD scanner.  Range is
        typically 0-255 (int) or 0-100 (normalised), but no specific
        normalisation is assumed.

    Returns
    -------
    grad_mag : (ny, nx) float64
        Gradient magnitude in BC-units / pixel.
    """
    bc_f = np.asarray(bc, dtype=np.float64)
    if bc_f.ndim != 2:
        raise ValueError(f"bc must be 2D, got shape {bc_f.shape}")
    dx = np.gradient(bc_f, axis=1)
    dy = np.gradient(bc_f, axis=0)
    return np.sqrt(dx * dx + dy * dy)


def bc_damage_mask(
    bc: np.ndarray,
    *,
    percentile: float = 95.0,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean mask where |nabla BC| exceeds the given percentile.

    The default 95th percentile labels approximately the top 5% of pixels
    (those at GBs or sharp damage transitions) as "damage."  Tune
    `percentile` for sparser (higher) or denser (lower) coverage.

    Parameters
    ----------
    bc : (ny, nx)
    percentile : float                threshold in [0, 100]
    valid_mask : (ny, nx) bool        restrict the percentile estimate
                                       (e.g. exclude unindexed pixels).
                                       Defaults to using all pixels.

    Returns
    -------
    mask : (ny, nx) bool
    """
    if not 0 < percentile < 100:
        raise ValueError(f"percentile must be in (0, 100), got {percentile}")
    grad = bc_gradient_magnitude(bc)
    if valid_mask is not None:
        thr = float(np.percentile(grad[valid_mask], percentile))
    else:
        thr = float(np.percentile(grad, percentile))
    return grad >= thr


def confidence_weighted_damage(
    bc: np.ndarray,
    mad: np.ndarray,
    *,
    bc_low_pct: float = 10.0,
    mad_high_deg: float = 1.5,
) -> np.ndarray:
    """Soft damage score combining low BC + high MAD.

    A pixel scores high when EITHER:
      - its BC is below the `bc_low_pct`-th percentile of BC values, OR
      - its MAD exceeds `mad_high_deg`.
    Score is the maximum of the two normalized signals.

    Returns a continuous score in [0, 1] per pixel.  Useful as a soft
    confidence weight for downstream analyses (e.g. weighted KAM, masked
    WTMM input).

    Parameters
    ----------
    bc : (ny, nx)
    mad : (ny, nx) float, in degrees
    bc_low_pct : float
        Percentile threshold below which BC is considered "low."
    mad_high_deg : float
        MAD threshold above which the indexing is considered unreliable.

    Returns
    -------
    score : (ny, nx) float64 in [0, 1]
    """
    bc_f  = np.asarray(bc,  dtype=np.float64)
    mad_f = np.asarray(mad, dtype=np.float64)
    if bc_f.shape != mad_f.shape:
        raise ValueError(f"bc {bc_f.shape} and mad {mad_f.shape} must match")

    # Low-BC component: 1 at the percentile floor, fades to 0 at the median
    bc_floor = float(np.percentile(bc_f, bc_low_pct))
    bc_med   = float(np.median(bc_f))
    if bc_med <= bc_floor:
        bc_score = np.zeros_like(bc_f)
    else:
        bc_score = np.clip((bc_med - bc_f) / (bc_med - bc_floor), 0.0, 1.0)

    # High-MAD component: 0 below threshold, fades to 1 over a fixed band
    # above the threshold (band = mad_high_deg, so MAD = 2*mad_high_deg => 1).
    mad_score = np.clip((mad_f - mad_high_deg) / max(mad_high_deg, 1e-6),
                         0.0, 1.0)

    return np.maximum(bc_score, mad_score)
