"""Kantelhardt-style spatial shuffles for null-hypothesis testing.

Three flavors:
  within_grain_shuffle    -- per-grain pixel permutation (UPSTREAM, on omega)
  global_shuffle          -- whole-image pixel permutation (UPSTREAM, on omega)
  bound_tuple_shuffle     -- per-grain (rho_1, ..., rho_N) tuple permutation
                              (DOWNSTREAM, on slip-system density vectors)

Within-grain shuffle preserves the per-grain marginal distribution; global
shuffle destroys grain structure entirely (extreme null).  Bound-tuple
preserves cross-system co-activation patterns at each pixel while destroying
spatial correlations.

The two upstream shuffles test "is multifractal scaling driven by spatial
correlations or by the marginal distribution?"; the downstream shuffle
tests the same question AFTER the non-linear NNLS projection -- a different
null hypothesis (cell 13 vs cell 45 of v1).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def within_grain_shuffle(
    field: np.ndarray,
    grain_id: np.ndarray,
    *,
    rng: Optional[np.random.Generator] = None,
    min_pixels: int = 2,
) -> np.ndarray:
    """Shuffle pixel values WITHIN each grain.

    Pixels with grain_id == 0 are left unchanged.  Field can be (ny, nx) or
    (ny, nx, c); for vector fields the ENTIRE c-vector at each pixel is
    permuted as a unit (not coordinate-by-coordinate).

    Parameters
    ----------
    field : (ny, nx) or (ny, nx, c)
    grain_id : (ny, nx) int
    rng : np.random.Generator, optional
    min_pixels : int       skip grains smaller than this

    Returns
    -------
    shuffled : same shape as field
    """
    rng = np.random.default_rng() if rng is None else rng
    out = field.copy()
    if field.ndim not in (2, 3):
        raise ValueError(f"field must be 2D or 3D, got {field.ndim}D")

    gids = [int(g) for g in np.unique(grain_id) if int(g) > 0]
    for gid in gids:
        ys, xs = np.where(grain_id == gid)
        if len(ys) < min_pixels:
            continue
        perm = rng.permutation(len(ys))
        if field.ndim == 2:
            out[ys, xs] = field[ys[perm], xs[perm]]
        else:
            out[ys, xs, :] = field[ys[perm], xs[perm], :]
    return out


def global_shuffle(
    field: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Shuffle ALL pixels (within valid_mask) without regard to grains.

    Restricts the shuffle to valid pixels; outside-mask pixels are
    unchanged.  Vector pixels are permuted as units (full c-vector).

    Parameters
    ----------
    field : (ny, nx) or (ny, nx, c)
    valid_mask : (ny, nx) bool, optional
    rng : np.random.Generator, optional
    """
    rng = np.random.default_rng() if rng is None else rng
    out = field.copy()
    if valid_mask is None:
        valid_mask = np.ones(field.shape[:2], dtype=bool)
    ys, xs = np.where(valid_mask)
    if len(ys) == 0:
        return out
    perm = rng.permutation(len(ys))
    if field.ndim == 2:
        out[ys, xs] = field[ys[perm], xs[perm]]
    elif field.ndim == 3:
        out[ys, xs, :] = field[ys[perm], xs[perm], :]
    else:
        raise ValueError(f"field must be 2D or 3D, got {field.ndim}D")
    return out


def bound_tuple_shuffle(
    rho_per_system: dict,
    grain_id: np.ndarray,
    *,
    rng: Optional[np.random.Generator] = None,
    min_pixels: int = 2,
) -> dict:
    """Per-grain shuffle of the (rho_1, rho_2, ..., rho_N) tuple at each pixel.

    Preserves: each system's marginal distribution exactly, AND the cross-
    system co-activation pattern at each pixel (the WHOLE tuple moves
    together).
    Destroys: only the spatial correlations within each grain.

    Canonical Kantelhardt analog for the scalar-WTMM spectra of GND density
    maps.

    Parameters
    ----------
    rho_per_system : dict
        keyed by some hashable (e.g. system name); values are (ny, nx) float
        arrays.  All values must share the same (ny, nx) shape.
    grain_id : (ny, nx) int
    rng : np.random.Generator, optional
    min_pixels : int

    Returns
    -------
    shuffled : dict with the same keys
    """
    rng = np.random.default_rng() if rng is None else rng
    if not rho_per_system:
        return {}
    keys = list(rho_per_system.keys())
    shape = rho_per_system[keys[0]].shape
    for k in keys:
        if rho_per_system[k].shape != shape:
            raise ValueError(f"rho field {k!r} has shape {rho_per_system[k].shape}; "
                             f"expected {shape}")
    out = {k: np.asarray(rho_per_system[k]).copy() for k in keys}

    gids = [int(g) for g in np.unique(grain_id) if int(g) > 0]
    for gid in gids:
        ys, xs = np.where(grain_id == gid)
        if len(ys) < min_pixels:
            continue
        perm = rng.permutation(len(ys))
        # Apply the SAME perm to every system -- preserves co-activation
        for k in keys:
            out[k][ys, xs] = rho_per_system[k][ys[perm], xs[perm]]
    return out
