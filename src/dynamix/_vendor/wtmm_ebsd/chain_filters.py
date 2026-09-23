"""WTMM chain filters: modulus, Holder-slope, and vertical-length thresholds.

Replaces the inline filter logic in the per-grain notebook's cells 57 and
59 (the click-driven threshold widgets).  These are pure data operations;
the click handlers stay in the notebook.

Public API
----------
chain_max_log2_modulus(chain) -> float
chain_ols_holder(chain) -> float
chain_max_slope_holder(chain) -> float
chain_length(chain) -> int
filter_by_modulus(chains, threshold) -> (kept, dropped)
filter_by_holder(chains, slope_cutoff, *, holder='ols') -> (kept, dropped)
filter_by_length(chains, min_len) -> (kept, dropped)
register_pruned_mode(svd_wtmm, source_mode, kept_chains, *, suffix='_pruned',
                      grain_id=None, ny_nx=None, threshold=None)
classify_chains_by_grain_position(chains, grain_id_initial, dauphine_mask, *,
                      min_grain_size=9) -> dict[str, list[int]]
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def chain_max_log2_modulus(chain: dict) -> float:
    """Maximum log2|W(a)| of a chain across its scales.  NaN if empty."""
    log2m = chain.get('log2_mod', None)
    if log2m is None or len(log2m) == 0:
        return float('nan')
    arr = np.asarray(log2m, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float('nan')
    return float(finite.max())


def chain_ols_holder(chain: dict) -> float:
    """OLS slope of log2|W| vs log2(scale) for a chain.  NaN if < 2 points."""
    log2s = np.asarray(chain.get('log2_scales', []), dtype=np.float64)
    log2m = np.asarray(chain.get('log2_mod',    []), dtype=np.float64)
    if len(log2s) < 2:
        return float('nan')
    finite = np.isfinite(log2s) & np.isfinite(log2m)
    if finite.sum() < 2:
        return float('nan')
    slope, _ = np.polyfit(log2s[finite], log2m[finite], 1)
    return float(slope)


def chain_max_slope_holder(chain: dict) -> float:
    """Maximum local slope of log2|W| vs log2(scale).  NaN if < 2 points."""
    log2s = np.asarray(chain.get('log2_scales', []), dtype=np.float64)
    log2m = np.asarray(chain.get('log2_mod',    []), dtype=np.float64)
    if len(log2s) < 2:
        return float('nan')
    ds = np.diff(log2s)
    dm = np.diff(log2m)
    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = dm / ds
    finite = slopes[np.isfinite(slopes)]
    if finite.size == 0:
        return float('nan')
    return float(finite.max())


def chain_length(chain: dict) -> int:
    """Vertical (across-scale) length of a chain.

    Returns the number of (a, |W|) samples the chain spans.  Used as the
    primary filter for partition-function statistics: a chain of length ≤ 1
    has zero inter-scale information and should never contribute to a
    log-log slope fit.

    Looks at ``'mod'`` first, falls back to ``'log2_mod'``, then
    ``'log2_scales'``.  Returns 0 if no length-bearing field is present.
    """
    for key in ('mod', 'log2_mod', 'log2_scales'):
        v = chain.get(key)
        if v is not None:
            return int(len(v))
    return 0


def filter_by_modulus(
    chains: list,
    threshold: float,
) -> tuple[list, list]:
    """Partition chains into (kept, dropped) by max-log2-modulus threshold.

    Kept = chains whose `chain_max_log2_modulus` >= threshold.
    """
    kept, dropped = [], []
    for ch in chains:
        m = chain_max_log2_modulus(ch)
        if np.isfinite(m) and m >= threshold:
            kept.append(ch)
        else:
            dropped.append(ch)
    return kept, dropped


def filter_by_holder(
    chains: list,
    slope_cutoff: float,
    *,
    holder: str = 'ols',
) -> tuple[list, list]:
    """Partition chains by Holder slope threshold.

    Kept = chains whose Holder >= slope_cutoff.

    Parameters
    ----------
    holder : str   'ols' (least-squares slope) or 'max' (max local slope)
    """
    if holder == 'ols':
        getter = chain_ols_holder
    elif holder == 'max':
        getter = chain_max_slope_holder
    else:
        raise ValueError(f"holder must be 'ols' or 'max'; got {holder!r}")
    kept, dropped = [], []
    for ch in chains:
        h = getter(ch)
        if np.isfinite(h) and h >= slope_cutoff:
            kept.append(ch)
        else:
            dropped.append(ch)
    return kept, dropped


def filter_by_length(
    chains: list,
    min_len: int,
) -> tuple[list, list]:
    """Partition chains by vertical (across-scale) length.

    Kept = chains whose vertical length is >= min_len.

    A chain of length 0 contributes nothing to the partition function;
    a chain of length 1 contributes a single (a, |W|) sample at one
    scale only and CANNOT contribute to any log-log slope.  Most callers
    want at least ``min_len=2`` (the partition-function default), and
    often higher (5–10) to require chains that span enough scales for
    a stable Holder fit.

    Parameters
    ----------
    chains : list of chain dicts
    min_len : int    minimum required vertical length (inclusive)

    Returns
    -------
    kept, dropped : tuple of lists
    """
    if min_len <= 0:
        return list(chains), []
    kept, dropped = [], []
    for ch in chains:
        if chain_length(ch) >= min_len:
            kept.append(ch)
        else:
            dropped.append(ch)
    return kept, dropped


def register_pruned_mode(
    svd_wtmm: dict,
    source_mode: str,
    kept_chains: list,
    *,
    suffix: str = '_pruned',
    grain_id: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    extra_meta: Optional[dict] = None,
) -> str:
    """Register a `<source_mode><suffix>` entry in svd_wtmm with the kept chains.

    Re-derives `chains_by_grain` from the kept chains, and copies `mod` and
    `ext_images` references from the source.  Holders are re-computed via
    polyfit since the kept chain count may differ.

    Returns the new mode name.
    """
    if source_mode not in svd_wtmm:
        raise KeyError(f"source mode {source_mode!r} not in svd_wtmm")
    src = svd_wtmm[source_mode]
    new_mode = f"{source_mode}{suffix}"
    holders = [{'h': chain_ols_holder(c)} for c in kept_chains]
    chains_by_grain: dict[int, list] = {}
    if grain_id is not None:
        ny, nx = grain_id.shape
        for ch in kept_chains:
            x0, y0 = int(ch['x'][0]), int(ch['y'][0])
            if 0 <= y0 < ny and 0 <= x0 < nx:
                gid0 = int(grain_id[y0, x0])
            else:
                gid0 = 0
            chains_by_grain.setdefault(gid0, []).append(ch)
    entry = {
        'chains': kept_chains,
        'holders': holders,
        'chains_by_grain': chains_by_grain,
        'mod': src.get('mod'),
        'ext_images': src.get('ext_images'),
        'pruned_from': source_mode,
        'threshold': threshold,
    }
    if extra_meta:
        entry.update(extra_meta)
    svd_wtmm[new_mode] = entry
    return new_mode


# =====================================================================
# Vectorised chain bucketing — by finest-scale anchor pixel position
# =====================================================================

def classify_chains_by_grain_position(
    chains: list[dict],
    grain_id_initial: np.ndarray,
    dauphine_mask: np.ndarray,
    *,
    min_grain_size: int = 9,
) -> dict[str, list[int]]:
    """Vectorised bucket assignment for WTMM chains by finest-scale anchor.

    Buckets:
      'intragrain'           — anchor pixel inside an INITIAL grain whose
                                size >= min_grain_size, AND not on a GB
                                between two such grains.
      'intergrain'           — anchor on a GB between two INITIAL grains
                                (each >= min_grain_size). Includes Dauphine
                                twin GBs.
      'small'                — anchor in an initial grain < min_grain_size,
                                or in an unindexed pixel.
      'dauphine_intergrain'  — subset of 'intergrain' where the anchor sits
                                on a Dauphine twin GB specifically (per
                                dauphine_mask). Not exclusive with
                                'intergrain' — chains here also appear there.

    Each value is a list of chain indices into the input `chains` list.

    All checks are vectorised: O(N_chains) lookups via fancy-indexing into
    the (ny, nx) grain map, no per-chain Python loop.

    Parameters
    ----------
    chains : list[dict]
        Each chain has at least 'x' and 'y' arrays (anchor pixel coords at
        each scale; index 0 = finest scale).
    grain_id_initial : (ny, nx) int
        Pre-Dauphine-merge grain ID map. Use INITIAL grains so each twin
        variant is its own grain — required to make Dauphine GBs visible
        as boundary pixels.
    dauphine_mask : (ny, nx) bool
        True at pixels on a Dauphine twin GB.
    min_grain_size : int
        Minimum INITIAL grain size in pixels for "intragrain" / "intergrain"
        eligibility. Pixels in smaller grains land in 'small'.

    Returns
    -------
    buckets : dict[str, list[int]]
    """
    bucket_keys = ('intragrain', 'intergrain', 'small', 'dauphine_intergrain')
    if not chains:
        return {k: [] for k in bucket_keys}

    ny, nx = grain_id_initial.shape
    n = len(chains)

    # Extract finest-scale anchor coords as (n,) arrays (vectorised).
    xs = np.empty(n, dtype=np.int64)
    ys = np.empty(n, dtype=np.int64)
    for i, ch in enumerate(chains):
        x_arr = ch.get('x', None)
        y_arr = ch.get('y', None)
        if x_arr is None or y_arr is None or len(x_arr) == 0:
            xs[i] = -1; ys[i] = -1
            continue
        xs[i] = int(x_arr[0]); ys[i] = int(y_arr[0])
    valid_anchor = (xs >= 0) & (ys >= 0)
    xs_c = np.clip(xs, 0, nx - 1)
    ys_c = np.clip(ys, 0, ny - 1)

    # Bulk grain-ID lookup at every anchor pixel.
    gid_init_anchor = grain_id_initial[ys_c, xs_c]

    # Per-grain size from the INITIAL map; gid_init_anchor==0 (unindexed)
    # always lands in 'small' regardless of size table.
    init_sizes = np.bincount(grain_id_initial.ravel())
    big_mask_for_anchor = (gid_init_anchor > 0) & \
                          (gid_init_anchor < len(init_sizes))
    is_big_init = np.zeros(n, dtype=bool)
    if big_mask_for_anchor.any():
        is_big_init[big_mask_for_anchor] = (
            init_sizes[gid_init_anchor[big_mask_for_anchor]] >= min_grain_size
        )

    # GB pixel mask: any 4-neighbour has a different INITIAL grain ID.
    gb_mask_init = np.zeros((ny, nx), dtype=bool)
    diff_r = grain_id_initial[:, :-1] != grain_id_initial[:, 1:]
    gb_mask_init[:, :-1] |= diff_r
    gb_mask_init[:,  1:] |= diff_r
    diff_d = grain_id_initial[:-1, :] != grain_id_initial[1:, :]
    gb_mask_init[:-1, :] |= diff_d
    gb_mask_init[ 1:, :] |= diff_d
    is_gb_anchor    = gb_mask_init[ys_c, xs_c] & valid_anchor

    is_dauph_anchor = dauphine_mask[ys_c, xs_c] & valid_anchor

    # Bucket assignment (mutually exclusive for intra/inter/small).
    is_small = ~is_big_init | ~valid_anchor
    is_intra = is_big_init & ~is_gb_anchor & valid_anchor
    is_inter = is_big_init &  is_gb_anchor & valid_anchor
    is_dauph_inter = is_inter & is_dauph_anchor

    return {
        'intragrain':          np.where(is_intra)[0].tolist(),
        'intergrain':          np.where(is_inter)[0].tolist(),
        'small':               np.where(is_small)[0].tolist(),
        'dauphine_intergrain': np.where(is_dauph_inter)[0].tolist(),
    }
