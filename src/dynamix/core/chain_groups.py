# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Chain grouping over the partition function's tilted measure -- the EBSD-workbook port.

The workbook's cell-52 controls
delineate GROUPS of cross-scale maxima chains by Boltzmann weight: at a chosen scale index and
moment order q, each surviving chain gets ``w = |T|^q / Z(q, a)`` and a membership rule splits
the population into dominant / non-dominant. "Topology from the partition function" is exactly
this per-line tilted measure (plus the tau(q) phase-transition breakpoint, which lives in
:mod:`dynamix.core.spectra`); it is never a spatial lasso and never an h-band.

Pure numpy over the ``chains`` list every full WTMM result carries (``{"x","y","mod",...}``
per chain, finest-anchored -- the wtmm_ebsd schema); the per-(chain, scale) modulus matrix is
built once per chains list and memoized by identity (the ``chain_stats._METRIC_CACHE`` idiom).
Subset partition tables delegate to ``wtmm_ebsd.partition.build_hd_from_chains`` UNCHANGED --
the Boltzmann-average / Chhabra-Jensen accumulation math lives there and is never transcribed
here (the ``wtmm_backend.partition2d`` law); ``wtmm_ebsd`` stays lazily imported.

One recorded divergence from the notebook: the absolute-threshold modes gate on
``q * ln|T| >= theta`` (a LOG-space threshold) rather than the notebook's raw ``|T|^q >= thresh``
-- ``|T|^q`` spans hundreds of decades over the workbook's own q range, so the raw comparison is
numerically meaningless for most of the slider's travel; log space is the monotone-equivalent
knob that stays finite. The percentile mode is the notebook's, unchanged.
"""
from __future__ import annotations

import numpy as np

__all__ = ["grouping_payload", "boltzmann_weights", "classify_chains", "subset_hd",
           "MODES"]

#: Membership rules, in the workbook's own order (cell 52 ``thresh_mode`` dropdown).
MODES = ("percentile", "mq_at_scale", "mq_sup")

_PAYLOAD_CACHE: "dict[int, tuple]" = {}
_PAYLOAD_CACHE_MAX = 4

_SUBSET_CACHE: "dict[tuple, tuple]" = {}
_SUBSET_CACHE_MAX = 16


def grouping_payload(chains: list, n_scales: int) -> dict:
    """The per-(chain, scale) modulus matrices, built once per ``chains`` list.

    Returns ``{"chain_len" (n_ch,), "mod_matrix", "cmax_matrix" (n_ch, n_sc)}``; NaN where a
    chain does not reach a scale. ``cmax_matrix`` is the running supremum from the finest scale
    up -- the chainmax convention (``maximum.accumulate``), NaN-masked back to the chain's own
    extent. Memoized on the list's identity with a strong reference (chains come out of the
    immutable engine-cached result, so identity implies identical content)."""
    hit = _PAYLOAD_CACHE.get(id(chains))
    if hit is not None and hit[0] is chains:
        return hit[1]
    n_ch = len(chains)
    chain_len = np.asarray([len(c["mod"]) for c in chains], dtype=np.int64)
    mod = np.full((n_ch, int(n_scales)), np.nan)
    for i, c in enumerate(chains):
        k = min(len(c["mod"]), mod.shape[1])
        mod[i, :k] = np.asarray(c["mod"], dtype=np.float64)[:k]
    with np.errstate(invalid="ignore"):
        cmax = np.fmax.accumulate(np.where(np.isnan(mod), -np.inf, mod), axis=1)
    cmax = np.where(np.isnan(mod), np.nan, cmax)
    payload = {"chain_len": chain_len, "mod_matrix": mod, "cmax_matrix": cmax}
    _PAYLOAD_CACHE[id(chains)] = (chains, payload)
    while len(_PAYLOAD_CACHE) > _PAYLOAD_CACHE_MAX:
        del _PAYLOAD_CACHE[next(iter(_PAYLOAD_CACHE))]
    return payload


def boltzmann_weights(payload: dict, scale_idx: int, q: float, *,
                      chainmax: bool = False):
    """``w_i = |T_i|^q / Z(q, a)`` over the chains alive at ``scale_idx`` (log-sum-exp safe:
    ``w = exp(q ln|T| - max) / sum``). Returns ``(weights, alive_idx)`` -- weights align with
    ``alive_idx`` into the chain population."""
    col = payload["cmax_matrix" if chainmax else "mod_matrix"][:, int(scale_idx)]
    alive = np.flatnonzero(np.isfinite(col) & (col > 0))
    if alive.size == 0:
        return np.zeros(0), alive
    logw = float(q) * np.log(col[alive])
    logw -= logw.max()
    w = np.exp(logw)
    return w / w.sum(), alive


def classify_chains(payload: dict, *, scale_idx: int, q: float, mode: str = "percentile",
                    dom_percentile: float = 20.0, log_thresh: float = 0.0,
                    chainmax: bool = False, min_len: int = 1) -> np.ndarray:
    """The cell-52 membership rules -> one boolean per chain (True = dominant).

    - ``"percentile"``: Boltzmann weight at ``scale_idx``; cutoff at the ``100 - p`` percentile
      of the nonzero weights; a chain is dominant when its weight reaches the cutoff.
    - ``"mq_at_scale"``: ``q * ln|T(a)| >= log_thresh`` at ``scale_idx`` (log-space; see the
      module docstring's recorded divergence).
    - ``"mq_sup"``: same threshold on the running-sup (chainmax) modulus.

    ``chainmax`` switches the CLASSIFICATION modulus for the first two modes (the workbook's
    ``classify_chainmax``, independent of any display choice); ``min_len`` excludes chains
    shorter than it from dominance outright (cell 52's ``min_vc`` gate)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    use_sup = chainmax or mode == "mq_sup"
    col = payload["cmax_matrix" if use_sup else "mod_matrix"][:, int(scale_idx)]
    n_ch = col.size
    out = np.zeros(n_ch, dtype=bool)
    alive = np.isfinite(col) & (col > 0) & (payload["chain_len"] >= int(min_len))
    if not alive.any():
        return out
    if mode == "percentile":
        logw = float(q) * np.log(col[alive])
        logw -= logw.max()
        w = np.exp(logw)
        w = w / w.sum()
        nz = w[w > 0]
        if nz.size == 0:
            return out
        cutoff = np.percentile(nz, 100.0 - float(dom_percentile))
        out[np.flatnonzero(alive)[w >= cutoff]] = True
    else:
        out[np.flatnonzero(alive)[float(q) * np.log(col[alive]) >= float(log_thresh)]] = True
    return out


def subset_hd(chains: list, scales, q_list, mask: np.ndarray, *,
              min_chain_len: int = 2):
    """Partition tables over a chain SUBSET -- ``(hd_std, hd_cmax)`` for the chains where
    ``mask`` is True, by delegation to ``wtmm_ebsd.partition.build_hd_from_chains`` (lazy
    import; the math is never transcribed here). Memoized on (chains identity, mask bytes,
    q-grid bytes, min_chain_len) -- a repeated gesture with the same membership is a dict hit,
    the workbook's own precompute posture."""
    mask = np.asarray(mask, dtype=bool)
    q_arr = np.asarray(q_list, dtype=np.float64)
    key = (id(chains), mask.tobytes(), q_arr.tobytes(), int(min_chain_len))
    hit = _SUBSET_CACHE.get(key)
    if hit is not None and hit[0] is chains:
        return hit[1]
    from dynamix._vendor.wtmm_ebsd.partition import build_hd_from_chains

    subset = [c for c, m in zip(chains, mask) if m]
    tables = build_hd_from_chains(subset, scales, q_arr, min_chain_len=int(min_chain_len))
    _SUBSET_CACHE[key] = (chains, tables)
    while len(_SUBSET_CACHE) > _SUBSET_CACHE_MAX:
        del _SUBSET_CACHE[next(iter(_SUBSET_CACHE))]
    return tables
