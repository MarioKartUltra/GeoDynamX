# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Per-chain Hölder / modulus / length statistics over the WTMM chain-dict schema.

This is a **reimplementation**, not a port. It exists so the ``chain_holder``/``chain_modulus``/
``chain_length`` filter devices (``dynamix.devices.chain_filters``) have a real, numpy-only
computation available when the optional ``wtmm_ebsd`` package is absent, and so those devices'
``reading``/``data_hints`` protocol methods have
one estimator to share with the future skeleton dialog. Per project rule, ``wtmm_ebsd`` is never
imported here, and none of its code is copied -- every function below is written fresh against the
documented reference semantics:

* :func:`per_chain_stats` mirrors the "best-principled", R²-carrying ``_per_chain_stats`` found in
  ``ebsd_gnd_filter_comparison.ipynb`` cell 32 (Creep/wavelet, read-only reference) -- with ONE
  deliberate change from that notebook: ``n`` (and the ``n < 3`` all-NaN gate) counts JOINTLY
  finite ``(log2_scales, log2_mod)`` pairs, not the raw sample count. The notebook version trusted
  its input to be clean and gated on ``len(log2_scales)`` alone; a device consuming arbitrary
  chains cannot make that assumption.
* :func:`stats_for` mirrors ``wtmm_ebsd.chain_filters.chain_ols_holder``/``chain_max_slope_holder``
  EXACTLY (including their looser ``n < 2`` gate, no R²) -- this is deliberately the SAME estimator
  ``filter_by_holder`` uses via the real backend, so the degraded-mode fallback in
  ``ChainHolderFilter.apply`` (when ``wtmm_ebsd`` is not installed) and the live reading/hints (which
  must describe whatever estimator produced the CURRENT ``chains``, real backend or not) agree with
  it to the last chain. ``per_chain_stats``'s stricter ``n < 3`` R²-carrying numbers are a DIFFERENT,
  more conservative statistic, on purpose -- see each docstring.

Scale axis: DynamiX chains already carry a real ``log2_scales`` array (``dynamix.core.wtmm_backend``
builds it as ``np.log2(scales)[:k]`` -- actual log2 of the physical/pixel scale values the wavelet
transform used, not an implicit index range), so every function here reads ``chain["log2_scales"]``
directly. There is no dyadic-voice approximation to make: the real octave spacing is already baked
into the array by the transform that produced it.
"""
from __future__ import annotations

import numpy as np


def per_chain_stats(chain: dict) -> tuple[float, float, float, float, int]:
    """``(h_ols, r2, max_slope, min_slope, n)`` for one chain.

    ``n`` is the count of INDICES where both ``log2_scales`` and ``log2_mod`` are finite. All of
    ``h_ols``/``r2``/``max_slope``/``min_slope`` are NaN when ``n < 3`` (``n`` itself is still the
    real count, never NaN -- a diagnostic of how close the chain came to being usable). This is the
    R²-carrying "chain hygiene" statistic (see module docstring); :func:`stats_for` is what the
    filter devices use for their own OLS/max-slope estimate.

    ``h_ols`` is the OLS slope of ``log2_mod`` against ``log2_scales`` (``np.polyfit(..., 1)[0]``);
    ``r2`` is the fit's coefficient of determination (``1 - ss_res / max(ss_tot, 1e-30)`` -- the
    ``max`` guards a chain whose finite ``log2_mod`` values are literally all equal, where
    ``ss_tot`` is exactly 0); ``max_slope``/``min_slope`` are the largest/smallest LOCAL slope
    (``diff(log2_mod) / diff(log2_scales)``) among the finite-point sequence, NaN if every local
    slope is non-finite (a repeated ``log2_scales`` value produces a zero denominator, guarded the
    same way ``diff(ds) > 1e-12`` guards the reference).
    """
    log2_s = np.asarray(chain.get("log2_scales", ()), dtype=np.float64)
    log2_m = np.asarray(chain.get("log2_mod", ()), dtype=np.float64)
    k = min(log2_s.size, log2_m.size)     # defensive: real chains always pair these 1:1
    log2_s, log2_m = log2_s[:k], log2_m[:k]

    finite = np.isfinite(log2_s) & np.isfinite(log2_m)
    n = int(finite.sum())
    if n < 3:
        return (float("nan"), float("nan"), float("nan"), float("nan"), n)

    s = log2_s[finite]
    m = log2_m[finite]
    coef = np.polyfit(s, m, 1)
    h_ols = float(coef[0])
    fit = np.polyval(coef, s)
    ss_res = float(np.sum((m - fit) ** 2))
    ss_tot = float(np.sum((m - m.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    ds = np.diff(s)
    dm = np.diff(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        sl = dm / np.where(np.abs(ds) > 1e-12, ds, np.nan)
    v = np.isfinite(sl)
    max_slope = float(np.max(sl[v])) if v.any() else float("nan")
    min_slope = float(np.min(sl[v])) if v.any() else float("nan")
    return (h_ols, r2, max_slope, min_slope, n)


def _chain_ols_holder(chain: dict) -> float:
    """OLS slope of ``log2_mod`` vs ``log2_scales``. NaN if fewer than 2 jointly-finite points --
    reimplements ``wtmm_ebsd.chain_filters.chain_ols_holder`` (:40) exactly, including its looser
    gate (2, not per_chain_stats's 3) and its lack of R²."""
    log2_s = np.asarray(chain.get("log2_scales", ()), dtype=np.float64)
    log2_m = np.asarray(chain.get("log2_mod", ()), dtype=np.float64)
    k = min(log2_s.size, log2_m.size)
    log2_s, log2_m = log2_s[:k], log2_m[:k]
    finite = np.isfinite(log2_s) & np.isfinite(log2_m)
    if finite.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(log2_s[finite], log2_m[finite], 1)
    return float(slope)


def _chain_max_slope_holder(chain: dict) -> float:
    """Maximum local slope of ``log2_mod`` vs ``log2_scales``. NaN if fewer than 2 points --
    reimplements ``wtmm_ebsd.chain_filters.chain_max_slope_holder`` (:53) exactly: the diff is
    taken over the RAW arrays (no joint-finite pre-filter), and only the resulting per-step slopes
    that come out non-finite are dropped."""
    log2_s = np.asarray(chain.get("log2_scales", ()), dtype=np.float64)
    log2_m = np.asarray(chain.get("log2_mod", ()), dtype=np.float64)
    k = min(log2_s.size, log2_m.size)
    log2_s, log2_m = log2_s[:k], log2_m[:k]
    if log2_s.size < 2:
        return float("nan")
    ds = np.diff(log2_s)
    dm = np.diff(log2_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = dm / ds
    finite = slopes[np.isfinite(slopes)]
    if finite.size == 0:
        return float("nan")
    return float(finite.max())


_ESTIMATORS = {"ols": _chain_ols_holder, "max": _chain_max_slope_holder}

#: Bounded identity memo: a filter tweak re-resolves the whole chain, so the SAME
#: cached transform ``chains`` list reaches ``stats_for`` on every tweak -- computing the metric
#: once and reusing it by ``id(chains)`` is EQSelect's "cache the per-chain metric" (its metric
#: registry). Strong ref held so a freed+realloc'd list can't collide on ``id``; small LRU since
#: only a couple of chain lists are ever live at once.
_METRIC_CACHE: "dict[tuple, tuple]" = {}
_METRIC_CACHE_MAX = 8


def _memoized(key, chains, compute):
    hit = _METRIC_CACHE.get(key)
    if hit is not None and hit[0] is chains:
        return hit[1]
    vals = compute()
    _METRIC_CACHE[key] = (chains, vals)
    while len(_METRIC_CACHE) > _METRIC_CACHE_MAX:
        del _METRIC_CACHE[next(iter(_METRIC_CACHE))]
    return vals


def _concat_log2(chains: list):
    """``(chain_id, counts, log2_s, log2_m)`` -- every chain's ``log2_scales``/``log2_mod`` flattened
    (each truncated to its own jointly-valid ``k = min(len_s, len_m)``, the per-chain functions'
    own defensive gate) with a per-point chain-id. The one Python pass over ``chains`` here reads
    already-built arrays (no per-chain polyfit), so it is ~30x cheaper than the per-chain loop it
    replaced; the actual estimators below are then pure vectorized numpy over the flat arrays."""
    n = len(chains)
    ls, lm, ks = [], [], np.empty(n, dtype=np.int64)
    for i, ch in enumerate(chains):
        s = np.asarray(ch.get("log2_scales", ()), dtype=np.float64)
        m = np.asarray(ch.get("log2_mod", ()), dtype=np.float64)
        k = min(s.size, m.size)
        ks[i] = k
        ls.append(s[:k]); lm.append(m[:k])
    log2_s = np.concatenate(ls) if n and ks.sum() else np.zeros(0)
    log2_m = np.concatenate(lm) if n and ks.sum() else np.zeros(0)
    chain_id = np.repeat(np.arange(n), ks)
    return chain_id, ks, log2_s, log2_m


def _vec_ols(chain_id, n, ls, lm) -> np.ndarray:
    """Vectorized per-chain OLS slope over jointly-finite pairs -- IDENTICAL semantics to
    :func:`_chain_ols_holder` (NaN below 2 finite pairs), one bincount pass instead of a polyfit
    per chain. ``tests/test_chain_stats.py`` pins it equal to the per-chain loop."""
    if n == 0:
        return np.zeros(0)
    finite = np.isfinite(ls) & np.isfinite(lm)
    w = finite.astype(np.float64)
    cnt = np.bincount(chain_id, weights=w, minlength=n)
    sx = np.bincount(chain_id, weights=np.where(finite, ls, 0.0), minlength=n)
    sy = np.bincount(chain_id, weights=np.where(finite, lm, 0.0), minlength=n)
    sxx = np.bincount(chain_id, weights=np.where(finite, ls * ls, 0.0), minlength=n)
    sxy = np.bincount(chain_id, weights=np.where(finite, ls * lm, 0.0), minlength=n)
    denom = cnt * sxx - sx * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (cnt * sxy - sx * sy) / denom
    slope[(cnt < 2) | ~np.isfinite(slope)] = np.nan
    return slope


def _vec_max(chain_id, ks, ls, lm) -> np.ndarray:
    """Vectorized per-chain MAX local slope -- IDENTICAL semantics to
    :func:`_chain_max_slope_holder` (diffs over the raw k-truncated arrays, non-finite step slopes
    dropped, NaN when a chain has < 2 points or no finite step)."""
    n = ks.size
    if n == 0:
        return np.zeros(0)
    out = np.full(n, -np.inf)
    if ls.size >= 2:
        within = chain_id[:-1] == chain_id[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            slopes = np.diff(lm) / np.diff(ls)
        ok = within & np.isfinite(slopes)
        if ok.any():
            np.maximum.at(out, chain_id[:-1][ok], slopes[ok])
    out[np.isneginf(out)] = np.nan
    out[ks < 2] = np.nan                       # a <2-point chain has no local slope at all
    return out


def stats_for(chains: list, estimator: str) -> np.ndarray:
    """Per-chain Hölder exponent, one estimator applied to every chain in ``chains``.

    ``estimator`` is ``"ols"`` (least-squares slope) or ``"max"`` (max local slope) -- the same
    two names, and the same NaN semantics, as ``wtmm_ebsd.chain_filters.filter_by_holder``'s own
    ``holder=`` keyword. Vectorized: a per-chain ``np.polyfit`` loop costs ~1 s over the 59k
    chains of a 2048 DEM window, and this runs up to THREE times per filter tweak (apply +
    reading + data_hints). The estimators below produce IDENTICAL values
    (``tests/test_chain_stats.py`` pins equality with the reference per-chain functions, retained
    above). Raises ``ValueError`` for an unknown estimator.
    """
    if estimator not in _ESTIMATORS:
        raise ValueError(f"estimator must be 'ols' or 'max'; got {estimator!r}")

    def _compute():
        chain_id, ks, ls, lm = _concat_log2(chains)
        return _vec_ols(chain_id, len(chains), ls, lm) if estimator == "ols" \
            else _vec_max(chain_id, ks, ls, lm)

    return _memoized((id(chains), "holder", estimator), chains, _compute)


def max_log2_modulus_for(chains: list) -> np.ndarray:
    """Per-chain maximum finite ``log2_mod``, NaN if missing/empty/all-non-finite -- reimplements
    ``wtmm_ebsd.chain_filters.chain_max_log2_modulus`` (:28). Vectorized via one ``reduceat`` over
    the flattened moduli, identical values to the per-chain loop it replaced."""
    n = len(chains)
    if n == 0:
        return np.zeros(0)

    def _compute():
        arrs = [np.asarray(ch.get("log2_mod", ()), dtype=np.float64) for ch in chains]
        counts = np.fromiter((a.size for a in arrs), np.int64, n)
        out = np.full(n, np.nan)
        if counts.sum() == 0:
            return out
        flat = np.concatenate(arrs)
        flat = np.where(np.isfinite(flat), flat, -np.inf)  # ignore non-finite in the per-chain max
        off = np.concatenate([[0], np.cumsum(counts)[:-1]])
        nonempty = counts > 0
        peaks = np.maximum.reduceat(flat, off[nonempty])
        vals = np.full(n, -np.inf)
        vals[nonempty] = peaks
        out[np.isfinite(vals)] = vals[np.isfinite(vals)]   # all-non-finite chain stays NaN
        return out

    return _memoized((id(chains), "max_log2_mod"), chains, _compute)


def length_for(chains: list) -> np.ndarray:
    """Per-chain vertical (across-scale) length -- reimplements ``wtmm_ebsd.chain_filters.
    chain_length`` (:69): looks at ``"mod"`` first, then ``"log2_mod"``, then ``"log2_scales"``;
    0 if none of those keys is present."""
    out = np.empty(len(chains), dtype=np.int64)
    for i, ch in enumerate(chains):
        n = 0
        for key in ("mod", "log2_mod", "log2_scales"):
            v = ch.get(key)
            if v is not None:
                n = len(v)
                break
        out[i] = n
    return out
