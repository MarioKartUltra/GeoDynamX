# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Per-(q, a) partition-function tables from V-chains, in the standard and the chain-max modes.

The app's own port (spec 09, Phase B) of the vendored ``wtmm_ebsd.partition.build_hd_from_chains``,
checked against its golden (``tests/golden/vendor/dh_from_chains.npz``). Same schema, same
derivation: at each scale ``a`` the chains alive there contribute ``|T_k(a)|^q`` to ``Z(q, a)``, the
Boltzmann-weighted mean of ``log2|T_k|`` gives ``h(q, a)``, and ``D(q, a) = q h - τ`` is the
per-scale Chhabra–Jensen form; the per-scale sample variances give ``σ`` on each. The chain-max
mode replaces each modulus by the running maximum along its chain. The accumulation over chains is
a numba kernel in float64 (the vendored one summed in float32 on mlx when it could; the two agree
to float32 tolerance), parallel over scales.
"""
from __future__ import annotations

import numpy as np

#: The default q grid (the vendored ``DEFAULT_Q_LIST``, 0.1 steps from -3 to 6).
DEFAULT_Q_LIST: np.ndarray = np.arange(-3.0, 6.0 + 1e-9, 0.1)

_LOG2 = np.log(2.0)
_KERNELS = None


def moment_grid(lo: float, hi: float, step: float, *, max_points: int = 10000) -> np.ndarray:
    """A bounded ascending moment grid; include hi only when the increment reaches it."""
    lo, hi, step = float(lo), float(hi), float(step)
    if not all(np.isfinite(v) for v in (lo, hi, step)) or step <= 0 or hi < lo:
        raise ValueError("moment bounds must be finite and ascending, with a positive increment")
    count = int(np.floor((hi - lo) / step + 1e-10)) + 1
    if count > max_points:
        raise ValueError(f"moment grid exceeds {max_points} points")
    return np.minimum(np.round(lo + step * np.arange(count), 10), hi)


def _kernels():
    """The accumulation kernel, compiled on first use (``cache=True``)."""
    global _KERNELS
    if _KERNELS is None:
        import numba

        @numba.njit(parallel=True, cache=True)
        def accumulate(M, q, Zq, ZqL, Z2q, ZqL2):
            """Per scale (column of ``M``, NaN where a chain is absent) and per q: the sums of
            ``m^q``, ``m^q ln m``, ``m^2q`` and ``m^q (ln m)^2`` over the chains alive there."""
            n_ch, n_sc = M.shape
            n_q = q.shape[0]
            for si in numba.prange(n_sc):
                for i in range(n_ch):
                    m = M[i, si]
                    if not (m == m):            # NaN: the chain is not at this scale
                        continue
                    ln_m = np.log(m)
                    for k in range(n_q):
                        mq = np.exp(q[k] * ln_m)
                        Zq[k, si] += mq
                        Z2q[k, si] += mq * mq
                        ZqL[k, si] += mq * ln_m
                        ZqL2[k, si] += mq * ln_m * ln_m

        _KERNELS = accumulate
    return _KERNELS


def _modulus_matrix(chains: list, n_sc: int) -> np.ndarray:
    """``(n_chains, n_sc)`` of |T| along each chain from the finest scale, NaN where the chain is
    absent or the modulus is not positive."""
    M = np.full((len(chains), n_sc), np.nan, dtype=np.float64)
    for i, ch in enumerate(chains):
        mods = np.abs(np.asarray(ch["mod"], dtype=np.float64))
        k = min(len(mods), n_sc)
        if k:
            M[i, :k] = np.where(mods[:k] > 0, mods[:k], np.nan)
    return M


def _accumulate(M: np.ndarray, q: np.ndarray) -> tuple:
    n_q, n_sc = len(q), M.shape[1]
    sums = tuple(np.zeros((n_q, n_sc), dtype=np.float64) for _ in range(4))
    if M.size:
        _kernels()(np.ascontiguousarray(M), np.ascontiguousarray(q), *sums)
    return sums


def _derive(Zq, ZqL, Z2q, ZqL2, Na, q, scales, log2_scales) -> dict:
    tau = np.full_like(Zq, np.nan)
    h = np.full_like(Zq, np.nan)
    D = np.full_like(Zq, np.nan)
    sig_tau = np.full_like(Zq, np.nan)
    sig_h = np.full_like(Zq, np.nan)
    sig_D = np.full_like(Zq, np.nan)
    N2 = np.maximum(Na, 1).astype(np.float64)[None, :]
    vZ = Zq > 0
    tau[vZ] = np.log(Zq[vZ]) / _LOG2
    h[vZ] = ZqL[vZ] / (_LOG2 * Zq[vZ])
    qmat = np.broadcast_to(q[:, None], Zq.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        D[vZ] = qmat[vZ] * ZqL[vZ] / (Zq[vZ] * _LOG2) - np.log(Zq[vZ]) / _LOG2
    # the sample variance of Z_q (chains i.i.d. at each scale, N fixed) plus the shot-noise floor
    var_Zq = np.maximum(Z2q - Zq * Zq / N2, 0.0)
    sig_tau_sv = np.sqrt(var_Zq) / np.maximum(Zq * _LOG2, 1e-300)
    sig_tau_shot = np.broadcast_to(1.0 / np.sqrt(N2) / _LOG2, Zq.shape)
    sig_tau[vZ] = np.sqrt(sig_tau_sv[vZ] ** 2 + sig_tau_shot[vZ] ** 2)
    # the canonical average's σ through the tilted distribution and an effective sample size
    Neff = np.where(Z2q > 0, Zq * Zq / np.maximum(Z2q, 1e-300), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_q_logm = ZqL2 / np.maximum(Zq, 1e-300) - (ZqL / np.maximum(Zq, 1e-300)) ** 2
    var_q_logm = np.maximum(var_q_logm, 0.0)
    sig_h[vZ] = np.sqrt(var_q_logm[vZ] / np.maximum(Neff[vZ], 1.0)) / _LOG2
    sig_D[vZ] = np.sqrt(qmat[vZ] ** 2 * sig_h[vZ] ** 2 + sig_tau[vZ] ** 2)
    return {"tau_qa": tau, "h_qa": h, "D_qa": D,
            "sigma_tau_qa": sig_tau, "sigma_h_qa": sig_h, "sigma_D_qa": sig_D,
            "N_a": Na, "q_list": q, "scales": scales, "log2_scales": log2_scales}


def build_hd_from_chains(chains: list, scales, q_list=None, *, min_chain_len: int = 2) -> tuple:
    """``(hd_std, hd_cmax)``, each a dict of ``(n_q, n_sc)`` tables ``tau_qa``, ``h_qa``, ``D_qa``
    and their ``sigma_*``, with ``N_a`` (chains alive per scale), ``q_list``, ``scales`` and
    ``log2_scales``. Chains shorter than ``min_chain_len`` are left out before the accumulation;
    ``q_list`` defaults to :data:`DEFAULT_Q_LIST`."""
    if min_chain_len > 1:
        chains = [ch for ch in chains if len(np.asarray(ch.get("mod", []))) >= min_chain_len]
    q = np.asarray(DEFAULT_Q_LIST if q_list is None else q_list, dtype=np.float64)
    scales_arr = np.asarray(scales, dtype=np.float64)
    log2_scales = np.log2(scales_arr)
    M = _modulus_matrix(chains, len(scales_arr))
    alive = np.isfinite(M)
    M_cmax = np.maximum.accumulate(np.where(alive, M, -np.inf), axis=1)
    M_cmax = np.where(alive & (M_cmax > 0), M_cmax, np.nan)
    Na = alive.sum(axis=0).astype(np.int64)
    std = _derive(*_accumulate(M, q), Na, q, scales_arr, log2_scales)
    cmax = _derive(*_accumulate(M_cmax, q), Na, q, scales_arr, log2_scales)
    return std, cmax
