"""Partition-function machinery for chain-based 2D WTMM multifractal analysis.

This is a faithful port of v1's `build_hd_from_chains` /
`fit_hq_Dq_weighted` / `compute_cumulants_log2` from the per-grain
Pantleon notebook (PARTITION_AND_SPECTRUM_CELL + CUMULANT_CELL).

Two distinct cumulant fitters are exposed because they ARE different
estimators of the same Taylor expansion `τ(q) = -c0 + c1 q − c2 q²/2 + c3 q³/6`:

  - `dupont_fit_polynomial`    : weighted polynomial fit of τ(q) directly
  - `venugopal_fit_cumulants`  : per-scale cumulants C_n(a) of log|T|, then
                                  slopes vs log(a) → c_n with sign convention

η-correction (fractional pseudo-integration of order η, Wendt et al. 2009):

  - `tau_orig(q)  = tau_int(q) - q·η`
  - `h_orig       = h_int      - η`
  - `D_orig(h)    = D_int(h+η)`            (D values unchanged, x-axis shifts)
  - `c_1_orig     = c_1_int    - η`
  - `c_0, c_2, c_3` unchanged

`fit_hq_Dq_weighted`, `dupont_fit_polynomial`, `venugopal_fit_cumulants`
take an `eta` and `frame` ∈ {'integrated', 'original'}.  When
`frame='original'` the returned spectra are shifted back to the
unintegrated signal frame; with `frame='integrated'` they are returned
as numerically computed (the integrated frame).
"""

from __future__ import annotations

import numpy as np

# MLX (Apple Silicon GPU) — optional, falls back to NumPy if missing
try:
    import mlx.core as _mx
    _HAVE_MLX = True
except ImportError:
    _mx = None
    _HAVE_MLX = False

# ---------------------------------------------------------------------------
# defaults / constants
# ---------------------------------------------------------------------------

#: default q grid for partition functions (matches v1)
DEFAULT_Q_LIST: np.ndarray = np.arange(-3.0, 6.0 + 1e-9, 0.1)

#: ln 2 (used to convert natural-log accumulators to log2)
_LOG2C = np.log(2.0)


# ---------------------------------------------------------------------------
# cumulants of log|T| per scale (Venugopal et al. 2005)
# ---------------------------------------------------------------------------

def compute_cumulants_log2(
    chains: list[dict],
    n_scales: int,
    use_chainmax: bool = False,
    *,
    min_chain_len: int = 2,
) -> np.ndarray:
    """Per-scale cumulants C_0..C_3 of log2|T(a)| over the chain ensemble.

    Returns a (4, n_scales) array.  Convention (matching v1):
        C[0, si] = log2(N(si))                  # support count, slope -> -c0
        C[1, si] = <log2|T|>                    # mean,          slope -> +c1
        C[2, si] = var(log2|T|)                 # variance,      slope -> -c2
        C[3, si] = <(log2|T| - mean)^3>         # third central, slope -> +c3

    Parameters
    ----------
    chains : list of chain dicts (each with 'mod' array of length ≤ n_scales)
    n_scales : number of analyzing scales
    use_chainmax : if True, replace |T(a)| along each chain with the running
        maximum from finest to scale a (chainmax convention).
    min_chain_len : int, default 2
        Drop chains whose vertical (across-scale) length is below this floor
        BEFORE building the cumulant matrix.  Length-0 chains contribute
        nothing; length-1 chains have a single sample at one scale only and
        cannot inform any inter-scale statistic.  Set higher (5+) to require
        chains spanning a wider scale plateau.

    Notes
    -----
    Vectorized (chains × scales matrix).  Scales with fewer than 5 surviving
    chains are returned as NaN to avoid spurious cumulants from tiny samples.
    """
    if min_chain_len > 1:
        chains = [ch for ch in chains if len(np.asarray(ch.get("mod", []))) >= min_chain_len]
    n_ch = len(chains)
    if n_ch == 0:
        return np.full((4, n_scales), np.nan)

    # Pad |T| to a (n_ch, n_sc) matrix; NaN where chain is short or modulus ≤ 0.
    M = np.full((n_ch, n_scales), np.nan, dtype=np.float64)
    for i, ch in enumerate(chains):
        mv = np.abs(np.asarray(ch["mod"], dtype=np.float64))
        k = min(len(mv), n_scales)
        if k == 0:
            continue
        M[i, :k] = np.where(mv[:k] > 0, mv[:k], np.nan)

    if use_chainmax:
        alive = np.isfinite(M)
        Mfill = np.where(alive, M, -np.inf)
        M = np.maximum.accumulate(Mfill, axis=1)
        M = np.where(alive & (M > 0), M, np.nan)

    L = np.log2(M)                                    # NaN-safe
    C = np.full((4, n_scales), np.nan)
    N = np.sum(np.isfinite(L), axis=0)
    mask5 = N >= 5
    if not mask5.any():
        return C

    mean = np.nanmean(L, axis=0)
    var = np.nanvar(L, axis=0)                        # ddof=0 (population)
    L_cent = L - mean[None, :]
    cube = np.where(np.isfinite(L_cent), L_cent ** 3, np.nan)
    third = np.nanmean(cube, axis=0)

    C[0, mask5] = np.log2(N[mask5].astype(float))
    C[1, mask5] = mean[mask5]
    C[2, mask5] = var[mask5]
    C[3, mask5] = third[mask5]
    return C


# ---------------------------------------------------------------------------
# partition function with per-scale moments and sample-variance accumulators
# ---------------------------------------------------------------------------

def build_hd_from_chains(
    chains: list[dict],
    scales: np.ndarray,
    q_list: np.ndarray | None = None,
    *,
    use_mlx: bool | None = None,
    min_chain_len: int = 2,
) -> tuple[dict, dict]:
    """Per-(q, scale) partition function for both standard and chainmax modes.

    Returns ``(hd_std, hd_cmax)``, each a dict with keys::

        q_list        (n_q,)              q grid
        scales        (n_sc,)              scale values (whatever units are passed)
        log2_scales   (n_sc,)              log2(scales)
        N_a           (n_sc,)              chain count at each scale
        tau_qa        (n_q, n_sc)          log2 Z(q, a), per-scale
        h_qa          (n_q, n_sc)          Σ_k μ_k(q) log2|T_k|, Boltzmann avg
        D_qa          (n_q, n_sc)          q·h - tau, per-scale Chhabra-Jensen
        sigma_tau_qa  (n_q, n_sc)          per-scale sample-variance σ on tau
        sigma_h_qa    (n_q, n_sc)          per-scale σ on h
        sigma_D_qa    (n_q, n_sc)          per-scale σ on D

    The slopes of ``tau_qa, h_qa, D_qa`` vs ``log2(scales)`` (over a chosen
    scale plateau) give ``τ(q), h(q), D(q)`` respectively.  See
    :func:`fit_hq_Dq_weighted` for the WLS slope fit with sample-variance
    error propagation.

    Parameters
    ----------
    chains : list of dicts with at least ``'mod'`` (array of |T| along chain).
    scales : (n_sc,) array of analyzing scales.
    q_list : optional q grid; defaults to ``DEFAULT_Q_LIST``.
    min_chain_len : int, default 2
        Drop chains whose vertical (across-scale) length is below this floor
        BEFORE the partition-function accumulation.  A length-0 chain
        contributes nothing; a length-1 chain inflates ``N_a[0]`` and
        ``Z_q[:, 0]`` without carrying any inter-scale signal.  Raise to
        require chains spanning a wider scale plateau (5–10 typical for
        cumulant fits).
    """
    if min_chain_len > 1:
        chains = [ch for ch in chains if len(np.asarray(ch.get("mod", []))) >= min_chain_len]
    if q_list is None:
        q_list = DEFAULT_Q_LIST
    q_arr = np.asarray(q_list, dtype=np.float64)
    n_q = len(q_arr)
    n_sc = len(scales)

    def _zeros() -> np.ndarray:
        return np.zeros((n_q, n_sc), dtype=np.float64)

    n_chains = len(chains)
    scales_arr = np.asarray(scales, dtype=np.float64)
    log2_scales = np.log2(scales_arr)

    # ---- Build (n_chains, n_sc) padded modulus matrix ----
    # NaN where chain is short or modulus <= 0.  This drops the per-chain
    # / per-scale Python loop entirely.
    M = np.full((n_chains, n_sc), np.nan, dtype=np.float64)
    for i, ch in enumerate(chains):
        mods = np.abs(np.asarray(ch["mod"], dtype=np.float64))
        k = min(len(mods), n_sc)
        if k == 0:
            continue
        M[i, :k] = np.where(mods[:k] > 0, mods[:k], np.nan)

    # ---- Chainmax: running max along each chain (axis=1) ----
    alive = np.isfinite(M)
    Mfill = np.where(alive, M, -np.inf)
    M_cmax = np.maximum.accumulate(Mfill, axis=1)
    # Restore NaNs where the chain doesn't exist; clip silly negatives
    M_cmax = np.where(alive & (M_cmax > 0), M_cmax, np.nan)

    # Per-scale chain counts (Na) come from the standard mask
    Na = alive.sum(axis=0).astype(np.int64)

    use_mlx = (_HAVE_MLX if use_mlx is None else (use_mlx and _HAVE_MLX))

    def _accumulate_per_scale_numpy(M_in: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """NumPy fallback (matches v1's vectorisation pattern).

        Per scale: ONE np.power for m^q, then m2q = mq * mq (cheap square).
        No q=0 mask correction — np.power(m_positive, 0) is 1.0 natively.

        Returns (Zq, ZqL, Z2q, ZqL2), each (n_q, n_sc).
        """
        Zq   = np.zeros((n_q, n_sc), dtype=np.float64)
        ZqL  = np.zeros((n_q, n_sc), dtype=np.float64)
        Z2q  = np.zeros((n_q, n_sc), dtype=np.float64)
        ZqL2 = np.zeros((n_q, n_sc), dtype=np.float64)
        for si in range(n_sc):
            col = M_in[:, si]
            valid = np.isfinite(col)
            if not valid.any():
                continue
            m = col[valid]                               # (n_valid_si,)
            ln_m = np.log(m)
            # (n_q, n_valid) — q-axis OUTER for axis=1 reductions
            mq = np.power(m[None, :], q_arr[:, None])    # (n_q, n_valid)
            m2q = mq * mq                                 # cheap elementwise square
            Zq[:, si]   = mq.sum(axis=1)
            Z2q[:, si]  = m2q.sum(axis=1)
            ZqL[:, si]  = (mq * ln_m[None, :]).sum(axis=1)
            ZqL2[:, si] = (mq * (ln_m[None, :] ** 2)).sum(axis=1)
        return Zq, ZqL, Z2q, ZqL2

    def _accumulate_all_mlx(M_in: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """MLX (Apple Silicon GPU) all-at-once kernel.

        Computes ``mq[ch, si, q] = exp(q * log(m[ch, si]))`` as a single 3D
        tensor, then reduces over the chain axis.  Memory: roughly
        ``n_chains × n_sc × n_q × 4`` bytes (float32) for the intermediate.
        For 8k × 44 × 91 that's ~128 MB per intermediate — fits comfortably
        on M-series unified memory.
        """
        n_chains_local = M_in.shape[0]
        valid_np = np.isfinite(M_in)
        # Fill invalid with 1.0 so log = 0 and m^q = 1.  We mask out the
        # contribution after computing mq via the valid mask.
        M_safe = np.where(valid_np, M_in, 1.0).astype(np.float32, copy=False)
        log_M_np = np.where(valid_np, np.log(M_safe), 0.0).astype(np.float32)
        valid_f32 = valid_np.astype(np.float32)

        log_M = _mx.array(log_M_np)                        # (n_chains, n_sc)
        valid_mlx = _mx.array(valid_f32)                    # (n_chains, n_sc)
        q_mlx = _mx.array(q_arr.astype(np.float32))          # (n_q,)

        # Broadcast: (n_chains, n_sc, 1) × (1, 1, n_q) → (n_chains, n_sc, n_q)
        qlogm = log_M[..., None] * q_mlx[None, None, :]
        mq = _mx.exp(qlogm) * valid_mlx[..., None]           # mask invalid -> 0
        m2q = mq * mq

        log_M_b = log_M[..., None]                           # (n_chains, n_sc, 1)

        # Reduce over chain axis -> (n_sc, n_q); transpose to (n_q, n_sc)
        Zq   = mq.sum(axis=0).T
        Z2q  = m2q.sum(axis=0).T
        ZqL  = (mq * log_M_b).sum(axis=0).T
        ZqL2 = (mq * (log_M_b * log_M_b)).sum(axis=0).T

        _mx.eval(Zq, ZqL, Z2q, ZqL2)
        return (np.asarray(Zq, dtype=np.float64),
                np.asarray(ZqL, dtype=np.float64),
                np.asarray(Z2q, dtype=np.float64),
                np.asarray(ZqL2, dtype=np.float64))

    _accumulate = _accumulate_all_mlx if use_mlx else _accumulate_per_scale_numpy
    Zq_s, ZqL_s, Z2q_s, ZqL2_s = _accumulate(M)
    Zq_c, ZqL_c, Z2q_c, ZqL2_c = _accumulate(M_cmax)

    def _derive(Zq, ZqL, Z2q, ZqL2) -> dict:
        tau = np.full_like(Zq, np.nan)
        h = np.full_like(Zq, np.nan)
        D = np.full_like(Zq, np.nan)
        sig_tau = np.full_like(Zq, np.nan)
        sig_h = np.full_like(Zq, np.nan)
        sig_D = np.full_like(Zq, np.nan)

        N2 = np.maximum(Na, 1).astype(np.float64)[None, :]
        vZ = Zq > 0
        tau[vZ] = np.log(Zq[vZ]) / _LOG2C
        h[vZ] = ZqL[vZ] / (_LOG2C * Zq[vZ])
        qmat = np.broadcast_to(q_arr[:, None], Zq.shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            D[vZ] = (qmat[vZ] * ZqL[vZ] / (Zq[vZ] * _LOG2C)
                     - np.log(Zq[vZ]) / _LOG2C)

        # Sample variance of Z_q (chains i.i.d. at each scale, given N fixed).
        var_Zq = np.maximum(Z2q - Zq * Zq / N2, 0.0)
        sig_tau_sv = np.sqrt(var_Zq) / np.maximum(Zq * _LOG2C, 1e-300)
        # Shot-noise floor on tau (chain count itself fluctuates ~sqrt(N))
        sig_tau_shot = 1.0 / np.sqrt(N2) / _LOG2C
        sig_tau[vZ] = np.sqrt(sig_tau_sv[vZ] ** 2
                              + np.broadcast_to(sig_tau_shot, Zq.shape)[vZ] ** 2)

        # Canonical-average σ via tilted distribution + effective sample size
        Neff = np.where(Z2q > 0, Zq * Zq / np.maximum(Z2q, 1e-300), 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            var_q_logm = (ZqL2 / np.maximum(Zq, 1e-300)
                          - (ZqL / np.maximum(Zq, 1e-300)) ** 2)
        var_q_logm = np.maximum(var_q_logm, 0.0)
        sig_h[vZ] = np.sqrt(var_q_logm[vZ]
                            / np.maximum(Neff[vZ], 1.0)) / _LOG2C

        # σ(D) by quadrature (ignores Cov(h, log Z))
        sig_D[vZ] = np.sqrt(qmat[vZ] ** 2 * sig_h[vZ] ** 2
                            + sig_tau[vZ] ** 2)

        return {
            "tau_qa": tau, "h_qa": h, "D_qa": D,
            "sigma_tau_qa": sig_tau, "sigma_h_qa": sig_h, "sigma_D_qa": sig_D,
            "N_a": Na,
            "q_list": q_arr,
            "scales": scales_arr,
            "log2_scales": log2_scales,
        }

    return (_derive(Zq_s, ZqL_s, Z2q_s, ZqL2_s),
            _derive(Zq_c, ZqL_c, Z2q_c, ZqL2_c))


# ---------------------------------------------------------------------------
# WLS slope fit τ(q), h(q), D(q) over scale plateau (with η correction)
# ---------------------------------------------------------------------------

def _validate_frame(frame: str) -> str:
    if frame not in ("integrated", "original"):
        raise ValueError(
            f"frame must be 'integrated' or 'original', got {frame!r}"
        )
    return frame


def fit_hq_Dq_weighted(
    hd: dict,
    scale_range: tuple[float, float],
    eta: float = 0.0,
    frame: str = "original",
) -> dict:
    """Fit τ(q), h(q), D(q) by WLS over a scale plateau.

    The fit values come from log-log slopes of ``hd['tau_qa']``,
    ``hd['h_qa']``, ``hd['D_qa']`` against ``hd['log2_scales']``.  Error
    bars come from per-scale sample-variance σ propagated through the
    OLS slope formula::

        Var(slope) = Σ (x_i - xbar)² σ_i² / (Σ (x_i - xbar)²)²

    Parameters
    ----------
    hd : output of :func:`build_hd_from_chains` (one of the two dicts).
    scale_range : (smin, smax) in the same units as ``hd['scales']``.
    eta : fractional pseudo-integration order applied to coefficients.
        Default 0 (no correction).  See module docstring.
    frame : 'original' applies the η correction to recover the
        un-integrated signal's spectrum (subtracts ``q·η`` from τ,
        ``η`` from h).  'integrated' returns spectra in the
        as-computed (integrated) frame.

    Returns
    -------
    dict with keys::
        q_list   (n_q,)
        tau, h, D                fitted slopes (corrected per ``frame``)
        tau_err, h_err, D_err    1-σ error bars (frame-invariant)
    """
    frame = _validate_frame(frame)
    q_arr = hd["q_list"]
    scales = hd["scales"]
    log2_s = hd["log2_scales"]
    smin, smax = scale_range
    mask = (scales >= smin) & (scales <= smax)
    n_q = len(q_arr)

    tau = np.full(n_q, np.nan)
    h = np.full(n_q, np.nan)
    D = np.full(n_q, np.nan)
    tau_err = np.full(n_q, np.nan)
    h_err = np.full(n_q, np.nan)
    D_err = np.full(n_q, np.nan)

    x = log2_s[mask]
    if x.size < 3:
        return {
            "q_list": q_arr,
            "tau": tau, "h": h, "D": D,
            "tau_err": tau_err, "h_err": h_err, "D_err": D_err,
            "scale_range": (smin, smax),
            "eta": eta, "frame": frame,
        }

    xm = x.mean()
    dx = x - xm
    denom = float((dx * dx).sum())
    if denom <= 0:
        return {
            "q_list": q_arr,
            "tau": tau, "h": h, "D": D,
            "tau_err": tau_err, "h_err": h_err, "D_err": D_err,
            "scale_range": (smin, smax),
            "eta": eta, "frame": frame,
        }

    for qi in range(n_q):
        # OLS slope of value vs log2 a (no weighting on the slope itself —
        # WLS only enters through the propagated error bar).
        for key_val, out_arr in (
            ("tau_qa", tau), ("h_qa", h), ("D_qa", D)
        ):
            y = hd[key_val][qi, mask]
            v = np.isfinite(y)
            if v.sum() >= 3:
                slope, _ = np.polyfit(x[v], y[v], 1)
                out_arr[qi] = slope
        # propagated 1-σ on each slope
        for key_sig, err_arr in (
            ("sigma_tau_qa", tau_err),
            ("sigma_h_qa", h_err),
            ("sigma_D_qa", D_err),
        ):
            s = hd[key_sig][qi, mask]
            v = np.isfinite(s) & (s > 0)
            if v.sum() >= 3:
                var_slope = float((dx[v] ** 2 * s[v] ** 2).sum()) / (denom ** 2)
                err_arr[qi] = float(np.sqrt(max(var_slope, 0.0)))

    if frame == "original" and eta != 0.0:
        tau = tau - eta * q_arr
        h = h - eta
        # D unchanged at fixed q; D(h) shifts horizontally because h shifted.

    return {
        "q_list": q_arr,
        "tau": tau, "h": h, "D": D,
        "tau_err": tau_err, "h_err": h_err, "D_err": D_err,
        "scale_range": (smin, smax),
        "eta": eta, "frame": frame,
    }


# ---------------------------------------------------------------------------
# Dupont method: weighted polynomial fit of τ(q)
# ---------------------------------------------------------------------------

def dupont_fit_polynomial(
    q: np.ndarray,
    tau: np.ndarray,
    tau_err: np.ndarray | None = None,
    deg: int = 2,
) -> dict:
    """Weighted polynomial fit of τ(q) -> Taylor cumulants c_p.

    Convention (matches v1)::
        τ(q) = -c0 + c1·q − c2·q²/2 + c3·q³/6

    For ``deg=2`` returns c0, c1, c2 only (c3=NaN).  For ``deg≥3`` returns
    c0..c3 (higher orders not produced by this fitter).

    Parameters
    ----------
    q, tau : aligned arrays from :func:`fit_hq_Dq_weighted`.  τ is
        already in whatever frame ('original' / 'integrated') the caller
        chose; this function does not re-shift.
    tau_err : optional 1-σ on τ for WLS weighting.  None -> equal weights.
    deg : polynomial degree (2 or 3 typical).

    Returns
    -------
    dict with::
        c0, c1, c2, c3      (c3=NaN for deg<3)
        coefs               numpy polyfit output (high → low order)
        residuals           τ_data - τ_fit
        r2                  goodness-of-fit
        tau_fit             evaluated polynomial at the input q
        deg                 polynomial degree used
    """
    q = np.asarray(q, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    valid = np.isfinite(q) & np.isfinite(tau)
    out = {"c0": np.nan, "c1": np.nan, "c2": np.nan, "c3": np.nan,
           "coefs": None, "residuals": None, "r2": np.nan,
           "tau_fit": np.full_like(q, np.nan), "deg": deg, "n_pts": int(valid.sum())}
    if valid.sum() < deg + 1:
        return out

    qv, tv = q[valid], tau[valid]
    if tau_err is not None:
        te = np.asarray(tau_err, dtype=np.float64)[valid]
        good = np.isfinite(te) & (te > 0)
        w = np.where(good, 1.0 / te, 1.0)
    else:
        w = np.ones_like(qv)

    coefs = np.polyfit(qv, tv, deg, w=w)            # high → low order
    out["coefs"] = coefs
    out["tau_fit"] = np.polyval(coefs, q)
    out["residuals"] = tau - out["tau_fit"]
    ss_res = float(np.nansum((tv - np.polyval(coefs, qv)) ** 2))
    ss_tot = float(np.nansum((tv - np.nanmean(tv)) ** 2))
    out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Read off cumulants from coefficients (high → low):
    # τ(q) = a_d q^d + ... + a_2 q² + a_1 q + a_0
    # Compare to τ(q) = -c0 + c1 q - c2 q²/2 + c3 q³/6
    a0 = coefs[-1]
    a1 = coefs[-2] if deg >= 1 else 0.0
    a2 = coefs[-3] if deg >= 2 else 0.0
    a3 = coefs[-4] if deg >= 3 else None
    out["c0"] = -a0
    out["c1"] = a1
    out["c2"] = -2.0 * a2
    if a3 is not None:
        out["c3"] = 6.0 * a3
    return out


# ---------------------------------------------------------------------------
# Venugopal method: cumulants of log|T| per scale -> slopes vs log(a)
# ---------------------------------------------------------------------------

def venugopal_fit_cumulants(
    C: np.ndarray,
    log2_scales: np.ndarray,
    scale_range: tuple[float, float],
    scales: np.ndarray | None = None,
    eta: float = 0.0,
    frame: str = "original",
) -> dict:
    """Slopes of cumulants C_n(a) vs log2(a) over plateau -> c_0..c_3.

    Sign convention (matching v1)::
        c_0 = -slope(C_0)         # support count exponent (≈ d_F)
        c_1 = +slope(C_1)         # Hölder location  (η-shifts: c_1 -> c_1 - η)
        c_2 = -slope(C_2)         # intermittency
        c_3 = +slope(C_3)         # asymmetry

    Parameters
    ----------
    C : (4, n_sc) cumulant table from :func:`compute_cumulants_log2`.
    log2_scales : (n_sc,) log2 of scale array.
    scale_range : (smin, smax) in same units as ``scales``.
    scales : (n_sc,) scale values, used only to build the plateau mask.
        If None, ``scale_range`` is interpreted in log2 units of
        ``log2_scales`` directly.
    eta, frame : as in :func:`fit_hq_Dq_weighted`.

    Returns
    -------
    dict with::
        c0, c1, c2, c3
        slopes              raw OLS slopes for each n
        n_pts               # of plateau points used
        eta, frame
    """
    frame = _validate_frame(frame)
    if scales is None:
        smin, smax = scale_range
        mask = (log2_scales >= np.log2(smin)) & (log2_scales <= np.log2(smax))
    else:
        smin, smax = scale_range
        mask = (np.asarray(scales) >= smin) & (np.asarray(scales) <= smax)

    sign_map = {0: -1, 1: +1, 2: -1, 3: +1}
    slopes = np.full(4, np.nan)
    cn = np.full(4, np.nan)
    for n in range(4):
        y = C[n]
        v = np.isfinite(y) & mask
        if v.sum() >= 3:
            sl, _ = np.polyfit(log2_scales[v], y[v], 1)
            slopes[n] = sl
            cn[n] = sign_map[n] * sl

    if frame == "original" and eta != 0.0:
        # Only c_1 shifts under fractional integration.
        cn[1] = cn[1] - eta if np.isfinite(cn[1]) else cn[1]

    return {
        "c0": cn[0], "c1": cn[1], "c2": cn[2], "c3": cn[3],
        "slopes": slopes,
        "n_pts": int(mask.sum()),
        "eta": eta, "frame": frame,
        "scale_range": (smin, smax),
    }


# ---------------------------------------------------------------------------
# convenience: reconstruct τ(q) and D(h) from fitted cumulants
# ---------------------------------------------------------------------------

def tau_from_cumulants(
    q: np.ndarray,
    c0: float, c1: float, c2: float, c3: float = 0.0,
) -> np.ndarray:
    """τ(q) ≈ -c0 + c1·q − (c2/2)·q² + (c3/6)·q³  (lognormal expansion)."""
    q = np.asarray(q, dtype=np.float64)
    out = -c0 + c1 * q
    if np.isfinite(c2):
        out -= 0.5 * c2 * q ** 2
    if np.isfinite(c3):
        out += c3 * q ** 3 / 6.0
    return out


def Dh_parabola_from_cumulants(
    h_grid: np.ndarray,
    c0: float, c1: float, c2: float,
) -> np.ndarray:
    """Lognormal-parabola D(h) = c0 - (h - c1)² / (2 c2).

    Returns NaN where D < 0 (off-spectrum).  Use this for the Dupont
    overlay; for non-zero c3 prefer the parametric Legendre form
    via :func:`tau_from_cumulants` evaluated on a q-grid.
    """
    if not (np.isfinite(c0) and np.isfinite(c1) and np.isfinite(c2)) or c2 <= 0:
        return np.full_like(h_grid, np.nan, dtype=np.float64)
    h = np.asarray(h_grid, dtype=np.float64)
    D = c0 - (h - c1) ** 2 / (2.0 * c2)
    return np.where(D >= 0, D, np.nan)


# ---------------------------------------------------------------------------
# Auto-plateau selection (Freddie recipe: h-flatness gate + τ-fit)
# ---------------------------------------------------------------------------

def _longest_flat_window(h, log2_a, in_bounds_mask, flatness_thresh, min_octaves):
    """Longest contiguous index range where h is statistically flat.

    A window [i_start, i_end) is flat iff every entry is finite and within
    in_bounds, std(h)/max(|mean(h)|, 0.01) < flatness_thresh, and the
    log2(scale) span is >= min_octaves. Brute-force O(n_sc²) — n_sc is
    typically <100 so this is microseconds.

    Returns (i_start, i_end) or None.
    """
    n = len(h)
    finite_in = np.isfinite(h) & in_bounds_mask
    best = (0, 0); best_len = 0
    for i_start in range(n):
        if not finite_in[i_start]:
            continue
        for i_end in range(i_start + 3, n + 1):
            sl = slice(i_start, i_end)
            if not finite_in[sl].all():
                break  # window must be contiguous + finite
            window = h[sl]
            m = float(np.mean(window))
            s = float(np.std(window))
            if s / max(abs(m), 0.01) > flatness_thresh:
                continue
            if log2_a[i_end - 1] - log2_a[i_start] < min_octaves:
                continue
            if (i_end - i_start) > best_len:
                best = (i_start, i_end); best_len = i_end - i_start
    return best if best_len > 0 else None


def auto_select_plateau(
    hd: dict,
    q_test: tuple = (0.0, 2.0),
    s_bounds: tuple | None = None,
    min_octaves: float = 2.0,
    flatness_thresh: float = 0.05,
) -> tuple | None:
    """Auto-select scale plateau (s_min, s_max) per Freddie's recipe.

    Two-stage:
        1. h-flatness GATE — for each q in q_test, find the longest contiguous
           window where the local Hölder h(q, a) is statistically flat
           (std/|mean| < flatness_thresh) within shared bounds and at least
           min_octaves long. INTERSECT windows across q_test.
        2. The returned plateau is what `fit_hq_Dq_weighted` should use.

    Stationarity-first (h-flatness) avoids the failure mode where R²-on-τ
    rewards long windows that include slow curvature (which would still give
    R² > 0.99 even though the data isn't truly self-similar there).

    Parameters
    ----------
    hd : dict        from build_hd_from_chains. Needs h_qa, q_list, scales,
                     log2_scales.
    q_test : tuple   q values to test flatness at (default (0, 2) — central q
                     where data is most informative).
    s_bounds : (s_lo, s_hi) in the SAME units as hd['scales'] (px). If None,
               inherits from the inner part of the scales array (skip first
               and last).
    min_octaves : minimum log₂(s_max/s_min) of the returned plateau.
    flatness_thresh : max allowed std(h) / |mean(h)| within a flat window.

    Returns
    -------
    (s_min, s_max) : floats in the same units as hd['scales'], or None if no
                     window passed the flatness test for ALL q in q_test.
    """
    q_arr  = np.asarray(hd['q_list'])
    scales = np.asarray(hd['scales'])
    log2_s = np.asarray(hd['log2_scales'])
    h_qa   = np.asarray(hd['h_qa'])
    n_sc = len(scales)

    if s_bounds is None:
        s_lo, s_hi = float(scales[1]), float(scales[-2])
    else:
        s_lo, s_hi = float(s_bounds[0]), float(s_bounds[1])
    in_bounds = (scales >= s_lo) & (scales <= s_hi)
    if in_bounds.sum() < 4:
        return None

    q_indices = [int(np.argmin(np.abs(q_arr - qt))) for qt in q_test]

    windows = []
    for qi in q_indices:
        w = _longest_flat_window(h_qa[qi], log2_s, in_bounds,
                                  flatness_thresh, min_octaves)
        if w is None:
            return None  # one of the q's has no flat window in bounds
        windows.append(w)

    # Intersect windows across q_test
    i_start = max(w[0] for w in windows)
    i_end   = min(w[1] for w in windows)
    if i_end - i_start < 3:
        # Intersection too narrow: fall back to the longest single window
        windows.sort(key=lambda w: w[1] - w[0], reverse=True)
        i_start, i_end = windows[0]

    return (float(scales[i_start]), float(scales[i_end - 1]))
