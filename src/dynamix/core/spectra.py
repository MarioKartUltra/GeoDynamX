# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""Multifractal spectrum scale-window fitter over the hd partition tables (pure numpy, NO GUI).

This is a **reimplementation**, not a port. It exists so the interactive tau(q)/D(h) fitter has a real, numpy-only engine
when the optional ``wtmm_ebsd`` package is absent. Per project rule, ``wtmm_ebsd`` is never
imported here and none of its code is copied -- written fresh against the documented reference
semantics of ``wtmm_ebsd.partition.fit_hq_Dq_weighted``, and cross-checked against the installed oracle to
float tolerance in ``tests/test_spectra.py``.

Input is one of the two per-(q, scale) partition dicts every scalar WTMM run already stamps
(``result["hd_std"]`` / ``result["hd_cmax"]`` -- nosup / sup convention). The fit is a cheap slope
re-fit over a chosen scale window, so the interactive re-fit is milliseconds; no WTMM recompute.

One deliberate signature change from the reference: the window is ``log2_a_min``/``log2_a_max``
against ``hd["log2_scales"]`` (the draggable-window knob lives on a log2 axis) instead of the
reference's linear ``scale_range``. log2 is monotone, so the selected scales are identical.
"""
from __future__ import annotations

import numpy as np


def fit_spectra(hd: dict, log2_a_min: float, log2_a_max: float, *,
                eta: float = 0.0, frame: str = "original",
                method: str = "canonical", fit_mode: str = "naive",
                log2_L: "float | None" = None) -> dict:
    """Fit tau(q), h(q), D(q) as log-log slopes over a scale window.

    Slopes are plain OLS of ``hd["tau_qa"]`` / ``hd["h_qa"]`` / ``hd["D_qa"]`` against
    ``hd["log2_scales"]`` restricted to the window, one fit per q (vectorized over q). The
    "weighted" part of the reference is the error bar only: per-scale sample-variance sigma
    (``hd["sigma_*_qa"]``) propagated through the OLS slope formula

        Var(slope) = sum((x_i - xbar)^2 sigma_i^2) / (sum((x_i - xbar)^2))^2

    with ``xbar`` and the denominator over the FULL window (reference convention), while each
    sigma row contributes only its finite, positive entries.

    ``eta``/``frame`` undo the forward fractional pseudo-integration of order eta (Wendt 2009)
    at the fit stage: ``frame="original"`` with ``eta != 0`` subtracts ``eta*q`` from tau and
    ``eta`` from h -- D(h) shifts horizontally, D values and every error bar are frame-invariant.
    ``frame="integrated"`` returns the spectra as numerically computed. The caller must pass the
    SAME eta the forward transform applied (``fracint_alpha``); there is no way to infer it here.
    """
    if frame not in ("integrated", "original"):
        raise ValueError(f"frame must be 'integrated' or 'original', got {frame!r}")
    if fit_mode not in ("naive", "fixed_focus", "focus"):
        raise ValueError(f"fit_mode must be 'naive', 'fixed_focus' or 'focus', got {fit_mode!r}")
    if fit_mode != "naive" and log2_L is None:
        raise ValueError(
            "fit_mode='fixed_focus'/'focus' needs log2_L, the log2 field extent -- the focus "
            "anchor is the SIGNAL LENGTH (Mukli Eq. (7): one cell spanning the field makes "
            "S(q, L) q-independent by construction), never the fit window's own top scale")
    if method in ("microcanonical", "legendre"):
        # Both names are reserved, refused loudly rather than silently answering with the
        # canonical estimator. 'legendre' is the Legendre-transform estimator of tau(q) --
        # still the CANONICAL (Arneodo WTMM) formalism, unimplemented in the reference too.
        # 'microcanonical' is accepted here only because the reference conflates it with
        # legendre; properly it names Turiel's local-singularity-exponent formalism (with
        # most-singular-component detection) -- a different pipeline that would not be a
        # method of this partition-table fitter at all.
        raise NotImplementedError(
            f"method={method!r} is not implemented; only method='canonical' (Arneodo WTMM) "
            "is available. 'legendre' (Legendre transform of tau(q), canonical family) is "
            "unimplemented; Turiel's microcanonical formalism / MSC detection is a possible "
            "future feature and would not be a method of this fitter")
    if method != "canonical":
        raise ValueError(f"method must be 'canonical', got {method!r}")
    q_arr = np.asarray(hd["q_list"], dtype=np.float64)
    log2_s = np.asarray(hd["log2_scales"], dtype=np.float64)
    mask = (log2_s >= log2_a_min) & (log2_s <= log2_a_max)
    x = log2_s[mask]
    if x.size < 3:
        return _all_nan_result(q_arr, eta, frame)
    dx = x - x.mean()
    denom = float((dx * dx).sum())
    if denom <= 0:
        return _all_nan_result(q_arr, eta, frame)

    out = {"q_list": q_arr}
    for key, name in (("tau_qa", "tau"), ("h_qa", "h"), ("D_qa", "D")):
        Y = np.asarray(hd[key], dtype=np.float64)[:, mask]
        out[name] = _masked_slopes(x, Y)
    for key, name in (("sigma_tau_qa", "tau_err"), ("sigma_h_qa", "h_err"),
                      ("sigma_D_qa", "D_err")):
        S = np.asarray(hd[key], dtype=np.float64)[:, mask]
        out[name] = _propagated_slope_sigma(dx, denom, S)

    if fit_mode != "naive":
        # Focus regression over the PARTITION channel: fit the power-mean
        # columns Y_q = (tau_qa - log2 N_a) / q jointly through one focus (Schadner closed
        # forms; Mukli anchor), then reconstruct tau(q) = q*h_q + slope(log2 N_a) and
        # tau(0) = slope(log2 N_a) exactly. h/D channels keep their own per-q estimators.
        nz = q_arr != 0.0
        N_a = np.asarray(hd["N_a"], dtype=np.float64)[mask]
        tau_w = np.asarray(hd["tau_qa"], dtype=np.float64)[:, mask]
        focus_info: dict = {"fit_mode": fit_mode}
        if x.size < 5 or int(nz.sum()) < 3 or np.any(N_a <= 0) or \
                not np.all(np.isfinite(tau_w[nz])):
            focus_info.update(branch="unavailable",
                              reason="needs >= 5 finite scales, >= 3 nonzero q, N_a > 0")
        else:
            log2N = np.log2(N_a)
            s_N = float((dx * log2N).sum() / denom)
            Ycols = ((tau_w[nz] - log2N[None, :]) / q_arr[nz, None]).T
            try:
                f = focus_regression(x, Ycols, x0_min=float(log2_L),
                                     generalized=(fit_mode == "focus"))
            except ValueError as exc:
                focus_info.update(branch="unavailable", reason=str(exc))
            else:
                tau_f = np.array(out["tau"], dtype=np.float64)
                tau_f[nz] = q_arr[nz] * f["h_q"] + s_N
                tau_f[~nz] = s_N
                out["tau"] = tau_f
                focus_info.update(branch=f["branch"], x0=f["x0"], y0=f["y0"],
                                  sse=f["sse"], sse_naive=f["sse_naive"],
                                  mse_q=f["mse_q"], h_q_mean=f["h_q"],
                                  h_q_mean_naive=f["h_q_naive"], s_N=s_N)
        out["focus"] = focus_info

    if frame == "original" and eta != 0.0:
        out["tau"] = out["tau"] - eta * q_arr
        out["h"] = out["h"] - eta
        # D unchanged at fixed q; D(h) shifts horizontally because h shifted.
    out["eta"] = eta
    out["frame"] = frame
    return out


def legendre_dh(q_list, tau, n_h: int = 257) -> dict:
    """Legendre-Fenchel transform of a FITTED tau(q): ``D(h) = inf_q (q h - tau(q))``.

    **Hull-only, by theorem** (Jaffard 1994/1997; Brown-Michon-Peyriere 1992; Touchette-Beck
    2006 Thm 2-4): the transform returns the CONCAVE HULL of the true spectrum -- a kink in
    tau(q) comes back as a straight chord, indistinguishable from a genuinely affine D(h).
    This is a DIAGNOSTIC overlay to
    compare against the canonical parametric (h(q), D(q)) construction -- the Arneodo route
    never differentiates a fitted tau; it fits the three partition-function families Z/H/D and
    reads D(h) parametrically. Disagreement between the two IS
    the kink/phase-transition tell; the Touchette-Beck generalized (Gaussian) ensemble is the
    designed follow-up that would resolve which case holds.

    The h grid spans the finite samples' secant-slope range (the transform is undetermined
    outside it); a degenerate range (monofractal: tau affine in q) collapses to the single
    point. Returns ``{"h", "D"}``.
    """
    q = np.asarray(q_list, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    finite = np.isfinite(q) & np.isfinite(t)
    q, t = q[finite], t[finite]
    order = np.argsort(q)
    q, t = q[order], t[order]
    if q.size < 2:
        return {"h": np.zeros(0), "D": np.zeros(0)}
    slopes = np.diff(t) / np.diff(q)
    h_lo, h_hi = float(slopes.min()), float(slopes.max())
    if h_hi - h_lo < 1e-12:
        h = np.array([0.5 * (h_lo + h_hi)])
        return {"h": h, "D": np.array([float((q * h - t).min())])}
    h = np.linspace(h_lo, h_hi, int(n_h))
    D = (q[None, :] * h[:, None] - t[None, :]).min(axis=1)
    return {"h": h, "D": D}


def _lnq_side_fit(lnq: np.ndarray, tau: np.ndarray):
    """OLS of tau against ln q for one side; ``(slope, intercept, r2)`` or NaNs below 2 pts."""
    if lnq.size < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(lnq, tau, 1)
    resid = tau - (slope * lnq + intercept)
    ss_tot = float(((tau - tau.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), r2


def phase_transition_fit(q_list, tau, q_break: float, *, q_min: float = 0.01,
                         min_pts: int = 3) -> dict:
    """Two-segment fit of tau(q) against **ln q** split at ``q_break`` -- the EBSD workbook's
    cell-49 phase-transition surface: in the constant-specific-heat approximation tau(q) is
    linear in ln(q) within each phase, so a slope break at q* signals a phase transition
    (freezing of singularities). Positive branch only (``q > q_min``); a side with fewer
    than ``min_pts`` finite points comes back NaN (and so does ``ds``).

    Returns ``{slope_L, slope_R, ds, r2_L, r2_R, n_L, n_R}`` with ``ds = slope_L - slope_R``.
    """
    q = np.asarray(q_list, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    keep = np.isfinite(q) & np.isfinite(t) & (q > q_min)
    q, t = q[keep], t[keep]
    lnq = np.log(q)
    left = q <= q_break
    n_L, n_R = int(left.sum()), int((~left).sum())
    slope_L, _, r2_L = _lnq_side_fit(lnq[left], t[left]) if n_L >= min_pts \
        else (np.nan, np.nan, np.nan)
    slope_R, _, r2_R = _lnq_side_fit(lnq[~left], t[~left]) if n_R >= min_pts \
        else (np.nan, np.nan, np.nan)
    return {"slope_L": slope_L, "slope_R": slope_R, "ds": slope_L - slope_R,
            "r2_L": r2_L, "r2_R": r2_R, "n_L": n_L, "n_R": n_R}


def phase_transition_scan(q_list, tau, *, q_min: float = 0.01, min_pts: int = 3,
                          n_candidates: int = 100) -> dict:
    """Count-weighted mean-R² scan over candidate break points (the cell-49 auto q* marker):
    for each candidate, ``r2w = (n_L r2_L + n_R r2_R) / (n_L + n_R)``; the best candidate is
    the argmax. Candidates span the positive branch's interior so both sides can hold
    ``min_pts``. Returns ``{"q_grid", "r2w", "best_q"}`` (``best_q`` NaN when nothing fits).
    """
    q = np.asarray(q_list, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    keep = np.isfinite(q) & np.isfinite(t) & (q > q_min)
    q_pos = np.sort(q[keep])
    if q_pos.size < 2 * min_pts:
        return {"q_grid": np.zeros(0), "r2w": np.zeros(0), "best_q": np.nan}
    grid = np.linspace(q_pos[min_pts - 1], q_pos[-min_pts], int(n_candidates))
    r2w = np.full(grid.shape, np.nan)
    for i, qb in enumerate(grid):
        f = phase_transition_fit(q, t, float(qb), q_min=q_min, min_pts=min_pts)
        if np.isfinite(f["r2_L"]) and np.isfinite(f["r2_R"]):
            r2w[i] = (f["n_L"] * f["r2_L"] + f["n_R"] * f["r2_R"]) / (f["n_L"] + f["n_R"])
    best = np.nan if not np.isfinite(r2w).any() else float(grid[np.nanargmax(r2w)])
    return {"q_grid": grid, "r2w": r2w, "best_q": best}


def negative_branch_convexity_loss(q_list, tau, *, rel_tol: float = 1e-3):
    """The Pont/Turiel/Perez-Vicente 2006 truncation indicator: experimental tau(q) loses its
    required curvature sign at large NEGATIVE moments (the right-tail regime), and the curve
    must be truncated at the inflection. In this codebase's convention tau is CONCAVE; walking
    the negative branch outward from q = 0, the first point whose discrete second difference
    turns positive (beyond ``rel_tol`` of the largest curvature magnitude -- float dust never
    flags) is returned as the loss abscissa; ``None`` when the branch stays concave. An
    INDICATOR for display, never an automatic truncation."""
    q = np.asarray(q_list, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    finite = np.isfinite(q) & np.isfinite(t)
    q, t = q[finite], t[finite]
    order = np.argsort(q)
    q, t = q[order], t[order]
    if q.size < 3:
        return None
    d2 = np.diff(t, 2)                          # second difference at interior points q[1:-1]
    q_mid = q[1:-1]
    tol = rel_tol * float(np.abs(d2).max()) if d2.size else 0.0
    neg = q_mid < 0
    bad = neg & (d2 > tol)
    if not bad.any():
        return None
    return float(q_mid[bad].max())              # the loss CLOSEST to q = 0 on the way out


def _all_nan_result(q_arr: np.ndarray, eta: float, frame: str) -> dict:
    """Full key set, every spectrum value NaN -- a window too narrow (or too degenerate) to fit."""
    nan = np.full(q_arr.shape, np.nan)
    return {"q_list": q_arr, "tau": nan.copy(), "h": nan.copy(), "D": nan.copy(),
            "tau_err": nan.copy(), "h_err": nan.copy(), "D_err": nan.copy(),
            "eta": eta, "frame": frame}


def _masked_slopes(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-row OLS slope of Y against x, using each row's finite entries only."""
    V = np.isfinite(Y)
    n = V.sum(axis=1)
    Y0 = np.where(V, Y, 0.0)
    Xv = np.where(V, x[None, :], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xbar = Xv.sum(axis=1) / n
        dx = np.where(V, x[None, :] - xbar[:, None], 0.0)
        sxx = (dx * dx).sum(axis=1)
        sxy = (dx * Y0).sum(axis=1)          # sum(dx*(y-ybar)) == sum(dx*y): sum(dx)=0 per row
        slopes = sxy / sxx
    return np.where((n >= 3) & (sxx > 0), slopes, np.nan)


def _propagated_slope_sigma(dx: np.ndarray, denom: float, S: np.ndarray) -> np.ndarray:
    """Per-row 1-sigma on the OLS slope from per-scale sigmas S (finite, positive entries only).

    ``dx``/``denom`` come from the full window, not each row's valid subset -- the reference
    convention (the slope itself IS fit on the full-window x for its valid points, and the error
    bar deliberately uses the same abscissa geometry).
    """
    V = np.isfinite(S) & (S > 0)
    n = V.sum(axis=1)
    S0 = np.where(V, S, 0.0)
    var = (dx[None, :] ** 2 * S0 * S0).sum(axis=1) / (denom * denom)
    with np.errstate(invalid="ignore"):
        err = np.sqrt(np.maximum(var, 0.0))
    return np.where(n >= 3, err, np.nan)


# ---------------------------------------------------------------------------------------------
# Focus regression: Schadner, "Focus regression for multifractal analysis",
# Chaos Solitons Fractals 209:118368 (2026) -- generalized focus with non-iterative closed
# forms (his eqs. 5/7/9) -- and Mukli-Nagy-Eke, Physica A 417:150 (2015) -- the fixed-focus
# origin, whose Eq. (7) (one window at the signal length => S(q, L) is q-independent BY
# CONSTRUCTION) is the anchor rationale. Everything reduces to the naive per-q OLS slopes
# b_q, the per-q column means ybar_q, and sigma_X^2 (POPULATION,
# ddof=0 -- Schadner's eq. 5 is only exact then). The quadratic's two roots have product
# -sigma_X^2 exactly (Vieta -- the stable second root needs NO subtraction), its
# discriminant is computed in the additive non-negative form, and the winner among {feasible
# interior root, fixed-focus endpoint x0 = x0_min, monofractal endpoint x0 = inf} is chosen
# by SSE ARGMIN -- root sign classification alone picks the wrong extremum in ~4% of cases
# (measured over 6000 synthetic fits) while the argmin was never wrong. The x0 = inf
# endpoint is the identical-slopes / free-intercepts fit whose common slope equals the mean
# of the naive slopes (= the ANCOVA pooled slope, since all columns share X). Monotone-h_q
# enforcement (the formalism's validity condition) uses the exact threshold
# |xi| >= sigma_X^2 * max(db/dy); the guarantee otherwise rides on the
# power-mean ordering of the columns, which is exactly why callers must normalize the
# partition sums by N_a before fitting (log2 Z -> log2 Z - log2 N_a).


def focus_regression(X, Y, *, x0_min: float, weights=None,
                     enforce_monotone: bool = True,
                     generalized: bool = True) -> dict:
    """Joint fit of all moment-wise scaling functions through one focus.

    Parameters
    ----------
    X : (n_t,) log-scales (any log base, shared by every column).
    Y : (n_t, n_q) log moment-scaling functions -- POWER MEANS per column (for WTMM:
        ``(log2 Z(q, a) - log2 N_a) / q``, q = 0 excluded), same log base as ``X``.
    x0_min : the anchor -- log of the signal length / field extent (Mukli Eq. 7), NOT the
        top of the fit window: the focus is an extrapolation target at or beyond it.
    weights : optional (n_q,) pre-specified moment weights (e.g. 1/sigma_q^2). Must not be
        derived from this fit's own residuals (that would be IRLS -- iteration again).
    enforce_monotone : assert the formalism's h_q ordering, pushing |xi| out to the exact
        threshold (or the monofractal endpoint) when violated.

    Returns dict: ``h_q`` (the fitted slopes), ``h_q_naive``, ``x0``, ``y0``, ``sse``,
    ``sse_naive``, ``mse_q`` (per-moment error curve -- the Mukli adequacy diagnostic),
    ``branch`` in {"interior", "fixed", "mono"}.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 1 or Y.ndim != 2 or Y.shape[0] != X.size:
        raise ValueError(f"X must be (n_t,), Y (n_t, n_q); got {X.shape} and {Y.shape}")
    if X.size < 5:
        raise ValueError(f"focus regression needs >= 5 scales; got {X.size}")
    if Y.shape[1] < 3:
        raise ValueError(f"focus regression needs >= 3 moments; got {Y.shape[1]}")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        raise ValueError("X and Y must be finite everywhere -- one NaN cell poisons every "
                         "coefficient of a pooled fit (drop whole scales or whole q columns)")
    w = np.ones(Y.shape[1]) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (Y.shape[1],) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("weights must be positive, finite, one per moment column")

    n_t = X.size
    xb = float(X.mean())
    s2 = float(X.var())                          # POPULATION variance (ddof=0), load-bearing
    if s2 <= 0:
        raise ValueError("degenerate scales: sigma_X^2 == 0")
    Xc = X - xb
    b = (Xc @ Y) / float(Xc @ Xc)                # naive per-q OLS slopes
    yq = Y.mean(axis=0)                          # per-q column means
    sw = float(w.sum())
    bbar = float((w * b).sum() / sw)
    ybar = float((w * yq).sum() / sw)
    sse_naive = float((w[None, :] * (Y - (yq[None, :] + Xc[:, None] * b[None, :])) ** 2).sum())

    # weighted population moments over q
    B = float((w * (b - bbar) ** 2).sum() / sw)
    Psi = float((w * (yq - ybar) ** 2).sum() / sw)
    C = float((w * (b - bbar) * (yq - ybar)).sum() / sw)

    def beta_of(xi):
        if np.isinf(xi):
            return np.full_like(b, bbar)
        return (s2 * b + xi * (yq - ybar) + xi * xi * bbar) / (s2 + xi * xi)

    def fit_at(xi):
        bet = beta_of(xi)
        if np.isinf(xi):
            # identical slopes, per-q intercepts pinned at the column means
            resid_sq = (w * (s2 * (b - bet) ** 2)).sum()
            sse = sse_naive + n_t * float(resid_sq)
            mse_q = (w * ((Y - (yq[None, :] + Xc[:, None] * bet[None, :])) ** 2)
                     ).sum(axis=0) / n_t
            return {"xi": xi, "h_q": bet, "y0": np.nan, "sse": float(sse),
                    "mse_q": np.asarray(mse_q)}
        y0 = ybar - xi * bbar
        excess = (w * (s2 * (b - bet) ** 2 + (yq - y0 - xi * bet) ** 2)).sum()
        sse = sse_naive + n_t * float(excess)
        x0 = xb - xi
        mse_q = (w * ((Y - (y0 + (X[:, None] - x0) * bet[None, :])) ** 2)).sum(axis=0) / n_t
        return {"xi": xi, "h_q": bet, "y0": float(y0), "sse": float(sse),
                "mse_q": np.asarray(mse_q)}

    xi_ff = xb - float(x0_min)                   # feasible ray: xi <= xi_ff (< 0)
    candidates = {"fixed": fit_at(xi_ff), "mono": fit_at(-np.inf)}
    scale_ref = max(abs(s2 * B - Psi), s2 * abs(C), 1e-300)
    # generalized=False is Mukli's fixed-focus variant: the interior root is never a
    # candidate -- the focus stays pinned at the anchor (or recedes to the mono endpoint).
    if generalized and abs(C) > 1e-14 * scale_ref:
        bc = s2 * B - Psi
        delta = bc * bc + 4.0 * s2 * C * C       # additive form: structurally >= 0
        sq = np.sqrt(delta)
        xi_big = -(bc + (np.copysign(1.0, bc) if bc != 0 else 1.0) * sq) / (2.0 * C)
        for xi in (xi_big, -s2 / xi_big if xi_big != 0 else np.inf):   # Vieta, no subtraction
            if np.isfinite(xi) and xi <= xi_ff:
                candidates["interior"] = fit_at(float(xi))
                break

    def pick(cands):
        name = min(cands, key=lambda k: cands[k]["sse"])
        return name, cands[name]

    branch, best = pick(candidates)

    if enforce_monotone and np.any(np.diff(best["h_q"]) > 1e-12):
        db = np.diff(b)
        dy = np.diff(yq)
        rising = db > 0
        ok = {}
        if rising.any() and np.all(dy[rising] > 0):
            xi_need = -s2 * float(np.max(db[rising] / dy[rising])) * (1.0 + 1e-12)
            if xi_need <= xi_ff:
                ok["interior"] = fit_at(xi_need)
            else:
                ok["fixed"] = fit_at(xi_ff) if xi_ff <= xi_need else fit_at(xi_need)
        ok["mono"] = candidates["mono"]
        ok = {k: v for k, v in ok.items()
              if not np.any(np.diff(v["h_q"]) > 1e-12)}
        branch, best = pick(ok or {"mono": candidates["mono"]})

    return {
        "h_q": np.asarray(best["h_q"]), "h_q_naive": np.asarray(b),
        "x0": float(xb - best["xi"]) if np.isfinite(best["xi"]) else np.inf,
        "y0": best["y0"], "sse": best["sse"], "sse_naive": sse_naive,
        "mse_q": best["mse_q"], "branch": branch,
    }
