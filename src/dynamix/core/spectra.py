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
        # tau(0) = slope(log2 N_a) exactly. A scale inside the window where every chain has
        # died (N_a = 0, tau not finite) is left out of the joint fits, as a per-q fit
        # leaves it out of its row.
        nz = q_arr != 0.0
        N_a = np.asarray(hd["N_a"], dtype=np.float64)[mask]
        tau_w = np.asarray(hd["tau_qa"], dtype=np.float64)[:, mask]
        alive = (N_a > 0) & np.all(np.isfinite(tau_w[nz]), axis=0)
        xa = x[alive]
        focus_info: dict = {"fit_mode": fit_mode}
        if xa.size < 5 or int(nz.sum()) < 3:
            focus_info.update(branch="unavailable",
                              reason="needs >= 5 live scales (N_a > 0, finite tau) and >= 3 nonzero q")
        else:
            dxa = xa - xa.mean()
            log2N = np.log2(N_a[alive])
            s_N = float((dxa * log2N).sum() / float((dxa * dxa).sum()))
            Ycols = ((tau_w[nz][:, alive] - log2N[None, :]) / q_arr[nz, None]).T
            try:
                f = focus_regression(xa, Ycols, x0_min=float(log2_L),
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
                if fit_mode == "focus" and f["branch"] == "fixed":
                    focus_info["note"] = ("the free focus would sit inside the data: pinned at "
                                          "the field extent (the fixed-focus fit)")
                elif fit_mode == "focus" and f["branch"] == "mono":
                    focus_info["note"] = "no finite focus: identical slopes (the monofractal limit)"
        # The canonical channels have a focus by the same argument: at a = L one maximum
        # carries the whole Boltzmann weight, so h(q, L) = log2|T(L)| for every q and
        # D(q, L) = 0. Each family is fitted through its own focus; D(q) peaks at q = 0, so
        # no ordering is enforced on it. The rows left out of a joint fit (not finite across
        # the window) keep their per-q slopes.
        for key, name, monotone in (("h_qa", "h", True), ("D_qa", "D", False)):
            fam = _family_focus(xa, np.asarray(hd[key], dtype=np.float64)[:, mask][:, alive],
                                float(log2_L), generalized=(fit_mode == "focus"),
                                monotone=monotone)
            if fam is None:
                focus_info[name] = {"branch": "unavailable"}
                continue
            slopes = np.array(out[name], dtype=np.float64)
            slopes[fam["cols"]] = fam["h_q"]
            out[name] = slopes
            focus_info[name] = {k: fam[k] for k in ("branch", "x0", "y0", "sse", "sse_naive")}
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


def legendre_local(q_list, tau, *, m: int = 2, order: int = 3) -> dict:
    """xsmurf's ``legendre2`` (Decoster 1999, ``tcl_library/legendre.tcl``): at every q with
    ``m`` neighbours on each side, a polynomial of ``order`` through the 2m + 1 points around
    it, h(q) the derivative of that local fit at q and τ_fit(q) its value, D = q h − τ_fit. A
    moving window, never a global smooth, so a kink survives except within m steps of it:
    the result sits between the hull (:func:`legendre_dh`, the concave envelope by theorem)
    and the canonical parametric route, which never differentiates τ. xsmurf weights the
    points by 0.005|τ| + 1e−5, near-uniform, and differentiates by a 0.001 step; here the
    weights are uniform and the derivative is the fit's own. Returns ``{"q", "h", "D",
    "tau_fit"}`` over the interior q, empty below 2m + 1 finite points."""
    q = np.asarray(q_list, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    finite = np.isfinite(q) & np.isfinite(t)
    q, t = q[finite], t[finite]
    by_q = np.argsort(q)
    q, t = q[by_q], t[by_q]
    n = q.size
    if n < 2 * m + 1:
        empty = np.zeros(0)
        return {"q": empty, "h": empty, "D": empty, "tau_fit": empty}
    qs, hs, taus = [], [], []
    for i in range(m, n - m):
        window = slice(i - m, i + m + 1)
        poly = np.poly1d(np.polyfit(q[window] - q[i], t[window], min(order, 2 * m)))
        qs.append(float(q[i]))
        hs.append(float(poly.deriv()(0.0)))
        taus.append(float(poly(0.0)))
    qs, hs, taus = (np.asarray(v, dtype=np.float64) for v in (qs, hs, taus))
    return {"q": qs, "h": hs, "D": qs * hs - taus, "tau_fit": taus}


def kink_interval(q_list, tau, *, q_c: "float | None" = None) -> "dict | None":
    """The nonconcave interval off a kink in τ(q) (Touchette & Beck 2006, Figs. 2 and 3): the
    two one-sided derivatives at q_c, h_h = τ′(q_c⁻) on the left and h_l = τ′(q_c⁺) on the
    right, are the ends of the open interval (h_l, h_h) on which the true spectrum lies below
    its concave envelope D**(h), the envelope's straight chord of slope q_c across it. The kink
    is the largest of :func:`dynamix.core.nonconcave.kink_flags` unless ``q_c`` names one;
    ``None`` without a kink. Returns ``{"q_c", "tau_c", "h_l", "h_h"}``, τ_c interpolated at q_c."""
    from dynamix.core.nonconcave import kink_flags
    q = np.asarray(q_list, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    finite = np.isfinite(q) & np.isfinite(t)
    q, t = q[finite], t[finite]
    order = np.argsort(q)
    q, t = q[order], t[order]
    flags = kink_flags(q, t)
    if not flags:
        return None
    if q_c is None:
        flag = max(flags, key=lambda f: f["jump"])
    else:
        flag = min(flags, key=lambda f: abs(f["q"] - float(q_c)))
    at = float(flag["q"])
    return {"q_c": at, "tau_c": float(np.interp(at, q, t)), "h_l": float(flag["h_lo"]), "h_h": float(flag["h_hi"])}


def envelope_departure(h_env, D_env, h_gen, D_gen, *, tol: float = 0.02) -> "tuple | None":
    """Where a generalized spectrum lies below the concave envelope: the extremes of the h at
    which ``D_gen`` falls more than ``tol`` under the envelope read at those h, the envelope's
    (h_l, h_h) by the other route (:func:`kink_interval` reads it off τ); ``None`` where it
    never does. The envelope is interpolated on the generalized grid inside its own range."""
    h_e = np.asarray(h_env, dtype=np.float64)
    d_e = np.asarray(D_env, dtype=np.float64)
    ok = np.isfinite(h_e) & np.isfinite(d_e)
    h_e, d_e = h_e[ok], d_e[ok]
    order = np.argsort(h_e)
    h_e, d_e = h_e[order], d_e[order]
    h_g = np.asarray(h_gen, dtype=np.float64)
    d_g = np.asarray(D_gen, dtype=np.float64)
    if h_e.size < 2 or h_g.size == 0:
        return None
    inside = np.isfinite(h_g) & np.isfinite(d_g) & (h_g >= h_e.min()) & (h_g <= h_e.max())
    env_at = np.interp(h_g[inside], h_e, d_e)
    below = d_g[inside] < env_at - float(tol)
    if not below.any():
        return None
    where = h_g[inside][below]
    return float(where.min()), float(where.max())


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


def phase_transition_tangent(q_list, tau, h, D, *, q_pick: "float | None" = None):
    """The tangent to D(h) at the phase transition: the point ``(h(q_c), D(q_c))`` of the
    fitted spectrum at the negative-branch convexity-loss abscissa ``q_c``
    (:func:`negative_branch_convexity_loss`), with slope ``q_c`` because ``dD/dh = q`` under the
    Legendre transform. Its ``intercept`` is where it meets the y axis at ``h = 0``,
    ``D(q_c) - q_c h(q_c)``, which is ``-tau(q_c)`` for a Legendre-consistent spectrum. ``None``
    when the branch stays concave or the point is not finite. ``q_pick``: the user's q in place
    of the convexity-loss one (the fitted q nearest to it); ``"picked"`` says which."""
    q = np.asarray(q_list, dtype=np.float64)
    if q_pick is not None and np.isfinite(q).any():
        q_c = float(q[np.nanargmin(np.abs(q - float(q_pick)))])
    else:
        q_c = negative_branch_convexity_loss(q_list, tau)
    if q_c is None:
        return None
    i = int(np.argmin(np.abs(q - q_c)))
    h_c, D_c = float(np.asarray(h, dtype=np.float64)[i]), float(np.asarray(D, dtype=np.float64)[i])
    if not (np.isfinite(h_c) and np.isfinite(D_c)):
        return None
    return {"q_c": float(q_c), "h_c": h_c, "D_c": D_c, "intercept": D_c - float(q_c) * h_c,
            "picked": q_pick is not None}


def _family_focus(x, rows, x0_min: float, *, generalized: bool, monotone: bool):
    """:func:`focus_regression` over one family's rows (one per q) that are finite across the
    window; ``{"h_q", "cols", "x0", "y0", "branch", "sse", "sse_naive"}`` or None when fewer
    than 5 scales or 3 such rows are there."""
    rows = np.asarray(rows, dtype=np.float64)
    cols = np.flatnonzero(np.all(np.isfinite(rows), axis=1))
    if x.size < 5 or cols.size < 3:
        return None
    try:
        f = focus_regression(x, rows[cols].T, x0_min=float(x0_min), generalized=generalized,
                             enforce_monotone=monotone)
    except ValueError:
        return None
    return {"h_q": f["h_q"], "cols": cols, "x0": f["x0"], "y0": f["y0"], "branch": f["branch"],
            "sse": f["sse"], "sse_naive": f["sse_naive"]}


def partition_fit_lines(hd: dict, log2_a_min: float, log2_a_max: float, *,
                        fit_mode: str = "naive", log2_L: "float | None" = None) -> dict:
    """The lines a fitter draws over the partition-function families, in the tables' own
    (integrated) frame: per family, ``slope`` and ``intercept`` per q in log2 a. Naive: the
    per-q OLS line through the row's window mean. A focus branch (interior or fixed): the
    family's lines meet at its focus (x0, y0) -- the Z family's in power-mean space
    Y_q = (log2 Z - log2 N_a)/q (Mukli Eq. 17: log S(q, s) = H(q)(log s - log L) + log S(L)),
    carried back to log2 Z with the window's log2 N_a line. Also ``power_mean`` (the Z
    family's q != 0 rows over every scale and their lines), ``focus`` (the Z family's, or
    None), ``branch`` (the Z family's, or None in naive mode) and ``family_focus`` (per family,
    None without one)."""
    fit = fit_spectra(hd, log2_a_min, log2_a_max, frame="integrated", fit_mode=fit_mode,
                      log2_L=log2_L)
    q = np.asarray(hd["q_list"], dtype=np.float64)
    log2_s = np.asarray(hd["log2_scales"], dtype=np.float64)
    mask = (log2_s >= log2_a_min) & (log2_s <= log2_a_max)
    x = log2_s[mask]
    out = {"q_list": q, "lines": {}, "focus": None, "branch": None,
           "family_focus": {"tau_qa": None, "h_qa": None, "D_qa": None}}

    def through_means(slope, table):
        Y = np.asarray(table, dtype=np.float64)[:, mask]
        fin = np.isfinite(Y)
        n = fin.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ybar = np.where(fin, Y, 0.0).sum(axis=1) / n
            xbar = (fin * x[None, :]).sum(axis=1) / n
        return {"slope": np.asarray(slope, dtype=np.float64), "intercept": ybar - slope * xbar}

    for key, name in (("tau_qa", "tau"), ("h_qa", "h"), ("D_qa", "D")):
        out["lines"][key] = through_means(fit[name], hd[key])

    N_a = np.asarray(hd["N_a"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log2N = np.where(N_a > 0, np.log2(np.where(N_a > 0, N_a, 1.0)), np.nan)
    nz = q != 0.0
    pm_table = (np.asarray(hd["tau_qa"], dtype=np.float64)[nz] - log2N[None, :]) / q[nz, None]
    pm = {"q": q[nz], "table": pm_table}
    focus = fit.get("focus") or {}
    if focus.get("branch") in ("interior", "fixed"):
        x0, y0, h, s_N = focus["x0"], focus["y0"], np.asarray(focus["h_q_mean"]), focus["s_N"]
        nbar, xbar = float(np.mean(log2N[mask])), float(x.mean())
        pm["slope"], pm["intercept"] = h, y0 - h * x0
        z = out["lines"]["tau_qa"]
        z["slope"][nz] = q[nz] * h + s_N
        z["intercept"][nz] = q[nz] * (y0 - h * x0) + nbar - s_N * xbar
        out["focus"] = out["family_focus"]["tau_qa"] = (float(x0), float(y0))
    else:
        naive = through_means(_masked_slopes(x, pm_table[:, mask]), pm_table)
        pm["slope"], pm["intercept"] = naive["slope"], naive["intercept"]
    out["branch"] = focus.get("branch")
    out["power_mean"] = pm
    for key, name in (("h_qa", "h"), ("D_qa", "D")):
        fam = focus.get(name) or {}
        if fam.get("branch") in ("interior", "fixed"):
            x0, y0 = float(fam["x0"]), float(fam["y0"])
            slope = out["lines"][key]["slope"]
            out["lines"][key]["intercept"] = y0 - slope * x0
            out["family_focus"][key] = (x0, y0)
    return out


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


# ----------------------------------------------------------------------------- specific heat

def specific_heat(q_list, tau) -> np.ndarray:
    """The specific heat c(q) = -q² d²τ/dq² (Beck & Schlögl 1993 eq. 3; Klamut et al. 2020
    eq. 14) by second-order finite differences over the finite points of τ(q); a row whose τ is
    not finite comes back NaN. A log-normal cascade (τ quadratic in q) gives c₂ q² exactly."""
    q = np.asarray(q_list, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    out = np.full(q.shape, np.nan)
    ok = np.isfinite(q) & np.isfinite(t)
    if ok.sum() < 3:
        return out
    qf = q[ok]
    d2 = np.gradient(np.gradient(t[ok], qf, edge_order=2), qf, edge_order=2)
    out[ok] = -qf ** 2 * d2
    return out


def cumulant_heat(cumulants: dict, q_list) -> np.ndarray:
    """c(q) from a cascade's cumulants: the expansion τ(q) = c₁q - c₂q²/2 + c₃q³/6 - c₄q⁴/24
    has τ'' = -c₂ + c₃q - c₄q²/2, so c(q) = -q² τ''. A missing order counts as zero; a NaN one
    (no fit) makes the whole curve NaN."""
    q = np.asarray(q_list, dtype=np.float64)
    c2, c3, c4 = (float(cumulants.get(k) if cumulants.get(k) is not None else 0.0)
                  for k in ("c2", "c3", "c4"))
    return -q ** 2 * (-c2 + c3 * q - c4 * q ** 2 / 2.0)


def cascade_cumulants(mods_per_scale, scales_px, log2_a_min: float, log2_a_max: float, *,
                      max_order: int = 4) -> dict:
    """The cascade's cumulants c₁..c₄ from the per-scale moduli of the maxima: the n-th cumulant
    of ln|T| regressed on ln a over the scales inside the fit window (log2 a, the fitter's
    window), through the vendored ``wtmm.dupont_venugopal.fit_venugopal_cumulants`` (whose cₙ is
    the raw slope), with the cascade's alternating signs: c₁ = +s₁ (the mean of ln|T| grows with
    ln a), c₂ = -s₂ (the variance grows towards SMALL scales), c₃ = +s₃, c₄ = -s₄. Returns
    ``{"c1".."c4", "r2"}`` (the worst per-order R²); NaN when fewer than two scales fall inside
    the window."""
    from dynamix.core import wtmm_backend

    if not wtmm_backend.USE_VENDORED:
        return cumulants_from_moments(ln_moments(mods_per_scale), scales_px, log2_a_min,
                                      log2_a_max, max_order=max_order)
    from dynamix._vendor.wtmm.dupont_venugopal import fit_venugopal_cumulants

    ln_a = np.log(np.asarray(scales_px, dtype=np.float64))
    samples = {}
    for i, mods in enumerate(mods_per_scale):
        m = np.asarray(mods, dtype=np.float64)
        samples[i] = np.log(m[np.isfinite(m) & (m > 0)])
    ln2 = np.log(2.0)
    fit = fit_venugopal_cumulants(samples, ln_a, max_order=max_order,
                                  scale_range=(float(log2_a_min) * ln2, float(log2_a_max) * ln2))
    signs = (1.0, -1.0, 1.0, -1.0)
    out = {f"c{n}": signs[n - 1] * float(fit[f"c{n}"]) for n in range(1, max_order + 1)}
    out["r2"] = float(fit["r2"])
    return out


def ln_moments(mods_per_scale) -> np.ndarray:
    """Per scale, the mean, variance, third and fourth cumulant of ln|T| over the positive
    finite moduli -- the moments the cascade's cumulants are regressed from, small enough to
    travel in a partition table: ``(4, n_scales)``."""
    out = np.full((4, len(mods_per_scale)), np.nan)
    for k, mods in enumerate(mods_per_scale):
        m = np.asarray(mods, dtype=np.float64)
        x = np.log(m[np.isfinite(m) & (m > 0)])
        if x.size == 0:
            continue
        mean = x.mean()
        d = x - mean
        var = float((d * d).mean())
        out[:, k] = (mean, var, float((d ** 3).mean()), float((d ** 4).mean()) - 3.0 * var * var)
    return out


def cumulants_from_moments(moments, scales_px, log2_a_min: float, log2_a_max: float, *,
                           max_order: int = 4) -> dict:
    """:func:`cascade_cumulants` from per-scale moments of ln|T| (:func:`ln_moments`, or a
    moment table's ``lnT_moments``): the n-th moment regressed on ln a over the scales inside
    the window, with the cascade's alternating signs; ``{"c1".."c4", "r2"}`` (worst R²), NaN
    below two scales."""
    moments = np.asarray(moments, dtype=np.float64)[:max_order]
    ln_a = np.log(np.asarray(scales_px, dtype=np.float64))
    ln2 = np.log(2.0)
    lo, hi = sorted((float(log2_a_min) * ln2, float(log2_a_max) * ln2))
    valid = np.all(np.isfinite(moments), axis=0) & (ln_a >= lo) & (ln_a <= hi)
    out = {f"c{n}": float("nan") for n in range(1, max_order + 1)}
    out["r2"] = float("nan")
    if valid.sum() < 2:
        return out
    signs = (1.0, -1.0, 1.0, -1.0)
    r2_min = 1.0
    for n in range(max_order):
        slope, intercept = np.polyfit(ln_a[valid], moments[n, valid], 1)
        out[f"c{n + 1}"] = signs[n] * float(slope)
        y = moments[n, valid]
        pred = slope * ln_a[valid] + intercept
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2_min = min(r2_min, 1.0 - float(((y - pred) ** 2).sum()) / max(ss_tot, 1e-30))
    out["r2"] = float(r2_min)
    return out


def cumulant_regression(moments, scales_px, log2_a_min: float, log2_a_max: float, order: int) -> dict:
    """One log-cumulant on its own range: C_order(a) regressed on ln a over the scales inside the
    window, with the cascade sign of :func:`cumulants_from_moments` (c₁ = +slope, c₂ = −slope,
    c₃ = +slope, c₄ = −slope): ``c``, its standard error ``sigma`` from the residuals (n − 2
    degrees of freedom; NaN below three scales), ``r2``, ``n`` (the scales fitted) and the raw
    ``slope`` and ``intercept`` of the line against ln a. NaN throughout below two scales."""
    C = np.asarray(moments, dtype=np.float64)[order - 1]
    ln_a = np.log(np.asarray(scales_px, dtype=np.float64))
    ln2 = np.log(2.0)
    lo, hi = sorted((float(log2_a_min) * ln2, float(log2_a_max) * ln2))
    valid = np.isfinite(C) & (ln_a >= lo) & (ln_a <= hi)
    n = int(valid.sum())
    out = {"c": np.nan, "sigma": np.nan, "r2": np.nan, "n": n, "slope": np.nan, "intercept": np.nan}
    if n < 2:
        return out
    x, y = ln_a[valid], C[valid]
    slope, intercept = np.polyfit(x, y, 1)
    res = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    out.update(c=(1.0, -1.0, 1.0, -1.0)[order - 1] * float(slope), slope=float(slope),
               intercept=float(intercept), r2=1.0 - float((res ** 2).sum()) / max(ss_tot, 1e-30))
    if n >= 3:
        out["sigma"] = float(np.sqrt((res ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum()))
    return out


def cumulant_tau(cumulants: dict, q_list, *, c0: float = 0.0) -> np.ndarray:
    """tau(q) from the cascade's cumulants, tau = -c0 + c1 q - c2 q²/2 + c3 q³/6 - c4 q⁴/24
    (the workbook's reconstruction), with c0 the support's dimension (-tau(0))."""
    q = np.asarray(q_list, dtype=np.float64)
    c1, c2, c3, c4 = (float(cumulants.get(k) if cumulants.get(k) is not None else 0.0)
                      for k in ("c1", "c2", "c3", "c4"))
    return -float(c0) + c1 * q - c2 * q ** 2 / 2.0 + c3 * q ** 3 / 6.0 - c4 * q ** 4 / 24.0
