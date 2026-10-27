# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The directional thermodynamic formalism of Ben Slimane, Ben Abid, Ben Omrane and Halouani
(Symmetry 11:825, 2019) on the hyperbolic (fully separable) wavelet coefficients of
:func:`dynamix.core.dwt.hyperbolic`.

Three estimators read the same moment tables. The default, ``level_line``, places every pair at
Abry, Clausel, Jaffard, Roux and Vedel's anisotropic level m(j₁, j₂) = max(j₁/α₁, j₂/α₂) (Rev.
Mat. Iberoam. 31, 2015, Prop. 4.7 (3)): the same liminf as Γ_j's, since every pair of Γ_j has
j − 2 ≤ m ≤ j and the Γ_j cover every pair, read without Γ_j's floors and ±1 margins, which make
a finite regression a staircase. ``gamma`` is Γ_j read literally (eqs. 44–47, the union
envelope); ``legacy`` the level-family approximation. Finite regressions do not establish the
asymptotic bounds.

For an anisotropy α = (α₁, α₂), α_i > 0, α₁ + α₂ = 2, the anisotropic scaling function
η_α(p) is read along the index sets Γ_j(α) of pairs (j₁, j₂) of detail levels around
([jα₁], [jα₂]) together with the two families where one level runs from its own up to the
coarsest (their eqs. 44–47), as the liminf of log S_{j₁j₂}(p) / log 2^{−j} with
S_{j₁j₂}(p) the mean of |c|^p over the cells (Theorem 3, eq. 50). Here the liminf is read over
a finite range of levels as the slowest of the families: within each family the pair with the
largest S at every j (the slowest decay towards fine scales), its envelope's slope against j,
and the smallest slope over the three families. Then, in the direction of the first axis,
η(p, e₁) = sup_α η_α(p) / α₁ (Theorem 1, eq. 25), η(p, e₂) by symmetry, the maximising α₁ as
the anisotropy profile, and the directional Legendre spectrum of Theorem 6 (eq. 96),
L(H, e) = inf_α (1/α_min) inf_{p ≥ p_α} (α_e H p − η_α(p) + d) with η_α(p_α) = d (eq. 95), an
upper bound on the directional Hausdorff spectrum. The isotropic η(p) is the diagonal
α = (1, 1). A fractional Brownian sheet with exponents (H₁, H₂) reads η(p, e_i) = p H_i and
η(p) = p min(H₁, H₂) (their Theorem 7).
"""
from __future__ import annotations

import math

import numpy as np

from dynamix.core import dwt
from dynamix.core.leaders import _box

__all__ = ["ALPHA_STEP", "FLOOR", "analyse", "coefficients", "directional_legendre", "directional_scaling",
           "families", "hyperbolic_leaders", "leader_scaling", "level_line_scaling", "scaling_functions",
           "strip_maps", "structure_functions"]

#: The anisotropy grid's step on α₁ ∈ (0, 2).
ALPHA_STEP = 1.0 / 8.0

#: A pair whose mean |c| sits below this share of the largest pair's is rounding, never structure:
#: float64 transforms leave ~1e-16 of the largest coefficient where a field has no detail, and a
#: smooth modulation of the field's own period still reaches ~1e-10 at the finest level of sym4.
FLOOR = 1e-12


def alpha_grid(step: float = ALPHA_STEP) -> np.ndarray:
    """α₁ from ``step`` to 2 − ``step``; α₂ = 2 − α₁."""
    n = int(round(2.0 / step))
    return np.arange(1, n) * step


def coefficients(values, name: str, level: int, *, gamma: float = 0.0, edge: str = "mirror",
                 origin=(0, 0), block=None, complete=False) -> dict:
    """The hyperbolic detail moduli of the field continued by ``edge``
    (:data:`dynamix.core.leaders.EDGE_MODES`): ``mods[(j1, j2)]`` is |c| at the pair (j₁ along
    x, j₂ along y), L¹, each axis rolled by its own wavelet atom's shift and integrated by
    2^{γ(j₁+j₂)/2}; ``valid[(j1, j2)]`` the cells inside the field at least ``border`` cells from
    an edge along each axis (none under ``periodic``) and clear of non-finite pixels."""
    from dynamix.core.leaders import _extend

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"directional: a 2-D band, not shape {x.shape}")
    J = int(level)
    finite = np.isfinite(x)
    filled = np.where(finite, x, float(x[finite].mean()) if finite.any() else 0.0)
    W = (tuple(int(math.ceil(n/2**J))*2**J for n in x.shape) if edge == "periodic"
         else dwt.working_shape(x.shape, J))
    work = _extend(filled, W, edge)
    bad = _extend((~finite).astype(np.float64), W, edge) > 0.5
    reg = dwt.registration(name, J)
    seam = edge == "periodic" and W != x.shape
    e = 0 if edge == "periodic" and not seam else dwt.border_cells(name) - 1
    hyp = dwt.hyperbolic_complete(work, name, J) if complete else dwt.hyperbolic(work, name, J)
    size = x.shape if block is None else tuple(int(n) for n in block)
    mods, valid = {}, {}
    for (j1, j2), c in hyp.items():
        l1, l2 = j1 or J, j2 or J
        # Approximation atoms use their coarsest cell frame; details retain the
        # measured wavelet registration used by the ordinary hyperbolic path.
        shifts = (reg[(l2, "detail")][0] if j2 else 0,
                  reg[(l1, "detail")][0] if j1 else 0)
        rolled = np.roll(c, shifts, axis=(0, 1))
        mods[(j1, j2)] = np.abs(rolled) * 2.0 ** (gamma * (l1 + l2) / 2.0)
        rlo, clo = math.ceil(origin[0]/2**l2), math.ceil(origin[1]/2**l1)
        q_rows, q_cols = (origin[0]+size[0])//2**l2, (origin[1]+size[1])//2**l1
        v = np.zeros(c.shape, dtype=bool)
        if q_rows-rlo-2*e > 0 and q_cols-clo-2*e > 0:
            v[rlo+e:q_rows-e, clo+e:q_cols-e] = True
        b = bad.reshape(W[0] // 2 ** l2, 2 ** l2, W[1] // 2 ** l1, 2 ** l1).any(axis=(1, 3))
        if b.any():
            v &= ~_box(b, e, np.logical_or) if e else ~b
        valid[(j1, j2)] = v
    return {"mods": mods, "valid": valid, "levels": J, "shape": tuple(x.shape), "wavelet": name,
            "gamma": float(gamma), "border_cells": e, "edge": edge,
            "origin": tuple(origin), "data_shape": tuple(size), "seam": seam, "complete": complete}



def restrict(coef: dict, origin, size) -> dict:
    """``coef`` counting only the cells wholly inside the ``size`` region at ``origin`` of its
    field: an ROI analysed on its processing window keeps the halo for each level's support and
    border, and its statistics are the ROI's own cells."""
    J = coef["levels"]
    valid = {}
    for (j1, j2), v in coef["valid"].items():
        l1, l2 = j1 or J, j2 or J
        rlo, clo = math.ceil(origin[0] / 2 ** l2), math.ceil(origin[1] / 2 ** l1)
        rhi, chi = (origin[0] + size[0]) // 2 ** l2, (origin[1] + size[1]) // 2 ** l1
        box = np.zeros_like(v)
        box[rlo:max(rlo, rhi), clo:max(clo, chi)] = True
        valid[(j1, j2)] = v & box
    return {**coef, "valid": valid, "origin": tuple(origin), "data_shape": tuple(size)}

def structure_functions(coef: dict, p_list) -> np.ndarray:
    """log2 S_{j₁j₂}(p), the mean of |c|^p over the valid cells: ``(n_p, J, J)`` indexed
    ``[p, j1 - 1, j2 - 1]``, NaN where a pair keeps no cell."""
    p = np.asarray(p_list, dtype=np.float64)
    J = coef["levels"]
    out = np.full((p.size, J, J), np.nan)
    for (j1, j2), m in coef["mods"].items():
        v = coef["valid"][(j1, j2)]
        if not v.any():
            continue
        sample = m[v]
        sample = sample[sample > 0]
        if sample.size == 0:
            continue
        lm = np.log(sample)
        for i, pi in enumerate(p):
            top = (pi * lm).max()
            out[i, j1 - 1, j2 - 1] = (top + np.log(np.exp(pi * lm - top).mean())) / np.log(2.0)
    return out


def families(j: int, alpha1: float, J: int) -> tuple:
    """The three families of Γ_j(α) in the finest-first index, each a list of (j₁, j₂): the
    pairs within one level of ([jα₁], [jα₂]); those at j₁ = [jα₁] with j₂ running from [jα₂]
    up to the coarsest; those at j₂ = [jα₂] with j₁ running up."""
    a1 = min(max(int(math.floor(j * alpha1)), 1), J)
    a2 = min(max(int(math.floor(j * (2.0 - alpha1))), 1), J)
    around = [(x, y) for x in range(a1 - 1, a1 + 2) for y in range(a2 - 1, a2 + 2)
              if 1 <= x <= J and 1 <= y <= J]
    edge_x = [(a1, y) for y in range(a2, J + 1)]
    edge_y = [(x, a2) for x in range(a1, J + 1)]
    return around, edge_x, edge_y


def _slope(x, y) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def fitted_levels(j_min: int, j_max: int, alpha1: float, J: int, finite=None) -> np.ndarray:
    """The j of the fit for one anisotropy: from ``j_min`` while ([jα₁] + 1, [jα₂] + 1) still lie
    within the J levels and, given ``finite`` (a (J, J) mask of pairs with a value), every pair
    of the three families has one — a family clipped at the coarsest level, or reaching a pair
    without cells, has a saturating envelope whose slope falls short."""
    js = []
    for j in range(int(j_min), int(j_max) + 1):
        if math.floor(j * alpha1) + 1 > J or math.floor(j * (2.0 - alpha1)) + 1 > J:
            continue
        if math.floor(j * alpha1) < 1 or math.floor(j * (2.0 - alpha1)) < 1:
            continue                   # a family pinned at level 1 has a constant envelope
        if finite is not None and not all(finite[j1 - 1, j2 - 1] for fam in families(j, alpha1, J)
                                          for (j1, j2) in fam):
            continue
        js.append(j)
    return np.asarray(js, dtype=np.float64)


def scaling_functions(log2_S, j_min: int, j_max: int, *, step: float = ALPHA_STEP) -> dict:
    """η_α(p) on the anisotropy grid: ``{"alpha": α₁ grid, "eta_alpha": (n_p, n_alpha),
    "n_levels": per α}``, each the smallest over the three families of the slope against j of
    the family's envelope (its largest log2 S at every fitted j), over the levels no family
    clips or loses a pair (:func:`fitted_levels`); NaN below three such levels."""
    log2_S = np.asarray(log2_S, dtype=np.float64)
    n_p, J = log2_S.shape[0], log2_S.shape[1]
    alphas = alpha_grid(step)
    eta = np.full((n_p, alphas.size), np.nan)
    n_levels = np.zeros(alphas.size, dtype=int)
    fits = []
    finite = np.all(np.isfinite(log2_S), axis=0)
    for a, alpha1 in enumerate(alphas):
        js = fitted_levels(j_min, j_max, float(alpha1), J, finite)
        if any(len({math.floor(j * axis) for j in js}) < 2 for axis in (alpha1, 2.0 - alpha1)):
            js = js[:0]                # a family that never moves has a constant envelope: a zero slope
        n_levels[a] = js.size
        envelopes = np.full((n_p, 3, js.size), np.nan)
        members = [families(int(j), float(alpha1), J) for j in js]
        slopes_by_family = np.full((n_p, 3), np.nan)
        intercepts = np.full((n_p, 3), np.nan)
        fit = {"x": js.astype(float), "envelopes": envelopes, "members": members,
               "slopes": slopes_by_family, "intercepts": intercepts, "admissible": js.size >= 3,
               "xlabel": "Γ index j ≈ (j₁+j₂)/2"}
        fits.append(fit)
        if js.size < 3:
            continue
        fams = members
        for i in range(n_p):
            slopes = []
            for f in range(3):
                env = []
                for k, j in enumerate(js):
                    vals = [log2_S[i, j1 - 1, j2 - 1] for (j1, j2) in fams[k][f]]
                    vals = [v for v in vals if np.isfinite(v)]
                    env.append(max(vals) if vals else np.nan)
                slope = _slope(js, env)
                slopes.append(slope)
                envelopes[i, f] = env
                slopes_by_family[i, f] = slope
                intercepts[i, f] = np.mean(env) - slope * np.mean(js)
            slopes = [s for s in slopes if np.isfinite(s)]
            if slopes:
                eta[i, a] = min(slopes)
    return {"alpha": alphas, "eta_alpha": eta, "n_levels": n_levels,
            "diagnostic_fits": fits}


def resolution_families(j: int, alpha1: float) -> tuple:
    """Literal resolution-index Γ families of (44–47), including index -1.

    These indices increase toward finer resolution; they are not the DWT's
    finest-first level numbers. Index -1 is the scaling branch.
    """
    bounds = [(math.floor((j-1)*a)-1, math.floor(j*a)+1) for a in (alpha1, 2-alpha1)]
    (lo1, hi1), (lo2, hi2) = bounds
    high1, high2 = range(lo1, hi1+1), range(lo2, hi2+1)
    return ([(x, y) for x in high1 for y in high2],
            [(x, y) for x in high1 for y in range(0, lo2)],
            [(x, y) for x in range(0, lo1) for y in high2])


def _pair_tables(coef: dict, p) -> tuple:
    """Per level pair: log2 of the mean of |c|^p over its valid cells (zeros kept in the count:
    removing them changes sparse fields' exponents), and the mean |c| itself."""
    tables, mean_abs = {}, {}
    for pair, m in coef["mods"].items():
        sample = m[coef["valid"][pair]]
        if not sample.size:
            tables[pair], mean_abs[pair] = np.full(p.size, np.nan), np.nan
            continue
        mean_abs[pair] = float(sample.mean())
        positive = sample[sample > 0]
        vals = []
        for order in p:
            if not positive.size:
                vals.append(-np.inf)
                continue
            z = order * np.log(positive)
            top = z.max()
            vals.append((top + np.log(np.exp(z - top).sum()) - np.log(sample.size)) / np.log(2))
        tables[pair] = np.array(vals)
    return tables, mean_abs


def _slope_se(x, y) -> tuple:
    """The OLS slope of ``y`` on ``x`` and its standard error (NaN below three points)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    x, y = x[ok], y[ok]
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    sxx = float(((x - x.mean()) ** 2).sum())
    return float(slope), float(np.sqrt((resid ** 2).sum() / (x.size - 2) / sxx)) if sxx > 0 else float("nan")


#: When φ partners a frontier point: only where every detail partner is below the floor, always, never.
COARSE_PARTNER = ("fallback", "always", "never")


def level_line_scaling(coef, p_list, j_min, *, step=ALPHA_STEP, coarse_partner="fallback", floor=FLOOR):
    """η_α(p) read by the anisotropic level m(r₁, r₂) = max(r₁/α₁, r₂/α₂) (Abry et al. 2015, Prop.
    4.7 (3)) in resolution indices r = J − level (the scaling branch φ at −1).

    The level sets {m = const} are L-shaped; their two arms are the frontiers. The x-frontier at
    a fine index r₁ is the largest log2 S over the pairs (r₁, r₂) with r₂ ≤ r₁α₂/α₁, placed at
    m = r₁/α₁; the y-frontier is its transpose. η_α(p) is the smaller of the two frontiers' slopes
    against −m (the larger moment rules the level set). A frontier point is fitted when its fine
    cell, 2^{J−r} px on its own axis, lies between 2^{j_min} px and 2^J px; its coarse partner is
    any coarser detail index, and φ as ``coarse_partner`` says: "fallback" only where every detail
    partner is below the floor (a layered field), "always", or "never". φ pairs rest on the
    coarsest cells, so where details exist they add spread without bias (±0.05 against ±0.03 on
    1024² fractional Brownian sheets). A pair whose mean |c| is below ``floor`` of the largest
    pair's is rounding and never enters; an axis left with no detail pair carries no structure,
    and its frontier is empty."""
    J = coef["levels"]
    p = np.asarray(p_list, float)
    tables, mean_abs = _pair_tables(coef, p)

    def res(level):
        return -1 if level == 0 else J - level

    top = max((v for v in mean_abs.values() if np.isfinite(v)), default=0.0)
    live = {(res(l1), res(l2)): tables[(l1, l2)] for (l1, l2) in tables
            if np.isfinite(mean_abs[(l1, l2)]) and mean_abs[(l1, l2)] > floor * top
            and np.isfinite(tables[(l1, l2)]).all()}
    no_structure = {"x": not any(r1 >= 0 for r1, _ in live), "y": not any(r2 >= 0 for _, r2 in live)}
    if coarse_partner not in COARSE_PARTNER:
        raise ValueError(f"coarse_partner must be one of {COARSE_PARTNER}")
    fine = range(0, J - int(j_min) + 1)
    alphas = alpha_grid(step)
    eta = np.full((p.size, alphas.size), np.nan)
    eta_se = np.full((p.size, alphas.size), np.nan)
    n_levels = np.zeros(alphas.size, int)
    fits = []
    for a, alpha in enumerate(alphas):
        weights = {"x": (float(alpha), 2.0 - float(alpha)), "y": (2.0 - float(alpha), float(alpha))}
        arms = {}
        for axis, (a_fine, a_coarse) in weights.items():
            xs, env, members, fine_px = [], [], [], []
            for rf in fine:
                cap = rf * a_coarse / a_fine
                row = [k for k in live if (k[0] if axis == "x" else k[1]) == rf
                       and (k[1] if axis == "x" else k[0]) <= cap]
                details = [k for k in row if min(k) >= 0]
                phi = [k for k in row if min(k) < 0]
                pairs = sorted(details + phi if coarse_partner == "always" else
                               details if details or coarse_partner == "never" else phi)
                if not pairs:
                    continue
                values = np.stack([live[k] for k in pairs])
                xs.append(-rf / a_fine)
                env.append(values.max(axis=0))
                members.append(pairs)
                fine_px.append(2 ** (J - rf))
            env = np.asarray(env, dtype=np.float64).T.reshape(p.size, len(xs))
            fitted = [_slope_se(xs, env[i]) for i in range(p.size)] if len(xs) >= 3 else []
            arms[axis] = {"x": np.asarray(xs), "envelope": env, "members": members, "fine_px": fine_px,
                          "slope": np.array([f[0] for f in fitted]) if fitted else np.full(p.size, np.nan),
                          "se": np.array([f[1] for f in fitted]) if fitted else np.full(p.size, np.nan)}
        n_levels[a] = max(arm["x"].size for arm in arms.values())
        both = np.stack([arms["x"]["slope"], arms["y"]["slope"]])
        se = np.stack([arms["x"]["se"], arms["y"]["se"]])
        for i in range(p.size):
            ok = np.isfinite(both[:, i])
            if ok.any():
                k = int(np.flatnonzero(ok)[np.argmin(both[ok, i])])
                eta[i, a], eta_se[i, a] = both[k, i], se[k, i]
        fits.append({"arms": arms, "admissible": bool(np.isfinite(eta[:, a]).any()),
                     "xlabel": "−m, the anisotropic level max(r₁/α₁, r₂/α₂)"})
    return {"alpha": alphas, "eta_alpha": eta, "eta_alpha_se": eta_se, "n_levels": n_levels,
            "estimator": "level line (Abry et al. 2015, Prop. 4.7 (3))", "resolution_origin": J,
            "diagnostic_fits": fits, "no_structure": no_structure, "coarse_partner": coarse_partner,
            "floor": float(floor),
            "moment_pairs": np.asarray(list(tables), dtype=int),
            "log2_moments": np.stack(list(tables.values()), axis=1),
            "moment_counts": np.asarray([int(coef["valid"][pair].sum()) for pair in tables]),
            "moment_normalization": "mean over every valid cell; zeros retained in the count"}


def gamma_scaling(coef, p_list, j_min, *, step=ALPHA_STEP):
    """Finite Γ-union envelope regression over the hyperbolic detail pairs.

    Resolution index r=J-level anchors the coarsest details at r=0; the
    scaling branch is r=-1. The moment tables span the complete basis, but a
    fitted level needs every family index >= 0, so the scaling branch never
    enters a fit. No family is clipped to available fine levels.
    The envelope is fitted against -r, the logarithm of relative scale.
    """
    J = coef["levels"]
    p = np.asarray(p_list, float)
    tables, _ = _pair_tables(coef, p)
    alphas = alpha_grid(step)
    eta = np.full((p.size, alphas.size), np.nan)
    n_levels = np.zeros(alphas.size, int)
    fits = []
    def key(pair):
        return tuple(0 if r == -1 else J-r for r in pair)
    for a, alpha in enumerate(alphas):
        js, envelopes, members = [], [], []
        for r in range(1, J-int(j_min)+1):
            if any(math.floor((r-1)*axis)-1 < 0 for axis in (alpha, 2-alpha)):
                continue   # coarse scaling crossover is not an asymptotic fit level
            pairs = set(pair for fam in resolution_families(r, float(alpha)) for pair in fam)
            if any(min(pair) < -1 or max(pair) >= J or key(pair) not in tables for pair in pairs):
                continue
            values = np.stack([tables[key(pair)] for pair in sorted(pairs)])
            if not np.isfinite(values).all():
                continue
            js.append(r)
            envelopes.append(values.max(axis=0))
            members.append([[key(pair) for pair in sorted(pairs)]])
        n_levels[a] = len(js)
        resolved = all(len({math.floor(r*axis) for r in js}) >= 3 for axis in (alpha, 2-alpha))
        if not resolved:
            n_levels[a] = 0
        xx = -np.asarray(js, dtype=float)
        yy = np.asarray(envelopes).T.reshape(p.size, 1, len(js))
        slopes = np.full((p.size, 1), np.nan)
        intercepts = np.full((p.size, 1), np.nan)
        if len(js) >= 3 and resolved:
            for i in range(p.size):
                eta[i, a] = _slope(-np.asarray(js), np.asarray(envelopes)[:, i])
                slopes[i, 0] = eta[i, a]
                intercepts[i, 0] = np.mean(yy[i, 0]) - eta[i, a] * np.mean(xx)
        fits.append({"x": xx, "envelopes": yy, "members": members,
                     "slopes": slopes, "intercepts": intercepts,
                     "admissible": len(js) >= 3 and resolved, "xlabel": "−r (log₂ relative scale)"})
    return {"alpha": alphas, "eta_alpha": eta, "n_levels": n_levels,
            "estimator": "resolution-index Gamma union envelope regression",
            "resolution_origin": J, "diagnostic_fits": fits,
            "moment_pairs": np.asarray(list(tables), dtype=int),
            "log2_moments": np.stack(list(tables.values()), axis=1),
            "moment_counts": np.asarray([int(coef["valid"][pair].sum()) for pair in tables]),
            "moment_normalization": "mean over every valid cell; zeros retained in the count"}


def directional_scaling(scaling: dict, p_list=None) -> dict:
    """η(p, e₁) = sup_α η_α(p)/α₁ and η(p, e₂) by symmetry (α₂ = 2 − α₁), where the isotropic η(p)
    (the diagonal α₁ = 1) is positive, as Ben Slimane et al.'s eq. 50 requires, and the axis
    carries structure. The maximisers form an interval, every α whose ratio lies within one
    standard error of the largest (a plateau of maximisers is the formalism's own: a fractional
    Brownian sheet's is exact), and the sup is read as the mean ratio over it: each is an estimate
    of the same value, and the largest of noisy estimates is biased up. α₁*(p) is the interval's
    largest α. Given ``p_list``, H_e is the slope of η(p, e) against p where it is read."""
    alphas, eta = scaling["alpha"], np.asarray(scaling["eta_alpha"], dtype=np.float64)
    se = np.asarray(scaling.get("eta_alpha_se", np.zeros_like(eta)), dtype=np.float64)
    diag = int(np.argmin(np.abs(alphas - 1.0)))
    positive = eta[:, diag] > 0
    empty = scaling.get("no_structure", {})

    def best(axis_alpha, axis):
        n = eta.shape[0]
        out_eta, out_alpha, interval = np.full(n, np.nan), np.full(n, np.nan), np.full((2, n), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio, band = eta / axis_alpha[None, :], se / axis_alpha[None, :]
        for i in range(n):
            row = ratio[i]
            if not np.isfinite(row).any() or not positive[i] or empty.get(axis, False):
                continue
            k = int(np.nanargmax(row))
            tol = band[i, k] if np.isfinite(band[i, k]) else 0.0
            hits = np.flatnonzero(np.isfinite(row) & (row >= row[k] - max(tol, 1e-9)))
            out_eta[i], out_alpha[i] = row[hits].mean(), np.max(axis_alpha[hits])
            interval[:, i] = axis_alpha[hits].min(), axis_alpha[hits].max()
        return out_eta, out_alpha, interval

    eta_e1, alpha_star_e1, interval_e1 = best(alphas, "x")
    eta_e2, alpha_star_e2, interval_e2 = best(2.0 - alphas, "y")
    out = {"eta_e1": eta_e1, "eta_e2": eta_e2, "eta_iso": eta[:, diag],
           "alpha_star_e1": alpha_star_e1, "alpha_star_e2": alpha_star_e2,
           "alpha_star_e1_interval": interval_e1, "alpha_star_e2_interval": interval_e2,
           "eta_positive": positive}
    if p_list is not None:
        p = np.asarray(p_list, dtype=np.float64)
        for k in ("e1", "e2"):
            out[f"H_{k}"] = _slope(p, out[f"eta_{k}"]) if np.isfinite(out[f"eta_{k}"]).sum() >= 2 else float("nan")
    return out


def directional_legendre(scaling: dict, p_list, *, direction: int = 1, d: float = 2.0,
                         h_grid=None) -> dict:
    """L(H, e) = inf_α (1/α_min) inf_{p ≥ p_α} (α_e H p − η_α(p) + d), with p_α the first p at
    which η_α(p) reaches d (interpolated on the p grid); an anisotropy whose η_α never reaches
    d contributes nothing; the bound is capped at d (a dimension). Taken over p ≥ p_α only, it
    binds the increasing side of the spectrum and sits at d beyond its knee.
    ``{"h": grid, "L": values}``, NaN where no α contributes."""
    alphas, eta = scaling["alpha"], np.asarray(scaling["eta_alpha"], dtype=np.float64)
    p = np.asarray(p_list, dtype=np.float64)
    h = np.linspace(-0.5, 2.5, 121) if h_grid is None else np.asarray(h_grid, dtype=np.float64)
    L = np.full(h.size, np.inf)
    limited = np.zeros(h.size, dtype=bool)
    for a, alpha1 in enumerate(alphas):
        alpha2 = 2.0 - alpha1
        row = eta[:, a]
        ok = np.isfinite(row)
        if ok.sum() < 2 or np.nanmax(row) < d:
            continue
        order = np.argsort(p[ok])
        pp, ee = p[ok][order], row[ok][order]
        above = np.flatnonzero(ee >= d)
        k = above[0]
        p_alpha = pp[k] if k == 0 else pp[k - 1] + (d - ee[k - 1]) * (pp[k] - pp[k - 1]) / (ee[k] - ee[k - 1])
        keep = pp >= p_alpha - 1e-12
        # Include the interpolated admissibility endpoint, rather than starting at
        # the next sampled moment and making the bound depend on that grid gap.
        pp = np.r_[p_alpha, pp[keep]]
        ee = np.r_[np.interp(p_alpha, p[ok][order], row[ok][order]), ee[keep]]
        alpha_e = alpha1 if direction == 1 else alpha2
        terms = alpha_e * h[:, None] * pp[None, :] - ee[None, :] + d
        legendre = terms.min(axis=1) / min(alpha1, alpha2)
        lower = legendre < L
        limited = np.where(lower, terms.argmin(axis=1) == pp.size - 1, limited)
        L = np.minimum(L, legendre)
    L[~np.isfinite(L)] = np.nan
    L = np.minimum(L, d)
    reached = np.flatnonzero(L >= d - 1e-9)
    with np.errstate(invalid="ignore"):
        empty = L < 0.0
    return {"h": h, "L": np.where(empty, np.nan, L), "limited": limited & np.isfinite(L) & (L < d - 1e-9),
            "knee": float(h[reached[0]]) if reached.size else float("nan"),
            "empty_below": float(h[np.flatnonzero(np.isfinite(L) & ~empty)[0]]) if (np.isfinite(L) & ~empty).any()
            else float("nan")}


def hyperbolic_leaders(coef: dict) -> dict:
    """Abry et al.'s hyperbolic wavelet leaders (2015, eq. 3.3) on the detail pairs: d(l₁, l₂) is the
    sup of |c| over the dyadic rectangle's finer sub-rectangles, D(l₁, l₂) = max(|c(l₁, l₂)|,
    pool_x D(l₁ − 1, l₂), pool_y D(l₁, l₂ − 1)), then over its 3 × 3 neighbourhood. A leader is
    valid where every cell it reads is (the border and missing pixels propagate up)."""
    J = coef["levels"]
    D, V = {}, {}
    for l2 in range(1, J + 1):
        for l1 in range(1, J + 1):
            m = np.array(coef["mods"][(l1, l2)], dtype=np.float64)
            v = np.array(coef["valid"][(l1, l2)], dtype=bool)
            if l1 > 1:
                m = np.maximum(m, np.maximum(D[(l1 - 1, l2)][:, 0::2], D[(l1 - 1, l2)][:, 1::2]))
                v &= V[(l1 - 1, l2)][:, 0::2] & V[(l1 - 1, l2)][:, 1::2]
            if l2 > 1:
                m = np.maximum(m, np.maximum(D[(l1, l2 - 1)][0::2], D[(l1, l2 - 1)][1::2]))
                v &= V[(l1, l2 - 1)][0::2] & V[(l1, l2 - 1)][1::2]
            D[(l1, l2)], V[(l1, l2)] = m, v
    return {"levels": J, "leaders": {k: _box(m, 1, np.maximum) for k, m in D.items()},
            "valid": {k: _box(v, 1, np.logical_and) for k, v in V.items()}}


def leader_scaling(coef: dict, p_list, j_min: int, *, step=ALPHA_STEP, h_grid=None) -> dict:
    """The hyperbolic leaders' scaling function ω(p, α) by the level line, over p of both signs
    (the leaders bound the small coefficients too), and Abry et al.'s level-set bound per
    anisotropy, L_f(H, α) = inf_p (Hp − ω(p, α) + 2) (eq. 3.7), capped at d = 2 and empty below 0."""
    positive = np.asarray(p_list, dtype=np.float64)
    p = np.concatenate([-positive[::-1], positive])
    lead = hyperbolic_leaders(coef)
    scaling = level_line_scaling({"levels": lead["levels"], "mods": lead["leaders"], "valid": lead["valid"]},
                                 p, j_min, step=step, coarse_partner="never")
    h = np.linspace(-0.5, 2.5, 121) if h_grid is None else np.asarray(h_grid, dtype=np.float64)
    omega = scaling["eta_alpha"]
    L = np.full((omega.shape[1], h.size), np.nan)
    for a in range(omega.shape[1]):
        ok = np.isfinite(omega[:, a])
        if ok.sum() < 2:
            continue
        L[a] = (h[:, None] * p[ok][None, :] - omega[ok, a][None, :] + 2.0).min(axis=1)
    with np.errstate(invalid="ignore"):
        L = np.where(L < 0.0, np.nan, np.minimum(L, 2.0))
    return {"p": p, "alpha": scaling["alpha"], "eta_alpha": omega, "h": h, "L_alpha": L}


def strip_maps(lead: dict, j_min: int, shape, crop=None) -> dict:
    """Local directional exponents from the strip leaders (Abry et al. 2015, Thm 3.4): H_x at a
    pixel is the slope of log2 d(l₁, J) over the fine x-levels l₁ = j_min … J − 1 with the y-level
    held at the coarsest J, cells 2^{l₁} × 2^J px; H_y the transpose; the anisotropy
    α₁* = 2H_y/(H_x + H_y), 1 where isotropic (the fractional Brownian sheet's maximiser). A pixel
    whose strip reaches an invalid cell at any fitted level is NaN. Resolution: the coarsest fitted
    level along the fine axis, 3·2^J px along the coarse one. ``crop`` = (row, col, h, w) cuts an
    ROI out of the painted window."""
    J = lead["levels"]
    levels = np.arange(int(j_min), J, dtype=np.float64)
    rows, cols = (int(n) for n in shape)

    def painted(key, ly, lx):
        d, v = lead["leaders"][key], lead["valid"][key]
        with np.errstate(divide="ignore"):
            cells = np.where(v & (d > 0), np.log2(np.where(d > 0, d, 1.0)), np.nan)
        return np.repeat(np.repeat(cells, 2 ** ly, axis=0), 2 ** lx, axis=1)

    def slope(stack):
        x = (levels - levels.mean())[:, None, None]
        return (x * (stack - stack.mean(axis=0))).sum(axis=0) / float((x ** 2).sum())

    h_x = slope(np.stack([painted((int(l), J), J, int(l)) for l in levels]))
    h_y = slope(np.stack([painted((J, int(l)), int(l), J) for l in levels]))
    r0, c0, h, w = (0, 0, rows, cols) if crop is None else (int(v) for v in crop)
    h_x, h_y = h_x[r0:r0 + h, c0:c0 + w], h_y[r0:r0 + h, c0:c0 + w]
    with np.errstate(invalid="ignore", divide="ignore"):
        anisotropy = np.where((h_x > 0) & (h_y > 0), 2.0 * h_y / (h_x + h_y), np.nan)
    return {"h_x": h_x, "h_y": h_y, "anisotropy": anisotropy, "levels": levels, "coarse_level": J}


def analyse(values, name: str, level: int, *, j_min: int, p_list, gamma: float = 0.0,
            step: float = ALPHA_STEP, edge: str = "mirror", estimator="legacy", origin=(0, 0), block=None,
            coefficient_cache=None, coarse_partner: str = "fallback") -> dict:
    """Everything the directional formalism reads off one band: the structure functions, η_α
    on the grid, η(p, e₁), η(p, e₂), η(p), α₁*(p) and L(H, e₁), L(H, e₂)."""
    p = np.asarray(p_list, dtype=np.float64)
    if estimator not in ("level_line", "legacy", "gamma"):
        raise ValueError("directional estimator must be level_line, gamma or legacy")
    coef = coefficient_cache if coefficient_cache is not None else coefficients(
        values, name, level, gamma=gamma, edge=edge, origin=origin, block=block,
        complete=estimator in ("gamma", "level_line"))
    log2_S = structure_functions({**coef, "mods": {k: v for k, v in coef["mods"].items() if min(k) > 0}}, p)
    if estimator == "level_line":
        scaling = level_line_scaling(coef, p, j_min, step=step, coarse_partner=coarse_partner)
    elif estimator == "gamma":
        scaling = gamma_scaling(coef, p, j_min, step=step)
    else:
        scaling = scaling_functions(log2_S, j_min, coef["levels"], step=step)
    if estimator == "legacy":
        pairs = [(j1, j2) for j1 in range(1, level+1) for j2 in range(1, level+1)]
        scaling.update(moment_pairs=np.asarray(pairs),
                       log2_moments=np.stack([log2_S[:, j1-1, j2-1] for j1, j2 in pairs], axis=1),
                       moment_counts=np.asarray([int(coef["valid"][pair].sum()) for pair in pairs]),
                       moment_normalization="legacy mean over positive coefficients only")
    out = {"p": p, "levels": coef["levels"], "j_min": int(j_min), "log2_S": log2_S,
           "valid_cells": np.array([[int(coef["valid"][(j1, j2)].sum()) for j2 in range(1, coef["levels"] + 1)]
                                    for j1 in range(1, coef["levels"] + 1)]),
           "border_cells": coef["border_cells"], "estimator": estimator,
           "complete_basis": coef["complete"], "seam": coef["seam"],
           "origin": coef["origin"], "data_shape": coef["data_shape"], **scaling, **directional_scaling(scaling, p)}
    out["estimator_description"], out["estimator"] = scaling.get("estimator", estimator), estimator
    out["leaders"] = leader_scaling(coef, p, j_min, step=step)
    for direction in (1, 2):
        leg = directional_legendre(scaling, p, direction=direction)
        out[f"L_e{direction}"] = leg
    return out
