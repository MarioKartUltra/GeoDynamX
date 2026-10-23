# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""The directional thermodynamic formalism of Ben Slimane, Ben Abid, Ben Omrane and Halouani
(Symmetry 11:825, 2019) on the hyperbolic (fully separable) wavelet coefficients of
:func:`dynamix.core.dwt.hyperbolic`.

The default device estimator constructs Γ in increasing-resolution coordinates
from the complete separable basis, then fits the union envelope over resolved
levels. The legacy API estimator described below remains selectable. Finite
regressions do not establish the asymptotic bounds.

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

__all__ = ["ALPHA_STEP", "analyse", "coefficients", "directional_legendre", "directional_scaling",
           "families", "scaling_functions", "structure_functions"]

#: The anisotropy grid's step on α₁ ∈ (0, 2).
ALPHA_STEP = 1.0 / 8.0


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
    tables = {}
    for pair, m in coef["mods"].items():
        sample = m[coef["valid"][pair]]
        if not sample.size:
            tables[pair] = np.full(p.size, np.nan)
            continue
        vals = []
        for order in p:
            # For positive moments zeros contribute zero, but remain in the
            # normalization count. Removing them changes sparse-field exponents.
            positive = sample[sample > 0]
            if not positive.size:
                vals.append(-np.inf)
                continue
            z = order*np.log(positive)
            top = z.max()
            vals.append((top+np.log(np.exp(z-top).sum())-np.log(sample.size))/np.log(2))
        tables[pair] = np.array(vals)
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


def directional_scaling(scaling: dict) -> dict:
    """η(p, e₁) = max_α η_α(p)/α₁ with the largest maximising α₁ as α₁*(p); η(p, e₂) by
    symmetry (α₂ = 2 − α₁); η(p) the diagonal α₁ = 1."""
    alphas, eta = scaling["alpha"], np.asarray(scaling["eta_alpha"], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        r1 = eta / alphas[None, :]
        r2 = eta / (2.0 - alphas)[None, :]

    def best(ratio, axis_alpha):
        out_eta, out_alpha = np.full(ratio.shape[0], np.nan), np.full(ratio.shape[0], np.nan)
        for i in range(ratio.shape[0]):
            row = ratio[i]
            if not np.isfinite(row).any():
                continue
            top = np.nanmax(row)
            hits = np.flatnonzero(np.isfinite(row) & (row >= top - 1e-9))
            out_eta[i], out_alpha[i] = top, np.max(axis_alpha[hits])
        return out_eta, out_alpha

    eta_e1, alpha_star_e1 = best(r1, alphas)
    eta_e2, alpha_star_e2 = best(r2, 2.0 - alphas)
    diag = int(np.argmin(np.abs(alphas - 1.0)))
    return {"eta_e1": eta_e1, "eta_e2": eta_e2, "eta_iso": eta[:, diag],
            "alpha_star_e1": alpha_star_e1, "alpha_star_e2": alpha_star_e2}


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
        legendre = (alpha_e * h[:, None] * pp[None, :] - ee[None, :] + d).min(axis=1)
        L = np.minimum(L, legendre / min(alpha1, alpha2))
    L[~np.isfinite(L)] = np.nan
    return {"h": h, "L": np.minimum(L, d)}


def analyse(values, name: str, level: int, *, j_min: int, p_list, gamma: float = 0.0,
            step: float = ALPHA_STEP, edge: str = "mirror", estimator="legacy", origin=(0, 0), block=None,
            coefficient_cache=None) -> dict:
    """Everything the directional formalism reads off one band: the structure functions, η_α
    on the grid, η(p, e₁), η(p, e₂), η(p), α₁*(p) and L(H, e₁), L(H, e₂)."""
    p = np.asarray(p_list, dtype=np.float64)
    if estimator not in ("legacy", "gamma"):
        raise ValueError("directional estimator must be legacy or gamma")
    coef = coefficient_cache if coefficient_cache is not None else coefficients(
        values, name, level, gamma=gamma, edge=edge, origin=origin, block=block, complete=estimator == "gamma")
    log2_S = structure_functions({**coef, "mods": {k: v for k, v in coef["mods"].items() if min(k) > 0}}, p)
    scaling = (gamma_scaling(coef, p, j_min, step=step) if estimator == "gamma" else
               scaling_functions(log2_S, j_min, coef["levels"], step=step))
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
           "origin": coef["origin"], "data_shape": coef["data_shape"], **scaling, **directional_scaling(scaling)}
    for direction in (1, 2):
        leg = directional_legendre(scaling, p, direction=direction)
        out[f"L_e{direction}"] = leg
    return out
